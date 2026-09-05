"""Single-resident-model proxy for the self-contained audio.cpp candidate."""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from collections import OrderedDict, deque
from pathlib import Path
from secrets import token_hex
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))
from profile_library import (  # noqa: E402
    canonical_profile,
    clone_content_hash,
    live_profiles,
    valid_profile_id,
)


class _OwnedStreamingResponse(StreamingResponse):
    """Streaming response whose external resources survive only for its ASGI call.

    Starlette may fail while sending response headers, before the body iterator
    is entered.  A generator ``finally`` alone cannot release an already-held
    model-generation lock in that ordering, so the response itself owns one
    idempotent asynchronous cleanup callback as the outermost boundary.
    """

    def __init__(self, *args: Any, cleanup: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._cleanup = cleanup

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self._cleanup()

TEMPLATE = Path(os.environ.get("AUDIO_CPP_CONFIG_TEMPLATE", "/config/qwen3-tts-base-f16.json"))
ACTIVE_CONFIG = Path("/run/audio-cpp-active.json")
ENGINE_BIN = os.environ.get("AUDIO_CPP_SERVER_BIN", "/opt/audio.cpp/build/linux-cuda-release/bin/audiocpp_server")
ENGINE_URL = "http://127.0.0.1:8081"
ACTIVE_MODEL = os.environ.get("AUDIO_CPP_ACTIVE_MODEL", "qwen3-tts-1.7b-base-bf16")
NATIVE_INCREMENTAL_PCM_ENABLED = os.environ.get("AUDIO_CPP_NATIVE_INCREMENTAL_PCM", "false").lower() == "true"
NATIVE_LOAD_WARMUP_ENABLED = os.environ.get("AUDIO_CPP_NATIVE_LOAD_WARMUP", "false").lower() == "true"
NATIVE_LOAD_WARMUP_TIMEOUT_SECONDS = float(os.environ.get("AUDIO_CPP_NATIVE_LOAD_WARMUP_TIMEOUT_SECONDS", "180"))
try:
    TALKER_PREFIX_CACHE_SLOTS = int(os.environ.get("AUDIO_CPP_TALKER_PREFIX_CACHE_SLOTS", "0"))
except ValueError as exc:
    raise RuntimeError("AUDIO_CPP_TALKER_PREFIX_CACHE_SLOTS must be an integer from 0 through 64") from exc
if not 0 <= TALKER_PREFIX_CACHE_SLOTS <= 64:
    raise RuntimeError("AUDIO_CPP_TALKER_PREFIX_CACHE_SLOTS must be from 0 through 64")
VOICE_LIBRARY_DIR = Path(os.environ.get("VOICE_LIBRARY_DIR", "/voices"))
MIN_SYNTHESIS_FREE_MIB = 2048
# The 0.6B reserve is derived from the observed 7,266 -> 1,636 MiB load delta
# (~5,630 MiB), rounded up to 6,000 MiB before adding the unchanged synthesis
# floor. The 1.7B value remains the pre-existing total admission threshold: its
# residency delta has not been measured, so adding another floor would invent
# headroom and could make it unloadable on this 16 GiB Windows GPU.
MODEL_RESIDENCY_RESERVE_MIB = {
    "qwen3-tts-0.6b-base-bf16": 6000,
}
MIN_FREE_MIB = {
    "qwen3-tts-0.6b-base-bf16": MODEL_RESIDENCY_RESERVE_MIB["qwen3-tts-0.6b-base-bf16"] + MIN_SYNTHESIS_FREE_MIB,
    "qwen3-tts-1.7b-base-bf16": 10500,
}
MAX_BUSY_PERCENT = 85
MAX_SYNTHESIS_BUSY_PERCENT = 95
GPU_GUARD_SETTINGS_PATH = VOICE_LIBRARY_DIR / "candidate_gpu_guard.json"
VOICE_STUDIO_SETTINGS_PATH = VOICE_LIBRARY_DIR / "voice_studio_settings.json"
TUNING_PROFILES_PATH = VOICE_LIBRARY_DIR / "tts_profiles.json"
GPU_GUARD_MODES = {"enforced", "custom", "disabled"}
TUNING_PROVIDER = "qwen3tts-audiocpp"
TUNING_SCOPES = {"voice-studio", "realtime"}
MODEL_REQUIRED_LEFT_CONTEXT_FRAMES = 25
BUILTIN_TUNING_PROFILE_IDS = {"quality", "balanced", "low-latency"}
TUNING_VALUE_FIELDS = {
    "model",
    "clone_mode",
    "max_reference_seconds",
    "first_block_frames",
    "steady_block_frames",
    "left_context_frames",
    "text_lookahead",
    "phrase_flush_ms",
    "temperature",
    "top_k",
    "top_p",
    "repetition_penalty",
    "seed",
}
ENGINE_DECODE_MODE_HEADER = "X-AudioCPP-Qwen3-Decode-Mode"
OFFLINE_FULL_DECODE_MODE = "offline-full-decoder"
SUPPORTED_MASTER_OUTPUT_FORMATS = {"wav", "pcm", "flac", "mp3", "aac", "opus"}
EVENTS: deque[dict[str, Any]] = deque(maxlen=20)
REQUEST_OUTCOMES: OrderedDict[str, dict[str, Any]] = OrderedDict()
OUTCOME_TTL_SECONDS = 600.0
OUTCOME_CAPACITY = 256
LIVE_SYNTHESIS_WALL_SECONDS = 180.0


def _prune_request_outcomes() -> None:
    cutoff = time.monotonic() - OUTCOME_TTL_SECONDS
    for key, value in list(REQUEST_OUTCOMES.items()):
        if value["_updated"] < cutoff:
            del REQUEST_OUTCOMES[key]
    while len(REQUEST_OUTCOMES) > OUTCOME_CAPACITY:
        REQUEST_OUTCOMES.popitem(last=False)


def _begin_request_outcome(payload: dict[str, Any], model: str, mode: str) -> str:
    # Server-generated identities prevent collisions or prompt material from
    # being stored in this deliberately content-free diagnostic lookup.
    request_id = uuid4().hex
    payload["_outcome_id"] = request_id
    REQUEST_OUTCOMES[request_id] = {
        "requestId": request_id, "model": model, "mode": mode,
        "state": "pending", "reason": None, "generatedFrames": None,
        "generationCap": None, "eos": None, "audioSeconds": 0.0,
        "bytes": 0, "elapsedSeconds": None, "_updated": time.monotonic(),
    }
    _prune_request_outcomes()
    return request_id


def _finish_request_outcome(request_id: str, status: str, reason: str | None = None,
                            **fields: Any) -> None:
    record = REQUEST_OUTCOMES.get(request_id)
    if record is None or record["state"] in {"completed", "error", "cancelled", "limited"}:
        return
    allowed = {"generatedFrames", "generationCap", "eos", "audioSeconds", "bytes", "elapsedSeconds",
               "promptMs", "prefillMs", "talkerMs", "decodeMs"}
    record.update({key: value for key, value in fields.items() if key in allowed})
    record.update(state=status, reason=reason, _updated=time.monotonic())
    _prune_request_outcomes()


def _request_audio_limit_frames(payload: dict[str, Any], mode: str) -> int | None:
    if mode == OFFLINE_FULL_DECODE_MODE or payload.get("_internal_warmup"):
        # Full WAV keeps the engine's established capacity. Do not shorten a
        # user's long offline clip with a live-conversation heuristic.
        return None
    text = str(payload.get("input") or payload.get("text") or "")
    words = len(re.findall(r"\w+", text, re.UNICODE))
    characters = len(re.sub(r"\s+", "", text))
    estimated = max(words / 2.6, characters / 14.0)
    seconds = min(60.0, max(12.0, 3.0 * estimated + 5.0))
    return int(seconds / 0.08)


def _install_request_limit(payload: dict[str, Any], engine_payload: dict[str, Any], mode: str) -> int | None:
    cap = _request_audio_limit_frames(payload, mode)
    options = dict(engine_payload.get("options") or {})
    # The candidate owns this safety bound; arbitrary options cannot raise it.
    options.pop("qwen3_tts.max_generated_frames", None)
    if cap is not None:
        options["qwen3_tts.max_generated_frames"] = cap
    if options:
        engine_payload["options"] = options
    else:
        engine_payload.pop("options", None)
    return cap


def _generation_event_fields(event: dict[str, Any]) -> dict[str, Any]:
    values = (event.get("generated_frames"), event.get("generation_cap"))
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise HTTPException(502, "Malformed audio.cpp generation outcome")
    frames, cap = values
    termination = event.get("termination")
    if cap < 1 or frames > cap or termination not in {"eos", "max_generated_frames"}:
        raise HTTPException(502, "Malformed audio.cpp generation outcome")
    result = {"generatedFrames": frames, "generationCap": cap, "eos": termination == "eos"}
    for engine_key, key in (("prompt_ms", "promptMs"), ("prefill_ms", "prefillMs"), ("talker_ms", "talkerMs"), ("decode_ms", "decodeMs")):
        if engine_key in event:
            value = event[engine_key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 600000:
                raise HTTPException(502, "Malformed audio.cpp stage timing")
            result[key] = value
    return result


def _raise_generation_limit() -> None:
    raise HTTPException(422, {"state": "generation-limit", "reason": "max_generated_frames",
                              "message": "Synthesis reached its generated-audio limit; output is incomplete."})


async def _deadline_lines(lines: Any, deadline: float) -> Any:
    """Bound the entire engine stream, not each individual socket read."""
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("synthesis wall-clock deadline exceeded")
        try:
            yield await asyncio.wait_for(lines.__anext__(), remaining)
        except StopAsyncIteration:
            return

# A numeric epoch changes when this process replaces its child.  The nonce
# additionally distinguishes a new supervisor process whose counter begins at
# zero again, so frozen response snapshots cannot cross a candidate restart.
SUPERVISOR_INSTANCE_ID = uuid4().hex
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("audio_cpp_candidate")
app = FastAPI(title="audio.cpp candidate supervisor")


@app.get("/v1/audio/outcomes/{request_id}")
async def request_outcome(request_id: str) -> Response:
    if not re.fullmatch(r"[0-9a-f]{32}", request_id):
        raise HTTPException(422, "Invalid synthesis request ID")
    _prune_request_outcomes()
    outcome = REQUEST_OUTCOMES.get(request_id)
    if outcome is None:
        raise HTTPException(404, "Synthesis outcome is absent or expired")
    return JSONResponse({key: value for key, value in outcome.items() if not key.startswith("_")},
                        headers={"Cache-Control": "no-store"})


engine: subprocess.Popen[bytes] | None = None
switch_lock = asyncio.Lock()
generation_lock = asyncio.Lock()
state: dict[str, Any] = {
    "activeModel": ACTIVE_MODEL,
    "state": "loading",
    "reason": None,
    "lastError": None,
    "lastLoadElapsedS": None,
    "lastWarmupElapsedS": None,
    "lastWarmupProfileId": None,
    "lastWarmupStatus": None,
    "lastWarmupError": None,
    "engineEpoch": 0,
    "gpu": None,
    "lastAction": None,
}


def _configured_model_mode() -> str:
    """Return the engine session mode advertised and configured by this process."""
    return "streaming" if NATIVE_INCREMENTAL_PCM_ENABLED else "offline"


def _enforced_gpu_policy() -> dict[str, Any]:
    """Describe evidence-backed per-model load admission without invented reserves."""
    models: dict[str, dict[str, Any]] = {}
    for model_id, load_minimum_mib in MIN_FREE_MIB.items():
        residency_mib = MODEL_RESIDENCY_RESERVE_MIB.get(model_id)
        if residency_mib is not None:
            models[model_id] = {
                "loadMinimumFreeMiB": load_minimum_mib,
                "admissionKind": "measured-residency-plus-synthesis-floor",
                "residencyReserveMiB": residency_mib,
                "postLoadSynthesisReserveMiB": MIN_SYNTHESIS_FREE_MIB,
                "formula": "measured model residency reserve + post-load synthesis floor",
            }
        else:
            models[model_id] = {
                "loadMinimumFreeMiB": load_minimum_mib,
                "admissionKind": "existing-total-threshold",
                "residencyReserveMiB": None,
                "postLoadSynthesisReserveMiB": None,
                "formula": "existing total admission threshold; residency delta not yet measured",
            }
    return {
        "synthesisReserveMiB": MIN_SYNTHESIS_FREE_MIB,
        "formula": "per-model evidence-backed admission; measured residency + synthesis floor where available",
        "models": models,
    }


def _default_gpu_guard_settings() -> dict[str, int | str]:
    return {
        "mode": "disabled",
        # 1.7B is the candidate default, so expose its safe reserve as the
        # editable Custom starting point. Enforced mode remains model-specific.
        "load_min_free_mib": MIN_FREE_MIB[ACTIVE_MODEL],
        "synthesis_min_free_mib": MIN_SYNTHESIS_FREE_MIB,
        "load_max_utilization_percent": MAX_BUSY_PERCENT,
        "synthesis_max_utilization_percent": MAX_SYNTHESIS_BUSY_PERCENT,
    }


def _load_gpu_guard_settings() -> dict[str, int | str]:
    settings = _default_gpu_guard_settings()
    try:
        saved = json.loads(GPU_GUARD_SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return settings
    if isinstance(saved, dict):
        settings.update({key: saved[key] for key in settings if key in saved})
    return settings


gpu_guard_settings: dict[str, int | str] = _load_gpu_guard_settings()


def _save_gpu_guard_settings() -> None:
    """Atomically persist candidate-only GPU admission preferences."""
    VOICE_LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    temporary = GPU_GUARD_SETTINGS_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(gpu_guard_settings, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(GPU_GUARD_SETTINGS_PATH)


def _read_voice_studio_settings() -> dict[str, Any]:
    try:
        saved = json.loads(VOICE_STUDIO_SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return saved if isinstance(saved, dict) else {}


def _write_voice_studio_settings(payload: dict[str, Any]) -> dict[str, Any]:
    """Atomically persist non-secret Studio controls in the candidate volume."""
    allowed = {"endpoint", "model", "system_prompt", "mic_id", "llm_input_format", "llm_input_rate", "vad", "phrase"}
    settings = _read_voice_studio_settings()
    for key in allowed:
        value = payload.get(key)
        if isinstance(value, (str, int, float, bool)):
            settings[key] = value
        elif key in {"vad", "phrase"} and isinstance(value, dict):
            settings[key] = {str(k): str(v) for k, v in value.items()}
    # API keys remain page-session/browser-only by design.
    VOICE_LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    temporary = VOICE_STUDIO_SETTINGS_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(settings, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(VOICE_STUDIO_SETTINGS_PATH)
    _record_event("voice-studio-settings-saved")
    return settings


def _default_tuning_profiles() -> dict[str, Any]:
    """Candidate-owned, model-native tuning defaults (24 kHz PCM is fixed)."""
    return {
        "version": 2,
        # `selected` is a compatibility alias for older Realtime clients.
        "selected": {TUNING_PROVIDER: "balanced"},
        "selections": {
            "voice-studio": {TUNING_PROVIDER: "balanced"},
            "realtime": {TUNING_PROVIDER: "balanced"},
        },
        "profiles": {
        "quality": {"id": "quality", "name": "Quality", "revision": 2, "model": None, "clone_mode": "full_icl", "max_reference_seconds": 30, "first_block_frames": 6, "steady_block_frames": 16, "left_context_frames": MODEL_REQUIRED_LEFT_CONTEXT_FRAMES, "crossfade_samples": 0, "text_lookahead": 128, "phrase_flush_ms": 900, "temperature": 0.8, "top_k": 50, "top_p": 0.95, "repetition_penalty": 1.05, "seed": None},
        "balanced": {"id": "balanced", "name": "Balanced", "revision": 2, "model": None, "clone_mode": "full_icl", "max_reference_seconds": 20, "first_block_frames": 4, "steady_block_frames": 12, "left_context_frames": MODEL_REQUIRED_LEFT_CONTEXT_FRAMES, "crossfade_samples": 0, "text_lookahead": 64, "phrase_flush_ms": 500, "temperature": 1.0, "top_k": 50, "top_p": 0.95, "repetition_penalty": 1.05, "seed": None},
        "low-latency": {"id": "low-latency", "name": "Low Latency", "revision": 2, "model": None, "clone_mode": "full_icl", "max_reference_seconds": 12, "first_block_frames": 2, "steady_block_frames": 8, "left_context_frames": MODEL_REQUIRED_LEFT_CONTEXT_FRAMES, "crossfade_samples": 0, "text_lookahead": 32, "phrase_flush_ms": 250, "temperature": 1.0, "top_k": 40, "top_p": 0.9, "repetition_penalty": 1.05, "seed": None},
    }}


def _normalize_tuning_document(saved: Any) -> dict[str, Any]:
    """Migrate tuning definitions while keeping each surface selection independent."""
    defaults = _default_tuning_profiles()
    source = saved if isinstance(saved, dict) else {}
    source_profiles = source.get("profiles") if isinstance(source.get("profiles"), dict) else {}
    profiles = dict(source_profiles)
    for profile_id, default in defaults["profiles"].items():
        # Built-ins are versioned product defaults, not user-editable records.
        # Old documents may contain mutations from the earlier editor; replace
        # them during migration while leaving every custom profile untouched.
        profiles[profile_id] = dict(default)

    legacy_selected = source.get("selected") if isinstance(source.get("selected"), dict) else {}
    legacy_profile = str(legacy_selected.get(TUNING_PROVIDER) or "balanced")
    if legacy_profile not in profiles:
        legacy_profile = "balanced"
    saved_selections = source.get("selections") if isinstance(source.get("selections"), dict) else {}
    selections: dict[str, dict[str, str]] = {}
    for scope in sorted(TUNING_SCOPES):
        scoped = saved_selections.get(scope) if isinstance(saved_selections.get(scope), dict) else {}
        selected = str(scoped.get(TUNING_PROVIDER) or legacy_profile or "balanced")
        if selected not in profiles:
            selected = "balanced"
        selections[scope] = {TUNING_PROVIDER: selected}
    realtime_selected = selections["realtime"][TUNING_PROVIDER]
    return {
        **source,
        "version": defaults["version"],
        "profiles": profiles,
        "selections": selections,
        "selected": {TUNING_PROVIDER: realtime_selected},
    }


def _selected_tuning_profile(document: dict[str, Any], scope: str) -> str:
    """Resolve one surface's selection with Balanced as the durable fallback."""
    if scope not in TUNING_SCOPES:
        raise HTTPException(status_code=422, detail="Tuning scope must be voice-studio or realtime.")
    selected = str(document.get("selections", {}).get(scope, {}).get(TUNING_PROVIDER) or "balanced")
    return selected if selected in document.get("profiles", {}) else "balanced"


def _read_tuning_profiles() -> dict[str, Any]:
    try:
        saved = json.loads(TUNING_PROFILES_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        saved = {}
    return _normalize_tuning_document(saved)


def _write_tuning_profiles(document: dict[str, Any]) -> dict[str, Any]:
    document = _normalize_tuning_document(document)
    VOICE_LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    temporary = TUNING_PROFILES_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(TUNING_PROFILES_PATH)
    _record_event("tts-tuning-profiles-saved")
    return document


def _normalize_seed_value(value: Any) -> int | None:
    """Normalize UI randomness sentinels while accepting only a real uint32."""
    if isinstance(value, str):
        value = value.strip()
    if value is None or value == "" or value == -1 or value == "-1":
        return None
    if isinstance(value, bool):
        raise HTTPException(status_code=422, detail="seed must be null or uint32.")
    if isinstance(value, str):
        if not value.isdecimal():
            raise HTTPException(status_code=422, detail="seed must be null or uint32.")
        value = int(value)
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise HTTPException(status_code=422, detail="seed must be null or uint32.")
        value = int(value)
    if not isinstance(value, int) or not 0 <= value <= 2**32 - 1:
        raise HTTPException(status_code=422, detail="seed must be null or uint32.")
    return value


def _validate_tuning_profile(profile: dict[str, Any]) -> None:
    if profile.get("clone_mode", "full_icl") != "full_icl":
        raise HTTPException(status_code=422, detail="x_vector_only_mode is not supported by this Base candidate.")
    for field in ("first_block_frames", "steady_block_frames", "left_context_frames"):
        value = profile.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 300:
            raise HTTPException(status_code=422, detail=f"{field} must be 1-300 whole 80 ms codec frames.")
    if profile["steady_block_frames"] < profile["first_block_frames"]:
        raise HTTPException(status_code=422, detail="steady_block_frames must be at least first_block_frames.")
    if profile["left_context_frames"] + profile["steady_block_frames"] > 300:
        raise HTTPException(status_code=422, detail="left_context_frames + steady_block_frames must not exceed 300 codec frames.")
    if isinstance(profile.get("max_reference_seconds"), bool) or not isinstance(profile.get("max_reference_seconds"), int) or not 1 <= profile["max_reference_seconds"] <= 30:
        raise HTTPException(status_code=422, detail="max_reference_seconds must be 1-30.")
    for key, minimum, maximum in (("text_lookahead", 16, 512), ("phrase_flush_ms", 50, 3000), ("top_k", 1, 200)):
        if key in profile and (isinstance(profile[key], bool) or not isinstance(profile[key], int) or not minimum <= profile[key] <= maximum):
            raise HTTPException(status_code=422, detail=f"{key} must be a whole number from {minimum}-{maximum}.")
    for key, minimum, maximum in (("top_p", .05, 1), ("repetition_penalty", .8, 2)):
        if key in profile and (isinstance(profile[key], bool) or not isinstance(profile[key], (int, float)) or not minimum <= profile[key] <= maximum):
            raise HTTPException(status_code=422, detail=f"{key} must be {minimum}-{maximum}.")
    temperature = profile.get("temperature")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 < temperature <= 2:
        raise HTTPException(status_code=422, detail="temperature must be greater than 0 and at most 2.")
    profile["seed"] = _normalize_seed_value(profile.get("seed"))
    if profile.get("model") is not None and profile["model"] not in MIN_FREE_MIB:
        raise HTTPException(status_code=422, detail="model must be an exact candidate model id.")


def _guard_policy(model_id: str, operation: str) -> tuple[int, int, bool]:
    """Return free-VRAM threshold, utilization ceiling, and manual bypass flag."""
    mode = str(gpu_guard_settings["mode"])
    if operation == "synthesis":
        default_required = MIN_SYNTHESIS_FREE_MIB
        configured_required = int(gpu_guard_settings["synthesis_min_free_mib"])
        configured_utilization = int(gpu_guard_settings["synthesis_max_utilization_percent"])
    else:
        default_required = MIN_FREE_MIB[model_id]
        configured_required = int(gpu_guard_settings["load_min_free_mib"])
        configured_utilization = int(gpu_guard_settings["load_max_utilization_percent"])
    required = configured_required if mode == "custom" else default_required
    return required, configured_utilization, mode == "disabled"


def _record_event(action: str, *, model_id: str | None = None, **details: Any) -> dict[str, Any]:
    """Keep a small user-visible lifecycle trail without exposing request data."""
    event = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "action": action,
        "model": model_id or state.get("activeModel"),
        **details,
    }
    EVENTS.append(event)
    state["lastAction"] = action
    logger.info("candidate action=%s model=%s details=%s", action, event["model"], details)
    return event


def _native_request_diagnostics(
    payload: dict[str, Any],
    tuning_snapshot: dict[str, Any],
    engine_payload: dict[str, Any],
    reference_stats: dict[str, Any],
    model_id: str,
) -> dict[str, Any]:
    """Return non-text identifiers and frozen settings for one native request.

    This deliberately omits the phrase and reference transcript. The same
    dictionary is attached to every lifecycle event for the request, which
    makes a timing trace useful without turning the candidate event trail into
    a prompt log.
    """
    supplied_id = payload.get("_outcome_id", payload.get("request_id", payload.get("requestId")))
    if supplied_id is None:
        supplied_id = payload.get("response_id", payload.get("responseId"))
    request_id = str(supplied_id or token_hex(12)).strip()
    if not request_id or len(request_id) > 160:
        request_id = token_hex(12)
    effective = tuning_snapshot.get("effective", {})
    if not isinstance(effective, dict):
        effective = {}
    clone_id = str(payload.get("voice") or "")
    if clone_id.startswith("clone:"):
        clone_id = clone_id.removeprefix("clone:")
    else:
        clone_id = None
    return {
        "requestId": request_id,
        "provider": TUNING_PROVIDER,
        "modelId": model_id,
        "engineEpoch": int(payload.get("_engine_epoch") or state.get("engineEpoch") or 0),
        "supervisorInstanceId": SUPERVISOR_INSTANCE_ID,
        "tuningScope": tuning_snapshot.get("scope"),
        "tuningProfileId": tuning_snapshot.get("id"),
        "tuningProfileRevision": tuning_snapshot.get("revision"),
        "effectiveFirstBlockFrames": effective.get("first_block_frames"),
        "effectiveSteadyBlockFrames": effective.get("steady_block_frames"),
        "effectiveLeftContextFrames": effective.get("left_context_frames"),
        "responseSeed": engine_payload.get("seed"),
        "cloneId": clone_id,
        "cloneContentHash": payload.get("_frozen_clone_content_hash"),
        "cloneContentRevision": payload.get("_frozen_clone_content_revision"),
        **_reference_event_fields(reference_stats, "native-incremental-pcm"),
    }


def _engine_stage_timings(response: httpx.Response | Any) -> dict[str, str]:
    """Expose engine timing headers when a patched engine provides them.

    The pinned audio.cpp server does not promise a fixed stage-header schema,
    so this is intentionally pass-through telemetry rather than fabricated
    numbers. It is correlated with the supervisor request id in the event.
    """
    headers = getattr(response, "headers", {})
    return {
        str(name): str(value)
        for name, value in headers.items()
        if str(name).lower().startswith(("x-audiocpp-", "x-engine-"))
        and str(name).lower() != ENGINE_DECODE_MODE_HEADER.lower()
    }


def _template() -> dict[str, Any]:
    return json.loads(TEMPLATE.read_text(encoding="utf-8"))


def _models() -> dict[str, dict[str, Any]]:
    return {str(model["id"]): model for model in _template()["models"]}


def gpu_guard(
    model_id: str,
    minimum_free_mib: int | None = None,
    max_busy_percent: int = MAX_BUSY_PERCENT,
    *,
    operation: str = "load",
    apply_policy: bool = True,
) -> dict[str, Any]:
    try:
        line = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True, timeout=5).strip().splitlines()[0]
        free_mib, utilization = (int(value.strip()) for value in line.split(",")[:2])
    except Exception as exc:
        if apply_policy and str(gpu_guard_settings["mode"]) == "disabled":
            return {
                "ok": True,
                "bypassed": True,
                "reason": f"GPU guard manually disabled; GPU status unavailable ({type(exc).__name__}) and operation will be attempted.",
                "operation": operation,
                "guardMode": "disabled",
            }
        return {"ok": False, "reason": f"GPU status unavailable: {type(exc).__name__}: {exc}"}
    policy_required, policy_utilization, bypassed = _guard_policy(model_id, operation)
    required = minimum_free_mib if minimum_free_mib is not None else policy_required
    utilization_limit = max_busy_percent if minimum_free_mib is not None else policy_utilization
    details = {
        "freeMiB": free_mib,
        "requiredMiB": required,
        "utilizationPercent": utilization,
        "maxUtilizationPercent": utilization_limit,
        "operation": operation,
        "guardMode": gpu_guard_settings["mode"],
    }
    if operation == "load" and minimum_free_mib is None and str(gpu_guard_settings["mode"]) == "enforced":
        details.update(_enforced_gpu_policy()["models"][model_id])
    elif operation == "load" and minimum_free_mib is None and str(gpu_guard_settings["mode"]) == "custom":
        details["customAbsoluteLoadThreshold"] = True
    if apply_policy and bypassed:
        return {
            "ok": True,
            "bypassed": True,
            "reason": "GPU guard manually disabled by candidate user; load or synthesis may still fail if VRAM is exhausted.",
            **details,
        }
    if free_mib < required:
        if operation == "load" and details.get("admissionKind") == "measured-residency-plus-synthesis-floor":
            reason = (
                f"GPU load blocked: {free_mib} MiB free, but enforced admission requires {required} MiB "
                f"({details['residencyReserveMiB']} MiB measured/rounded model residency + "
                f"{details['postLoadSynthesisReserveMiB']} MiB post-load synthesis floor)."
            )
        elif operation == "load" and details.get("admissionKind") == "existing-total-threshold":
            reason = (
                f"GPU load blocked: {free_mib} MiB free, but enforced admission requires the existing "
                f"{required} MiB total threshold; this model's residency delta has not been measured, "
                "so no unverified reserve was added."
            )
        else:
            reason = f"GPU insufficient VRAM: {free_mib} MiB free, {required} MiB required; operation was not attempted."
        return {"ok": False, "reason": reason, **details}
    if utilization >= utilization_limit:
        return {"ok": False, "reason": f"GPU busy: {utilization}% utilization meets or exceeds the {utilization_limit}% limit; operation was not attempted.", **details}
    return {"ok": True, **details}


def _reconcile_engine() -> None:
    """Report a child killed by external GPU pressure as an error, not unload."""
    global engine
    if state["state"] == "loaded" and (engine is None or engine.poll() is not None):
        exit_code = engine.returncode if engine is not None else None
        message = f"audio.cpp child exited unexpectedly (exit={exit_code})."
        state.update(state="evicted", reason=message, lastError=message)
        _record_event("child-exit", exitCode=exit_code, gpu=gpu_guard(state["activeModel"], minimum_free_mib=0, apply_policy=False))
        engine = None


def _stop_engine(*, action: str = "release") -> None:
    global engine
    if engine and engine.poll() is None:
        engine.terminate()
        try:
            engine.wait(timeout=20)
        except subprocess.TimeoutExpired:
            engine.kill()
            engine.wait(timeout=10)
    engine = None
    _record_event(action)


def _start_engine(model_id: str) -> None:
    global engine
    model = dict(_models()[model_id])
    model["session_options"] = {
        **dict(model.get("session_options") or {}),
        # Retain the talker graph for native-streaming benchmarks.  This does
        # not change the BF16 checkpoint or permit a second resident model.
        "qwen3_tts.mem_saver": "false" if NATIVE_INCREMENTAL_PCM_ENABLED else "true",
    }
    if NATIVE_INCREMENTAL_PCM_ENABLED and TALKER_PREFIX_CACHE_SLOTS > 0:
        # Cache only immutable clone/language/control/reference talker state.
        # Phrase-dependent suffix rows are replayed and generated speech state
        # remains request-local inside audio.cpp.
        model["session_options"]["qwen3_tts.talker_prefix_cache_slots"] = str(
            TALKER_PREFIX_CACHE_SLOTS
        )
    config = _template()
    config["port"] = 8081
    config["models"] = [model]
    # Keep the published buffered candidate in Offline mode.  The experimental
    # engine accepts Streaming only when the separately enabled native path is
    # installed and validated.
    config["models"][0]["mode"] = _configured_model_mode()
    ACTIVE_CONFIG.write_text(json.dumps(config), encoding="utf-8")
    engine = subprocess.Popen([ENGINE_BIN, "--config", str(ACTIVE_CONFIG)])
    _record_event(
        "child-start",
        model_id=model_id,
        memSaver=not NATIVE_INCREMENTAL_PCM_ENABLED,
        talkerPrefixCacheSlots=TALKER_PREFIX_CACHE_SLOTS if NATIVE_INCREMENTAL_PCM_ENABLED else 0,
    )


async def _engine_ready() -> bool:
    async with httpx.AsyncClient(timeout=1.0) as client:
        try:
            return (await client.get(f"{ENGINE_URL}/health")).is_success
        except httpx.HTTPError:
            return False


async def _complete_native_load_warmup(model_id: str) -> None:
    """Bound the private warmup and leave a truthful, retryable state."""
    try:
        warmup = await asyncio.wait_for(
            _run_native_load_warmup(model_id),
            timeout=NATIVE_LOAD_WARMUP_TIMEOUT_SECONDS,
        )
        state.update(
            lastWarmupElapsedS=warmup.get("elapsedSeconds"),
            lastWarmupProfileId=warmup.get("profileId"),
            lastWarmupStatus=warmup.get("status"),
            lastWarmupError=None,
            reason=warmup.get("reason"),
        )
    except asyncio.CancelledError:
        _stop_engine(action="warmup-cancelled-release")
        message = "Native load warmup was cancelled; the candidate model was released."
        state.update(state="error", reason=message, lastError=message, lastWarmupStatus="cancelled", lastWarmupError=message)
        _record_event("model-warmup-cancelled", model_id=model_id)
        raise
    except Exception as exc:
        message = f"Native load warmup failed; the first synthesis may remain cold: {exc}"
        state.update(lastWarmupStatus="failed", lastWarmupError=message, reason=message)
        _record_event("model-warmup-failed", model_id=model_id, error=str(exc))
    if not await _engine_ready():
        _stop_engine(action="warmup-failed-release")
        message = "audio.cpp stopped responding during native load warmup."
        state.update(state="error", reason=message, lastError=message)
        raise HTTPException(status_code=503, detail={"state": "error", "message": message})


async def switch_model(model_id: str) -> dict[str, Any]:
    if model_id not in _models():
        raise HTTPException(status_code=404, detail="Unknown candidate model.")
    async with switch_lock:
        _reconcile_engine()
        warm_after_load = NATIVE_INCREMENTAL_PCM_ENABLED and NATIVE_LOAD_WARMUP_ENABLED
        if state["state"] == "loaded" and state["activeModel"] == model_id and await _engine_ready():
            if not warm_after_load or state.get("lastWarmupStatus") == "complete":
                _record_event("load-noop", model_id=model_id, reason="selected model is already resident and ready")
                return {**state, "singleResident": True, "events": list(EVENTS)}
            started = time.monotonic()
            state.update(
                state="warming",
                reason="Retrying the native decoder warmup.",
                lastError=None,
                lastWarmupStatus="pending",
                lastWarmupError=None,
            )
            await _complete_native_load_warmup(model_id)
            state.update(state="loaded", lastError=None, lastLoadElapsedS=round(time.monotonic() - started, 3))
            _record_event(
                "model-ready",
                model_id=model_id,
                loadElapsedS=state["lastLoadElapsedS"],
                warmupStatus=state["lastWarmupStatus"],
                warmupElapsedS=state["lastWarmupElapsedS"],
                warmupRetry=True,
            )
            return {**state, "singleResident": True, "events": list(EVENTS)}
        guard = gpu_guard(model_id, operation="load")
        if not guard["ok"]:
            state.update(reason=guard["reason"], lastError=guard["reason"])
            _record_event("gpu-blocked-load", model_id=model_id, gpu=guard)
            raise HTTPException(status_code=409, detail={"state": "blocked", **guard})
        state.update(state="loading", reason=None, lastError=None)
        state["engineEpoch"] = int(state.get("engineEpoch") or 0) + 1
        async with generation_lock:
            _record_event("load-requested", model_id=model_id, gpu=guard)
            if engine is not None:
                _stop_engine(action="old-model-released")  # release old model before the new child exists
            started = time.monotonic()
            _start_engine(model_id)
            for _ in range(120):
                if await _engine_ready():
                    break
                await asyncio.sleep(0.25)
            else:
                _stop_engine(action="load-failed-release")
                message = "audio.cpp did not become healthy after the model switch."
                state.update(activeModel=model_id, state="error", reason=message, lastError=message)
                raise HTTPException(status_code=503, detail={"state": "error", "message": message})
            state.update(
                activeModel=model_id,
                state="warming" if warm_after_load else "loaded",
                reason="Preparing the native decoder for consistent first-turn latency." if warm_after_load else None,
                lastError=None,
                lastWarmupElapsedS=None,
                lastWarmupProfileId=None,
                lastWarmupStatus="pending" if warm_after_load else "disabled",
                lastWarmupError=None,
                gpu=guard,
            )
        if warm_after_load:
            await _complete_native_load_warmup(model_id)
        state.update(
            state="loaded",
            lastError=None,
            lastLoadElapsedS=round(time.monotonic() - started, 3),
        )
        _record_event(
            "model-ready",
            model_id=model_id,
            loadElapsedS=state["lastLoadElapsedS"],
            warmupStatus=state["lastWarmupStatus"],
            warmupElapsedS=state["lastWarmupElapsedS"],
            gpu=guard,
        )
        return {**state, "singleResident": True, "events": list(EVENTS)}


@app.on_event("startup")
async def startup() -> None:
    # The engine is deliberately not started until Voice Studio explicitly
    # loads a model. This makes the initial resident set truthful and empty.
    state["state"] = "unloaded"
    _record_event("supervisor-started", model_id=ACTIVE_MODEL)


@app.on_event("shutdown")
async def shutdown() -> None: _stop_engine(action="supervisor-shutdown")


@app.get("/health")
async def health() -> dict[str, Any]:
    _reconcile_engine()
    loaded = state["state"] == "loaded" and await _engine_ready()
    return {
        "status": "ok",
        "backend": {
            "name": "audio.cpp CUDA candidate",
            "model_id": state["activeModel"] if loaded else None,
            "current_model_key": state["activeModel"],
            "loaded_models": [state["activeModel"]] if loaded else [],
            "runtime": {
                "state": state["state"],
                "last_error": state["lastError"],
                "last_load_elapsed_s": state["lastLoadElapsedS"],
                "gpu": state["gpu"],
                "events": list(EVENTS),
                "mem_saver": not NATIVE_INCREMENTAL_PCM_ENABLED,
                "native_incremental_pcm": NATIVE_INCREMENTAL_PCM_ENABLED,
                "native_load_warmup": NATIVE_LOAD_WARMUP_ENABLED,
                "native_load_warmup_timeout_seconds": NATIVE_LOAD_WARMUP_TIMEOUT_SECONDS,
                "cuda_graphs_disabled": None,
                "cuda_graphs_control_supported": False,
                "cuda_graphs_mode": "engine-default-uncontrolled",
                "talker_prefix_cache_slots": TALKER_PREFIX_CACHE_SLOTS if NATIVE_INCREMENTAL_PCM_ENABLED else 0,
                "talker_prefix_cache_state": (
                    "configured-experimental"
                    if NATIVE_INCREMENTAL_PCM_ENABLED and TALKER_PREFIX_CACHE_SLOTS > 0
                    else "disabled"
                ),
                "progressive_phrase_pcm": True,
                "sample_rate": 24000,
                "sample_format": "pcm_s16le",
                "load_headroom_mib": _guard_policy(str(state["activeModel"]), "load")[0],
                "synthesis_headroom_mib": _guard_policy(str(state["activeModel"]), "synthesis")[0],
                "enforced_gpu_policy": _enforced_gpu_policy(),
                "gpu_guard": dict(gpu_guard_settings),
            },
        },
        **state,
        "supervisorInstanceId": SUPERVISOR_INSTANCE_ID,
        "singleResident": True,
        "nativeIncrementalPcm": NATIVE_INCREMENTAL_PCM_ENABLED,
        "ttsMode": "native-incremental-pcm" if NATIVE_INCREMENTAL_PCM_ENABLED else "offline-buffered",
    }


@app.get("/control/status")
async def control_status() -> dict[str, Any]:
    _reconcile_engine()
    return {
        **state,
        "supervisorInstanceId": SUPERVISOR_INSTANCE_ID,
        "singleResident": True,
        "availableModels": list(_models()),
        "events": list(EVENTS),
        "gpu_guard": dict(gpu_guard_settings),
        "enforced_gpu_policy": _enforced_gpu_policy(),
    }


@app.post("/control/switch")
async def control_switch(payload: dict[str, str]) -> dict[str, Any]: return await switch_model(str(payload.get("model") or ""))


def _update_gpu_guard(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and persist explicit candidate-only GPU admission settings."""
    mode = str(payload.get("mode", gpu_guard_settings["mode"])).strip().lower()
    if mode not in GPU_GUARD_MODES:
        raise HTTPException(status_code=422, detail="GPU guard mode must be enforced, custom, or disabled.")
    updated = dict(gpu_guard_settings)
    updated["mode"] = mode
    for key in ("load_min_free_mib", "synthesis_min_free_mib"):
        if key in payload:
            try:
                value = int(payload[key])
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=422, detail=f"{key} must be an integer.") from exc
            if not 0 <= value <= 16384:
                raise HTTPException(status_code=422, detail=f"{key} must be between 0 and 16384 MiB.")
            updated[key] = value
    for key in ("load_max_utilization_percent", "synthesis_max_utilization_percent"):
        if key in payload:
            try:
                value = int(payload[key])
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=422, detail=f"{key} must be an integer.") from exc
            if not 1 <= value <= 100:
                raise HTTPException(status_code=422, detail=f"{key} must be between 1 and 100 percent.")
            updated[key] = value
    gpu_guard_settings.clear()
    gpu_guard_settings.update(updated)
    _save_gpu_guard_settings()
    _record_event("gpu-guard-updated", mode=mode, settings=dict(gpu_guard_settings))
    return {
        "gpu_guard": dict(gpu_guard_settings),
        "enforced_gpu_policy": _enforced_gpu_policy(),
        "warning": "Disabled mode bypasses admission checks only; it cannot prevent a CUDA out-of-memory failure.",
    }


@app.get("/control/gpu-guard")
async def control_gpu_guard() -> dict[str, Any]:
    return {"gpu_guard": dict(gpu_guard_settings), "enforced_gpu_policy": _enforced_gpu_policy()}


@app.post("/control/gpu-guard")
async def control_gpu_guard_update(payload: dict[str, Any]) -> dict[str, Any]:
    return _update_gpu_guard(payload)


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    active = str(state["activeModel"])
    mode = _configured_model_mode()
    return {"object": "list", "data": [{"id": model_id, "object": "model", "owned_by": "engine", "family": "qwen3_tts", "task": "tts", "mode": mode, "active": model_id == active} for model_id in _models()]}


@app.get("/v1/voices")
async def voices() -> dict[str, Any]:
    """Candidate-private Base clone inventory for compatible callers."""
    return {"object": "list", "data": _voice_profiles()}


def _candidate_profile_path(profile_id: str) -> Path:
    if not valid_profile_id(profile_id):
        raise HTTPException(status_code=422, detail="Invalid clone profile id.")
    path = VOICE_LIBRARY_DIR / "profiles" / profile_id
    if not path.is_dir():
        raise HTTPException(status_code=404, detail="Clone profile is unavailable.")
    return path


def _candidate_profile_payload(profile_id: str, *, include_audio: bool = False) -> dict[str, Any]:
    path = _candidate_profile_path(profile_id)
    meta = canonical_profile(VOICE_LIBRARY_DIR, profile_id, normalize=True)
    if meta is None:
        raise HTTPException(status_code=409, detail="Clone profile metadata or reference audio is unavailable.")
    reference = path / str(meta["ref_audio_filename"])
    result = {**meta, "id": profile_id, "voice": f"clone:{profile_id}"}
    if include_audio:
        if not reference.is_file():
            raise HTTPException(status_code=409, detail="Clone profile reference audio is missing.")
        result["ref_audio"] = base64.b64encode(reference.read_bytes()).decode("ascii")
        frozen_excerpts: list[dict[str, str]] = []
        for excerpt in meta.get("reference_excerpts") or []:
            if not isinstance(excerpt, dict):
                continue
            filename = str(excerpt.get("ref_audio_filename") or "").strip()
            transcript = str(excerpt.get("ref_text") or "").strip()
            excerpt_path = path / filename
            if filename and Path(filename).name == filename and transcript and excerpt_path.is_file():
                frozen_excerpts.append(
                    {
                        "ref_audio": base64.b64encode(excerpt_path.read_bytes()).decode("ascii"),
                        "ref_text": transcript,
                    }
                )
        result["reference_excerpts"] = frozen_excerpts
    return result


def _candidate_profile_response() -> dict[str, Any]:
    selected = None
    try:
        selected = str(json.loads((VOICE_LIBRARY_DIR / "selected_profile.json").read_text(encoding="utf-8")).get("profile_id") or "")
    except (OSError, ValueError):
        pass
    profiles = [_candidate_profile_payload(item["id"]) for item in _voice_profiles()]
    live_ids = {str(profile["id"]) for profile in profiles}
    if selected not in live_ids:
        selected = None
    return {
        "backend": "qwen3tts-audiocpp",
        "writable": True,
        "selectedVoice": f"clone:{selected}" if selected else None,
        "defaultVoice": profiles[0]["voice"] if profiles else None,
        "voices": profiles,
    }


@app.get("/v1/voices/profiles")
async def voice_profiles() -> dict[str, Any]:
    return _candidate_profile_response()


@app.get("/v1/voices/profiles/{profile_id}")
async def voice_profile(profile_id: str) -> dict[str, Any]:
    return _candidate_profile_payload(profile_id, include_audio=True)


@app.post("/v1/voices/profiles")
async def create_voice_profile(payload: dict[str, Any]) -> dict[str, Any]:
    source_id = str(payload.get("source_profile_id") or "")
    if source_id and not payload.get("ref_audio"):
        source = _candidate_profile_payload(source_id, include_audio=True)
        payload = {**source, **payload, "ref_audio": source["ref_audio"]}
    profile_id = token_hex(6)
    _write_candidate_profile(profile_id, payload)
    return _candidate_profile_response()


@app.patch("/v1/voices/profiles/{profile_id}")
async def edit_voice_profile(profile_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    current = _candidate_profile_payload(profile_id, include_audio=True)
    _write_candidate_profile(profile_id, {**current, **payload, "ref_audio": current["ref_audio"]})
    _record_event("profile-edited", profile_id=profile_id)
    return _candidate_profile_response()


@app.delete("/v1/voices/profiles/{profile_id}")
async def delete_voice_profile(profile_id: str) -> dict[str, Any]:
    path = _candidate_profile_path(profile_id)
    shutil.rmtree(path)
    selected_path = VOICE_LIBRARY_DIR / "selected_profile.json"
    try:
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        selected = {}
    if isinstance(selected, dict) and selected.get("profile_id") == profile_id:
        selected_path.unlink(missing_ok=True)
    _record_event("profile-deleted", profile_id=profile_id)
    return _candidate_profile_response()


@app.post("/v1/voices/profiles/{profile_id}/select")
async def select_voice_profile(profile_id: str) -> dict[str, Any]:
    _candidate_profile_path(profile_id)
    VOICE_LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    temporary = VOICE_LIBRARY_DIR / ".selected_profile.json.tmp"
    temporary.write_text(json.dumps({"profile_id": profile_id}, indent=2) + "\n", encoding="utf-8")
    temporary.replace(VOICE_LIBRARY_DIR / "selected_profile.json")
    return _candidate_profile_response()


@app.get("/v1/voice-studio/settings")
async def voice_studio_settings() -> dict[str, Any]:
    return {"settings": _read_voice_studio_settings(), "apiKeyPersisted": False}


@app.put("/v1/voice-studio/settings")
async def save_voice_studio_settings(payload: dict[str, Any]) -> dict[str, Any]:
    return {"settings": _write_voice_studio_settings(payload), "apiKeyPersisted": False}


@app.get("/v1/tuning/profiles")
async def get_tuning_profiles() -> dict[str, Any]:
    return _read_tuning_profiles()


def _profile_values_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Read the complete safe schema from Save As or legacy flat requests."""
    nested = payload.get("values")
    if nested is not None and not isinstance(nested, dict):
        raise HTTPException(status_code=422, detail="Profile values must be an object.")
    values = dict(nested or {})
    unknown = set(values) - TUNING_VALUE_FIELDS
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported tuning profile field(s): {', '.join(sorted(unknown))}.",
        )
    for key in TUNING_VALUE_FIELDS:
        if key in payload:
            values[key] = payload[key]
    if values.get("clone_mode", "full_icl") != "full_icl":
        raise HTTPException(
            status_code=422,
            detail="x_vector_only_mode is not supported by this Base candidate.",
        )
    if "seed" in values:
        values["seed"] = _normalize_seed_value(values["seed"])
    return values


def _created_profile_response(
    document: dict[str, Any], profile_id: str, scope: str | None,
) -> dict[str, Any]:
    """Keep the document response compatible while identifying Save As state."""
    selected = _selected_tuning_profile(document, scope) if scope else None
    return {
        **document,
        "profile": document["profiles"][profile_id],
        "created_profile_id": profile_id,
        "selected_profile_id": selected,
        "selected_scope": scope,
    }


@app.post("/v1/tuning/profiles")
async def create_tuning_profile(payload: dict[str, Any]) -> dict[str, Any]:
    document = _read_tuning_profiles()
    provider = str(payload.get("provider") or TUNING_PROVIDER)
    if provider != TUNING_PROVIDER:
        raise HTTPException(status_code=422, detail="Tuning profiles are available only for qwen3tts-audiocpp.")
    profile_id = str(payload.get("id") or token_hex(5))
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", profile_id):
        raise HTTPException(status_code=422, detail="Invalid tuning profile id.")
    if profile_id in document["profiles"]:
        raise HTTPException(status_code=409, detail="A tuning profile with that id already exists; built-ins and custom profiles are never overwritten.")
    clone_from = str(payload.get("clone_from") or "balanced")
    source_profile = document["profiles"].get(clone_from)
    if not isinstance(source_profile, dict):
        raise HTTPException(status_code=404, detail="Unknown source tuning profile.")
    source = dict(source_profile)
    source.update(_profile_values_from_payload(payload))
    if "name" in payload:
        source["name"] = str(payload["name"])
    source.update(id=profile_id, revision=1, clone_mode="full_icl", crossfade_samples=0)
    _validate_tuning_profile(source)
    document["profiles"][profile_id] = source
    scope: str | None = None
    if bool(payload.get("select")):
        scope = str(payload.get("scope") or "realtime")
        if scope not in TUNING_SCOPES:
            raise HTTPException(status_code=422, detail="Tuning scope must be voice-studio or realtime.")
        document.setdefault("selections", {}).setdefault(scope, {})[TUNING_PROVIDER] = profile_id
        if scope == "realtime":
            document.setdefault("selected", {})[TUNING_PROVIDER] = profile_id
    written = _write_tuning_profiles(document)
    return _created_profile_response(written, profile_id, scope)


@app.patch("/v1/tuning/profiles/{profile_id}")
async def patch_tuning_profile(profile_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    document = _read_tuning_profiles()
    profile = document["profiles"].get(profile_id)
    if not isinstance(profile, dict):
        raise HTTPException(status_code=404, detail="Unknown tuning profile.")
    if profile_id in BUILTIN_TUNING_PROFILE_IDS:
        raise HTTPException(status_code=409, detail="Built-in tuning profiles are immutable; clone one before editing.")
    expected_revision = payload.get("revision")
    if isinstance(expected_revision, bool) or not isinstance(expected_revision, int):
        raise HTTPException(status_code=422, detail="A numeric profile revision is required for updates.")
    if expected_revision != profile.get("revision", 1):
        raise HTTPException(status_code=409, detail={"state": "conflict", "current_revision": profile.get("revision", 1)})
    unknown = set(payload) - ({"revision", "name", "values"} | TUNING_VALUE_FIELDS)
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unsupported tuning profile field(s): {', '.join(sorted(unknown))}.")
    if "name" in payload:
        profile["name"] = str(payload["name"])
    profile.update(_profile_values_from_payload(payload))
    profile.update(clone_mode="full_icl", crossfade_samples=0, revision=int(profile.get("revision", 1)) + 1)
    _validate_tuning_profile(profile)
    return _write_tuning_profiles(document)


@app.delete("/v1/tuning/profiles/{profile_id}")
async def delete_tuning_profile(profile_id: str) -> dict[str, Any]:
    document = _read_tuning_profiles()
    if profile_id in BUILTIN_TUNING_PROFILE_IDS:
        raise HTTPException(status_code=409, detail="Built-in tuning profiles cannot be deleted; reset them instead.")
    document["profiles"].pop(profile_id, None)
    for scope in TUNING_SCOPES:
        if document.get("selections", {}).get(scope, {}).get(TUNING_PROVIDER) == profile_id:
            document["selections"][scope][TUNING_PROVIDER] = "balanced"
    if document.get("selected", {}).get(TUNING_PROVIDER) == profile_id:
        document["selected"][TUNING_PROVIDER] = "balanced"
    return _write_tuning_profiles(document)


@app.post("/v1/tuning/profiles/{profile_id}/clone")
async def clone_tuning_profile(profile_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    document = _read_tuning_profiles()
    source = document["profiles"].get(profile_id)
    if not isinstance(source, dict):
        raise HTTPException(status_code=404, detail="Unknown tuning profile.")
    clone_id = str(payload.get("id") or token_hex(5))
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", clone_id):
        raise HTTPException(status_code=422, detail="Invalid clone id.")
    if clone_id in document["profiles"]:
        raise HTTPException(status_code=409, detail="A tuning profile with that id already exists.")
    clone = dict(source)
    clone.update(_profile_values_from_payload(payload))
    clone.update(id=clone_id, name=str(payload.get("name") or f"{source['name']} copy"), revision=1, clone_mode="full_icl", crossfade_samples=0)
    _validate_tuning_profile(clone)
    document["profiles"][clone_id] = clone
    scope: str | None = None
    if bool(payload.get("select")):
        scope = str(payload.get("scope") or "realtime")
        if scope not in TUNING_SCOPES:
            raise HTTPException(status_code=422, detail="Tuning scope must be voice-studio or realtime.")
        document.setdefault("selections", {}).setdefault(scope, {})[TUNING_PROVIDER] = clone_id
        if scope == "realtime":
            document.setdefault("selected", {})[TUNING_PROVIDER] = clone_id
    written = _write_tuning_profiles(document)
    return _created_profile_response(written, clone_id, scope)


@app.post("/v1/tuning/profiles/{profile_id}/reset")
async def reset_tuning_profile(profile_id: str) -> dict[str, Any]:
    defaults = _default_tuning_profiles()["profiles"]
    if profile_id not in defaults:
        raise HTTPException(status_code=409, detail="Only built-in tuning profiles can be reset.")
    document = _read_tuning_profiles()
    document["profiles"][profile_id] = defaults[profile_id]
    return _write_tuning_profiles(document)


@app.get("/v1/tuning/export")
async def export_tuning_profiles() -> dict[str, Any]:
    return {"schema": "qwen3tts-audiocpp.tuning/v1", "document": _read_tuning_profiles()}


@app.post("/v1/tuning/import")
async def import_tuning_profiles(payload: dict[str, Any]) -> dict[str, Any]:
    incoming = payload.get("document")
    if payload.get("schema") != "qwen3tts-audiocpp.tuning/v1" or not isinstance(incoming, dict) or not isinstance(incoming.get("profiles"), dict):
        raise HTTPException(status_code=422, detail="Invalid tuning profile export.")
    document = _read_tuning_profiles()
    imported: dict[str, dict[str, Any]] = {}
    for profile_id, profile in incoming["profiles"].items():
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", str(profile_id)) or not isinstance(profile, dict):
            raise HTTPException(status_code=422, detail="Invalid imported profile.")
        candidate = dict(profile)
        candidate.update(
            id=str(profile_id),
            clone_mode="full_icl",
            crossfade_samples=0,
            revision=1,
        )
        _validate_tuning_profile(candidate)
        if profile_id in BUILTIN_TUNING_PROFILE_IDS:
            continue
        if profile_id in document["profiles"]:
            raise HTTPException(
                status_code=409,
                detail={
                    "state": "profile-import-conflict",
                    "profile_id": profile_id,
                    "message": "Imported custom profiles never overwrite an existing id.",
                },
            )
        imported[str(profile_id)] = candidate
    document["profiles"].update(imported)
    return _write_tuning_profiles(document)


@app.put("/v1/tuning/selection")
async def select_tuning_profile(payload: dict[str, Any]) -> dict[str, Any]:
    document = _read_tuning_profiles()
    provider = str(payload.get("provider") or TUNING_PROVIDER)
    profile_id = str(payload.get("profile_id") or "balanced")
    scope = str(payload.get("scope") or "realtime")
    if provider != TUNING_PROVIDER:
        raise HTTPException(status_code=422, detail="Tuning profile selection is available only for qwen3tts-audiocpp.")
    if scope not in TUNING_SCOPES:
        raise HTTPException(status_code=422, detail="Tuning scope must be voice-studio or realtime.")
    if profile_id not in document["profiles"]:
        raise HTTPException(status_code=404, detail="Unknown tuning profile.")
    document.setdefault("selections", {}).setdefault(scope, {})[provider] = profile_id
    if scope == "realtime":
        document.setdefault("selected", {})[provider] = profile_id
    return _write_tuning_profiles(document)


@app.post("/v1/tuning/resolve")
async def resolve_tuning_profile(payload: dict[str, Any]) -> dict[str, Any]:
    document = _read_tuning_profiles()
    provider = str(payload.get("provider") or TUNING_PROVIDER)
    scope = str(payload.get("scope") or "realtime")
    if provider != TUNING_PROVIDER:
        raise HTTPException(status_code=422, detail="Tuning profiles are available only for qwen3tts-audiocpp.")
    selected = _selected_tuning_profile(document, scope)
    profile_id = str(payload.get("profile_id") or selected)
    profile = document["profiles"].get(profile_id)
    if not isinstance(profile, dict):
        raise HTTPException(status_code=404, detail="Unknown tuning profile.")
    if profile.get("model") and profile["model"] != state["activeModel"]:
        raise HTTPException(status_code=409, detail={"active": state["activeModel"], "required": profile["model"], "auto_switch": False})
    overrides = payload.get("overrides") or {}
    if not isinstance(overrides, dict):
        raise HTTPException(status_code=422, detail="Temporary overrides must be an object.")
    supported = TUNING_VALUE_FIELDS - {"clone_mode"}
    unknown = set(overrides) - supported
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unsupported tuning override field(s): {', '.join(sorted(unknown))}.")
    requested_model = overrides.get("model")
    if requested_model and requested_model != state["activeModel"]:
        raise HTTPException(status_code=409, detail={"active": state["activeModel"], "required": requested_model, "auto_switch": False})
    bounds = {"text_lookahead": (16, 512), "phrase_flush_ms": (50, 3000), "top_k": (1, 200), "top_p": (.05, 1), "repetition_penalty": (.8, 2)}
    for key, (minimum, maximum) in bounds.items():
        if key in overrides and (not isinstance(overrides[key], (int, float)) or not minimum <= overrides[key] <= maximum):
            raise HTTPException(status_code=422, detail=f"{key} must be {minimum}-{maximum}.")
    if "temperature" in overrides:
        temperature = overrides["temperature"]
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 < temperature <= 2:
            raise HTTPException(status_code=422, detail="temperature must be greater than 0 and at most 2.")
    if "seed" in overrides:
        overrides = dict(overrides)
        overrides["seed"] = _normalize_seed_value(overrides["seed"])
    candidate = dict(profile)
    candidate.update(overrides)
    _validate_tuning_profile(candidate)
    native_fields = ["first_block_frames", "steady_block_frames", "left_context_frames"]
    effective_fields = ["temperature", "top_k", "top_p", "repetition_penalty", "seed"]
    if NATIVE_INCREMENTAL_PCM_ENABLED:
        effective_fields.extend(native_fields)
    warnings = [
        "crossfade is sample-aligned and fixed at zero",
        "codec frames are 80 ms",
        "blank and -1 seeds request engine randomness; explicit seeds are uint32",
        "max_reference_seconds applies only when the clone stores a transcript-matched excerpt",
        "lookahead and flush apply to the buffered phrase queues",
        *(["native decoder block/context fields are inactive until native incremental PCM is enabled"] if not NATIVE_INCREMENTAL_PCM_ENABLED else []),
    ]
    if int(candidate["left_context_frames"]) < MODEL_REQUIRED_LEFT_CONTEXT_FRAMES:
        warnings.append(
            f"Reduced decoder context is experimental and may degrade quality; the model requires {MODEL_REQUIRED_LEFT_CONTEXT_FRAMES} frames."
        )
    return {
        "profile": profile,
        "scope": scope,
        "transport": {"sample_rate": 24000, "format": "pcm_s16le", "read_only": True},
        "capabilities": {"full_icl": True, "x_vector_only": False, "native_incremental_pcm": NATIVE_INCREMENTAL_PCM_ENABLED},
        "effectiveFields": effective_fields,
        "conditionalFields": ["max_reference_seconds"],
        "inactiveFields": ["crossfade_samples", "x_vector_only_mode", *(native_fields if not NATIVE_INCREMENTAL_PCM_ENABLED else [])],
        "phraseQueueFields": ["text_lookahead", "phrase_flush_ms"],
        "referenceDurationLimit": {
            "requestedSeconds": int(candidate["max_reference_seconds"]),
            "condition": "matched-excerpt-only",
            "effective": None,
            "pairing": "resolved-per-request",
        },
        "warnings": warnings,
        "temporaryOverrides": overrides,
    }


@app.post("/v1/voices/profiles/{profile_id}")
async def import_voice_profile(profile_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return _write_candidate_profile(profile_id, payload)


@app.get("/v1/backend/models")
async def backend_models() -> dict[str, Any]:
    _reconcile_engine()
    active = str(state["activeModel"])
    loaded = state["state"] == "loaded" and await _engine_ready()
    return {
        "available": list(_models()),
        "current": active,
        "loaded_models": [active] if loaded else [],
        "state": "loaded" if loaded else state["state"],
        "last_error": state["lastError"],
        "runtime": {
            "last_load_elapsed_s": state["lastLoadElapsedS"],
            "gpu": state["gpu"],
            "gpu_now": gpu_guard(active, minimum_free_mib=0, apply_policy=False),
            "load_headroom_mib": _guard_policy(active, "load")[0],
            "synthesis_headroom_mib": _guard_policy(active, "synthesis")[0],
            "enforced_gpu_policy": _enforced_gpu_policy(),
            "gpu_guard": dict(gpu_guard_settings),
            "mem_saver": not NATIVE_INCREMENTAL_PCM_ENABLED,
            "native_incremental_pcm": NATIVE_INCREMENTAL_PCM_ENABLED,
            "talker_prefix_cache_slots": TALKER_PREFIX_CACHE_SLOTS if NATIVE_INCREMENTAL_PCM_ENABLED else 0,
            "talker_prefix_cache_state": (
                "configured-experimental"
                if NATIVE_INCREMENTAL_PCM_ENABLED and TALKER_PREFIX_CACHE_SLOTS > 0
                else "disabled"
            ),
            "progressive_phrase_pcm": True,
            "sample_rate": 24000,
            "sample_format": "pcm_s16le",
            "tts_mode": "native-incremental-pcm" if NATIVE_INCREMENTAL_PCM_ENABLED else "offline-buffered",
        },
        "events": list(EVENTS),
        "last_action": state["lastAction"],
        "engineEpoch": int(state.get("engineEpoch") or 0),
        "supervisorInstanceId": SUPERVISOR_INSTANCE_ID,
        "singleResident": True,
    }


@app.post("/v1/backend/models/switch")
async def backend_switch(payload: dict[str, str]) -> dict[str, Any]:
    return await switch_model(str(payload.get("model_key") or ""))


@app.post("/v1/backend/models/unload")
async def backend_unload() -> dict[str, Any]:
    async with switch_lock:
        async with generation_lock:
            _stop_engine(action="explicit-unload")
            state.update(state="unloaded", reason="Model released by Voice Studio.", lastError=None)
        return {"current": state["activeModel"], "state": "unloaded", "loaded_models": [], "singleResident": True, "events": list(EVENTS)}


def _apply_clone_profile(payload: dict[str, Any]) -> None:
    voice = str(payload.get("voice") or "")
    if not voice.startswith("clone:"):
        return
    profile_id = voice.removeprefix("clone:")
    if not valid_profile_id(profile_id):
        raise HTTPException(status_code=422, detail="Invalid clone profile id.")
    frozen = payload.pop("clone_snapshot", None)
    if frozen is not None:
        if not isinstance(frozen, dict):
            raise HTTPException(status_code=422, detail="clone_snapshot must be an object.")
        frozen_id = str(frozen.get("profile_id") or "")
        content_hash = str(frozen.get("content_hash") or "")
        content_revision = frozen.get("content_revision")
        raw = str(frozen.get("ref_audio") or "")
        transcript = str(frozen.get("ref_text") or "").strip()
        raw_excerpts = frozen.get("reference_excerpts") or []
        if (
            frozen_id != profile_id
            or not re.fullmatch(r"[0-9a-f]{64}", content_hash)
            or isinstance(content_revision, bool)
            or not isinstance(content_revision, int)
            or content_revision < 1
            or not raw
            or not transcript
            or not isinstance(raw_excerpts, list)
        ):
            raise HTTPException(status_code=422, detail="Invalid frozen clone snapshot.")
        try:
            reference_audio = base64.b64decode(raw, validate=True)
        except Exception as exc:
            raise HTTPException(status_code=422, detail="Frozen clone reference audio is invalid.") from exc
        matched_pairs: list[dict[str, str]] = []
        hash_excerpts: list[tuple[bytes, str]] = []
        for excerpt in raw_excerpts:
            if not isinstance(excerpt, dict):
                raise HTTPException(status_code=422, detail="Frozen clone excerpts must be objects.")
            excerpt_encoded = str(excerpt.get("ref_audio") or "")
            excerpt_text = str(excerpt.get("ref_text") or "").strip()
            if not excerpt_encoded or not excerpt_text:
                raise HTTPException(status_code=422, detail="Frozen clone excerpts require audio and transcript.")
            try:
                excerpt_audio = base64.b64decode(excerpt_encoded, validate=True)
            except Exception as exc:
                raise HTTPException(status_code=422, detail="Frozen clone excerpt audio is invalid.") from exc
            hash_excerpts.append((excerpt_audio, excerpt_text))
            matched_pairs.append({"ref_audio": excerpt_encoded, "ref_text": excerpt_text})
        if clone_content_hash(reference_audio, transcript, hash_excerpts) != content_hash:
            raise HTTPException(status_code=409, detail="Frozen clone content hash does not match its reference material.")
        # Do not reopen mutable library metadata here.  These bytes were
        # captured when the response snapshot was created, so later profile
        # edits cannot change a later phrase in the same answer.
        payload.update(
            task_type="Base",
            ref_audio=raw,
            ref_text=transcript,
            x_vector_only_mode=False,
            _matched_reference_pairs=matched_pairs,
            _frozen_clone_content_hash=content_hash,
            _frozen_clone_content_revision=content_revision,
        )
        return
    profile_root = VOICE_LIBRARY_DIR / "profiles" / profile_id
    try:
        meta = json.loads((profile_root / "meta.json").read_text(encoding="utf-8"))
        reference = profile_root / str(meta["ref_audio_filename"])
    except (OSError, KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=f"Clone profile `{profile_id}` is unavailable: {exc}") from exc
    if not reference.is_file():
        raise HTTPException(status_code=404, detail=f"Clone profile `{profile_id}` is missing its reference audio.")
    matched_pairs: list[dict[str, str]] = []
    excerpts = meta.get("reference_excerpts")
    if isinstance(excerpts, list):
        for excerpt in excerpts:
            if not isinstance(excerpt, dict):
                continue
            filename = str(excerpt.get("ref_audio_filename") or "").strip()
            transcript = str(excerpt.get("ref_text") or "").strip()
            if not filename or Path(filename).name != filename or not transcript:
                continue
            excerpt_path = profile_root / filename
            if excerpt_path.is_file():
                matched_pairs.append(
                    {
                        "ref_audio": base64.b64encode(excerpt_path.read_bytes()).decode("ascii"),
                        "ref_text": transcript,
                    }
                )
    payload.update(
        task_type="Base",
        ref_audio=base64.b64encode(reference.read_bytes()).decode("ascii"),
        ref_text=str(meta.get("ref_text") or ""),
        x_vector_only_mode=bool(meta.get("x_vector_only_mode")),
        _matched_reference_pairs=matched_pairs,
    )


def _normalize_gradio_seed(payload: dict[str, Any]) -> None:
    """Omit randomness sentinels and forward an explicit uint32 seed."""
    if "seed" not in payload:
        return
    seed = _normalize_seed_value(payload.get("seed"))
    if seed is None:
        payload.pop("seed", None)
    else:
        payload["seed"] = seed


def _container_reachable_llm_endpoint(endpoint: str) -> str:
    """Normalize an OpenAI-compatible LLM URL and map host loopback into Docker.

    The Studio accepts either a complete ``/v1/chat/completions`` URL or the
    common server-base form (for example ``http://127.0.0.1:8818``).  Sending
    the latter verbatim is a valid HTTP request but reaches llama.cpp's root
    and produces an unhelpful 404 during a live turn.
    """
    parsed = urlsplit(endpoint)
    path = parsed.path.rstrip("/")
    if parsed.scheme in {"http", "https"} and parsed.hostname:
        if not path:
            path = "/v1/chat/completions"
        elif path == "/v1":
            path = "/v1/chat/completions"
    else:
        return endpoint
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        return urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, parsed.fragment))
    host = "host.docker.internal"
    if parsed.port:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, host, path, parsed.query, parsed.fragment))


def _transcode_llm_audio(audio: bytes, source_format: str, target_format: str) -> bytes:
    """Convert a complete browser turn to a truthful llama.cpp WAV or MP3 input."""
    if target_format not in {"wav", "mp3"}:
        raise HTTPException(status_code=422, detail="LLM microphone format must be wav or mp3.")
    with tempfile.TemporaryDirectory(prefix="audio-cpp-llm-input-") as directory:
        source = Path(directory) / f"recording.{source_format}"
        output = Path(directory) / f"recording.{target_format}"
        source.write_bytes(audio)
        codec_args = ["-c:a", "pcm_s16le"] if target_format == "wav" else ["-c:a", "libmp3lame", "-b:a", "320k"]
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-v", "error", "-y", "-i", str(source),
                    "-vn", "-ac", "1", *codec_args, str(output),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise HTTPException(status_code=503, detail=f"Could not convert the browser microphone recording to {target_format.upper()}.") from exc
        if result.returncode != 0 or not output.is_file() or not output.stat().st_size:
            detail = result.stderr.strip() or f"ffmpeg produced no {target_format.upper()} output"
            raise HTTPException(status_code=422, detail=f"Browser microphone audio could not be converted to {target_format.upper()}: {detail}")
        return output.read_bytes()


def _transcode_llm_audio_to_wav(audio: bytes, source_format: str) -> bytes:
    """Backward-compatible wrapper used by focused tests and older callers."""
    return _transcode_llm_audio(audio, source_format, "wav")


def _llamacpp_audio_part(data_uri: str, requested_format: str = "wav") -> dict[str, Any]:
    """Convert browser recorder data into the WAV/MP3 formats accepted by the LLM."""
    header, separator, encoded = data_uri.partition(",")
    if not separator or not header.startswith("data:audio/") or ";base64" not in header:
        raise HTTPException(status_code=422, detail="Streaming Playground requires base64 audio data.")
    audio_format = header.removeprefix("data:audio/").split(";", 1)[0].lower()
    if audio_format in {"x-wav", "wave"}:
        audio_format = "wav"
    if audio_format in {"mpeg", "x-mp3"}:
        audio_format = "mp3"
    try:
        audio = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Streaming Playground requires valid base64 audio data.") from exc
    if not audio:
        raise HTTPException(status_code=422, detail="Streaming Playground received empty microphone audio.")
    requested_format = requested_format.strip().lower() or "wav"
    if requested_format not in {"wav", "mp3"}:
        raise HTTPException(status_code=422, detail="LLM microphone format must be wav or mp3.")
    if audio_format != requested_format:
        audio = _transcode_llm_audio(audio, audio_format or "wav", requested_format)
        audio_format = requested_format
    return {
        "type": "input_audio",
        "input_audio": {"data": base64.b64encode(audio).decode("ascii"), "format": audio_format},
    }


def _llamacpp_history(messages: list[Any]) -> list[dict[str, Any]]:
    """Keep text history valid when earlier Studio turns originated from audio."""
    safe: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        if role not in {"user", "assistant", "system"}:
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            safe.append({"role": role, "content": content})
        elif role == "user" and message.get("audio_data_url"):
            safe.append({"role": "user", "content": "[Earlier spoken user turn]"})
    return safe


def _voice_profiles() -> list[dict[str, Any]]:
    """Expose the same uncached, canonical Base inventory used by Gradio."""
    return [
        {
            "id": str(meta["profile_id"]),
            "voice": f"clone:{meta['profile_id']}",
            "name": str(meta["name"]),
            "task": "Base",
            "language": str(meta["language"]),
            "provider": TUNING_PROVIDER,
            "content_revision": int(meta["content_revision"]),
            "content_hash": str(meta["content_hash"]),
        }
        for meta in live_profiles(VOICE_LIBRARY_DIR, normalize=True)
    ]


def _write_candidate_profile(profile_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Atomically import a compatible Base profile into the private candidate library."""
    if not valid_profile_id(profile_id):
        raise HTTPException(status_code=422, detail="Invalid clone profile id.")
    encoded = str(payload.get("ref_audio") or "")
    if bool(payload.get("x_vector_only_mode")):
        raise HTTPException(status_code=422, detail="x_vector_only_mode is unavailable; import a full ICL reference and transcript.")
    reference_text = str(payload.get("ref_text") or "").strip()
    if not reference_text:
        raise HTTPException(status_code=422, detail="A full ICL reference transcript is required.")
    if not encoded:
        raise HTTPException(status_code=422, detail="Clone reference audio is required.")
    try:
        audio = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Invalid clone reference audio.") from exc
    if not audio:
        raise HTTPException(status_code=422, detail="Clone reference audio is empty.")
    prepared_excerpts: list[tuple[str, bytes, str]] = []
    excerpts = payload.get("reference_excerpts") or []
    if not isinstance(excerpts, list):
        raise HTTPException(status_code=422, detail="reference_excerpts must be a list of matched audio/transcript pairs.")
    for index, excerpt in enumerate(excerpts, start=1):
        if not isinstance(excerpt, dict):
            raise HTTPException(status_code=422, detail="Each reference excerpt must be an object.")
        excerpt_text = str(excerpt.get("ref_text") or "").strip()
        excerpt_encoded = str(excerpt.get("ref_audio") or "")
        if not excerpt_text or not excerpt_encoded:
            raise HTTPException(status_code=422, detail="Each reference excerpt requires audio and its exact transcript.")
        try:
            excerpt_audio, _ = _decode_reference_wav(excerpt_encoded, "matched reference excerpt")
        except HTTPException as exc:
            raise HTTPException(status_code=422, detail=exc.detail) from exc
        prepared_excerpts.append((f"ref_excerpt_{index}.wav", excerpt_audio, excerpt_text))
    current = canonical_profile(VOICE_LIBRARY_DIR, profile_id, normalize=True)
    content_hash = clone_content_hash(
        audio,
        reference_text,
        [(excerpt_audio, excerpt_text) for _, excerpt_audio, excerpt_text in prepared_excerpts],
    )
    previous_revision = (
        int(current.get("content_revision", 1))
        if isinstance(current, dict) and isinstance(current.get("content_revision"), int)
        else 0
    )
    content_revision = (
        previous_revision
        if isinstance(current, dict) and current.get("content_hash") == content_hash
        else previous_revision + 1
    )
    profile_dir = VOICE_LIBRARY_DIR / "profiles" / profile_id
    profile_dir.mkdir(parents=True, exist_ok=True)
    reference_name = "ref_audio.wav"
    temp_audio = profile_dir / f".{reference_name}.tmp"
    temp_meta = profile_dir / ".meta.json.tmp"
    temp_audio.write_bytes(audio)
    temp_audio.replace(profile_dir / reference_name)
    for excerpt_name, excerpt_audio, _ in prepared_excerpts:
        temp_excerpt = profile_dir / f".{excerpt_name}.tmp"
        temp_excerpt.write_bytes(excerpt_audio)
        temp_excerpt.replace(profile_dir / excerpt_name)
    metadata = {
        "profile_id": profile_id,
        "name": str(payload.get("name") or profile_id),
        "task_type": "Base",
        "created_at": str(payload.get("created_at") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
        "language": str(payload.get("language") or "Auto"),
        "voice": f"clone:{profile_id}",
        "instructions": str(payload.get("instructions") or ""),
        "ref_text": reference_text,
        "x_vector_only_mode": False,
        "ref_audio_filename": reference_name,
        "origin": str(payload.get("origin") or "audio.cpp Base clone"),
        "provider": TUNING_PROVIDER,
        "content_revision": max(1, content_revision),
        "content_hash": content_hash,
        "reference_excerpts": [
            {"ref_audio_filename": filename, "ref_text": transcript}
            for filename, _, transcript in prepared_excerpts
        ],
    }
    temp_meta.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    temp_meta.replace(profile_dir / "meta.json")
    _record_event("profile-imported", profile_id=profile_id)
    return {
        "id": profile_id,
        "voice": f"clone:{profile_id}",
        "name": metadata["name"],
        "content_revision": metadata["content_revision"],
        "content_hash": metadata["content_hash"],
    }


def _render_master_wav(master_wav: bytes, requested_format: str) -> tuple[bytes, str, str, dict[str, int | str]]:
    """Return exactly one requested output, using audio.cpp's complete WAV as master."""
    fmt = (requested_format or "wav").strip().lower()
    if fmt not in SUPPORTED_MASTER_OUTPUT_FORMATS:
        raise HTTPException(status_code=400, detail=f"Unsupported audio format `{fmt}`.")
    try:
        with wave.open(io.BytesIO(master_wav), "rb") as reader:
            if reader.getframerate() != 24000 or reader.getsampwidth() != 2:
                raise HTTPException(
                    status_code=502,
                    detail="audio.cpp WAV master must be model-native 24 kHz PCM16; it is never resampled or requantized by the candidate.",
                )
            metadata: dict[str, int | str] = {
                "codec": "pcm_s16le",
                "container": "WAV",
                "quality": "native lossless master",
                "sampleRate": reader.getframerate(),
                "channels": reader.getnchannels(),
                "bitsPerSample": reader.getsampwidth() * 8,
                "durationSeconds": round(reader.getnframes() / reader.getframerate(), 3),
            }
            if fmt == "pcm":
                metadata["container"] = "raw PCM"
                metadata["quality"] = "native PCM extracted without re-encoding"
                return reader.readframes(reader.getnframes()), "audio/pcm", "pcm", metadata
    except wave.Error as exc:
        raise HTTPException(status_code=502, detail=f"audio.cpp returned an invalid WAV master: {exc}") from exc
    if fmt == "wav":
        return master_wav, "audio/wav", "wav", metadata

    encoders = {
        # Never resample or downmix the native WAV master.  Lossless formats
        # preserve it bit-for-bit; lossy formats use intentionally high rates.
        "mp3": ("libmp3lame", ["-b:a", "320k"], "audio/mpeg", "mp3", "mp3", "MP3", "high-quality 320 kbps"),
        "flac": ("flac", ["-compression_level", "8"], "audio/flac", "flac", "flac", "FLAC", "lossless"),
        "aac": ("aac", ["-b:a", "320k"], "audio/aac", "aac", "aac", "ADTS", "high-quality 320 kbps"),
        # libopus caps this mono stream at 256 kbps.  Asking ffmpeg for
        # 320 kbps fails instead of silently clamping, so use the codec's
        # highest supported rate while preserving the 24 kHz master.
        "opus": ("libopus", ["-b:a", "256k"], "audio/ogg", "opus", "opus", "OGG", "maximum-quality 256 kbps"),
    }
    encoder, encoder_args, media_type, extension, codec, container, quality = encoders[fmt]
    with tempfile.TemporaryDirectory(prefix="audio-cpp-output-") as directory:
        source = Path(directory) / "master.wav"
        output = Path(directory) / f"output.{extension}"
        source.write_bytes(master_wav)
        try:
            result = subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-i", str(source), "-c:a", encoder, *encoder_args, str(output)],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise HTTPException(status_code=503, detail=f"Could not encode `{fmt}` output.") from exc
        if result.returncode != 0 or not output.exists():
            raise HTTPException(status_code=502, detail=f"Failed to encode `{fmt}` output: {result.stderr.strip()}")
        metadata["codec"] = codec
        metadata["container"] = container
        metadata["quality"] = quality
        return output.read_bytes(), media_type, extension, metadata


async def _assert_synthesis_ready(payload: dict[str, Any]) -> str:
    """Check candidate admission before returning a buffered or chunked response."""
    model_id = str(payload.get("model") or state["activeModel"])
    if model_id != state["activeModel"]:
        raise HTTPException(status_code=409, detail="Selected model is not active; switch it in Voice Studio before synthesis.")
    if state["state"] != "loaded":
        raise HTTPException(status_code=409, detail="No candidate model is loaded. Use Voice Studio model controls first.")
    guard = gpu_guard(model_id, operation="synthesis")
    if not guard["ok"]:
        state.update(reason=guard["reason"], lastError=guard["reason"])
        _record_event("gpu-blocked-synthesis", model_id=model_id, gpu=guard)
        raise HTTPException(status_code=409, detail={"state": "blocked", **guard})
    expected_epoch_raw = payload.pop("expected_engine_epoch", None)
    expected_instance_raw = payload.pop("expected_supervisor_instance_id", None)
    expected_epoch: int | None = None
    expected_instance: str | None = None
    if expected_epoch_raw is not None:
        if isinstance(expected_epoch_raw, bool):
            raise HTTPException(status_code=422, detail="expected_engine_epoch must be a non-negative integer.")
        if isinstance(expected_epoch_raw, int):
            expected_epoch = expected_epoch_raw
        elif isinstance(expected_epoch_raw, str) and expected_epoch_raw.isdecimal():
            expected_epoch = int(expected_epoch_raw)
        else:
            raise HTTPException(status_code=422, detail="expected_engine_epoch must be a non-negative integer.")
        if expected_epoch < 0:
            raise HTTPException(status_code=422, detail="expected_engine_epoch must be a non-negative integer.")
    if expected_instance_raw is not None:
        if not isinstance(expected_instance_raw, str) or not re.fullmatch(r"[0-9a-f]{32}", expected_instance_raw):
            raise HTTPException(status_code=422, detail="expected_supervisor_instance_id must be a supervisor instance nonce.")
        expected_instance = expected_instance_raw
    if (expected_epoch is None) != (expected_instance is None):
        raise HTTPException(
            status_code=422,
            detail="expected_engine_epoch and expected_supervisor_instance_id must be supplied together.",
        )
    admission_epoch = int(state.get("engineEpoch") or 0)
    if expected_epoch is not None and (
        expected_epoch != admission_epoch or expected_instance != SUPERVISOR_INSTANCE_ID
    ):
        raise HTTPException(
            status_code=409,
            detail="Candidate engine lifecycle changed before synthesis admission; retry with the current supervisor identity.",
        )
    if not await _engine_ready():
        raise HTTPException(status_code=503, detail="Candidate model is switching/loading; retry when Voice Studio reports loaded.")
    if state["state"] != "loaded" or state["activeModel"] != model_id or admission_epoch != int(state.get("engineEpoch") or 0):
        raise HTTPException(status_code=409, detail="Candidate model changed while the synthesis request was being admitted; retry the request.")
    payload["_engine_epoch"] = admission_epoch
    return model_id


def _revalidate_generation_admission(payload: dict[str, Any], model_id: str, *, internal_warmup: bool = False) -> None:
    """Reject a request that queued behind a model transition."""
    expected_epoch = payload.get("_engine_epoch")
    # Route-level admission always stamps an epoch.  Direct contract tests and
    # internal helper callers without one are intentionally left unchanged.
    if expected_epoch is None and not internal_warmup:
        return
    if expected_epoch is not None and int(expected_epoch) != int(state.get("engineEpoch") or 0):
        raise HTTPException(status_code=409, detail="Candidate model changed before synthesis began; retry the request.")
    if state["activeModel"] != model_id:
        raise HTTPException(status_code=409, detail="Selected model is no longer active; retry after the model switch completes.")
    allowed_states = {"warming"} if internal_warmup else {"loaded"}
    if state["state"] not in allowed_states:
        raise HTTPException(status_code=409, detail="Candidate model is switching or warming; retry when it reports loaded.")


def _native_pcm_requested(payload: dict[str, Any]) -> bool:
    return bool(payload.get("stream")) and str(payload.get("response_format") or "wav").lower() == "pcm"


def _delivery_mode_for_payload(payload: dict[str, Any]) -> str:
    if "response_format" not in payload and "stream" not in payload:
        return "profile-resolution"
    if _native_pcm_requested(payload):
        return "native-incremental-pcm"
    if str(payload.get("response_format") or "wav").lower() == "pcm":
        return "buffered-fallback"
    return "offline-full-decoder"


def _resolve_request_tuning(
    payload: dict[str, Any], *, force_offline_full: bool = False,
) -> dict[str, Any]:
    """Resolve one immutable request policy without mixing offline and stream knobs."""
    tuning = payload.get("tuning") or {}
    if not isinstance(tuning, dict):
        raise HTTPException(status_code=422, detail="tuning must be an object.")
    provider = tuning.get("provider")
    if provider is not None and provider != TUNING_PROVIDER:
        raise HTTPException(status_code=422, detail="tuning is available only for qwen3tts-audiocpp.")
    scope = str(tuning.get("scope") or "realtime")
    frozen_effective = tuning.get("effective")
    if frozen_effective is not None:
        # HF Realtime resolves the named profile when the user applies it, then
        # freezes this bounded value snapshot for one assistant response.  Do
        # not reread the mutable profile document for later phrase requests:
        # the profile may have been edited or deleted after the answer began.
        if provider != TUNING_PROVIDER or not isinstance(frozen_effective, dict):
            raise HTTPException(status_code=422, detail="Frozen tuning requires the qwen3tts-audiocpp provider and an effective object.")
        unknown_effective = set(frozen_effective) - TUNING_VALUE_FIELDS
        if unknown_effective:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported frozen tuning field(s): {', '.join(sorted(unknown_effective))}.",
            )
        required_effective = TUNING_VALUE_FIELDS - {"model", "clone_mode"}
        missing_effective = required_effective - set(frozen_effective)
        if missing_effective:
            raise HTTPException(
                status_code=422,
                detail=f"Frozen tuning is missing field(s): {', '.join(sorted(missing_effective))}.",
            )
        profile_id = str(tuning.get("profile_id") or "")
        if not profile_id:
            raise HTTPException(status_code=422, detail="Frozen tuning requires profile_id.")
        revision = tuning.get("profile_revision")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise HTTPException(status_code=422, detail="Frozen tuning requires a positive profile_revision.")
        profile = dict(frozen_effective)
        profile.setdefault("clone_mode", "full_icl")
        _validate_tuning_profile(profile)
    else:
        document = _read_tuning_profiles()
        selected = _selected_tuning_profile(document, scope)
        profile_id = str(tuning.get("profile_id") or selected)
        profile = document["profiles"].get(profile_id)
        if not isinstance(profile, dict):
            raise HTTPException(status_code=404, detail="Unknown tuning profile.")
        revision = profile.get("revision", 1)
    required_model = tuning.get("model") or profile.get("model")
    if required_model and required_model != state["activeModel"]:
        raise HTTPException(status_code=409, detail={"active": state["activeModel"], "required": required_model, "auto_switch": False})
    allowed_names = TUNING_VALUE_FIELDS - {"clone_mode"}
    allowed = {key: profile[key] for key in allowed_names if key in profile}
    overrides = tuning.get("overrides") or {}
    if not isinstance(overrides, dict) or any(key not in allowed_names for key in overrides):
        raise HTTPException(status_code=422, detail="Unsupported tuning override.")
    seed_supplied = "seed" in overrides or "seed" in payload
    if "seed" in overrides:
        overrides = dict(overrides)
        overrides["seed"] = _normalize_seed_value(overrides["seed"])
        request_seed = overrides["seed"]
    elif "seed" in payload:
        request_seed = _normalize_seed_value(payload["seed"])
    else:
        request_seed = None
    candidate = dict(profile)
    candidate.update(overrides)
    _validate_tuning_profile(candidate)
    if candidate.get("model") and candidate["model"] != state["activeModel"]:
        raise HTTPException(status_code=409, detail={"active": state["activeModel"], "required": candidate["model"], "auto_switch": False})
    allowed.update(overrides)
    delivery_mode = OFFLINE_FULL_DECODE_MODE if force_offline_full else _delivery_mode_for_payload(payload)
    offline_full_quality = delivery_mode == "offline-full-decoder"
    # Full-WAV synthesis is a fixed offline policy, not the mutable streaming
    # selection.  Use the shipped Quality sampler defaults even if a built-in
    # profile was edited for interactive experiments.
    quality = _default_tuning_profiles()["profiles"]["quality"]
    # audio.cpp accepts these sampler fields directly. Lookahead and flush
    # belong to the browser phrase queue and are never sent to the engine.
    sampler_source = quality if offline_full_quality else allowed
    engine_fields = {
        key: sampler_source[key]
        for key in ("temperature", "top_k", "top_p", "repetition_penalty", "seed")
        if key in sampler_source and sampler_source[key] is not None
    }
    # A seed is request identity rather than a quality preset.  Preserve a
    # caller's explicit uint32 even when Full WAV pins every other sampler to
    # Quality; null/blank/-1 deliberately removes a profile seed for random
    # generation.
    if seed_supplied:
        if request_seed is None:
            engine_fields.pop("seed", None)
        else:
            engine_fields["seed"] = request_seed
    if offline_full_quality:
        effective = {
            key: value
            for key, value in allowed.items()
            if key == "model"
        }
        effective.update(engine_fields)
        inactive_fields = [
            "max_reference_seconds", "first_block_frames",
            "steady_block_frames", "left_context_frames", "text_lookahead",
            "phrase_flush_ms",
        ]
        phrase_queue_fields: dict[str, Any] = {}
    else:
        effective = allowed
        inactive_fields = [
            *([] if NATIVE_INCREMENTAL_PCM_ENABLED else ["first_block_frames", "steady_block_frames", "left_context_frames"]),
        ]
        phrase_queue_fields = {
            key: allowed[key]
            for key in ("text_lookahead", "phrase_flush_ms")
            if key in allowed
        }
    warnings: list[str] = []
    if not offline_full_quality and int(candidate["left_context_frames"]) < MODEL_REQUIRED_LEFT_CONTEXT_FRAMES:
        warnings.append(
            f"Reduced decoder context is experimental and may degrade quality; the model requires {MODEL_REQUIRED_LEFT_CONTEXT_FRAMES} frames."
        )
    return {
        "id": profile_id,
        "scope": scope,
        "revision": revision,
        "policy": "offline-full-quality" if offline_full_quality else "profile-streaming",
        "delivery_mode": delivery_mode,
        "effective": effective,
        "engine_fields": engine_fields,
        "reference_limit_seconds": int(candidate["max_reference_seconds"]),
        "reference_limit_condition": "matched-excerpt-only",
        "inactive_fields": inactive_fields,
        "phrase_queue_fields": phrase_queue_fields,
        "warnings": warnings,
    }


@app.post("/v1/audio/speech")
async def speech(request: Request) -> Response:
    payload = await request.json()
    payload.setdefault("response_format", "wav")
    payload.setdefault("stream", False)
    payload["_tuning_snapshot"] = _resolve_request_tuning(payload)
    _apply_clone_profile(payload)
    model_id = await _assert_synthesis_ready(payload)
    if _native_pcm_requested(payload):
        if not NATIVE_INCREMENTAL_PCM_ENABLED:
            raise HTTPException(status_code=409, detail="Native incremental PCM is not enabled in this candidate build; use buffered phrase PCM or deploy a validated native build.")
        if not (payload.get("task_type") == "Base" or payload.get("ref_audio")):
            raise HTTPException(status_code=422, detail="Native streaming currently supports Base clone requests only.")
        return await _native_clone_pcm_response(payload, model_id, request=request)
    async with generation_lock:
        _revalidate_generation_admission(payload, model_id)
        _record_event("generation-start", model_id=model_id, requestedFormat=str(payload.get("response_format") or "wav"))
        if payload.get("task_type") == "Base" or payload.get("ref_audio"):
            return await _voice_clone_response(payload, request=request)
        payload.pop("_tuning_snapshot", None)
        payload.pop("tuning", None)
        _strip_private_engine_fields(payload)
        payload["stream"] = False
        async with httpx.AsyncClient(timeout=600.0) as client:
            response = await _post_engine_with_disconnect(client, payload, request=request)
        return Response(content=response.content, status_code=response.status_code, media_type=response.headers.get("content-type"))


@app.post("/v1/audio/voice-clone")
async def voice_clone(request: Request) -> Response:
    """Compatibility adapter for the copied Gradio Base-profile playground."""
    payload = await request.json()
    payload.setdefault("response_format", "wav")
    payload.setdefault("stream", False)
    # This endpoint is the Studio's format-independent full-quality contract.
    # Even PCM means raw frames extracted from one completed offline WAV master,
    # never the buffered or incremental streaming decoder.
    payload["_tuning_snapshot"] = _resolve_request_tuning(
        payload, force_offline_full=True,
    )
    model_id = await _assert_synthesis_ready(payload)
    async with generation_lock:
        _revalidate_generation_admission(payload, model_id)
        _record_event("generation-start", model_id=model_id, requestedFormat=str(payload.get("response_format") or "wav"))
        return await _voice_clone_response(payload, request=request)


def _decode_reference_wav(raw: str, label: str) -> tuple[bytes, float]:
    """Decode one complete WAV and return its immutable bytes and duration."""
    try:
        audio = base64.b64decode(raw, validate=True)
        with wave.open(io.BytesIO(audio), "rb") as reader:
            rate = reader.getframerate()
            frames = reader.getnframes()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid {label} WAV: {exc}") from exc
    if rate <= 0 or frames <= 0:
        raise HTTPException(status_code=400, detail=f"{label.capitalize()} WAV is empty.")
    return audio, frames / rate


def _prepare_reference_pair(
    payload: dict[str, Any],
    tuning_snapshot: dict[str, Any],
    filename: str,
) -> tuple[Path, dict[str, Any], str]:
    """Select one transcript-matched reference without audio-only truncation.

    A duration limit can select an explicitly stored excerpt only when that
    excerpt carries its own exact transcript.  Otherwise the complete source
    audio and complete source transcript remain inseparable.  Full-WAV policy
    always uses the complete pair regardless of the requested stream limit.
    """
    full_audio, source_seconds = _decode_reference_wav(
        str(payload.get("ref_audio") or ""), "clone reference"
    )
    full_text = str(payload.get("ref_text") or "").strip()
    if not full_text:
        raise HTTPException(status_code=400, detail="A Base clone reference transcript is required.")
    requested_limit = int(tuning_snapshot.get("reference_limit_seconds", 30))
    delivery_mode = str(
        tuning_snapshot.get("delivery_mode") or _delivery_mode_for_payload(payload)
    )
    chosen_audio = full_audio
    chosen_text = full_text
    used_seconds = source_seconds
    pairing = "full"
    limit_applied = False
    excerpts = payload.pop("_matched_reference_pairs", [])
    if (
        delivery_mode != "offline-full-decoder"
        and source_seconds > requested_limit
        and isinstance(excerpts, list)
    ):
        candidates: list[tuple[float, bytes, str]] = []
        for excerpt in excerpts:
            if not isinstance(excerpt, dict):
                continue
            excerpt_text = str(excerpt.get("ref_text") or "").strip()
            if not excerpt_text:
                continue
            try:
                excerpt_audio, excerpt_seconds = _decode_reference_wav(
                    str(excerpt.get("ref_audio") or ""), "matched reference excerpt"
                )
            except HTTPException:
                continue
            if 0 < excerpt_seconds <= requested_limit and excerpt_seconds < source_seconds:
                candidates.append((excerpt_seconds, excerpt_audio, excerpt_text))
        if candidates:
            used_seconds, chosen_audio, chosen_text = max(candidates, key=lambda item: item[0])
            pairing = "matched-excerpt"
            limit_applied = True
    reference = Path(tempfile.gettempdir()) / filename
    reference.write_bytes(chosen_audio)
    stats = {
        "source_seconds": round(source_seconds, 3),
        "requested_limit_seconds": requested_limit,
        "used_seconds": round(used_seconds, 3),
        "limit_applied": limit_applied,
        "pairing": pairing,
        # Compatibility alias for existing clients.  It no longer means that
        # the supervisor cut a WAV while retaining the original transcript.
        "truncated": limit_applied,
    }
    return reference, stats, chosen_text


def _reference_headers(stats: dict[str, Any], delivery_mode: str) -> dict[str, str]:
    return {
        "X-TTS-Reference-Source-Seconds": str(stats["source_seconds"]),
        "X-TTS-Reference-Requested-Limit-Seconds": str(stats["requested_limit_seconds"]),
        "X-TTS-Reference-Used-Seconds": str(stats["used_seconds"]),
        "X-TTS-Reference-Limit-Applied": str(bool(stats["limit_applied"])).lower(),
        "X-TTS-Reference-Pairing": str(stats["pairing"]),
        "X-TTS-Reference-Truncated": str(bool(stats["truncated"])).lower(),
        "X-TTS-Delivery-Mode": delivery_mode,
    }


def _reference_event_fields(stats: dict[str, Any], delivery_mode: str) -> dict[str, Any]:
    return {
        "sourceReferenceSeconds": stats["source_seconds"],
        "requestedReferenceLimitSeconds": stats["requested_limit_seconds"],
        "usedReferenceSeconds": stats["used_seconds"],
        "referenceLimitApplied": stats["limit_applied"],
        "referencePairing": stats["pairing"],
        "deliveryMode": delivery_mode,
    }


def _apply_engine_tuning(engine_payload: dict[str, Any]) -> dict[str, Any]:
    """Strip supervisor-only fields and forward only supported sampler knobs."""
    snapshot = engine_payload.pop("_tuning_snapshot", {})
    engine_payload.pop("tuning", None)
    for key in ("temperature", "top_k", "top_p", "repetition_penalty", "seed"):
        engine_payload.pop(key, None)
    if not isinstance(snapshot, dict):
        return {}
    fields = snapshot.get("engine_fields", {})
    if isinstance(fields, dict):
        for key in ("temperature", "top_k", "top_p", "repetition_penalty", "seed"):
            if key in fields:
                engine_payload[key] = fields[key]
    return snapshot


def _strip_private_engine_fields(engine_payload: dict[str, Any]) -> None:
    """Keep supervisor bookkeeping out of audio.cpp's public request schema."""
    for key in list(engine_payload):
        if key.startswith("_") or key in {"request_id", "requestId", "response_id", "responseId"}:
            engine_payload.pop(key, None)


def _engine_decode_mode(response: httpx.Response | Any) -> str | None:
    """Read result-derived engine proof without trusting request metadata."""
    headers = getattr(response, "headers", {})
    return headers.get(ENGINE_DECODE_MODE_HEADER) or headers.get(
        ENGINE_DECODE_MODE_HEADER.lower()
    )


def _validate_engine_decode_mode(
    response: httpx.Response | Any, expected: str | None,
) -> str:
    """Require result-derived proof whenever a specific decoder was requested."""
    actual = _engine_decode_mode(response)
    if expected and not actual:
        raise HTTPException(
            status_code=502,
            detail={
                "state": "decoder-mode-proof-missing",
                "expected": expected,
            },
        )
    if actual and expected and actual != expected:
        raise HTTPException(
            status_code=502,
            detail={
                "state": "decoder-mode-mismatch",
                "expected": expected,
                "actual": actual,
            },
        )
    return actual or "unreported"


_NATIVE_SSE_DONE = object()


def _native_sse_event(line: str) -> dict[str, Any] | object | None:
    """Decode one audio.cpp SSE data line without treating transport text as PCM.

    Native PCM is admitted only after the engine emits its own decoder-mode
    artifact.  The candidate never converts request metadata into that proof.
    """
    if not line.startswith("data:"):
        return None
    data = line[5:].strip()
    if not data:
        return None
    if data == "[DONE]":
        return _NATIVE_SSE_DONE
    try:
        value = json.loads(data)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail="audio.cpp native SSE contained invalid JSON") from exc
    if not isinstance(value, dict):
        raise HTTPException(status_code=502, detail="audio.cpp native SSE event must be an object")
    return value


async def _await_native_decode_mode_proof(lines: Any) -> str:
    """Consume only engine SSE metadata until native decode proof is observed.

    The caller retains the same iterator for PCM relay, so the proof is not
    replayed and the first audio delta remains the next unread event.  PCM or
    completion before the proof is a fail-closed protocol violation.
    """
    while True:
        try:
            line = await lines.__anext__()
        except StopAsyncIteration as exc:
            raise HTTPException(
                status_code=502,
                detail={"state": "decoder-mode-proof-missing", "expected": "native-incremental-pcm"},
            ) from exc
        event = _native_sse_event(line)
        if event is _NATIVE_SSE_DONE:
            raise HTTPException(
                status_code=502,
                detail={
                    "state": "decoder-mode-proof-missing",
                    "message": "audio.cpp native SSE ended before decoder-mode proof",
                    "expected": "native-incremental-pcm",
                },
            )
        if event is None:
            continue
        assert isinstance(event, dict)
        event_type = str(event.get("type") or "")
        if event_type == "error":
            error = event.get("error")
            message = error.get("message") if isinstance(error, dict) else None
            raise HTTPException(
                status_code=502,
                detail={
                    "state": "native-stream-error",
                    "message": str(message or "audio.cpp native stream failed before decoder-mode proof"),
                    "chunksReceived": 0,
                },
            )
        if event_type == "speech.decode_mode":
            candidate_mode = str(event.get("mode") or "")
            if candidate_mode != "native-incremental-pcm":
                raise HTTPException(
                    status_code=502,
                    detail={
                        "state": "decoder-mode-mismatch",
                        "expected": "native-incremental-pcm",
                        "actual": candidate_mode or None,
                    },
                )
            return candidate_mode
        if event_type in {"speech.audio.delta", "speech.audio.done"}:
            raise HTTPException(
                status_code=502,
                detail={"state": "decoder-mode-proof-missing", "expected": "native-incremental-pcm"},
            )


def _decode_native_pcm_delta(event: dict[str, Any]) -> bytes:
    """Decode one engine-owned PCM delta without altering its boundaries."""

    encoded = event.get("audio")
    if not isinstance(encoded, str):
        raise HTTPException(status_code=502, detail="audio.cpp native SSE PCM delta is missing audio")
    try:
        chunk = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=502, detail="audio.cpp native SSE PCM delta is malformed") from exc
    if not chunk:
        return b""
    if len(chunk) % 2:
        raise HTTPException(status_code=502, detail="audio.cpp native SSE PCM delta is not PCM16 aligned")
    return chunk


async def _prefetch_native_pcm_proof(
    lines: Any,
    *,
    verified_decode_mode: str,
    minimum_chunks: int = 2,
) -> list[tuple[bytes, float]]:
    """Require genuine engine delta boundaries before committing headers.

    HTTP client read boundaries are not engine chunk boundaries: intermediaries
    may coalesce two SSE deltas into one byte read or split one delta across
    reads.  This preflight consumes two *decoded* nonempty speech.audio.delta
    events from the same iterator, buffers them once, and lets the relay emit
    those exact bytes exactly once.
    """

    prefetched: list[tuple[bytes, float]] = []
    while len(prefetched) < minimum_chunks:
        try:
            line = await lines.__anext__()
        except StopAsyncIteration as exc:
            raise HTTPException(
                status_code=502,
                detail={
                    "state": "native-stream-not-incremental",
                    "message": f"audio.cpp native SSE ended before {minimum_chunks} nonempty PCM delta events",
                    "chunksReceived": len(prefetched),
                },
            ) from exc
        event = _native_sse_event(line)
        if event is _NATIVE_SSE_DONE:
            raise HTTPException(
                status_code=502,
                detail={
                    "state": "native-stream-not-incremental",
                    "message": f"audio.cpp native SSE ended before {minimum_chunks} nonempty PCM delta events",
                    "chunksReceived": len(prefetched),
                },
            )
        if event is None:
            continue
        assert isinstance(event, dict)
        event_type = str(event.get("type") or "")
        if event_type == "error":
            error = event.get("error")
            message = error.get("message") if isinstance(error, dict) else None
            raise HTTPException(
                status_code=502,
                detail={
                    "state": "native-stream-error",
                    "message": str(message or "audio.cpp native stream failed before incremental proof"),
                    "chunksReceived": len(prefetched),
                },
            )
        if event_type == "speech.decode_mode":
            candidate_mode = str(event.get("mode") or "")
            if candidate_mode != verified_decode_mode:
                raise HTTPException(
                    status_code=502,
                    detail={
                        "state": "decoder-mode-mismatch",
                        "expected": verified_decode_mode,
                        "actual": candidate_mode or None,
                    },
                )
            continue
        if event_type == "speech.audio.done":
            raise HTTPException(
                status_code=502,
                detail={
                    "state": "native-stream-not-incremental",
                    "message": f"audio.cpp native SSE completed before {minimum_chunks} nonempty PCM delta events",
                    "chunksReceived": len(prefetched),
                },
            )
        if event_type == "speech.generation":
            generation = _generation_event_fields(event)
            if not generation["eos"]:
                _raise_generation_limit()
            raise HTTPException(502, "audio.cpp generation completed before two native PCM events")
        if event_type != "speech.audio.delta":
            continue
        chunk = _decode_native_pcm_delta(event)
        if chunk:
            prefetched.append((chunk, time.monotonic()))
    return prefetched


def _is_native_terminal_transport_line(line: str) -> bool:
    """Return whether a post-completion SSE line is only a blank separator."""

    return not line.strip()


async def _await_native_stage_with_disconnect(
    awaitable: Any,
    *,
    request: Request | None,
    stage: str,
    model_id: str,
    internal_warmup: bool,
) -> Any:
    """Await one pre-header native stage while retaining downstream ownership.

    Starlette cannot cancel a response iterator until the StreamingResponse has
    been returned.  Native proof deliberately happens before that boundary, so
    every blocking stage must observe a browser disconnect itself or the sole
    generation lock can remain pinned for the upstream timeout.
    """

    task = asyncio.ensure_future(awaitable)
    try:
        if request is None:
            return await task
        while not task.done():
            done, _ = await asyncio.wait({task}, timeout=0.05)
            if done:
                break
            if await request.is_disconnected():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                if not internal_warmup:
                    _record_event(
                        "generation-cancelled",
                        model_id=model_id,
                        nativeIncremental=True,
                        reason="client-disconnected",
                        stage=stage,
                    )
                raise asyncio.CancelledError(
                    f"downstream client disconnected during native {stage}"
                )
        return await task
    except BaseException:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        raise


async def _post_engine_with_disconnect(
    client: httpx.AsyncClient,
    payload: dict[str, Any],
    *,
    request: Request | None,
) -> httpx.Response:
    """Own a buffered upstream POST until completion or client disconnect.

    A regular ``await client.post`` gives FastAPI no cancellation point tied to
    the downstream socket while the engine is synthesizing.  Because the caller
    holds the single-generation lock, an abandoned ten-minute request could
    otherwise block cancellation, model lifecycle, and every successor turn.
    """

    upstream = asyncio.create_task(
        client.post(f"{ENGINE_URL}/v1/audio/speech", json=payload)
    )
    try:
        if request is None:
            return await upstream
        while not upstream.done():
            done, _ = await asyncio.wait({upstream}, timeout=0.05)
            if done:
                break
            if await request.is_disconnected():
                upstream.cancel()
                try:
                    await upstream
                except asyncio.CancelledError:
                    pass
                _record_event(
                    "generation-cancelled",
                    model_id=state.get("activeModel"),
                    reason="client-disconnected",
                )
                raise asyncio.CancelledError("downstream client disconnected")
        return await upstream
    except BaseException:
        if not upstream.done():
            upstream.cancel()
            try:
                await upstream
            except asyncio.CancelledError:
                pass
        raise


async def _voice_clone_response(
    payload: dict[str, Any], *, request: Request | None = None
) -> Response:
    if bool(payload.get("x_vector_only_mode")):
        raise HTTPException(status_code=422, detail="x_vector_only_mode is unavailable; use a full ICL reference and transcript.")
    raw = str(payload.get("ref_audio") or "")
    if not raw:
        raise HTTPException(status_code=400, detail="A Base clone reference WAV is required.")
    engine_payload = dict(payload)
    tuning_snapshot = _apply_engine_tuning(engine_payload)
    delivery_mode = str(
        tuning_snapshot.get("delivery_mode") or _delivery_mode_for_payload(payload)
    )
    # Gradio uses -1 as its "random seed" sentinel. audio.cpp validates seed
    # as unsigned, so omission is the compatible way to request randomness.
    _normalize_gradio_seed(engine_payload)
    requested_format = str(payload.get("response_format") or "wav").strip().lower()
    if requested_format not in SUPPORTED_MASTER_OUTPUT_FORMATS:
        raise HTTPException(status_code=400, detail=f"Unsupported audio format `{requested_format}`.")
    options = engine_payload.get("options") or {}
    if not isinstance(options, dict):
        raise HTTPException(status_code=422, detail="options must be an object.")
    options = dict(options)
    options.pop("qwen3_tts.decode_mode", None)
    expected_decode_mode: str | None = None
    # Buffered phrase delivery is a completed-response fallback.  It must use
    # the original offline decoder even when the resident model also supports
    # native incremental PCM; otherwise the engine performs the slower native
    # block loop internally and merely buffers those blocks before returning.
    # Keep the outward delivery label distinct so callers can still tell a
    # completed PCM phrase from a Full Quality export.
    if delivery_mode in {OFFLINE_FULL_DECODE_MODE, "buffered-fallback"}:
        options["qwen3_tts.decode_mode"] = "offline_full"
        expected_decode_mode = OFFLINE_FULL_DECODE_MODE
    if options:
        engine_payload["options"] = options
    else:
        engine_payload.pop("options", None)
    reference, reference_stats, reference_text = _prepare_reference_pair(
        engine_payload,
        tuning_snapshot,
        f"audio-cpp-reference-{token_hex(8)}.wav",
    )
    lifecycle_headers = {
        "X-TTS-Engine-Epoch": str(int(payload.get("_engine_epoch") or state.get("engineEpoch") or 0)),
        "X-TTS-Supervisor-Instance-Id": SUPERVISOR_INSTANCE_ID,
    }
    engine_payload.update(
        model=state["activeModel"],
        voice_ref=str(reference),
        reference_text=reference_text,
        response_format="wav",
        stream=False,
    )
    cap = _install_request_limit(payload, engine_payload, delivery_mode)
    outcome_id = _begin_request_outcome(payload, str(state["activeModel"]), delivery_mode)
    _finish_request_outcome(outcome_id, "producing", generationCap=cap)
    lifecycle_headers["X-TTS-Request-Id"] = outcome_id
    started = time.monotonic()
    diagnostics = _native_request_diagnostics(payload, tuning_snapshot, engine_payload, reference_stats, str(state["activeModel"]))
    diagnostics["deliveryMode"] = delivery_mode
    _strip_private_engine_fields(engine_payload)
    try:
        wall_seconds = 600.0 if delivery_mode == OFFLINE_FULL_DECODE_MODE else LIVE_SYNTHESIS_WALL_SECONDS
        async with httpx.AsyncClient(timeout=wall_seconds) as client:
            response = await asyncio.wait_for(
                _post_engine_with_disconnect(client, engine_payload, request=request),
                timeout=wall_seconds,
            )
        if response.is_error:
            termination = response.headers.get("X-AudioCPP-Termination", "")
            # Completed-result exceptions cannot carry engine artifact headers.
            # Match only the engine-owned marker, never infer degeneration from
            # arbitrary error prose or repeat a failed request in another mode.
            if "qwen3_tts.max_generated_frames_exhausted" in response.text[:4096]:
                termination = "max_generated_frames"
            _finish_request_outcome(outcome_id, "limited" if termination == "max_generated_frames" else "error",
                                    "max_generated_frames" if termination == "max_generated_frames" else "engine-error")
            if termination == "max_generated_frames":
                _raise_generation_limit()
            proven_decoder_mode = _engine_decode_mode(response)
            upstream_decoder_mode = proven_decoder_mode or "unreported"
            _record_event(
                "generation-error",
                model_id=state["activeModel"],
                status=response.status_code,
                decoderMode=upstream_decoder_mode,
                **_reference_event_fields(reference_stats, delivery_mode),
            )
            error_headers = {
                "X-TTS-Decoder-Mode": upstream_decoder_mode,
                **lifecycle_headers,
                **_reference_headers(reference_stats, delivery_mode),
            }
            if proven_decoder_mode:
                error_headers[ENGINE_DECODE_MODE_HEADER] = proven_decoder_mode
            return Response(
                content=response.content,
                status_code=response.status_code,
                media_type=response.headers.get("content-type"),
                headers=error_headers,
            )
        try:
            decoder_mode = _validate_engine_decode_mode(response, expected_decode_mode)
        except HTTPException as exc:
            _record_event(
                "generation-error",
                model_id=state["activeModel"],
                status=exc.status_code,
                decoderMode=_engine_decode_mode(response),
                expectedDecoderMode=expected_decode_mode,
                **_reference_event_fields(reference_stats, delivery_mode),
            )
            raise
        termination = response.headers.get("X-AudioCPP-Termination", "")
        if termination == "max_generated_frames":
            _finish_request_outcome(outcome_id, "limited", "max_generated_frames")
            _raise_generation_limit()
        if delivery_mode == "buffered-fallback" and termination != "eos":
            raise HTTPException(502, "audio.cpp buffered synthesis lacks engine-owned EOS telemetry")
        generation_fields = {}
        if termination == "eos":
            try:
                generation_fields = _generation_event_fields({
                    "generated_frames": int(response.headers["X-AudioCPP-Generated-Frames"]),
                    "generation_cap": int(response.headers["X-AudioCPP-Generation-Cap"]),
                    "termination": termination,
                })
            except (KeyError, ValueError) as exc:
                raise HTTPException(502, "Malformed audio.cpp completed generation telemetry") from exc
        try:
            content, media_type, extension, metadata = _render_master_wav(
                response.content, requested_format,
            )
        except HTTPException as exc:
            _record_event(
                "generation-error",
                model_id=state["activeModel"],
                status=exc.status_code,
                decoderMode=decoder_mode,
                requestedFormat=requested_format,
                stage="completed-master-export",
                **_reference_event_fields(reference_stats, delivery_mode),
            )
            raise
        # The engine permits EOS immediately after exactly cap codec frames.
        # Duration alone cannot distinguish that valid final lookahead step
        # from a cap termination; buffered output already requires engine EOS.
        _finish_request_outcome(outcome_id, "completed", termination or "complete-unreported-eos",
                                audioSeconds=metadata.get("durationSeconds"), bytes=len(content),
                                elapsedSeconds=round(time.monotonic() - started, 3),
                                **generation_fields)
        _record_event(
            "generation-complete",
            model_id=state["activeModel"],
            requestedFormat=requested_format,
            returnedFormat=extension,
            bytes=len(content),
            durationSeconds=metadata.get("durationSeconds"),
            decoderMode=decoder_mode,
            codec=metadata.get("codec"),
            container=metadata.get("container"),
            sampleRate=metadata.get("sampleRate"),
            bitsPerSample=metadata.get("bitsPerSample"),
            channels=metadata.get("channels"),
            elapsedSeconds=round(time.monotonic() - started, 3),
            realtimeFactor=round((time.monotonic() - started) / metadata["durationSeconds"], 3) if metadata.get("durationSeconds") else None,
            **diagnostics,
        )
        result_headers = {
            "X-TTS-Model": state["activeModel"],
            "X-TTS-Codec": str(metadata["codec"]),
            "X-TTS-Container": str(metadata["container"]),
            "X-TTS-Quality": str(metadata["quality"]),
            "X-TTS-Sample-Rate": str(metadata["sampleRate"]),
            "X-TTS-Bits-Per-Sample": str(metadata["bitsPerSample"]),
            "X-TTS-Channels": str(metadata["channels"]),
            "X-TTS-Duration-Seconds": str(metadata["durationSeconds"]),
            "X-TTS-Format": extension,
            "X-TTS-Decoder-Mode": decoder_mode,
            **lifecycle_headers,
            **_reference_headers(reference_stats, delivery_mode),
        }
        if "seed" in engine_payload:
            result_headers["X-TTS-Seed"] = str(engine_payload["seed"])
        for name in ("X-AudioCPP-Generated-Frames", "X-AudioCPP-Generation-Cap", "X-AudioCPP-Termination"):
            if name in response.headers:
                result_headers[name] = response.headers[name]
        if decoder_mode != "unreported":
            result_headers[ENGINE_DECODE_MODE_HEADER] = decoder_mode
        return Response(
            content=content,
            media_type=media_type,
            headers=result_headers,
        )
    except asyncio.CancelledError:
        _finish_request_outcome(outcome_id, "cancelled", "client-cancelled")
        raise
    except (TimeoutError, httpx.TimeoutException) as exc:
        _finish_request_outcome(outcome_id, "error", "wall-timeout")
        raise HTTPException(504, "Synthesis exceeded its wall-clock deadline", headers=lifecycle_headers) from exc
    except BaseException as exc:
        _finish_request_outcome(outcome_id, "error", "synthesis-error")
        if isinstance(exc, HTTPException):
            exc.headers = {**(exc.headers or {}), **lifecycle_headers}
        raise
    finally:
        reference.unlink(missing_ok=True)


async def _native_clone_pcm_response(
    payload: dict[str, Any],
    model_id: str,
    *,
    request: Request | None = None,
) -> StreamingResponse:
    """Relay one engine request as raw, incrementally-produced PCM16 blocks.

    The engine is the sole producer of stream chunks.  The supervisor never
    reconstructs or re-chunks them, which preserves cancellation identity and
    prevents a buffered result from being replayed beside native PCM.
    """
    if bool(payload.get("x_vector_only_mode")):
        raise HTTPException(status_code=422, detail="Native PCM supports full ICL Base cloning only; x_vector_only_mode is unavailable.")
    raw = str(payload.get("ref_audio") or "")
    if not raw:
        raise HTTPException(status_code=400, detail="A Base clone reference WAV is required.")
    internal_warmup = bool(payload.get("_internal_warmup"))
    engine_payload = dict(payload)
    tuning_snapshot = _apply_engine_tuning(engine_payload)
    delivery_mode = "native-incremental-pcm"
    effective = tuning_snapshot.get("effective", {}) if isinstance(tuning_snapshot, dict) else {}
    raw_options = engine_payload.get("options") or {}
    if not isinstance(raw_options, dict):
        raise HTTPException(status_code=422, detail="options must be an object.")
    options = dict(raw_options)
    options.pop("qwen3_tts.decode_mode", None)
    for key in {"first_block_frames", "steady_block_frames", "left_context_frames"}:
        if key in effective:
            options[f"qwen3_tts.stream_{key}"] = effective[key]
    if options:
        engine_payload["options"] = options
    _normalize_gradio_seed(engine_payload)
    reference, reference_stats, reference_text = _prepare_reference_pair(
        engine_payload,
        tuning_snapshot,
        f"audio-cpp-reference-{token_hex(8)}.wav",
    )
    engine_payload.update(
        model=model_id,
        voice_ref=str(reference),
        reference_text=reference_text,
        response_format="pcm",
        stream=True,
        # SSE lets the engine deliver a result-derived mode artifact before
        # any PCM delta.  Raw chunked PCM has no metadata channel for that
        # proof, so it is not accepted as the candidate-native relay source.
        stream_format="sse",
    )
    cap = _install_request_limit(payload, engine_payload, delivery_mode)
    outcome_id = _begin_request_outcome(payload, model_id, delivery_mode)
    _finish_request_outcome(outcome_id, "pending", generationCap=cap)
    diagnostics = _native_request_diagnostics(
        payload, tuning_snapshot, engine_payload, reference_stats, model_id,
    )
    # These identifiers belong to the candidate's response lifecycle, not to
    # audio.cpp's public synthesis schema. Capture them above, then keep the
    # engine request limited to its supported fields.
    for key in ("request_id", "requestId", "response_id", "responseId"):
        engine_payload.pop(key, None)
    _strip_private_engine_fields(engine_payload)

    # Establish the upstream response and consume its engine-owned decode-mode
    # SSE proof before this endpoint commits downstream headers.  Raw PCM has
    # no metadata channel, so the proof must come from this same SSE iterator;
    # any PCM before proof is rejected rather than self-labelled as native.
    client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, read=600.0))
    response: httpx.Response | None = None
    stream_lines: Any = None
    verified_decode_mode: str | None = None
    prefetched_pcm: list[tuple[bytes, float]] = []
    preheader_proof_metrics: dict[str, float | int] = {}
    native_started = time.monotonic()
    lock_held = False
    closed = False

    async def close_upstream() -> None:
        nonlocal closed, lock_held
        if closed:
            return
        closed = True
        try:
            if response is not None:
                try:
                    await response.aclose()
                except BaseException:
                    # Cleanup remains best effort after disconnect/cancel. One
                    # broken close must not wedge every later generation.
                    pass
            try:
                await client.aclose()
            except BaseException:
                pass
        finally:
            if lock_held:
                generation_lock.release()
                lock_held = False

    async def cleanup_native_response() -> None:
        try:
            await close_upstream()
        finally:
            reference.unlink(missing_ok=True)
            _finish_request_outcome(outcome_id, "cancelled", "transport-closed")

    try:
        await _await_native_stage_with_disconnect(
            generation_lock.acquire(),
            request=request,
            stage="generation-lock",
            model_id=model_id,
            internal_warmup=internal_warmup,
        )
        lock_held = True
        _revalidate_generation_admission(payload, model_id, internal_warmup=internal_warmup)
        _finish_request_outcome(outcome_id, "producing")
        engine_started = time.monotonic()
        diagnostics["admissionQueueMs"] = round((engine_started - native_started) * 1000, 3)
        upstream_request = client.build_request("POST", f"{ENGINE_URL}/v1/audio/speech", json=engine_payload)
        response = await _await_native_stage_with_disconnect(
            asyncio.wait_for(client.send(upstream_request, stream=True), LIVE_SYNTHESIS_WALL_SECONDS),
            request=request,
            stage="upstream-send",
            model_id=model_id,
            internal_warmup=internal_warmup,
        )
        if response.is_error:
            detail = (
                await _await_native_stage_with_disconnect(
                    response.aread(),
                    request=request,
                    stage="upstream-error-body",
                    model_id=model_id,
                    internal_warmup=internal_warmup,
                )
            )[:2048]
            if not internal_warmup:
                _record_event(
                    "generation-error",
                    model_id=model_id,
                    status=response.status_code,
                    nativeIncremental=True,
                    **diagnostics,
                )
            raise HTTPException(
                status_code=502,
                detail=f"audio.cpp native stream failed ({response.status_code}): {detail.decode('utf-8', errors='replace')}",
            )
        engine_timings = _engine_stage_timings(response)
        stream_lines = _deadline_lines(response.aiter_lines(), engine_started + LIVE_SYNTHESIS_WALL_SECONDS)
        verified_decode_mode = await _await_native_stage_with_disconnect(
            _await_native_decode_mode_proof(stream_lines),
            request=request,
            stage="decode-mode-proof",
            model_id=model_id,
            internal_warmup=internal_warmup,
        )
        # Prove engine incrementality at the SSE event boundary, before raw
        # HTTP headers are committed downstream. HTTP byte-read boundaries are
        # deliberately not used as evidence because they can split or coalesce.
        prefetched_pcm = await _await_native_stage_with_disconnect(
            _prefetch_native_pcm_proof(
                stream_lines,
                verified_decode_mode=verified_decode_mode,
            ),
            request=request,
            stage="incremental-pcm-proof",
            model_id=model_id,
            internal_warmup=internal_warmup,
        )
        # The exact-once two-delta proof deliberately holds downstream headers
        # until a second engine-owned boundary exists.  Keep its timing
        # visible so native performance work can distinguish engine TTFT from
        # the intentional proof window rather than misattribute it to the
        # browser or PCM player.
        proof_first = prefetched_pcm[0][1] - native_started
        proof_ready = prefetched_pcm[-1][1] - native_started
        preheader_proof_metrics = {
            "preheaderProofFirstPcmMs": round(proof_first * 1000.0, 3),
            "preheaderProofReadyMs": round(proof_ready * 1000.0, 3),
            "preheaderProofAdditionalWaitMs": round((proof_ready - proof_first) * 1000.0, 3),
            "preheaderProofChunks": len(prefetched_pcm),
        }
    except BaseException as exc:
        detail = getattr(exc, "detail", None)
        limited = isinstance(detail, dict) and detail.get("state") == "generation-limit"
        _finish_request_outcome(outcome_id,
                                "limited" if limited else "cancelled" if isinstance(exc, asyncio.CancelledError) else "error",
                                "max_generated_frames" if limited else "client-cancelled" if isinstance(exc, asyncio.CancelledError) else "native-preheader-error")
        if isinstance(exc, HTTPException):
            exc.headers = {**(exc.headers or {}), "X-TTS-Request-Id": outcome_id}
        await cleanup_native_response()
        raise

    assert response is not None and stream_lines is not None
    assert verified_decode_mode == "native-incremental-pcm"
    if not internal_warmup:
        _record_event(
            "native-pcm-preheader-proof",
            model_id=model_id,
            **preheader_proof_metrics,
            **diagnostics,
        )
        _record_event(
            "generation-start",
            model_id=model_id,
            requestedFormat="pcm",
            nativeIncremental=True,
            decoderMode=verified_decode_mode,
            **diagnostics,
        )
        if engine_timings:
            _record_event(
                "native-pcm-engine-headers",
                model_id=model_id,
                engineTimingHeaders=engine_timings,
                **diagnostics,
            )

    async def relay() -> Any:
        started = native_started
        bytes_sent = 0
        chunks_sent = 0
        saw_audio_done = False
        saw_done_marker = False
        first_chunk_s: float | None = None
        last_chunk_at: float | None = None
        cadence_total_ms = 0.0
        cadence_count = 0
        cadence_max_ms = 0.0
        generation_fields: dict[str, Any] = {}
        try:
            for chunk, received_at in prefetched_pcm:
                if cap is not None and bytes_sent + len(chunk) > cap * 1920 * 2:
                    _raise_generation_limit()
                if first_chunk_s is None:
                    first_chunk_s = received_at - started
                    if not internal_warmup:
                        _record_event(
                            "native-pcm-first",
                            model_id=model_id,
                            firstPcmSeconds=round(first_chunk_s, 3),
                            bytes=len(chunk),
                            **diagnostics,
                        )
                if last_chunk_at is not None:
                    cadence_ms = (received_at - last_chunk_at) * 1000.0
                    cadence_total_ms += cadence_ms
                    cadence_count += 1
                    cadence_max_ms = max(cadence_max_ms, cadence_ms)
                last_chunk_at = received_at
                chunks_sent += 1
                bytes_sent += len(chunk)
                yield chunk
            async for line in stream_lines:
                event = _native_sse_event(line)
                if saw_done_marker:
                    if event is None and _is_native_terminal_transport_line(line):
                        continue
                    raise HTTPException(
                        status_code=502,
                        detail="audio.cpp native SSE emitted data after its terminal [DONE] marker",
                    )
                if saw_audio_done:
                    if event is _NATIVE_SSE_DONE:
                        saw_done_marker = True
                        continue
                    if event is None and _is_native_terminal_transport_line(line):
                        continue
                    raise HTTPException(
                        status_code=502,
                        detail="audio.cpp native SSE emitted a structured event after completion",
                    )
                if event is _NATIVE_SSE_DONE:
                    raise HTTPException(
                        status_code=502,
                        detail={
                            "state": "native-stream-incomplete",
                            "message": "audio.cpp native SSE ended before speech.audio.done",
                            "chunksReceived": chunks_sent,
                            "bytesReceived": bytes_sent,
                        },
                    )
                if event is None:
                    continue
                assert isinstance(event, dict)
                event_type = str(event.get("type") or "")
                if event_type == "speech.generation":
                    if generation_fields:
                        raise HTTPException(502, "audio.cpp native SSE repeated generation telemetry")
                    generation_fields = _generation_event_fields(event)
                    if not generation_fields["eos"]:
                        _raise_generation_limit()
                    continue
                if event_type == "error":
                    error = event.get("error")
                    message = error.get("message") if isinstance(error, dict) else None
                    raise HTTPException(
                        status_code=502,
                        detail={
                            "state": "native-stream-error",
                            "message": str(message or "audio.cpp native stream failed after headers"),
                            "chunksReceived": chunks_sent,
                            "bytesReceived": bytes_sent,
                        },
                    )
                if event_type == "speech.decode_mode":
                    candidate_mode = str(event.get("mode") or "")
                    if candidate_mode != verified_decode_mode:
                        raise HTTPException(
                            status_code=502,
                            detail={
                                "state": "decoder-mode-mismatch",
                                "expected": "native-incremental-pcm",
                                "actual": candidate_mode or None,
                            },
                        )
                    continue
                if event_type == "speech.audio.done":
                    if not generation_fields:
                        raise HTTPException(502, "audio.cpp native SSE lacks engine-owned generation telemetry")
                    done_mode = str(event.get("decode_mode") or verified_decode_mode)
                    if done_mode != verified_decode_mode:
                        raise HTTPException(
                            status_code=502,
                            detail={
                                "state": "decoder-mode-mismatch",
                                "expected": verified_decode_mode,
                                "actual": done_mode or None,
                            },
                        )
                    saw_audio_done = True
                    continue
                if event_type != "speech.audio.delta":
                    continue
                if generation_fields:
                    raise HTTPException(502, "audio.cpp native SSE emitted PCM after generation telemetry")
                if verified_decode_mode != "native-incremental-pcm":
                    raise HTTPException(
                        status_code=502,
                        detail={"state": "decoder-mode-proof-missing", "expected": "native-incremental-pcm"},
                    )
                chunk = _decode_native_pcm_delta(event)
                if not chunk:
                    continue
                if cap is not None and bytes_sent + len(chunk) > cap * 1920 * 2:
                    _raise_generation_limit()
                if first_chunk_s is None:
                    first_chunk_s = time.monotonic() - started
                    if not internal_warmup:
                        _record_event(
                            "native-pcm-first",
                            model_id=model_id,
                            firstPcmSeconds=round(first_chunk_s, 3),
                            bytes=len(chunk),
                            **diagnostics,
                        )
                now = time.monotonic()
                if last_chunk_at is not None:
                    cadence_ms = (now - last_chunk_at) * 1000.0
                    cadence_total_ms += cadence_ms
                    cadence_count += 1
                    cadence_max_ms = max(cadence_max_ms, cadence_ms)
                last_chunk_at = now
                chunks_sent += 1
                bytes_sent += len(chunk)
                yield chunk
            if chunks_sent == 0:
                raise HTTPException(status_code=502, detail="audio.cpp native SSE produced no PCM delta")
            if not saw_audio_done:
                raise HTTPException(
                    status_code=502,
                    detail={
                        "state": "native-stream-incomplete",
                        "message": "audio.cpp native SSE ended without speech.audio.done",
                        "chunksReceived": chunks_sent,
                        "bytesReceived": bytes_sent,
                    },
                )
            if not saw_done_marker:
                raise HTTPException(
                    status_code=502,
                    detail={
                        "state": "native-stream-incomplete",
                        "message": "audio.cpp native SSE ended without terminal [DONE]",
                        "chunksReceived": chunks_sent,
                        "bytesReceived": bytes_sent,
                    },
                )
            if generation_fields and bytes_sent != generation_fields["generatedFrames"] * 1920 * 2:
                raise HTTPException(502, "audio.cpp generation frame count differs from emitted PCM")
            _finish_request_outcome(outcome_id, "completed", "eos",
                                    **generation_fields, bytes=bytes_sent, audioSeconds=bytes_sent / 48000,
                                    elapsedSeconds=round(time.monotonic() - started, 3))
            if not internal_warmup:
                elapsed_seconds = time.monotonic() - started
                audio_seconds = bytes_sent / (24000 * 2)
                _record_event(
                    "generation-complete",
                    model_id=model_id,
                    requestedFormat="pcm",
                    returnedFormat="pcm",
                    nativeIncremental=True,
                    chunks=chunks_sent,
                    bytes=bytes_sent,
                    firstChunkSeconds=round(first_chunk_s, 3) if first_chunk_s is not None else None,
                    elapsedSeconds=round(elapsed_seconds, 3),
                    audioSeconds=round(audio_seconds, 3),
                    realtimeFactor=round(elapsed_seconds / audio_seconds, 3) if audio_seconds else None,
                    chunkCadenceMeanMs=round(cadence_total_ms / cadence_count, 3) if cadence_count else None,
                    chunkCadenceMaxMs=round(cadence_max_ms, 3) if cadence_count else None,
                    **generation_fields,
                    **diagnostics,
                )
        except asyncio.CancelledError:
            _finish_request_outcome(outcome_id, "cancelled", "client-cancelled", bytes=bytes_sent,
                                    audioSeconds=bytes_sent / 48000)
            if not internal_warmup:
                _record_event(
                    "generation-cancelled",
                    model_id=model_id,
                    nativeIncremental=True,
                    chunks=chunks_sent,
                    **diagnostics,
                )
            raise
        except BaseException as exc:
            detail = getattr(exc, "detail", None)
            limited = isinstance(detail, dict) and detail.get("state") == "generation-limit"
            _finish_request_outcome(outcome_id, "limited" if limited else "error",
                                    "max_generated_frames" if limited else "wall-timeout" if isinstance(exc, TimeoutError) else "native-stream-error",
                                    **generation_fields, bytes=bytes_sent, audioSeconds=bytes_sent / 48000,
                                    elapsedSeconds=round(time.monotonic() - started, 3))
            if not internal_warmup:
                detail = getattr(exc, "detail", None)
                _record_event(
                    "generation-error",
                    model_id=model_id,
                    nativeIncremental=True,
                    chunks=chunks_sent,
                    bytes=bytes_sent,
                    error=detail if detail is not None else str(exc),
                    **diagnostics,
                )
            raise
        finally:
            await cleanup_native_response()

    return _OwnedStreamingResponse(
        relay(),
        cleanup=cleanup_native_response,
        media_type="audio/pcm",
        headers={
            "X-TTS-Model": model_id,
            "X-TTS-Codec": "pcm_s16le",
            "X-TTS-Container": "raw PCM",
            "X-TTS-Format": "pcm",
            "X-TTS-Sample-Rate": "24000",
            "X-TTS-Bits-Per-Sample": "16",
            "X-TTS-Channels": "1",
            # These exact values are admitted only after consuming the
            # engine-owned speech.decode_mode event from this response.
            "X-TTS-Streaming-Mode": "native-incremental-pcm",
            "X-TTS-Decoder-Mode": verified_decode_mode,
            # This proof is generated only after the supervisor consumes two
            # nonempty engine speech.audio.delta SSE events. Downstream HTTP
            # read sizes are explicitly not part of the native proof.
            "X-TTS-Native-Engine-Chunk-Proof": "two-distinct-sse-delta-events",
            "X-TTS-Native-Preheader-Proof-First-PCM-Ms": str(
                preheader_proof_metrics["preheaderProofFirstPcmMs"]
            ),
            "X-TTS-Native-Preheader-Proof-Ready-Ms": str(
                preheader_proof_metrics["preheaderProofReadyMs"]
            ),
            "X-TTS-Native-Preheader-Proof-Additional-Wait-Ms": str(
                preheader_proof_metrics["preheaderProofAdditionalWaitMs"]
            ),
            "X-TTS-Request-Id": str(diagnostics["requestId"]),
            "X-TTS-Engine-Epoch": str(diagnostics["engineEpoch"]),
            "X-TTS-Supervisor-Instance-Id": str(diagnostics["supervisorInstanceId"]),
            "X-TTS-Tuning-Scope": str(diagnostics["tuningScope"] or ""),
            "X-TTS-Tuning-Profile-Id": str(diagnostics["tuningProfileId"] or ""),
            "X-TTS-Tuning-Profile-Revision": str(diagnostics["tuningProfileRevision"] or ""),
            **_reference_headers(reference_stats, delivery_mode),
        },
    )


async def _run_native_load_warmup(model_id: str) -> dict[str, Any]:
    """Consume one private native utterance so explicit Load absorbs lazy GPU work."""
    guard = gpu_guard(model_id, operation="synthesis")
    if not guard["ok"]:
        reason = "Native load warmup skipped because the synthesis GPU guard is not satisfied."
        _record_event("model-warmup-skipped", model_id=model_id, reason=reason, gpu=guard)
        return {"status": "skipped", "reason": reason, "profileId": None, "elapsedSeconds": None}
    inventory = _candidate_profile_response()
    voices = inventory.get("voices") or []
    if not isinstance(voices, list) or not voices:
        reason = "Native load warmup skipped because no live Base clone profile is available."
        _record_event("model-warmup-skipped", model_id=model_id, reason=reason)
        return {"status": "skipped", "reason": reason, "profileId": None, "elapsedSeconds": None}
    voice_ids = [str(item.get("id") or "") for item in voices if isinstance(item, dict)]
    preferred = str(inventory.get("selectedVoice") or inventory.get("defaultVoice") or "")
    if preferred.startswith("clone:"):
        preferred = preferred.removeprefix("clone:")
    profile_id = preferred if preferred in voice_ids else next((item for item in voice_ids if item), "")
    if not profile_id:
        reason = "Native load warmup skipped because the live clone inventory has no valid profile ID."
        _record_event("model-warmup-skipped", model_id=model_id, reason=reason)
        return {"status": "skipped", "reason": reason, "profileId": None, "elapsedSeconds": None}
    profile = _candidate_profile_payload(profile_id, include_audio=True)
    payload: dict[str, Any] = {
        "input": "Ready.",
        "task_type": "Base",
        "ref_audio": profile["ref_audio"],
        "ref_text": profile["ref_text"],
        "response_format": "pcm",
        "stream": True,
        "_internal_warmup": True,
        "_engine_epoch": int(state.get("engineEpoch") or 0),
        "tuning": {
            "provider": TUNING_PROVIDER,
            "profile_id": "balanced",
            "scope": "realtime",
            "overrides": {"seed": 321},
        },
    }
    payload["_tuning_snapshot"] = _resolve_request_tuning(payload)
    response = await _native_clone_pcm_response(payload, model_id)
    started = time.monotonic()
    chunks = 0
    byte_count = 0
    async for chunk in response.body_iterator:
        if chunk:
            chunks += 1
            byte_count += len(chunk)
    if chunks == 0 or byte_count == 0:
        raise RuntimeError("native warmup returned no PCM")
    elapsed = round(time.monotonic() - started, 3)
    _record_event(
        "model-warmup-complete",
        model_id=model_id,
        profileId=profile_id,
        elapsedSeconds=elapsed,
        chunks=chunks,
        bytes=byte_count,
    )
    return {
        "status": "complete",
        "reason": None,
        "profileId": profile_id,
        "elapsedSeconds": elapsed,
    }


@app.post("/v1/voice-studio/llamacpp-audio-turn/stream")
async def llamacpp_audio_turn(request: Request) -> StreamingResponse:
    """Pass through the Studio's user-configured llama.cpp SSE turn."""
    payload = await request.json()
    endpoint = _container_reachable_llm_endpoint(str(payload.pop("endpoint", "")).strip())
    api_key = str(payload.pop("api_key", "")).strip()
    if not endpoint.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="A valid llama.cpp HTTP endpoint is required.")
    messages = [{"role": "system", "content": payload.pop("system_prompt", "")}]
    messages.extend(_llamacpp_history(payload.pop("history", [])))
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": payload.pop("prompt", "Respond to the spoken message.")},
                _llamacpp_audio_part(
                    str(payload.pop("audio_data_url", "")),
                    str(payload.pop("llm_input_format", "wav")),
                ),
            ],
        }
    )
    upstream = {"model": payload.pop("model", ""), "messages": messages, "stream": True, **payload}
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    async def relay():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", endpoint, json=upstream, headers=headers) as response:
                if response.is_error:
                    body = await response.aread()
                    yield f'data: {{"error": {{"message": {json.dumps(body.decode(errors="replace"))}}}}}\n\n'.encode()
                    return
                async for chunk in response.aiter_bytes():
                    yield chunk

    return StreamingResponse(relay(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
