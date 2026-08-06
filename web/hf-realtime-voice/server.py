"""
Tiny server for the speech-to-speech demo.

The demo used to ship as a `sdk: static` Space, but the web-search tool needs a
search key the browser must NOT see. A static Space has no runtime process, so it
can't hold a secret the front-end uses. This server fixes that: it serves the
unchanged front-end AND exposes a same-origin `/api/search` proxy that holds the
Serper key server-side (see docs/adr/0001).

Everything lives in one container; the speech-to-speech backend stays a separate,
load-balanced service the browser talks to over WebSocket as before. The load
balancer's address is a secret too (like the Serper key): the browser never sees
it. `/api/session` proxies the session handshake server-side so only the
per-session compute URL the LB hands back (which the browser must dial) is exposed.

On the deployed Space the server also meters conversation time by HF login tier
(anonymous / signed-in / PRO) — see `limiter.py` and `auth.py`. That whole feature
is off unless BOTH `LOAD_BALANCER_URL` and `SPACE_ID` are set, so it runs only on
the live Space, never locally (even with the LB exported for testing).

Endpoints:
  GET  /api/config           -> { search, lb, allowDirect, auth }
  GET  /api/me               -> login + tier + remaining budget (LB mode only)
  POST /api/search           -> { results, answer }  Google via Serper.dev
  POST /api/session          -> proxies <LB>/session: a grant, or a queue ticket
  GET  /api/queue/{id}       -> proxies <LB>/queue/{id}: position, or a grant on claim
  DELETE /api/queue/{id}     -> leave the queue (explicit "Leave queue" button)
  POST /api/queue/end        -> leave the queue (sendBeacon on teardown)
  POST /api/session/heartbeat-> extend the reservation; { expired }
  POST /api/session/end      -> reconcile + refund (sendBeacon on teardown)
  /*                         -> static files (index.html, main.js, ...)

When every compute slot is busy the load balancer hands back a queue ticket
instead of a grant; the browser polls /api/queue/{id} until it reaches the front
and a slot frees. Waiting reserves nothing — the daily budget is only reserved at
the moment a slot is actually claimed (a grant), never while queued.
"""

import asyncio
import base64
import json
import logging
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from secrets import token_hex
from typing import Any

import auth
import httpx
import limiter
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logger = logging.getLogger("s2s.search")

SERPER_KEY = os.environ.get("SERPER_API_KEY", "").strip()
# Speech-to-speech load balancer URL. When set, the browser POSTs /api/session
# (which proxies <lb>/session here, server-side) and connects to the URL the LB
# returns (the original flow). The LB address itself is never sent to the browser.
# When empty, the user may instead set a direct s2s server URL in Settings and the
# browser connects to it straight (no load balancer).
LOAD_BALANCER_URL = os.environ.get("LOAD_BALANCER_URL", "").strip()
# HF injects SPACE_ID ("owner/space") into every Space runtime; it's absent
# locally and on a plain `docker run`. We meter conversation time ONLY on the
# deployed Space — i.e. when BOTH the LB is configured AND we're on a Space.
# Off-Space (local dev, even with the LB exported) the app still proxies the LB,
# but nothing is metered: no budget, no reservations, no sign-in gating.
SPACE_ID = os.environ.get("SPACE_ID", "").strip()
LIMITER_ENABLED = bool(LOAD_BALANCER_URL) and bool(SPACE_ID)
SERPER_URL = "https://google.serper.dev/search"
# Cap results so the tool output stays small enough to feed back to the model.
MAX_RESULTS = 5
LOCAL_UI_API_VERSION = 20
HERE = os.path.dirname(os.path.abspath(__file__))
_repo_runtime_dir = Path(HERE).parents[1] / ".runtime"
# In the repository the legacy shared runtime is two levels above the UI.
# The self-contained candidate installs this module at /opt/voice-studio, so
# that parent does not exist there; candidate settings instead live under its
# private VOICE_LIBRARY_DIR.
_legacy_runtime_dir = (Path(HERE).parents[2] / ".runtime") if len(Path(HERE).parents) > 2 else _repo_runtime_dir
_ui_settings_default = (
    Path(os.environ["VOICE_LIBRARY_DIR"]) / "hf_realtime_ui_settings.json"
    if os.environ.get("VOICE_LIBRARY_DIR")
    else _repo_runtime_dir / "hf_realtime_ui_settings.json"
)
UI_SETTINGS_PATH = Path(os.environ.get("S2S_UI_SETTINGS_PATH", str(_ui_settings_default)))
AUDIO_CPP_VALIDATION_PATH = UI_SETTINGS_PATH.with_name("audio_cpp_validation.json")
TTS_VALIDATION_PATH = UI_SETTINGS_PATH.with_name("tts_backend_validation.json")
_LEGACY_UI_SETTINGS_PATH = _legacy_runtime_dir / UI_SETTINGS_PATH.name
_LEGACY_AUDIO_CPP_VALIDATION_PATH = _legacy_runtime_dir / AUDIO_CPP_VALIDATION_PATH.name
_LEGACY_TTS_VALIDATION_PATH = _legacy_runtime_dir / TTS_VALIDATION_PATH.name
PUBLIC_UI_SETTING_KEYS = {
    "directUrl",
    "voice",
    "instructions",
    "noiseGate",
    "echoGuard",
    "echoCalibrations",
    "fullBufferTts",
    "liveTranscript",
    "maxResponseTokens",
    "ttsBackend",
    "modelProvider",
    "modelUrl",
    "modelName",
    "voiceByBackend",
    "ttsProfileByBackend",
}
DEFAULT_QWEN3_VOICE_ID = "16d9bb336799"
DEFAULT_QWEN3_VOICE = f"clone:{DEFAULT_QWEN3_VOICE_ID}"
TTS_BACKENDS = {
    "faster": {
        "endpoint": "http://127.0.0.1:8881/v1", "requiredModel": "1.7B-Base",
        "displayName": "FasterQwen3TTS", "profileMode": "local", "explicitValidation": False,
    },
    "groxaxo": {
        "endpoint": "http://127.0.0.1:8882/v1", "requiredModel": None,
        "displayName": "Groxaxo candidate", "profileMode": "readonly", "explicitValidation": False,
    },
    # This endpoint is intentionally opt-in.  audio.cpp is a separately managed
    # candidate and is never started, stopped, or selected by this application.
    "qwen3tts-audiocpp": {
        "endpoint": os.environ.get("AUDIO_CPP_TTS_BASE_URL", "http://127.0.0.1:8890/v1"),
        "requiredModel": "qwen3-tts",
        "displayName": "Qwen3TTS audio.cpp", "profileMode": "remote", "explicitValidation": True,
    },
}


def _canonical_tts_provider(provider: Any) -> Any:
    return "qwen3tts-audiocpp" if provider == "audio-cpp" else provider


def _normalize_tts_provider_settings(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    normalized["ttsBackend"] = _canonical_tts_provider(normalized.get("ttsBackend"))
    for key in ("voiceByBackend", "ttsProfileByBackend"):
        value = normalized.get(key)
        if not isinstance(value, dict):
            continue
        providers = dict(value)
        if "audio-cpp" in providers and "qwen3tts-audiocpp" not in providers:
            providers["qwen3tts-audiocpp"] = providers["audio-cpp"]
        providers.pop("audio-cpp", None)
        normalized[key] = providers
    return normalized


DEFAULT_VOICE_LIBRARY_DIR = Path(
    os.environ.get(
        "VOICE_LIBRARY_DIR",
        r"C:\Users\yepyy\Documents\Codex\2026-05-24\files-mentioned-by-the-user-i\qwen3-tts-candidate\voice_library_from_original",
    )
).expanduser()

app = FastAPI(title="s2s-demo")


def _read_public_ui_settings() -> dict[str, Any]:
    """Load local non-secret UI settings, if the managed frontend has saved any."""
    paths = (UI_SETTINGS_PATH, _LEGACY_UI_SETTINGS_PATH) if UI_SETTINGS_PATH == _ui_settings_default else (UI_SETTINGS_PATH,)
    for path in paths:
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(saved, dict):
            return _normalize_tts_provider_settings(saved)
    return {}


def _write_public_ui_settings(payload: dict[str, Any]) -> dict[str, Any]:
    """Atomically save frontend preferences without ever retaining API keys."""
    payload = _normalize_tts_provider_settings(payload)
    existing = _read_public_ui_settings()
    for key in PUBLIC_UI_SETTING_KEYS:
        value = payload.get(key)
        if key == "voiceByBackend" and isinstance(value, dict):
            existing[key] = {
                str(provider)[:64]: str(voice)[:256]
                for provider, voice in value.items()
                if provider in TTS_BACKENDS and isinstance(voice, str) and voice.startswith("clone:")
            }
        elif key == "ttsProfileByBackend" and isinstance(value, dict):
            existing[key] = {
                str(provider): str(profile_id)
                for provider, profile_id in value.items()
                if provider in TTS_BACKENDS
                and isinstance(profile_id, str)
                and re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", profile_id)
            }
        elif key == "echoCalibrations" and isinstance(value, dict):
            limits = {
                "delayMs": (0.0, 500.0),
                "suppressionStrength": (0.0, 1.0),
                "leakageThreshold": (0.05, 1.0),
                "doubleTalkSensitivity": (0.0, 1.0),
                "echoTailMs": (0.0, 1000.0),
            }
            calibrations: dict[str, dict[str, float]] = {}
            for pair, calibration in list(value.items())[:16]:
                if not isinstance(pair, str) or not isinstance(calibration, dict):
                    continue
                cleaned: dict[str, float] = {}
                for field, (minimum, maximum) in limits.items():
                    candidate = calibration.get(field)
                    if isinstance(candidate, bool) or not isinstance(candidate, (int, float)):
                        continue
                    cleaned[field] = max(minimum, min(maximum, float(candidate)))
                if cleaned:
                    calibrations[pair[:512]] = cleaned
            existing[key] = calibrations
        elif isinstance(value, (str, int, float, bool)):
            existing[key] = value
    UI_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = UI_SETTINGS_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(UI_SETTINGS_PATH)
    return existing


def _read_audio_cpp_validation() -> dict[str, str]:
    saved: Any = None
    paths = (
        (AUDIO_CPP_VALIDATION_PATH, _LEGACY_AUDIO_CPP_VALIDATION_PATH)
        if AUDIO_CPP_VALIDATION_PATH == _ui_settings_default.with_name("audio_cpp_validation.json")
        else (AUDIO_CPP_VALIDATION_PATH,)
    )
    for path in paths:
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
            break
        except (OSError, ValueError):
            continue
    if not isinstance(saved, dict):
        return {}
    model_id = saved.get("model_id")
    profile_id = saved.get("profile_id")
    if not isinstance(model_id, str) or not isinstance(profile_id, str):
        return {}
    return {"model_id": model_id, "profile_id": profile_id, "validated_at": str(saved.get("validated_at") or "")}


def _write_audio_cpp_validation(model_id: str, profile_id: str) -> dict[str, str]:
    """Remember a successful candidate capability probe, not a model lifecycle action."""
    saved = {
        "model_id": model_id,
        "profile_id": profile_id,
        "validated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    AUDIO_CPP_VALIDATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = AUDIO_CPP_VALIDATION_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(saved, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(AUDIO_CPP_VALIDATION_PATH)
    return saved


def _read_tts_validations() -> dict[str, Any]:
    saved: Any = None
    paths = (
        (TTS_VALIDATION_PATH, _LEGACY_TTS_VALIDATION_PATH)
        if TTS_VALIDATION_PATH == _ui_settings_default.with_name("tts_backend_validation.json")
        else (TTS_VALIDATION_PATH,)
    )
    for path in paths:
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
            break
        except (OSError, ValueError):
            continue
    if not isinstance(saved, dict):
        saved = {}
    records = saved.get("_records")
    if not isinstance(records, dict):
        records = {}
        saved["_records"] = records
    # Migrate the original one-entry-per-backend shape without discarding it.
    # Keeping individual model+clone records means validating one candidate
    # clone no longer makes another already-validated clone unusable.
    for backend in TTS_BACKENDS:
        entry = saved.get(backend)
        if not isinstance(entry, dict) or not entry.get("speech"):
            continue
        backend_records = records.setdefault(backend, {})
        if isinstance(backend_records, dict):
            backend_records.setdefault(_tts_validation_key(entry.get("model"), str(entry.get("voice") or "")), entry)
    legacy = _read_audio_cpp_validation()
    if legacy and "qwen3tts-audiocpp" not in saved:
        entry = {
            "model": legacy.get("model_id"), "voice": f"clone:{legacy.get('profile_id')}",
            "validatedAt": legacy.get("validated_at", ""), "speech": True,
        }
        saved["qwen3tts-audiocpp"] = entry
        records.setdefault("qwen3tts-audiocpp", {})[
            _tts_validation_key(entry.get("model"), str(entry.get("voice") or ""))
        ] = entry
    return saved


def _tts_validation_key(model: Any, voice: str) -> str:
    return f"{str(model or '')}|{voice}"


def _find_tts_validation(backend: str, model: str | None, voice: str) -> dict[str, Any]:
    saved = _read_tts_validations()
    records = saved.get("_records") or {}
    backend_records = records.get(backend) if isinstance(records, dict) else None
    if isinstance(backend_records, dict):
        entry = backend_records.get(_tts_validation_key(model, voice))
        if isinstance(entry, dict) and entry.get("speech"):
            return entry
    latest = saved.get(backend)
    if (
        isinstance(latest, dict)
        and latest.get("speech")
        and latest.get("model") == model
        and latest.get("voice") == voice
    ):
        return latest
    return {}


def _write_tts_validation(
    backend: str,
    model: str | None,
    voice: str,
    *,
    delivery_mode: str | None = None,
    native_streaming: bool | None = None,
) -> dict[str, Any]:
    saved = _read_tts_validations()
    entry = {
        "model": model, "voice": voice, "speech": True,
        "validatedAt": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    if delivery_mode:
        entry["deliveryMode"] = delivery_mode
    if native_streaming is not None:
        entry["nativeStreaming"] = bool(native_streaming)
    saved[backend] = entry
    records = saved.setdefault("_records", {})
    backend_records = records.setdefault(backend, {})
    backend_records[_tts_validation_key(model, voice)] = entry
    TTS_VALIDATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = TTS_VALIDATION_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(saved, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(TTS_VALIDATION_PATH)
    return entry

# Wire HF OAuth before the app serves (no-op unless the OAuth env is present).
# Sign-in only matters when we're metering (prod Space), so gate it on that.
AUTH_ENABLED = LIMITER_ENABLED and auth.attach(app)


@app.on_event("startup")
async def _startup():
    """Stand up the usage DB and a periodic sweeper — metered (prod Space) only."""
    if not LIMITER_ENABLED:
        return
    limiter.init()
    asyncio.create_task(_sweeper())


async def _sweeper():
    while True:
        await asyncio.sleep(limiter.REAP_AFTER_SEC)
        try:
            await asyncio.to_thread(limiter.sweep)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("usage sweep failed: %r", exc)


class SearchRequest(BaseModel):
    query: str
    # Optional user-supplied key (fallback when the deploy has no server key).
    # Used for this request only; never stored.
    key: str | None = None


class ModelTestRequest(BaseModel):
    provider: str = "local"
    base_url: str | None = None
    model: str | None = None
    api_key: str | None = None


class ProfileCreateRequest(BaseModel):
    name: str
    ref_text: str = ""
    language: str = "Auto"
    source_profile_id: str | None = None


class ProfileEditRequest(BaseModel):
    name: str | None = None
    ref_text: str | None = None
    language: str | None = None


class ProfileImportRequest(ProfileCreateRequest):
    audio_base64: str
    audio_filename: str = "ref_audio.wav"


class ProfileSelectRequest(BaseModel):
    profile_id: str


class ModelControlRequest(BaseModel):
    model_id: str | None = None


class AudioCppValidationRequest(BaseModel):
    model_id: str | None = None
    profile_id: str | None = None


class AudioCppModelSwitchRequest(BaseModel):
    model_id: str


@app.get("/api/ui-settings")
def get_public_ui_settings():
    return {"settings": _read_public_ui_settings(), "apiKeysPersisted": False}


@app.put("/api/ui-settings")
def save_public_ui_settings(payload: dict[str, Any]):
    return {"settings": _write_public_ui_settings(payload), "apiKeysPersisted": False}


class VoiceStudioSynthesisRequest(BaseModel):
    model_id: str
    profile_id: str
    text: str


@app.post("/api/model/test")
async def test_model_endpoint(req: ModelTestRequest):
    if req.provider == "local":
        base_url = os.environ.get("GEMMA_AUDIO_BASE_URL", "http://127.0.0.1:8818/v1")
        model = os.environ.get("GEMMA_AUDIO_MODEL", "gemma-4-12b-it-qat")
        api_key = os.environ.get("GEMMA_API_KEY") or os.environ.get("LLAMA_CPP_API_KEY")
    elif req.provider == "remote":
        base_url = (req.base_url or "").strip()
        model = (req.model or "").strip()
        api_key = req.api_key or None
        if not base_url or not model:
            raise HTTPException(status_code=400, detail="Remote URL and model name are required.")
    else:
        raise HTTPException(status_code=400, detail="Unknown model provider.")

    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=5.0), headers=headers) as http:
            response = await http.get(f"{base_url}/models")
            response.raise_for_status()
            models = response.json().get("data") or []
            advertised = next((str(item.get("id")) for item in models if item.get("id")), None)
            context_window = None
            try:
                props = await http.get(f"{base_url.removesuffix('/v1')}/props")
                props.raise_for_status()
                context_window = props.json().get("default_generation_settings", {}).get("n_ctx")
            except Exception:
                pass
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Model endpoint unavailable: {type(exc).__name__}") from exc
    return {
        "provider": req.provider,
        "model": advertised or model,
        "configured_model": model,
        "context_window": int(context_window) if context_window else None,
        "api_key_set": bool(api_key),
    }


@app.get("/api/config")
def config():
    """Client bootstrap: whether web search is available, whether the deploy runs
    behind a load balancer (so the browser uses the /api/session proxy + limiter),
    whether HF sign-in is available, and whether the user may instead set a direct
    s2s server URL. The LB address itself is intentionally NOT included."""
    return {
        "apiVersion": LOCAL_UI_API_VERSION,
        "search": bool(SERPER_KEY),
        "lb": bool(LOAD_BALANCER_URL),
        "allowDirect": not LOAD_BALANCER_URL,
        "auth": AUTH_ENABLED,
    }


def _load_base_clone_profiles(library_dir: Path) -> list[dict]:
    profiles_dir = library_dir / "profiles"
    if not profiles_dir.exists():
        return []

    voices = []
    for meta_file in sorted(profiles_dir.glob("*/meta.json")):
        try:
            data = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Skipping unreadable voice profile metadata: %s", meta_file)
            continue
        if data.get("task_type") != "Base":
            continue
        profile_id = str(data.get("profile_id") or meta_file.parent.name).strip()
        if not profile_id:
            continue
        name = str(data.get("name") or profile_id).strip()
        voices.append(
            {
                "id": profile_id,
                "voice": f"clone:{profile_id}",
                "name": name,
                "task_type": data.get("task_type"),
                "created_at": data.get("created_at") or "",
                "ref_text": data.get("ref_text") or "",
                "language": data.get("language") or "Auto",
            }
        )

    voices.sort(key=lambda item: (item["id"] != DEFAULT_QWEN3_VOICE_ID, item["name"].lower()))
    return voices


def _profile_path(library_dir: Path, profile_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,80}", profile_id):
        raise HTTPException(status_code=400, detail="Invalid clone profile id.")
    return library_dir / "profiles" / profile_id


def _read_profile(library_dir: Path, profile_id: str) -> dict[str, Any]:
    meta_path = _profile_path(library_dir, profile_id) / "meta.json"
    if not meta_path.is_file():
        raise HTTPException(status_code=404, detail="Clone profile was not found.")
    try:
        profile = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail="Clone profile metadata is unreadable.") from exc
    if profile.get("task_type") != "Base":
        raise HTTPException(status_code=400, detail="Only Base clone profiles are supported by FasterQwen3TTS.")
    return profile


def _ensure_profile_library_writable(library_dir: Path) -> Path:
    profiles_dir = library_dir / "profiles"
    try:
        profiles_dir.mkdir(parents=True, exist_ok=True)
        probe = profiles_dir / f".s2s-write-test-{token_hex(4)}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise HTTPException(
            status_code=409,
            detail=f"Configured Faster voice library is not writable: {library_dir}",
        ) from exc
    return profiles_dir


def _new_profile_id(library_dir: Path) -> str:
    while True:
        profile_id = token_hex(6)
        if not (library_dir / "profiles" / profile_id).exists():
            return profile_id


def _write_profile(directory: Path, profile: dict[str, Any]) -> None:
    (directory / "meta.json").write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")


def _profile_response(library_dir: Path) -> dict[str, Any]:
    selected_path = library_dir / "selected_profile.json"
    selected = DEFAULT_QWEN3_VOICE_ID
    try:
        selected = str(json.loads(selected_path.read_text(encoding="utf-8")).get("profile_id") or selected)
    except (OSError, json.JSONDecodeError):
        pass
    return {
        "defaultVoice": DEFAULT_QWEN3_VOICE,
        "selectedVoice": f"clone:{selected}",
        "libraryDir": str(library_dir),
        "writable": os.access(library_dir, os.W_OK),
        "voices": _load_base_clone_profiles(library_dir),
    }


@app.get("/api/qwen3/voices")
def qwen3_voices():
    """Return saved Faster-side Base clone profiles for the realtime settings UI."""
    return _profile_response(DEFAULT_VOICE_LIBRARY_DIR)


def _normalize_remote_voices(payload: Any) -> list[dict[str, Any]]:
    items = payload.get("data") if isinstance(payload, dict) else None
    if items is None and isinstance(payload, dict):
        items = payload.get("voices")
    normalized: list[dict[str, Any]] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        raw_voice = str(item.get("voice") or item.get("id") or "")
        if not raw_voice.startswith("clone:"):
            if str(item.get("task") or item.get("task_type") or "").lower() == "base" and raw_voice:
                raw_voice = f"clone:{raw_voice}"
            else:
                continue
        normalized.append({
            "id": str(item.get("profile_id") or item.get("id") or raw_voice.removeprefix("clone:")),
            "voice": raw_voice,
            "name": str(item.get("name") or raw_voice.removeprefix("clone:")),
            "task_type": "Base",
            "language": str(item.get("language") or "Auto"),
            "ref_text": str(item.get("ref_text") or ""),
        })
    return normalized


async def _backend_voice_inventory(backend: str) -> dict[str, Any]:
    if backend not in TTS_BACKENDS:
        raise HTTPException(status_code=404, detail="Unknown TTS backend.")
    config = TTS_BACKENDS[backend]
    endpoint = str(config["endpoint"]).rstrip("/")
    base = {
        "backend": backend,
        "displayName": config.get("displayName", backend),
        "reachable": False,
        "writable": config.get("profileMode") in {"local", "remote"},
        "management": config.get("profileMode"),
        "defaultVoice": DEFAULT_QWEN3_VOICE if backend == "faster" else None,
        "selectedVoice": None,
        "voices": [],
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=2.0)) as http:
            if backend == "qwen3tts-audiocpp":
                response = await http.get(f"{endpoint}/voices/profiles")
                response.raise_for_status()
                remote = _scope_audio_cpp_profile_response(response.json())
                base.update(
                    reachable=True,
                    voices=remote["voices"],
                    defaultVoice=remote.get("defaultVoice"),
                    selectedVoice=remote.get("selectedVoice"),
                )
                return base
            response = await http.get(f"{endpoint}/voices")
            response.raise_for_status()
            remote_voices = _normalize_remote_voices(response.json())
            base["reachable"] = True
            if backend == "faster":
                local = _load_base_clone_profiles(DEFAULT_VOICE_LIBRARY_DIR)
                live_names = {
                    str(item["voice"]).removeprefix("clone:").casefold()
                    for item in remote_voices
                }
                reconciled = [item for item in local if str(item.get("name") or "").casefold() in live_names]
                base["voices"] = reconciled
                profile_state = _profile_response(DEFAULT_VOICE_LIBRARY_DIR)
                base["selectedVoice"] = profile_state["selectedVoice"]
                if not any(item["voice"] == DEFAULT_QWEN3_VOICE for item in reconciled) and reconciled:
                    base["defaultVoice"] = reconciled[0]["voice"]
            else:
                base["voices"] = remote_voices
                base["writable"] = False
                base["defaultVoice"] = remote_voices[0]["voice"] if remote_voices else None
    except Exception as exc:
        base["error"] = f"{type(exc).__name__}: {exc}"
    return base


def _scope_audio_cpp_profile_response(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize only the candidate's live inventory without cross-provider filtering."""
    result = dict(payload)
    voices = _normalize_remote_voices(payload)
    available = {str(profile["voice"]) for profile in voices}
    default_voice = str(payload.get("defaultVoice") or "")
    selected_voice = str(payload.get("selectedVoice") or "")
    fallback = str(voices[0]["voice"]) if voices else None
    result["voices"] = voices
    result["defaultVoice"] = default_voice if default_voice in available else fallback
    result["selectedVoice"] = selected_voice if selected_voice in available else result["defaultVoice"]
    return result


@app.get("/api/tts/backends/{backend}/voices")
async def backend_voices(backend: str, response: Response):
    response.headers["Cache-Control"] = "no-store, max-age=0"
    response.headers["Pragma"] = "no-cache"
    return await _backend_voice_inventory(backend)


async def _remote_profile_request(backend: str, method: str, suffix: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    endpoint = str(TTS_BACKENDS[backend]["endpoint"]).rstrip("/")
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=3.0)) as http:
        response = await http.request(method, f"{endpoint}/voices/profiles{suffix}", json=payload)
        if response.is_error:
            try:
                detail = response.json().get("detail")
            except Exception:
                detail = response.text
            raise HTTPException(status_code=response.status_code, detail=detail or "Backend profile operation failed.")
        result = response.json()
        return _scope_audio_cpp_profile_response(result) if backend == "qwen3tts-audiocpp" else result


@app.post("/api/tts/backends/{backend}/profiles")
async def create_backend_profile(backend: str, request: Request):
    payload = await request.json()
    if backend == "faster":
        if payload.get("audio_base64"):
            return import_qwen3_profile(ProfileImportRequest(**payload))
        return create_qwen3_profile(ProfileCreateRequest(**payload))
    if backend == "qwen3tts-audiocpp":
        if payload.get("audio_base64"):
            payload["ref_audio"] = payload.pop("audio_base64")
        return await _remote_profile_request(backend, "POST", "", payload)
    raise HTTPException(status_code=409, detail="The selected backend exposes clone inventory only; manage it in its own Voice Studio.")


@app.patch("/api/tts/backends/{backend}/profiles/{profile_id}")
async def edit_backend_profile(backend: str, profile_id: str, request: Request):
    payload = await request.json()
    if backend == "faster":
        return edit_qwen3_profile(profile_id, ProfileEditRequest(**payload))
    if backend == "qwen3tts-audiocpp":
        return await _remote_profile_request(backend, "PATCH", f"/{profile_id}", payload)
    raise HTTPException(status_code=409, detail="The selected backend profile library is read-only here.")


@app.delete("/api/tts/backends/{backend}/profiles/{profile_id}")
async def delete_backend_profile(backend: str, profile_id: str):
    if backend == "faster":
        return delete_qwen3_profile(profile_id)
    if backend == "qwen3tts-audiocpp":
        return await _remote_profile_request(backend, "DELETE", f"/{profile_id}")
    raise HTTPException(status_code=409, detail="The selected backend profile library is read-only here.")


@app.post("/api/tts/backends/{backend}/profiles/{profile_id}/select")
async def select_backend_profile(backend: str, profile_id: str):
    if backend == "faster":
        return select_qwen3_profile(ProfileSelectRequest(profile_id=profile_id))
    if backend == "qwen3tts-audiocpp":
        return await _remote_profile_request(backend, "POST", f"/{profile_id}/select")
    return await _backend_voice_inventory(backend)


@app.post("/api/qwen3/profiles")
def create_qwen3_profile(req: ProfileCreateRequest):
    """Create a Base clone profile by copying an existing compatible reference."""
    library_dir = DEFAULT_VOICE_LIBRARY_DIR
    profiles_dir = _ensure_profile_library_writable(library_dir)
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="A profile name is required.")
    if not req.source_profile_id:
        raise HTTPException(status_code=400, detail="Creating a clone needs an existing Base profile or an audio import.")
    source_dir = _profile_path(library_dir, req.source_profile_id)
    source = _read_profile(library_dir, req.source_profile_id)
    reference_name = str(source.get("ref_audio_filename") or "ref_audio.wav")
    source_audio = source_dir / reference_name
    if not source_audio.is_file():
        raise HTTPException(status_code=409, detail="The selected source profile has no reference audio.")
    profile_id = _new_profile_id(library_dir)
    target_dir = profiles_dir / profile_id
    target_dir.mkdir()
    shutil.copy2(source_audio, target_dir / "ref_audio.wav")
    source.update({
        "profile_id": profile_id,
        "name": name,
        "ref_text": req.ref_text or str(source.get("ref_text") or ""),
        "language": req.language or str(source.get("language") or "Auto"),
        "ref_audio_filename": "ref_audio.wav",
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "origin": "Clone(Base)",
    })
    _write_profile(target_dir, source)
    return _profile_response(library_dir)


@app.post("/api/qwen3/profiles/import")
def import_qwen3_profile(req: ProfileImportRequest):
    """Import a browser-supplied WAV reference into the configured Faster library."""
    library_dir = DEFAULT_VOICE_LIBRARY_DIR
    profiles_dir = _ensure_profile_library_writable(library_dir)
    name = req.name.strip()
    if not name or not req.ref_text.strip():
        raise HTTPException(status_code=400, detail="Import requires a profile name and reference transcript.")
    try:
        audio = base64.b64decode(req.audio_base64, validate=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Reference audio is not valid base64.") from exc
    if not audio.startswith(b"RIFF") or b"WAVE" not in audio[:16]:
        raise HTTPException(status_code=400, detail="Only WAV reference audio is supported for this clone library.")
    if len(audio) > 64 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Reference audio exceeds the 64 MiB import limit.")
    profile_id = _new_profile_id(library_dir)
    target_dir = profiles_dir / profile_id
    target_dir.mkdir()
    (target_dir / "ref_audio.wav").write_bytes(audio)
    _write_profile(target_dir, {
        "profile_id": profile_id,
        "name": name,
        "task_type": "Base",
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "language": req.language or "Auto",
        "voice": "",
        "instructions": "",
        "ref_text": req.ref_text.strip(),
        "x_vector_only_mode": False,
        "ref_audio_filename": "ref_audio.wav",
        "origin": "Clone(Base)",
    })
    return _profile_response(library_dir)


@app.patch("/api/qwen3/profiles/{profile_id}")
def edit_qwen3_profile(profile_id: str, req: ProfileEditRequest):
    library_dir = DEFAULT_VOICE_LIBRARY_DIR
    _ensure_profile_library_writable(library_dir)
    profile = _read_profile(library_dir, profile_id)
    if req.name is not None:
        if not req.name.strip():
            raise HTTPException(status_code=400, detail="Profile name cannot be empty.")
        profile["name"] = req.name.strip()
    if req.ref_text is not None:
        profile["ref_text"] = req.ref_text
    if req.language is not None:
        profile["language"] = req.language or "Auto"
    _write_profile(_profile_path(library_dir, profile_id), profile)
    return _profile_response(library_dir)


@app.delete("/api/qwen3/profiles/{profile_id}")
def delete_qwen3_profile(profile_id: str):
    if profile_id == DEFAULT_QWEN3_VOICE_ID:
        raise HTTPException(status_code=409, detail="The configured J.A.R.V.I.S default profile cannot be deleted.")
    library_dir = DEFAULT_VOICE_LIBRARY_DIR
    _ensure_profile_library_writable(library_dir)
    profile_dir = _profile_path(library_dir, profile_id)
    _read_profile(library_dir, profile_id)
    shutil.rmtree(profile_dir)
    return _profile_response(library_dir)


@app.post("/api/qwen3/profiles/select")
def select_qwen3_profile(req: ProfileSelectRequest):
    library_dir = DEFAULT_VOICE_LIBRARY_DIR
    _ensure_profile_library_writable(library_dir)
    _read_profile(library_dir, req.profile_id)
    (library_dir / "selected_profile.json").write_text(
        json.dumps({"profile_id": req.profile_id}, indent=2) + "\n", encoding="utf-8"
    )
    return _profile_response(library_dir)


@app.get("/api/local-pipeline")
async def local_pipeline():
    """Best-effort live description of the local runtime backing the UI."""
    gemma_base = os.environ.get("GEMMA_AUDIO_BASE_URL", "http://127.0.0.1:8818/v1").rstrip("/")
    tts_base = os.environ.get("QWEN3_TTS_API_BASE_URL", "http://127.0.0.1:8881/v1").rstrip("/")
    status = {
        "mode": "local-direct-audio",
        "vad": {"name": "Silero VAD", "device": "CPU", "sampleRate": 16000},
        "gemma": {"baseUrl": gemma_base, "model": os.environ.get("GEMMA_AUDIO_MODEL", "gemma-4-12b-it-qat")},
        "tts": {
            "baseUrl": tts_base,
            "backend": "qwen3-tts-faster",
            "apiModel": "qwen3-tts",
            "model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            "voice": DEFAULT_QWEN3_VOICE,
            "container": "281c411e5cfe02d0b0d903f67cd3fb712ab38bc705eb3edbedd8a2041c78702b",
        },
        "tools": {"serper": bool(SERPER_KEY), "camera": True},
    }
    async with httpx.AsyncClient(timeout=1.5) as http:
        try:
            gemma = await http.get(f"{gemma_base}/models")
            status["gemma"]["reachable"] = gemma.status_code == 200
            if gemma.status_code == 200:
                data = gemma.json()
                models = data.get("data") or []
                if models and isinstance(models[0], dict):
                    status["gemma"]["model"] = models[0].get("id") or status["gemma"]["model"]
        except Exception:
            status["gemma"]["reachable"] = False
        try:
            props = await http.get(f"{gemma_base.removesuffix('/v1')}/props")
            if props.status_code == 200:
                n_ctx = props.json().get("default_generation_settings", {}).get("n_ctx")
                status["gemma"]["contextWindow"] = int(n_ctx) if n_ctx else None
        except Exception:
            status["gemma"]["contextWindow"] = None
        try:
            health_base = tts_base.removesuffix("/v1")
            tts = await http.get(f"{health_base}/health")
            status["tts"]["reachable"] = tts.status_code == 200
            if tts.status_code == 200:
                status["tts"]["health"] = tts.json()
        except Exception:
            status["tts"]["reachable"] = False
    return status


async def _faster_model_inventory() -> dict[str, Any]:
    """Describe the fixed Faster service without pretending it has a model API."""
    endpoint = TTS_BACKENDS["faster"]["endpoint"]
    result: dict[str, Any] = {
        "provider": "faster",
        "activeModel": "1.7B-Base",
        "models": [{
            "id": "1.7B-Base",
            "displayName": "Qwen3-TTS 1.7B Base",
            "default": True,
            "loaded": False,
            "precision": "backend-managed",
        }],
        "controls": {
            "inventory": True,
            "load": False,
            "switch": False,
            "unload": False,
            "reason": "The stable FasterQwen3TTS API is clone-only and exposes no model lifecycle endpoints.",
        },
    }
    try:
        async with httpx.AsyncClient(timeout=1.5) as http:
            health = await http.get(f"{endpoint.removesuffix('/v1')}/health")
            health.raise_for_status()
            payload = health.json()
            result["models"][0]["loaded"] = bool(payload.get("model_loaded"))
            result["health"] = payload
    except Exception as exc:
        result["error"] = type(exc).__name__
    return result


@app.get("/api/qwen3/models")
async def qwen3_models():
    """Inventory/control contract for the configured Faster service.

    The production API does not implement lifecycle operations, so load/switch/
    unload are represented explicitly as unavailable rather than simulated.
    """
    return await _faster_model_inventory()


@app.post("/api/qwen3/models/{action}")
async def qwen3_model_control(action: str, req: ModelControlRequest):
    if action not in {"load", "switch", "unload"}:
        raise HTTPException(status_code=404, detail="Unknown Faster model operation.")
    inventory = await _faster_model_inventory()
    raise HTTPException(
        status_code=409,
        detail=inventory["controls"]["reason"],
    )


def _audio_cpp_model_ids() -> tuple[str, ...]:
    configured = os.environ.get(
        "AUDIO_CPP_TTS_MODELS",
        "qwen3-tts-0.6b-base-bf16,qwen3-tts-1.7b-base-bf16",
    )
    return tuple(item.strip() for item in configured.split(",") if item.strip())


async def _sync_audio_cpp_profile(profile_id: str) -> None:
    """Copy a selected Faster-compatible Base profile into candidate-private storage.

    The original library remains read-only; this only creates/updates the isolated
    candidate copy so the same stable ``clone:<id>`` value works at port 7862.
    """
    profile = _read_profile(DEFAULT_VOICE_LIBRARY_DIR, profile_id)
    reference_name = str(profile.get("ref_audio_filename") or "ref_audio.wav")
    reference = _profile_path(DEFAULT_VOICE_LIBRARY_DIR, profile_id) / reference_name
    if not reference.is_file():
        raise HTTPException(status_code=409, detail="The selected clone profile has no reference audio to synchronize.")
    endpoint = str(TTS_BACKENDS["qwen3tts-audiocpp"]["endpoint"]).rstrip("/")
    payload = {
        "name": str(profile.get("name") or profile_id),
        "language": str(profile.get("language") or "Auto"),
        "ref_text": str(profile.get("ref_text") or ""),
        "x_vector_only_mode": bool(profile.get("x_vector_only_mode")),
        "ref_audio": base64.b64encode(reference.read_bytes()).decode("ascii"),
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=3.0)) as http:
            response = await http.post(f"{endpoint}/voices/profiles/{profile_id}", json=payload)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Could not synchronize the clone to audio.cpp: {exc}") from exc


async def _probe_audio_cpp_candidate(
    *, run_speech_probe: bool = False, model_id: str | None = None, profile_id: str | None = None
) -> dict[str, Any]:
    """Validate audio.cpp strictly before it can be selected as a realtime TTS.

    Native PCM is accepted only when the running candidate advertises it and a
    direct chunked speech probe succeeds.  A completed-phrase request remains an
    explicit, truthfully labelled fallback.  No lifecycle action is performed;
    probes use only the already-resident model.
    """
    config = TTS_BACKENDS["qwen3tts-audiocpp"]
    endpoint = str(config["endpoint"]).rstrip("/")
    control_endpoint = os.environ.get("AUDIO_CPP_CONTROL_URL", f"{endpoint.removesuffix('/v1')}/control").rstrip("/")
    candidate_models = _audio_cpp_model_ids()
    requested_model = model_id or None
    if requested_model and requested_model not in candidate_models:
        raise HTTPException(status_code=400, detail="The requested audio.cpp model is not configured for this candidate.")
    result: dict[str, Any] = {
        "id": "qwen3tts-audiocpp",
        "endpoint": endpoint,
        "displayName": "Qwen3TTS audio.cpp",
        "profileMode": "remote",
        "explicitValidation": True,
        "reachable": False,
        "streaming": False,
        "cloneCompatible": False,
        "currentModel": None,
        "requiredModel": requested_model,
        "ready": False,
        "capabilityChecked": run_speech_probe,
        "limitations": [],
    }
    try:
        # Health/model polling should stay quick, but the one-time Base-clone
        # synthesis deliberately follows the candidate's normal speech timeout.
        # A 1.7B cold/warm request can exceed the old eight-second probe window.
        timeout = httpx.Timeout(600.0, connect=5.0) if run_speech_probe else httpx.Timeout(8.0, connect=2.0)
        async with httpx.AsyncClient(timeout=timeout) as http:
            health = await http.get(f"{endpoint.removesuffix('/v1')}/health")
            health.raise_for_status()
            result["reachable"] = True
            result["health"] = health.json()
            models_response = await http.get(f"{endpoint}/models")
            models_response.raise_for_status()
            models = models_response.json().get("data") or []
            ids = [str(item.get("id")) for item in models if isinstance(item, dict) and item.get("id")]
            result["models"] = ids
            runtime = (health.json().get("backend") or {}).get("runtime") or {}
            current_model = (health.json().get("backend") or {}).get("model_id")
            result["currentModel"] = current_model
            result["progressivePcm"] = bool(runtime.get("progressive_phrase_pcm"))
            result["nativeIncrementalPcm"] = bool(runtime.get("native_incremental_pcm"))
            result["deliveryMode"] = (
                "native-incremental-pcm"
                if result["nativeIncrementalPcm"]
                else "progressive-buffered-pcm"
            )
            result["bufferedFallback"] = not result["nativeIncrementalPcm"]
            if not result["nativeIncrementalPcm"]:
                result["limitations"].append(
                    "This candidate uses completed phrase PCM; native model-level PCM is unavailable."
                )
            if requested_model and current_model != requested_model:
                result["error"] = f"Load {requested_model!r} in the isolated audio.cpp Voice Studio before selecting this provider."
                result["state"] = runtime.get("state") or "unloaded"
                return result
            if current_model not in ids:
                result["error"] = "Load one of the configured audio.cpp models in Voice Studio before selecting this provider."
                result["state"] = runtime.get("state") or "unloaded"
                return result
            result["requiredModel"] = current_model
            if not run_speech_probe:
                public_settings = _read_public_ui_settings()
                voice_by_backend = public_settings.get("voiceByBackend") or {}
                selected_voice = str(
                    voice_by_backend.get("qwen3tts-audiocpp")
                    or (
                        public_settings.get("voice")
                        if public_settings.get("ttsBackend") == "qwen3tts-audiocpp"
                        else ""
                    )
                    or ""
                )
                validation = _find_tts_validation(
                    "qwen3tts-audiocpp", current_model, selected_voice
                )
                validation_voice = str(validation.get("voice") or "")
                validation_profile = validation_voice.removeprefix("clone:")
                if validation.get("model") == current_model and validation.get("speech"):
                    voices_response = await http.get(f"{endpoint}/voices")
                    voices_response.raise_for_status()
                    candidate_voice_ids = {
                        str(item.get("id")) for item in voices_response.json().get("data") or [] if isinstance(item, dict)
                    }
                    if validation_profile in candidate_voice_ids:
                        validated_native = bool(validation.get("nativeStreaming"))
                        validated_mode = str(
                            validation.get("deliveryMode")
                            or (
                                "native-incremental-pcm"
                                if validated_native
                                else "progressive-buffered-pcm"
                            )
                        )
                        if result["nativeIncrementalPcm"] and not validated_native:
                            result["limitations"].append(
                                "The stored speech validation proves buffered fallback only; revalidate to prove native chunks."
                            )
                        result.update(
                            capabilityChecked=True,
                            cloneCompatible=True,
                            ready=True,
                            streaming=validated_native,
                            nativeStreaming=validated_native,
                            bufferedFallback=not validated_native,
                            deliveryMode=validated_mode,
                            mode=validated_mode,
                            validation=validation,
                        )
                        return result
                result["error"] = "Validate the loaded audio.cpp model and selected Base clone once from TTS Backend."
                return result
            voices_response = await http.get(f"{endpoint}/voices")
            voices_response.raise_for_status()
            candidate_voice_ids = {str(item.get("id")) for item in voices_response.json().get("data") or [] if isinstance(item, dict)}
            if profile_id and profile_id not in candidate_voice_ids:
                result["error"] = "The selected clone profile has not been synchronized to the isolated audio.cpp library."
                result["cloneCompatible"] = False
                return result
            payload: dict[str, Any] = {
                "model": current_model,
                "input": "Capability check.",
                "voice": f"clone:{profile_id}" if profile_id else "default",
                "response_format": "pcm",
                "stream": bool(result["nativeIncrementalPcm"]),
            }
            if not profile_id:
                result["limitations"].append("Import and select a Base clone profile before validating audio.cpp voice cloning.")
            probe_started = asyncio.get_running_loop().time()
            first_chunk_ms: float | None = None
            chunks: list[bytes] = []
            streaming_mode_header = ""
            try:
                async with http.stream(
                    "POST",
                    f"{endpoint}/audio/speech",
                    json=payload,
                ) as speech:
                    speech.raise_for_status()
                    content_type = speech.headers.get("content-type", "").lower()
                    streaming_mode_header = str(
                        speech.headers.get("x-tts-streaming-mode", "")
                    )
                    async for chunk in speech.aiter_bytes():
                        if not chunk:
                            continue
                        if first_chunk_ms is None:
                            first_chunk_ms = (asyncio.get_running_loop().time() - probe_started) * 1000
                        chunks.append(chunk)
            except httpx.HTTPError as native_exc:
                if not result["nativeIncrementalPcm"]:
                    raise
                result["nativeProbeError"] = str(native_exc)
                payload["stream"] = False
                probe_started = asyncio.get_running_loop().time()
                async with http.stream(
                    "POST",
                    f"{endpoint}/audio/speech",
                    json=payload,
                ) as speech:
                    speech.raise_for_status()
                    content_type = speech.headers.get("content-type", "").lower()
                    chunks = []
                    async for chunk in speech.aiter_bytes():
                        if not chunk:
                            continue
                        if first_chunk_ms is None:
                            first_chunk_ms = (asyncio.get_running_loop().time() - probe_started) * 1000
                        chunks.append(chunk)
                result.update(
                    bufferedFallback=True,
                    deliveryMode="buffered-fallback",
                    nativeStreaming=False,
                )
                result["limitations"].append(
                    "Native PCM probe failed; validation succeeded through the completed-phrase fallback."
                )
            native_header_valid = streaming_mode_header == "native-incremental-pcm"
            native_chunk_evidence = len(chunks) >= 2
            result["nativeStreaming"] = bool(
                result["nativeIncrementalPcm"]
                and not result.get("nativeProbeError")
                and native_header_valid
                and native_chunk_evidence
            )
            if result["nativeIncrementalPcm"] and not result["nativeStreaming"] and not result.get("nativeProbeError"):
                failures = []
                if not native_header_valid:
                    failures.append("the exact X-TTS-Streaming-Mode: native-incremental-pcm header was absent")
                if not native_chunk_evidence:
                    failures.append("fewer than two nonempty PCM chunks were observed")
                result["nativeProbeError"] = "; ".join(failures)
                result.update(
                    bufferedFallback=True,
                    deliveryMode="buffered-fallback",
                )
                result["limitations"].append(
                    "Native PCM validation failed: " + result["nativeProbeError"] + ". "
                    "Returned speech is reported only as buffered capability."
                )
            result["bufferedSpeech"] = bool(chunks) and not result["nativeStreaming"]
            result["speechProbe"] = {
                "chunks": len(chunks),
                "bytes": sum(len(chunk) for chunk in chunks),
                "firstChunkMs": round(first_chunk_ms, 3) if first_chunk_ms is not None else None,
                "streamingModeHeader": streaming_mode_header or None,
                "nativeHeaderValid": native_header_valid,
                "incrementalEvidence": native_chunk_evidence,
            }
            result["contentType"] = content_type
            result["streaming"] = bool(result["nativeStreaming"])
            result["cloneCompatible"] = bool(profile_id and profile_id in candidate_voice_ids)
            if not chunks:
                result["error"] = "Candidate returned no speech bytes."
            elif profile_id:
                result["ready"] = True
                result["mode"] = result["deliveryMode"]
            else:
                result["error"] = "Candidate speech works; select a Base clone profile to test its direct audio.cpp clone request."
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


@app.post("/api/tts/audio-cpp/validate")
async def validate_audio_cpp_candidate(req: AudioCppValidationRequest):
    if req.profile_id:
        await _sync_audio_cpp_profile(req.profile_id)
    result = await _probe_audio_cpp_candidate(
        run_speech_probe=True, model_id=req.model_id, profile_id=req.profile_id
    )
    if result.get("ready") and req.profile_id and result.get("currentModel"):
        result["validation"] = _write_audio_cpp_validation(str(result["currentModel"]), req.profile_id)
    return result


@app.post("/api/tts/audio-cpp/select-model")
async def select_audio_cpp_model(req: AudioCppModelSwitchRequest):
    if req.model_id not in _audio_cpp_model_ids():
        raise HTTPException(status_code=400, detail="The requested audio.cpp model is not configured for this candidate.")
    raise HTTPException(
        status_code=409,
        detail=(
            "Load or switch audio.cpp models explicitly in the isolated candidate Voice Studio. "
            "The speech-to-speech UI never changes candidate model residency."
        ),
    )


def _audio_cpp_proxy_base() -> str:
    return str(TTS_BACKENDS["qwen3tts-audiocpp"]["endpoint"]).rstrip("/")


@app.get("/api/audio-cpp/settings")
async def proxy_audio_cpp_settings() -> Response:
    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=5.0)) as http:
        upstream = await http.get(f"{_audio_cpp_proxy_base()}/voice-studio/settings")
    return Response(content=upstream.content, status_code=upstream.status_code, media_type="application/json")


@app.put("/api/audio-cpp/settings")
async def save_proxy_audio_cpp_settings(request: Request) -> Response:
    payload = await request.json()
    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=5.0)) as http:
        upstream = await http.put(f"{_audio_cpp_proxy_base()}/voice-studio/settings", json=payload)
    return Response(content=upstream.content, status_code=upstream.status_code, media_type="application/json")


@app.api_route("/api/audio-cpp/tuning/{path:path}", methods=["GET", "POST", "PATCH", "PUT", "DELETE"])
async def proxy_audio_cpp_tuning(path: str, request: Request) -> Response:
    """Same-origin proxy only: candidate-private supervisor owns tuning state."""
    allowed = {"profiles", "selection", "resolve", "export", "import"}
    if not path or path.split("/", 1)[0] not in allowed or ".." in path:
        raise HTTPException(status_code=404, detail="Unknown tuning endpoint.")
    kwargs: dict[str, Any] = {}
    if request.method in {"POST", "PATCH", "PUT"}:
        kwargs["json"] = await request.json()
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0)) as http:
        upstream = await http.request(request.method, f"{_audio_cpp_proxy_base()}/tuning/{path}", **kwargs)
    return Response(content=upstream.content, status_code=upstream.status_code, media_type="application/json")


@app.post("/api/audio-cpp/audio/speech")
async def proxy_audio_cpp_speech(request: Request) -> Response:
    """Relay native PCM incrementally while retaining complete buffered replies."""
    payload = await request.json()
    http = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=5.0, read=None))
    try:
        upstream_request = http.build_request(
            "POST", f"{_audio_cpp_proxy_base()}/audio/speech", json=payload
        )
        upstream = await http.send(upstream_request, stream=True)
    except BaseException:
        await http.aclose()
        raise
    headers = {
        key: value for key, value in upstream.headers.items() if key.lower().startswith("x-tts-")
    }
    media_type = upstream.headers.get("content-type", "application/octet-stream")
    if upstream.is_error:
        content = await upstream.aread()
        await upstream.aclose()
        await http.aclose()
        return Response(
            content=content,
            status_code=upstream.status_code,
            media_type=media_type,
            headers=headers,
        )

    async def relay():
        try:
            async for chunk in upstream.aiter_raw():
                if await request.is_disconnected():
                    break
                if chunk:
                    yield chunk
        except asyncio.CancelledError:
            raise
        finally:
            await upstream.aclose()
            await http.aclose()

    return StreamingResponse(
        relay(),
        status_code=upstream.status_code,
        media_type=media_type,
        headers=headers,
    )


@app.post("/api/audio-cpp/audio/voice-clone")
async def proxy_audio_cpp_voice_clone(request: Request) -> Response:
    payload = await request.json()
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=5.0)) as http:
        upstream = await http.post(f"{_audio_cpp_proxy_base()}/audio/voice-clone", json=payload)
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/octet-stream"),
        headers={key: value for key, value in upstream.headers.items() if key.lower().startswith("x-tts-")},
    )


@app.post("/api/audio-cpp/llamacpp-audio-turn/stream")
async def proxy_audio_cpp_llamacpp_turn(request: Request) -> StreamingResponse:
    """Keep Studio llama.cpp SSE on the same browser origin as Gradio."""
    payload = await request.json()

    async def relay():
        async with httpx.AsyncClient(timeout=None) as http:
            async with http.stream(
                "POST",
                f"{_audio_cpp_proxy_base()}/voice-studio/llamacpp-audio-turn/stream",
                json=payload,
            ) as upstream:
                if upstream.is_error:
                    body = await upstream.aread()
                    yield f'data: {{"error": {{"message": {json.dumps(body.decode(errors="replace"))}}}}}\n\n'.encode()
                    return
                async for chunk in upstream.aiter_bytes():
                    yield chunk

    return StreamingResponse(relay(), media_type="text/event-stream")


@app.post("/api/voice-studio/test")
async def voice_studio_test(req: VoiceStudioSynthesisRequest):
    """Candidate test generation using the public supervisor/runtime contract."""
    if req.model_id not in _audio_cpp_model_ids():
        raise HTTPException(status_code=400, detail="Unknown candidate model.")
    profile = _read_profile(DEFAULT_VOICE_LIBRARY_DIR, req.profile_id)
    endpoint = str(TTS_BACKENDS["qwen3tts-audiocpp"]["endpoint"]).rstrip("/")
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=5.0)) as http:
        health = await http.get(f"{endpoint.removesuffix('/v1')}/health")
        health.raise_for_status()
        backend = health.json().get("backend", {})
        runtime = backend.get("runtime", {})
        current_model = backend.get("model_id") or backend.get("current_model_key")
        if runtime.get("state") != "loaded" or current_model != req.model_id:
            raise HTTPException(
                status_code=409,
                detail=(
                    "The requested audio.cpp model is not resident. Load it explicitly in the "
                    "candidate Voice Studio first; the speech-to-speech UI never switches models."
                ),
            )
        await _sync_audio_cpp_profile(req.profile_id)
        response = await http.post(
            f"{endpoint}/audio/speech",
            json={
                "model": req.model_id,
                "input": req.text.strip(),
                "voice": f"clone:{req.profile_id}",
                "response_format": "wav",
                "language": str(profile.get("language") or "Auto"),
                "stream": False,
            },
        )
        response.raise_for_status()
    return Response(
        content=response.content,
        media_type=response.headers.get("content-type", "audio/wav"),
        headers={key: value for key, value in response.headers.items() if key.lower().startswith("x-tts-")},
    )


async def _probe_tts_backend(name: str, config: dict) -> dict:
    endpoint = config["endpoint"]
    result = {
        "id": name,
        "endpoint": endpoint,
        "displayName": config.get("displayName", name),
        "profileMode": config.get("profileMode", "readonly"),
        "explicitValidation": bool(config.get("explicitValidation")),
        "reachable": False,
        "streaming": False,
        "cloneCompatible": False,
        "currentModel": None,
        "requiredModel": config.get("requiredModel"),
        "ready": False,
    }
    try:
        async with httpx.AsyncClient(timeout=1.5) as http:
            if name == "qwen3tts-audiocpp":
                return await _probe_audio_cpp_candidate()
            if name == "faster":
                response = await http.get(f"{endpoint.removesuffix('/v1')}/health")
                response.raise_for_status()
                health = response.json()
                caps = health.get("capabilities") or {}
                result.update(
                    reachable=True,
                    streaming=bool(caps.get("native_pcm_streaming")),
                    cloneCompatible=bool(caps.get("clone_only")),
                    currentModel="1.7B-Base",
                )
                result["ready"] = bool(
                    result["streaming"] and result["cloneCompatible"] and health.get("model_loaded")
                )
            else:
                response = await http.get(f"{endpoint}/backend/models")
                response.raise_for_status()
                status = response.json()
                current = str(status.get("current") or "")
                loaded = status.get("loaded_models") or []
                result.update(
                    reachable=True,
                    streaming=True,
                    cloneCompatible=current.endswith("B-Base"),
                    currentModel=current or None,
                )
                result["ready"] = bool(
                    status.get("state") == "loaded" and current in loaded and current.endswith("B-Base")
                )
    except Exception as exc:
        result["error"] = str(exc)
    return result


@app.get("/api/tts/backends")
async def tts_backends():
    statuses = await asyncio.gather(
        *(_probe_tts_backend(name, config) for name, config in TTS_BACKENDS.items())
    )
    return {"default": "faster", "backends": statuses}


def _voice_for_backend_request(backend: str, voice: str) -> str:
    if backend not in {"faster", "groxaxo"} or not voice.startswith("clone:"):
        return voice
    profile_id = voice.removeprefix("clone:")
    try:
        profile = _read_profile(DEFAULT_VOICE_LIBRARY_DIR, profile_id)
    except HTTPException:
        return voice
    name = str(profile.get("name") or "").strip()
    return f"clone:{name}" if name else voice


@app.post("/api/tts/backends/{backend}/validate")
async def validate_tts_backend(backend: str, request: Request):
    if backend not in TTS_BACKENDS:
        raise HTTPException(status_code=404, detail="Unknown TTS backend.")
    payload = await request.json()
    voice = str(payload.get("voice") or "")
    inventory = await _backend_voice_inventory(backend)
    voices = {str(item.get("voice")) for item in inventory.get("voices") or []}
    if not inventory.get("reachable"):
        raise HTTPException(status_code=503, detail=inventory.get("error") or "Selected TTS backend is unreachable.")
    if not voice or voice not in voices:
        raise HTTPException(status_code=409, detail="The selected Base clone is not present in the selected backend inventory.")

    if backend == "qwen3tts-audiocpp":
        profile_id = voice.removeprefix("clone:")
        result = await _probe_audio_cpp_candidate(run_speech_probe=True, profile_id=profile_id)
        if not result.get("ready"):
            return result
        validation = _write_tts_validation(
            backend,
            str(result.get("currentModel") or ""),
            voice,
            delivery_mode=str(result.get("deliveryMode") or result.get("mode") or ""),
            native_streaming=bool(result.get("nativeStreaming")),
        )
        _write_audio_cpp_validation(str(result.get("currentModel") or ""), profile_id)
        result["validation"] = validation
        return result

    status = await _probe_tts_backend(backend, TTS_BACKENDS[backend])
    if not status.get("ready"):
        raise HTTPException(status_code=409, detail=status.get("error") or "Selected backend has no ready Base model.")
    endpoint = str(TTS_BACKENDS[backend]["endpoint"]).rstrip("/")
    model = "qwen3-tts" if backend == "faster" else str(status.get("currentModel") or "qwen3-tts")
    started = asyncio.get_running_loop().time()
    request_payload = {
        "model": model,
        "input": "Backend validation complete.",
        "voice": _voice_for_backend_request(backend, voice),
        "response_format": "pcm",
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=5.0)) as http:
            response = await http.post(f"{endpoint}/audio/speech", json=request_payload)
            response.raise_for_status()
            audio = response.content
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Selected backend speech validation failed: {exc}") from exc
    if not audio:
        raise HTTPException(status_code=502, detail="Selected backend returned no speech bytes.")
    validation = _write_tts_validation(backend, str(status.get("currentModel") or model), voice)
    return {
        **status,
        "ready": True,
        "cloneCompatible": True,
        "speechProbe": {"bytes": len(audio), "elapsedMs": round((asyncio.get_running_loop().time() - started) * 1000)},
        "validation": validation,
    }


@app.get("/api/me")
async def me(request: Request):
    """Login state, tier, and remaining daily budget. Only meaningful in LB mode;
    sets the anonymous tracking cookie when first seen."""
    if not LIMITER_ENABLED:
        return {"enabled": False}
    view = auth.user_view(request)
    tier, keys, set_cookie = auth.resolve_identity(request)
    unlimited = limiter.budget_for(tier) is None
    rem = None if unlimited else await asyncio.to_thread(limiter.remaining, keys, tier)
    out = {
        "enabled": True,
        "auth": AUTH_ENABLED,
        **view,
        "remainingSec": rem,
        "limitSec": limiter.budget_for(tier),
        "loginUrl": auth.OAUTH_LOGIN_PATH if AUTH_ENABLED else None,
        "logoutUrl": auth.OAUTH_LOGOUT_PATH if AUTH_ENABLED else None,
    }
    resp = JSONResponse(out)
    if set_cookie:
        auth.set_anon_cookie(resp, set_cookie)
    return resp


@app.post("/api/search")
async def search(req: SearchRequest):
    """Proxy a Google search via Serper.dev. The key stays on the server unless
    the user brought their own (then theirs is used for this request only)."""
    query = (req.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Empty query.")

    key = (req.key or "").strip() or SERPER_KEY
    if not key:
        # No server key and the user didn't supply one — search is unavailable.
        raise HTTPException(status_code=503, detail="Search is not configured.")

    headers = {"X-API-KEY": key, "Content-Type": "application/json"}
    payload = {"q": query, "num": MAX_RESULTS}
    try:
        async with httpx.AsyncClient(timeout=12.0) as http:
            resp = await http.post(SERPER_URL, headers=headers, json=payload)
    except httpx.RequestError as exc:
        logger.warning("Serper unreachable: %r", exc)
        raise HTTPException(status_code=502, detail="Search provider unreachable.")

    if resp.status_code != 200:
        # Serper's error body carries the real reason (e.g. "Not enough
        # credits") and contains no key, so it's safe to log and relay.
        body = resp.text[:300]
        logger.warning("Serper error %s: %s", resp.status_code, body)
        msg = None
        try:
            msg = resp.json().get("message")
        except Exception:
            pass
        detail = f"Search provider error ({resp.status_code})"
        if msg:
            detail += f": {msg}"
        raise HTTPException(status_code=502, detail=detail)

    data = resp.json()
    results = []
    for item in (data.get("organic") or [])[:MAX_RESULTS]:
        results.append(
            {
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "url": item.get("link", ""),
            }
        )

    # A direct answer when Google has one — saves the model a hop.
    box = data.get("answerBox") or {}
    answer = box.get("answer") or box.get("snippet") or None
    if not answer:
        kg = data.get("knowledgeGraph") or {}
        answer = kg.get("description") or None

    return JSONResponse({"query": query, "answer": answer, "results": results})


@app.post("/api/session")
async def session(request: Request):
    """Proxy the session handshake to the load balancer, keeping its URL secret,
    and meter conversation time by tier.

    The browser POSTs here (same-origin); we resolve the caller's tier, refuse if
    today's budget is already spent (402), otherwise POST <LOAD_BALANCER_URL>/session
    and relay the JSON back. The LB body carries a per-session `connect_url`
    (compute host + short-lived token) the browser must dial directly — that one
    URL is unavoidably exposed, but the stable load-balancer address is not. On a
    successful grant we reserve the first time chunk against the day's budget."""
    if not LOAD_BALANCER_URL:
        # No LB configured — this deploy is direct-mode only; the browser should
        # never call this. 404 so it's indistinguishable from a missing route.
        raise HTTPException(status_code=404, detail="Not found.")

    tier, keys, set_cookie = auth.resolve_identity(request)
    # Metering runs only on the deployed Space; off-Space the LB still proxies but
    # nothing is tracked. Within metering, unlimited tiers (pro, org) aren't either.
    tracked = LIMITER_ENABLED and limiter.budget_for(tier) is not None

    # Refuse before troubling the LB if the day's budget is already gone. Done
    # here (at enqueue) so we never put a user who can't talk into the queue.
    if tracked:
        rem = await asyncio.to_thread(limiter.remaining, keys, tier)
        if rem is not None and rem <= 0:
            resp = JSONResponse(
                {"tier": tier, "reason": "limit", "remainingSec": 0}, status_code=402
            )
            if set_cookie:
                auth.set_anon_cookie(resp, set_cookie)
            return resp

    url = f"{LOAD_BALANCER_URL.rstrip('/')}/session"
    try:
        async with httpx.AsyncClient(timeout=15.0) as http:
            lb = await http.post(url, headers={"Content-Type": "application/json"}, content="{}")
    except httpx.RequestError as exc:
        logger.warning("Load balancer unreachable: %r", exc)
        raise HTTPException(status_code=502, detail="Speech service unreachable.")

    # The queue is full: the LB replies 503 {state:"at_capacity"}. Relay it as-is
    # so the client shows a soft "try again shortly", not a hard error.
    if lb.status_code == 503:
        body = _safe_json(lb)
        if body.get("state") == "at_capacity":
            resp = JSONResponse({"state": "at_capacity"}, status_code=503)
            if set_cookie:
                auth.set_anon_cookie(resp, set_cookie)
            return resp

    if lb.status_code != 200:
        # The LB's error body may name the reason (e.g. capacity); it carries no
        # secret, so relay a trimmed copy.
        logger.warning("Session handshake failed %s: %s", lb.status_code, lb.text[:300])
        raise HTTPException(status_code=502, detail=f"Session handshake failed ({lb.status_code}).")

    data = lb.json()

    # Busy pool: the LB queued us. Relay the ticket untouched — crucially with NO
    # reservation, so waiting in line never costs the day's budget.
    if data.get("state") == "queued":
        data["tier"] = tier
        resp = JSONResponse(data)
        if set_cookie:
            auth.set_anon_cookie(resp, set_cookie)
        return resp

    # A slot was free: reserve the first chunk now and return the grant.
    return await _finalize_grant(data, keys, tier, tracked, set_cookie)


@app.get("/api/queue/{queue_id}")
async def queue_status(queue_id: str, request: Request):
    """Poll a waiting ticket: relay the position, or — when the head of the line
    claims a freed slot — reserve the budget now and return the grant. Re-checks the
    daily budget at claim, since a multi-minute wait could have spent it elsewhere."""
    if not LOAD_BALANCER_URL:
        raise HTTPException(status_code=404, detail="Not found.")

    tier, keys, set_cookie = auth.resolve_identity(request)
    tracked = LIMITER_ENABLED and limiter.budget_for(tier) is not None

    url = f"{LOAD_BALANCER_URL.rstrip('/')}/queue/{queue_id}"
    try:
        async with httpx.AsyncClient(timeout=15.0) as http:
            lb = await http.get(url)
    except httpx.RequestError as exc:
        logger.warning("Load balancer unreachable: %r", exc)
        raise HTTPException(status_code=502, detail="Speech service unreachable.")

    if lb.status_code == 404:
        # Ticket unknown/expired (reaped after we stopped polling). Tell the client
        # to start over rather than spin.
        resp = JSONResponse({"state": "expired"}, status_code=404)
        if set_cookie:
            auth.set_anon_cookie(resp, set_cookie)
        return resp

    if lb.status_code != 200:
        logger.warning("Queue poll failed %s: %s", lb.status_code, lb.text[:300])
        raise HTTPException(status_code=502, detail=f"Queue poll failed ({lb.status_code}).")

    data = lb.json()

    if data.get("state") == "queued":
        data["tier"] = tier
        resp = JSONResponse(data)
        if set_cookie:
            auth.set_anon_cookie(resp, set_cookie)
        return resp

    # Claimed a slot. Re-check the budget: it may have been spent in another tab
    # during the wait. If so, refuse — the just-claimed slot is now a pending
    # session on the LB and its pending-timeout reaper reclaims it shortly.
    if tracked:
        rem = await asyncio.to_thread(limiter.remaining, keys, tier)
        if rem is not None and rem <= 0:
            resp = JSONResponse(
                {"tier": tier, "reason": "limit", "remainingSec": 0}, status_code=402
            )
            if set_cookie:
                auth.set_anon_cookie(resp, set_cookie)
            return resp

    return await _finalize_grant(data, keys, tier, tracked, set_cookie)


@app.delete("/api/queue/{queue_id}")
async def queue_leave(queue_id: str):
    """Leave the queue from the explicit 'Leave queue' button (a real fetch)."""
    if not LOAD_BALANCER_URL:
        raise HTTPException(status_code=404, detail="Not found.")
    await _lb_leave(queue_id)
    return {"ok": True}


@app.post("/api/queue/end")
async def queue_end(request: Request):
    """Leave the queue on teardown/tab-close (navigator.sendBeacon, which can only
    POST). Body: { queueId }. Best-effort; the LB reaps the ticket on TTL anyway."""
    if not LOAD_BALANCER_URL:
        raise HTTPException(status_code=404, detail="Not found.")
    qid = await _queue_id(request)
    if qid:
        await _lb_leave(qid)
    return {"ok": True}


async def _finalize_grant(data, keys, tier, tracked, set_cookie):
    """Shared grant tail (fast path or queue claim): reserve the first chunk, attach
    the metering fields the client needs, and set the anon cookie."""
    remaining = None
    if tracked and data.get("session_id"):
        await asyncio.to_thread(limiter.begin, data["session_id"], keys, tier)
        remaining = await asyncio.to_thread(limiter.remaining, keys, tier)

    data.update({
        "tier": tier,
        "limited": tracked,
        "remainingSec": remaining,
        "heartbeatSec": limiter.HEARTBEAT_SEC,
    })
    resp = JSONResponse(data)
    if set_cookie:
        auth.set_anon_cookie(resp, set_cookie)
    return resp


async def _lb_leave(queue_id: str) -> None:
    """Best-effort: tell the LB to drop a waiting ticket."""
    url = f"{LOAD_BALANCER_URL.rstrip('/')}/queue/{queue_id}"
    try:
        async with httpx.AsyncClient(timeout=5.0) as http:
            await http.delete(url)
    except httpx.RequestError as exc:
        logger.warning("Queue leave failed: %r", exc)


def _safe_json(response) -> dict:
    try:
        body = response.json()
    except Exception:
        return {}
    return body if isinstance(body, dict) else {}


async def _queue_id(request: Request) -> str:
    """Pull `queueId` from a JSON body, tolerating sendBeacon's blob posts."""
    try:
        data = await request.json()
    except Exception:
        return ""
    return (data or {}).get("queueId", "") if isinstance(data, dict) else ""


async def _session_id(request: Request) -> str:
    """Pull `sessionId` from a JSON body, tolerating sendBeacon's blob posts."""
    try:
        data = await request.json()
    except Exception:
        return ""
    return (data or {}).get("sessionId", "") if isinstance(data, dict) else ""


@app.post("/api/session/heartbeat")
async def session_heartbeat(request: Request):
    """Extend the live reservation one chunk at a time. `expired` once the day's
    budget is spent — the client then tears down."""
    if not LIMITER_ENABLED:
        raise HTTPException(status_code=404, detail="Not found.")
    sid = await _session_id(request)
    alive = bool(sid) and await asyncio.to_thread(limiter.heartbeat, sid)
    return {"expired": not alive}


@app.post("/api/session/end")
async def session_end(request: Request):
    """Clean teardown: reconcile to real elapsed time and refund the unused
    chunk. Sent via navigator.sendBeacon, so it must succeed without a response."""
    if not LIMITER_ENABLED:
        raise HTTPException(status_code=404, detail="Not found.")
    sid = await _session_id(request)
    if sid:
        await asyncio.to_thread(limiter.end, sid)
    return {"ok": True}


# Static front-end. Registered last so the /api routes win. `html=True` serves
# The recovered copied Faster-side Gradio Voice Studio is an additional testing
# surface. The root HF Realtime page remains independent for future streaming
# integration work.
try:
    import gradio as gr
    from gradio_voice_studio import build_app as build_gradio_voice_studio
    app = gr.mount_gradio_app(
        app,
        build_gradio_voice_studio(
            os.environ.get("AUDIO_CPP_TTS_BASE_URL", "http://127.0.0.1:8080/v1").removesuffix("/v1"),
            DEFAULT_VOICE_LIBRARY_DIR,
        ),
        path="/voice-studio",
    )
except ImportError:
    # The regular Realtime server stays runnable outside the candidate image.
    pass

# index.html at "/". The repo is public anyway, so serving the dir is fine.
app.mount("/", StaticFiles(directory=HERE, html=True), name="static")
