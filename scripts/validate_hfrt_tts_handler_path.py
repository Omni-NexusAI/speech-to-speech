"""Explicit, single-request acceptance check for HFRT's candidate TTS handler path.

This constructs one response-owned ``TTSInput`` and calls the production
``Qwen3TTSHandler`` snapshot/process path. It does not open a WebSocket, call
an LLM, record a microphone, save settings, load/switch a model, or write PCM
unless ``--audio-dir`` is supplied. It is therefore a TTS-handler transport
check, not an end-to-end HFRT or listening acceptance test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from copy import deepcopy
from pathlib import Path
from queue import Empty, Queue
from threading import Event
from typing import Any

import httpx
import numpy as np

from speech_to_speech.TTS.qwen3_tts_handler import Qwen3TTSHandler
from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.pipeline.messages import AudioOutput, TTSInput


EFFECTIVE_FIELDS = (
    "model", "clone_mode", "max_reference_seconds", "first_block_frames",
    "steady_block_frames", "left_context_frames", "text_lookahead",
    "phrase_flush_ms", "temperature", "top_k", "top_p",
    "repetition_penalty", "seed",
)
OUTCOME_FIELDS = (
    "candidate_request_id", "generated_codec_frames", "generation_cap_frames",
    "engine_prompt_ms", "engine_prefill_ms", "engine_talker_ms",
    "engine_decode_ms", "engine_eos",
)
_SAFE_FAILURE_REASONS = {
    "stale before phrase dispatch": "stale_before_dispatch",
    "an earlier phrase in this response failed": "prior_phrase_failed",
    "response snapshot setup failed": "snapshot_setup_failed",
    "cancelled while waiting for phrase flush": "cancelled_during_flush",
    "stale before playback queue commit": "stale_before_queue_commit",
    "barge-in/stop/replacement": "cancelled",
    "provider returned no audio": "empty_audio",
    "wall-timeout": "wall_timeout",
    "native-stream-error": "native_stream_error",
    "native-preheader-error": "native_preheader_error",
    "engine-error": "engine_error",
    "synthesis-error": "synthesis_error",
    "client-cancelled": "client_cancelled",
    "transport-closed": "transport_closed",
    "incomplete-synthesis": "incomplete_synthesis",
}
_SAFE_OUTCOME_STATES = {"completed", "error", "cancelled", "missing"}


def _resident_profile(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read only the currently resident candidate and selected immutable profile."""
    base = args.candidate.rstrip("/")
    with httpx.Client(timeout=5, trust_env=False) as client:
        health = client.get(f"{base}/health").raise_for_status().json()
        profiles = client.get(f"{base}/v1/tuning/profiles").raise_for_status().json()
    model = health.get("activeModel") or health.get("current")
    instance = health.get("supervisorInstanceId")
    if (
        health.get("state") != "loaded"
        or health.get("singleResident") is not True
        or not isinstance(model, str)
        or not model.endswith("base-bf16")
        or not isinstance(health.get("engineEpoch"), int)
        or health["engineEpoch"] < 0
        or not isinstance(instance, str)
        or not instance.strip()
    ):
        raise RuntimeError("Acceptance requires one already-resident Base candidate with lifecycle identity")
    selected = (profiles.get("profiles") or {}).get(args.profile)
    if not isinstance(selected, dict) or selected.get("revision") != args.profile_revision:
        raise RuntimeError("Selected candidate profile/revision is not currently available")
    effective = {field: selected[field] for field in EFFECTIVE_FIELDS if field in selected}
    if len(effective) != len(EFFECTIVE_FIELDS):
        raise RuntimeError("Selected candidate profile has no complete effective tuning snapshot")
    return health, effective


def _runtime_config(args: argparse.Namespace, effective: dict[str, Any]) -> RuntimeConfig:
    tuning = {
        "provider": "qwen3tts-audiocpp",
        "scope": "realtime",
        "profile_id": args.profile,
        "profile_revision": args.profile_revision,
        "effective": deepcopy(effective),
        "overrides": {"seed": args.seed},
        "delivery_mode": args.delivery,
    }
    runtime = RuntimeConfig()
    runtime.local_pipeline.update({
        "tts_backend": "qwen3tts-audiocpp",
        "tts_tuning": tuning,
        "assistant_language": args.language,
    })
    # This mirrors realtime owner admission: snapshot these values once before
    # the handler constructs the request, rather than letting a later setting
    # update retarget the one diagnostic phrase.
    runtime.response_synthesis_configs[1] = {
        "local_pipeline": deepcopy(runtime.local_pipeline),
        "voice": args.voice,
    }
    return runtime


def _metric_rows(metrics: Queue) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    while True:
        try:
            event = metrics.get_nowait()
        except Empty:
            return rows
        detail = getattr(event, "detail", {})
        detail = detail if isinstance(detail, dict) else {}
        sanitized_detail = {key: detail[key] for key in OUTCOME_FIELDS + (
            "mode", "first_pcm_ms", "generation_ms", "audio_duration_ms",
            "rtf", "source_sample_rate", "profile_id", "profile_revision",
        ) if key in detail}
        raw_reason = detail.get("reason")
        if isinstance(raw_reason, str):
            sanitized_detail["reason"] = _SAFE_FAILURE_REASONS.get(raw_reason, "unspecified")
        raw_outcome_state = detail.get("outcome_state")
        if raw_outcome_state in _SAFE_OUTCOME_STATES:
            sanitized_detail["outcome_state"] = raw_outcome_state
        rows.append({
            "stage": getattr(event, "stage", None),
            "status": getattr(event, "status", None),
            "elapsed_ms": getattr(event, "elapsed_ms", None),
            "detail": sanitized_detail,
        })


def _validate_frozen_identity(snapshot: Any, health: dict[str, Any]) -> None:
    """Reject a candidate reload or retarget between health admission and request freeze."""
    expected = (
        str(health.get("activeModel")),
        str(health.get("engineEpoch")),
        str(health.get("supervisorInstanceId")),
    )
    frozen = (
        getattr(snapshot, "model", None),
        getattr(snapshot, "model_epoch", None),
        getattr(snapshot, "model_instance_id", None),
    )
    if frozen != expected:
        raise RuntimeError(
            "Candidate lifecycle changed between health admission and frozen TTS request"
        )


def _completion_is_proven(done: dict[str, Any] | None, outcome: Any) -> bool:
    """Require one correlated engine-complete request before reporting success."""
    if not isinstance(outcome, dict) or not isinstance(done, dict):
        return False
    request_id = outcome.get("candidate_request_id")
    detail = done.get("detail")
    return (
        isinstance(request_id, str)
        and re.fullmatch(r"[0-9a-f]{32}", request_id) is not None
        and outcome.get("engine_eos") is True
        and isinstance(detail, dict)
        and detail.get("candidate_request_id") == request_id
        and detail.get("engine_eos") is True
    )


def _safe_failure_code(error: Exception) -> str:
    """Classify known provider failures without exposing error text or request material."""
    message = str(error)
    if message.startswith("Acceptance requires one already-resident Base candidate"):
        return "candidate_unavailable"
    if message.startswith("Selected candidate profile/revision is not currently available"):
        return "profile_unavailable"
    if message.startswith("Selected candidate profile has no complete effective tuning snapshot"):
        return "profile_incomplete"
    if message.startswith("Candidate lifecycle changed"):
        return "lifecycle_changed"
    if message.startswith("audio.cpp synthesis failed: wall-timeout") or "'reason': 'wall_timeout'" in message:
        return "wall_timeout"
    if message.startswith("audio.cpp synthesis failed: engine-error") or "'reason': 'engine_error'" in message:
        return "engine_error"
    metric_reason_codes = {
        "stale_before_dispatch": "lifecycle_changed",
        "stale_before_queue_commit": "lifecycle_changed",
        "prior_phrase_failed": "completion_unverified",
        "snapshot_setup_failed": "completion_unverified",
        "cancelled_during_flush": "completion_unverified",
        "cancelled": "completion_unverified",
        "empty_audio": "completion_unverified",
        "native_stream_error": "completion_unverified",
        "native_preheader_error": "completion_unverified",
        "synthesis_error": "completion_unverified",
        "client_cancelled": "completion_unverified",
        "transport_closed": "completion_unverified",
        "incomplete_synthesis": "completion_unverified",
    }
    for reason, code in metric_reason_codes.items():
        if f"'reason': '{reason}'" in message:
            return code
    if message.startswith("HFRT TTS handler did not complete audio"):
        return "completion_unverified"
    return "unspecified"


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not isinstance(args.voice, str) or not args.voice.startswith("clone:") or not args.voice.removeprefix("clone:"):
        raise RuntimeError("Acceptance requires one explicit candidate clone:<id>")
    final_audio = None
    temporary_audio = None
    if args.audio_dir:
        args.audio_dir.mkdir(parents=True, exist_ok=True)
        final_audio = args.audio_dir / "tts-handler-acceptance.pcm"
        if final_audio.exists():
            raise RuntimeError(f"Refusing to overwrite existing PCM artifact: {final_audio}")
    try:
        health, effective = _resident_profile(args)
        runtime = _runtime_config(args, effective)
        metrics: Queue = Queue()
        # ``openai-api`` is deferred: production setup creates no local model and
        # per-request candidate admission below only reads health/clone/profile
        # state.  Construct through BaseHandler's normal queue/event contract.
        handler = Qwen3TTSHandler(
            Event(), Queue(), Queue(),
            setup_args=(Event(),),
            setup_kwargs={
                "backend": "openai-api",
                "api_base_url": f"{args.candidate.rstrip('/')}/v1",
                "api_model": str(health.get("activeModel")),
                "api_voice": args.voice,
                "text_output_queue": metrics,
            },
        )
        # Setup intentionally honors normal runtime environment defaults.  This
        # bounded diagnostic must not: it has no credential input and always sends
        # directly to the explicit candidate selected on its own command line.
        handler.api_base_url = f"{args.candidate.rstrip('/')}/v1"
        handler.audio_cpp_api_base_url = f"{args.candidate.rstrip('/')}/v1"
        handler.api_key = None
        if final_audio is not None:
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=".tts-handler-", suffix=".pcm", dir=args.audio_dir, delete=False,
            ) as handle:
                temporary_audio = Path(handle.name)
        item = TTSInput(
            text=args.text,
            language_code=args.language,
            runtime_config=runtime,
            turn_id="tts-handler-acceptance",
            turn_revision=1,
            input_epoch=1,
            response_epoch=1,
            response_id="tts-handler-acceptance",
        )
        snapshot = handler._response_synthesis_snapshot(item)
        _validate_frozen_identity(snapshot, health)
        chunks: list[bytes] = []
        samples = 0
        for output in handler.process(item):
            if not isinstance(output, AudioOutput):
                continue
            pcm = np.asarray(output.audio, dtype=np.int16)
            samples += int(pcm.size)
            if temporary_audio is not None:
                chunks.append(pcm.astype("<i2", copy=False).tobytes())
        rows = _metric_rows(metrics)
        done = next((row for row in reversed(rows) if row["stage"] == "tts" and row["status"] == "done"), None)
        failure = next((row for row in reversed(rows) if row["stage"] == "tts" and row["status"] in {
            "failed", "runaway_aborted", "empty_audio", "cancelled_before_audio", "cancelled_after_audio",
        }), None)
        outcome = getattr(handler, "_last_tts_outcome", {})
        completion_proven = _completion_is_proven(done, outcome)
        if failure or samples <= 0 or not completion_proven:
            diagnostic = failure or {"reason": "missing_completed_metric"}
            raise RuntimeError(f"HFRT TTS handler did not complete audio: {diagnostic}")
        if temporary_audio is not None and final_audio is not None:
            temporary_audio.write_bytes(b"".join(chunks))
            os.replace(temporary_audio, final_audio)
            temporary_audio = None
        return {
        "scope": "HFRT Qwen3TTSHandler TTSInput/snapshot/process path only; excludes WebSocket, LLM, microphone, and listening",
        "identity": {
            "admitted_model": health.get("activeModel"),
            "admitted_engine_epoch": health.get("engineEpoch"),
            "admitted_supervisor_instance_id": health.get("supervisorInstanceId"),
            "snapshot_model": snapshot.model,
            "snapshot_engine_epoch": snapshot.model_epoch,
            "snapshot_supervisor_instance_id": snapshot.model_instance_id,
        },
        "request": {
            "text_sha256": hashlib.sha256(args.text.encode("utf-8")).hexdigest(),
            "voice": args.voice,
            "clone_content_revision": snapshot.clone_content_revision,
            "clone_content_hash": snapshot.clone_content_hash,
            "profile_id": snapshot.profile_id,
            "profile_revision": snapshot.profile_revision,
            "seed": snapshot.seed,
            "language": snapshot.language,
            "delivery_mode": snapshot.delivery_streaming_mode,
        },
        "audio": {
            "samples": samples,
            "source_sample_rate": snapshot.delivery_sample_rate,
            "duration_ms": round(samples * 1000 / int(snapshot.delivery_sample_rate or 24000), 3),
            "exported": final_audio is not None,
        },
        "metrics": rows,
        "outcome": {
            "completion_proven": completion_proven,
            **{key: outcome.get(key) for key in OUTCOME_FIELDS if key in outcome},
        },
    }
    except Exception as error:
        raise RuntimeError(f"HFRT TTS handler failed: {_safe_failure_code(error)}") from None
    finally:
        if temporary_audio is not None:
            temporary_audio.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Explicitly send one TTS synthesis request")
    parser.add_argument("--candidate", default="http://127.0.0.1:8890")
    parser.add_argument("--voice", required=True, help="Explicit selected candidate clone:<id>")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--profile-revision", type=int, required=True)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--language", default="English")
    parser.add_argument("--delivery", choices=("buffered_phrase", "native_incremental_pcm"), default="buffered_phrase")
    parser.add_argument("--text", default="Count slowly from one to five. Then count backwards from five to one.")
    parser.add_argument("--audio-dir", type=Path, help="Optional explicit raw PCM export directory")
    args = parser.parse_args()
    if not args.execute:
        parser.error("Use --execute to send one explicit TTS handler diagnostic request")
    if not 0 <= args.seed <= 0xFFFFFFFF or args.profile_revision < 1:
        parser.error("seed must be uint32 and profile revision must be positive")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
