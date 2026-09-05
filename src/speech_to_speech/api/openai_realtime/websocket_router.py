import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from queue import Empty, Queue
from threading import Event as ThreadingEvent
from typing import Any, Callable, TypeVar

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from openai.types.realtime import (
    ConversationItemCreateEvent,
    InputAudioBufferAppendEvent,
    InputAudioBufferCommitEvent,
    ResponseAudioDeltaEvent,
    ResponseCancelEvent,
    ResponseCreateEvent,
    SessionUpdateEvent,
)
from starlette.websockets import WebSocketState

from speech_to_speech.api.openai_realtime.pipeline_unit import PipelineUnit, SessionState
from speech_to_speech.api.openai_realtime.service import PipelineRuntimeServerEvent, ServerEvent, build_error_event
from speech_to_speech.LLM.direct_history_compaction import normalize_history_compaction
from speech_to_speech.pipeline.control import SESSION_END, PipelineControlMessage, is_control_message
from speech_to_speech.pipeline.events import (
    AssistantTextEvent,
    PartialTranscriptionEvent,
    PipelineEvent,
    PipelineMetricEvent,
    ResponseFailedEvent,
    ResponseOutputCompleteEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    TokenUsageEvent,
    TranscriptionCompletedEvent,
)
from speech_to_speech.pipeline.log_context import pipeline_log_ctx
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, PIPELINE_END, AudioOutput
from speech_to_speech.pipeline.model_operations import ModelOperationToken
from speech_to_speech.pipeline.response_ownership import ResponseOwner

logger = logging.getLogger(__name__)
MAX_AUDIO_BATCH_BYTES = 6400
# How long the release path waits for SESSION_END to propagate through the
# handler chain back to output_queue before clearing unit.session. Tests
# monkeypatch this to a small value since their fixtures usually skip the
# real handler chain.
SESSION_END_DRAIN_TIMEOUT_S = 10.0
BACKEND_RUNTIME_API_VERSION = 7
MODEL_CANCEL_TIMEOUT_S = 2.0
QItem = TypeVar("QItem")
_AUDIO_CPP_TUNING_BOUNDS: dict[str, tuple[float, float]] = {
    "max_reference_seconds": (1, 30),
    "first_block_frames": (1, 300),
    "steady_block_frames": (1, 300),
    "left_context_frames": (1, 300),
    "text_lookahead": (16, 512),
    "phrase_flush_ms": (50, 3000),
    "temperature": (0, 2),
    "top_k": (1, 200),
    "top_p": (0.05, 1),
    "repetition_penalty": (0.8, 2),
    "seed": (0, 2**32 - 1),
}
_AUDIO_CPP_INTEGER_TUNING_FIELDS = {
    "max_reference_seconds",
    "first_block_frames",
    "steady_block_frames",
    "left_context_frames",
    "text_lookahead",
    "phrase_flush_ms",
    "top_k",
    "seed",
}
_AUDIO_CPP_MODELS = {
    "qwen3-tts-0.6b-base-bf16",
    "qwen3-tts-1.7b-base-bf16",
}
_AUDIO_CPP_EFFECTIVE_TUNING_KEYS = frozenset({
    "model",
    "clone_mode",
    *_AUDIO_CPP_TUNING_BOUNDS,
})


def _validate_audio_cpp_tuning_values(
    values: Any,
    *,
    require_complete: bool,
    allow_null_model_seed: bool,
) -> dict[str, Any]:
    """Validate only engine-safe candidate fields at the HFRT trust boundary."""

    if not isinstance(values, dict):
        raise ValueError("tts_tuning values must be an object")
    keys = set(values)
    if keys - _AUDIO_CPP_EFFECTIVE_TUNING_KEYS:
        raise ValueError("tts_tuning contains unsupported effective fields")
    if require_complete and keys != _AUDIO_CPP_EFFECTIVE_TUNING_KEYS:
        raise ValueError("tts_tuning effective snapshot is incomplete")

    validated: dict[str, Any] = {}
    if "clone_mode" in values:
        if values["clone_mode"] != "full_icl":
            raise ValueError("tts_tuning supports Base full_icl cloning only")
        validated["clone_mode"] = "full_icl"
    if "model" in values:
        model = values["model"]
        if model is None and allow_null_model_seed:
            validated["model"] = None
        elif model in _AUDIO_CPP_MODELS:
            validated["model"] = model
        else:
            raise ValueError("tts_tuning model is invalid")

    for key in _AUDIO_CPP_TUNING_BOUNDS:
        if key not in values:
            continue
        value = values[key]
        if key == "seed" and value is None and allow_null_model_seed:
            validated[key] = None
            continue
        bounds = _AUDIO_CPP_TUNING_BOUNDS[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or (key in _AUDIO_CPP_INTEGER_TUNING_FIELDS and not isinstance(value, int))
            or not bounds[0] <= value <= bounds[1]
        ):
            raise ValueError("tts_tuning effective value is out of bounds")
        validated[key] = value
    if "temperature" in validated and validated["temperature"] <= 0:
        raise ValueError("tts_tuning temperature must be greater than zero")
    context = validated.get("left_context_frames")
    steady = validated.get("steady_block_frames")
    if context is not None and steady is not None and context + steady > 300:
        raise ValueError("tts_tuning left context plus steady block must not exceed 300 frames")
    return validated


def _validated_audio_cpp_tuning(tuning: Any) -> dict[str, Any]:
    """Validate the browser's candidate-only session tuning snapshot.

    The candidate supervisor remains authoritative for the named profile and
    engine fields.  ``resolved`` carries only the two supervisor-resolved
    phrase-queue values needed locally by HF Realtime; it is bounded again at
    this trust boundary before it can affect scheduling.
    """
    if not isinstance(tuning, dict):
        raise ValueError("tts_tuning must be an object")
    tuning_keys = set(tuning)
    modern_required_keys = {"provider", "profile_id", "profile_revision", "effective", "overrides"}
    modern_keys = {
        "provider",
        "profile_id",
        "profile_revision",
        "effective",
        "overrides",
        "delivery_mode",
    }
    if tuning_keys - (modern_keys | {"resolved"}):
        raise ValueError("tts_tuning contains unsupported fields")
    if not modern_required_keys.issubset(tuning_keys):
        raise ValueError(
            "tts_tuning requires provider, profile_id, profile_revision, effective, and overrides"
        )
    if tuning.get("provider") != "qwen3tts-audiocpp":
        raise ValueError("tts_tuning is available only for qwen3tts-audiocpp")
    profile_id = tuning.get("profile_id")
    overrides = tuning.get("overrides", {})
    if not isinstance(profile_id, str) or not profile_id or not isinstance(overrides, dict):
        raise ValueError("tts_tuning requires profile_id and object overrides")

    validated_overrides = _validate_audio_cpp_tuning_values(
        overrides,
        require_complete=False,
        allow_null_model_seed=True,
    )

    validated: dict[str, Any] = {
        "provider": "qwen3tts-audiocpp",
        "profile_id": profile_id,
        "overrides": validated_overrides,
    }
    # Engine capability is not user consent to use the experimental native
    # relay.  Buffered phrase delivery is the compatibility/default path until
    # the caller explicitly opts into native PCM for this conversation.
    delivery_mode = tuning.get("delivery_mode", "buffered_phrase")
    if delivery_mode not in {"buffered_phrase", "native_incremental_pcm"}:
        raise ValueError("tts_tuning delivery_mode is invalid")
    validated["delivery_mode"] = delivery_mode
    revision = tuning.get("profile_revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision <= 0:
        raise ValueError("tts_tuning profile_revision must be a positive integer")
    validated["profile_revision"] = revision
    validated["effective"] = _validate_audio_cpp_tuning_values(
        tuning.get("effective"),
        require_complete=True,
        allow_null_model_seed=True,
    )

    resolved = tuning.get("resolved")
    if resolved is not None:
        if not isinstance(resolved, dict) or set(resolved) != {"text_lookahead", "phrase_flush_ms"}:
            raise ValueError("tts_tuning resolved phrase queue is invalid")
        validated_resolved: dict[str, int] = {}
        for key in ("text_lookahead", "phrase_flush_ms"):
            value = resolved.get(key)
            minimum, maximum = _AUDIO_CPP_TUNING_BOUNDS[key]
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                raise ValueError("tts_tuning resolved phrase queue is out of bounds")
            validated_resolved[key] = value
        validated["resolved"] = validated_resolved
    return validated


class _PipelineConfigError(ValueError):
    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


def _validated_playback_policy(value: Any, *, target_backend: str) -> dict[str, Any]:
    """Validate the local browser policy committed for the next response.

    PCM transport remains server-owned.  This object carries only bounded
    playback admission choices so the server can echo the exact frozen policy
    with the response lifecycle even if a later config acknowledgement
    overtakes that lifecycle event on the socket.
    """
    if not isinstance(value, dict) or set(value) != {
        "prime_target_ms",
        "continuity_mode",
        "native_streaming",
        "max_prime_ms",
    }:
        raise _PipelineConfigError("playback_policy has invalid fields", "invalid_playback_policy")
    continuity = value.get("continuity_mode")
    if continuity not in {"adaptive", "fast-start"}:
        raise _PipelineConfigError("playback_policy continuity mode is invalid", "invalid_playback_policy")
    native_streaming = value.get("native_streaming")
    if not isinstance(native_streaming, bool):
        raise _PipelineConfigError("playback_policy native_streaming must be boolean", "invalid_playback_policy")
    prime = value.get("prime_target_ms")
    maximum = value.get("max_prime_ms")
    if (
        isinstance(prime, bool)
        or not isinstance(prime, int)
        or isinstance(maximum, bool)
        or not isinstance(maximum, int)
        or not 0 <= prime <= maximum <= 2000
    ):
        raise _PipelineConfigError("playback_policy prime values are out of bounds", "invalid_playback_policy")
    if target_backend != "qwen3tts-audiocpp":
        native_streaming = False
        prime = 0
    return {
        "prime_target_ms": prime,
        "continuity_mode": continuity,
        "native_streaming": native_streaming,
        "max_prime_ms": maximum,
    }


async def _prepare_pipeline_config_update(
    service: Any,
    runtime_config: Any,
    config: dict[str, Any],
) -> tuple[dict[str, Any], Any]:
    """Validate a pipeline update without mutating the live session.

    The caller commits both returned objects together only after every field,
    candidate tuning value, and model endpoint has passed validation.
    """
    proposed_pipeline = dict(runtime_config.local_pipeline)
    proposed_endpoint = runtime_config.model_endpoint

    for key in ("full_buffer_tts", "live_transcription"):
        if key in config:
            value = config[key]
            if not isinstance(value, bool):
                raise _PipelineConfigError(f"{key} must be a boolean", "invalid_pipeline_config")
            proposed_pipeline[key] = value

    if "max_response_tokens" in config:
        try:
            proposed_pipeline["max_response_tokens"] = min(
                1024, max(64, int(config["max_response_tokens"]))
            )
        except (TypeError, ValueError) as exc:
            raise _PipelineConfigError(
                "Response limit must be between 64 and 1024 tokens",
                "invalid_max_response_tokens",
            ) from exc

    if "history_compaction" in config:
        value = config["history_compaction"]
        if not isinstance(value, dict):
            raise _PipelineConfigError("history_compaction must be an object", "invalid_history_compaction")
        try:
            proposed_pipeline["history_compaction"] = normalize_history_compaction(value)
        except (TypeError, ValueError) as exc:
            raise _PipelineConfigError("Invalid history_compaction values", "invalid_history_compaction") from exc

    previous_backend = proposed_pipeline.get("tts_backend", "faster")
    target_backend = previous_backend
    if "tts_backend" in config:
        target_backend = config["tts_backend"]
        if target_backend == "audio-cpp":
            target_backend = "qwen3tts-audiocpp"
        if target_backend not in {"faster", "groxaxo", "qwen3tts-audiocpp"}:
            raise _PipelineConfigError("Unknown TTS backend", "invalid_tts_backend")
        proposed_pipeline["tts_backend"] = target_backend

    if "tts_tuning" in config:
        if target_backend != "qwen3tts-audiocpp":
            raise _PipelineConfigError(
                "tts_tuning is available only for qwen3tts-audiocpp",
                "invalid_tts_tuning",
            )
        try:
            proposed_pipeline["tts_tuning"] = _validated_audio_cpp_tuning(config["tts_tuning"])
        except ValueError as exc:
            raise _PipelineConfigError(str(exc), "invalid_tts_tuning") from exc
    elif target_backend != "qwen3tts-audiocpp" or previous_backend != "qwen3tts-audiocpp":
        # A provider switch never inherits a different provider's synthesis
        # snapshot.  The candidate client sends its resolved profile explicitly.
        proposed_pipeline.pop("tts_tuning", None)

    # Source clock is server-owned and derives solely from the committed
    # provider. Browser microphone/VAD input remains the fixed 16 kHz path.
    proposed_pipeline["audio_output_sample_rate"] = (
        24000 if target_backend == "qwen3tts-audiocpp" else 16000
    )

    if "playback_policy" in config:
        proposed_pipeline["playback_policy"] = _validated_playback_policy(
            config["playback_policy"],
            target_backend=target_backend,
        )
    elif target_backend != previous_backend:
        proposed_pipeline["playback_policy"] = {
            "prime_target_ms": 0,
            "continuity_mode": "adaptive",
            "native_streaming": False,
            "max_prime_ms": 2000,
        }

    if "model_endpoint" in config:
        model_config = config["model_endpoint"]
        if not isinstance(model_config, dict):
            raise _PipelineConfigError("Model endpoint must be an object", "invalid_model_provider")
        provider = model_config.get("provider", "local")
        if provider not in {"local", "remote"}:
            raise _PipelineConfigError(
                "Model provider must be local or remote",
                "invalid_model_provider",
            )
        try:
            proposed_endpoint = await asyncio.to_thread(
                service.validate_model_endpoint,
                provider=provider,
                base_url=model_config.get("base_url"),
                model=model_config.get("model"),
                api_key=model_config.get("api_key") or None,
            )
        except Exception as exc:
            logger.warning(
                "Model endpoint validation failed provider=%s error=%s",
                provider,
                type(exc).__name__,
            )
            raise _PipelineConfigError(
                f"Selected model endpoint is unavailable: {exc}",
                "model_endpoint_unavailable",
            ) from exc

    return proposed_pipeline, proposed_endpoint


async def _send_event(ws: WebSocket, event: ServerEvent) -> bool:
    # Skip cleanly when the ws is already closing/closed — happens during Ctrl-C
    # shutdown, where the lifespan starts closing sockets while the route handler
    # or send loop is still in flight pushing events.
    if ws.application_state != WebSocketState.CONNECTED:
        return False
    try:
        await ws.send_json(event.model_dump())
        return True
    except WebSocketDisconnect:
        logger.debug("Skipped event: ws disconnected mid-send")
    except RuntimeError as e:
        # Race: ws closed between the state check above and the send. Starlette
        # raises a plain RuntimeError("Unexpected ASGI message 'websocket.send'
        # after sending 'websocket.close' ...") — harmless during shutdown.
        msg = str(e)
        if "websocket.close" in msg or "websocket.disconnect" in msg or "response already completed" in msg:
            logger.debug(f"Skipped event: ws already closed ({msg})")
        else:
            logger.error(f"Failed to send event to client: {e}")
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to send event to client: {e}")
    return False


async def _send_events(ws: WebSocket, events: list[ServerEvent]) -> None:
    for event in events:
        await _send_event(ws, event)


def _keep_audio_sentinel(item: Any) -> bool:
    return _is_audio_done(item)


def _keep_user_text_event(item: Any) -> bool:
    return isinstance(
        item,
        (
            SpeechStartedEvent,
            SpeechStoppedEvent,
            PartialTranscriptionEvent,
            TranscriptionCompletedEvent,
            TokenUsageEvent,
            PipelineMetricEvent,
        ),
    )


def _audio_payload(item: Any) -> Any:
    return item.audio if isinstance(item, AudioOutput) else item


def _audio_generation(item: Any) -> int | None:
    return item.cancel_generation if isinstance(item, AudioOutput) else None


def _audio_response_epoch(item: Any) -> int | None:
    return item.response_epoch if isinstance(item, AudioOutput) else None


def _audio_input_epoch(item: Any) -> int | None:
    return item.input_epoch if isinstance(item, AudioOutput) else None


def _audio_response_id(item: Any) -> str | None:
    return item.response_id if isinstance(item, AudioOutput) else None


def _audio_sample_rate(item: Any) -> int:
    """Return one trusted source clock; legacy queue entries stay at 16 kHz."""

    rate = item.source_sample_rate if isinstance(item, AudioOutput) else 16000
    return rate if isinstance(rate, int) and not isinstance(rate, bool) and rate > 0 else 16000


def _audio_identity(item: Any) -> tuple[int | None, int | None, str | None, int]:
    """The immutable response identity and source clock for one PCM batch."""

    return (
        _audio_input_epoch(item),
        _audio_response_epoch(item),
        _audio_response_id(item),
        _audio_sample_rate(item),
    )


def _flush_queue(q: Queue[QItem], *, preserve: Callable[[QItem], bool] | None = None) -> None:
    """Drain a queue, optionally preserving items matching *preserve*.

    Preserved items are re-inserted at the **front** of the queue
    (atomically under the queue's mutex) so they are processed before
    anything a pipeline thread may have enqueued during the drain.
    """
    preserved: list[QItem] = []
    while True:
        try:
            item = q.get_nowait()
            if preserve and preserve(item):
                preserved.append(item)
        except Empty:
            break
    if preserved:
        with q.mutex:
            for item in reversed(preserved):
                q.queue.appendleft(item)
            q.not_empty.notify(len(preserved))


def _belongs_to_cancelled_response(
    item: Any,
    *,
    owner: ResponseOwner | None,
    cancel_generation: int | None,
) -> bool:
    """Return whether one queued output belongs to the response being cancelled.

    Cancellation may wait up to two seconds for a provider transport. A newer
    response can be admitted during that wait, so a blanket queue drain would
    delete the successor. Modern pipeline output carries an immutable response
    epoch; the cancellation generation is retained only for legacy output.
    """

    response_epoch = getattr(item, "response_epoch", None)
    if owner is not None and response_epoch is not None:
        return response_epoch == owner.response_epoch
    generation = getattr(item, "cancel_generation", None)
    return (
        cancel_generation is not None
        and generation is not None
        and generation == cancel_generation
    )


def _flush_cancelled_response_output(
    q: Queue[QItem],
    *,
    owner: ResponseOwner | None,
    cancel_generation: int | None,
) -> None:
    """Remove only the cancelled owner's queued output, preserving successors."""

    _flush_queue(
        q,
        preserve=lambda item: not _belongs_to_cancelled_response(
            item,
            owner=owner,
            cancel_generation=cancel_generation,
        ),
    )


async def _drain_pending_response_events(
    ws: WebSocket | None,
    unit: PipelineUnit,
    session_id: str | None,
) -> None:
    if session_id is None:
        return

    preserved: list[Any] = []
    drained_assistant = 0
    drained_usage = 0
    response_boundary_seen = False
    try:
        while True:
            try:
                item = unit.text_output_queue.get_nowait()
            except Empty:
                break
            if isinstance(item, SpeechStartedEvent):
                # The next turn has started. Retain its response-bearing events
                # for the normal send loop while still allowing token accounting
                # already queued behind this marker to settle the prior turn.
                response_boundary_seen = True
                preserved.append(item)
                continue
            if response_boundary_seen and isinstance(
                item,
                (
                    AssistantTextEvent,
                    PartialTranscriptionEvent,
                    TranscriptionCompletedEvent,
                    ResponseFailedEvent,
                    ResponseOutputCompleteEvent,
                ),
            ):
                preserved.append(item)
                continue
            if isinstance(
                item,
                (
                    TokenUsageEvent,
                    AssistantTextEvent,
                    PartialTranscriptionEvent,
                    TranscriptionCompletedEvent,
                    PipelineMetricEvent,
                    ResponseFailedEvent,
                    ResponseOutputCompleteEvent,
                ),
            ):
                if isinstance(item, TokenUsageEvent):
                    drained_usage += 1
                elif isinstance(item, AssistantTextEvent):
                    drained_assistant += 1
                if _generation_is_discardable(unit, getattr(item, "cancel_generation", None)):
                    continue
                events = unit.service.dispatch_pipeline_event(session_id, item)
                if ws is not None and events:
                    await _send_events(ws, events)
            else:
                preserved.append(item)
    finally:
        if preserved:
            with unit.text_output_queue.mutex:
                for item in reversed(preserved):
                    unit.text_output_queue.queue.appendleft(item)
                unit.text_output_queue.not_empty.notify(len(preserved))

    if drained_assistant or drained_usage:
        logger.debug(
            "Pipeline %d: drained %d assistant event(s) and %d token usage event(s) before response completion",
            unit.index,
            drained_assistant,
            drained_usage,
        )


def _clean_unit(unit: PipelineUnit, preserve: Callable[[Any], bool] | None = None) -> None:
    """Cancel in-flight work and flush queues for a single pipeline unit.

    Every queue in the handler chain is drained so pending speculative turns
    cannot sit ahead of SESSION_END and keep the sole pipeline slot occupied.
    SESSION_END is enqueued by the route handler *after* this returns to serve
    as the soft reset signal for stateful handlers.
    """
    unit.cancel_scope.cancel()
    unit.model_operations.cancel_and_wait("session_end", MODEL_CANCEL_TIMEOUT_S)
    for handler in unit.handlers:
        if getattr(handler, "model_operations", None) is unit.model_operations:
            continue
        cancel_active = getattr(handler, "cancel_active", None)
        if callable(cancel_active):
            try:
                cancel_active()
            except Exception:
                logger.debug("Handler active-stream cancellation failed", exc_info=True)
    queues: list[Queue[Any]] = [
        unit.input_queue,
        unit.text_prompt_queue,
        unit.output_queue,
        unit.text_output_queue,
    ]
    for handler in unit.handlers:
        for attr in ("queue_in", "queue_out"):
            queue = getattr(handler, attr, None)
            if isinstance(queue, Queue):
                queues.append(queue)

    seen: set[int] = set()
    edge_queue_ids = {id(unit.output_queue), id(unit.text_output_queue)}
    for queue in queues:
        queue_id = id(queue)
        if queue_id in seen:
            continue
        seen.add(queue_id)
        _flush_queue(queue, preserve=preserve if queue_id in edge_queue_ids else None)
    unit.response_playing.clear()
    unit.model_operations.reset()
    unit.cancel_scope.reset()
    unit.should_listen.set()


def _cancel_handler_transports(unit: PipelineUnit, reason: str) -> None:
    """Close handler-local transports before cancellation yields to a successor."""

    for handler in unit.handlers:
        if getattr(handler, "model_operations", None) is unit.model_operations:
            continue
        cancel_active = getattr(handler, "cancel_active", None)
        if callable(cancel_active):
            try:
                cancel_active()
            except Exception:
                logger.debug("Handler cancellation failed during %s", reason, exc_info=True)


def _operation_matches_response(
    operation: ModelOperationToken | None,
    owner: ResponseOwner | None,
    cancel_generation: int | None,
) -> bool:
    """Bind a coordinator token to the response being superseded."""

    if operation is None:
        return False
    if cancel_generation is not None and operation.cancel_generation != cancel_generation:
        return False
    if owner is None:
        return True
    if owner.turn_id is not None and operation.turn_id != owner.turn_id:
        return False
    if owner.turn_revision is not None and operation.turn_revision != owner.turn_revision:
        return False
    return True


def _capture_superseded_response(unit: PipelineUnit, supersession: Any) -> dict[str, Any]:
    """Invalidate transport generation at the synchronous VAD boundary."""

    cancelled_generation = unit.cancel_scope.generation
    operation = unit.model_operations.active_token()
    if not _operation_matches_response(operation, supersession.previous, cancelled_generation):
        operation = None
    elif operation is not None:
        unit.model_operations.request_cancel_token(operation, supersession.reason)
    # This must happen before VAD can claim/start the accepted successor.
    unit.cancel_scope.cancel()
    _cancel_handler_transports(unit, supersession.reason)
    return {
        "operation": operation,
        "cancel_generation": cancelled_generation,
    }


async def _cancel_active_generation(
    unit: PipelineUnit,
    reason: str,
    *,
    response_owner: ResponseOwner | None = None,
    model_operation: ModelOperationToken | None = None,
    capture_current_operation: bool = True,
    cancel_handler_transports: bool = True,
) -> None:
    """Cancel one response without ending or poisoning its conversation."""
    if response_owner is None and unit.session is not None and unit.session.session_id:
        response_owner = unit.service._state(unit.session.session_id).response_ownership.active()
    if model_operation is None and capture_current_operation:
        # Capture synchronously before the first await. A successor may acquire
        # as soon as handler transports release the old operation; the later
        # wait must remain bound to this exact token.
        model_operation = unit.model_operations.request_cancel(reason)
    elif model_operation is not None:
        unit.model_operations.request_cancel_token(model_operation, reason)
    # Close handler-local transports before the first await. Once cancellation
    # yields, VAD or a manual response.create may install a successor and these
    # identity-unbound compatibility methods would otherwise close that new
    # response instead of the captured one.
    if cancel_handler_transports:
        _cancel_handler_transports(unit, reason)
    result = await asyncio.to_thread(
        unit.model_operations.wait_for_cancellation,
        model_operation,
        reason,
        MODEL_CANCEL_TIMEOUT_S,
    )
    operation = result.operation
    unit.text_output_queue.put(
        PipelineMetricEvent(
            stage="gemma",
            status="detached" if result.detached else "cancelled",
            at_s=time.time(),
            elapsed_ms=result.elapsed_ms,
            turn_id=operation.turn_id if operation else None,
            turn_revision=operation.turn_revision if operation else None,
            input_epoch=response_owner.input_epoch if response_owner else None,
            response_epoch=response_owner.response_epoch if response_owner else None,
            response_id=response_owner.response_id if response_owner else None,
            # This event is synthesized by the router from its own completed
            # cancellation operation.  It is safe to emit even for legacy SDK
            # streams that began before response-epoch ownership existed.
            authoritative_terminal=True,
            detail={
                "reason": reason,
                "operation": operation.kind if operation else None,
                "operation_id": operation.operation_id if operation else None,
                "released": result.released,
            },
        )
    )


def _to_audio_bytes(chunk: Any) -> bytes:
    chunk = _audio_payload(chunk)
    if isinstance(chunk, PipelineControlMessage):
        raise TypeError(f"unexpected control message on audio output queue: {chunk!r}")
    if isinstance(chunk, np.ndarray) or hasattr(chunk, "tobytes"):
        return chunk.tobytes()
    return chunk


def _is_audio_done(item: Any) -> bool:
    payload = _audio_payload(item)
    return isinstance(payload, bytes) and payload == AUDIO_RESPONSE_DONE


def _is_pipeline_end(item: Any) -> bool:
    payload = _audio_payload(item)
    return isinstance(payload, bytes) and payload == PIPELINE_END


def _generation_is_discardable(unit: PipelineUnit, generation: int | None) -> bool:
    """Whether output tagged with *generation* should be dropped.

    A generation is discardable if it has been superseded (``is_stale``) or if the
    cancel scope is in its post-cancel discard window and this is not the current
    live generation. Shared by audio and assistant-text so the two paths stay in
    lockstep: dropping text whenever ``discarding`` is set (without this generation
    check) silently swallows the transcript of a fresh response when ``discarding``
    lingers — e.g. a superseded speculative turn whose TTS never emitted an
    AUDIO_RESPONSE_DONE sentinel, so response_done() never cleared the flag.
    """
    if generation is not None and unit.cancel_scope.is_stale(generation):
        return True
    if unit.cancel_scope.discarding and generation != unit.cancel_scope.generation:
        return True
    return False


def _should_discard_audio(
    unit: PipelineUnit,
    item: Any,
    *,
    service: Any | None = None,
    conn_id: str | None = None,
) -> bool:
    """Reject stale PCM by both cancellation generation and response epoch."""
    if _generation_is_discardable(unit, _audio_generation(item)):
        return True
    response_epoch = _audio_response_epoch(item)
    if response_epoch is not None and service is not None and conn_id is not None:
        return not service._state(conn_id).response_ownership.admits_output(response_epoch)
    return False


def _audio_identity_is_admissible(
    service: Any,
    conn_id: str | None,
    identity: tuple[int | None, int | None, str | None, int],
) -> bool:
    """Recheck a dequeued PCM batch against its captured ownership tuple."""

    input_epoch, response_epoch, response_id, _source_rate = identity
    if response_epoch is None:
        return True
    if conn_id is None or input_epoch is None:
        return False
    owner = service._state(conn_id).response_ownership.owner_for_response_epoch(response_epoch)
    return bool(
        owner is not None
        and owner.input_epoch == input_epoch
        # A direct-audio owner can legitimately have no response ID until the
        # encoder allocates its first response.created event.  Once a concrete
        # ID is present, it must be the one captured with this batch.
        and (response_id is None or owner.response_id == response_id)
        and service.is_response_epoch_output_admissible(conn_id, response_epoch)
    )


async def _release_unit_after_drain(unit: PipelineUnit, session: Any, session_id: str) -> None:
    """Wait indefinitely for SESSION_END to propagate, then release the unit.

    Runs in its own asyncio task so the route handler's finally block can return
    immediately. The unit stays unavailable for new claims (unit.session != None)
    until SESSION_END travels all the way through the handler chain back to
    output_queue — observed by the send loop, which sets session.drained.

    Intentionally has no timeout-fallback release. If a handler (e.g. an LM HTTP
    call) is still busy past SESSION_END_DRAIN_TIMEOUT_S, releasing the unit
    would let a new client claim it while stale output from the previous session
    is still in flight — that output would be dispatched under the new session.
    We accept reduced pool capacity over a cross-session leak; operators can see
    stuck units in `/v1/pool` (long `released_at` age).
    """
    elapsed = 0.0
    warned = False
    while not session.drained.is_set():
        await asyncio.sleep(0.05)
        elapsed += 0.05
        if not warned and elapsed >= SESSION_END_DRAIN_TIMEOUT_S:
            logger.warning(
                f"Pipeline {unit.index}: SESSION_END not drained after {elapsed:.1f}s — "
                f"unit will remain unavailable until handlers finish (session {session_id})"
            )
            warned = True
    unit.service.unregister(session_id)
    unit.session = None
    logger.info(f"Pipeline {unit.index} released (session {session_id} ended)")


def create_app(
    pool: list[PipelineUnit],
    stop_event: ThreadingEvent,
    runtime_info: dict[str, Any] | None = None,
) -> FastAPI:
    emit_runtime_event = runtime_info is not None
    started_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    runtime = {
        "api_version": BACKEND_RUNTIME_API_VERSION,
        "started_at_utc": started_at_utc,
        "pid": os.getpid(),
        "mode": "realtime",
        "diagnostic_stages": [
            "mic",
            "echo_guard",
            "vad",
            "transcription",
            "gemma",
            "context",
            "tool",
            "tts",
            "playback",
        ],
        **(runtime_info or {}),
    }
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # One send loop per pipeline unit; each polls its own queues and forwards
        # to the websocket currently attached via unit.session.
        send_tasks = [asyncio.create_task(_send_loop_for(unit)) for unit in pool]
        yield
        for task in send_tasks:
            task.cancel()
        for task in send_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        for unit in pool:
            sess = unit.session
            if sess is not None:
                try:
                    await sess.websocket.close()
                except Exception:
                    pass

    app = FastAPI(lifespan=lifespan)

    def _claim_unit(ws: WebSocket) -> PipelineUnit | None:
        """Atomically (between asyncio yield points) reserve the first idle unit.

        Creates a placeholder SessionState that the caller fills in with the
        session_id after RealtimeService.register().
        """
        for unit in pool:
            if unit.session is None:
                unit.session = SessionState(websocket=ws)
                return unit
        return None

    @app.websocket("/v1/realtime")
    async def realtime_endpoint(ws: WebSocket) -> None:
        await ws.accept()

        unit = _claim_unit(ws)
        if unit is None:
            logger.warning(f"Rejected connection: all {len(pool)} pipeline slots in use")
            # Stateless error event — rejection is not chargeable to any unit's usage metrics.
            await _send_event(
                ws,
                build_error_event(
                    f"All {len(pool)} session slots are in use. Disconnect an existing client first.",
                    error_type="session_limit_reached",
                ),
            )
            await ws.close(code=1008, reason="All session slots are in use")
            return

        pipeline_log_ctx.set(unit.index)
        session_id = unit.service.register()
        # _claim_unit guarantees unit.session is not None for the returned unit.
        assert unit.session is not None
        unit.session.session_id = session_id
        logger.info(f"Client connected to pipeline {unit.index} (session {session_id})")

        # Defensive: drain edge queues and reset events so stale data from a
        # previous session that survived SESSION_END propagation doesn't leak.
        _clean_unit(unit)
        unit.service._state(session_id).runtime_config.local_pipeline[
            "_capture_superseded_response"
        ] = lambda supersession, bound_unit=unit: _capture_superseded_response(
            bound_unit,
            supersession,
        )

        try:
            await _send_event(ws, unit.service.build_session_created(session_id))
            if emit_runtime_event:
                await _send_event(
                    ws,
                    PipelineRuntimeServerEvent(
                        event_id=f"event_runtime_{session_id}",
                        runtime=runtime,
                    ),
                )

            while not stop_event.is_set():
                try:
                    raw = await asyncio.wait_for(ws.receive_json(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                if raw.get("type") in {"local.pipeline.update", "pipeline.config.update"}:
                    config = raw.get("config") if isinstance(raw.get("config"), dict) else {}
                    rt_cfg = unit.service._state(session_id).runtime_config
                    try:
                        proposed_pipeline, proposed_endpoint = await _prepare_pipeline_config_update(
                            unit.service,
                            rt_cfg,
                            config,
                        )
                    except _PipelineConfigError as exc:
                        await _send_event(ws, unit.service.make_error(str(exc), exc.code))
                        continue
                    # Commit only after the entire update has validated.  Invalid
                    # tuning or endpoint data cannot partially change providers.
                    with rt_cfg.history_maintenance_lock:
                        rt_cfg.local_pipeline = proposed_pipeline
                        rt_cfg.model_endpoint = proposed_endpoint
                    await ws.send_json(
                        {
                            "type": (
                                "pipeline.config.updated"
                                if raw.get("type") == "pipeline.config.update"
                                else "local.pipeline.updated"
                            ),
                            "config": {
                                **{k: v for k, v in rt_cfg.local_pipeline.items() if not k.startswith("_")},
                                "model_endpoint": rt_cfg.model_endpoint.redacted(),
                            },
                        }
                    )
                    continue

                # Local capability declaration. It has no OpenAI Realtime
                # schema equivalent: browser worklets can acknowledge actual
                # rendered playback, while all other clients settle on their
                # first delivered PCM for compatibility.
                if raw.get("type") == "pipeline.playback.capability":
                    rendered_ack = raw.get("rendered_playback_ack")
                    if not isinstance(rendered_ack, bool):
                        await _send_event(
                            ws,
                            unit.service.make_error(
                                "pipeline.playback.capability requires boolean rendered_playback_ack",
                                "invalid_playback_capability",
                            ),
                        )
                        continue
                    unit.service.set_rendered_playback_ack_supported(session_id, rendered_ack)
                    continue

                # Local extension: OpenAI Realtime has no browser-rendered
                # audio acknowledgement.  The worklet sends this exactly once
                # for its first scheduled sample, so response completion alone
                # cannot make an unheard answer non-supersedable.
                if raw.get("type") == "pipeline.playback.started":
                    response_id = raw.get("response_id")
                    response_epoch = raw.get("response_epoch")
                    if (
                        not isinstance(response_id, str)
                        or not response_id
                        or isinstance(response_epoch, bool)
                        or not isinstance(response_epoch, int)
                        or response_epoch < 1
                    ):
                        await _send_event(
                            ws,
                            unit.service.make_error(
                                "pipeline.playback.started requires response_id and positive response_epoch",
                                "invalid_playback_started",
                            ),
                        )
                        continue
                    acknowledged = unit.service.handle_playback_started(
                        session_id,
                        response_id=response_id,
                        response_epoch=response_epoch,
                    )
                    if acknowledged is not None:
                        await _send_event(ws, acknowledged)
                        await _send_events(
                            ws,
                            unit.service.take_deferred_settlement_events(session_id),
                        )
                    continue

                event = unit.service.parse_client_event(raw)
                if event is None:
                    await _send_event(
                        ws,
                        unit.service.make_error(
                            f"Unknown or invalid event: {raw.get('type')}", "unknown_or_invalid_event"
                        ),
                    )
                    continue

                if isinstance(event, InputAudioBufferAppendEvent):
                    chunks = unit.service.handle_audio_append(session_id, event)
                    rt_cfg = unit.service._state(session_id).runtime_config
                    for chunk in chunks:
                        unit.input_queue.put((chunk, rt_cfg))

                elif isinstance(event, InputAudioBufferCommitEvent):
                    err = unit.service.handle_audio_commit(session_id)
                    if err:
                        await _send_event(ws, err)

                elif isinstance(event, SessionUpdateEvent):
                    result = unit.service.handle_session_update(session_id, event)
                    if result:
                        await _send_event(ws, result)

                elif isinstance(event, ConversationItemCreateEvent):
                    events = unit.service.handle_conversation_item_create(session_id, event)
                    if events:
                        await _send_events(ws, events)

                elif isinstance(event, ResponseCreateEvent):
                    result = unit.service.handle_response_create(session_id, event)
                    if result:
                        if result.type != "error":
                            unit.cancel_scope.new_response()
                        await _send_event(ws, result)
                        if result.type != "error":
                            await _send_events(
                                ws,
                                unit.service.take_deferred_settlement_events(session_id),
                            )
                            owner_event = unit.service.response_owner_event(session_id)
                            if owner_event is not None:
                                await _send_event(ws, owner_event)

                elif isinstance(event, ResponseCancelEvent):
                    state = unit.service._state(session_id)
                    events = None
                    cancelled_owner = None
                    cancelled_operation = None
                    cancelled_generation = None
                    was_active = False
                    # Capture ownership and every identity-free cancellation
                    # side effect while the same transaction blocks VAD/manual
                    # successor admission. Otherwise response.cancel can read A,
                    # let B claim, then accidentally cancel B's generation or
                    # compatibility transport before its owner check rejects A.
                    with state.response_ownership.transaction():
                        cancelled_owner = state.response_ownership.active()
                        cancelled_generation = unit.cancel_scope.generation
                        was_active = state.in_response or state.response_pending
                        if was_active:
                            operation = unit.model_operations.active_token()
                            if _operation_matches_response(
                                operation,
                                cancelled_owner,
                                cancelled_generation,
                            ):
                                cancelled_operation = operation
                                if cancelled_operation is not None:
                                    unit.model_operations.request_cancel_token(
                                        cancelled_operation,
                                        "response_cancel",
                                    )
                            unit.cancel_scope.cancel()
                            # These compatibility transports do not carry an
                            # owner parameter. Close them before finish_response
                            # can promote a queued successor and before unlock.
                            _cancel_handler_transports(unit, "response_cancel")
                            if cancelled_owner is not None:
                                events = unit.service.handle_response_cancel_if_owner(
                                    session_id,
                                    input_epoch=cancelled_owner.input_epoch,
                                    response_epoch=cancelled_owner.response_epoch,
                                    response_id=cancelled_owner.response_id,
                                )
                            else:
                                # Legacy response flags without epoch ownership.
                                cancelled_operation = unit.model_operations.request_cancel(
                                    "response_cancel"
                                )
                                events = unit.service.handle_response_cancel(session_id)
                    if was_active:
                        if cancelled_owner is not None:
                            _flush_cancelled_response_output(
                                unit.output_queue,
                                owner=cancelled_owner,
                                cancel_generation=cancelled_generation,
                            )
                            _flush_cancelled_response_output(
                                unit.text_output_queue,
                                owner=cancelled_owner,
                                cancel_generation=cancelled_generation,
                            )
                        else:
                            _flush_queue(unit.output_queue, preserve=_keep_audio_sentinel)
                            _flush_queue(unit.text_output_queue, preserve=_keep_user_text_event)
                        await _cancel_active_generation(
                            unit,
                            "response_cancel",
                            response_owner=cancelled_owner,
                            model_operation=cancelled_operation,
                            capture_current_operation=False,
                            cancel_handler_transports=False,
                        )
                    elif cancelled_owner is not None:
                        events = unit.service.handle_response_cancel_if_owner(
                            session_id,
                            input_epoch=cancelled_owner.input_epoch,
                            response_epoch=cancelled_owner.response_epoch,
                            response_id=cancelled_owner.response_id,
                        )
                    else:
                        events = unit.service.handle_response_cancel(session_id)
                    if events:
                        await _send_events(ws, events)
                    await _send_events(
                        ws,
                        unit.service.take_deferred_settlement_events(session_id),
                    )
                    active_after_cancel = state.response_ownership.active()
                    if (
                        active_after_cancel is None
                        or cancelled_owner is None
                        or active_after_cancel.response_epoch == cancelled_owner.response_epoch
                    ):
                        unit.response_playing.clear()

        except WebSocketDisconnect:
            logger.info(f"Client {session_id} disconnected from pipeline {unit.index}")
        except Exception as e:
            logger.error(f"Client {session_id} on pipeline {unit.index} error: {type(e).__name__}: {e}", exc_info=True)
        finally:
            # Hold the session reference: the send loop's snapshot will still resolve
            # to this object until we clear unit.session, so any handler output that
            # arrives during the drain window is sent to the now-closed ws (silently
            # dropped) instead of leaking to whichever client claims this unit next.
            old_session = unit.session
            if old_session is not None:
                old_session.released_at = time.monotonic()
            _clean_unit(unit)
            unit.input_queue.put(SESSION_END)
            # Spawn the drain-and-release as a separate task so the route handler's
            # finally returns immediately. Awaiting here is unreliable: after
            # WebSocketDisconnect propagates, subsequent awaits in the same task
            # can be skipped/cancelled by Starlette's runner and never resume.
            asyncio.create_task(_release_unit_after_drain(unit, old_session, session_id))

    @app.get("/v1/usage")
    async def usage_endpoint() -> dict[str, Any]:
        # Aggregate usage across the pool. Numeric fields sum; dict fields (e.g.
        # errors_by_type) merge with numeric leaves summed too, so per-unit error
        # counts don't get dropped by the first-unit's value.
        def _merge(into: dict[str, Any], src: dict[str, Any]) -> None:
            for k, v in src.items():
                if isinstance(v, (int, float)):
                    into[k] = into.get(k, 0) + v
                elif isinstance(v, dict):
                    sub = into.setdefault(k, {})
                    if isinstance(sub, dict):
                        _merge(sub, v)
                else:
                    into.setdefault(k, v)

        total: dict[str, Any] = {}
        for unit in pool:
            _merge(total, unit.service.get_usage())
        return total

    @app.get("/v1/pool")
    async def pool_endpoint() -> dict[str, Any]:
        now = time.monotonic()

        def _state(u: PipelineUnit) -> dict[str, Any]:
            s = u.session
            if s is None:
                return {"index": u.index, "state": "idle", "session_id": None}
            if s.released_at is None:
                return {"index": u.index, "state": "active", "session_id": s.session_id}
            # released by client but SESSION_END hasn't drained yet → unit
            # is still occupied; surface elapsed time so operators can spot
            # stuck handlers.
            return {
                "index": u.index,
                "state": "draining",
                "session_id": s.session_id,
                "draining_for_s": round(now - s.released_at, 2),
            }

        return {
            "size": len(pool),
            "in_use": sum(1 for u in pool if u.session is not None),
            "units": [_state(u) for u in pool],
            "runtime": runtime,
        }

    async def _send_loop_for(unit: PipelineUnit) -> None:
        """Per-pipeline send loop. Polls this unit's output queues and forwards
        to the websocket currently attached via unit.session.

        Per-session scratch (pending_output_item) lives on SessionState, so it
        disappears together with the websocket when the session is released —
        no stale sentinel can leak into the next claim.
        """
        pipeline_log_ctx.set(unit.index)
        while not stop_event.is_set():
            try:
                # Snapshot the session once per iteration; if the route releases the
                # unit mid-iteration, we continue against the prior snapshot which is
                # consistent (its websocket is still valid until ws.close() returns).
                session = unit.session
                ws = session.websocket if session is not None else None
                session_id = session.session_id if session is not None else None

                # Text events first (speech_started cancels active response).
                try:
                    text_msg = unit.text_output_queue.get_nowait()
                    is_speech_start = isinstance(text_msg, SpeechStartedEvent)
                    legacy_active_response = False
                    if is_speech_start and session_id:
                        # Capture before dispatch: AudioHandler closes the
                        # interrupted OpenAI response while producing the
                        # speech-start events, so its post-dispatch flags are
                        # intentionally already clear.
                        pre_dispatch_state = unit.service._state(session_id)
                        legacy_active_response = bool(
                            unit.response_playing.is_set()
                            and (
                                pre_dispatch_state.in_response
                                or pre_dispatch_state.current_response_id is not None
                            )
                        )

                    if isinstance(text_msg, AssistantTextEvent) and _generation_is_discardable(
                        unit, text_msg.cancel_generation
                    ):
                        pass
                    elif ws is not None and isinstance(text_msg, PipelineEvent) and session_id:
                        events = unit.service.dispatch_pipeline_event(session_id, text_msg)
                        if events:
                            await _send_events(ws, events)
                        await _send_events(
                            ws,
                            unit.service.take_deferred_settlement_events(session_id),
                        )

                    if is_speech_start and session_id:
                        supersession = unit.service.take_pending_supersession(
                            session_id,
                            text_msg.input_epoch,
                        )
                        if supersession is None:
                            logger.debug(
                                "Skipping stale/unmatched speech-start input_epoch=%s",
                                text_msg.input_epoch,
                            )
                            continue
                        response_state = unit.service._state(session_id)
                        active_cfg = response_state.runtime_config
                        # ``response_playing`` alone is a queue-worker hint,
                        # not ownership. ``legacy_active_response`` was bound
                        # to the real pre-dispatch protocol response above.
                        should_interrupt = bool(
                            (
                                supersession
                                and supersession.previous is not None
                                and (
                                    supersession.pre_audible
                                    or (
                                        text_msg.interrupt_response
                                        and active_cfg.interrupt_response_enabled
                                    )
                                )
                            )
                            or (
                                legacy_active_response
                                and text_msg.interrupt_response
                                and active_cfg.interrupt_response_enabled
                            )
                        )
                        if should_interrupt:
                            cancelled_owner = supersession.previous if supersession else None
                            captured_cancel = unit.service.take_pending_transport_cancellation(
                                session_id,
                                text_msg.input_epoch,
                            )
                            if isinstance(captured_cancel, dict):
                                cancelled_generation = captured_cancel.get("cancel_generation")
                                cancelled_operation = captured_cancel.get("operation")
                                cancellation_already_captured = True
                                should_cancel_transport = True
                            else:
                                candidate_generation = unit.cancel_scope.generation
                                candidate_operation = unit.model_operations.active_token()
                                operation_matches = _operation_matches_response(
                                    candidate_operation,
                                    cancelled_owner,
                                    candidate_generation,
                                )
                                # A speech-start event may reach this loop after
                                # its completed-but-unheard predecessor was
                                # replaced and a successor already acquired the
                                # model.  Never bump the mutable cancel scope or
                                # close handler transports unless they are bound
                                # to the exact superseded operation.  The only
                                # ownerless exception is the legacy protocol
                                # response path, where pre-dispatch state is the
                                # sole available identity.
                                should_cancel_transport = bool(
                                    operation_matches
                                    or (legacy_active_response and cancelled_owner is None)
                                )
                                cancelled_generation = (
                                    candidate_generation if should_cancel_transport else None
                                )
                                cancelled_operation = (
                                    candidate_operation if operation_matches else None
                                )
                                if should_cancel_transport:
                                    unit.cancel_scope.cancel()
                                cancellation_already_captured = False
                            if should_cancel_transport and (
                                (supersession and supersession.requires_transport_cancel)
                                or legacy_active_response
                            ):
                                await _cancel_active_generation(
                                    unit,
                                    "pre_audible_supersession"
                                    if supersession and supersession.pre_audible
                                    else "barge_in",
                                    response_owner=cancelled_owner,
                                    model_operation=cancelled_operation,
                                    capture_current_operation=False,
                                    cancel_handler_transports=not cancellation_already_captured,
                                )
                            # A prior iteration may have pulled a partial PCM
                            # batch off the queue while waiting for text/output
                            # completion. It belongs to the invalidated owner
                            # and must not be reintroduced after the selective
                            # queue cleanup. A successor admitted during the
                            # transport wait must remain untouched.
                            if (
                                session is not None
                                and session.pending_output_item is not None
                                and _belongs_to_cancelled_response(
                                    session.pending_output_item,
                                    owner=cancelled_owner,
                                    cancel_generation=cancelled_generation,
                                )
                            ):
                                session.pending_output_item = None
                            if cancelled_owner is not None:
                                _flush_cancelled_response_output(
                                    unit.output_queue,
                                    owner=cancelled_owner,
                                    cancel_generation=cancelled_generation,
                                )
                                _flush_cancelled_response_output(
                                    unit.text_output_queue,
                                    owner=cancelled_owner,
                                    cancel_generation=cancelled_generation,
                                )
                            else:
                                _flush_queue(unit.output_queue, preserve=_keep_audio_sentinel)
                                _flush_queue(unit.text_output_queue, preserve=_keep_user_text_event)
                            active_after_cancel = response_state.response_ownership.active()
                            if unit.response_playing.is_set() and (
                                active_after_cancel is None
                                or cancelled_owner is None
                                or active_after_cancel.response_epoch == cancelled_owner.response_epoch
                            ):
                                unit.response_playing.clear()
                            logger.info(
                                "Pipeline %d: superseded response epoch=%s pre_audible=%s; queue flushed",
                                unit.index,
                                supersession.previous.response_epoch if supersession and supersession.previous else None,
                                supersession.pre_audible if supersession else False,
                            )
                        elif supersession and supersession.previous is not None:
                            logger.info(
                                "Pipeline %d: speech during audible response retained because interrupt_response is disabled",
                                unit.index,
                            )
                except Empty:
                    pass

                try:
                    if session is not None and session.pending_output_item is not None:
                        audio_chunk = session.pending_output_item
                        session.pending_output_item = None
                    else:
                        audio_chunk = unit.output_queue.get_nowait()

                    if _is_pipeline_end(audio_chunk):
                        audio_generation = _audio_generation(audio_chunk)
                        await _drain_pending_response_events(ws, unit, session_id)
                        if ws is not None and session_id:
                            terminal_events = unit.service.response.finish_audio_terminal_if_owner(
                                session_id,
                                input_epoch=_audio_input_epoch(audio_chunk),
                                response_epoch=_audio_response_epoch(audio_chunk),
                                response_id=_audio_response_id(audio_chunk),
                            )
                            if terminal_events is None:
                                unit.cancel_scope.response_done(audio_generation)
                                logger.info(
                                    "Pipeline %d: response terminal became stale during event drain",
                                    unit.index,
                                )
                                continue
                            await _send_events(ws, terminal_events)
                            await _send_events(
                                ws,
                                unit.service.take_deferred_settlement_events(session_id),
                            )
                        break

                    if _is_audio_done(audio_chunk):
                        audio_generation = _audio_generation(audio_chunk)
                        if _should_discard_audio(
                            unit,
                            audio_chunk,
                            service=unit.service,
                            conn_id=session_id,
                        ):
                            if session_id and _audio_response_epoch(audio_chunk) is None:
                                stale_state = unit.service._state(session_id)
                                if stale_state.response_ownership.active() is None:
                                    stale_state.response_pending = False
                            unit.cancel_scope.response_done(audio_generation)
                            unit.should_listen.set()
                            logger.info(f"Pipeline {unit.index}: stale response complete, listening re-enabled")
                            continue
                        if (
                            session_id
                            and unit.service._state(session_id).in_response
                            and unit.service._state(session_id).await_text_output_complete
                            and not unit.service._state(session_id).text_output_complete
                        ):
                            # The LLM side writes an explicit terminal marker before
                            # TTS emits this sentinel. Hold it until that marker is
                            # dispatched; this is a response barrier, not a timer.
                            if session is not None:
                                session.pending_output_item = audio_chunk
                            continue
                        await _drain_pending_response_events(ws, unit, session_id)
                        if ws is not None and session_id:
                            terminal_events = unit.service.response.finish_audio_terminal_if_owner(
                                session_id,
                                input_epoch=_audio_input_epoch(audio_chunk),
                                response_epoch=_audio_response_epoch(audio_chunk),
                                response_id=_audio_response_id(audio_chunk),
                            )
                            if terminal_events is None:
                                unit.cancel_scope.response_done(audio_generation)
                                logger.info(
                                    "Pipeline %d: audio completion became stale during event drain",
                                    unit.index,
                                )
                                continue
                            await _send_events(ws, terminal_events)
                            await _send_events(
                                ws,
                                unit.service.take_deferred_settlement_events(session_id),
                            )
                        unit.response_playing.clear()
                        unit.cancel_scope.response_done(audio_generation)
                        unit.should_listen.set()
                        logger.info(f"Pipeline {unit.index}: response complete, listening re-enabled")
                        continue

                    # SESSION_END travels from input_queue through every handler to
                    # output_queue. Observing it here means the chain has fully reset;
                    # signal the release path so it can clear unit.session.
                    if is_control_message(audio_chunk, SESSION_END.kind):
                        if session is not None:
                            session.drained.set()
                            logger.debug(f"Pipeline {unit.index}: SESSION_END drained")
                        continue

                    if is_control_message(audio_chunk):
                        continue

                    if _should_discard_audio(
                        unit,
                        audio_chunk,
                        service=unit.service,
                        conn_id=session_id,
                    ):
                        continue

                    batch_identity = _audio_identity(audio_chunk)
                    source_sample_rate = _audio_sample_rate(audio_chunk)
                    audio_chunk = _to_audio_bytes(audio_chunk)

                    audio_batch = bytearray(audio_chunk)
                    while len(audio_batch) < MAX_AUDIO_BATCH_BYTES:
                        try:
                            next_chunk = unit.output_queue.get_nowait()
                        except Empty:
                            break

                        if (
                            _is_pipeline_end(next_chunk)
                            or _is_audio_done(next_chunk)
                            or is_control_message(next_chunk, SESSION_END.kind)
                        ):
                            # Only stash if we still have a session; otherwise drop it.
                            if session is not None:
                                session.pending_output_item = next_chunk
                            break

                        if _should_discard_audio(
                            unit,
                            next_chunk,
                            service=unit.service,
                            conn_id=session_id,
                        ):
                            continue

                        if _audio_identity(next_chunk) != batch_identity:
                            # Do not merge epochs or response IDs: a detached
                            # completion can arrive beside the first PCM of the
                            # next accepted turn. Preserve its order for the
                            # following iteration.
                            if session is not None:
                                session.pending_output_item = next_chunk
                            break

                        next_audio = _to_audio_bytes(next_chunk)
                        if len(audio_batch) + len(next_audio) > MAX_AUDIO_BATCH_BYTES:
                            if session is not None:
                                session.pending_output_item = next_chunk
                            break
                        audio_batch.extend(next_audio)

                    # Pulling a chunk is not admission. A newer speech-start
                    # can invalidate this owner while batching, so confirm the
                    # immutable tuple again before conversion allocates or
                    # mutates a protocol response.
                    if not _audio_identity_is_admissible(unit.service, session_id, batch_identity):
                        continue

                    if not unit.response_playing.is_set():
                        unit.response_playing.set()
                        unit.should_listen.set()

                    if ws is not None and session_id:
                        input_epoch, response_epoch, response_id, _ = batch_identity
                        encoded_events = unit.service.encode_audio_chunk(
                            session_id,
                            bytes(audio_batch),
                            source_sample_rate=source_sample_rate,
                            input_epoch=input_epoch,
                            response_epoch=response_epoch,
                            response_id=response_id,
                        )
                        if not encoded_events:
                            continue
                        # Bind the socket-side verification to the response ID
                        # returned by this encode, rather than looking up the
                        # mutable current_response_id after another task runs.
                        encoded_response_id = next(
                            (
                                event.response_id
                                for event in encoded_events
                                if isinstance(event, ResponseAudioDeltaEvent)
                            ),
                            response_id,
                        )
                        encoded_identity = (
                            input_epoch,
                            response_epoch,
                            encoded_response_id,
                            source_sample_rate,
                        )
                        delivered_pcm = False
                        for encoded_event in encoded_events:
                            # The await below can yield to a speech-start path;
                            # check immediately before every protocol event so
                            # an old PCM batch cannot attach to a newer turn.
                            if not _audio_identity_is_admissible(
                                unit.service,
                                session_id,
                                encoded_identity,
                            ):
                                break
                            sent = await _send_event(ws, encoded_event)
                            if (
                                sent
                                and isinstance(encoded_event, ResponseAudioDeltaEvent)
                                and bool(encoded_event.delta)
                            ):
                                delivered_pcm = True
                        if delivered_pcm and _audio_identity_is_admissible(
                            unit.service,
                            session_id,
                            encoded_identity,
                        ):
                            await _send_events(
                                ws,
                                unit.service.handle_first_delivered_pcm(session_id),
                            )
                except Empty:
                    pass

                await asyncio.sleep(0.01)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Pipeline {unit.index} send loop error: {e}")
                await asyncio.sleep(0.1)

    return app
