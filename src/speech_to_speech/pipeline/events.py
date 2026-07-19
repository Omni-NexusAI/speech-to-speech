"""Typed events flowing on ``text_output_queue``.

These are internal pipeline events produced by VAD, TranscriptionNotifier, and
LMOutputProcessor, consumed by the realtime ``WebSocketRouter`` send-loop and
``RealtimeService.dispatch_pipeline_event``.  They replace the raw ``dict``
literals that were previously put on the queue.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from pydantic import BaseModel, Field


class PipelineEvent(BaseModel):
    """Base for all text_output_queue events.

    The ``type`` field mirrors the former dict ``"type"`` key and acts as a
    Pydantic discriminator.
    """

    type: str


# ── VAD events ────────────────────────────────────────────────────────


class SpeechStartedEvent(PipelineEvent):
    type: Literal["speech_started"] = "speech_started"
    audio_start_ms: int = 0
    turn_id: str | None = None
    turn_revision: int | None = None
    reopened: bool = False
    interrupt_response: bool = Field(default=True, exclude=True)


class SpeechStoppedEvent(PipelineEvent):
    type: Literal["speech_stopped"] = "speech_stopped"
    duration_s: float = 0.0
    audio_end_ms: int = 0
    turn_id: str | None = None
    turn_revision: int | None = None


# ── Transcription events (TranscriptionNotifier) ─────────────────────


class PartialTranscriptionEvent(PipelineEvent):
    type: Literal["partial_transcription"] = "partial_transcription"
    delta: str
    turn_id: str | None = None
    turn_revision: int | None = None


class TranscriptionCompletedEvent(PipelineEvent):
    type: Literal["transcription_completed"] = "transcription_completed"
    transcript: str
    language_code: Optional[str] = None
    turn_id: str | None = None
    turn_revision: int | None = None
    speech_stopped_at_s: float | None = Field(default=None, exclude=True)
    context_committed: bool = Field(default=False, exclude=True)
    # Direct-audio mode may publish a UI-only placeholder when Gemma omits
    # optional transcript metadata. It must never enter model context.
    display_only: bool = Field(default=False, exclude=True)
    # Direct-audio Gemma already generated the assistant response for this turn.
    # The realtime service must close the transcription item without starting the
    # regular text-only response path a second time.
    direct_audio_completed: bool = Field(default=False, exclude=True)


# ── LLM output events (LMOutputProcessor) ────────────────────────────


class AssistantTextEvent(PipelineEvent):
    type: Literal["assistant_text"] = "assistant_text"
    text: str
    tools: list[ResponseFunctionToolCall] = Field(default_factory=list)
    turn_id: str | None = None
    turn_revision: int | None = None
    # Response generation that produced this text, mirroring AudioOutput. Lets the
    # send loop discard stale assistant text by the same generation-aware rule as
    # audio, instead of blanket-dropping while cancel_scope.discarding is set.
    cancel_generation: int | None = None


class TokenUsageEvent(PipelineEvent):
    type: Literal["token_usage"] = "token_usage"
    input_tokens: int = 0
    output_tokens: int = 0
    turn_id: str | None = None
    turn_revision: int | None = None


class ResponseFailedEvent(PipelineEvent):
    """Signals that a response could not be generated (e.g. invalid out-of-band input).

    Dispatched to the service so it can close the in-progress response with
    ``status="failed"`` instead of the usual ``completed``.
    """

    type: Literal["response_failed"] = "response_failed"
    message: str = ""
    turn_id: str | None = None
    turn_revision: int | None = None


class ResponseOutputCompleteEvent(PipelineEvent):
    """Text/tool output for one response is fully queued before TTS completion."""

    type: Literal["response_output_complete"] = "response_output_complete"
    turn_id: str | None = None
    turn_revision: int | None = None
    cancel_generation: int | None = None


class PipelineMetricEvent(PipelineEvent):
    """Low-overhead timing/status event for the realtime diagnostics panel."""

    type: Literal["pipeline_metric"] = "pipeline_metric"
    stage: str
    status: str
    at_s: float
    elapsed_ms: float | None = None
    turn_id: str | None = None
    turn_revision: int | None = None
    detail: dict[str, Any] = Field(default_factory=dict)
