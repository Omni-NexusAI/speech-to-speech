import logging
import threading
import time
from collections.abc import Mapping
from queue import Queue
from threading import Event as ThreadingEvent
from typing import Any, Callable, Literal, Optional, TypeVar, Union

import httpx
from openai.types.realtime import (
    ConversationItem,
    ConversationItemCreatedEvent,
    ConversationItemCreateEvent,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    ConversationItemInputAudioTranscriptionDeltaEvent,
    InputAudioBufferAppendEvent,
    InputAudioBufferSpeechStartedEvent,
    InputAudioBufferSpeechStoppedEvent,
    RealtimeError,
    RealtimeErrorEvent,
    ResponseAudioDeltaEvent,
    ResponseAudioDoneEvent,
    ResponseAudioTranscriptDoneEvent,
    ResponseCancelEvent,
    ResponseCreatedEvent,
    ResponseCreateEvent,
    ResponseDoneEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    SessionCreatedEvent,
    SessionUpdatedEvent,
    SessionUpdateEvent,
)
from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from speech_to_speech.api.openai_realtime.handlers import (
    AudioHandler,
    ConversationHandler,
    ResponseHandler,
    SessionHandler,
)
from speech_to_speech.api.openai_realtime.runtime_config import ModelEndpointConfig, RuntimeConfig
from speech_to_speech.LLM.chat import Chat, make_user_message
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
from speech_to_speech.pipeline.messages import GenerateResponseRequest
from speech_to_speech.pipeline.queue_types import TextPromptItem
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.utils.utils import _generate_id

logger = logging.getLogger(__name__)

PIPELINE_SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512
BYTES_PER_SAMPLE = 2
CHUNK_SIZE_BYTES = CHUNK_SAMPLES * BYTES_PER_SAMPLE

_ResponseStatus = Literal["completed", "cancelled", "failed", "incomplete", "in_progress"]
_StatusReason = Literal["turn_detected", "client_cancelled", "max_output_tokens", "content_filter"]

_EVENT_TYPE_TO_MODEL: dict[str, type[BaseModel]] = {
    "input_audio_buffer.append": InputAudioBufferAppendEvent,
    "session.update": SessionUpdateEvent,
    "conversation.item.create": ConversationItemCreateEvent,
    "response.create": ResponseCreateEvent,
    "response.cancel": ResponseCancelEvent,
}

ClientEvent = Union[
    InputAudioBufferAppendEvent,
    SessionUpdateEvent,
    ConversationItemCreateEvent,
    ResponseCreateEvent,
    ResponseCancelEvent,
]

ServerEvent = Union[
    SessionCreatedEvent,
    SessionUpdatedEvent,
    RealtimeErrorEvent,
    InputAudioBufferSpeechStartedEvent,
    InputAudioBufferSpeechStoppedEvent,
    ConversationItemCreatedEvent,
    ConversationItemInputAudioTranscriptionDeltaEvent,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    ResponseCreatedEvent,
    ResponseDoneEvent,
    ResponseAudioDeltaEvent,
    ResponseAudioDoneEvent,
    ResponseAudioTranscriptDoneEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    "PipelineMetricServerEvent",
    "PipelineRuntimeServerEvent",
]

RealtimeEvent = Union[ClientEvent, ServerEvent]


_UsageMetricsT = TypeVar("_UsageMetricsT", bound="UsageMetrics")


class UsageMetrics(BaseModel):
    """Per-response usage counters.

    Supports ``+=`` for rolling per-response metrics into a global total
    and ``reset()`` for clearing per-response state after rollup.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    audio_duration_s: float = 0.0
    responses_completed: int = 0
    responses_cancelled: int = 0
    tool_calls: int = 0
    turns: int = 0

    def __iadd__(self: _UsageMetricsT, other: "UsageMetrics") -> _UsageMetricsT:
        for field in UsageMetrics.model_fields:
            setattr(self, field, getattr(self, field) + getattr(other, field))
        return self

    def reset(self) -> None:
        for field, info in UsageMetrics.model_fields.items():
            setattr(self, field, info.default)


class GlobalUsageMetrics(UsageMetrics):
    """Server-wide metrics that extend per-response counters with
    connection and error tracking."""

    connections: int = 0
    # connection duration in seconds.
    # latency tts, llm, vad, stt (mean, max, p90)
    errors_by_type: dict[str, int] = Field(default_factory=dict)

    def record_error(self, error_type: str) -> None:
        self.errors_by_type[error_type] = self.errors_by_type.get(error_type, 0) + 1

    @property
    def total_errors(self) -> int:
        return sum(self.errors_by_type.values())


class PipelineMetricServerEvent(BaseModel):
    type: Literal["pipeline.metric"] = "pipeline.metric"
    event_id: str
    stage: str
    status: str
    at_s: float
    elapsed_ms: float | None = None
    turn_id: str | None = None
    turn_revision: int | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class PipelineRuntimeServerEvent(BaseModel):
    type: Literal["pipeline.runtime"] = "pipeline.runtime"
    event_id: str
    runtime: dict[str, Any] = Field(default_factory=dict)


class ConnState(BaseModel):
    """Per-connection mutable state, including all protocol-level IDs."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: str = Field(default_factory=lambda: _generate_id("session"))
    conversation_id: str = Field(default_factory=lambda: _generate_id("conv"))
    runtime_config: RuntimeConfig = Field(default_factory=RuntimeConfig)
    in_response: bool = False
    response_pending: bool = False
    audio_buffer_has_data: bool = False
    audio_remainder: bytes = b""
    current_response_id: Optional[str] = None
    current_item_id: Optional[str] = None
    content_index: int = 0
    input_content_index: int = 0
    input_audio_duration_s: float = 0.0
    last_item_id: Optional[str] = None
    current_response_params: RealtimeResponseCreateParams | None = None
    pending_output_text_parts: list[str] = Field(default_factory=list)
    response_usage: UsageMetrics = Field(default_factory=UsageMetrics)
    speculative_turn_id: Optional[str] = None
    speculative_turn_revision: Optional[int] = None
    speculative_user_turn_id: Optional[str] = None
    speculative_user_turn_revision: Optional[int] = None
    speculative_user_speech_stopped_at_s: Optional[float] = None
    speculative_user_item_id: Optional[str] = None
    speculative_input_item_id: Optional[str] = None
    speculative_audio_duration_s: float = 0.0
    # Client conversation.item.create items that arrived while a response was
    # generating. Applying them mid-generation races the LLM handler's chat
    # write-back (cross-thread), so they are buffered here and flushed in order
    # once the response completes. See ConversationHandler.flush_deferred_items.
    deferred_items: list[ConversationItem] = Field(default_factory=list)
    # Tool continuations are a call-ID-bound transaction. The browser may only
    # create one post-tool response after every matching output reaches Chat.
    pending_tool_call_ids: set[str] = Field(default_factory=set)
    tool_followup_ready: bool = False
    tool_followup_started: bool = False
    tool_followup_requested: bool = False
    tool_followup_response: RealtimeResponseCreateParams | None = None
    current_response_is_tool_followup: bool = False
    text_output_complete: bool = False
    await_text_output_complete: bool = False
    history_tokens: int = 0
    history_token_source: str = "empty"
    history_token_fingerprint: str = ""
    history_token_pending: bool = False


class RealtimeService:
    """Translates between OpenAI Realtime protocol events and internal pipeline messages.

    One instance is shared across all WebSocket connections.  Per-connection
    state (response lifecycle, audio buffer) is tracked internally by
    connection id.
    """

    def __init__(
        self,
        text_prompt_queue: Queue[TextPromptItem] | None = None,
        should_listen: ThreadingEvent | None = None,
        chat_size: int = 10,
        speculative_turns: SpeculativeTurnTracker | None = None,
        context_tokenizer_base_url: str | None = None,
        default_model_name: str = "gemma-4-12b-it-qat",
        default_model_api_key: str | None = None,
    ) -> None:
        self.text_prompt_queue = text_prompt_queue
        self.should_listen = should_listen
        self._chat_size = chat_size
        self.speculative_turns = speculative_turns
        self.context_tokenizer_base_url = context_tokenizer_base_url.rstrip("/") if context_tokenizer_base_url else None
        self.default_model_endpoint = ModelEndpointConfig(
            provider="local",
            base_url=self.context_tokenizer_base_url or "http://127.0.0.1:8818/v1",
            model=default_model_name,
            api_key=default_model_api_key,
        )
        self._conns: dict[str, ConnState] = {}
        self.total_usage = GlobalUsageMetrics()

        self.audio = AudioHandler(self)
        self.session = SessionHandler(self)
        self.response = ResponseHandler(self)
        self.conversation = ConversationHandler(self)

        self._pipeline_dispatch: dict[type[PipelineEvent], Callable[..., list[ServerEvent]]] = {
            SpeechStartedEvent: self.audio.on_speech_started,
            SpeechStoppedEvent: self.audio.on_speech_stopped,
            TokenUsageEvent: self._on_token_usage,
            PartialTranscriptionEvent: self.conversation.on_partial_transcription,
            TranscriptionCompletedEvent: self._on_transcription_completed,
            ResponseFailedEvent: self._on_response_failed,
            ResponseOutputCompleteEvent: self._on_response_output_complete,
            PipelineMetricEvent: self._on_pipeline_metric,
        }

    @staticmethod
    def _headers(api_key: str | None) -> dict[str, str]:
        return {"Authorization": f"Bearer {api_key}"} if api_key else {}

    @staticmethod
    def _normalize_model_base_url(base_url: str) -> str:
        normalized = base_url.strip().rstrip("/")
        if not normalized:
            raise ValueError("Model endpoint URL is required")
        return normalized if normalized.endswith("/v1") else f"{normalized}/v1"

    def validate_model_endpoint(
        self,
        *,
        provider: Literal["local", "remote"],
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ) -> ModelEndpointConfig:
        if provider == "local":
            source = self.default_model_endpoint
            base_url = source.base_url
            model = source.model
            api_key = source.api_key
        elif not model or not model.strip():
            raise ValueError("Remote model name is required")

        normalized = self._normalize_model_base_url(base_url or "")
        headers = self._headers(api_key)
        timeout = httpx.Timeout(5.0, connect=5.0)
        with httpx.Client(timeout=timeout, headers=headers) as client:
            response = client.get(f"{normalized}/models")
            response.raise_for_status()
            models = response.json().get("data") or []
            advertised = next((str(item.get("id")) for item in models if item.get("id")), None)
            context_window = None
            try:
                props = client.get(f"{normalized.removesuffix('/v1')}/props")
                props.raise_for_status()
                value = props.json().get("default_generation_settings", {}).get("n_ctx")
                context_window = int(value) if value else None
            except Exception:
                logger.debug("Selected model endpoint does not expose llama.cpp /props")
        return ModelEndpointConfig(
            provider=provider,
            base_url=normalized,
            model=str(model),
            api_key=api_key,
            advertised_model=advertised,
            context_window=context_window,
        )

    def _probe_context_window(self, endpoint: ModelEndpointConfig) -> int | None:
        try:
            response = httpx.get(
                f"{endpoint.base_url.removesuffix('/v1')}/props",
                headers=self._headers(endpoint.api_key),
                timeout=1.0,
            )
            response.raise_for_status()
            value = response.json().get("default_generation_settings", {}).get("n_ctx")
            return int(value) if value else None
        except Exception:
            logger.warning("Could not detect llama.cpp context window", exc_info=True)
            return None

    def _refresh_history_tokens(
        self,
        conn_id: str,
        content: str,
        fingerprint: str,
        endpoint: ModelEndpointConfig,
    ) -> None:
        """Refresh llama.cpp token counts off the websocket/send-loop thread."""
        try:
            response = httpx.post(
                f"{endpoint.base_url.removesuffix('/v1')}/tokenize",
                headers=self._headers(endpoint.api_key),
                json={"content": content, "add_special": False, "with_pieces": False},
                timeout=2.0,
            )
            response.raise_for_status()
            tokens = len(response.json().get("tokens") or [])
        except Exception:
            logger.debug("Could not refresh retained-history token count", exc_info=True)
            tokens = None
        st = self._conns.get(conn_id)
        if st is None:
            return
        if tokens is not None and st.runtime_config.chat.history_token_text() == content:
            st.history_tokens = tokens
            st.history_token_source = "llama.cpp"
            st.history_token_fingerprint = fingerprint
        st.history_token_pending = False

    def _history_tokens(self, conn_id: str, chat: Chat) -> tuple[int, str]:
        content = chat.history_token_text()
        st = self._state(conn_id)
        if not content:
            st.history_tokens = 0
            st.history_token_source = "empty"
            st.history_token_fingerprint = ""
            return 0, "empty"

        fingerprint = str(hash(content))
        if fingerprint != st.history_token_fingerprint:
            # Keep the last exact value while llama.cpp is busy, but never report
            # zero for non-empty history. The estimate is replaced asynchronously.
            estimate = max(1, len(content) // 4)
            st.history_tokens = max(st.history_tokens, estimate)
            st.history_token_source = "estimated" if st.history_token_source == "empty" else "cached"
            endpoint = st.runtime_config.model_endpoint
            if endpoint.base_url and not st.history_token_pending:
                st.history_token_pending = True
                threading.Thread(
                    target=self._refresh_history_tokens,
                    args=(conn_id, content, fingerprint, endpoint.model_copy(deep=True)),
                    name=f"context-tokenizer-{conn_id[-8:]}",
                    daemon=True,
                ).start()
            elif not endpoint.base_url:
                st.history_token_fingerprint = fingerprint
        return st.history_tokens, st.history_token_source

    def context_detail(self, conn_id: str) -> dict[str, Any]:
        chat = self._state(conn_id).runtime_config.chat
        history_tokens, token_source = self._history_tokens(conn_id, chat)
        detail: dict[str, Any] = {
            **chat.stats(),
            "history_tokens": history_tokens,
            "token_source": token_source,
        }
        context_window = self._state(conn_id).runtime_config.model_endpoint.context_window
        detail["max_tokens"] = context_window
        detail["percent"] = (
            round(100 * detail["history_tokens"] / context_window, 2) if context_window else None
        )
        detail["policy"] = "visible_trim"
        return detail

    def context_metric(self, conn_id: str, status: str = "updated") -> PipelineMetricServerEvent:
        return PipelineMetricServerEvent(
            event_id=self._next_event_id(),
            stage="context",
            status=status,
            at_s=time.time(),
            detail=self.context_detail(conn_id),
        )

    # ── Connection lifecycle ─────────────────────

    def register(self) -> str:
        """Register a new connection and return its session_id."""
        if self.speculative_turns:
            self.speculative_turns.reset()
        state = ConnState(
            runtime_config=RuntimeConfig(
                chat=Chat(self._chat_size),
                model_endpoint=self.default_model_endpoint.model_copy(deep=True),
            )
        )
        state.runtime_config.local_pipeline["_session_id"] = state.session_id
        self._conns[state.session_id] = state
        self.total_usage.connections += 1
        return state.session_id

    def unregister(self, conn_id: str) -> None:
        st = self._conns.pop(conn_id, None)
        if st is not None:
            # Suppress any in-flight compaction splice so a daemon worker can't
            # mutate a Chat tied to a closed session, and don't make further
            # billable LLM calls on its behalf once the splice is suppressed.
            st.runtime_config.chat.close()
            self.total_usage += st.response_usage
            logger.info(
                "Session %s unregistered — cumulative: input_tokens=%d, output_tokens=%d, audio=%.2fs",
                conn_id,
                self.total_usage.input_tokens,
                self.total_usage.output_tokens,
                self.total_usage.audio_duration_s,
            )

    def _state(self, conn_id: str) -> ConnState:
        return self._conns[conn_id]

    @property
    def connection_ids(self) -> list[str]:
        return list(self._conns)

    # ── Client event parsing ─────────────────────

    @staticmethod
    def _next_event_id() -> str:
        return _generate_id("event")

    def parse_client_event(self, raw: Mapping[str, object]) -> Optional[ClientEvent]:
        raw_type = raw.get("type")
        event_type: Optional[str] = raw_type if isinstance(raw_type, str) else None
        if event_type is None:
            logger.warning("Client event missing 'type' field")
            return None
        model_cls = _EVENT_TYPE_TO_MODEL.get(event_type)
        if model_cls is None:
            logger.warning(f"Unknown client event type: {event_type}")
            return None
        try:
            return model_cls.model_validate(raw)  # type: ignore[return-value]
        except ValidationError as e:
            logger.error(f"Invalid {event_type} payload: {e}")
            return None

    # ── Client event handlers ────────────────────

    def build_session_created(self, conn_id: str) -> SessionCreatedEvent:
        return self.session.build_session_created(conn_id)

    def handle_session_update(self, conn_id: str, event: SessionUpdateEvent) -> RealtimeErrorEvent | SessionUpdatedEvent | None:
        return self.session.handle_session_update(conn_id, event)

    def handle_audio_append(self, conn_id: str, event: InputAudioBufferAppendEvent) -> list[bytes]:
        return self.audio.handle_audio_append(conn_id, event)

    def handle_audio_commit(self, conn_id: str) -> RealtimeErrorEvent | None:
        return self.audio.handle_audio_commit(conn_id)

    def encode_audio_chunk(self, conn_id: str, audio: bytes) -> list[ServerEvent]:
        return self.audio.encode_audio_chunk(conn_id, audio)

    def handle_response_create(self, conn_id: str, event: ResponseCreateEvent) -> ServerEvent | None:
        return self.response.handle_response_create(conn_id, event)

    def handle_response_cancel(self, conn_id: str) -> list[ServerEvent]:
        return self.response.handle_response_cancel(conn_id)

    def finish_response(
        self,
        conn_id: str,
        status: _ResponseStatus = "completed",
        reason: _StatusReason | None = None,
    ) -> list[ServerEvent]:
        return self.response.finish_response(conn_id, status, reason)

    def handle_conversation_item_create(self, conn_id: str, event: ConversationItemCreateEvent) -> list[ServerEvent]:
        return self.conversation.handle_conversation_item_create(conn_id, event)

    def dispatch_pipeline_event(self, conn_id: str, event: PipelineEvent) -> list[ServerEvent]:
        """Route a pipeline text_output_queue event to the appropriate handler."""
        events = self._dispatch_pipeline_event(conn_id, event, wait_for_pending_reopen=True)
        return [] if events is None else events

    def try_dispatch_pipeline_event(self, conn_id: str, event: PipelineEvent) -> list[ServerEvent] | None:
        """Non-blocking dispatch.

        Returns ``None`` when dispatch must be retried after a speculative
        reopen candidate resolves.
        """
        return self._dispatch_pipeline_event(conn_id, event, wait_for_pending_reopen=False)

    def should_defer_pipeline_event(self, event: PipelineEvent) -> bool:
        if self.speculative_turns is None or not isinstance(event, (AssistantTextEvent, TokenUsageEvent)):
            return False
        return self.speculative_turns.has_pending_reopen_or_grace(
            getattr(event, "turn_id", None),
            getattr(event, "turn_revision", None),
        )

    def _dispatch_pipeline_event(
        self,
        conn_id: str,
        event: PipelineEvent,
        *,
        wait_for_pending_reopen: bool,
    ) -> list[ServerEvent] | None:
        is_stale = self._is_stale_turn_event(event, wait_for_pending_reopen=wait_for_pending_reopen)
        if is_stale is None:
            return None
        if is_stale:
            logger.info(
                "Ignoring stale %s for turn=%s rev=%s",
                event.type,
                getattr(event, "turn_id", None),
                getattr(event, "turn_revision", None),
            )
            return []

        self._observe_turn_event(event)
        if isinstance(event, AssistantTextEvent):
            return self.response.on_assistant_text(
                conn_id,
                event,
                wait_for_pending_reopen=wait_for_pending_reopen,
            )
        handler = self._pipeline_dispatch.get(type(event))
        if handler is None:
            logger.debug("Unhandled pipeline event type: %s", type(event).__name__)
            return []
        return handler(conn_id, event)

    def _is_stale_turn_event(self, event: PipelineEvent, *, wait_for_pending_reopen: bool = True) -> bool | None:
        if self.speculative_turns is None:
            return False
        if not isinstance(
            event,
            (
                PartialTranscriptionEvent,
                TranscriptionCompletedEvent,
                AssistantTextEvent,
                TokenUsageEvent,
                ResponseOutputCompleteEvent,
            ),
        ):
            return False
        turn_id = getattr(event, "turn_id", None)
        turn_revision = getattr(event, "turn_revision", None)
        if isinstance(event, (AssistantTextEvent, TokenUsageEvent, ResponseOutputCompleteEvent)):
            is_latest: bool | None
            if wait_for_pending_reopen:
                is_latest = self.speculative_turns.is_latest_after_reopen_grace(turn_id, turn_revision)
            else:
                is_latest = self.speculative_turns.try_is_latest_after_reopen_grace(turn_id, turn_revision)
            if is_latest is None:
                return None
            return not is_latest
        return not self.speculative_turns.is_latest(turn_id, turn_revision)

    def _observe_turn_event(self, event: PipelineEvent) -> None:
        if self.speculative_turns is None:
            return
        self.speculative_turns.observe(
            getattr(event, "turn_id", None),
            getattr(event, "turn_revision", None),
        )

    # ── STT → LM bridge ────────────────────────────

    def _on_transcription_completed(self, conn_id: str, event: TranscriptionCompletedEvent) -> list[ServerEvent]:
        """Handle a final STT transcription: emit protocol event, append to chat, trigger LM."""
        st = self._state(conn_id)
        same_speculative_turn = event.turn_id is not None and event.turn_id == st.speculative_user_turn_id
        if same_speculative_turn:
            st.response_usage.audio_duration_s -= st.speculative_audio_duration_s
        else:
            st.speculative_audio_duration_s = 0.0

        events = self.conversation.on_transcription_completed(conn_id, event)
        if event.turn_id is not None:
            st.speculative_audio_duration_s = st.input_audio_duration_s

        cfg = st.runtime_config
        transcript = event.transcript
        context_transcript = None if event.display_only else transcript
        if context_transcript and not event.context_committed:
            if same_speculative_turn and st.speculative_user_item_id:
                replaced = cfg.chat.replace_user_message_text(st.speculative_user_item_id, context_transcript)
                if not replaced:
                    item = cfg.chat.add_item(make_user_message(context_transcript))
                    st.speculative_user_item_id = item.id
            else:
                item = cfg.chat.add_item(make_user_message(context_transcript))
                st.speculative_user_item_id = item.id
        elif not event.context_committed and same_speculative_turn and st.speculative_user_item_id:
            cfg.chat.remove_user_message(st.speculative_user_item_id)
            st.speculative_user_item_id = None
        elif not event.context_committed and event.turn_id is not None and event.turn_id != st.speculative_user_turn_id:
            st.speculative_user_item_id = None

        if event.turn_id is not None:
            st.speculative_user_turn_id = event.turn_id
            st.speculative_user_turn_revision = event.turn_revision
            st.speculative_user_speech_stopped_at_s = event.speech_stopped_at_s

        # Direct Gemma commits complete user/assistant/tool history itself.
        # Publish the authoritative tokenized value from the service rather
        # than a partial stats-only metric from a pipeline handler.
        if self._state(conn_id).runtime_config.model_endpoint.context_window is not None:
            events.append(self.context_metric(conn_id, "committed" if event.context_committed else "updated"))

        queue = self.text_prompt_queue
        if event.direct_audio_completed:
            # Gemma direct-audio already owns this answer and its phrase chunks
            # are flowing through the TTS path. Starting GenerateResponseRequest
            # here would make a second Gemma call and a duplicate playback.
            logger.info(
                "Direct-audio completion turn=%s rev=%s: normal generation suppressed",
                event.turn_id,
                event.turn_revision,
            )
        elif queue and context_transcript:
            st.response_pending = True
            queue.put(
                GenerateResponseRequest(
                    runtime_config=cfg,
                    language_code=event.language_code,
                    turn_id=event.turn_id,
                    turn_revision=event.turn_revision,
                    speech_stopped_at_s=event.speech_stopped_at_s,
                )
            )

        return events

    # ── Metrics ────────────────────────────────────

    def _on_token_usage(self, conn_id: str, event: TokenUsageEvent) -> list[ServerEvent]:
        """Accumulate input/output token counts on the connection's usage metrics."""
        if self.speculative_turns and not self.speculative_turns.is_latest(
            event.turn_id,
            event.turn_revision,
        ):
            logger.debug("Dropping stale token usage for turn=%s rev=%s", event.turn_id, event.turn_revision)
            return []
        st = self._state(conn_id)
        st.response_usage.input_tokens += event.input_tokens
        st.response_usage.output_tokens += event.output_tokens
        logger.info(
            "Token usage (response): input=%d, output=%d",
            st.response_usage.input_tokens,
            st.response_usage.output_tokens,
        )
        return []

    def _on_response_failed(self, conn_id: str, event: ResponseFailedEvent) -> list[ServerEvent]:
        """Surface the failure to the client and close the response as ``failed``.

        Emitted when generation failed (e.g. invalid out-of-band input, or the
        provider rejecting an empty context). A top-level ``error`` event carries
        the human-readable reason — ``response.done.status_details.error`` only
        has code/type, no message — then ``finish_response`` closes the slot.

        Idempotent: gated on an active response, and ``finish_response`` is itself
        a no-op once the slot is closed, so a later EndOfResponse-driven close does
        nothing.
        """
        logger.info("Response failed: %s", event.message)
        if not self._state(conn_id).in_response:
            return []
        events: list[ServerEvent] = [self.make_error(event.message, "response_failed")]
        events.extend(self.response.finish_response(conn_id, status="failed"))
        return events

    def _on_response_output_complete(
        self, conn_id: str, event: ResponseOutputCompleteEvent
    ) -> list[ServerEvent]:
        self._state(conn_id).text_output_complete = True
        if self._state(conn_id).runtime_config.model_endpoint.context_window is not None:
            return [self.context_metric(conn_id, "committed")]
        return []

    def _on_pipeline_metric(self, conn_id: str, event: PipelineMetricEvent) -> list[ServerEvent]:
        return [
            PipelineMetricServerEvent(
                event_id=self._next_event_id(),
                stage=event.stage,
                status=event.status,
                at_s=event.at_s,
                elapsed_ms=event.elapsed_ms,
                turn_id=event.turn_id,
                turn_revision=event.turn_revision,
                detail=event.detail,
            )
        ]

    def get_usage(self) -> dict[str, Any]:
        """Return cumulative usage metrics across all completed responses."""
        data = self.total_usage.model_dump()
        data["total_tokens"] = data["input_tokens"] + data["output_tokens"]
        data["total_errors"] = self.total_usage.total_errors
        return data

    # ── Error ───────────────────────────────────

    def make_error(self, message: str, _type: str) -> RealtimeErrorEvent:
        self.total_usage.record_error(_type)
        return build_error_event(message, _type)


def build_error_event(message: str, error_type: str) -> RealtimeErrorEvent:
    """Construct a RealtimeErrorEvent without touching any service-instance state.

    Used by the websocket route handler on pool rejection, where no unit's
    service should be charged with the error in its usage metrics.
    """
    return RealtimeErrorEvent(
        type="error",
        error=RealtimeError(message=message, type=error_type),
        event_id=_generate_id("event"),
    )
