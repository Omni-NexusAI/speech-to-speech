import logging
import threading
import time
from collections.abc import Mapping
from copy import deepcopy
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
from speech_to_speech.LLM.direct_history_compaction import DirectHistoryMaintenance, normalize_history_compaction
from speech_to_speech.pipeline.model_operations import ModelOperationCoordinator
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
from speech_to_speech.pipeline.response_ownership import (
    ResponseOwner,
    ResponseOwnershipTracker,
    Supersession,
    response_epoch_history_transaction,
)
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
    "PipelineSpeechStartedServerEvent",
    "PipelineResponseServerEvent",
    "PipelineRuntimeServerEvent",
]

RealtimeEvent = Union[ClientEvent, ServerEvent]


_RESPONSE_SCOPED_METRIC_STAGES = frozenset({"gemma", "tts", "playback", "end_to_end"})
_TOOL_CALL_TOMBSTONE_LIMIT = 128


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
    input_epoch: int | None = None
    response_epoch: int | None = None
    response_id: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class PipelineSpeechStartedServerEvent(BaseModel):
    """Local speech-start decision made from authoritative ownership state."""

    type: Literal["pipeline.input_audio.speech_started"] = "pipeline.input_audio.speech_started"
    event_id: str
    input_epoch: int
    effective_interrupt: bool
    response_epoch: int | None = None
    reason: str


class PipelineResponseServerEvent(BaseModel):
    """Additive local ownership state; OpenAI response fields stay unchanged."""

    type: Literal["pipeline.response"] = "pipeline.response"
    event_id: str
    input_epoch: int
    response_epoch: int
    state: str
    response_id: str | None = None
    reason: str | None = None
    # Local browser extension: the response-owned PCM source clock.  This is
    # deliberately separate from mutable pipeline configuration, whose next
    # acknowledgement may already describe a later response.
    output_sample_rate: int | None = None
    playback_policy: dict[str, Any] | None = None


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
    response_audio_emitted: bool = False
    # The browser treats one response as one PCM clock. Freeze the negotiated
    # output transport when ownership is claimed so a live settings update
    # cannot retime even the first chunk of an already-pending response.
    response_output_sample_rate: int | None = None
    response_output_sample_rates: dict[int, int] = Field(default_factory=dict)
    # OpenAI-compatible clients do not know the local worklet acknowledgement.
    # They retain the historical first-delivered-PCM audible boundary unless a
    # browser explicitly declares rendered-sample acknowledgement support.
    rendered_playback_ack_supported: bool = False
    response_ownership: ResponseOwnershipTracker = Field(default_factory=ResponseOwnershipTracker)
    # Speech-start delivery is asynchronous. Keep each input epoch's decision
    # independently so an older queued start cannot consume a newer one.
    pending_supersessions: dict[int, Supersession] = Field(default_factory=dict)
    pending_transport_cancellations: dict[int, Any] = Field(default_factory=dict)
    # Manual response.create is not admitted between a confirmed VAD start and
    # that input epoch's accepted stop/response claim.
    input_capture_pending: bool = False
    # Checkpoints make user/assistant/tool writes provisional until the browser
    # acknowledges its first rendered sample for the response epoch.
    provisional_chat_checkpoints: dict[int, Any] = Field(default_factory=dict)
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
    # A normal STT completion can arrive while its accepted response epoch is
    # deliberately queued behind an audible, non-interrupted owner.  Keep it
    # response-scoped until promotion rather than dropping it as stale or
    # blocking the single websocket output loop while waiting.
    deferred_transcriptions: dict[int, TranscriptionCompletedEvent] = Field(default_factory=dict)
    deferred_settlement_events: list[Any] = Field(default_factory=list)
    # Tool continuations are a call-ID-bound transaction. The browser may only
    # create one post-tool response after every matching output reaches Chat.
    pending_tool_call_ids: set[str] = Field(default_factory=set)
    # A bounded idempotency ledger prevents duplicate model terminal events or
    # repeated client output submissions from executing a call twice.
    completed_tool_call_ids: list[str] = Field(default_factory=list)
    tool_followup_ready: bool = False
    tool_followup_started: bool = False
    tool_followup_requested: bool = False
    tool_followup_response: RealtimeResponseCreateParams | None = None
    # When an accepted speech successor replaces a separate tool continuation,
    # consume the browser's one inevitable response.create instead of starting
    # a second answer after that successor.
    suppress_next_tool_followup_create: bool = False
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

    tool_call_tombstone_limit = _TOOL_CALL_TOMBSTONE_LIMIT

    def __init__(
        self,
        text_prompt_queue: Queue[TextPromptItem] | None = None,
        should_listen: ThreadingEvent | None = None,
        chat_size: int = 10,
        speculative_turns: SpeculativeTurnTracker | None = None,
        context_tokenizer_base_url: str | None = None,
        default_model_name: str = "gemma-4-12b-it-qat",
        default_model_api_key: str | None = None,
        model_operations: ModelOperationCoordinator | None = None,
        direct_audio_session: bool = False,
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
        self._history_maintenance = DirectHistoryMaintenance(model_operations) if model_operations else None
        self._model_operations = model_operations
        # This is deliberately supplied by the pipeline's selected STT mode,
        # rather than inferred from a model endpoint.  A remote endpoint is a
        # valid non-direct conversation too, while every direct Gemma session
        # must begin with token-managed history before its first client item.
        self._direct_audio_session = direct_audio_session

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
        runtime_config = self._state(conn_id).runtime_config
        chat = runtime_config.chat
        history_tokens, token_source = self._history_tokens(conn_id, chat)
        detail: dict[str, Any] = {
            **chat.stats(),
            "history_tokens": history_tokens,
            "token_source": token_source,
        }
        context_window = runtime_config.model_endpoint.context_window
        detail["max_tokens"] = context_window
        detail["percent"] = (
            round(100 * detail["history_tokens"] / context_window, 2) if context_window else None
        )
        detail["policy"] = "token_compaction" if getattr(chat, "_managed_history", False) else "visible_trim"
        local = runtime_config.local_pipeline
        policy = local.get("history_compaction")
        telemetry = local.get("_history_compaction_telemetry")
        if isinstance(policy, dict):
            detail["history_compaction"] = {
                "policy": normalize_history_compaction(policy),
                "status": deepcopy(telemetry) if isinstance(telemetry, dict) else {"status": "idle", "last_failure": None},
            }
        return detail

    def context_metric(self, conn_id: str, status: str = "updated") -> PipelineMetricServerEvent:
        st = self._state(conn_id)
        owner = st.response_ownership.active()
        return PipelineMetricServerEvent(
            event_id=self._next_event_id(),
            stage="context",
            status=status,
            at_s=time.time(),
            input_epoch=owner.input_epoch if owner else st.response_ownership.input_epoch,
            response_epoch=owner.response_epoch if owner else None,
            response_id=owner.response_id if owner else None,
            detail=self.context_detail(conn_id),
        )

    # ── Conversation response ownership ─────────────────────────────

    def _owner_event(self, conn_id: str, owner: ResponseOwner) -> PipelineResponseServerEvent:
        """Return lifecycle metadata with the complete admission-time policy."""
        st = self._state(conn_id)
        frozen = st.runtime_config.response_synthesis_configs.get(owner.response_epoch, {})
        playback_policy = deepcopy(frozen.get("playback_policy")) if isinstance(frozen, dict) else None
        frozen_pipeline = frozen.get("local_pipeline") if isinstance(frozen, dict) else None
        output_sample_rate = st.response_output_sample_rates.get(
            owner.response_epoch,
            st.response_output_sample_rate,
        )
        if isinstance(playback_policy, dict):
            playback_policy["source_sample_rate"] = output_sample_rate
            if isinstance(frozen_pipeline, dict):
                playback_policy["provider"] = frozen_pipeline.get("tts_backend")
                tuning = frozen_pipeline.get("tts_tuning")
                if isinstance(tuning, dict):
                    playback_policy["profile_id"] = tuning.get("profile_id")
                    playback_policy["profile_revision"] = tuning.get("profile_revision")
        # ``response.done`` only completes the transport.  For browsers that
        # acknowledge actual worklet rendering, make the remaining provisional
        # interval explicit rather than presenting it as a fully settled turn.
        lifecycle_state = (
            "playback_pending"
            if owner.state == "completed" and not owner.playback_started
            else owner.state
        )
        return PipelineResponseServerEvent(
            event_id=_generate_id("event"),
            input_epoch=owner.input_epoch,
            response_epoch=owner.response_epoch,
            state=lifecycle_state,
            response_id=owner.response_id,
            reason=owner.reason,
            output_sample_rate=output_sample_rate,
            playback_policy=playback_policy,
        )

    def build_speech_started_decision(
        self,
        *,
        input_epoch: int,
        effective_interrupt: bool,
        response_epoch: int | None,
        reason: str,
    ) -> PipelineSpeechStartedServerEvent:
        """Describe whether this accepted speech start invalidated playback."""

        return PipelineSpeechStartedServerEvent(
            event_id=_generate_id("event"),
            input_epoch=input_epoch,
            effective_interrupt=effective_interrupt,
            response_epoch=response_epoch,
            reason=reason,
        )

    def observe_speech_started(
        self,
        conn_id: str,
        *,
        reason: str,
        interrupt_response: bool = True,
    ) -> Supersession:
        """Advance capture identity and invalidate only when policy permits it."""
        st = self._state(conn_id)
        active_operation = self._model_operations.active_token() if self._model_operations else None
        if active_operation is not None and active_operation.kind == "history_compaction":
            self._model_operations.cancel_and_wait("newer_speech", 2.0)
        # Admission identity, invalidation, rollback, and the adjacent frozen
        # response configuration form one transaction.  Without this outer
        # lock a response.create can claim a new owner between input_started()
        # and the old checkpoint rollback, allowing that rollback to erase the
        # newly admitted response's Chat history.
        with st.response_ownership.transaction():
            active = st.response_ownership.active()
            should_interrupt = bool(interrupt_response and st.runtime_config.interrupt_response_enabled)
            # An unheard provisional answer is always superseded by confirmed newer
            # speech. Once playback is actually audible, an interruption-disabled
            # session keeps that answer authoritative through its terminal event.
            invalidate_active = active is None or not active.audible or should_interrupt
            supersession = st.response_ownership.input_started(
                reason=reason,
                invalidate_active=invalidate_active,
            )
            st.runtime_config.local_pipeline["_input_epoch"] = supersession.input_epoch
            st.pending_supersessions[supersession.input_epoch] = supersession
            st.input_capture_pending = True
            previous = supersession.previous
            for queued in supersession.superseded_queued:
                # A queued owner has never been promoted and therefore has not
                # mutated Chat. Its claim-time checkpoint predates any trailing
                # history still being committed by the audible owner. Restoring
                # it would erase that valid tail when a third input supersedes
                # the queue; discard the unused checkpoint instead.
                st.provisional_chat_checkpoints.pop(queued.response_epoch, None)
                self._release_response_synthesis_config(st, queued.response_epoch)
                st.response_output_sample_rates.pop(queued.response_epoch, None)
                st.deferred_transcriptions.pop(queued.response_epoch, None)
                st.deferred_settlement_events.append(
                    PipelineResponseServerEvent(
                        event_id=_generate_id("event"),
                        input_epoch=supersession.input_epoch,
                        response_epoch=queued.response_epoch,
                        state="cancelled",
                        response_id=queued.response_id,
                        reason=reason,
                        output_sample_rate=None,
                    )
                )
            if supersession.invalidated and previous is not None and (
                previous.state not in {"completed", "cancelled"} or not previous.audible
            ):
                # The old owner can now finish only as detached stale work. Remove
                # its admission-time TTS config immediately rather than waiting for
                # an optional/late terminal event to clean it up.
                self._release_response_synthesis_config(st, previous.response_epoch)
                st.response_output_sample_rates.pop(previous.response_epoch, None)
                st.deferred_settlement_events.append(
                    PipelineResponseServerEvent(
                        event_id=_generate_id("event"),
                        input_epoch=supersession.input_epoch,
                        response_epoch=previous.response_epoch,
                        state="cancelled",
                        response_id=previous.response_id,
                        reason=reason,
                        output_sample_rate=st.response_output_sample_rate,
                    )
                )
            if previous is not None and supersession.pre_audible:
                self.rollback_provisional_response(conn_id, previous.response_epoch, reason=reason)
                # Do not let a provisional function call manufacture a second
                # follow-up after the final user input is accepted.
                stale_call_ids = set(st.pending_tool_call_ids)
                if stale_call_ids:
                    st.deferred_items = [
                        item
                        for item in st.deferred_items
                        if not (
                            getattr(item, "type", None) == "function_call_output"
                            and getattr(item, "call_id", None) in stale_call_ids
                        )
                    ]
                st.pending_tool_call_ids.clear()
                st.tool_followup_ready = False
                st.tool_followup_started = False
                st.tool_followup_requested = False
                st.tool_followup_response = None
                # Safe client items must be visible to the replacement response's
                # context immediately after rollback.  Hold only their protocol
                # acknowledgements until the queued speech-start reaches the
                # WebSocket; stale tool outputs were removed above.
                st.deferred_settlement_events.extend(self.conversation.flush_deferred_items(conn_id))
            legacy_protocol_response = previous is None and st.in_response and should_interrupt
            owned_protocol_response = (
                previous is not None
                and previous.response_id is not None
                and st.in_response
                and st.current_response_id == previous.response_id
            )
            if (legacy_protocol_response and should_interrupt) or (
                supersession.invalidated and owned_protocol_response
            ):
                # VAD observes ownership before its SpeechStartedEvent reaches
                # the asynchronous websocket send loop. Close only the exact
                # previous protocol presentation now, while the ownership lock
                # still prevents the accepted stop from claiming its successor.
                # A delayed speech-start event can then never close that newer
                # response through mutable ``st.in_response`` state. Raw-byte
                # compatibility clients predate response ownership, so their
                # sole implicit protocol response is also settled here when no
                # owner exists; it must not remain orphaned across barge-in.
                st.deferred_settlement_events.extend(
                    self.response.finish_response(
                        conn_id,
                        status="cancelled",
                        reason="turn_detected",
                    )
                )
            if supersession.requires_transport_cancel:
                # This fixes the direct-Gemma gap where no protocol response exists
                # yet, so neither legacy response flag was a reliable authority.
                st.response_pending = False
                logger.info(
                    "Superseded response epoch=%s for input epoch=%d (reason=%s, pre_audible=%s)",
                    supersession.previous.response_epoch if supersession.previous else None,
                    supersession.input_epoch,
                    reason,
                    supersession.pre_audible,
                )
        # Transport teardown must not run under the ownership/history lock,
        # but it still runs synchronously before the VAD callback returns and
        # therefore before that same VAD worker can claim the successor.
        if supersession.requires_transport_cancel:
            capture_cancel = st.runtime_config.local_pipeline.get("_capture_superseded_response")
            if callable(capture_cancel):
                st.pending_transport_cancellations[supersession.input_epoch] = capture_cancel(
                    supersession
                )
        return supersession

    def pending_supersession(self, conn_id: str, input_epoch: int | None) -> Supersession | None:
        if input_epoch is None:
            return None
        return self._state(conn_id).pending_supersessions.get(input_epoch)

    def take_pending_supersession(
        self,
        conn_id: str,
        input_epoch: int | None,
    ) -> Supersession | None:
        st = self._state(conn_id)
        if input_epoch is None:
            return None
        return st.pending_supersessions.pop(input_epoch, None)

    def take_pending_transport_cancellation(self, conn_id: str, input_epoch: int | None) -> Any:
        if input_epoch is None:
            return None
        return self._state(conn_id).pending_transport_cancellations.pop(input_epoch, None)

    def close_rejected_input_capture(self, conn_id: str, *, input_epoch: int | None) -> bool:
        """Release the capture gate for one exact phantom/discarded VAD pair.

        A delayed stop from an older segment must not clear a newer microphone
        capture. Valid accepted stops use ``claim_pending_response`` instead,
        which clears this same gate while installing response ownership.
        """

        if input_epoch is None:
            return False
        st = self._state(conn_id)
        with st.response_ownership.transaction():
            if input_epoch != st.response_ownership.input_epoch:
                return False
            if not st.input_capture_pending:
                return False
            st.input_capture_pending = False
            st.pending_supersessions.pop(input_epoch, None)
            st.pending_transport_cancellations.pop(input_epoch, None)
            return True

    def claim_pending_response(
        self,
        conn_id: str,
        *,
        turn_id: str | None,
        turn_revision: int | None,
        input_epoch: int | None = None,
        response: RealtimeResponseCreateParams | None = None,
    ) -> ResponseOwner:
        st = self._state(conn_id)
        # Claim and install the rollback/frozen-routing boundary atomically.
        # input_started() must see either no owner or a fully initialized owner;
        # otherwise it can invalidate the epoch before its checkpoint exists and
        # leave stale configuration/history behind.
        with st.response_ownership.transaction():
            accepted_capture = bool(
                turn_id is not None
                and st.input_capture_pending
                and (input_epoch is None or input_epoch == st.response_ownership.input_epoch)
            )
            # A delayed stop for an older, retained audible response must not
            # release a newer microphone capture gate.  Epoch-aware producers
            # clear only their exact capture; legacy callers retain the old
            # behavior because they cannot name a competing capture.
            if turn_id is not None and (
                input_epoch is None or input_epoch == st.response_ownership.input_epoch
            ):
                st.input_capture_pending = False
            current = st.response_ownership.active()
            existing = st.response_ownership.owner_for_turn(turn_id, turn_revision)
            if existing is not None and existing.state not in {"completed", "cancelled"}:
                return existing
            if (
                current is not None
                and current.state not in {"completed", "cancelled"}
                and current.turn_id == turn_id
                and current.turn_revision == turn_revision
            ):
                return current
            queue_behind_audible = bool(
                current is not None
                and current.audible
                and current.state not in {"completed", "cancelled"}
                and current.input_epoch < st.response_ownership.input_epoch
            )
            owner = st.response_ownership.claim_pending(
                turn_id=turn_id,
                turn_revision=turn_revision,
                activate=not queue_behind_audible,
            )
            st.provisional_chat_checkpoints[owner.response_epoch] = st.runtime_config.chat.checkpoint()
            output_sample_rate = self._configured_output_sample_rate(st, response=response)
            st.response_output_sample_rates[owner.response_epoch] = output_sample_rate
            if not queue_behind_audible:
                st.response_output_sample_rate = output_sample_rate
            st.runtime_config.response_synthesis_configs[owner.response_epoch] = (
                self._freeze_response_synthesis_config(st, response=response)
            )
            st.runtime_config.local_pipeline.update(
                {
                    "_input_epoch": owner.input_epoch,
                    "_response_epoch": owner.response_epoch,
                    "_response_id": owner.response_id,
                }
            )
            if not queue_behind_audible:
                st.response_pending = True
            if (
                accepted_capture
                and st.tool_followup_requested
                and st.tool_followup_ready
                and not st.pending_tool_call_ids
            ):
                # The browser already supplied response.create for the tool
                # transaction, but confirmed speech B arrived before that
                # continuation could start. Its call/output pair is committed;
                # consume only the stale delimiter so B remains the one model
                # request and the external tool is never replayed.
                st.tool_followup_requested = False
                st.tool_followup_response = None
                st.tool_followup_ready = False
                st.tool_followup_started = False
                logger.info(
                    "Consumed queued tool follow-up at accepted speech claim "
                    "input_epoch=%d response_epoch=%d",
                    owner.input_epoch,
                    owner.response_epoch,
                )
        logger.debug(
            "Claimed %s response input_epoch=%d response_epoch=%d turn=%s rev=%s",
            "queued" if queue_behind_audible else "pending",
            owner.input_epoch,
            owner.response_epoch,
            turn_id,
            turn_revision,
        )
        return owner

    def activate_next_queued_response(self, conn_id: str) -> ResponseOwner | None:
        """Promote one accepted non-interrupting turn after the audible owner ends."""
        st = self._state(conn_id)
        if st.pending_tool_call_ids:
            logger.info(
                "Queued response promotion waiting for tool outputs (pending=%d)",
                len(st.pending_tool_call_ids),
            )
            return None
        # `promote_next` wakes the queued model worker. Hold the same re-entrant
        # ownership lock until its rollback checkpoint and frozen routing state
        # are installed, so the worker cannot commit history against the old
        # claim-time checkpoint in between promotion and refresh.
        with st.response_ownership.transaction():
            owner = st.response_ownership.promote_next()
            if owner is None:
                return None
            # Queued model work has not mutated chat yet.  Refresh its rollback
            # boundary at promotion so a later pre-audible supersession cannot
            # restore the older claim-time snapshot and erase assistant/tool
            # history committed by the audible owner while this turn waited.
            st.provisional_chat_checkpoints[owner.response_epoch] = st.runtime_config.chat.checkpoint()
            st.response_output_sample_rate = st.response_output_sample_rates.get(owner.response_epoch)
            st.runtime_config.local_pipeline.update(
                {
                    "_input_epoch": owner.input_epoch,
                    "_response_epoch": owner.response_epoch,
                    "_response_id": owner.response_id,
                }
            )
            st.response_pending = True
            st.deferred_settlement_events.append(self._owner_event(conn_id, owner))
            deferred_transcription = st.deferred_transcriptions.pop(owner.response_epoch, None)
            if deferred_transcription is not None:
                admitted_events = self._dispatch_pipeline_event(
                    conn_id,
                    deferred_transcription,
                    wait_for_pending_reopen=True,
                )
                if admitted_events:
                    st.deferred_settlement_events.extend(admitted_events)
        logger.info(
            "Promoted queued response input_epoch=%d response_epoch=%d turn=%s rev=%s",
            owner.input_epoch,
            owner.response_epoch,
            owner.turn_id,
            owner.turn_revision,
        )
        return owner

    def promote_queued_after_tool_transaction(
        self, conn_id: str, *, drain_events: bool = True,
    ) -> list[ServerEvent]:
        """Let the newest accepted speech answer replace a separate tool follow-up.

        The originating function-call/output pair is committed first so Chat
        remains structurally valid.  The already-admitted speech response then
        becomes the single successor; a browser response.create for the old tool
        transaction is consumed, never replayed as another answer.
        """

        st = self._state(conn_id)
        if st.in_response or st.pending_tool_call_ids or not st.response_ownership.has_queued():
            return []
        if st.tool_followup_requested:
            st.tool_followup_requested = False
            st.tool_followup_response = None
        else:
            st.suppress_next_tool_followup_create = True
        st.tool_followup_ready = False
        st.tool_followup_started = False
        owner = self.activate_next_queued_response(conn_id)
        if owner is None:
            return []
        logger.info(
            "Promoted accepted speech epoch=%d after tool output; separate tool follow-up suppressed",
            owner.response_epoch,
        )
        return self.take_deferred_settlement_events(conn_id) if drain_events else []

    @staticmethod
    def _freeze_response_synthesis_config(
        st: ConnState,
        *,
        response: RealtimeResponseCreateParams | None = None,
    ) -> dict[str, Any]:
        """Copy only response-scoped TTS inputs at ownership admission.

        RuntimeConfig remains shared for VAD/LLM/session behavior.  TTS must
        instead consume this tiny immutable value set so an acknowledged
        provider/profile/voice change can apply to the next answer without
        retargeting an answer that has not emitted its first phrase. Assistant
        language is inferred for each answer after admission and is frozen from
        its first authoritative TTS input, never from prior-turn runtime residue.
        """
        local_pipeline = st.runtime_config.local_pipeline
        selected: dict[str, Any] = {}
        for key in (
            "tts_backend",
            "tts_tuning",
            "tts_model_epoch",
            "candidate_model_epoch",
        ):
            if key in local_pipeline:
                selected[key] = deepcopy(local_pipeline[key])
        response_audio = response.audio if response is not None else None
        response_output = response_audio.output if response_audio is not None else None
        audio = st.runtime_config.session.audio
        session_output = audio.output if audio is not None else None
        # Response-level audio settings are fieldwise overrides. A response may
        # choose only a format/rate and must still inherit the session's selected
        # clone; choosing the whole response output object would freeze ``None``
        # and silently fall back to the handler's unrelated default voice.
        response_voice = getattr(response_output, "voice", None) if response_output is not None else None
        session_voice = getattr(session_output, "voice", None) if session_output is not None else None
        voice_value = response_voice or session_voice
        voice = str(voice_value) if voice_value else None
        playback_policy = deepcopy(local_pipeline.get("playback_policy"))
        if not isinstance(playback_policy, dict):
            playback_policy = {
                "prime_target_ms": 0,
                "continuity_mode": "fast-start",
                "native_streaming": False,
                "max_prime_ms": 2000,
            }
        return {
            "local_pipeline": selected,
            "voice": voice,
            "playback_policy": playback_policy,
        }

    @staticmethod
    def _release_response_synthesis_config(st: ConnState, response_epoch: int | None) -> None:
        if response_epoch is not None:
            st.runtime_config.response_synthesis_configs.pop(response_epoch, None)

    def claim_manual_response(
        self,
        conn_id: str,
        response: RealtimeResponseCreateParams | None = None,
    ) -> ResponseOwner:
        st = self._state(conn_id)
        owner = st.response_ownership.active()
        if owner is not None and owner.state not in {"completed", "cancelled"}:
            return owner
        if owner is not None and owner.state == "completed" and not owner.audible:
            # ``response.done`` only closes the transport.  Until the browser
            # acknowledges a rendered sample, the prior answer remains a
            # provisional history transaction.  A manual follow-up replaces
            # that unanswered response, so roll it back *before* claiming the
            # new owner/checkpoint.  Settled tool-only and heard responses have
            # no checkpoint and are intentionally untouched.
            self.rollback_provisional_response(
                conn_id,
                owner.response_epoch,
                reason="manual_response_replaced_unheard",
            )
            self._release_response_synthesis_config(st, owner.response_epoch)
            st.response_output_sample_rates.pop(owner.response_epoch, None)
        return self.claim_pending_response(
            conn_id,
            turn_id=None,
            turn_revision=None,
            response=response,
        )

    @staticmethod
    def _configured_output_sample_rate(
        st: ConnState,
        *,
        response: RealtimeResponseCreateParams | None = None,
    ) -> int:
        """Return the response-admission transport clock from current settings.

        This is intentionally evaluated once at ownership claim. The audio
        handler may still choose its source-rate fallback for legacy implicit
        responses that bypass the normal pipeline, but a pending runtime
        response must never re-read mutable local-pipeline configuration.
        """
        response_audio = response.audio if response is not None else None
        response_output = response_audio.output if response_audio is not None else None
        response_format = response_output.format if response_output is not None else None
        response_rate = getattr(response_format, "rate", None)
        if isinstance(response_rate, int) and not isinstance(response_rate, bool) and response_rate > 0:
            return response_rate
        local_pipeline = getattr(st.runtime_config, "local_pipeline", None)
        negotiated = local_pipeline.get("audio_output_sample_rate") if isinstance(local_pipeline, dict) else None
        if isinstance(negotiated, int) and not isinstance(negotiated, bool) and negotiated > 0:
            return negotiated
        audio_cfg = st.runtime_config.session.audio
        if audio_cfg is not None and audio_cfg.output is not None:
            rate = getattr(audio_cfg.output.format, "rate", None)
            if isinstance(rate, int) and not isinstance(rate, bool) and rate > 0:
                return rate
        return PIPELINE_SAMPLE_RATE

    def bind_response_id(self, conn_id: str, response_id: str) -> ResponseOwner | None:
        st = self._state(conn_id)
        owner = st.response_ownership.active()
        if owner is None:
            return None
        updated = st.response_ownership.transition(owner.response_epoch, "priming", response_id=response_id)
        st.runtime_config.local_pipeline["_response_id"] = response_id
        return updated

    def rollback_provisional_response(self, conn_id: str, response_epoch: int, *, reason: str) -> bool:
        """Remove only the unheard response transaction for *response_epoch*."""
        st = self._state(conn_id)
        # Pair rollback with the direct-Gemma admission/Chat.add_item guard.
        # Once VAD marks the owner stale under this lock, a late worker sees
        # rejection rather than adding history after its checkpoint restores.
        with st.response_ownership.transaction():
            checkpoint = st.provisional_chat_checkpoints.pop(response_epoch, None)
            if checkpoint is None:
                return False
            rolled_back = st.runtime_config.chat.rollback(checkpoint)
        logger.info(
            "Provisional response epoch=%d rollback=%s reason=%s",
            response_epoch,
            rolled_back,
            reason,
        )
        return rolled_back

    def settle_provisional_response(self, conn_id: str, response_epoch: int) -> bool:
        """Commit a delivered non-audio or browser-audible response transaction."""
        return self._state(conn_id).provisional_chat_checkpoints.pop(response_epoch, None) is not None

    def set_rendered_playback_ack_supported(self, conn_id: str, supported: bool) -> None:
        """Select the audible boundary before the first response PCM is sent.

        Browser clients opt in and acknowledge the worklet's first rendered
        sample.  Clients that do not opt in remain compatible with the public
        Realtime contract, where first delivered PCM is the best available
        evidence that output became audible.
        """
        self._state(conn_id).rendered_playback_ack_supported = bool(supported)

    def response_owner_event(self, conn_id: str) -> PipelineResponseServerEvent | None:
        """Return the current owner as a local protocol extension.

        The browser uses this to associate its first rendered sample with the
        epoch that produced the response.  It is intentionally separate from
        OpenAI Realtime events, whose schema has no response epoch field.
        """
        st = self._state(conn_id)
        if not st.rendered_playback_ack_supported:
            return None
        owner = st.response_ownership.active()
        return self._owner_event(conn_id, owner) if owner is not None else None

    def mark_response_producing(self, conn_id: str) -> ResponseOwner | None:
        st = self._state(conn_id)
        owner = st.response_ownership.active()
        return st.response_ownership.transition(owner.response_epoch, "producing") if owner else None

    def mark_response_completed(self, conn_id: str, *, reason: str | None = None) -> ResponseOwner | None:
        st = self._state(conn_id)
        owner = st.response_ownership.active()
        completed = st.response_ownership.complete(owner.response_epoch, reason=reason) if owner else None
        if completed is not None:
            self._release_response_synthesis_config(st, completed.response_epoch)
            st.response_output_sample_rates.pop(completed.response_epoch, None)
        return completed

    @staticmethod
    def _maybe_schedule_history_maintenance(st: ConnState) -> None:
        # Compaction must never snapshot a current/provisional response.  The
        # completed owner can remain current only for playback acknowledgement;
        # that lifecycle state is safe once protocol output itself has settled.
        if st.in_response or st.response_pending or st.input_capture_pending or st.pending_tool_call_ids:
            return
        if not st.runtime_config.local_pipeline.get("_history_maintenance_needed", False):
            return
        scheduler = st.runtime_config.local_pipeline.get("_schedule_history_maintenance")
        if callable(scheduler) and scheduler(st.runtime_config):
            st.runtime_config.local_pipeline.pop("_history_maintenance_needed", None)

    def maybe_schedule_history_maintenance(self, conn_id: str) -> None:
        self._maybe_schedule_history_maintenance(self._state(conn_id))

    def mark_response_cancelled(self, conn_id: str, *, reason: str | None = None) -> ResponseOwner | None:
        st = self._state(conn_id)
        # Mark output inadmissible before rolling history back, under the same
        # RLock used by every response-epoch Chat mutation. A model worker can
        # therefore observe either the old live transaction or the cancelled
        # rolled-back one, never the vulnerable state between them.
        with st.response_ownership.transaction():
            owner = st.response_ownership.active()
            if owner is None:
                return None
            cancelled = st.response_ownership.transition(owner.response_epoch, "cancelled", reason=reason)
            if not owner.audible:
                self.rollback_provisional_response(conn_id, owner.response_epoch, reason=reason or "cancelled")
            self._release_response_synthesis_config(st, owner.response_epoch)
            st.response_output_sample_rates.pop(owner.response_epoch, None)
            # Direct/Gemma work can be cancelled before a protocol response exists.
            # Its frozen transport belongs to that abandoned owner, not the next
            # accepted turn.
            if st.current_response_id is None and not st.in_response:
                st.response_output_sample_rate = None
        return cancelled

    def handle_playback_started(
        self, conn_id: str, *, response_id: str, response_epoch: int
    ) -> PipelineResponseServerEvent | None:
        st = self._state(conn_id)
        tracker = st.response_ownership
        # Audible admission, checkpoint settlement, deferred history, and any
        # tool-continuation release are one transaction. A new VAD epoch cannot
        # checkpoint a successor in the middle and later lose valid audible
        # history when that successor is rolled back.
        with tracker.transaction():
            before = tracker.active()
            owner = tracker.playback_started(
                response_id=response_id,
                response_epoch=response_epoch,
            )
            if owner is None:
                logger.debug(
                    "Ignoring stale playback acknowledgement response=%s epoch=%d",
                    response_id,
                    response_epoch,
                )
                return None
            if before is not None and before.response_epoch == response_epoch and before.playback_started:
                # The browser can retry after a reconnect race. The state
                # change is idempotent, and the local extension is exact-once.
                return None
            # Heard output commits the transaction. A completed transport
            # without this acknowledgement remains rollbackable.
            st.provisional_chat_checkpoints.pop(response_epoch, None)
            # The worklet can render the first sample while the model worker is
            # still committing the response's function call / assistant item to
            # Chat.  Audibility makes the provisional checkpoint durable, but
            # client items must remain deferred until that write-back reaches
            # the protocol terminal.  If transport already finished, settle
            # them here; otherwise finish_response() performs this exact drain.
            if not st.in_response:
                st.deferred_settlement_events.extend(self.conversation.flush_deferred_items(conn_id))
                followup = self.response.start_queued_tool_followup_if_ready(conn_id)
                if followup is not None:
                    # Conversation acknowledgements precede the new response,
                    # matching hosted tool-output/response.create ordering once.
                    st.deferred_settlement_events.append(followup)
            boundary = "browser render" if st.rendered_playback_ack_supported else "delivered PCM"
            logger.info("Playback acknowledged at %s boundary response epoch=%d", boundary, response_epoch)
            self._maybe_schedule_history_maintenance(st)
            return self._owner_event(conn_id, owner)

    def handle_first_delivered_pcm(self, conn_id: str) -> list[ServerEvent]:
        """Commit a nonbrowser response only after its first PCM delta was sent.

        Encoding is not delivery: the WebSocket can close or reject a send
        after audio conversion succeeds. Browser clients opt into the stronger
        worklet-render acknowledgement and therefore never use this fallback.
        """
        st = self._state(conn_id)
        if st.rendered_playback_ack_supported or not st.response_audio_emitted:
            return []
        owner = st.response_ownership.active()
        response_id = st.current_response_id
        if owner is None or owner.playback_started or response_id is None:
            return []
        self.handle_playback_started(
            conn_id,
            response_id=response_id,
            response_epoch=owner.response_epoch,
        )
        # Browser-local lifecycle extensions are filtered here; standard
        # conversation acknowledgements and a queued tool follow-up remain.
        return self.take_deferred_settlement_events(conn_id)

    def take_deferred_settlement_events(self, conn_id: str) -> list[ServerEvent]:
        st = self._state(conn_id)
        # Producers run on VAD/model threads while the websocket router drains
        # asynchronously.  Copy-and-clear must share the ownership lock with
        # every producer or an append between those two list operations can be
        # erased without ever reaching the client.
        with st.response_ownership.transaction():
            events = list(st.deferred_settlement_events)
            st.deferred_settlement_events.clear()
        if not st.rendered_playback_ack_supported:
            # ``pipeline.response`` is a browser-local diagnostics extension,
            # not part of the OpenAI Realtime SDK schema. Standard clients keep
            # their original event stream and first-valid-PCM audible boundary.
            events = [event for event in events if not isinstance(event, PipelineResponseServerEvent)]
        return events

    def queue_deferred_settlement_events(
        self,
        conn_id: str,
        events: ServerEvent | list[ServerEvent],
    ) -> None:
        """Atomically append one or more deferred protocol settlement events."""

        st = self._state(conn_id)
        queued = events if isinstance(events, list) else [events]
        if not queued:
            return
        with st.response_ownership.transaction():
            st.deferred_settlement_events.extend(queued)

    def response_owner_for_turn(
        self, conn_id: str, turn_id: str | None, turn_revision: int | None
    ) -> ResponseOwner | None:
        return self._state(conn_id).response_ownership.owner_for_turn(turn_id, turn_revision)

    # ── Connection lifecycle ─────────────────────

    def register(self) -> str:
        """Register a new connection and return its session_id."""
        if self.speculative_turns:
            self.speculative_turns.reset()
        chat = Chat(self._chat_size)
        if self._direct_audio_session:
            # Direct turns may arrive before their final transcription metadata.
            # Do not let the legacy doubled turn cap drop that truthful context
            # while waiting for the first direct response to commit it.
            chat.enable_token_managed_history()
        state = ConnState(
            runtime_config=RuntimeConfig(
                chat=chat,
                model_endpoint=self.default_model_endpoint.model_copy(deep=True),
            )
        )
        state.runtime_config.local_pipeline["_session_id"] = state.session_id
        state.runtime_config.local_pipeline["_history_maintenance_admissible"] = (
            lambda session_id=state.session_id: (
                (current := self._conns.get(session_id)) is not None
                and not current.runtime_config.local_pipeline.get("_history_closed")
                and not current.in_response
                and not current.response_pending
                and not current.input_capture_pending
                and not current.pending_tool_call_ids
            )
        )
        if self._history_maintenance is not None:
            state.runtime_config.local_pipeline["_schedule_history_maintenance"] = self._history_maintenance.schedule
        self._conns[state.session_id] = state
        state.runtime_config.local_pipeline["_claim_response_owner"] = (
            lambda turn_id, turn_revision, session_id=state.session_id: self.claim_pending_response(
                session_id,
                turn_id=turn_id,
                turn_revision=turn_revision,
            )
        )
        state.runtime_config.local_pipeline["_observe_speech_started"] = (
            lambda reason, interrupt_response=True, session_id=state.session_id: self.observe_speech_started(
                session_id,
                reason=reason,
                interrupt_response=interrupt_response,
            )
        )
        state.runtime_config.local_pipeline["_response_epoch_is_current"] = (
            lambda response_epoch, session_id=state.session_id: self.is_response_epoch_current(
                session_id,
                response_epoch,
            )
        )
        state.runtime_config.local_pipeline["_response_epoch_admits_output"] = (
            lambda response_epoch, session_id=state.session_id: self.is_response_epoch_output_admissible(
                session_id,
                response_epoch,
            )
        )
        state.runtime_config.local_pipeline["_response_epoch_history_transaction"] = (
            lambda response_epoch, tracker=state.response_ownership: tracker.history_transaction(response_epoch)
        )
        state.runtime_config.local_pipeline["_wait_response_epoch_current"] = (
            lambda response_epoch, tracker=state.response_ownership: tracker.wait_until_current(response_epoch)
        )
        self.total_usage.connections += 1
        return state.session_id

    def unregister(self, conn_id: str) -> None:
        st = self._conns.pop(conn_id, None)
        if st is not None:
            with st.runtime_config.history_maintenance_lock:
                st.runtime_config.local_pipeline["_history_closed"] = True
                st.response_ownership.close()
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

    def is_response_epoch_current(self, conn_id: str, response_epoch: int | None) -> bool:
        st = self._conns.get(conn_id)
        return st is not None and st.response_ownership.is_current(response_epoch)

    def is_response_epoch_output_admissible(self, conn_id: str, response_epoch: int | None) -> bool:
        """Return whether text, TTS, PCM, or terminal output may still escape.

        A completed response remains ``current`` briefly so a browser can send
        its late first-render acknowledgement.  That compatibility window must
        not reopen model or TTS output after the response is terminal.
        """

        st = self._conns.get(conn_id)
        return st is not None and st.response_ownership.admits_output(response_epoch)

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

    def encode_audio_chunk(
        self,
        conn_id: str,
        audio: bytes,
        *,
        source_sample_rate: int = 16000,
        output_sample_rate: int | None = None,
        input_epoch: int | None = None,
        response_epoch: int | None = None,
        response_id: str | None = None,
    ) -> list[ServerEvent]:
        """Encode provider PCM without inferring ownership from mutable session state."""

        return self.audio.encode_audio_chunk(
            conn_id,
            audio,
            source_sample_rate=source_sample_rate,
            output_sample_rate=output_sample_rate,
            input_epoch=input_epoch,
            response_epoch=response_epoch,
            response_id=response_id,
        )

    def handle_response_create(self, conn_id: str, event: ResponseCreateEvent) -> ServerEvent | None:
        return self.response.handle_response_create(conn_id, event)

    def handle_response_cancel(self, conn_id: str) -> list[ServerEvent]:
        return self.response.handle_response_cancel(conn_id)

    def handle_response_cancel_if_owner(
        self,
        conn_id: str,
        *,
        input_epoch: int | None,
        response_epoch: int | None,
        response_id: str | None,
    ) -> list[ServerEvent] | None:
        return self.response.handle_response_cancel_if_owner(
            conn_id,
            input_epoch=input_epoch,
            response_epoch=response_epoch,
            response_id=response_id,
        )

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
        response_epoch_was_explicit = getattr(event, "response_epoch", None) is not None
        if not self._bind_event_response_epoch(conn_id, event):
            logger.info(
                "Ignoring response-scoped %s metric without an unambiguous response identity",
                event.stage,
            )
            return []
        response_epoch = getattr(event, "response_epoch", None)
        if (
            isinstance(event, TranscriptionCompletedEvent)
            and response_epoch is not None
            and self._state(conn_id).response_ownership.is_queued(response_epoch)
        ):
            # Do not block this output-loop thread on the ownership condition:
            # it must keep sending the audible predecessor's terminal events,
            # which are what eventually promote this exact queued epoch.
            self._state(conn_id).deferred_transcriptions[response_epoch] = event
            logger.info(
                "Deferred final transcription for queued response epoch=%d turn=%s rev=%s",
                response_epoch,
                event.turn_id,
                event.turn_revision,
            )
            return []
        is_stale = self._is_stale_turn_event(
            conn_id,
            event,
            wait_for_pending_reopen=wait_for_pending_reopen,
            response_epoch_is_authoritative=response_epoch_was_explicit,
        )
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

        st = self._state(conn_id)

        def dispatch_admitted() -> list[ServerEvent] | None:
            self._observe_turn_event(event)
            if isinstance(event, AssistantTextEvent):
                return self.response.on_assistant_text(
                    conn_id,
                    event,
                    wait_for_pending_reopen=wait_for_pending_reopen,
                    response_epoch_is_authoritative=response_epoch_was_explicit,
                )
            handler = self._pipeline_dispatch.get(type(event))
            if handler is None:
                logger.debug("Unhandled pipeline event type: %s", type(event).__name__)
                return []
            return handler(conn_id, event)

        # Admission plus every response-bound side effect is one ownership
        # transaction. VAD may replace an owner on another thread immediately
        # after the outer stale check; without this second compare an old text,
        # usage, failure, or terminal-barrier event can mutate the newly
        # claimed response.
        if response_epoch is not None:
            with st.response_ownership.transaction():
                if isinstance(event, PipelineMetricEvent) and event.authoritative_terminal:
                    owner = st.response_ownership.owner_for_response_epoch(response_epoch)
                    admitted = bool(
                        owner is not None
                        and owner.state == "cancelled"
                        and getattr(event, "input_epoch", None) == owner.input_epoch
                        and (
                            getattr(event, "response_id", None) is None
                            or getattr(event, "response_id", None) == owner.response_id
                        )
                    )
                else:
                    admitted = st.response_ownership.matches_active_identity(
                        input_epoch=getattr(event, "input_epoch", None),
                        response_epoch=response_epoch,
                        response_id=getattr(event, "response_id", None),
                    )
                if not admitted:
                    logger.info(
                        "Ignoring response-bound %s after owner changed (epoch=%s)",
                        event.type,
                        response_epoch,
                    )
                    return []
                return dispatch_admitted()
        return dispatch_admitted()

    def _bind_event_response_epoch(self, conn_id: str, event: PipelineEvent) -> bool:
        """Attach the owner selected at speech-stop to later pipeline events.

        Pipeline message propagation remains backwards-compatible: handlers
        without epoch fields are resolved through their turn/revision key at
        the service boundary before any protocol side effect occurs.
        """
        if isinstance(event, (SpeechStartedEvent, SpeechStoppedEvent)):
            return True
        if not hasattr(event, "response_epoch"):
            return True
        explicit_epoch = getattr(event, "response_epoch", None)
        if explicit_epoch is not None:
            # Older in-process producers may have learned the response epoch
            # before a concrete response id existed. Complete the immutable
            # tuple from that same owner, never from the mutable active owner.
            owner = self._state(conn_id).response_ownership.owner_for_response_epoch(explicit_epoch)
            if owner is not None:
                if getattr(event, "input_epoch", None) is None:
                    event.input_epoch = owner.input_epoch
                if hasattr(event, "response_id") and getattr(event, "response_id", None) is None:
                    event.response_id = owner.response_id
            return True
        turn_id = getattr(event, "turn_id", None)
        turn_revision = getattr(event, "turn_revision", None)
        if (
            isinstance(event, PipelineMetricEvent)
            and event.stage in _RESPONSE_SCOPED_METRIC_STAGES
            and not event.authoritative_terminal
            and turn_id is None
            and turn_revision is None
        ):
            # Manual and out-of-band turns deliberately use a ``(None, None)``
            # turn key.  Looking that key up after a replacement response has
            # claimed ownership can silently relabel a late metric as belonging
            # to the new response.  Response-scoped producers must therefore
            # carry an epoch (or a concrete turn key) rather than relying on the
            # compatibility binder.
            return False
        owner = self.response_owner_for_turn(
            conn_id,
            turn_id,
            turn_revision,
        )
        if owner is not None:
            event.response_epoch = owner.response_epoch
            event.input_epoch = owner.input_epoch
            if hasattr(event, "response_id") and getattr(event, "response_id", None) is None:
                event.response_id = owner.response_id
        return True

    def _is_stale_turn_event(
        self,
        conn_id: str,
        event: PipelineEvent,
        *,
        wait_for_pending_reopen: bool = True,
        response_epoch_is_authoritative: bool = False,
    ) -> bool | None:
        response_epoch = getattr(event, "response_epoch", None)
        if response_epoch is not None:
            ownership = self._state(conn_id).response_ownership
            if not ownership.admits_output(response_epoch):
                owner = ownership.owner_for_response_epoch(response_epoch)
                if (
                    isinstance(event, PipelineMetricEvent)
                    and event.authoritative_terminal
                    and owner is not None
                    and owner.state == "cancelled"
                    and getattr(event, "input_epoch", None) == owner.input_epoch
                    and (
                        getattr(event, "response_id", None) is None
                        or getattr(event, "response_id", None) == owner.response_id
                    )
                ):
                    # Router-owned cancellation may describe a just-superseded
                    # response. A second terminal for an already-completed
                    # response is still stale and must not re-enter metrics.
                    return False
                return True
            # A response epoch is the authoritative output owner.  A newer
            # accepted input may already exist while an audible response is
            # deliberately allowed to finish (interrupt_response=false).  In
            # that case the speculative turn tracker quite correctly points at
            # the newer input, but it must not revoke output ownership from the
            # still-active audible response.
            if response_epoch_is_authoritative:
                return False
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
                ResponseFailedEvent,
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
        with response_epoch_history_transaction(
            runtime_config=st.runtime_config,
            response_epoch=event.response_epoch,
        ) as admitted:
            if not admitted:
                logger.info(
                    "Dropping superseded final transcription before history commit turn=%s rev=%s epoch=%s",
                    event.turn_id,
                    event.turn_revision,
                    event.response_epoch,
                )
                return []
            return self._on_admitted_transcription_completed(conn_id, event)

    def _on_admitted_transcription_completed(
        self,
        conn_id: str,
        event: TranscriptionCompletedEvent,
    ) -> list[ServerEvent]:
        """Commit one transcription while its response epoch owns the history lock."""
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
            producing = self.mark_response_producing(conn_id)
            if producing is not None:
                self.queue_deferred_settlement_events(conn_id, self._owner_event(conn_id, producing))
        elif queue and context_transcript:
            producing = self.mark_response_producing(conn_id)
            if producing is not None:
                self.queue_deferred_settlement_events(conn_id, self._owner_event(conn_id, producing))
            owner = self.response_owner_for_turn(conn_id, event.turn_id, event.turn_revision)
            queue.put(
                GenerateResponseRequest(
                    runtime_config=cfg,
                    language_code=event.language_code,
                    turn_id=event.turn_id,
                    turn_revision=event.turn_revision,
                    speech_stopped_at_s=event.speech_stopped_at_s,
                    input_epoch=owner.input_epoch if owner else None,
                    response_epoch=owner.response_epoch if owner else None,
                    response_id=owner.response_id if owner else None,
                )
            )
        else:
            # A VAD turn still claims ownership before model admission, but an
            # empty/non-contextual final transcript must not leave that owner
            # pending and block the next accepted turn.
            st.response_pending = False
            cancelled = self.mark_response_cancelled(conn_id, reason="empty_transcript")
            if cancelled is not None:
                st.provisional_chat_checkpoints.pop(cancelled.response_epoch, None)
                self.queue_deferred_settlement_events(conn_id, self._owner_event(conn_id, cancelled))
            events.extend(self.conversation.flush_deferred_items(conn_id))

        return events

    # ── Metrics ────────────────────────────────────

    def _on_token_usage(self, conn_id: str, event: TokenUsageEvent) -> list[ServerEvent]:
        """Accumulate usage after the service's authoritative admission gate.

        ``_dispatch_pipeline_event`` has already resolved response-epoch
        ownership before routing here.  Re-checking only speculative turn
        identity would incorrectly drop usage for an audible response that is
        deliberately finishing while newer, non-interrupting speech is queued.
        """
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
        st = self._state(conn_id)
        if not st.in_response:
            owner = st.response_ownership.active()
            if owner is None:
                return []
            self.mark_response_cancelled(conn_id, reason="failed")
            events: list[ServerEvent] = [self.make_error(event.message, "response_failed")]
            cancelled = st.response_ownership.active()
            if cancelled is not None:
                self.queue_deferred_settlement_events(conn_id, self._owner_event(conn_id, cancelled))
            events.extend(self.conversation.flush_deferred_items(conn_id))
            return events
        events: list[ServerEvent] = [self.make_error(event.message, "response_failed")]
        events.extend(self.response.finish_response(conn_id, status="failed"))
        return events

    def _on_response_output_complete(
        self, conn_id: str, event: ResponseOutputCompleteEvent
    ) -> list[ServerEvent]:
        st = self._state(conn_id)
        if event.response_epoch is not None and not st.response_ownership.admits_output(event.response_epoch):
            return []
        st.text_output_complete = True
        if self._state(conn_id).runtime_config.model_endpoint.context_window is not None:
            return [self.context_metric(conn_id, "committed")]
        return []

    def _on_pipeline_metric(self, conn_id: str, event: PipelineMetricEvent) -> list[ServerEvent]:
        metric = PipelineMetricServerEvent(
            event_id=self._next_event_id(),
            stage=event.stage,
            status=event.status,
            at_s=event.at_s,
            elapsed_ms=event.elapsed_ms,
            turn_id=event.turn_id,
            turn_revision=event.turn_revision,
            input_epoch=event.input_epoch,
            response_epoch=event.response_epoch,
            response_id=event.response_id,
            detail=event.detail,
        )
        if event.stage != "tts" or event.status not in {"runaway_aborted", "failed"}:
            return [metric]

        # A TTS limit/failure is terminal only for the exact output owner that
        # reported it.  The dispatcher has already admitted response-scoped
        # events, but retain this local identity check so direct callers and
        # future routing changes cannot clear a successor's PCM clock.
        st = self._state(conn_id)
        owner = st.response_ownership.active()
        if (
            owner is None
            or event.response_epoch is None
            or owner.response_epoch != event.response_epoch
            or not st.response_ownership.matches_active_identity(
                input_epoch=event.input_epoch,
                response_epoch=event.response_epoch,
                response_id=event.response_id,
            )
        ):
            return [metric]

        # ``finish_response`` clears only this response's protocol/output
        # state.  ``mark_response_cancelled`` rolls history back only when it
        # was never audible, preserving an interrupted audible answer exactly
        # as the normal barge-in lifecycle does.  Do not advance input epoch:
        # synthesis failure is not new user speech.
        terminal = self.response.finish_response(
            conn_id,
            status="failed",
            ownership_reason="synthesis-failed",
        )
        return [metric, *terminal]

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
