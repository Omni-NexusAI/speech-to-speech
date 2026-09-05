from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from threading import Lock, local
from time import perf_counter, time
from typing import Any, Optional

import httpx
from nltk import sent_tokenize
from openai import OpenAI
from openai.types.realtime.conversation_item import (
    RealtimeConversationItemAssistantMessage,
    RealtimeConversationItemFunctionCall,
)
from openai.types.realtime.realtime_conversation_item_assistant_message import (
    Content as AssistantContent,
)
from openai.types.responses import ResponseFunctionToolCall
from pydantic import BaseModel, ConfigDict, Field

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.LLM.chat import (
    Chat,
    ChatItemError,
    SupportedItem,
    build_active_chat,
    make_system_message,
    make_user_message,
)
from speech_to_speech.LLM.compaction_prompt import CompactGenerateFn, build_compactor
from speech_to_speech.LLM.text_prompt import build_text_system_prompt
from speech_to_speech.LLM.utils import remove_unspeechable, resolve_auto_language
from speech_to_speech.LLM.voice_prompt import build_voice_system_prompt
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.events import PipelineMetricEvent
from speech_to_speech.pipeline.handler_types import LLMIn, LLMOut
from speech_to_speech.pipeline.messages import (
    DirectAssistantRequest,
    EndOfResponse,
    LLMResponseChunk,
    TokenUsage,
)
from speech_to_speech.pipeline.model_operations import ModelOperationCoordinator, ModelOperationToken
from speech_to_speech.pipeline.response_ownership import (
    response_epoch_admission,
    response_epoch_history_transaction,
    response_output_allowed,
    wait_for_response_epoch_admission,
)
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.utils.utils import is_out_of_band, response_wants_audio

logger = logging.getLogger(__name__)


class _ModelOperationCancelled(RuntimeError):
    """Internal signal for cancellation captured before transport startup."""


# â”€â”€ Normalised provider events â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Each backend's stream/response is mapped to this small vocabulary so the shared
# speech-pipeline logic (sentence batching, cancellation, history, token usage)
# lives in one place. Subclasses differ only in how they produce these events.


class TextDelta(BaseModel):
    """Incremental assistant text. Always RAW (unfiltered); the base applies
    ``remove_unspeechable`` for the audio path."""

    text: str


class AssistantMessage(BaseModel):
    """A complete assistant turn to write back to history."""

    content: list[AssistantContent]


class ToolCall(BaseModel):
    """A complete function tool call (``call_id`` / ``id`` already regenerated)."""

    item: ResponseFunctionToolCall


class Usage(BaseModel):
    """Token accounting for the turn."""

    input_tokens: int
    output_tokens: int


ProviderEvent = TextDelta | AssistantMessage | ToolCall | Usage


class _Turn(BaseModel):
    """Per-request context threaded through generation (immutable for the turn)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    language_code: Optional[str]
    gen: int | None
    runtime_config: Any
    response: Any
    turn_id: str | None
    turn_revision: int | None
    speech_stopped_at_s: float | None
    wants_audio: bool
    input_epoch: int | None = None
    response_epoch: int | None = None
    response_id: str | None = None
    tool_calls_allowed: bool = True


class _GenState(BaseModel):
    """Mutable accumulators collected while consuming a turn's events."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tools: list[ResponseFunctionToolCall] = Field(default_factory=list)
    pending: list[SupportedItem] = Field(default_factory=list)
    clean_text: str = ""  # filtered text, kept only for the debug log
    input_tokens: int = 0
    output_tokens: int = 0


class BaseOpenAICompatibleHandler(BaseHandler[LLMIn, LLMOut], ABC):
    """Shared lifecycle for OpenAI-compatible LLM backends (Responses & Chat
    Completions).

    Subclasses implement four hooks â€” :meth:`warmup`,
    :meth:`_build_compaction_generate_fn`, :meth:`_serialize`, :meth:`_request`,
    :meth:`_iter_events` and :meth:`_build_optional_kwargs` â€” and inherit the
    request/response orchestration: speculative-turn gating, cancellation,
    sentence batching, text-only vs audio handling, history write-back, token
    usage, out-of-band handling and error termination.
    """

    # â”€â”€ setup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def setup(
        self,
        model_name: str = "gpt-5.4-mini",
        device: str = "cuda",
        gen_kwargs: dict[str, Any] = {},
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        stream: bool = True,
        user_role: str = "user",
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
        disable_thinking: bool = True,
        reasoning_effort: Optional[str] = None,
        request_timeout_s: float = 30.0,
        warmup: bool = False,
        stream_batch_sentences: int = 3,
        enable_lang_prompt: bool = False,
        compact_history: bool = False,
        model_operations: ModelOperationCoordinator | None = None,
        text_output_queue: Any | None = None,
        **_kwargs: Any,
    ) -> None:
        self.cancel_scope = cancel_scope
        self.model_operations = model_operations
        self.text_output_queue = text_output_queue
        self.speculative_turns = speculative_turns
        self.model_name = model_name
        self.stream = stream
        self.stream_batch_sentences = max(1, stream_batch_sentences)
        self.enable_lang_prompt = enable_lang_prompt
        self.gen_kwargs = dict(gen_kwargs)
        self.request_timeout_s = float(request_timeout_s)
        self.request_timeout = httpx.Timeout(
            self.request_timeout_s,
            connect=min(5.0, self.request_timeout_s),
        )

        self.user_role = user_role
        self.base_url = base_url
        self.api_key = api_key
        self.disable_thinking = disable_thinking
        self.reasoning_effort = reasoning_effort
        self.client = OpenAI(
            api_key=api_key or "not-needed",
            base_url=base_url,
            max_retries=0,
            timeout=self.request_timeout,
        )
        self._active_response: Any = None
        self._active_client: Any = None
        self._active_response_lock = Lock()
        self._operation_context = local()
        self._extra_body = self._build_extra_body(base_url, disable_thinking, reasoning_effort)
        self.compactor = build_compactor(self._build_compaction_generate_fn()) if compact_history else None
        if warmup:
            self.warmup()

    def _client_for(self, runtime_config: Any) -> tuple[OpenAI, str, Optional[dict[str, Any]]]:
        endpoint = getattr(runtime_config, "model_endpoint", None)
        if endpoint is None or endpoint.provider == "local":
            if getattr(self.client, "is_closed", False):
                self.client = OpenAI(
                    api_key=self.api_key or "not-needed",
                    base_url=self.base_url,
                    max_retries=0,
                    timeout=self.request_timeout,
                )
            return self.client, self.model_name, self._extra_body
        base_url = endpoint.base_url
        api_key = endpoint.api_key
        model_name = endpoint.model
        client = OpenAI(
            api_key=api_key or "not-needed",
            base_url=base_url,
            max_retries=0,
            timeout=self.request_timeout,
        )
        extra_body = self._build_extra_body(base_url, self.disable_thinking, self.reasoning_effort)
        return client, model_name, extra_body

    def _set_active_client(self, client: Any) -> bool:
        with self._active_response_lock:
            self._active_client = client
        context = getattr(getattr(self, "_operation_context", None), "value", None)
        if context is None:
            return True
        coordinator, operation = context
        bound = coordinator.bind_cancel(operation, self.cancel_active)
        cancellation_requested = getattr(coordinator, "cancellation_requested", None)
        cancelled = callable(cancellation_requested) and cancellation_requested(operation)
        if not bound or cancelled:
            self.cancel_active()
            return False
        return True

    def _model_operation_cancelled(self) -> bool:
        context = getattr(getattr(self, "_operation_context", None), "value", None)
        if context is None:
            return False
        coordinator, operation = context
        cancellation_requested = getattr(coordinator, "cancellation_requested", None)
        return bool(
            not coordinator.is_current(operation)
            or (callable(cancellation_requested) and cancellation_requested(operation))
        )

    @staticmethod
    def _is_official_openai(base_url: Optional[str]) -> bool:
        """Whether ``base_url`` points at the official OpenAI server.

        Normalises a trailing slash so ``https://api.openai.com/v1/`` is also
        recognised; the official server rejects the provider-specific extra_body
        keys we send to vLLM / the HF router.
        """
        if base_url is None:
            return False
        return base_url.rstrip("/") == "https://api.openai.com/v1"

    @classmethod
    def _build_extra_body(
        cls,
        base_url: Optional[str],
        disable_thinking: bool,
        reasoning_effort: Optional[str],
    ) -> Optional[dict[str, Any]]:
        """Build the provider-specific ``extra_body`` used to disable reasoning.

        Providers differ in how reasoning is turned off: vLLM/Qwen honour
        ``chat_template_kwargs.enable_thinking=false``, while others (e.g. GLM via
        the HF router) ignore that and require ``reasoning_effort='none'``. A
        non-empty ``reasoning_effort`` therefore takes precedence; otherwise we fall
        back to the chat-template flag. None of this applies to the official
        OpenAI server, which rejects unknown extra_body keys.
        """
        if base_url is None or cls._is_official_openai(base_url):
            return None
        if reasoning_effort:
            return {"reasoning_effort": reasoning_effort}
        if disable_thinking:
            return {"chat_template_kwargs": {"enable_thinking": False}}
        return None

    # â”€â”€ subclass hooks â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€--

    @abstractmethod
    def warmup(self) -> None:
        """Issue a cheap request so the model/connection is ready before serving."""
        ...

    @abstractmethod
    def _build_compaction_generate_fn(self) -> CompactGenerateFn:
        """Return a ``(system, user) -> text`` fn used to compact long histories."""
        ...

    @abstractmethod
    def _serialize(self, active_chat: Chat) -> Any:
        """Serialise the chat to the backend's request payload (input/messages)."""
        ...

    @abstractmethod
    def _request(self, api_input: Any, optional_kwargs: dict[str, Any], runtime_config: Any) -> Any:
        """Issue the create() call and return the response or stream."""
        ...

    @abstractmethod
    def _iter_stream_events(self, api_response: Any) -> Iterator[ProviderEvent]:
        """Map a streaming response to normalised :data:`ProviderEvent`s."""
        ...

    @abstractmethod
    def _iter_response_events(self, api_response: Any) -> Iterator[ProviderEvent]:
        """Map a non-streaming response to normalised :data:`ProviderEvent`s."""
        ...

    def _iter_events(self, api_response: Any) -> Iterator[ProviderEvent]:
        """Dispatch to the stream/non-stream mapper. ``self.stream`` is the single
        source of truth (it set the request's ``stream=`` flag), so the response
        type always matches it."""
        if self.stream:
            yield from self._iter_stream_events(api_response)
        else:
            yield from self._iter_response_events(api_response)

    @abstractmethod
    def _build_optional_kwargs(self, req_tools: Any, req_tool_choice: Any) -> dict[str, Any]:
        """Build the per-request tools/tool_choice kwargs in the backend's shape."""
        ...

    # â”€â”€ speculative-turn / cancellation gating â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _turn_is_latest(
        self,
        turn_id: str | None,
        turn_revision: int | None,
        *,
        runtime_config: Any | None = None,
        response_epoch: int | None = None,
    ) -> bool:
        admission = response_epoch_admission(
            runtime_config=runtime_config,
            response_epoch=response_epoch,
            cancel_scope=self.cancel_scope,
        )
        if admission is not None:
            return admission
        return self.speculative_turns is None or self.speculative_turns.is_latest(turn_id, turn_revision)

    def _generation_is_stale(self, gen: int | None) -> bool:
        return gen is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(gen)

    def _turn_output_allowed(
        self,
        turn_id: str | None,
        turn_revision: int | None,
        *,
        runtime_config: Any | None = None,
        response_epoch: int | None = None,
    ) -> bool:
        return response_output_allowed(
            runtime_config=runtime_config,
            response_epoch=response_epoch,
            turn_id=turn_id,
            turn_revision=turn_revision,
            speculative_turns=self.speculative_turns,
            cancel_scope=self.cancel_scope,
        )

    def _apply_config(
        self,
        chat: Chat,
        instructions: Optional[str],
        wants_audio: bool = True,
    ) -> None:
        if instructions:
            builder = build_voice_system_prompt if wants_audio else build_text_system_prompt
            full_instructions = builder(instructions)
            chat.add_item(make_system_message(full_instructions))

    # â”€â”€ output helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€--

    def _chunk(
        self,
        turn: _Turn,
        *,
        text: str = "",
        tools: list[ResponseFunctionToolCall] | None = None,
        language_code: Optional[str] = None,
    ) -> LLMResponseChunk:
        return LLMResponseChunk(
            text=text,
            language_code=language_code if language_code is not None else turn.language_code,
            tools=tools or [],
            runtime_config=turn.runtime_config,
            response=turn.response,
            turn_id=turn.turn_id,
            turn_revision=turn.turn_revision,
            speech_stopped_at_s=turn.speech_stopped_at_s,
            cancel_generation=turn.gen,
            input_epoch=turn.input_epoch,
            response_epoch=turn.response_epoch,
            response_id=turn.response_id,
        )

    def _record_tool_call(self, state: _GenState, turn: _Turn, item: ResponseFunctionToolCall) -> Iterator[LLMOut]:
        """Emit a tool call, persisting it (and any assistant text seen so far)
        to history *before* it is forwarded to the client.

        The function_call must already exist in the conversation by the time the
        client returns its ``function_call_output``; otherwise a fast client
        races ahead of the deferred end-of-turn write-back and the output is
        rejected ("No function_call with call_id ... found"), which makes the
        model re-issue the same tool call. The call lands in ``_pending_tool_calls``
        (not serialized until its output pairs it), so eager recording is safe.

        Out-of-band turns never touch the default conversation, and a stale turn
        records nothing (it is not forwarded to the client either)."""
        state.tools.append(item)
        fc_item = RealtimeConversationItemFunctionCall(
            type="function_call",
            name=item.name,
            arguments=item.arguments,
            call_id=item.call_id,
            id=item.id,
            status=item.status,
        )
        if self._generation_is_stale(turn.gen) or not self._turn_output_allowed(
            turn.turn_id,
            turn.turn_revision,
            runtime_config=turn.runtime_config,
            response_epoch=turn.response_epoch,
        ):
            logger.info("LLM generation cancelled (stale speculative turn)")
            return
        if not is_out_of_band(turn.response):
            # Flush assistant text accumulated before this call first (so history
            # order matches what the client received), then persist the call â€”
            # all before the chunk leaves for the client.
            chat = turn.runtime_config.chat
            with response_epoch_history_transaction(
                runtime_config=turn.runtime_config,
                response_epoch=turn.response_epoch,
            ) as admitted:
                if not admitted:
                    logger.info("LLM tool call cancelled before atomic history commit")
                    return
                for pending_item in state.pending:
                    chat.add_item(pending_item)
                state.pending.clear()
                chat.add_item(fc_item)
        yield self._chunk(turn, tools=[item])

    def _record_disallowed_tool_call(self, turn: _Turn, assistant_text_length: int) -> None:
        """Record a contract breach without retaining provider-supplied details."""
        self._emit_model_metric(
            turn,
            "tool_contract",
            detail={
                "operation": "post_tool",
                "finish_reason_category": "other",
                "assistant_text_length": assistant_text_length,
                "native_tool_fragment_count": 1,
                "completed_call_count": 0,
                "malformed_call_category": "native_call_disallowed",
            },
        )

    # â”€â”€ consumption â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€--

    def _consume_streaming(self, events: Iterator[ProviderEvent], state: _GenState, turn: _Turn) -> Iterator[LLMOut]:
        cancelled = False
        printable_text = ""
        sentence_batch: list[str] = []

        def _flush(batch: list[str]) -> Iterator[LLMOut]:
            if not batch:
                return
            if not self._turn_output_allowed(
                turn.turn_id,
                turn.turn_revision,
                runtime_config=turn.runtime_config,
                response_epoch=turn.response_epoch,
            ):
                logger.info("LLM generation cancelled (stale speculative turn)")
                return
            yield self._chunk(turn, text=" ".join(batch))

        for event in events:
            if self._generation_is_stale(turn.gen) or not self._turn_is_latest(
                turn.turn_id,
                turn.turn_revision,
                runtime_config=turn.runtime_config,
                response_epoch=turn.response_epoch,
            ):
                logger.info("LLM generation cancelled (interruption)")
                cancelled = True
                break

            if isinstance(event, Usage):
                state.input_tokens = event.input_tokens
                state.output_tokens = event.output_tokens
            elif isinstance(event, AssistantMessage):
                state.pending.append(
                    RealtimeConversationItemAssistantMessage(type="message", role="assistant", content=event.content)
                )
            elif isinstance(event, ToolCall):
                if not turn.tool_calls_allowed:
                    logger.warning("Provider returned a tool call while tool_choice=none; dropping it")
                    self._record_disallowed_tool_call(turn, len(state.clean_text))
                    continue
                # Flush any pending spoken text before emitting the tool call.
                if printable_text.strip():
                    sentence_batch.append(printable_text.strip())
                    printable_text = ""
                if sentence_batch:
                    if not self._turn_output_allowed(
                        turn.turn_id,
                        turn.turn_revision,
                        runtime_config=turn.runtime_config,
                        response_epoch=turn.response_epoch,
                    ):
                        logger.info("LLM generation cancelled (stale speculative turn)")
                        cancelled = True
                        break
                    yield from _flush(sentence_batch)
                    sentence_batch = []
                yield from self._record_tool_call(state, turn, event.item)
            elif isinstance(event, TextDelta):
                if not turn.wants_audio:
                    # Text-only: forward verbatim. Keep every character (no
                    # remove_unspeechable, which strips TTS-unfriendly symbols) and
                    # don't sentence-split (sent_tokenize collapses newlines/markdown).
                    state.clean_text += event.text
                    if event.text:
                        if not self._turn_output_allowed(
                            turn.turn_id,
                            turn.turn_revision,
                            runtime_config=turn.runtime_config,
                            response_epoch=turn.response_epoch,
                        ):
                            logger.info("LLM generation cancelled (stale speculative turn)")
                            cancelled = True
                            break
                        yield self._chunk(turn, text=event.text)
                    continue
                new_text = remove_unspeechable(event.text)
                state.clean_text += new_text
                printable_text += new_text
                sentences = sent_tokenize(printable_text)
                if len(sentences) > 1:
                    for s in sentences[:-1]:
                        sentence_batch.append(s)
                        if len(sentence_batch) >= self.stream_batch_sentences:
                            if not self._turn_output_allowed(
                                turn.turn_id,
                                turn.turn_revision,
                                runtime_config=turn.runtime_config,
                                response_epoch=turn.response_epoch,
                            ):
                                logger.info("LLM generation cancelled (stale speculative turn)")
                                cancelled = True
                                break
                            yield from _flush(sentence_batch)
                            sentence_batch = []
                    if cancelled:
                        break
                    printable_text = sentences[-1]

        if not cancelled:
            if printable_text.strip():
                sentence_batch.append(printable_text.strip())
            if sentence_batch:
                if self._generation_is_stale(turn.gen):
                    logger.info("LLM generation cancelled (interruption)")
                else:
                    logger.debug(f"Clean text: {state.clean_text}")
                    yield from _flush(sentence_batch)
            logger.info(f"Tools: {state.tools}")

    def _consume_nonstreaming(self, events: Iterator[ProviderEvent], state: _GenState, turn: _Turn) -> Iterator[LLMOut]:
        if self._generation_is_stale(turn.gen) or not self._turn_is_latest(
            turn.turn_id,
            turn.turn_revision,
            runtime_config=turn.runtime_config,
            response_epoch=turn.response_epoch,
        ):
            logger.info("LLM generation cancelled (interruption)")
            return
        for event in events:
            if isinstance(event, Usage):
                state.input_tokens = event.input_tokens
                state.output_tokens = event.output_tokens
            elif isinstance(event, AssistantMessage):
                state.pending.append(
                    RealtimeConversationItemAssistantMessage(type="message", role="assistant", content=event.content)
                )
            elif isinstance(event, ToolCall):
                if not turn.tool_calls_allowed:
                    logger.warning("Provider returned a tool call while tool_choice=none; dropping it")
                    self._record_disallowed_tool_call(turn, len(state.clean_text))
                    continue
                yield from self._record_tool_call(state, turn, event.item)
            elif isinstance(event, TextDelta):
                # Text-only keeps every character verbatim; audio strips
                # TTS-unfriendly symbols via remove_unspeechable.
                spoken = event.text if not turn.wants_audio else remove_unspeechable(event.text)
                state.clean_text += spoken
                out = spoken if not turn.wants_audio else spoken.strip()
                if (
                    out
                    and not self._generation_is_stale(turn.gen)
                    and self._turn_output_allowed(
                        turn.turn_id,
                        turn.turn_revision,
                        runtime_config=turn.runtime_config,
                        response_epoch=turn.response_epoch,
                    )
                ):
                    yield self._chunk(turn, text=out)
        logger.debug(f"Clean text: {state.clean_text}")
        logger.info(f"Tools: {state.tools}")

    # â”€â”€ orchestration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _generate(
        self,
        active_chat: Chat,
        original_chat: Chat,
        turn: _Turn,
        optional_kwargs: dict[str, Any],
    ) -> Iterator[LLMOut]:
        if not hasattr(self, "_active_response_lock"):
            self._active_response_lock = Lock()
            self._active_response = None
            self._active_client = None
        api_response: Any = None
        state = _GenState()
        error_message: str | None = None
        api_input = self._serialize(active_chat)
        request_started_s = perf_counter()
        first_output = True
        # Images the model actually sees this turn; only these are stripped on
        # write-back, so an image a fast client injects mid-generation for the
        # next turn survives (it is not in this serialized snapshot).
        consumed_image_ids = active_chat.image_message_ids()
        if not api_input:
            # Nothing to send: empty `instructions` and no `input` (in the response,
            # the default conversation, or the out-of-band context). The provider
            # would reject this; fail with a clear message instead of an opaque error.
            error_message = "Cannot generate a response: no instructions and no input were provided."

        try:
            if error_message is None:
                self._emit_model_metric(turn, "waiting_headers")
                api_response = self._request(api_input, optional_kwargs, turn.runtime_config)
                if self._model_operation_cancelled():
                    raise _ModelOperationCancelled()
            if api_response is not None:
                self._emit_model_metric(
                    turn,
                    "generating",
                    elapsed_ms=(perf_counter() - request_started_s) * 1000,
                )
                with self._active_response_lock:
                    self._active_response = api_response
                events = self._iter_events(api_response)
                outputs = (
                    self._consume_streaming(events, state, turn)
                    if self.stream
                    else self._consume_nonstreaming(events, state, turn)
                )
                for output in outputs:
                    if first_output and isinstance(output, LLMResponseChunk) and (output.text or output.tools):
                        self._emit_model_metric(
                            turn,
                            "first_token",
                            elapsed_ms=(perf_counter() - request_started_s) * 1000,
                        )
                        first_output = False
                    yield output
        except _ModelOperationCancelled:
            logger.info(
                "Language-model transport was cancelled before output admission "
                "turn=%s revision=%s response_epoch=%s",
                turn.turn_id,
                turn.turn_revision,
                turn.response_epoch,
            )
        except httpx.ReadTimeout:
            error_message = "Language model response timed out."
            self._emit_model_metric(
                turn,
                "timeout",
                elapsed_ms=(perf_counter() - request_started_s) * 1000,
            )
            logger.warning(
                "OpenAI API read timed out after %.1fs; ending the current response",
                self.request_timeout_s,
            )
        except Exception as exc:
            # Any other generation failure must still terminate the response: record
            # the error and fall through to the EndOfResponse below. Without this the
            # exception would escape process() and no EndOfResponse would be emitted,
            # leaving st.in_response stuck and locking every subsequent response.
            logger.exception("LLM generation failed; ending the current response")
            if error_message is None:
                error_message = f"Language model generation failed: {exc}"
        finally:
            with self._active_response_lock:
                if self._active_response is api_response:
                    self._active_response = None
                active_client = getattr(self, "_active_client", None)
                self._active_client = None
            if api_response is not None and hasattr(api_response, "close"):
                try:
                    api_response.close()
                except Exception:
                    pass
            if active_client is not None and active_client is not getattr(self, "client", None):
                try:
                    active_client.close()
                except Exception:
                    pass

        # Camera images are single-use once the request has reached the provider.
        # Use the request snapshot so an image added for the next turn survives.
        if not is_out_of_band(turn.response) and (api_response is not None or error_message is not None):
            original_chat.strip_images(consumed_image_ids)

        if (
            error_message is None
            and not self._generation_is_stale(turn.gen)
            and self._turn_output_allowed(
                turn.turn_id,
                turn.turn_revision,
                runtime_config=turn.runtime_config,
                response_epoch=turn.response_epoch,
            )
        ):
            history_admitted = True
            # Out-of-band responses emit output and usage but never write back to the
            # default conversation (their context was a throwaway chat).
            if not is_out_of_band(turn.response):
                # Tool calls (and any assistant text preceding them) were already
                # written eagerly in _record_tool_call; only trailing items remain.
                with response_epoch_history_transaction(
                    runtime_config=turn.runtime_config,
                    response_epoch=turn.response_epoch,
                ) as admitted:
                    history_admitted = admitted
                    if admitted:
                        for item in state.pending:
                            original_chat.add_item(item)
                        original_chat.strip_images(consumed_image_ids)
                        original_chat.trim_if_needed(self.compactor)
            if history_admitted and (state.input_tokens or state.output_tokens):
                yield TokenUsage(
                    input_tokens=state.input_tokens,
                    output_tokens=state.output_tokens,
                    runtime_config=turn.runtime_config,
                    turn_id=turn.turn_id,
                    turn_revision=turn.turn_revision,
                    input_epoch=turn.input_epoch,
                    response_epoch=turn.response_epoch,
                    response_id=turn.response_id,
                )
        final_status = "cancelled" if self._generation_is_stale(turn.gen) else "complete"
        if error_message is not None:
            final_status = "failed"
        self._emit_model_metric(
            turn,
            final_status,
            elapsed_ms=(perf_counter() - request_started_s) * 1000,
        )
        yield EndOfResponse(
            runtime_config=turn.runtime_config,
            turn_id=turn.turn_id,
            turn_revision=turn.turn_revision,
            cancel_generation=turn.gen,
            error=error_message,
            input_epoch=turn.input_epoch,
            response_epoch=turn.response_epoch,
            response_id=turn.response_id,
        )

    def cancel_active(self) -> None:
        """Close the current provider stream so cancellation releases the slot."""
        lock = getattr(self, "_active_response_lock", None)
        if lock is None:
            return
        with lock:
            response = self._active_response
            client = self._active_client
            self._active_response = None
            self._active_client = None
        for resource in (response, client):
            if resource is not None and hasattr(resource, "close"):
                try:
                    resource.close()
                except Exception:
                    logger.debug("Provider transport was already closed during cancellation")

    def _emit_model_metric(
        self,
        turn: _Turn,
        status: str,
        *,
        elapsed_ms: float | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        text_output_queue = getattr(self, "text_output_queue", None)
        if text_output_queue is None:
            return
        text_output_queue.put(
            PipelineMetricEvent(
                stage="gemma",
                status=status,
                at_s=time(),
                elapsed_ms=elapsed_ms,
                turn_id=turn.turn_id,
                turn_revision=turn.turn_revision,
                input_epoch=turn.input_epoch,
                response_epoch=turn.response_epoch,
                response_id=turn.response_id,
                detail={"operation": "post_tool", **(detail or {})},
            )
        )

    def process(self, request: LLMIn) -> Iterator[LLMOut]:
        """Process a language model request and yield LLMResponseChunks."""
        if isinstance(request, DirectAssistantRequest):
            if request.text or request.tools:
                yield LLMResponseChunk(
                    text=request.text,
                    language_code=request.language_code,
                    runtime_config=request.runtime_config,
                    response=request.response,
                    turn_id=request.turn_id,
                    turn_revision=request.turn_revision,
                    speech_stopped_at_s=request.speech_stopped_at_s,
                    tools=request.tools,
                    cancel_generation=request.cancel_generation,
                    input_epoch=request.input_epoch,
                    response_epoch=request.response_epoch,
                    response_id=request.response_id,
                )
            if request.is_final:
                yield EndOfResponse(
                    runtime_config=request.runtime_config,
                    turn_id=request.turn_id,
                    turn_revision=request.turn_revision,
                    cancel_generation=request.cancel_generation,
                    error=request.error,
                    input_epoch=request.input_epoch,
                    response_epoch=request.response_epoch,
                    response_id=request.response_id,
                )
            return
        runtime_config = request.runtime_config
        response = request.response
        turn_id = request.turn_id
        turn_revision = request.turn_revision
        speech_stopped_at_s = request.speech_stopped_at_s
        input_epoch = request.input_epoch
        response_epoch = request.response_epoch
        response_id = request.response_id
        if not wait_for_response_epoch_admission(
            runtime_config=runtime_config,
            response_epoch=response_epoch,
        ):
            logger.info("Skipping cancelled queued LLM request for response epoch=%s", response_epoch)
            return
        if not self._turn_is_latest(
            turn_id,
            turn_revision,
            runtime_config=runtime_config,
            response_epoch=response_epoch,
        ):
            logger.info("Skipping stale LLM request for turn=%s rev=%s", turn_id, turn_revision)
            yield EndOfResponse(
                runtime_config=runtime_config,
                turn_id=turn_id,
                turn_revision=turn_revision,
                input_epoch=input_epoch,
                response_epoch=response_epoch,
                response_id=response_id,
            )
            return

        original_chat = runtime_config.chat
        if is_out_of_band(response):
            try:
                active_chat = build_active_chat(original_chat, response)
            except ChatItemError as exc:
                logger.info("Out-of-band response rejected: %s", exc)
                yield EndOfResponse(
                    runtime_config=runtime_config,
                    turn_id=turn_id,
                    turn_revision=turn_revision,
                    error=str(exc),
                    input_epoch=input_epoch,
                    response_epoch=response_epoch,
                    response_id=response_id,
                )
                return
        else:
            active_chat = original_chat.copy()
        language_code = request.language_code
        instructions = (
            response.instructions if response and response.instructions else runtime_config.session.instructions
        ) or ""
        req_tools = response.tools if response and response.tools else runtime_config.session.tools
        req_tool_choice = (
            response.tool_choice if response and response.tool_choice else runtime_config.session.tool_choice
        )
        wants_audio = response_wants_audio(response)
        self._apply_config(active_chat, instructions, wants_audio)
        language_code, lang_name = resolve_auto_language(language_code)
        if lang_name and self.enable_lang_prompt:
            active_chat.add_item(make_user_message(f"Please reply to my message in {lang_name}."))

        optional_kwargs = self._build_optional_kwargs(req_tools, req_tool_choice)

        # The shared model-operation coordinator closes a blocked provider stream
        # on barge-in; the generation tag still rejects any late detached output.
        gen = self.cancel_scope.generation if self.cancel_scope else None

        turn = _Turn(
            language_code=language_code,
            gen=gen,
            runtime_config=runtime_config,
            response=response,
            turn_id=turn_id,
            turn_revision=turn_revision,
            speech_stopped_at_s=speech_stopped_at_s,
            wants_audio=wants_audio,
            input_epoch=input_epoch,
            response_epoch=response_epoch,
            response_id=response_id,
            tool_calls_allowed=req_tool_choice != "none",
        )
        operation: ModelOperationToken | None = None
        coordinator = getattr(self, "model_operations", None)
        if coordinator is not None:
            session_id = str(runtime_config.local_pipeline.get("_session_id") or "") or None
            operation = coordinator.acquire(
                kind="post_tool",
                session_id=session_id,
                turn_id=turn_id,
                turn_revision=turn_revision,
                cancel_generation=gen,
                stale=lambda: self._generation_is_stale(gen)
                or not self._turn_is_latest(
                    turn_id,
                    turn_revision,
                    runtime_config=runtime_config,
                    response_epoch=response_epoch,
                ),
            )
            if operation is None:
                yield EndOfResponse(
                    runtime_config=runtime_config,
                    turn_id=turn_id,
                    turn_revision=turn_revision,
                    cancel_generation=gen,
                    input_epoch=input_epoch,
                    response_epoch=response_epoch,
                    response_id=response_id,
                )
                return
        try:
            if operation is not None and coordinator is not None:
                self._operation_context.value = (coordinator, operation)
            if operation is not None and coordinator is not None and not coordinator.is_current(operation):
                yield EndOfResponse(
                    runtime_config=runtime_config,
                    turn_id=turn_id,
                    turn_revision=turn_revision,
                    cancel_generation=gen,
                    input_epoch=input_epoch,
                    response_epoch=response_epoch,
                    response_id=response_id,
                )
                return
            yield from self._generate(active_chat, original_chat, turn, optional_kwargs)
        finally:
            # Several focused adapters construct a lightweight handler without
            # ``setup``/``__init__``.  Cancellation still has to release the
            # coordinator in that path; do not turn terminal cleanup into a
            # second failure merely because no thread-local context was made.
            operation_context = getattr(self, "_operation_context", None)
            if operation_context is not None and hasattr(operation_context, "value"):
                del operation_context.value
            if operation is not None and coordinator is not None:
                coordinator.release(operation)

    @property
    def timing_log_level(self) -> int:
        return logging.INFO

    def should_log_timing(self, output: LLMOut) -> bool:
        return isinstance(output, LLMResponseChunk) and self.last_time > self.min_time_to_debug
