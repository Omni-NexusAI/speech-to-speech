"""Identity propagation regressions for response-owned realtime output."""

from __future__ import annotations

from contextlib import contextmanager
from queue import Queue
from types import SimpleNamespace

import numpy as np
from openai.types.realtime import ConversationItemCreateEvent, ResponseCreateEvent
from openai.types.realtime.realtime_conversation_item_function_call import (
    RealtimeConversationItemFunctionCall,
)

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.LLM.language_model import LanguageModelHandler
from speech_to_speech.pipeline.events import (
    PipelineMetricEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    TranscriptionCompletedEvent,
)
from speech_to_speech.pipeline.messages import (
    DirectAssistantResponse,
    GenerateResponseRequest,
    LLMResponseChunk,
    TokenUsage,
    TTSInput,
    VADAudio,
)
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.STT.gemma_audio_handler import GemmaAudioSTTHandler
from speech_to_speech.STT.transcription_notifier import TranscriptionNotifier
from speech_to_speech.TTS.qwen3_tts_handler import Qwen3TTSHandler


def test_service_generation_request_carries_claimed_response_epochs() -> None:
    """The normal STT path must give LM/TTS a stale-output identity."""
    prompt_queue: Queue[GenerateResponseRequest] = Queue()
    service = RealtimeService(text_prompt_queue=prompt_queue)
    conn_id = service.register()
    try:
        owner = service.claim_pending_response(conn_id, turn_id="turn_a", turn_revision=0)

        service._on_transcription_completed(
            conn_id,
            TranscriptionCompletedEvent(
                transcript="final input",
                turn_id="turn_a",
                turn_revision=0,
            ),
        )

        request = prompt_queue.get_nowait()
        assert request.input_epoch == owner.input_epoch
        assert request.response_epoch == owner.response_epoch
        # An implicit VAD response has no OpenAI response object yet.  The epoch,
        # not a guessed response id, is its authoritative downstream identity.
        assert request.response_id is None
    finally:
        service.unregister(conn_id)


def test_direct_audio_notifier_preserves_response_identity() -> None:
    """Direct Gemma output cannot lose the identity before it reaches LM/TTS."""
    notifier = TranscriptionNotifier.__new__(TranscriptionNotifier)
    notifier.setup(text_output_queue=Queue())

    outputs = list(
        notifier.process(
            DirectAssistantResponse(
                text="A direct answer.",
                is_final=True,
                turn_id="turn_a",
                turn_revision=0,
                input_epoch=7,
                response_epoch=11,
                response_id="resp_11",
            )
        )
    )

    assert len(outputs) == 1
    request = outputs[0]
    assert request.input_epoch == 7
    assert request.response_epoch == 11
    assert request.response_id == "resp_11"


def test_gemma_direct_response_resolves_identity_without_name_error() -> None:
    runtime = RuntimeConfig()
    runtime.local_pipeline.update(
        {"_input_epoch": 5, "_response_epoch": 8, "_response_id": "resp_8"}
    )
    handler = GemmaAudioSTTHandler.__new__(GemmaAudioSTTHandler)

    output = handler._direct(
        VADAudio(audio=np.zeros(16, dtype=np.float32), runtime_config=runtime),
        "answer",
        is_final=True,
    )

    assert (output.input_epoch, output.response_epoch, output.response_id) == (5, 8, "resp_8")


def test_empty_turn_settles_its_provisional_checkpoint() -> None:
    """A cancelled empty turn cannot roll back a later client item."""
    service = RealtimeService(text_prompt_queue=Queue())
    conn_id = service.register()
    try:
        service.dispatch_pipeline_event(conn_id, SpeechStartedEvent(turn_id="turn_empty", turn_revision=0))
        service.dispatch_pipeline_event(
            conn_id,
            SpeechStoppedEvent(duration_s=1.0, turn_id="turn_empty", turn_revision=0),
        )
        st = service._state(conn_id)
        owner = st.response_ownership.active()
        assert owner is not None
        assert owner.response_epoch in st.provisional_chat_checkpoints

        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(transcript="", turn_id="turn_empty", turn_revision=0),
        )

        assert owner.response_epoch not in st.provisional_chat_checkpoints
    finally:
        service.unregister(conn_id)


def test_pending_direct_owner_defers_new_client_items_until_it_settles() -> None:
    """A direct Gemma answer is transactional before an OpenAI response exists."""
    service = RealtimeService(text_prompt_queue=Queue())
    conn_id = service.register()
    try:
        service.claim_pending_response(conn_id, turn_id="turn_direct", turn_revision=0)
        item = ConversationItemCreateEvent(
            type="conversation.item.create",
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "belongs to the next turn"}],
            },
        )

        assert service.handle_conversation_item_create(conn_id, item) == []
        st = service._state(conn_id)
        assert st.deferred_items == [item.item]
        assert st.runtime_config.chat.buffer == []
    finally:
        service.unregister(conn_id)


def test_response_create_cannot_start_a_second_generation_while_direct_turn_is_pending() -> None:
    """The VAD/direct-audio gap is already one owned model operation."""
    prompts: Queue[GenerateResponseRequest] = Queue()
    service = RealtimeService(text_prompt_queue=prompts)
    conn_id = service.register()
    try:
        owner = service.claim_pending_response(conn_id, turn_id="direct-pending", turn_revision=0)

        result = service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create"))

        assert result is not None
        assert result.type == "error"
        assert result.error.type == "conversation_already_has_active_response"
        assert prompts.empty()
        active = service._state(conn_id).response_ownership.active()
        assert active is not None and active.response_epoch == owner.response_epoch
        assert service._state(conn_id).in_response is False
    finally:
        service.unregister(conn_id)


def test_tts_rejects_a_stale_response_epoch_from_its_runtime_snapshot() -> None:
    """Epoch cancellation must work even when no legacy cancel generation is present."""
    runtime_config = RuntimeConfig()
    runtime_config.local_pipeline["_response_epoch_is_current"] = lambda _epoch: False
    handler = Qwen3TTSHandler.__new__(Qwen3TTSHandler)
    handler.cancel_scope = None

    assert handler._input_generation_is_stale(
        TTSInput(text="late phrase", runtime_config=runtime_config, response_epoch=9)
    )


def test_transformers_llm_admits_current_epoch_over_newer_speculative_turn() -> None:
    """The local-model stream loop uses the same epoch-first ownership contract."""

    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_new", 0)
    runtime = RuntimeConfig()
    runtime.local_pipeline["_response_epoch_is_current"] = lambda epoch: epoch == 12
    handler = object.__new__(LanguageModelHandler)
    handler.speculative_turns = tracker
    handler.cancel_scope = None

    assert handler._turn_is_latest(
        "turn_old",
        0,
        runtime_config=runtime,
        response_epoch=12,
    )
    assert handler._turn_output_allowed(
        "turn_old",
        0,
        runtime_config=runtime,
        response_epoch=12,
    )


def test_transformers_llm_does_not_run_a_cancelled_queued_epoch() -> None:
    """The normal local-model entry point honours retained-turn admission."""

    runtime = RuntimeConfig()
    waited: list[int] = []
    runtime.local_pipeline["_wait_response_epoch_current"] = (
        lambda epoch: (waited.append(epoch), False)[1]
    )
    handler = object.__new__(LanguageModelHandler)
    handler.speculative_turns = None
    handler.cancel_scope = None

    outputs = list(
        handler.process(
            GenerateResponseRequest(
                runtime_config=runtime,
                response_epoch=41,
            )
        )
    )

    assert waited == [41]
    assert outputs == []


def test_transformers_llm_history_guard_rejects_stale_final_commit() -> None:
    """Local-model history and trailing output share the atomic epoch decision."""

    @contextmanager
    def reject_history(_epoch):
        yield False

    runtime = RuntimeConfig()
    runtime.local_pipeline["_response_epoch_is_current"] = lambda epoch: epoch == 42
    runtime.local_pipeline["_response_epoch_history_transaction"] = reject_history
    handler = object.__new__(LanguageModelHandler)
    handler.speculative_turns = None
    handler.cancel_scope = None
    handler.enable_lang_prompt = False
    handler.compactor = None
    handler.tokenizer = SimpleNamespace(encode=lambda _text: [1, 2])
    handler._apply_instructions = lambda *_args, **_kwargs: None

    def fake_generate(_chat, _language, _generation, ctx, _runtime, _response):
        ctx.generated_text = "stale assistant"
        ctx.raw_generated_text = "stale assistant"
        ctx.printable_text = "stale assistant"
        ctx.input_tokens = 3
        ctx.tools = []
        if False:
            yield None

    handler._generate = fake_generate
    outputs = list(
        handler.process(
            GenerateResponseRequest(
                runtime_config=runtime,
                response_epoch=42,
            )
        )
    )

    assert runtime.chat.buffer == []
    assert not any(isinstance(output, (LLMResponseChunk, TokenUsage)) for output in outputs)


def test_service_rejects_ambiguous_response_scoped_metric_identity() -> None:
    """A late manual-turn metric cannot be rebound to the newest owner."""
    service = RealtimeService(text_prompt_queue=Queue())
    conn_id = service.register()
    try:
        owner = service.claim_manual_response(conn_id)

        assert (
            service.dispatch_pipeline_event(
                conn_id,
                PipelineMetricEvent(stage="tts", status="done", at_s=1.0),
            )
            == []
        )

        events = service.dispatch_pipeline_event(
            conn_id,
            PipelineMetricEvent(
                stage="tts",
                status="done",
                at_s=2.0,
                input_epoch=owner.input_epoch,
                response_epoch=owner.response_epoch,
            ),
        )
        assert len(events) == 1
        assert events[0].response_epoch == owner.response_epoch
        assert events[0].input_epoch == owner.input_epoch
    finally:
        service.unregister(conn_id)


def test_authoritative_cancel_metric_can_describe_the_just_staled_response() -> None:
    """Router cancellation diagnostics retain old identity without reopening it."""
    service = RealtimeService(text_prompt_queue=Queue())
    conn_id = service.register()
    try:
        owner = service.claim_pending_response(conn_id, turn_id="turn_old", turn_revision=0)
        service.observe_speech_started(conn_id, reason="barge_in")

        stale_provider_metric = PipelineMetricEvent(
            stage="tts",
            status="done",
            at_s=1.0,
            input_epoch=owner.input_epoch,
            response_epoch=owner.response_epoch,
        )
        assert service.dispatch_pipeline_event(conn_id, stale_provider_metric) == []

        events = service.dispatch_pipeline_event(
            conn_id,
            PipelineMetricEvent(
                stage="gemma",
                status="cancelled",
                at_s=2.0,
                input_epoch=owner.input_epoch,
                response_epoch=owner.response_epoch,
                authoritative_terminal=True,
            ),
        )
        assert len(events) == 1
        assert events[0].response_epoch == owner.response_epoch
        assert events[0].status == "cancelled"
    finally:
        service.unregister(conn_id)


def test_playback_ack_flushes_items_deferred_behind_pre_audible_response() -> None:
    """A worklet acknowledgement must not strand deferred client chronology."""
    service = RealtimeService(text_prompt_queue=Queue())
    conn_id = service.register()
    try:
        # A browser declares this before any session/pipeline configuration.
        # Legacy clients settle on their first delivered PCM instead.
        service.set_rendered_playback_ack_supported(conn_id, True)
        owner = service.claim_pending_response(conn_id, turn_id="turn_a", turn_revision=0)
        response_id, _ = service.response._ensure_response(conn_id)
        item = ConversationItemCreateEvent(
            type="conversation.item.create",
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "queued client item"}],
            },
        )
        assert service.handle_conversation_item_create(conn_id, item) == []
        # PCM was emitted but the browser's worklet has not rendered it yet.
        # This is the race window that must retain the transaction until ack.
        service.encode_audio_chunk(conn_id, b"\x00\x00" * 32)
        service.response.finish_response(conn_id)
        st = service._state(conn_id)
        assert st.deferred_items == [item.item]

        service.handle_playback_started(
            conn_id,
            response_id=response_id,
            response_epoch=owner.response_epoch,
        )

        assert st.deferred_items == []
        assert st.runtime_config.chat.buffer[-1] == item.item
    finally:
        service.unregister(conn_id)


def test_pre_audible_supersession_keeps_safe_deferred_context_for_replacement() -> None:
    """A fresh input must see safe client context after it rolls back the unheard turn."""
    service = RealtimeService(text_prompt_queue=Queue())
    conn_id = service.register()
    try:
        old = service.claim_pending_response(conn_id, turn_id="turn_old", turn_revision=0)
        safe_item = ConversationItemCreateEvent(
            type="conversation.item.create",
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "keep this with the replacement"}],
            },
        )
        assert service.handle_conversation_item_create(conn_id, safe_item) == []

        supersession = service.observe_speech_started(conn_id, reason="new_speech")
        st = service._state(conn_id)

        assert supersession.previous is not None
        assert supersession.previous.response_epoch == old.response_epoch
        assert st.deferred_items == []
        assert st.runtime_config.chat.buffer[-1] == safe_item.item
    finally:
        service.unregister(conn_id)


def test_pre_audible_supersession_drops_stale_deferred_tool_output_once() -> None:
    """A rolled-back tool transaction cannot trigger a second follow-up later."""
    service = RealtimeService(text_prompt_queue=Queue())
    conn_id = service.register()
    try:
        service.claim_pending_response(conn_id, turn_id="turn_old", turn_revision=0)
        st = service._state(conn_id)
        st.pending_tool_call_ids.add("call_stale")
        safe_item = ConversationItemCreateEvent(
            type="conversation.item.create",
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "final user context"}],
            },
        )
        stale_output = ConversationItemCreateEvent(
            type="conversation.item.create",
            item={
                "type": "function_call_output",
                "call_id": "call_stale",
                "output": "obsolete tool result",
            },
        )
        assert service.handle_conversation_item_create(conn_id, safe_item) == []
        assert service.handle_conversation_item_create(conn_id, stale_output) == []

        service.observe_speech_started(conn_id, reason="new_speech")

        assert st.pending_tool_call_ids == set()
        assert st.tool_followup_ready is False
        assert st.tool_followup_started is False
        assert st.tool_followup_requested is False
        assert st.deferred_items == []
        assert st.runtime_config.chat.buffer[-1] == safe_item.item
        assert all(
            getattr(item, "call_id", None) != "call_stale"
            for item in st.runtime_config.chat.buffer
        )
    finally:
        service.unregister(conn_id)


def test_playback_ack_applies_deferred_tool_output_exactly_once() -> None:
    """A repeated browser acknowledgement cannot reapply or retrigger a tool output."""
    service = RealtimeService(text_prompt_queue=Queue())
    conn_id = service.register()
    try:
        st = service._state(conn_id)
        st.runtime_config.chat.add_item(
            RealtimeConversationItemFunctionCall(
                type="function_call", call_id="call_live", name="lookup", arguments="{}"
            )
        )
        st.pending_tool_call_ids.add("call_live")
        owner = service.claim_pending_response(conn_id, turn_id="turn_live", turn_revision=0)
        response_id, _ = service.response._ensure_response(conn_id)
        output = ConversationItemCreateEvent(
            type="conversation.item.create",
            item={"type": "function_call_output", "call_id": "call_live", "output": "result"},
        )
        assert service.handle_conversation_item_create(conn_id, output) == []

        assert service.handle_playback_started(
            conn_id, response_id=response_id, response_epoch=owner.response_epoch
        ) is not None
        # First rendered audio settles audibility, but the active model thread
        # may still be writing its assistant/tool transaction.  Do not flush
        # client chronology until that terminal barrier has landed.
        assert service.take_deferred_settlement_events(conn_id) == []
        assert st.pending_tool_call_ids == {"call_live"}
        assert st.tool_followup_ready is False
        assert sum(item.type == "function_call_output" for item in st.runtime_config.chat.buffer) == 0

        applied = service.finish_response(conn_id)
        assert [event.type for event in applied].count("conversation.item.created") == 1
        assert st.pending_tool_call_ids == set()
        assert st.tool_followup_ready is True
        assert sum(item.type == "function_call_output" for item in st.runtime_config.chat.buffer) == 1

        assert service.handle_playback_started(
            conn_id, response_id=response_id, response_epoch=owner.response_epoch
        ) is None
        assert service.take_deferred_settlement_events(conn_id) == []
        assert sum(item.type == "function_call_output" for item in st.runtime_config.chat.buffer) == 1
    finally:
        service.unregister(conn_id)
