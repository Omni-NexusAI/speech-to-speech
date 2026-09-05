"""Regression coverage for final-turn response ownership.

These tests deliberately exercise the server-side authority rather than a
browser response flag: transport completion is not audible until the local
worklet acknowledgement arrives.
"""

from __future__ import annotations

import pytest
from openai.types.realtime import ConversationItemCreateEvent, ResponseCreateEvent
from openai.types.realtime.conversation_item import RealtimeConversationItemFunctionCall

from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.LLM.chat import make_assistant_message, make_user_message
from speech_to_speech.pipeline.events import (
    AssistantTextEvent,
    PipelineMetricEvent,
    ResponseOutputCompleteEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    TokenUsageEvent,
    TranscriptionCompletedEvent,
)
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.VAD.vad_handler import VADHandler


def _claimed_service() -> tuple[RealtimeService, str]:
    service = RealtimeService()
    conn_id = service.register()
    return service, conn_id


def test_completed_transport_is_still_pre_audible_until_browser_ack() -> None:
    service, conn_id = _claimed_service()
    try:
        owner = service.claim_pending_response(conn_id, turn_id="turn_a", turn_revision=0)
        service.bind_response_id(conn_id, "resp_a")
        service.mark_response_completed(conn_id)

        supersession = service.observe_speech_started(conn_id, reason="same_turn_revision")

        assert supersession.previous is not None
        assert supersession.previous.state == "completed"
        assert supersession.pre_audible is True
        assert supersession.requires_output_suppression is True
        assert service._state(conn_id).response_ownership.is_current(owner.response_epoch) is False
    finally:
        service.unregister(conn_id)


def test_transport_complete_reports_playback_pending_until_render_ack() -> None:
    """The local lifecycle distinguishes protocol completion from audibility."""

    service, conn_id = _claimed_service()
    try:
        service.set_rendered_playback_ack_supported(conn_id, True)
        owner = service.claim_pending_response(conn_id, turn_id="turn_pending", turn_revision=0)
        service.bind_response_id(conn_id, "resp_pending")
        service.mark_response_completed(conn_id)

        event = service.response_owner_event(conn_id)

        assert event is not None
        assert event.state == "playback_pending"
        assert service.handle_playback_started(
            conn_id,
            response_id="resp_pending",
            response_epoch=owner.response_epoch,
        ).state == "completed"
    finally:
        service.unregister(conn_id)


def test_pulled_old_pcm_cannot_be_encoded_for_newer_owner() -> None:
    """A batched old tuple never allocates or attaches to the replacement response."""

    service, conn_id = _claimed_service()
    try:
        old = service.claim_pending_response(conn_id, turn_id="turn_old_pcm", turn_revision=0)
        service.bind_response_id(conn_id, "resp_old_pcm")
        service.observe_speech_started(conn_id, reason="new_confirmed_speech")
        replacement = service.claim_pending_response(conn_id, turn_id="turn_new_pcm", turn_revision=0)

        assert service.encode_audio_chunk(
            conn_id,
            b"\x00\x00" * 160,
            input_epoch=old.input_epoch,
            response_epoch=old.response_epoch,
            response_id="resp_old_pcm",
        ) == []
        active = service._state(conn_id).response_ownership.active()
        assert active is not None and active.response_epoch == replacement.response_epoch
        assert service._state(conn_id).current_response_id is None
    finally:
        service.unregister(conn_id)


def test_playback_ack_is_idempotent_and_makes_completed_owner_heard() -> None:
    service, conn_id = _claimed_service()
    try:
        owner = service.claim_pending_response(conn_id, turn_id="turn_a", turn_revision=0)
        service.bind_response_id(conn_id, "resp_a")
        service.mark_response_completed(conn_id)

        first = service.handle_playback_started(
            conn_id, response_id="resp_a", response_epoch=owner.response_epoch
        )
        second = service.handle_playback_started(
            conn_id, response_id="resp_a", response_epoch=owner.response_epoch
        )
        supersession = service.observe_speech_started(conn_id, reason="next_turn")

        assert first is not None and first.state == "completed"
        assert second is None
        assert supersession.pre_audible is False
        assert supersession.requires_output_suppression is False
    finally:
        service.unregister(conn_id)


def test_completed_epoch_rejects_late_output_but_still_accepts_render_ack() -> None:
    service, conn_id = _claimed_service()
    try:
        owner = service.claim_pending_response(conn_id, turn_id="turn_a", turn_revision=0)
        service.bind_response_id(conn_id, "resp_a")
        service.mark_response_completed(conn_id)

        assert service._state(conn_id).response_ownership.is_current(owner.response_epoch)
        assert not service._state(conn_id).response_ownership.admits_output(owner.response_epoch)
        assert service.dispatch_pipeline_event(
            conn_id,
            AssistantTextEvent(
                text="late duplicate",
                turn_id="turn_a",
                turn_revision=0,
                input_epoch=owner.input_epoch,
                response_epoch=owner.response_epoch,
                response_id="resp_a",
            ),
        ) == []
        assert service.dispatch_pipeline_event(
            conn_id,
            PipelineMetricEvent(
                stage="tts",
                status="done",
                at_s=1.0,
                input_epoch=owner.input_epoch,
                response_epoch=owner.response_epoch,
                response_id="resp_a",
            ),
        ) == []
        assert service.dispatch_pipeline_event(
            conn_id,
            ResponseOutputCompleteEvent(
                turn_id="turn_a",
                turn_revision=0,
                input_epoch=owner.input_epoch,
                response_epoch=owner.response_epoch,
                response_id="resp_a",
            ),
        ) == []
        assert service.handle_playback_started(
            conn_id, response_id="resp_a", response_epoch=owner.response_epoch
        ) is not None
    finally:
        service.unregister(conn_id)


def test_pre_audible_supersession_rolls_back_user_assistant_and_pending_tool_context() -> None:
    service, conn_id = _claimed_service()
    try:
        chat = service._state(conn_id).runtime_config.chat
        chat.add_item(make_user_message("retained history"))
        service.claim_pending_response(conn_id, turn_id="turn_a", turn_revision=0)
        chat.add_item(make_user_message("provisional input"))
        chat.add_item(make_assistant_message("provisional answer"))
        chat.add_item(
            RealtimeConversationItemFunctionCall(
                type="function_call",
                id="fc_provisional",
                call_id="call_provisional",
                name="test_tool",
                arguments="{}",
            )
        )
        state = service._state(conn_id)
        state.pending_tool_call_ids.add("call_provisional")
        state.tool_followup_ready = True
        state.tool_followup_requested = True

        service.observe_speech_started(conn_id, reason="forced_horizon_split")

        assert [item.role for item in chat.buffer] == ["user"]
        assert "retained history" in chat.history_token_text()
        assert "provisional input" not in chat.history_token_text()
        assert "provisional answer" not in chat.history_token_text()
        assert state.pending_tool_call_ids == set()
        assert state.tool_followup_ready is False
        assert state.tool_followup_requested is False
    finally:
        service.unregister(conn_id)


def test_stale_response_epoch_cannot_emit_late_assistant_text() -> None:
    service, conn_id = _claimed_service()
    try:
        owner = service.claim_pending_response(conn_id, turn_id="turn_a", turn_revision=0)
        service.observe_speech_started(conn_id, reason="synthetic_final")

        events = service.dispatch_pipeline_event(
            conn_id,
            AssistantTextEvent(
                text="late answer",
                turn_id="turn_a",
                turn_revision=0,
                input_epoch=owner.input_epoch,
                response_epoch=owner.response_epoch,
            ),
        )

        assert events == []
    finally:
        service.unregister(conn_id)


def test_response_epochs_are_monotonic_across_same_turn_revisions() -> None:
    service, conn_id = _claimed_service()
    try:
        first = service.claim_pending_response(conn_id, turn_id="turn_a", turn_revision=0)
        service.observe_speech_started(conn_id, reason="revision_reopen")
        second = service.claim_pending_response(conn_id, turn_id="turn_a", turn_revision=1)

        assert second.input_epoch > first.input_epoch
        assert second.response_epoch > first.response_epoch
        assert service.response_owner_for_turn(conn_id, "turn_a", 0) is not None
        assert service.response_owner_for_turn(conn_id, "turn_a", 1) == second
    finally:
        service.unregister(conn_id)


@pytest.mark.parametrize("reason", ["fixed_horizon", "revision_limit", "synthetic_final"])
def test_all_forced_input_splits_supersede_before_legacy_response_flags(reason: str) -> None:
    service, conn_id = _claimed_service()
    try:
        owner = service.claim_pending_response(conn_id, turn_id="turn_a", turn_revision=0)
        state = service._state(conn_id)
        # Direct Gemma can be active before OpenAI response flags are set.
        state.in_response = False
        state.response_pending = False

        supersession = service.observe_speech_started(conn_id, reason=reason)

        assert supersession.previous == owner
        assert supersession.requires_transport_cancel is True
        assert supersession.pre_audible is True
        assert state.response_ownership.is_current(owner.response_epoch) is False
    finally:
        service.unregister(conn_id)


def test_post_audible_barge_in_retains_committed_context_once() -> None:
    service, conn_id = _claimed_service()
    try:
        service.set_rendered_playback_ack_supported(conn_id, True)
        chat = service._state(conn_id).runtime_config.chat
        owner = service.claim_pending_response(conn_id, turn_id="turn_a", turn_revision=0)
        chat.add_item(make_user_message("heard input"))
        chat.add_item(make_assistant_message("heard answer"))
        service.bind_response_id(conn_id, "resp_a")
        assert service.handle_playback_started(conn_id, response_id="resp_a", response_epoch=owner.response_epoch)

        supersession = service.observe_speech_started(conn_id, reason="barge_in")

        assert supersession.pre_audible is False
        assert service._state(conn_id).response_ownership.is_current(owner.response_epoch) is False
        assert "heard input" in chat.history_token_text()
        assert "heard answer" in chat.history_token_text()
        lifecycle = service.take_deferred_settlement_events(conn_id)
        assert lifecycle[-1].response_epoch == owner.response_epoch
        assert lifecycle[-1].state == "cancelled"
        assert lifecycle[-1].reason == "barge_in"
    finally:
        service.unregister(conn_id)


def test_stale_terminal_event_is_rejected_after_input_epoch_advances() -> None:
    service, conn_id = _claimed_service()
    try:
        owner = service.claim_pending_response(conn_id, turn_id="turn_a", turn_revision=0)
        service.observe_speech_started(conn_id, reason="same_turn_revision")

        assert service.dispatch_pipeline_event(
            conn_id,
            ResponseOutputCompleteEvent(
                turn_id="turn_a",
                turn_revision=0,
                input_epoch=owner.input_epoch,
                response_epoch=owner.response_epoch,
            ),
        ) == []
    finally:
        service.unregister(conn_id)


def test_synthetic_final_advances_input_before_claim_and_queued_start_is_idempotent() -> None:
    """A queued synthetic start cannot invalidate the response claimed behind it."""
    service, conn_id = _claimed_service()
    try:
        service.set_rendered_playback_ack_supported(conn_id, True)
        old = service.claim_pending_response(conn_id, turn_id="turn_old", turn_revision=0)
        runtime = service._state(conn_id).runtime_config

        input_epoch = VADHandler._observe_speech_identity(runtime, reason="synthetic_final")
        started = SpeechStartedEvent(
            turn_id="turn_final",
            turn_revision=0,
            interrupt_response=False,
            input_epoch=input_epoch,
        )
        identity = VADHandler._claim_response_identity(runtime, "turn_final", 0)
        replacement = service._state(conn_id).response_ownership.active()

        # Dispatch happens after the final VAD audio has already claimed its
        # owner, matching the cross-queue ordering that previously raced.
        service.dispatch_pipeline_event(conn_id, started)

        assert replacement is not None
        assert identity[:2] == (input_epoch, replacement.response_epoch)
        assert service._state(conn_id).response_ownership.active() == replacement
        assert service._state(conn_id).response_ownership.is_current(old.response_epoch) is False
        lifecycle = service.take_deferred_settlement_events(conn_id)
        assert any(
            event.type == "pipeline.response"
            and event.response_epoch == old.response_epoch
            and event.state == "cancelled"
            and event.reason == "synthetic_final"
            for event in lifecycle
        )
    finally:
        service.unregister(conn_id)


def test_pending_producing_and_completed_lifecycle_events_share_one_epoch() -> None:
    """Local diagnostics receive every authoritative response-owner transition."""
    from queue import Queue

    service = RealtimeService(text_prompt_queue=Queue())
    conn_id = service.register()
    try:
        service.set_rendered_playback_ack_supported(conn_id, True)
        service.dispatch_pipeline_event(conn_id, SpeechStartedEvent())
        service.dispatch_pipeline_event(
            conn_id,
            SpeechStoppedEvent(duration_s=1.0, turn_id="turn_a", turn_revision=0),
        )
        pending = service.take_deferred_settlement_events(conn_id)
        owner = service._state(conn_id).response_ownership.active()
        assert owner is not None
        assert [(event.response_epoch, event.state) for event in pending] == [
            (owner.response_epoch, "pending")
        ]

        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(
                transcript="accepted input",
                turn_id="turn_a",
                turn_revision=0,
            ),
        )
        producing = service.take_deferred_settlement_events(conn_id)
        assert [(event.response_epoch, event.state) for event in producing] == [
            (owner.response_epoch, "producing")
        ]

        service.response._ensure_response(conn_id)
        service.finish_response(conn_id)
        completed = service.take_deferred_settlement_events(conn_id)
        assert completed[-1].response_epoch == owner.response_epoch
        assert completed[-1].state == "playback_pending"
    finally:
        service.unregister(conn_id)


def test_no_audio_tool_response_commits_dispatched_call_and_applies_output_once() -> None:
    """A tool already sent to the client is never rolled back or replayed."""
    service, conn_id = _claimed_service()
    try:
        owner = service.claim_pending_response(conn_id, turn_id="turn_tool", turn_revision=0)
        service.response._ensure_response(conn_id)
        st = service._state(conn_id)
        st.runtime_config.chat.add_item(
            RealtimeConversationItemFunctionCall(
                type="function_call",
                id="fc_once",
                call_id="call_once",
                name="test_tool",
                arguments="{}",
            )
        )
        st.pending_tool_call_ids.add("call_once")
        output = ConversationItemCreateEvent(
            type="conversation.item.create",
            item={
                "type": "function_call_output",
                "call_id": "call_once",
                "output": "done",
            },
        )
        assert service.handle_conversation_item_create(conn_id, output) == []

        service.response.mark_no_audio_terminal_if_needed(conn_id)
        service.finish_response(conn_id)
        service.finish_response(conn_id)

        types = [item.type for item in st.runtime_config.chat.buffer]
        assert types.count("function_call") == 1
        assert types.count("function_call_output") == 1
        assert st.pending_tool_call_ids == set()
        assert st.tool_followup_ready is True
        assert owner.response_epoch not in st.provisional_chat_checkpoints
    finally:
        service.unregister(conn_id)


def test_no_audio_terminal_rolls_back_text_history_that_never_became_audible() -> None:
    """A text transcript/response id is not a substitute for rendered PCM."""
    service, conn_id = _claimed_service()
    try:
        owner = service.claim_pending_response(conn_id, turn_id="turn_silent", turn_revision=0)
        state = service._state(conn_id)
        chat = state.runtime_config.chat
        chat.add_item(make_user_message("unheard input"))
        chat.add_item(make_assistant_message("unheard text-only transcript"))
        service.response._ensure_response(conn_id)

        service.response.mark_no_audio_terminal_if_needed(conn_id)
        service.finish_response(conn_id)

        assert "unheard input" not in chat.history_token_text()
        assert "unheard text-only transcript" not in chat.history_token_text()
        assert owner.response_epoch not in state.provisional_chat_checkpoints
        recorded = state.response_ownership.owner_for_response_epoch(owner.response_epoch)
        assert recorded is not None and recorded.state == "cancelled"
    finally:
        service.unregister(conn_id)


def test_manual_response_replaces_completed_unheard_transaction_before_new_claim() -> None:
    """A late manual create must not retain an unheard completed predecessor."""
    service, conn_id = _claimed_service()
    try:
        first = service.claim_manual_response(conn_id)
        state = service._state(conn_id)
        chat = state.runtime_config.chat
        chat.add_item(make_user_message("old unheard input"))
        chat.add_item(make_assistant_message("old unheard answer"))
        service.bind_response_id(conn_id, "resp_old")
        service.mark_response_completed(conn_id)

        second = service.claim_manual_response(conn_id)

        assert second.response_epoch > first.response_epoch
        assert "old unheard input" not in chat.history_token_text()
        assert "old unheard answer" not in chat.history_token_text()
        assert first.response_epoch not in state.provisional_chat_checkpoints
        assert second.response_epoch in state.provisional_chat_checkpoints
    finally:
        service.unregister(conn_id)


def test_empty_provider_result_emits_one_epoch_only_cancelled_terminal() -> None:
    """A claimed audio turn without model output still closes browser processing."""
    service, conn_id = _claimed_service()
    try:
        service.set_rendered_playback_ack_supported(conn_id, True)
        service.observe_speech_started(conn_id, reason="accepted_speech")
        owner = service.claim_pending_response(conn_id, turn_id="turn_empty", turn_revision=0)
        state = service._state(conn_id)
        state.input_audio_duration_s = 0.75
        service.mark_response_producing(conn_id)

        # A provider can legally terminate without yielding text, tools, or
        # audio. The audio worker may observe more than one terminal marker.
        service.response.mark_no_audio_terminal_if_needed(conn_id)
        first_finish = service.finish_response(conn_id)
        service.response.mark_no_audio_terminal_if_needed(conn_id)
        second_finish = service.finish_response(conn_id)
        lifecycle = service.take_deferred_settlement_events(conn_id)

        assert first_finish == []
        assert second_finish == []
        terminals = [event for event in lifecycle if event.type == "pipeline.response"]
        assert len(terminals) == 1
        terminal = terminals[0]
        assert terminal.input_epoch == owner.input_epoch
        assert terminal.response_epoch == owner.response_epoch
        assert terminal.response_id is None
        assert terminal.state == "cancelled"
        assert terminal.reason == "no_audio"
        assert state.response_pending is False
        assert owner.response_epoch not in state.provisional_chat_checkpoints
    finally:
        service.unregister(conn_id)


def test_duplicate_finish_cannot_rollback_completed_pcm_before_render_ack() -> None:
    service, conn_id = _claimed_service()
    try:
        service.set_rendered_playback_ack_supported(conn_id, True)
        owner = service.claim_pending_response(conn_id, turn_id="turn_pcm", turn_revision=0)
        state = service._state(conn_id)
        state.runtime_config.chat.add_item(make_user_message("Keep this heard input."))
        state.runtime_config.chat.add_item(make_assistant_message("Keep this queued answer."))
        service.encode_audio_chunk(conn_id, b"\x00\x00" * 160)
        active = state.response_ownership.active()
        assert active is not None and active.response_id is not None
        response_id = active.response_id

        service.finish_response(conn_id)
        history_after_first = list(state.runtime_config.chat.buffer)
        duplicate = service.finish_response(conn_id)

        terminal = state.response_ownership.active()
        assert duplicate == []
        assert terminal is not None and terminal.state == "completed"
        assert list(state.runtime_config.chat.buffer) == history_after_first
        assert service.handle_playback_started(
            conn_id, response_id=response_id, response_epoch=owner.response_epoch
        ) is not None
    finally:
        service.unregister(conn_id)


def test_nonbrowser_first_delivered_pcm_is_the_audible_compatibility_boundary() -> None:
    """Legacy/nonbrowser clients cannot send the local rendered-sample ack."""
    service = RealtimeService()
    conn_id = service.register()
    try:
        service.observe_speech_started(conn_id, reason="new_speech")
        owner = service.claim_pending_response(conn_id, turn_id="turn_pcm", turn_revision=0)

        events = service.encode_audio_chunk(conn_id, b"\x00\x00" * 160)

        event_types = [event.type for event in events]
        assert event_types == ["response.created", "response.output_audio.delta"]
        assert "pipeline.response" not in event_types
        assert service._state(conn_id).response_ownership.active().playback_started is False
        assert owner.response_epoch in service._state(conn_id).provisional_chat_checkpoints

        # The router invokes this only after the nonempty delta send succeeds.
        assert service.handle_first_delivered_pcm(conn_id) == []
        assert service._state(conn_id).response_ownership.active().playback_started is True
        assert owner.response_epoch not in service._state(conn_id).provisional_chat_checkpoints
    finally:
        service.unregister(conn_id)


def test_invalid_or_empty_pcm_never_commits_and_next_valid_chunk_settles_once() -> None:
    service = RealtimeService()
    conn_id = service.register()
    try:
        service.observe_speech_started(conn_id, reason="new_speech")
        owner = service.claim_pending_response(conn_id, turn_id="turn_retry", turn_revision=0)
        st = service._state(conn_id)

        with pytest.raises(ValueError, match="source_sample_rate"):
            service.encode_audio_chunk(conn_id, b"\x00\x00", source_sample_rate=0)
        assert st.current_response_id is None
        assert st.response_ownership.active().playback_started is False
        assert owner.response_epoch in st.provisional_chat_checkpoints

        empty_events = service.encode_audio_chunk(conn_id, b"")
        assert not any(
            event.type == "pipeline.response" and getattr(event, "state", None) == "audible"
            for event in empty_events
        )
        assert st.response_ownership.active().playback_started is False
        assert st.response_audio_emitted is False
        assert owner.response_epoch in st.provisional_chat_checkpoints

        valid_events = service.encode_audio_chunk(conn_id, b"\x00\x00" * 160)
        assert not any(event.type == "pipeline.response" for event in valid_events)
        assert st.response_ownership.active().playback_started is False
        assert owner.response_epoch in st.provisional_chat_checkpoints
        assert service.handle_first_delivered_pcm(conn_id) == []
        assert st.response_ownership.active().playback_started is True
        assert owner.response_epoch not in st.provisional_chat_checkpoints
        later_events = service.encode_audio_chunk(conn_id, b"\x00\x00" * 160)
        assert not any(
            event.type == "pipeline.response" and getattr(event, "state", None) == "audible"
            for event in later_events
        )
    finally:
        service.unregister(conn_id)


def test_pre_audible_rollback_restores_history_after_size_trim() -> None:
    """A provisional 31st turn cannot evict the oldest heard turn on rollback."""
    service, conn_id = _claimed_service()
    try:
        chat = service._state(conn_id).runtime_config.chat
        for index in range(chat.size):
            chat.add_item(make_user_message(f"heard user {index}"))
            chat.add_item(make_assistant_message(f"heard answer {index}"))

        service.claim_pending_response(conn_id, turn_id="turn_provisional", turn_revision=0)
        chat.add_item(make_user_message("unheard user"))
        chat.add_item(make_assistant_message("unheard answer"))
        chat.trim_if_needed()
        assert "heard user 0" not in chat.history_token_text()

        service.observe_speech_started(conn_id, reason="same_turn_revision")

        restored = chat.history_token_text()
        assert "heard user 0" in restored
        assert f"heard answer {chat.size - 1}" in restored
        assert "unheard user" not in restored
        assert "unheard answer" not in restored
    finally:
        service.unregister(conn_id)


def test_playback_ack_flushes_deferred_tool_output_and_starts_followup_once() -> None:
    """A queued tool continuation crosses the rendered-ack barrier exactly once."""
    from queue import Queue

    from speech_to_speech.pipeline.messages import GenerateResponseRequest

    prompt_queue: Queue = Queue()
    service = RealtimeService(text_prompt_queue=prompt_queue)
    conn_id = service.register()
    try:
        service.set_rendered_playback_ack_supported(conn_id, True)
        owner = service.claim_pending_response(conn_id, turn_id="turn_tool", turn_revision=0)
        service.bind_response_id(conn_id, "resp_tool")
        st = service._state(conn_id)
        call = RealtimeConversationItemFunctionCall(
            type="function_call",
            call_id="call_tool",
            name="lookup",
            arguments="{}",
        )
        st.runtime_config.chat.add_item(call)
        st.pending_tool_call_ids.add("call_tool")

        deferred = ConversationItemCreateEvent(
            type="conversation.item.create",
            item={
                "type": "function_call_output",
                "call_id": "call_tool",
                "output": '{"ok":true}',
            },
        )
        assert service.handle_conversation_item_create(conn_id, deferred) == []
        assert service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create")) is None
        service.mark_response_completed(conn_id)

        first = service.handle_playback_started(
            conn_id,
            response_id="resp_tool",
            response_epoch=owner.response_epoch,
        )
        events = service.take_deferred_settlement_events(conn_id)

        assert first is not None
        assert [event.type for event in events] == [
            "pipeline.response",
            "conversation.item.created",
            "response.created",
        ]
        assert events[0].state == "producing"
        queued = prompt_queue.get_nowait()
        assert isinstance(queued, GenerateResponseRequest)
        assert st.tool_followup_started is True
        assert service.handle_playback_started(
            conn_id,
            response_id="resp_tool",
            response_epoch=owner.response_epoch,
        ) is None
        assert prompt_queue.empty()
    finally:
        service.unregister(conn_id)


def test_pre_audible_terminal_holds_deferred_work_until_ack_or_newer_input() -> None:
    """Transport completion cannot launch a tool follow-up before audibility."""

    from queue import Queue

    prompt_queue: Queue = Queue()
    service = RealtimeService(text_prompt_queue=prompt_queue)
    conn_id = service.register()
    try:
        service.set_rendered_playback_ack_supported(conn_id, True)
        owner = service.claim_pending_response(conn_id, turn_id="turn_unheard", turn_revision=0)
        service.bind_response_id(conn_id, "resp_unheard")
        st = service._state(conn_id)
        st.in_response = True
        st.current_response_id = "resp_unheard"
        st.current_item_id = "item_unheard"
        st.response_audio_emitted = True
        st.tool_followup_ready = True
        st.tool_followup_requested = True
        st.tool_followup_response = None

        deferred = ConversationItemCreateEvent(
            type="conversation.item.create",
            item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": "new input"}]},
        )
        assert service.handle_conversation_item_create(conn_id, deferred) == []
        terminal = service.finish_response(conn_id)

        assert not any(event.type == "response.created" for event in terminal)
        assert len(st.deferred_items) == 1
        assert st.tool_followup_started is False
        assert prompt_queue.empty()

        # A confirmed newer input supersedes the unheard transaction before an
        # acknowledgement can start its queued continuation.
        service.audio.on_speech_started(conn_id, SpeechStartedEvent())

        assert st.response_ownership.is_current(owner.response_epoch) is False
        assert st.tool_followup_ready is False
        assert st.tool_followup_requested is False
        assert st.tool_followup_started is False
        assert prompt_queue.empty()
    finally:
        service.unregister(conn_id)


def test_early_playback_ack_waits_for_terminal_before_deferred_history_flush() -> None:
    """First rendered PCM commits audibility, not unfinished model write-back."""

    service = RealtimeService()
    conn_id = service.register()
    try:
        service.set_rendered_playback_ack_supported(conn_id, True)
        owner = service.claim_pending_response(conn_id, turn_id="turn_early_ack", turn_revision=0)
        response_id, _ = service.response._ensure_response(conn_id)
        st = service._state(conn_id)
        deferred = ConversationItemCreateEvent(
            type="conversation.item.create",
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "arrived during response"}],
            },
        )
        assert service.handle_conversation_item_create(conn_id, deferred) == []
        checkpoint_size = len(st.runtime_config.chat.buffer)

        acknowledged = service.handle_playback_started(
            conn_id,
            response_id=response_id,
            response_epoch=owner.response_epoch,
        )

        assert acknowledged is not None
        assert st.response_ownership.active().playback_started is True
        assert owner.response_epoch not in st.provisional_chat_checkpoints
        assert len(st.deferred_items) == 1
        assert len(st.runtime_config.chat.buffer) == checkpoint_size
        assert not any(
            event.type == "conversation.item.created"
            for event in service.take_deferred_settlement_events(conn_id)
        )

        terminal = service.finish_response(conn_id)
        assert len(st.deferred_items) == 0
        assert len(st.runtime_config.chat.buffer) == checkpoint_size + 1
        assert sum(event.type == "conversation.item.created" for event in terminal) == 1
        assert service.finish_response(conn_id) == []
    finally:
        service.unregister(conn_id)


def test_manual_response_create_cannot_duplicate_a_pending_direct_audio_turn() -> None:
    """Pending ownership is active before direct Gemma has a protocol response."""
    from queue import Queue

    text_prompt_queue: Queue = Queue()
    service = RealtimeService(text_prompt_queue=text_prompt_queue)
    conn_id = service.register()
    try:
        service.observe_speech_started(conn_id, reason="new_speech")
        owner = service.claim_pending_response(conn_id, turn_id="turn_direct", turn_revision=0)

        result = service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create"))

        assert result is not None
        assert result.type == "error"
        assert result.error.type == "conversation_already_has_active_response"
        assert text_prompt_queue.empty()
        assert service._state(conn_id).response_ownership.active() == owner
        assert service._state(conn_id).response_pending is True
    finally:
        service.unregister(conn_id)


def test_queued_non_interrupt_turn_waits_then_reaches_gemma_exactly_once() -> None:
    """Accepted successor audio is deferred, not dropped, behind audible output."""
    from threading import Event, Thread
    from types import SimpleNamespace

    import numpy as np

    from speech_to_speech.STT.gemma_audio_handler import GemmaAudioSTTHandler

    service = RealtimeService()
    conn_id = service.register()
    try:
        old = service.claim_pending_response(conn_id, turn_id="turn_old", turn_revision=0)
        response_id, _ = service.response._ensure_response(conn_id)
        assert service.handle_playback_started(
            conn_id,
            response_id=response_id,
            response_epoch=old.response_epoch,
        ) is not None
        state = service._state(conn_id)
        service.observe_speech_started(
            conn_id,
            reason="new_speech",
            interrupt_response=False,
        )
        queued = service.claim_pending_response(conn_id, turn_id="turn_new", turn_revision=0)
        assert state.response_ownership.is_queued(queued.response_epoch)

        wait_entered = Event()
        wait_for_activation = state.runtime_config.local_pipeline["_wait_response_epoch_current"]
        state.runtime_config.local_pipeline["_wait_response_epoch_current"] = (
            lambda epoch: (wait_entered.set(), wait_for_activation(epoch))[1]
        )
        requests: list[int] = []
        outputs: list[object] = []
        handler = object.__new__(GemmaAudioSTTHandler)
        handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=False)

        def fake_process_direct(vad_audio, generation):
            requests.append(vad_audio.response_epoch)
            yield SimpleNamespace(text="one response", response_epoch=vad_audio.response_epoch)

        handler._process_direct = fake_process_direct
        vad_audio = SimpleNamespace(
            audio=np.zeros(1600, dtype=np.float32),
            mode="final",
            runtime_config=state.runtime_config,
            turn_id="turn_new",
            turn_revision=0,
            created_at_s=0.0,
            input_epoch=queued.input_epoch,
            response_epoch=queued.response_epoch,
            response_id=None,
        )
        worker = Thread(target=lambda: outputs.extend(handler.process(vad_audio)), daemon=True)
        worker.start()
        assert wait_entered.wait(1.0)
        assert requests == []

        service.finish_response(conn_id)
        worker.join(1.0)

        assert not worker.is_alive()
        assert requests == [queued.response_epoch]
        assert len(outputs) == 1
        assert outputs[0].text == "one response"
    finally:
        service.unregister(conn_id)


def test_audible_retained_response_keeps_its_token_usage_after_newer_input() -> None:
    """Speculative input identity cannot revoke metrics from the active owner."""

    tracker = SpeculativeTurnTracker()
    service = RealtimeService(speculative_turns=tracker)
    conn_id = service.register()
    try:
        tracker.observe("turn_old", 0)
        owner = service.claim_pending_response(conn_id, turn_id="turn_old", turn_revision=0)
        response_id, _ = service.response._ensure_response(conn_id)
        assert service.handle_playback_started(
            conn_id,
            response_id=response_id,
            response_epoch=owner.response_epoch,
        ) is not None

        service.observe_speech_started(
            conn_id,
            reason="new_speech",
            interrupt_response=False,
        )
        tracker.observe("turn_old", 1)
        assert not tracker.is_latest("turn_old", 0)

        assert service.dispatch_pipeline_event(
            conn_id,
            TokenUsageEvent(
                input_tokens=17,
                output_tokens=23,
                turn_id="turn_old",
                turn_revision=0,
                input_epoch=owner.input_epoch,
                response_epoch=owner.response_epoch,
                response_id=response_id,
            ),
        ) == []
        usage = service._state(conn_id).response_usage
        assert (usage.input_tokens, usage.output_tokens) == (17, 23)
    finally:
        service.unregister(conn_id)


def test_promoted_queue_refreshes_checkpoint_before_pre_audible_rollback() -> None:
    """Rolling back the queued successor cannot erase the heard owner's late history."""

    service = RealtimeService()
    conn_id = service.register()
    try:
        old = service.claim_pending_response(conn_id, turn_id="turn_old", turn_revision=0)
        response_id, _ = service.response._ensure_response(conn_id)
        assert service.handle_playback_started(
            conn_id,
            response_id=response_id,
            response_epoch=old.response_epoch,
        ) is not None
        state = service._state(conn_id)
        service.observe_speech_started(
            conn_id,
            reason="new_speech",
            interrupt_response=False,
        )
        queued = service.claim_pending_response(conn_id, turn_id="turn_queued", turn_revision=0)
        assert state.response_ownership.is_queued(queued.response_epoch)

        # These commits occur after the successor's original claim-time
        # checkpoint, while the audible owner is still finishing.
        state.runtime_config.chat.add_item(make_user_message("heard old input"))
        state.runtime_config.chat.add_item(make_assistant_message("audible old tail"))

        service.finish_response(conn_id)
        assert state.response_ownership.is_current(queued.response_epoch)
        service.observe_speech_started(conn_id, reason="supersede_queued")

        history = state.runtime_config.chat.history_token_text()
        assert "heard old input" in history
        assert "audible old tail" in history
    finally:
        service.unregister(conn_id)


def test_render_ack_capable_browser_stays_pre_audible_until_worklet_ack() -> None:
    """Browser queueing PCM is not playback when its worklet can acknowledge render."""
    service = RealtimeService()
    conn_id = service.register()
    try:
        service.set_rendered_playback_ack_supported(conn_id, True)
        service.observe_speech_started(conn_id, reason="new_speech")
        owner = service.claim_pending_response(conn_id, turn_id="turn_browser", turn_revision=0)

        service.encode_audio_chunk(conn_id, b"\x00\x00" * 160)

        current = service._state(conn_id).response_ownership.active()
        assert current.playback_started is False
        assert owner.response_epoch in service._state(conn_id).provisional_chat_checkpoints

        response_id = service._state(conn_id).current_response_id
        assert response_id is not None
        assert service.handle_playback_started(
            conn_id,
            response_id=response_id,
            response_epoch=owner.response_epoch,
        ) is not None
        assert owner.response_epoch not in service._state(conn_id).provisional_chat_checkpoints
    finally:
        service.unregister(conn_id)


def test_audible_noninterrupt_capture_keeps_client_items_deferred_until_terminal() -> None:
    """A queued successor cannot reorder Chat while its audible owner writes back."""

    service, conn_id = _claimed_service()
    try:
        owner = service.claim_pending_response(conn_id, turn_id="turn_a", turn_revision=0)
        response_id, _ = service.response._ensure_response(conn_id)
        assert service.handle_playback_started(
            conn_id,
            response_id=response_id,
            response_epoch=owner.response_epoch,
        ) is not None
        st = service._state(conn_id)
        deferred = ConversationItemCreateEvent(
            type="conversation.item.create",
            item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": "for B"}]},
        )
        assert service.handle_conversation_item_create(conn_id, deferred) == []
        assert len(st.deferred_items) == 1

        service.audio.on_speech_started(
            conn_id,
            SpeechStartedEvent(turn_id="turn_b", turn_revision=0, interrupt_response=False),
        )

        assert len(st.deferred_items) == 1
        assert "for B" not in st.runtime_config.chat.history_token_text()
        service.finish_response(conn_id)
        assert len(st.deferred_items) == 0
        assert "for B" in st.runtime_config.chat.history_token_text()
    finally:
        service.unregister(conn_id)


def test_delayed_old_stop_cannot_release_newer_capture_gate() -> None:
    """A retained audible A stop cannot make response.create race B capture."""

    service, conn_id = _claimed_service()
    try:
        owner_a = service.claim_pending_response(conn_id, turn_id="turn_a", turn_revision=0)
        response_id, _ = service.response._ensure_response(conn_id)
        assert service.handle_playback_started(
            conn_id,
            response_id=response_id,
            response_epoch=owner_a.response_epoch,
        ) is not None
        supersession = service.observe_speech_started(
            conn_id,
            reason="new_speech",
            interrupt_response=False,
        )
        st = service._state(conn_id)
        assert st.input_capture_pending is True

        service.audio.on_speech_stopped(
            conn_id,
            SpeechStoppedEvent(
                duration_s=1.0,
                turn_id="turn_a",
                turn_revision=0,
                input_epoch=owner_a.input_epoch,
                response_epoch=owner_a.response_epoch,
            ),
        )
        assert st.input_capture_pending is True
        blocked = service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create"))
        assert blocked is not None and blocked.type == "error"
        assert blocked.error.type == "input_audio_buffer_capture_in_progress"

        service.audio.on_speech_stopped(
            conn_id,
            SpeechStoppedEvent(
                duration_s=1.0,
                turn_id="turn_b",
                turn_revision=0,
                input_epoch=supersession.input_epoch,
            ),
        )
        assert st.input_capture_pending is False
        queued_b = st.response_ownership.owner_for_turn("turn_b", 0)
        assert queued_b is not None
        assert st.response_ownership.is_queued(queued_b.response_epoch)
    finally:
        service.unregister(conn_id)


def test_tool_followup_is_consumed_while_new_speech_capture_is_pending() -> None:
    """Accepted B speech outranks A's hosted-style tool response delimiter."""

    from queue import Queue

    prompt_queue: Queue = Queue()
    service = RealtimeService(text_prompt_queue=prompt_queue)
    conn_id = service.register()
    try:
        owner_a = service.claim_pending_response(conn_id, turn_id="turn_a", turn_revision=0)
        response_id, _ = service.response._ensure_response(conn_id)
        st = service._state(conn_id)
        st.response_audio_emitted = True
        call = RealtimeConversationItemFunctionCall(
            type="function_call",
            call_id="call_a",
            name="inspect",
            arguments="{}",
        )
        st.runtime_config.chat.add_item(call)
        st.pending_tool_call_ids.add("call_a")
        assert service.handle_playback_started(
            conn_id,
            response_id=response_id,
            response_epoch=owner_a.response_epoch,
        ) is not None
        service.finish_response(conn_id)

        supersession = service.observe_speech_started(
            conn_id,
            reason="new_speech",
            interrupt_response=False,
        )
        output = ConversationItemCreateEvent(
            type="conversation.item.create",
            item={"type": "function_call_output", "call_id": "call_a", "output": '{"ok":true}'},
        )
        assert service.handle_conversation_item_create(conn_id, output)
        assert st.tool_followup_ready is True

        assert service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create")) is None
        assert prompt_queue.empty()
        assert st.tool_followup_requested is False
        assert st.tool_followup_ready is False

        service.audio.on_speech_stopped(
            conn_id,
            SpeechStoppedEvent(
                duration_s=1.0,
                turn_id="turn_b",
                turn_revision=0,
                input_epoch=supersession.input_epoch,
            ),
        )
        owner_b = st.response_ownership.owner_for_turn("turn_b", 0)
        assert owner_b is not None and st.response_ownership.is_current(owner_b.response_epoch)
        assert prompt_queue.empty()
        history = st.runtime_config.chat.history_token_text()
        assert "inspect" in history
        assert "ok" in history
    finally:
        service.unregister(conn_id)


def test_deferred_tool_followup_cannot_start_between_new_speech_start_and_stop() -> None:
    """A terminal cannot launch its continuation inside B's capture window."""

    from queue import Queue

    prompt_queue: Queue = Queue()
    service = RealtimeService(text_prompt_queue=prompt_queue)
    conn_id = service.register()
    try:
        owner_a = service.claim_pending_response(conn_id, turn_id="turn_a", turn_revision=0)
        response_id, _ = service.response._ensure_response(conn_id)
        st = service._state(conn_id)
        st.response_audio_emitted = True
        call = RealtimeConversationItemFunctionCall(
            type="function_call",
            call_id="call_a_capture_gap",
            name="inspect",
            arguments="{}",
        )
        st.runtime_config.chat.add_item(call)
        st.pending_tool_call_ids.add(call.call_id)
        assert service.handle_playback_started(
            conn_id,
            response_id=response_id,
            response_epoch=owner_a.response_epoch,
        ) is not None

        deferred_output = ConversationItemCreateEvent(
            type="conversation.item.create",
            item={
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": '{"ok":true}',
            },
        )
        assert service.handle_conversation_item_create(conn_id, deferred_output) == []
        assert service.handle_response_create(
            conn_id,
            ResponseCreateEvent(type="response.create"),
        ) is None
        assert st.tool_followup_requested is True

        supersession = service.observe_speech_started(
            conn_id,
            reason="turn_b",
            interrupt_response=False,
        )
        assert st.input_capture_pending is True

        service.finish_response(conn_id)

        assert prompt_queue.empty()
        assert st.tool_followup_started is False
        assert st.tool_followup_requested is True
        assert st.tool_followup_ready is True
        assert st.pending_tool_call_ids == set()

        service.audio.on_speech_stopped(
            conn_id,
            SpeechStoppedEvent(
                duration_s=1.0,
                turn_id="turn_b",
                turn_revision=0,
                input_epoch=supersession.input_epoch,
            ),
        )
        owner_b = st.response_ownership.owner_for_turn("turn_b", 0)

        assert owner_b is not None
        assert st.response_ownership.is_current(owner_b.response_epoch)
        assert st.response_pending is True
        assert prompt_queue.empty(), "A's tool continuation must not become a second model request"
        assert st.tool_followup_requested is False
        assert st.tool_followup_ready is False
        history = st.runtime_config.chat.history_token_text()
        assert "inspect" in history
        assert "ok" in history
    finally:
        service.unregister(conn_id)


def test_deferred_settlement_drain_cannot_erase_concurrent_append() -> None:
    """Producer and router copy/clear share one ownership transaction."""

    from threading import Event, Thread

    service, conn_id = _claimed_service()
    try:
        entered = Event()
        release = Event()

        class BlockingList(list):
            def __iter__(self):
                entered.set()
                assert release.wait(1.0)
                return super().__iter__()

        st = service._state(conn_id)
        st.deferred_settlement_events = BlockingList(["old"])
        drained: list[list[object]] = []
        drain_thread = Thread(
            target=lambda: drained.append(service.take_deferred_settlement_events(conn_id)),
            daemon=True,
        )
        drain_thread.start()
        assert entered.wait(1.0)
        writer = Thread(
            target=lambda: service.queue_deferred_settlement_events(conn_id, "new"),
            daemon=True,
        )
        writer.start()
        release.set()
        drain_thread.join(1.0)
        writer.join(1.0)

        assert drained == [["old"]]
        assert service.take_deferred_settlement_events(conn_id) == ["new"]
    finally:
        service.unregister(conn_id)
