from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from openai.types.realtime import (
    RealtimeResponse,
    ResponseAudioDoneEvent,
    ResponseAudioTranscriptDoneEvent,
    ResponseCreatedEvent,
    ResponseCreateEvent,
    ResponseDoneEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)
from openai.types.realtime.realtime_response import Audio, AudioOutput
from openai.types.realtime.realtime_response_status import RealtimeResponseStatus
from openai.types.realtime.realtime_response_usage import RealtimeResponseUsage

from speech_to_speech.api.openai_realtime.handlers.base import RealtimeBaseHandler
from speech_to_speech.LLM.chat import ChatItemError
from speech_to_speech.pipeline.events import AssistantTextEvent
from speech_to_speech.pipeline.messages import GenerateResponseRequest
from speech_to_speech.pipeline.response_ownership import response_epoch_history_transaction
from speech_to_speech.utils.utils import _generate_id, is_out_of_band, response_wants_audio

if TYPE_CHECKING:
    from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams

    from speech_to_speech.api.openai_realtime.service import ServerEvent, _ResponseStatus, _StatusReason

logger = logging.getLogger(__name__)


class ResponseHandler(RealtimeBaseHandler):
    """Owns the response lifecycle: create, cancel, finish, and ID management."""

    # ── ID / state helpers ────────────────────────

    def _ensure_response(self, conn_id: str) -> tuple[str, str]:
        """Ensure a response and output item exist, creating them if needed."""
        st = self._state(conn_id)
        if st.current_response_id is None:
            st.current_response_id = _generate_id("resp")
            self._start_item(conn_id)
            st.in_response = True
            st.response_audio_emitted = False
            self._service.bind_response_id(conn_id, st.current_response_id)
        st.response_pending = False
        return st.current_response_id, self._current_item_id(conn_id)

    def _end_response(
        self,
        conn_id: str,
        status: _ResponseStatus = "completed",
        *,
        ownership_reason: str | None = None,
    ) -> None:
        st = self._state(conn_id)
        if status == "cancelled":
            st.response_usage.responses_cancelled += 1
        else:
            st.response_usage.responses_completed += 1
        self._service.total_usage += st.response_usage
        logger.info(
            "Response done (status=%s) — this response: input_tokens=%d, output_tokens=%d, audio=%.2fs"
            " | cumulative: input_tokens=%d, output_tokens=%d, audio=%.2fs",
            status,
            st.response_usage.input_tokens,
            st.response_usage.output_tokens,
            st.response_usage.audio_duration_s,
            self._service.total_usage.input_tokens,
            self._service.total_usage.output_tokens,
            self._service.total_usage.audio_duration_s,
        )
        st.response_usage.reset()
        owner = st.response_ownership.active()
        wants_audio = response_wants_audio(st.current_response_params)
        if status != "completed":
            # The public Realtime status reason has a deliberately small
            # upstream-compatible vocabulary.  Keep richer local causes on
            # the ownership lifecycle without fabricating a protocol reason.
            self._service.mark_response_cancelled(conn_id, reason=ownership_reason or status)
        else:
            self._service.mark_response_completed(conn_id, reason=status)
            if owner is not None and not wants_audio:
                self._service.settle_provisional_response(conn_id, owner.response_epoch)
        st.current_response_id = None
        st.current_item_id = None
        st.content_index = 0
        st.in_response = False
        st.response_pending = False
        st.response_audio_emitted = False
        # The next response may negotiate a new provider/transport clock, but
        # an active response was pinned when ownership was claimed, before
        # Gemma/TTS or its first PCM chunk started.
        st.response_output_sample_rate = None
        st.current_response_params = None
        st.pending_output_text_parts = []
        if status == "completed":
            self._service.maybe_schedule_history_maintenance(conn_id)
        # A queued user turn must not overtake an external tool transaction
        # already dispatched by this audible response.  Its matching output is
        # a promotion barrier; ConversationHandler commits it first and then
        # promotes the queued speech as the one successor response.
        if not st.pending_tool_call_ids:
            if st.tool_followup_ready and st.response_ownership.has_queued():
                self._service.promote_queued_after_tool_transaction(conn_id, drain_events=False)
            else:
                self._service.activate_next_queued_response(conn_id)

    def _start_item(self, conn_id: str) -> str:
        """Generate a new item ID, reset content index, and store it."""
        st = self._state(conn_id)
        item_id = _generate_id("item")
        st.current_item_id = item_id
        st.content_index = 0
        st.input_audio_duration_s = 0.0
        return item_id

    def _current_item_id(self, conn_id: str) -> str:
        return self._state(conn_id).current_item_id or self._start_item(conn_id)

    def _next_content_index(self, conn_id: str) -> int:
        """Return the current content index and advance it."""
        st = self._state(conn_id)
        idx = st.content_index
        st.content_index += 1
        return idx

    def _build_response(
        self,
        conn_id: str,
        status: _ResponseStatus,
        reason: _StatusReason | None = None,
    ) -> RealtimeResponse:
        """Build a fully-populated RealtimeResponse from the current connection state."""
        st = self._state(conn_id)
        status_details = None
        if reason or status in ("completed", "cancelled", "incomplete", "failed"):
            status_details = RealtimeResponseStatus(type=status, reason=reason)  # type: ignore[arg-type]

        rp = st.current_response_params
        metadata = rp.metadata if rp and rp.metadata else None

        voice: Optional[str] = None
        owner = st.response_ownership.active()
        frozen = (
            st.runtime_config.response_synthesis_configs.get(owner.response_epoch)
            if owner is not None
            else None
        )
        # Presence is authoritative. A frozen ``None`` means the TTS handler's
        # own default, not permission to reread a session voice changed after
        # this response was admitted.
        if isinstance(frozen, dict) and "voice" in frozen:
            frozen_voice = frozen.get("voice")
            voice = str(frozen_voice) if frozen_voice else None
        else:
            if rp and rp.audio and rp.audio.output and rp.audio.output.voice:
                voice = str(rp.audio.output.voice)
            if not voice:
                audio_cfg = st.runtime_config.session.audio
                audio_output = audio_cfg.output if audio_cfg is not None else None
                voice = str(audio_output.voice) if audio_output is not None and audio_output.voice else None

        # Out-of-band responses are not threaded into any conversation: report a null id.
        conversation_id = None if is_out_of_band(rp) else st.conversation_id

        return RealtimeResponse(
            id=st.current_response_id,
            object="realtime.response",
            status=status,
            status_details=status_details,
            audio=Audio(output=AudioOutput(voice=str(voice) if voice else None)),  # type: ignore[arg-type]
            conversation_id=conversation_id,
            metadata=metadata,
            usage=RealtimeResponseUsage(
                input_tokens=st.response_usage.input_tokens,
                output_tokens=st.response_usage.output_tokens,
                total_tokens=st.response_usage.input_tokens + st.response_usage.output_tokens,
            ),
        )

    # ── Public handlers ───────────────────────────

    def handle_response_create(self, conn_id: str, event: ResponseCreateEvent) -> ServerEvent | None:
        """Trigger a response.

        Returns a ``ResponseCreatedEvent`` on success, a ``RealtimeErrorEvent``
        on failure, or ``None`` if there is no text_prompt_queue.
        """
        st = self._state(conn_id)
        # Manual admission, optional response.input history, presentation
        # binding, and queue publication are one short ownership transaction.
        # Without this guard a VAD thread could claim a newer direct-audio turn
        # after the idle check but before _start_generation(), causing this
        # response.create to bind itself to that unrelated owner.
        with st.response_ownership.transaction():
            return self._handle_response_create_locked(conn_id, event)

    def _handle_response_create_locked(self, conn_id: str, event: ResponseCreateEvent) -> ServerEvent | None:
        """Handle ``response.create`` while the ownership transaction is held."""
        st = self._state(conn_id)
        if st.suppress_next_tool_followup_create:
            st.suppress_next_tool_followup_create = False
            logger.info("Consumed superseded tool response.create; accepted speech already owns the successor")
            return None
        if st.tool_followup_started:
            return self.make_error(
                message="The tool follow-up response has already been requested.",
                _type="duplicate_tool_followup",
            )
        tool_transaction = bool(st.pending_tool_call_ids or st.tool_followup_ready)
        if tool_transaction:
            # Hosted-compatible ordering sends response.create immediately after
            # function_call_output. Queue it behind the originating response and
            # optional deferred camera image; the backend owns the barrier.
            if st.tool_followup_started or st.tool_followup_requested:
                logger.info("Ignoring duplicate response.create for the active tool transaction")
                return None
            st.tool_followup_requested = True
            st.tool_followup_response = event.response
            if st.tool_followup_ready and not st.in_response and not st.pending_tool_call_ids:
                if st.input_capture_pending:
                    # New speech has crossed the VAD admission boundary but its
                    # stop has not claimed a response yet.  Commit the completed
                    # call/output/image transaction and consume this hosted-style
                    # delimiter; starting the old tool continuation here would
                    # occupy the sole model slot ahead of the accepted user turn.
                    self._service.queue_deferred_settlement_events(
                        conn_id,
                        self._service.conversation.flush_tool_transaction_tail(conn_id),
                    )
                    st.tool_followup_requested = False
                    st.tool_followup_response = None
                    st.tool_followup_ready = False
                    st.tool_followup_started = False
                    logger.info(
                        "Consumed tool response.create while accepted speech capture is pending"
                    )
                    return None
                if st.response_ownership.has_queued():
                    # response.create is the protocol delimiter after the
                    # function output and optional camera image. Commit that
                    # entire transaction before promoting accepted speech, and
                    # consume this create instead of generating a second answer.
                    self._service.queue_deferred_settlement_events(
                        conn_id,
                        self._service.conversation.flush_tool_transaction_tail(conn_id),
                    )
                    self._service.promote_queued_after_tool_transaction(
                        conn_id,
                        drain_events=False,
                    )
                    return None
                return self._start_generation(conn_id, event.response, tool_followup=True)
            logger.info("Queued one response.create for the active tool transaction")
            return None
        if event.response:
            if event.response.tool_choice and not isinstance(event.response.tool_choice, str):
                return self.make_error(
                    message="Only string tool_choice values are supported for now (auto, required, none).",
                    _type="tool_choice_not_supported",
                )
        if st.input_capture_pending:
            return self.make_error(
                message="Cannot create a response while accepted speech capture is still in progress.",
                _type="input_audio_buffer_capture_in_progress",
            )
        owner = st.response_ownership.active()
        owner_blocks_manual_create = owner is not None and owner.state in {"pending", "producing", "priming"}
        if st.in_response or st.response_pending or owner_blocks_manual_create:
            return self.make_error(
                # Preserve the public error wording for existing OpenAI
                # Realtime clients; authoritative ownership now broadens what
                # counts as "in progress" before a protocol response exists.
                message="Cannot create response while another response is in progress.",
                _type="conversation_already_has_active_response",
            )

        out_of_band = is_out_of_band(event.response)

        # In-band: response.input items are added to the default conversation here so
        # they appear in history. Out-of-band: leave the default conversation untouched —
        # the input rides along on the request and seeds a throwaway chat in the LM.
        if not out_of_band and event.response and event.response.input:
            owner = self._service.claim_manual_response(conn_id, response=event.response)
            with response_epoch_history_transaction(
                runtime_config=st.runtime_config,
                response_epoch=owner.response_epoch,
            ) as admitted:
                if not admitted:
                    return self.make_error(
                        message="The response was superseded before its input could be committed.",
                        _type="response_superseded",
                    )
                for input_item in event.response.input:
                    try:
                        self._service.conversation._append_item(conn_id, input_item)
                    except ChatItemError as exc:
                        # The checkpoint predates every response.input mutation,
                        # so one invalid item rolls the entire in-band input back
                        # and leaves the next response.create usable.
                        self._service.mark_response_cancelled(
                            conn_id,
                            reason="invalid_input_item",
                        )
                        st.response_pending = False
                        st.provisional_chat_checkpoints.pop(owner.response_epoch, None)
                        return self.make_error(message=str(exc), _type="invalid_input_item")

        logger.debug("response.create received, LLM generation triggered")
        return self._start_generation(conn_id, event.response, tool_followup=False)

    def handle_response_cancel(self, conn_id: str) -> list[ServerEvent]:
        """Cancel the in-progress response and re-enable listening."""
        events = self.finish_response(conn_id, status="cancelled", reason="client_cancelled")
        should_listen = self._should_listen(conn_id)
        if should_listen:
            should_listen.set()
        logger.info("Response cancelled, listening re-enabled")
        return events

    def handle_response_cancel_if_owner(
        self,
        conn_id: str,
        *,
        input_epoch: int | None,
        response_epoch: int | None,
        response_id: str | None,
    ) -> list[ServerEvent] | None:
        """Cancel only the response identity captured with a client request.

        Transport cancellation is allowed to wait. The final presentation
        mutation is therefore guarded by the captured identity so a successor
        admitted during that wait cannot be closed by the older request.
        """

        st = self._state(conn_id)
        with st.response_ownership.transaction():
            if response_epoch is not None and not st.response_ownership.matches_active_identity(
                input_epoch=input_epoch,
                response_epoch=response_epoch,
                response_id=response_id,
                require_output_admission=False,
            ):
                return None
            return self.handle_response_cancel(conn_id)

    def _cancel_owner_without_protocol_response(self, conn_id: str, *, reason: str) -> None:
        """Cancel an owner that never produced an OpenAI response, exactly once.

        A positive-duration accepted turn can reach its terminal without one
        assistant text chunk, tool call, or PCM block. Such a turn has no
        ``response_id`` and therefore cannot carry a stock ``response.done``.
        Browser clients still need the local epoch lifecycle terminal to leave
        their pending/processing state. Queue it only on the first transition;
        repeated audio sentinels or cleanup calls remain idempotent.
        """
        st = self._state(conn_id)
        before = st.response_ownership.active()
        if (
            before is None
            or before.response_id is not None
            or not st.response_ownership.admits_output(before.response_epoch)
        ):
            return
        cancelled = self._service.mark_response_cancelled(conn_id, reason=reason)
        st.response_pending = False
        if cancelled is not None:
            owner_event = self._service.response_owner_event(conn_id)
            if owner_event is not None:
                self._service.queue_deferred_settlement_events(conn_id, owner_event)

    def mark_no_audio_terminal_if_needed(self, conn_id: str) -> None:
        """Rollback an audio response that reached its terminal with no PCM."""
        st = self._state(conn_id)
        if response_wants_audio(st.current_response_params) and not st.response_audio_emitted:
            owner = st.response_ownership.active()
            if owner is not None and st.pending_tool_call_ids:
                # The function call has already been dispatched to the client.
                # Commit it exactly once even though this response produced no
                # playable audio; rollback would orphan its matching output.
                self._service.settle_provisional_response(conn_id, owner.response_epoch)
            else:
                # Text may already have created the protocol response and
                # written assistant history, even though no PCM was ever
                # delivered.  A response id is therefore not evidence that
                # this transaction became audible.  Cancel through the
                # authoritative owner so its provisional checkpoint rolls
                # back before ``response.done`` clears the presentation flags.
                if owner is not None and owner.response_id is None:
                    self._cancel_owner_without_protocol_response(conn_id, reason="no_audio")
                else:
                    self._service.mark_response_cancelled(conn_id, reason="no_audio")

    def finish_audio_terminal_if_owner(
        self,
        conn_id: str,
        *,
        input_epoch: int | None,
        response_epoch: int | None,
        response_id: str | None,
    ) -> list[ServerEvent] | None:
        """Close one audio sentinel only while its captured owner is active.

        ``AUDIO_RESPONSE_DONE`` can wait behind text/tool dispatch. New speech
        may replace its response during that await, so the final comparison,
        no-audio rollback, and presentation closeout must be one transaction.
        ``None`` means the sentinel became stale; an empty list is a valid
        idempotent close for the matching owner.
        """

        st = self._state(conn_id)
        with st.response_ownership.transaction():
            if response_epoch is not None and not st.response_ownership.matches_active_identity(
                input_epoch=input_epoch,
                response_epoch=response_epoch,
                response_id=response_id,
            ):
                return None
            self.mark_no_audio_terminal_if_needed(conn_id)
            return self.finish_response(conn_id)

    def finish_response(
        self,
        conn_id: str,
        status: _ResponseStatus = "completed",
        reason: _StatusReason | None = None,
        *,
        ownership_reason: str | None = None,
    ) -> list[ServerEvent]:
        """Close the current response (audio/text done + response done).

        Audio responses emit ``response.output_audio.done`` for any terminal
        status. Text-only responses emit a single ``response.output_text.done``
        carrying the full streamed text, but only on ``status="completed"`` —
        a cancelled or failed text response sends no audio, so it just closes
        with ``response.done``.
        """
        st = self._state(conn_id)
        events: list[ServerEvent] = []
        # ``response.done`` is only a transport boundary for audio.  Capture
        # this before _end_response clears the per-response presentation
        # fields, so client inserts and queued tool continuations cannot become
        # irrevocable while a browser may still supersede unheard PCM.
        owner_before_terminal = st.response_ownership.active()
        await_audible_settlement = bool(
            status == "completed"
            and response_wants_audio(st.current_response_params)
            and st.response_audio_emitted
            and owner_before_terminal is not None
            and not owner_before_terminal.playback_started
        )
        if not st.in_response:
            owner = st.response_ownership.active()
            if owner is not None and status != "completed":
                self._cancel_owner_without_protocol_response(conn_id, reason=status)
            elif owner is not None and status == "completed":
                self._cancel_owner_without_protocol_response(conn_id, reason="no_audio")
        if st.in_response:
            resp_id, item_id = self._ensure_response(conn_id)
            if response_wants_audio(st.current_response_params):
                events.append(
                    ResponseAudioDoneEvent(
                        type="response.output_audio.done",
                        event_id=self._next_event_id(),
                        content_index=0,
                        item_id=item_id,
                        output_index=0,
                        response_id=resp_id,
                    )
                )
            elif status == "completed" and st.pending_output_text_parts:
                events.append(
                    ResponseTextDoneEvent(
                        type="response.output_text.done",
                        event_id=self._next_event_id(),
                        content_index=0,
                        item_id=item_id,
                        output_index=0,
                        response_id=resp_id,
                        text="".join(st.pending_output_text_parts),
                    )
                )
            events.append(
                ResponseDoneEvent(
                    type="response.done",
                    event_id=self._next_event_id(),
                    response=self._build_response(conn_id, status, reason),
                )
            )
            self._end_response(conn_id, status, ownership_reason=ownership_reason)
            owner_event = self._service.response_owner_event(conn_id)
            # activate_next_queued_response() already publishes the successor
            # lifecycle event. Do not append that same event a second time
            # merely because _end_response changed the active owner.
            if (
                owner_event is not None
                and (
                    owner_before_terminal is None
                    or owner_event.response_epoch == owner_before_terminal.response_epoch
                )
            ):
                self._service.queue_deferred_settlement_events(conn_id, owner_event)
            st.text_output_complete = False
            st.await_text_output_complete = False
            if st.current_response_is_tool_followup:
                st.current_response_is_tool_followup = False
                st.tool_followup_started = False
        # A browser-aware audio response stays rollbackable after transport
        # completion until its first rendered sample is acknowledged.  The
        # non-browser path invokes the same settlement through its first
        # successfully delivered PCM delta.  Cancellation/no-audio/text paths
        # have no such future acknowledgement and retain their normal drain.
        if not await_audible_settlement:
            # Apply any client items that arrived mid-generation now that
            # in_response is cleared and the generation's own write-back has
            # landed. Done outside the in_response guard so a stray terminal
            # call still drains the buffer.
            events.extend(self._service.conversation.flush_deferred_items(conn_id))
            if status == "completed":
                followup = self.start_queued_tool_followup_if_ready(conn_id)
                if followup is not None:
                    events.append(followup)
        else:
            logger.debug(
                "Holding deferred response settlement until playback acknowledgement "
                "response epoch=%d",
                owner_before_terminal.response_epoch,
            )
        if events and self._service.context_tokenizer_base_url:
            events.append(self._service.context_metric(conn_id, "committed"))
        return events

    def start_queued_tool_followup_if_ready(self, conn_id: str) -> ResponseCreatedEvent | None:
        """Start one requested tool continuation after its transaction commits."""
        st = self._state(conn_id)
        if not (
            st.tool_followup_requested
            and st.tool_followup_ready
            and not st.pending_tool_call_ids
            and not st.tool_followup_started
            and not st.in_response
            and not st.input_capture_pending
        ):
            return None
        return self._start_generation(
            conn_id,
            st.tool_followup_response,
            tool_followup=True,
        )

    # ── Pipeline event handlers ───────────────────

    def on_assistant_text(
        self,
        conn_id: str,
        event: AssistantTextEvent,
        *,
        wait_for_pending_reopen: bool = True,
        response_epoch_is_authoritative: bool = False,
    ) -> list[ServerEvent] | None:
        """Handle assistant_text: emit transcript and/or tool-call events."""
        st = self._state(conn_id)
        response_epoch = getattr(event, "response_epoch", None)
        epoch_owned = bool(
            response_epoch is not None
            and st.response_ownership.admits_output(response_epoch)
        )
        if self._service.speculative_turns and not (
            epoch_owned and response_epoch_is_authoritative
        ):
            commit_result: bool | None
            if wait_for_pending_reopen:
                commit_result = self._service.speculative_turns.commit_if_latest_after_reopen_grace(
                    event.turn_id,
                    event.turn_revision,
                )
            else:
                commit_result = self._service.speculative_turns.try_commit_if_latest_after_reopen_grace(
                    event.turn_id,
                    event.turn_revision,
                )
            if commit_result is None:
                return None
            if not commit_result:
                logger.debug("Dropping stale assistant text for turn=%s rev=%s", event.turn_id, event.turn_revision)
                return []
        events: list[ServerEvent] = []
        # Audio-producing turns are announced by AudioHandler. A tool-only
        # direct-audio turn has no audio, so it must announce its own response
        # lifecycle for the browser to wait for response.done before follow-up.
        response_was_missing = st.current_response_id is None
        resp_id, item_id = self._ensure_response(conn_id)
        if response_was_missing:
            events.append(
                ResponseCreatedEvent(
                    type="response.created",
                    event_id=self._next_event_id(),
                    response=self._build_response(conn_id, "in_progress"),
                )
            )
            owner_event = self._service.response_owner_event(conn_id)
            if owner_event is not None:
                events.append(owner_event)
        st.last_item_id = item_id
        output_idx = 0
        if event.text:
            if response_wants_audio(st.current_response_params):
                events.append(
                    ResponseAudioTranscriptDoneEvent(
                        type="response.output_audio_transcript.done",
                        event_id=self._next_event_id(),
                        content_index=0,
                        item_id=item_id,
                        output_index=output_idx,
                        response_id=resp_id,
                        transcript=event.text,
                    )
                )
            else:
                # Stream the delta now; the matching response.output_text.done is
                # emitted once at close in finish_response, carrying the per-chunk
                # parts collected here, space-joined ("" when there were none).
                st.pending_output_text_parts.append(event.text)
                events.append(
                    ResponseTextDeltaEvent(
                        type="response.output_text.delta",
                        event_id=self._next_event_id(),
                        content_index=0,
                        item_id=item_id,
                        output_index=output_idx,
                        response_id=resp_id,
                        delta=event.text,
                    )
                )
            output_idx += 1
        if event.tools:
            accepted_tool_calls = 0
            for tool in event.tools:
                if tool.call_id in st.pending_tool_call_ids or tool.call_id in st.completed_tool_call_ids:
                    logger.info("Ignoring duplicate function-call terminal event (call_id=%s)", tool.call_id)
                    continue
                st.pending_tool_call_ids.add(tool.call_id)
                accepted_tool_calls += 1
                st.tool_followup_ready = False
                st.tool_followup_started = False
                st.tool_followup_requested = False
                st.tool_followup_response = None
                events.append(
                    ResponseFunctionCallArgumentsDoneEvent(
                        type="response.function_call_arguments.done",
                        event_id=self._next_event_id(),
                        call_id=tool.call_id,
                        name=tool.name,
                        arguments=tool.arguments,
                        item_id=item_id,
                        output_index=output_idx,
                        response_id=resp_id,
                    )
                )
                output_idx += 1
            st.response_usage.tool_calls += accepted_tool_calls
        return events
    def _start_generation(
        self,
        conn_id: str,
        response: RealtimeResponseCreateParams | None,
        *,
        tool_followup: bool,
    ) -> ResponseCreatedEvent:
        st = self._state(conn_id)
        with st.response_ownership.transaction():
            owner = self._service.claim_manual_response(conn_id, response=response)
            producing = self._service.mark_response_producing(conn_id)
            if producing is not None:
                st.deferred_settlement_events.append(self._service._owner_event(conn_id, producing))
            out_of_band = is_out_of_band(response)
            st.in_response = True
            st.response_pending = False
            st.text_output_complete = False
            st.await_text_output_complete = True
            st.current_response_params = response
            st.current_response_id = _generate_id("resp")
            self._start_item(conn_id)
            self._service.bind_response_id(conn_id, st.current_response_id)
            owner = st.response_ownership.active() or owner
            st.current_response_is_tool_followup = tool_followup
            if tool_followup:
                st.tool_followup_ready = False
                st.tool_followup_started = True
                st.tool_followup_requested = False
                st.tool_followup_response = None
                logger.info("Tool follow-up generation started (stage=followup_start)")

            queue = self._queue(conn_id)
            if queue:
                queue.put(
                    GenerateResponseRequest(
                        runtime_config=st.runtime_config,
                        response=response,
                        turn_id=None if out_of_band else st.speculative_user_turn_id,
                        turn_revision=None if out_of_band else st.speculative_user_turn_revision,
                        speech_stopped_at_s=None if out_of_band else st.speculative_user_speech_stopped_at_s,
                        input_epoch=owner.input_epoch,
                        response_epoch=owner.response_epoch,
                        response_id=st.current_response_id,
                    )
                )
            return ResponseCreatedEvent(
                type="response.created",
                event_id=self._next_event_id(),
                response=self._build_response(conn_id, "in_progress"),
            )
