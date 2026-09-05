from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from openai.types.realtime import (
    ConversationItem,
    ConversationItemCreatedEvent,
    ConversationItemCreateEvent,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    ConversationItemInputAudioTranscriptionDeltaEvent,
)
from openai.types.realtime.conversation_item_input_audio_transcription_completed_event import (
    UsageTranscriptTextUsageDuration,
)

from speech_to_speech.api.openai_realtime.handlers.base import RealtimeBaseHandler
from speech_to_speech.LLM.chat import ChatItemError, add_supported_item
from speech_to_speech.pipeline.events import PartialTranscriptionEvent, TranscriptionCompletedEvent

if TYPE_CHECKING:
    from speech_to_speech.api.openai_realtime.service import ServerEvent

logger = logging.getLogger(__name__)


class ConversationHandler(RealtimeBaseHandler):
    """Owns conversation item injection and pipeline-to-protocol translation."""

    def handle_conversation_item_create(
        self,
        conn_id: str,
        event: ConversationItemCreateEvent,
    ) -> list[ServerEvent]:
        """Inject a text message or function-call output into the LLM context.

        Items are added to the LLM chat context but do NOT trigger response
        generation on their own.  A subsequent ``response.create`` event is
        required to trigger the model.

        While a response is generating, ordinary items are *deferred*: applying them now
        would race the LLM handler's end-of-turn chat write-back, which runs on
        the pipeline thread (e.g. a ``function_call_output`` arriving before its
        ``function_call`` is recorded, or an image stripped before the next
        turn reads it). Deferred items are flushed, in order, once the response
        completes — see :meth:`flush_deferred_items`.
        """
        st = self._state(conn_id)
        # The defer/apply decision and Chat mutation share the same ownership
        # lock used by response claim and rollback. Otherwise a client item can
        # be acknowledged just after a new checkpoint is captured and then be
        # erased when that new provisional response is superseded.
        with st.response_ownership.transaction():
            # Direct Gemma can own a provisional answer before an OpenAI response
            # object exists. Defer all client inserts until that transaction settles
            # so rollback cannot truncate unrelated next-turn items.
            provisional = bool(st.provisional_chat_checkpoints)
            is_tool_output = getattr(event.item, "type", None) == "function_call_output"
            call_id = getattr(event.item, "call_id", None) if is_tool_output else None
            matching_pending_output = bool(is_tool_output and call_id in st.pending_tool_call_ids)
            # A local model can publish the protocol function-call event before its
            # final Chat write-back records that call. Even a matching fast client
            # output must therefore wait while the originating response is still
            # active. Once it ends, only that matching output may pass a queued
            # successor's provisional checkpoint so call -> output ordering is
            # committed before the successor model is promoted.
            matching_output_unblocks_queued_successor = bool(
                matching_pending_output and st.response_ownership.has_queued()
            )
            if st.in_response or (
                provisional and not matching_output_unblocks_queued_successor
            ):
                st.deferred_items.append(event.item)
                logger.debug("Deferred conversation item until the provisional response settles")
                return []
            events = self._apply_item(conn_id, event.item)
            if matching_pending_output and not st.pending_tool_call_ids:
                # Do not promote a queued speech successor on the output item
                # alone. Browser tools may send an optional camera image after the
                # function output and use response.create as the transaction
                # delimiter. Promoting here would let Gemma snapshot Chat before
                # that image is present.
                if st.tool_followup_requested:
                    if st.response_ownership.has_queued():
                        events.extend(self.flush_tool_transaction_tail(conn_id))
                        events.extend(self._service.promote_queued_after_tool_transaction(conn_id))
                    else:
                        followup = self._service.response.start_queued_tool_followup_if_ready(conn_id)
                        if followup is not None:
                            events.append(followup)
            return events

    def _apply_item(self, conn_id: str, item: ConversationItem) -> list[ServerEvent]:
        """Add one item to the chat and build its ``conversation.item.created``."""
        st = self._state(conn_id)
        is_tool_output = getattr(item, "type", None) == "function_call_output"
        call_id = getattr(item, "call_id", None) if is_tool_output else None
        if call_id and call_id in st.completed_tool_call_ids:
            logger.info("Ignoring duplicate completed tool output (call_id=%s)", call_id)
            return [
                self.make_error(
                    message=f"Tool output for call_id '{call_id}' has already been accepted.",
                    _type="duplicate_tool_output",
                )
            ]
        try:
            self._append_item(conn_id, item)
        except ChatItemError as exc:
            return [self.make_error(str(exc), "invalid_conversation_item")]

        if not item:
            return []
        if is_tool_output:
            if call_id:
                st.completed_tool_call_ids.append(call_id)
                del st.completed_tool_call_ids[: -self._service.tool_call_tombstone_limit]
            if call_id in st.pending_tool_call_ids:
                st.pending_tool_call_ids.remove(call_id)
                logger.info(
                    "Tool output accepted (stage=output_ack call_id=%s pending=%d)",
                    call_id,
                    len(st.pending_tool_call_ids),
                )
                if not st.pending_tool_call_ids:
                    st.tool_followup_ready = True
                    logger.info("Tool transaction ready for one follow-up response (call_id=%s)", call_id)
        event = ConversationItemCreatedEvent(
            type="conversation.item.created",
            event_id=self._next_event_id(),
            previous_item_id=st.last_item_id,
            item=item,
        )
        st.last_item_id = item.id
        events: list[ServerEvent] = [event]
        if self._service.context_tokenizer_base_url:
            events.append(self._service.context_metric(conn_id, "updated"))
        return events

    def flush_deferred_items(self, conn_id: str) -> list[ServerEvent]:
        """Apply items buffered during a response, in arrival order.

        Called at response completion (after the generation's own write-back),
        so a ``function_call_output`` pairs with its now-recorded ``function_call``
        and an image survives the just-finished response's ``strip_images``.
        """
        st = self._state(conn_id)
        with st.response_ownership.transaction():
            if not st.deferred_items:
                return []
            if st.provisional_chat_checkpoints:
                # A queued speech successor owns its own checkpoint.  That must not
                # hold the audible predecessor's already-dispatched tool output
                # behind the successor it structurally precedes in Chat.
                items = [
                    item
                    for item in st.deferred_items
                    if getattr(item, "type", None) == "function_call_output"
                    and getattr(item, "call_id", None) in st.pending_tool_call_ids
                ]
                if not items:
                    return []
                selected_ids = {id(item) for item in items}
                st.deferred_items = [item for item in st.deferred_items if id(item) not in selected_ids]
            else:
                items = st.deferred_items
                st.deferred_items = []
            events: list[ServerEvent] = []
            for item in items:
                events.extend(self._apply_item(conn_id, item))
            if not st.pending_tool_call_ids and st.tool_followup_requested:
                if st.response_ownership.has_queued():
                    events.extend(self.flush_tool_transaction_tail(conn_id))
                    events.extend(self._service.promote_queued_after_tool_transaction(conn_id))
                else:
                    followup = self._service.response.start_queued_tool_followup_if_ready(conn_id)
                    if followup is not None:
                        events.append(followup)
            return events

    def flush_tool_transaction_tail(self, conn_id: str) -> list[ServerEvent]:
        """Commit items preceding the tool response.create delimiter.

        A queued user-speech response already owns a provisional checkpoint,
        so the general deferred flush intentionally selects only its matching
        function output. Once response.create arrives, any remaining image or
        ordinary items are known to belong before that successor. Apply them in
        wire order; queued-response promotion refreshes its checkpoint after
        this method returns.
        """

        st = self._state(conn_id)
        with st.response_ownership.transaction():
            items = st.deferred_items
            st.deferred_items = []
            events: list[ServerEvent] = []
            for item in items:
                events.extend(self._apply_item(conn_id, item))
            return events

    def _append_item(self, conn_id: str, item: ConversationItem) -> None:
        """Narrow ``ConversationItem`` to ``SupportedItem`` and delegate to ``Chat.add_item``.

        Raises :class:`ChatItemError` on validation failure or unsupported type.
        """
        add_supported_item(self._state(conn_id).runtime_config.chat, item)

    # ── Pipeline event handlers ────────────────────

    def on_partial_transcription(self, conn_id: str, event: PartialTranscriptionEvent) -> list[ServerEvent]:
        """Handle partial_transcription: emit transcription delta event."""
        return [
            ConversationItemInputAudioTranscriptionDeltaEvent(
                type="conversation.item.input_audio_transcription.delta",
                event_id=self._next_event_id(),
                content_index=self._next_input_content_index(conn_id),
                item_id=self._input_item_id(conn_id),
                delta=event.delta,
            )
        ]

    def on_transcription_completed(self, conn_id: str, event: TranscriptionCompletedEvent) -> list[ServerEvent]:
        """Handle transcription_completed: accumulate duration and emit completed event."""
        st = self._state(conn_id)
        st.response_usage.audio_duration_s += st.input_audio_duration_s
        return [
            ConversationItemInputAudioTranscriptionCompletedEvent(
                type="conversation.item.input_audio_transcription.completed",
                event_id=self._next_event_id(),
                content_index=0,
                item_id=self._input_item_id(conn_id),
                transcript=event.transcript,
                usage=UsageTranscriptTextUsageDuration(
                    seconds=st.input_audio_duration_s,
                    type="duration",
                ),
            )
        ]
