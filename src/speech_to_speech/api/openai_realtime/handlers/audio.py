from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING

from openai.types.realtime import (
    InputAudioBufferAppendEvent,
    InputAudioBufferSpeechStartedEvent,
    InputAudioBufferSpeechStoppedEvent,
    RealtimeErrorEvent,
    ResponseAudioDeltaEvent,
    ResponseCreatedEvent,
)

from speech_to_speech.api.openai_realtime.handlers.base import RealtimeBaseHandler
from speech_to_speech.api.openai_realtime.utils import resample
from speech_to_speech.pipeline.events import SpeechStartedEvent, SpeechStoppedEvent

if TYPE_CHECKING:
    from speech_to_speech.api.openai_realtime.service import ServerEvent

logger = logging.getLogger(__name__)

PIPELINE_SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512
BYTES_PER_SAMPLE = 2
CHUNK_SIZE_BYTES = CHUNK_SAMPLES * BYTES_PER_SAMPLE


class AudioHandler(RealtimeBaseHandler):
    """Owns inbound audio decoding/chunking and outbound audio encoding."""

    def _start_input_item(self, conn_id: str, *, preserve_active_response: bool = False) -> str:
        response = self._service.response
        st = self._state(conn_id)
        if not preserve_active_response:
            item_id = response._start_item(conn_id)
        else:
            response_item_id = st.current_item_id
            response_content_index = st.content_index
            item_id = response._start_item(conn_id)
            st.current_item_id = response_item_id
            st.content_index = response_content_index
        st.input_content_index = 0
        return item_id

    def handle_audio_append(self, conn_id: str, event: InputAudioBufferAppendEvent) -> list[bytes]:
        """Decode base64 audio, resample to pipeline rate, and split into 512-sample PCM16 chunks for the VAD."""
        try:
            pcm_bytes = base64.b64decode(event.audio)
        except Exception as e:
            logger.error(f"Base64 decode error: {e}")
            return []

        st = self._state(conn_id)

        audio_cfg = st.runtime_config.session.audio
        if audio_cfg is not None and audio_cfg.input is not None:
            client_in_rate = getattr(audio_cfg.input.format, "rate", None) or PIPELINE_SAMPLE_RATE
        else:
            client_in_rate = PIPELINE_SAMPLE_RATE
        pcm_bytes = resample(pcm_bytes, client_in_rate, PIPELINE_SAMPLE_RATE)

        pcm_bytes = st.audio_remainder + pcm_bytes

        chunks = []
        for i in range(0, len(pcm_bytes), CHUNK_SIZE_BYTES):
            chunk = pcm_bytes[i : i + CHUNK_SIZE_BYTES]
            if len(chunk) == CHUNK_SIZE_BYTES:
                chunks.append(chunk)
            else:
                st.audio_remainder = chunk
                break
        else:
            st.audio_remainder = b""

        if chunks:
            st.audio_buffer_has_data = True
        return chunks

    def handle_audio_commit(self, conn_id: str) -> RealtimeErrorEvent | None:
        """Commit the audio buffer. Returns an error if no audio was appended."""
        st = self._state(conn_id)
        if not st.audio_buffer_has_data:
            return self.make_error(
                message="Input audio buffer is empty, nothing to commit.",
                _type="input_audio_buffer_commit_empty",
            )
        st.audio_buffer_has_data = False
        logger.debug("Audio buffer committed")
        return None

    # ── Pipeline event handlers ────────────────────

    def on_speech_started(self, conn_id: str, event: SpeechStartedEvent) -> list[ServerEvent]:
        """Handle VAD speech_started: cancel active response if interrupts enabled, start new input item."""
        events: list[ServerEvent] = []
        st = self._state(conn_id)
        was_in_response = st.in_response
        supersession = None
        # VAD has now confirmed newer speech.  Invalidate the old owner before
        # the router attempts transport cancellation; direct Gemma can be
        # active here even though no OpenAI response exists yet.
        current_input_epoch = st.response_ownership.input_epoch
        if event.input_epoch is None:
            supersession = self._service.observe_speech_started(
                conn_id,
                reason="speculative_reopen" if event.reopened else "new_speech",
                interrupt_response=event.interrupt_response,
            )
        elif event.input_epoch < current_input_epoch:
            logger.debug(
                "Ignoring stale speech-start input_epoch=%d current=%d",
                event.input_epoch,
                current_input_epoch,
            )
            return []
        elif event.input_epoch > current_input_epoch:
            # Compatibility for a producer that stamps the next epoch before
            # the per-session callback is installed.  Larger jumps cannot be
            # valid within one conversation.
            supersession = self._service.observe_speech_started(
                conn_id,
                reason="speculative_reopen" if event.reopened else "new_speech",
                interrupt_response=event.interrupt_response,
            )
            if supersession.input_epoch != event.input_epoch:
                logger.warning(
                    "Dropping speech-start with noncontiguous input_epoch=%d current=%d",
                    event.input_epoch,
                    supersession.input_epoch,
                )
                return []
        else:
            supersession = self._service.pending_supersession(conn_id, event.input_epoch)
        if supersession is None:
            logger.debug(
                "Ignoring speech-start without its matching supersession input_epoch=%s",
                event.input_epoch,
            )
            return []
        if event.input_epoch is None:
            # Legacy producers do not stamp speech-start events. Preserve the
            # epoch allocated above on the same event object so the router can
            # retrieve this exact supersession and perform transport/queue
            # cleanup instead of treating the event as unmatched.
            event.input_epoch = supersession.input_epoch
        # Pre-audible supersession already rolls back and flushes safe client
        # inserts synchronously inside ``observe_speech_started``.  Do not
        # drain here: when interruption is disabled an audible predecessor may
        # still be writing assistant/tool history, and its terminal barrier is
        # the first safe point to apply the queued successor's client items.
        st.text_output_complete = False
        st.await_text_output_complete = True
        # Keep an audible predecessor's already-dispatched tool transaction
        # intact when interruption is disabled. The newly accepted speech may
        # queue behind it, but cannot overtake call -> output in Chat. An
        # actually invalidated predecessor is safe to discard (the service has
        # already handled the pre-audible rollback case).
        if supersession.invalidated and (st.pending_tool_call_ids or st.tool_followup_ready):
            logger.info("Superseding unresolved tool transaction for a newer user turn")
            self._service._state(conn_id).runtime_config.chat.discard_pending_tool_calls(
                set(st.pending_tool_call_ids)
            )
            st.pending_tool_call_ids.clear()
            st.tool_followup_ready = False
            st.tool_followup_started = False
            st.tool_followup_requested = False
            st.tool_followup_response = None
        # The producer already invalidated/closed the previous response under
        # the input-epoch ownership transaction. A delayed speech-start must
        # never close whichever successor has since become current.
        is_reopen = bool(event.reopened and event.turn_id is not None and event.turn_id == st.speculative_turn_id)
        preserve_active_response = st.in_response
        if is_reopen:
            input_item_id = st.speculative_input_item_id
            if input_item_id is None:
                input_item_id = self._start_input_item(
                    conn_id,
                    preserve_active_response=preserve_active_response,
                )
                st.speculative_input_item_id = input_item_id
            elif not preserve_active_response:
                st.current_item_id = input_item_id
                st.content_index = 0
            st.input_audio_duration_s = 0.0
            st.input_content_index = 0
        else:
            input_item_id = self._start_input_item(
                conn_id,
                preserve_active_response=preserve_active_response,
            )
            st.speculative_input_item_id = input_item_id
            st.response_usage.turns += 1
        st.speculative_turn_id = event.turn_id
        st.speculative_turn_revision = event.turn_revision
        st.last_item_id = input_item_id
        previous = supersession.previous if supersession is not None else None
        effective_interrupt = bool(
            (supersession is not None and supersession.invalidated and previous is not None)
            or (
                was_in_response
                and event.interrupt_response
                and st.runtime_config.interrupt_response_enabled
            )
        )
        if st.rendered_playback_ack_supported and (previous is not None or was_in_response):
            # This additive local event precedes the stock speech-start event so
            # the browser can distinguish a real barge-in from capture that is
            # intentionally queued behind an audible response. It is gated on
            # the browser's existing playback-ack capability so nonbrowser and
            # OpenAI SDK clients receive only the stock compatible event.
            events.append(
                self._service.build_speech_started_decision(
                    input_epoch=st.response_ownership.input_epoch,
                    effective_interrupt=effective_interrupt,
                    response_epoch=previous.response_epoch if previous is not None else None,
                    reason=(
                        "pre_audible_supersession"
                        if supersession is not None and supersession.pre_audible
                        else "barge_in"
                        if effective_interrupt
                        else "interrupt_disabled"
                    ),
                )
            )
        events.append(
            InputAudioBufferSpeechStartedEvent(
                type="input_audio_buffer.speech_started",
                event_id=self._next_event_id(),
                audio_start_ms=event.audio_start_ms,
                item_id=input_item_id,
            )
        )
        return events

    def on_speech_stopped(self, conn_id: str, event: SpeechStoppedEvent) -> list[ServerEvent]:
        """Handle VAD speech_stopped: record duration and emit stopped event."""
        st = self._state(conn_id)
        if event.duration_s:
            st.input_audio_duration_s = event.duration_s
        # A zero-duration stop closes a phantom/discarded VAD pair; it is not
        # an accepted turn and must not create provisional response ownership.
        # Valid VAD output either carries its already-claimed epoch or has a
        # positive duration on the legacy path.
        if event.response_epoch is not None or event.duration_s > 0:
            owner = self._service.claim_pending_response(
                conn_id,
                turn_id=event.turn_id,
                turn_revision=event.turn_revision,
                input_epoch=event.input_epoch,
            )
            active = st.response_ownership.active()
            if active is not None and active.response_epoch == owner.response_epoch:
                self._service.queue_deferred_settlement_events(
                    conn_id,
                    self._service._owner_event(conn_id, owner),
                )
        else:
            self._service.close_rejected_input_capture(
                conn_id,
                input_epoch=event.input_epoch,
            )
        return [
            InputAudioBufferSpeechStoppedEvent(
                type="input_audio_buffer.speech_stopped",
                event_id=self._next_event_id(),
                audio_end_ms=event.audio_end_ms,
                item_id=self._input_item_id(conn_id),
            )
        ]

    # ── Outbound audio encoding ──────────────────

    def encode_audio_chunk(
        self,
        conn_id: str,
        audio: bytes,
        *,
        source_sample_rate: int = PIPELINE_SAMPLE_RATE,
        output_sample_rate: int | None = None,
        input_epoch: int | None = None,
        response_epoch: int | None = None,
        response_id: str | None = None,
    ) -> list[ServerEvent]:
        """Encode a raw PCM audio chunk, emitting ResponseCreated on the first chunk.

        When ``handle_response_create`` already allocated the response,
        ``current_response_id`` is set and no duplicate event is emitted.
        For the implicit-response path (VAD -> STT -> LLM -> TTS, no
        ``response.create``), ``current_response_id`` is still ``None``
        and the event is emitted here on the first audio chunk.
        """
        st = self._state(conn_id)
        # Queue ownership is captured by the producer, not reconstructed from
        # whichever response happens to be current when this slow conversion
        # completes.  Untagged legacy PCM remains supported, but a tagged item
        # must match the active tuple exactly at every side-effect boundary.
        if not self._pcm_identity_admitted(
            conn_id,
            input_epoch=input_epoch,
            response_epoch=response_epoch,
            response_id=response_id,
        ):
            return []
        if not isinstance(source_sample_rate, int) or isinstance(source_sample_rate, bool) or source_sample_rate <= 0:
            raise ValueError("source_sample_rate must be a positive integer")

        # The candidate's native PCM is 24 kHz. The normal response path
        # freezes its destination clock at ownership claim, before Gemma/TTS
        # starts. Once present, it is authoritative: a later config update or
        # explicit caller value belongs to the next response and cannot retime
        # current PCM. The fallback resolution below serves legacy implicit
        # callers that bypass ownership claim.
        client_out_rate = st.response_output_sample_rate
        if client_out_rate is None:
            client_out_rate = output_sample_rate
            if client_out_rate is None:
                local_pipeline = getattr(st.runtime_config, "local_pipeline", None)
                negotiated = local_pipeline.get("audio_output_sample_rate") if isinstance(local_pipeline, dict) else None
                if isinstance(negotiated, int) and not isinstance(negotiated, bool) and negotiated > 0:
                    client_out_rate = negotiated
            if client_out_rate is None and source_sample_rate != PIPELINE_SAMPLE_RATE:
                client_out_rate = source_sample_rate
            if client_out_rate is None:
                rp = st.current_response_params
                if rp and rp.audio and rp.audio.output and rp.audio.output.format:
                    client_out_rate = getattr(rp.audio.output.format, "rate", None)
                if client_out_rate is None:
                    audio_cfg = st.runtime_config.session.audio
                    if audio_cfg is not None and audio_cfg.output is not None:
                        client_out_rate = getattr(audio_cfg.output.format, "rate", None) or PIPELINE_SAMPLE_RATE
                    else:
                        client_out_rate = PIPELINE_SAMPLE_RATE
        if not isinstance(client_out_rate, int) or isinstance(client_out_rate, bool) or client_out_rate <= 0:
            raise ValueError("output sample rate must be a positive integer")
        audio = resample(audio, source_sample_rate, client_out_rate)

        if not self._pcm_identity_admitted(
            conn_id,
            input_epoch=input_epoch,
            response_epoch=response_epoch,
            response_id=response_id,
        ):
            return []

        # Do not claim a protocol response or settle ownership until the PCM
        # clock and conversion have succeeded.  A malformed/failed chunk must
        # leave the provisional turn rollbackable and the next valid chunk able
        # to emit its response.created event.
        response = self._service.response
        # Recheck and allocate under the ownership lock. A VAD thread must not
        # be able to replace A with B between A's admission and
        # ``_ensure_response``; that would bind A's generated response id to B
        # and leave B emitting deltas for an unseen response.created event.
        with st.response_ownership.transaction():
            if response_epoch is not None and not st.response_ownership.matches_active_identity(
                input_epoch=input_epoch,
                response_epoch=response_epoch,
                response_id=response_id,
            ):
                return []

            events: list[ServerEvent] = []
            # A stale owner must never cause ``_ensure_response`` to allocate a
            # response object for a newer owner.  The matching check also catches a
            # residual current_response_id from an already superseded transport.
            if (
                response_epoch is not None
                and st.current_response_id is not None
                and st.current_response_id != response_id
            ):
                owner = st.response_ownership.owner_for_response_epoch(response_epoch)
                if owner is None or owner.response_id != st.current_response_id:
                    return []
            need_created = st.current_response_id is None
            resp_id, item_id = response._ensure_response(conn_id)
            if not self._pcm_identity_admitted(
                conn_id,
                input_epoch=input_epoch,
                response_epoch=response_epoch,
                # Once the response has been allocated, use its concrete identity
                # for the last check instead of reading it later from ConnState.
                response_id=resp_id,
            ):
                return []
            # Legacy implicit responses freeze only after conversion and response
            # allocation succeed. Normal response ownership froze the rate earlier.
            if st.response_output_sample_rate is None:
                st.response_output_sample_rate = client_out_rate
            if need_created:
                events.append(
                    ResponseCreatedEvent(
                        type="response.created",
                        event_id=self._next_event_id(),
                        response=response._build_response(conn_id, "in_progress"),
                    )
                )
                owner_event = self._service.response_owner_event(conn_id)
                if owner_event is not None:
                    events.append(owner_event)
            if audio:
                st.response_audio_emitted = True
            b64 = base64.b64encode(audio).decode("ascii")
            events.append(
                ResponseAudioDeltaEvent(
                    type="response.output_audio.delta",
                    event_id=self._next_event_id(),
                    content_index=response._next_content_index(conn_id),
                    delta=b64,
                    item_id=item_id,
                    output_index=0,
                    response_id=resp_id,
                )
            )
            return events

    def _pcm_identity_admitted(
        self,
        conn_id: str,
        *,
        input_epoch: int | None,
        response_epoch: int | None,
        response_id: str | None,
    ) -> bool:
        """Validate an explicitly owned PCM item without mutable-ID rebinding."""

        # Raw legacy byte entries have no response ownership metadata.  Keep
        # their original OpenAI-compatible behavior; all modern AudioOutput
        # entries carry at least a response epoch and take the strict path.
        if response_epoch is None:
            return True
        st = self._state(conn_id)
        owner = st.response_ownership.owner_for_response_epoch(response_epoch)
        if owner is None or not st.response_ownership.admits_output(response_epoch):
            return False
        if input_epoch is None or input_epoch != owner.input_epoch:
            return False
        if response_id is not None and owner.response_id != response_id:
            return False
        return True
