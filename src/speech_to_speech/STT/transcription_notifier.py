from __future__ import annotations

import logging
from queue import Queue
from threading import Event
from typing import Iterator, Union

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.LLM.chat import make_user_message
from speech_to_speech.pipeline.events import PartialTranscriptionEvent, TranscriptionCompletedEvent
from speech_to_speech.pipeline.handler_types import LLMIn, STTOut
from speech_to_speech.pipeline.messages import (
    DirectAssistantRequest,
    DirectAssistantResponse,
    GenerateResponseRequest,
    PartialTranscription,
    Transcription,
)
from speech_to_speech.pipeline.queue_types import TextEventItem

logger = logging.getLogger(__name__)


class TranscriptionNotifier(BaseHandler[STTOut, Union[STTOut, LLMIn]]):
    """Sits between STT and LLM.

    For **realtime mode** (no ``runtime_config``): emits transcription events
    on ``text_output_queue`` for protocol translation but yields nothing -- the
    ``RealtimeService`` builds ``GenerateResponseRequest`` directly.

    For **legacy mode** (``runtime_config`` provided): appends the user
    message to ``runtime_config.chat`` and yields a
    ``GenerateResponseRequest`` so the LLM handler receives a uniform input
    type regardless of pipeline mode.
    """

    def setup(
        self,
        text_output_queue: Queue[TextEventItem] | None = None,
        runtime_config: RuntimeConfig | None = None,
        should_listen: Event | None = None,
    ) -> None:
        self.text_output_queue = text_output_queue
        self.runtime_config = runtime_config
        self.should_listen = should_listen
        self._finalized_direct_transcripts: set[tuple[str | None, int | None]] = set()

    def process(self, transcription: STTOut) -> Iterator[Union[STTOut, LLMIn]]:
        if isinstance(transcription, DirectAssistantResponse):
            text = transcription.text.strip()
            transcript = (transcription.transcript or "").strip()
            runtime_config = transcription.runtime_config or self.runtime_config
            if transcript and self.text_output_queue is not None:
                transcript_key = (transcription.turn_id, transcription.turn_revision)
                should_finalize = (
                    (transcription.is_final or transcription.transcript_finalized)
                    and transcript_key not in self._finalized_direct_transcripts
                )
                if should_finalize:
                    self._finalized_direct_transcripts.add(transcript_key)
                    self.text_output_queue.put(
                        TranscriptionCompletedEvent(
                            transcript=transcript,
                            language_code=transcription.language_code,
                            turn_id=transcription.turn_id,
                            turn_revision=transcription.turn_revision,
                            speech_stopped_at_s=transcription.speech_stopped_at_s,
                            context_committed=transcription.context_committed,
                            direct_audio_completed=True,
                        )
                    )
                elif not transcription.is_final:
                    self.text_output_queue.put(
                        PartialTranscriptionEvent(
                            delta=transcript,
                            turn_id=transcription.turn_id,
                            turn_revision=transcription.turn_revision,
                        )
                    )
            if self.runtime_config is not None and transcript:
                self.runtime_config.chat.add_item(make_user_message(transcript))
            if transcription.is_final:
                self._finalized_direct_transcripts.discard((transcription.turn_id, transcription.turn_revision))
            if not text and not transcription.tools and not transcription.is_final:
                return
            if text:
                logger.info("Direct audio assistant response chunk: %s", text)
            yield DirectAssistantRequest(
                text=text,
                transcript=transcript or None,
                is_final=transcription.is_final,
                tools=transcription.tools,
                language_code=transcription.language_code,
                turn_id=transcription.turn_id,
                turn_revision=transcription.turn_revision,
                speech_stopped_at_s=transcription.speech_stopped_at_s,
                runtime_config=runtime_config,
                response=transcription.response,
                context_committed=transcription.context_committed,
                cancel_generation=transcription.cancel_generation,
            )
            return
        if isinstance(transcription, PartialTranscription):
            if self.text_output_queue and transcription.text:
                self.text_output_queue.put(
                    PartialTranscriptionEvent(
                        delta=str(transcription.text),
                        turn_id=transcription.turn_id,
                        turn_revision=transcription.turn_revision,
                    )
                )
                logger.debug("Partial transcription: %s", str(transcription.text)[:80])
            return

        if isinstance(transcription, Transcription):
            text = transcription.text
            language_code = transcription.language_code
            turn_id = transcription.turn_id
            turn_revision = transcription.turn_revision
            speech_stopped_at_s = transcription.speech_stopped_at_s
        else:
            text = transcription
            language_code = None
            turn_id = None
            turn_revision = None
            speech_stopped_at_s = None

        transcript = str(text)
        # Always close the client-visible transcription item. Empty final STT
        # results should not trigger the LLM, but clients may already have
        # received partial deltas and still need a completed event.
        if self.text_output_queue is not None:
            self.text_output_queue.put(
                TranscriptionCompletedEvent(
                    transcript=transcript,
                    language_code=language_code,
                    turn_id=turn_id,
                    turn_revision=turn_revision,
                    speech_stopped_at_s=speech_stopped_at_s,
                )
            )

        if not transcript:
            logger.debug("Transcription completed with empty transcript")
            if self.should_listen is not None:
                self.should_listen.set()
                logger.debug("Empty transcription completed; listening re-enabled")
            return

        if language_code:
            logger.info("Transcription completed (language=%s): %s", language_code, transcript)
        else:
            logger.info("Transcription completed: %s", transcript)

        if self.runtime_config is not None:
            self.runtime_config.chat.add_item(make_user_message(transcript))
            yield GenerateResponseRequest(
                runtime_config=self.runtime_config,
                language_code=language_code,
                turn_id=turn_id,
                turn_revision=turn_revision,
                speech_stopped_at_s=speech_stopped_at_s,
            )
