from queue import Queue
from threading import Event, Thread

from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.LLM.lm_output_processor import LMOutputProcessor
from speech_to_speech.pipeline.events import ResponseOutputCompleteEvent
from speech_to_speech.pipeline.messages import EndOfResponse, LLMResponseChunk, TTSInput
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker


def _processor(tracker: SpeculativeTurnTracker) -> LMOutputProcessor:
    processor = LMOutputProcessor.__new__(LMOutputProcessor)
    processor.setup(text_output_queue=Queue(), speculative_turns=tracker)
    return processor


def test_stale_end_of_response_is_not_forwarded_to_tts():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 1)
    processor = _processor(tracker)

    outputs = list(processor.process(EndOfResponse(turn_id="turn_1", turn_revision=0)))

    assert outputs == []


def test_latest_end_of_response_is_forwarded_to_tts():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 1)
    processor = _processor(tracker)

    outputs = list(processor.process(EndOfResponse(turn_id="turn_1", turn_revision=1)))

    assert len(outputs) == 1
    assert outputs[0].turn_id == "turn_1"
    assert outputs[0].turn_revision == 1
    terminal = processor.text_output_queue.get_nowait()
    assert isinstance(terminal, ResponseOutputCompleteEvent)
    assert terminal.turn_id == "turn_1"


def test_current_response_epoch_overrides_newer_speculative_turn_for_chunk_and_terminal():
    """An audible retained answer remains authoritative when interruption is disabled."""

    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_new", 0)
    processor = _processor(tracker)
    runtime = RuntimeConfig()
    runtime.local_pipeline["_response_epoch_is_current"] = lambda epoch: epoch == 41

    chunk = list(
        processor.process(
            LLMResponseChunk(
                text="retained phrase",
                runtime_config=runtime,
                turn_id="turn_old",
                turn_revision=0,
                response_epoch=41,
                response=RealtimeResponseCreateParams(output_modalities=["audio"]),
            )
        )
    )
    terminal = list(
        processor.process(
            EndOfResponse(
                runtime_config=runtime,
                turn_id="turn_old",
                turn_revision=0,
                response_epoch=41,
            )
        )
    )

    assert [item.text for item in chunk] == ["retained phrase"]
    assert len(terminal) == 1
    assert terminal[0].runtime_config is runtime


def test_obsolete_response_epoch_is_rejected_even_when_turn_tracker_is_latest():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_old", 0)
    processor = _processor(tracker)
    runtime = RuntimeConfig()
    runtime.local_pipeline["_response_epoch_is_current"] = lambda _epoch: False

    outputs = list(
        processor.process(
            LLMResponseChunk(
                text="late phrase",
                runtime_config=runtime,
                turn_id="turn_old",
                turn_revision=0,
                response_epoch=40,
                response=RealtimeResponseCreateParams(output_modalities=["audio"]),
            )
        )
    )

    assert outputs == []
    assert processor.text_output_queue.empty()


def test_completed_service_epoch_rejects_late_chunk_and_terminal_output():
    """A late playback-ACK window cannot reopen model/TTS admission."""

    service = RealtimeService()
    conn_id = service.register()
    try:
        runtime = service._state(conn_id).runtime_config
        owner = service.claim_pending_response(conn_id, turn_id="turn_done", turn_revision=0)
        service.mark_response_completed(conn_id)
        assert service.is_response_epoch_current(conn_id, owner.response_epoch) is True
        assert service.is_response_epoch_output_admissible(conn_id, owner.response_epoch) is False

        tracker = SpeculativeTurnTracker()
        tracker.observe("turn_done", 0)
        processor = _processor(tracker)
        chunk = list(
            processor.process(
                LLMResponseChunk(
                    text="late assistant text",
                    runtime_config=runtime,
                    turn_id="turn_done",
                    turn_revision=0,
                    response_epoch=owner.response_epoch,
                    response=RealtimeResponseCreateParams(output_modalities=["audio"]),
                )
            )
        )
        terminal = list(
            processor.process(
                EndOfResponse(
                    runtime_config=runtime,
                    turn_id="turn_done",
                    turn_revision=0,
                    response_epoch=owner.response_epoch,
                )
            )
        )

        assert chunk == []
        assert terminal == []
        assert processor.text_output_queue.empty()
    finally:
        service.unregister(conn_id)


def test_cancel_generation_is_forwarded_to_tts():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
    processor = _processor(tracker)

    outputs = list(
        processor.process(
            LLMResponseChunk(
                text="hello",
                turn_id="turn_1",
                turn_revision=0,
                cancel_generation=7,
            )
        )
    )

    assert len(outputs) == 1
    assert outputs[0].cancel_generation == 7


def test_response_identity_epochs_are_forwarded_to_tts_and_terminal():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
    processor = _processor(tracker)

    outputs = list(
        processor.process(
            LLMResponseChunk(
                text="hello",
                turn_id="turn_1",
                turn_revision=0,
                cancel_generation=7,
                input_epoch=101,
                response_epoch=202,
                response_id="response-202",
                response=RealtimeResponseCreateParams(output_modalities=["audio"]),
            )
        )
    )
    terminal = list(
        processor.process(
            EndOfResponse(
                turn_id="turn_1",
                turn_revision=0,
                input_epoch=101,
                response_epoch=202,
                response_id="response-202",
            )
        )
    )

    assert outputs[0].input_epoch == 101
    assert outputs[0].response_epoch == 202
    assert outputs[0].response_id == "response-202"
    assert terminal[0].input_epoch == 101
    assert terminal[0].response_epoch == 202
    assert terminal[0].response_id == "response-202"


def test_text_only_chunk_is_not_forwarded_to_tts():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
    processor = _processor(tracker)

    outputs = list(
        processor.process(
            LLMResponseChunk(
                text="hello",
                turn_id="turn_1",
                turn_revision=0,
                response=RealtimeResponseCreateParams(output_modalities=["text"]),
            )
        )
    )

    assert outputs == []
    # The assistant text still reaches clients even when TTS is skipped.
    event = processor.text_output_queue.get_nowait()
    assert event.text == "hello"


def test_audio_chunk_is_forwarded_to_tts():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
    processor = _processor(tracker)

    outputs = list(
        processor.process(
            LLMResponseChunk(
                text="hello",
                turn_id="turn_1",
                turn_revision=0,
                response=RealtimeResponseCreateParams(output_modalities=["audio"]),
            )
        )
    )

    assert len(outputs) == 1
    assert isinstance(outputs[0], TTSInput)
    assert outputs[0].text == "hello"


def test_empty_modalities_is_forwarded_to_tts():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
    processor = _processor(tracker)

    outputs = list(
        processor.process(
            LLMResponseChunk(
                text="hello",
                turn_id="turn_1",
                turn_revision=0,
                response=RealtimeResponseCreateParams(output_modalities=[]),
            )
        )
    )

    assert len(outputs) == 1
    assert isinstance(outputs[0], TTSInput)


def test_pending_reopen_holds_assistant_chunk_until_cancelled():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
    candidate_revision = tracker.begin_reopen_candidate("turn_1", 0)
    processor = _processor(tracker)
    done = Event()
    outputs = []

    def run_processor():
        outputs.extend(
            processor.process(
                LLMResponseChunk(
                    text="hello",
                    turn_id="turn_1",
                    turn_revision=0,
                )
            )
        )
        done.set()

    thread = Thread(target=run_processor)
    thread.start()

    assert not done.wait(0.05)
    tracker.cancel_reopen_candidate("turn_1", candidate_revision)
    assert done.wait(1.0)
    thread.join(timeout=1.0)

    assert len(outputs) == 1
    assert outputs[0].text == "hello"
    event = processor.text_output_queue.get_nowait()
    assert event.text == "hello"


def test_reopen_grace_holds_assistant_chunk_until_elapsed():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
    tracker.start_reopen_grace("turn_1", 0, grace_s=0.08)
    processor = _processor(tracker)
    done = Event()
    outputs = []

    def run_processor():
        outputs.extend(
            processor.process(
                LLMResponseChunk(
                    text="hello",
                    turn_id="turn_1",
                    turn_revision=0,
                )
            )
        )
        done.set()

    thread = Thread(target=run_processor)
    thread.start()

    assert not done.wait(0.02)
    assert done.wait(1.0)
    thread.join(timeout=1.0)

    assert len(outputs) == 1
    assert outputs[0].text == "hello"
    event = processor.text_output_queue.get_nowait()
    assert event.text == "hello"


def test_confirmed_reopen_drops_held_assistant_chunk():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
    candidate_revision = tracker.begin_reopen_candidate("turn_1", 0)
    processor = _processor(tracker)
    done = Event()
    outputs = []

    def run_processor():
        outputs.extend(
            processor.process(
                LLMResponseChunk(
                    text="hello",
                    turn_id="turn_1",
                    turn_revision=0,
                )
            )
        )
        done.set()

    thread = Thread(target=run_processor)
    thread.start()

    assert not done.wait(0.05)
    assert tracker.confirm_reopen_candidate("turn_1", 0, candidate_revision)
    assert done.wait(1.0)
    thread.join(timeout=1.0)

    assert outputs == []
    assert processor.text_output_queue.empty()
