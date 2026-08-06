from queue import Queue
from threading import Event, Thread
from time import perf_counter, sleep

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.pipeline.control import SESSION_END
from speech_to_speech.pipeline.messages import EndOfResponse, TTSInput
from speech_to_speech.TTS.qwen3_tts_handler import MAX_COALESCED_TTS_CHARS, Qwen3TTSHandler


def _make_handler():
    handler = object.__new__(Qwen3TTSHandler)
    handler.queue_in = Queue()
    handler.queue_out = Queue()
    handler.stop_event = Event()
    return handler


def _candidate_runtime(*, text_lookahead: int, phrase_flush_ms: int) -> RuntimeConfig:
    runtime = RuntimeConfig()
    runtime.local_pipeline.update(
        {
            "tts_backend": "qwen3tts-audiocpp",
            "tts_tuning": {
                "provider": "qwen3tts-audiocpp",
                "profile_id": "test",
                "overrides": {},
                "resolved": {
                    "text_lookahead": text_lookahead,
                    "phrase_flush_ms": phrase_flush_ms,
                },
            },
        }
    )
    return runtime


def test_coalesce_pending_tts_input_merges_ready_sentences_and_absorbs_response_end():
    handler = _make_handler()

    handler.queue_in.put(TTSInput(text="Second sentence.", language_code="en"))
    handler.queue_in.put(TTSInput(text="Third sentence.", language_code="en"))
    handler.queue_in.put(EndOfResponse())

    text, lang, saw_end = handler._coalesce_pending_tts_input(TTSInput(text="First sentence.", language_code="en"))

    assert text == "First sentence. Second sentence. Third sentence."
    assert lang == "en"
    assert saw_end is True
    remaining = handler.queue_in.get_nowait()
    assert isinstance(remaining, EndOfResponse)


def test_coalesce_pending_tts_input_stops_before_control_messages():
    handler = _make_handler()

    handler.queue_in.put(SESSION_END)
    text, lang, saw_end = handler._coalesce_pending_tts_input(TTSInput(text="Hello.", language_code="en"))

    assert text == "Hello."
    assert lang == "en"
    assert saw_end is False
    assert handler.queue_in.get_nowait() == SESSION_END


def test_coalesce_pending_tts_input_keeps_long_remainder_queued():
    handler = _make_handler()
    first = "A" * (MAX_COALESCED_TTS_CHARS - 10)
    second = "B" * 40
    handler.queue_in.put(TTSInput(text=second, language_code="en", turn_id="turn", turn_revision=0))

    text, lang, saw_end = handler._coalesce_pending_tts_input(
        TTSInput(text=first, language_code="en", turn_id="turn", turn_revision=0)
    )

    assert text == first
    assert lang == "en"
    assert saw_end is False
    assert handler.queue_in.get_nowait().text == second


def test_audio_cpp_resolved_lookahead_starts_after_target_without_consuming_extra_phrase():
    handler = _make_handler()
    runtime = _candidate_runtime(text_lookahead=16, phrase_flush_ms=1000)
    handler.queue_in.put(
        TTSInput(text="second phrase.", runtime_config=runtime, turn_id="turn", turn_revision=0)
    )
    handler.queue_in.put(
        TTSInput(text="third phrase.", runtime_config=runtime, turn_id="turn", turn_revision=0)
    )

    text, _, saw_end = handler._coalesce_pending_tts_input(
        TTSInput(text="First", runtime_config=runtime, turn_id="turn", turn_revision=0)
    )

    assert text == "First second phrase."
    assert saw_end is False
    assert handler.queue_in.get_nowait().text == "third phrase."


def test_audio_cpp_punctuation_complete_first_phrase_dispatches_without_lookahead_delay():
    handler = _make_handler()
    runtime = _candidate_runtime(text_lookahead=256, phrase_flush_ms=3000)
    handler.queue_in.put(
        TTSInput(text="Second phrase.", runtime_config=runtime, turn_id="turn", turn_revision=0)
    )
    started = perf_counter()

    first, _, first_saw_end = handler._coalesce_pending_tts_input(
        TTSInput(text="First phrase.", runtime_config=runtime, turn_id="turn", turn_revision=0)
    )
    second_item = handler.queue_in.get_nowait()
    second, _, second_saw_end = handler._coalesce_pending_tts_input(second_item)

    assert [first, second] == ["First phrase.", "Second phrase."]
    assert first_saw_end is False
    assert second_saw_end is False
    assert perf_counter() - started < 0.1


def test_audio_cpp_coalescing_never_crosses_cancel_generation_or_provider():
    handler = _make_handler()
    runtime = _candidate_runtime(text_lookahead=256, phrase_flush_ms=1000)
    faster_runtime = _candidate_runtime(text_lookahead=256, phrase_flush_ms=1000)
    faster_runtime.local_pipeline["tts_backend"] = "faster"
    handler.queue_in.put(
        TTSInput(
            text="stale generation",
            runtime_config=runtime,
            turn_id="turn",
            turn_revision=0,
            cancel_generation=6,
        )
    )

    text, _, _ = handler._coalesce_pending_tts_input(
        TTSInput(
            text="active fragment",
            runtime_config=runtime,
            turn_id="turn",
            turn_revision=0,
            cancel_generation=7,
        )
    )

    assert text == "active fragment"
    assert handler.queue_in.get_nowait().text == "stale generation"

    handler.queue_in.put(
        TTSInput(
            text="foreign provider",
            runtime_config=faster_runtime,
            turn_id="turn",
            turn_revision=0,
            cancel_generation=7,
        )
    )
    text, _, _ = handler._coalesce_pending_tts_input(
        TTSInput(
            text="candidate fragment",
            runtime_config=runtime,
            turn_id="turn",
            turn_revision=0,
            cancel_generation=7,
        )
    )

    assert text == "candidate fragment"
    assert handler.queue_in.get_nowait().text == "foreign provider"


def test_audio_cpp_phrase_flush_exits_promptly_when_generation_is_cancelled():
    handler = _make_handler()
    runtime = _candidate_runtime(text_lookahead=256, phrase_flush_ms=3000)
    generation = {"stale": False}
    handler.cancel_scope = type(
        "Scope",
        (),
        {"is_stale": lambda _self, _value: generation["stale"]},
    )()

    def cancel_later():
        sleep(0.03)
        generation["stale"] = True

    canceller = Thread(target=cancel_later)
    canceller.start()
    started = perf_counter()
    text, _, _ = handler._coalesce_pending_tts_input(
        TTSInput(
            text="Incomplete fragment",
            runtime_config=runtime,
            turn_id="turn",
            turn_revision=0,
            cancel_generation=7,
        )
    )
    elapsed = perf_counter() - started
    canceller.join(timeout=1)

    assert text == "Incomplete fragment"
    assert 0.02 <= elapsed < 0.25


def test_audio_cpp_resolved_flush_waits_for_a_later_phrase_and_releases_on_response_end():
    handler = _make_handler()
    runtime = _candidate_runtime(text_lookahead=128, phrase_flush_ms=250)

    def enqueue_later():
        sleep(0.02)
        handler.queue_in.put(
            TTSInput(text="arrived later.", runtime_config=runtime, turn_id="turn", turn_revision=0)
        )
        handler.queue_in.put(EndOfResponse(turn_id="turn", turn_revision=0))

    producer = Thread(target=enqueue_later)
    producer.start()
    started = perf_counter()
    text, _, saw_end = handler._coalesce_pending_tts_input(
        TTSInput(text="Short", runtime_config=runtime, turn_id="turn", turn_revision=0)
    )
    elapsed = perf_counter() - started
    producer.join(timeout=1)

    assert text == "Short arrived later."
    assert saw_end is True
    assert 0.01 <= elapsed < 0.2
    assert isinstance(handler.queue_in.get_nowait(), EndOfResponse)


def test_audio_cpp_resolved_flush_expires_without_more_text():
    handler = _make_handler()
    runtime = _candidate_runtime(text_lookahead=128, phrase_flush_ms=50)
    started = perf_counter()

    text, _, saw_end = handler._coalesce_pending_tts_input(
        TTSInput(text="Short", runtime_config=runtime, turn_id="turn", turn_revision=0)
    )

    elapsed = perf_counter() - started
    assert text == "Short"
    assert saw_end is False
    assert 0.04 <= elapsed < 0.3


def test_faster_ignores_audio_cpp_phrase_queue_snapshot_and_keeps_ready_queue_behavior():
    handler = _make_handler()
    runtime = _candidate_runtime(text_lookahead=16, phrase_flush_ms=3000)
    runtime.local_pipeline["tts_backend"] = "faster"
    handler.queue_in.put(
        TTSInput(text="Second.", runtime_config=runtime, turn_id="turn", turn_revision=0)
    )

    started = perf_counter()
    text, _, _ = handler._coalesce_pending_tts_input(
        TTSInput(text="First.", runtime_config=runtime, turn_id="turn", turn_revision=0)
    )

    assert text == "First. Second."
    assert perf_counter() - started < 0.05
