import asyncio
import base64
from copy import deepcopy
from pathlib import Path
from queue import Queue

import numpy as np
import pytest

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.api.openai_realtime.websocket_router import (
    _PipelineConfigError,
    _prepare_pipeline_config_update,
    _validated_audio_cpp_tuning,
)
from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, AudioOutput, EndOfResponse, TTSInput
from speech_to_speech.TTS.kokoro_handler import KokoroTTSHandler
from speech_to_speech.TTS.qwen3_tts_handler import Qwen3TTSHandler


def _candidate_runtime(
    *, profile_id="balanced", revision=3, seed=None, lookahead=24, flush_ms=250, model="qwen3-tts-1.7b-base-bf16",
):
    effective = _frozen_effective(seed=seed)
    effective["model"] = model
    runtime = RuntimeConfig()
    runtime.local_pipeline.update(
        {
            "tts_backend": "qwen3tts-audiocpp",
            "assistant_language": "en",
            "tts_model_epoch": "model-epoch-a",
            "tts_tuning": {
                "provider": "qwen3tts-audiocpp",
                "profile_id": profile_id,
                "profile_revision": revision,
                "overrides": {"seed": seed},
                "effective": effective,
                "resolved": {
                    "text_lookahead": lookahead,
                    "phrase_flush_ms": flush_ms,
                },
            },
        }
    )
    return runtime


def _frozen_effective(*, seed=17):
    """The full candidate policy that must outlive a mutable named profile."""
    return {
        "model": "qwen3-tts-1.7b-base-bf16",
        "clone_mode": "full_icl",
        "max_reference_seconds": 20,
        "first_block_frames": 4,
        "steady_block_frames": 12,
        "left_context_frames": 72,
        "text_lookahead": 64,
        "phrase_flush_ms": 500,
        "temperature": 0.7,
        "top_k": 42,
        "top_p": 0.9,
        "repetition_penalty": 1.05,
        "seed": seed,
    }


def _handler():
    handler = object.__new__(Qwen3TTSHandler)
    handler.backend = "openai_api"
    handler._initial_speaker = None
    handler.speaker = None
    handler.ref_audio = "clone-reference.wav"
    handler._initial_ref_audio = handler.ref_audio
    handler.ref_text = "Matched reference transcript."
    handler.queue_in = Queue()
    handler._resolve_api_provider = lambda _runtime: (
        "qwen3tts-audiocpp",
        "http://candidate/v1",
        "qwen3-tts-1.7b-base-bf16",
    )
    handler._resolve_api_voice = lambda _runtime, _response: "clone:stable-clone"
    handler._freeze_candidate_clone = lambda _endpoint, _voice: {
        "profile_id": "stable-clone",
        "content_revision": 4,
        "content_hash": "a" * 64,
        "ref_audio": base64.b64encode(b"first-reference-bytes").decode("ascii"),
        "ref_text": "Matched reference transcript.",
        "reference_excerpts": [],
    }
    handler._audio_cpp_native_lifecycle_epochs = {
        ("http://candidate/v1", "qwen3-tts-1.7b-base-bf16"): "7",
    }
    handler._audio_cpp_native_lifecycle_instances = {
        ("http://candidate/v1", "qwen3-tts-1.7b-base-bf16"): "a" * 32,
    }
    return handler


def _kokoro_handler():
    handler = object.__new__(KokoroTTSHandler)
    handler.voice = "bm_fable"
    handler.lang_code = "b"
    handler.speed = 1.0
    return handler


def test_kokoro_response_snapshot_keeps_voice_language_and_speed_after_live_update():
    """Non-Qwen phrase synthesis must not reread a mutable session voice."""
    handler = _kokoro_handler()
    runtime = RuntimeConfig()
    runtime.response_synthesis_configs[18] = {"voice": "af_bella"}
    first = TTSInput(
        text="First phrase.",
        language_code="en",
        runtime_config=runtime,
        response_epoch=18,
        response_id="resp-eighteen",
    )
    snapshot = handler._response_synthesis_snapshot(first)

    runtime.response_synthesis_configs[18]["voice"] = "bf_emma"
    handler.voice = "bf_emma"
    handler.lang_code = "b"
    handler.speed = 1.75
    later = handler._response_synthesis_snapshot(
        TTSInput(
            text="Later phrase.",
            language_code="fr",
            runtime_config=runtime,
            response_epoch=18,
            response_id="resp-eighteen",
        )
    )

    assert later is snapshot
    assert snapshot.voice == "af_bella"
    assert snapshot.lang_code == "b"
    assert snapshot.speed == 1.0


def test_kokoro_releases_response_snapshots_at_terminal_and_on_cancellation():
    """Long sessions retain only active Kokoro response snapshots."""

    handler = _kokoro_handler()
    runtime = RuntimeConfig()

    # A normal answer may contain multiple phrases, but its snapshot must be
    # released at the terminal rather than accumulate for every prior answer.
    for epoch in range(1, 33):
        runtime.response_synthesis_configs[epoch] = {"voice": "af_bella"}
        item = TTSInput(text="Phrase.", runtime_config=runtime, response_epoch=epoch)
        snapshot = handler._response_synthesis_snapshot(item)
        assert handler._response_synthesis_snapshots[snapshot.key] is snapshot

        assert list(handler.process(EndOfResponse(runtime_config=runtime, response_epoch=epoch))) == [
            AUDIO_RESPONSE_DONE
        ]
        assert snapshot.key not in handler._response_synthesis_snapshots

    assert handler._response_synthesis_snapshots == {}

    # Cancellation can arrive before its terminal sentinel.  The stale phrase
    # admission path must also release the existing frozen snapshot.
    runtime.local_pipeline["_response_epoch_is_current"] = lambda _epoch: False
    cancelled = TTSInput(text="Cancelled.", runtime_config=runtime, response_epoch=99)
    snapshot = handler._response_synthesis_snapshot(cancelled)

    assert list(handler.process(cancelled)) == []
    assert snapshot.key not in handler._response_synthesis_snapshots


def test_session_end_clears_response_scoped_snapshot_state_before_epoch_reuse():
    handler = _handler()
    first_runtime = _candidate_runtime(profile_id="balanced", revision=3, seed=101)
    first = handler._response_synthesis_snapshot(
        TTSInput(text="First session.", runtime_config=first_runtime, input_epoch=1, response_epoch=1)
    )
    handler._start_response_phrase(first)
    handler._metric_first_phrase_turns = {first.key}

    handler.on_session_end()

    assert handler._response_synthesis_snapshots == {}
    assert handler._response_phrase_counts == {}
    assert handler._metric_first_phrase_turns == set()

    second_runtime = _candidate_runtime(profile_id="quality", revision=8, seed=202)
    second = handler._response_synthesis_snapshot(
        TTSInput(text="Second session.", runtime_config=second_runtime, input_epoch=1, response_epoch=1)
    )
    assert second is not first
    assert second.profile_id == "quality"
    assert second.profile_revision == 8
    assert second.seed == 202


def test_response_snapshot_freezes_profile_clone_language_model_and_random_seed():
    handler = _handler()
    runtime = _candidate_runtime(seed=None)
    first = TTSInput(
        text="First sentence.",
        language_code="en",
        runtime_config=runtime,
        input_epoch=41,
        response_epoch=7,
        response_id="response-seven",
    )
    snapshot = handler._response_synthesis_snapshot(first)

    # Simulate a live Diagnostics update after the answer already started.
    runtime.local_pipeline["assistant_language"] = "fr"
    runtime.local_pipeline["tts_model_epoch"] = "model-epoch-b"
    runtime.local_pipeline["tts_tuning"]["profile_id"] = "low-latency"
    runtime.local_pipeline["tts_tuning"]["profile_revision"] = 99
    runtime.local_pipeline["tts_tuning"]["effective"]["seed"] = 123
    runtime.local_pipeline["tts_tuning"]["overrides"]["seed"] = 123
    handler._resolve_api_voice = lambda _runtime, _response: "clone:changed-clone"
    second = TTSInput(
        text="Second sentence.",
        language_code="fr",
        runtime_config=runtime,
        input_epoch=41,
        response_epoch=7,
        response_id="response-seven",
    )

    assert handler._response_synthesis_snapshot(second) is snapshot
    assert snapshot.provider == "qwen3tts-audiocpp"
    assert snapshot.endpoint == "http://candidate/v1"
    assert snapshot.model == "qwen3-tts-1.7b-base-bf16"
    assert snapshot.model_epoch == "7"
    assert snapshot.model_instance_id == "a" * 32
    assert snapshot.voice == "clone:stable-clone"
    assert snapshot.profile_id == "balanced"
    assert snapshot.profile_revision == 3
    assert snapshot.language == "English"
    assert snapshot.seed_policy == "response_random"
    assert 0 <= snapshot.seed <= 0xFFFFFFFF
    assert snapshot.tuning["overrides"]["seed"] == snapshot.seed


def test_new_response_freezes_first_phrase_language_not_prior_turn_runtime_value():
    """A German answer followed by English cannot carry German into the new snapshot."""

    service = RealtimeService()
    conn_id = service.register()
    try:
        runtime = service._state(conn_id).runtime_config
        runtime.local_pipeline.update(deepcopy(_candidate_runtime(seed=13).local_pipeline))
        runtime.local_pipeline["assistant_language"] = "German"
        owner = service.claim_pending_response(conn_id, turn_id="turn_english", turn_revision=0)

        admission = runtime.response_synthesis_configs[owner.response_epoch]["local_pipeline"]
        assert "assistant_language" not in admission

        snapshot = _handler()._response_synthesis_snapshot(
            TTSInput(
                text="This answer is English.",
                language_code="English",
                runtime_config=runtime,
                response_epoch=owner.response_epoch,
            )
        )

        assert snapshot.language == "English"
    finally:
        service.unregister(conn_id)


def test_response_snapshot_preserves_explicit_profile_seed():
    handler = _handler()
    snapshot = handler._response_synthesis_snapshot(
        TTSInput(text="A sentence.", runtime_config=_candidate_runtime(seed=1337), response_epoch=8)
    )

    assert snapshot.seed == 1337
    assert snapshot.seed_policy == "profile"
    assert snapshot.tuning["overrides"]["seed"] == 1337


def test_response_snapshot_preserves_revisioned_null_model_profile_against_resident_model():
    """Built-in profiles may follow the resident Base model without losing their revision."""

    handler = _handler()
    runtime = _candidate_runtime(profile_id="low-latency", revision=2, seed=None, model=None)
    snapshot = handler._response_synthesis_snapshot(
        TTSInput(text="A stable phrase.", runtime_config=runtime, response_epoch=48)
    )
    request_tuning = handler._audio_cpp_request_tuning(snapshot)

    # The response owns the health-derived concrete model/lifecycle, while the
    # profile's null model remains the candidate contract for "follow resident".
    assert snapshot.model == "qwen3-tts-1.7b-base-bf16"
    assert snapshot.profile_id == "low-latency"
    assert snapshot.profile_revision == 2
    assert snapshot.tuning["effective"]["model"] is None
    assert request_tuning is not None
    assert request_tuning["profile_id"] == "low-latency"
    assert request_tuning["profile_revision"] == 2
    assert request_tuning["effective"]["model"] is None


def test_response_snapshot_rejects_unknown_profile_model_even_when_null_is_supported():
    """Null means resident model; arbitrary model names must never become a fallback."""

    effective = _frozen_effective(seed=None)
    effective["model"] = "not-a-qwen3-base-model"
    assert Qwen3TTSHandler._audio_cpp_complete_effective_tuning(effective, 2) is None


def test_response_snapshot_honors_profile_seed_from_frozen_effective_snapshot():
    """A complete browser snapshot may have no temporary override at all."""
    handler = _handler()
    runtime = _candidate_runtime(seed=None)
    runtime.local_pipeline["tts_tuning"].update(
        {
            "profile_revision": 3,
            "effective": _frozen_effective(seed=1337),
            "overrides": {},
        }
    )

    snapshot = handler._response_synthesis_snapshot(
        TTSInput(text="A sentence.", runtime_config=runtime, response_epoch=81)
    )

    assert snapshot.seed == 1337
    assert snapshot.seed_policy == "profile"
    assert snapshot.tuning["overrides"]["seed"] == 1337


def test_later_candidate_phrases_coalesce_using_the_frozen_profile_after_first_phrase():
    handler = _handler()
    runtime = _candidate_runtime(lookahead=512, flush_ms=250)
    first = TTSInput(text="First phrase.", runtime_config=runtime, response_epoch=9)
    snapshot = handler._response_synthesis_snapshot(first)

    immediate, _, _ = handler._coalesce_pending_tts_input(first, snapshot)
    handler._start_response_phrase(snapshot)
    handler.queue_in.put(TTSInput(text="Third phrase.", runtime_config=runtime, response_epoch=9))
    combined, _, _ = handler._coalesce_pending_tts_input(
        TTSInput(text="Second phrase.", runtime_config=runtime, response_epoch=9), snapshot
    )

    assert immediate == "First phrase."
    assert combined == "Second phrase. Third phrase."


def test_frozen_effective_snapshot_drives_candidate_phrase_queue_not_legacy_resolved():
    """The signed-off browser schema has no mutable ``resolved`` scheduling source."""
    handler = _handler()
    runtime = _candidate_runtime(seed=17, lookahead=16, flush_ms=50)
    runtime.local_pipeline["tts_tuning"].update(
        {
            "profile_revision": 3,
            "effective": _frozen_effective(),
        }
    )
    snapshot = handler._response_synthesis_snapshot(
        TTSInput(text="Incomplete", runtime_config=runtime, response_epoch=82)
    )

    assert handler._candidate_phrase_queue_settings_from_snapshot(snapshot) == (64, 0.5)


def test_phrase_coalescing_never_crosses_response_epochs():
    handler = _handler()
    runtime = _candidate_runtime(lookahead=128, flush_ms=250)
    current = TTSInput(text="Current", runtime_config=runtime, response_epoch=10)
    snapshot = handler._response_synthesis_snapshot(current)
    handler._start_response_phrase(snapshot)
    handler.queue_in.put(TTSInput(text="Foreign", runtime_config=runtime, response_epoch=11))

    combined, _, _ = handler._coalesce_pending_tts_input(current, snapshot)

    assert combined == "Current"
    assert handler.queue_in.get_nowait().response_epoch == 11


def test_phrase_coalescing_never_crosses_input_epochs_inside_one_response():
    handler = _handler()
    runtime = _candidate_runtime(lookahead=128, flush_ms=250)
    current = TTSInput(text="Current", runtime_config=runtime, input_epoch=20, response_epoch=10)
    snapshot = handler._response_synthesis_snapshot(current)
    handler._start_response_phrase(snapshot)
    handler.queue_in.put(TTSInput(text="Foreign", runtime_config=runtime, input_epoch=21, response_epoch=10))

    combined, _, _ = handler._coalesce_pending_tts_input(current, snapshot)

    assert combined == "Current"
    assert handler.queue_in.get_nowait().input_epoch == 21


def test_later_phrases_consume_ready_sentences_in_order():
    handler = _handler()
    runtime = _candidate_runtime(lookahead=512, flush_ms=250)
    first = TTSInput(text="First phrase.", runtime_config=runtime, input_epoch=22, response_epoch=11)
    snapshot = handler._response_synthesis_snapshot(first)
    handler._start_response_phrase(snapshot)
    handler.queue_in.put(TTSInput(text="Third phrase.", runtime_config=runtime, input_epoch=22, response_epoch=11))
    handler.queue_in.put(TTSInput(text="Fourth phrase.", runtime_config=runtime, input_epoch=22, response_epoch=11))
    handler.queue_in.put(EndOfResponse(input_epoch=22, response_epoch=11))

    combined, _, _ = handler._coalesce_pending_tts_input(
        TTSInput(text="Second phrase.", runtime_config=runtime, input_epoch=22, response_epoch=11), snapshot
    )

    assert combined == "Second phrase. Third phrase. Fourth phrase."


def test_final_tail_flushes_immediately_at_end_of_response():
    handler = _handler()
    runtime = _candidate_runtime(lookahead=512, flush_ms=3000)
    tail = TTSInput(text="unfinished final tail", runtime_config=runtime, input_epoch=23, response_epoch=12)
    snapshot = handler._response_synthesis_snapshot(tail)
    handler._start_response_phrase(snapshot)
    handler.queue_in.put(EndOfResponse(input_epoch=23, response_epoch=12))

    combined, _, saw_end = handler._coalesce_pending_tts_input(tail, snapshot)

    assert combined == "unfinished final tail"
    assert saw_end is True


def test_snapshot_metric_detail_exposes_response_identity_and_stability_fields():
    handler = _handler()
    snapshot = handler._response_synthesis_snapshot(
        TTSInput(text="Metric.", runtime_config=_candidate_runtime(seed=5), input_epoch=12, response_epoch=13)
    )

    detail = handler._snapshot_metric_detail(snapshot, phrase_index=2)

    assert detail["input_epoch"] == 12
    assert detail["response_epoch"] == 13
    assert detail["profile_id"] == "balanced"
    assert detail["profile_revision"] == 3
    assert detail["seed"] == 5
    assert detail["phrase_index"] == 2
    assert detail["clone_fingerprint"]
    assert detail["clone_content_revision"] == 4
    assert detail["clone_content_hash"] == "a" * 64


def test_process_reuses_frozen_candidate_seed_and_tuning_for_later_phrases():
    handler = _handler()
    handler.cancel_scope = None
    handler.speculative_turns = None
    handler.api_streaming_supported = False
    handler.api_voice = "clone:default"
    handler.api_response_format = "pcm"
    handler.api_sample_rate = 24000
    handler.blocksize = 4
    handler.text_output_queue = None
    handler._last_tts_response_headers = {}
    captured = []

    def synthesize(_text, _voice, **kwargs):
        captured.append(kwargs["tts_tuning"])
        return iter([np.array([1, 2, 3, 4], dtype=np.int16)])

    handler._process_openai_api = synthesize
    runtime = _candidate_runtime(seed=None, lookahead=16)
    first = TTSInput(text="First sentence.", runtime_config=runtime, response_epoch=14)
    assert len(list(handler.process(first))) == 1

    runtime.local_pipeline["tts_tuning"]["profile_id"] = "quality"
    runtime.local_pipeline["tts_tuning"]["profile_revision"] = 44
    runtime.local_pipeline["tts_tuning"]["effective"]["seed"] = 22
    runtime.local_pipeline["tts_tuning"]["overrides"]["seed"] = 22
    second = TTSInput(text="A later sentence that is long enough.", runtime_config=runtime, response_epoch=14)
    assert len(list(handler.process(second))) == 1

    assert len(captured) == 2
    assert captured[0]["profile_id"] == captured[1]["profile_id"] == "balanced"
    assert captured[0]["overrides"]["seed"] == captured[1]["overrides"]["seed"]
    assert "revision" not in captured[0]
    assert "revision" not in captured[1]
    assert "resolved" not in captured[0]
    assert "resolved" not in captured[1]
    assert captured[0]["profile_revision"] == captured[1]["profile_revision"] == 3
    assert captured[0]["effective"] == captured[1]["effective"]
    assert captured[0]["effective"] == _frozen_effective(seed=None)


def test_candidate_without_complete_tuning_resolves_once_then_freezes_selection(monkeypatch):
    """A profile-refresh race must not roll back an otherwise complete answer.

    The browser may not yet have a complete revisioned policy when it starts a
    fresh Realtime session.  In that narrow case the candidate owns the selected
    Realtime profile, while HF Realtime still freezes one response seed and
    never forwards partial browser policy.
    """

    handler = _handler()
    handler._openai_api_headers = lambda: {}
    selected = {"id": "balanced", "revision": 7, **_frozen_effective(seed=None)}
    resolutions = []
    def resolve(url, **kwargs):
        import httpx
        resolutions.append((url, deepcopy(kwargs)))
        return httpx.Response(200, json={"profile": deepcopy(selected)}, request=httpx.Request("POST", url))
    monkeypatch.setattr("speech_to_speech.TTS.qwen3_tts_handler.httpx.post", resolve)
    handler.cancel_scope = None
    handler.speculative_turns = None
    handler.api_streaming_supported = False
    handler.api_voice = "clone:default"
    handler.api_response_format = "pcm"
    handler.api_sample_rate = 24000
    handler.blocksize = 4
    handler.text_output_queue = None
    handler._last_tts_response_headers = {}
    captured = []
    handler._process_openai_api = lambda _text, _voice, **kwargs: (
        captured.append(deepcopy(kwargs["tts_tuning"]))
        or iter([np.array([1, 2, 3, 4], dtype=np.int16)])
    )
    runtime = RuntimeConfig()
    runtime.local_pipeline["tts_backend"] = "qwen3tts-audiocpp"

    first = TTSInput(text="First phrase.", runtime_config=runtime, response_epoch=145)
    second = TTSInput(text="Second phrase.", runtime_config=runtime, response_epoch=145)
    assert len(list(handler.process(first))) == 1
    selected.update(id="quality", revision=8, seed=99)
    assert len(list(handler.process(second))) == 1

    assert len(captured) == 2
    assert captured[0]["provider"] == "qwen3tts-audiocpp"
    assert captured[0]["scope"] == "realtime"
    assert captured[0]["profile_id"] == captured[1]["profile_id"] == "balanced"
    assert captured[0]["profile_revision"] == captured[1]["profile_revision"] == 7
    assert captured[0]["effective"] == captured[1]["effective"]
    assert len(resolutions) == 1
    assert captured[0]["overrides"]["seed"] == captured[1]["overrides"]["seed"]


def test_same_model_engine_reload_keeps_the_frozen_epoch_on_later_phrases():
    """A same-ID child restart must be rejected by the candidate, not hidden."""

    handler = _handler()
    handler._audio_cpp_native_lifecycle_epochs = {
        ("http://candidate/v1", "qwen3-tts-1.7b-base-bf16"): "7",
    }
    handler._audio_cpp_native_lifecycle_instances = {
        ("http://candidate/v1", "qwen3-tts-1.7b-base-bf16"): "a" * 32,
    }
    handler.cancel_scope = None
    handler.speculative_turns = None
    handler.api_streaming_supported = False
    handler.api_voice = "clone:default"
    handler.api_response_format = "pcm"
    handler.api_sample_rate = 24000
    handler.blocksize = 4
    handler.text_output_queue = None
    handler._last_tts_response_headers = {}
    lifecycles = []

    def synthesize(_text, _voice, **kwargs):
        lifecycles.append((kwargs.get("expected_engine_epoch"), kwargs.get("expected_supervisor_instance_id")))
        return iter([np.array([1, 2, 3, 4], dtype=np.int16)])

    handler._process_openai_api = synthesize
    runtime = _candidate_runtime(seed=17)
    first = TTSInput(text="First phrase.", runtime_config=runtime, response_epoch=90)
    assert len(list(handler.process(first))) == 1

    # This models an unload/reload of the same Base model after phrase one.
    # The second request keeps seven, so the supervisor's now-eight admission
    # rejects it instead of synthesizing with a different engine lifecycle.
    handler._audio_cpp_native_lifecycle_epochs[("http://candidate/v1", "qwen3-tts-1.7b-base-bf16")] = "8"
    second = TTSInput(text="Second phrase.", runtime_config=runtime, response_epoch=90)
    assert len(list(handler.process(second))) == 1

    frozen = handler._response_synthesis_snapshots[("response_epoch", 90)]
    assert frozen.model_epoch == "7"
    assert frozen.model_instance_id == "a" * 32
    assert lifecycles == [(7, "a" * 32), (7, "a" * 32)]


def test_non_candidate_tts_request_does_not_receive_candidate_engine_epoch():
    """Faster/Groxaxo stay outside the candidate's private lifecycle guard."""

    handler = _handler()
    handler._resolve_api_provider = lambda _runtime: (
        "faster", "http://faster/v1", "1.7B-Base"
    )
    handler.cancel_scope = None
    handler.speculative_turns = None
    handler.api_streaming_supported = True
    handler.api_voice = "clone:default"
    handler.api_response_format = "pcm"
    handler.api_sample_rate = 16000
    handler.blocksize = 4
    handler.text_output_queue = None
    captured = []
    handler._process_openai_api = lambda _text, _voice, **kwargs: (
        captured.append(kwargs) or iter([np.array([1, 2, 3, 4], dtype=np.int16)])
    )

    assert len(list(handler.process(TTSInput(
        text="Faster phrase.", runtime_config=RuntimeConfig(), response_epoch=91
    )))) == 1
    assert "expected_engine_epoch" not in captured[0]


def test_phrase_requests_forward_one_immutable_effective_profile_snapshot():
    """A saved-profile edit after phrase one cannot alter phrase two's engine policy."""
    handler = _handler()
    handler.cancel_scope = None
    handler.speculative_turns = None
    handler.api_streaming_supported = False
    handler.api_voice = "clone:default"
    handler.api_response_format = "pcm"
    handler.api_sample_rate = 24000
    handler.blocksize = 4
    handler.text_output_queue = None
    handler._last_tts_response_headers = {}
    captured = []

    def synthesize(_text, _voice, **kwargs):
        captured.append(kwargs["tts_tuning"])
        return iter([np.array([1, 2, 3, 4], dtype=np.int16)])

    handler._process_openai_api = synthesize
    runtime = _candidate_runtime(seed=17)
    frozen_effective = _frozen_effective()
    runtime.local_pipeline["tts_tuning"].update(
        {
            "profile_revision": 3,
            "effective": deepcopy(frozen_effective),
        }
    )
    first = TTSInput(text="First sentence.", runtime_config=runtime, response_epoch=15)
    assert len(list(handler.process(first))) == 1

    # This models an external profile edit/resolution after phrase one.  The
    # original effective values must travel with the response snapshot, not be
    # looked up again by the supervisor from a mutable profile id.
    runtime.local_pipeline["tts_tuning"]["effective"].update(
        {
            "max_reference_seconds": 1,
            "temperature": 1.8,
            "top_k": 1,
            "top_p": 0.1,
            "repetition_penalty": 1.8,
            "seed": 999,
        }
    )
    second = TTSInput(text="Second sentence.", runtime_config=runtime, response_epoch=15)
    assert len(list(handler.process(second))) == 1

    assert len(captured) == 2
    assert captured[0]["profile_id"] == captured[1]["profile_id"] == "balanced"
    assert captured[0]["effective"] == frozen_effective
    assert captured[1]["effective"] == frozen_effective
    assert captured[0]["overrides"]["seed"] == captured[1]["overrides"]["seed"] == 17


def test_websocket_preserves_validated_frozen_effective_profile_snapshot():
    """The local session boundary cannot discard the immutable policy before TTS sees it."""
    effective = _frozen_effective()
    tuning = _validated_audio_cpp_tuning(
        {
            "provider": "qwen3tts-audiocpp",
            "profile_id": "response-frozen",
            "profile_revision": 9,
            "effective": effective,
            "overrides": {"seed": 17},
        }
    )

    assert tuning["profile_revision"] == 9
    assert tuning["effective"] == effective


@pytest.mark.parametrize(
    "mutate",
    [
        lambda tuning: tuning["effective"].update({"unexpected": 1}),
        lambda tuning: tuning["effective"].pop("top_p"),
        lambda tuning: tuning["effective"].update({"max_reference_seconds": 31}),
        lambda tuning: tuning["effective"].update({"temperature": 0}),
        lambda tuning: tuning["effective"].update({"left_context_frames": 289, "steady_block_frames": 12}),
        lambda tuning: tuning["effective"].update({"clone_mode": "x_vector_only"}),
    ],
    ids=(
        "unknown",
        "missing",
        "scalar-out-of-bounds",
        "zero-temperature",
        "decoder-total-out-of-bounds",
        "unsupported-clone-mode",
    ),
)
def test_websocket_rejects_noncanonical_frozen_effective_snapshot(mutate):
    tuning = {
        "provider": "qwen3tts-audiocpp",
        "profile_id": "response-frozen",
        "profile_revision": 9,
        "effective": _frozen_effective(),
        "overrides": {"seed": 17},
    }
    mutate(tuning)

    with pytest.raises(ValueError):
        _validated_audio_cpp_tuning(tuning)


def test_candidate_session_wire_contains_profile_revision_and_complete_effective_snapshot():
    """The browser must send a complete immutable value policy, not just queue knobs."""
    source = (Path(__file__).parents[1] / "web" / "hf-realtime-voice" / "main.js").read_text(encoding="utf-8")
    session_wire = source.split("function activeTtsTuning", 1)[1].split(
        "function activePlaybackConfig", 1
    )[0]

    assert "profile_revision" in session_wire
    assert "effective" in session_wire
    assert 'clone_mode: "full_icl"' in session_wire
    for field in _frozen_effective():
        if field == "clone_mode":
            continue
        assert f'"{field}"' in source


@pytest.mark.parametrize("backend", ("faster", "groxaxo"))
def test_non_candidate_pipeline_config_rejects_frozen_candidate_tuning_without_mutation(backend):
    """Candidate-only snapshots cannot leak into legacy provider configuration."""
    runtime = RuntimeConfig()
    runtime.local_pipeline.update({"tts_backend": backend, "legacy_marker": backend})
    before = deepcopy(runtime.local_pipeline)
    config = {
        "tts_backend": backend,
        "tts_tuning": {
            "provider": "qwen3tts-audiocpp",
            "profile_id": "response-frozen",
            "profile_revision": 9,
            "effective": _frozen_effective(),
            "overrides": {"seed": 17},
        },
    }

    with pytest.raises(_PipelineConfigError):
        asyncio.run(_prepare_pipeline_config_update(object(), runtime, config))

    assert runtime.local_pipeline == before


def test_profile_switch_applies_to_next_response_only_and_snapshot_releases_at_terminal():
    handler = _handler()
    runtime = _candidate_runtime(seed=7, profile_id="balanced", revision=3)
    first = TTSInput(text="First.", runtime_config=runtime, input_epoch=30, response_epoch=30)
    frozen = handler._response_synthesis_snapshot(first)

    runtime.local_pipeline["tts_tuning"]["profile_id"] = "quality"
    runtime.local_pipeline["tts_tuning"]["profile_revision"] = 4
    assert handler._response_synthesis_snapshot(
        TTSInput(text="Later.", runtime_config=runtime, input_epoch=30, response_epoch=30)
    ) is frozen
    assert frozen.profile_id == "balanced"

    assert list(handler.process(EndOfResponse(input_epoch=30, response_epoch=30)))
    assert not handler._response_synthesis_snapshots
    next_response = handler._response_synthesis_snapshot(
        TTSInput(text="Next answer.", runtime_config=runtime, input_epoch=31, response_epoch=31)
    )
    assert next_response.profile_id == "quality"
    assert next_response.profile_revision == 4


def test_response_snapshot_stays_until_terminal_not_between_phrases():
    handler = _handler()
    runtime = _candidate_runtime(seed=99)
    first = TTSInput(text="First.", runtime_config=runtime, input_epoch=34, response_epoch=34)
    snapshot = handler._response_synthesis_snapshot(first)
    handler._start_response_phrase(snapshot)
    handler._start_response_phrase(snapshot)

    assert handler._response_synthesis_snapshots[snapshot.key] is snapshot
    assert handler._response_phrase_index(snapshot) == 2
    handler._release_response_synthesis_snapshot(EndOfResponse(input_epoch=34, response_epoch=34))
    assert snapshot.key not in handler._response_synthesis_snapshots
    assert snapshot.key not in handler._response_phrase_counts


def test_stale_response_epoch_never_starts_external_tts_request():
    handler = _handler()
    handler.speculative_turns = None
    handler.cancel_scope = type("Scope", (), {"is_response_epoch_stale": lambda _self, _epoch: True})()
    calls = []
    handler._process_openai_api = lambda *_args, **_kwargs: calls.append(True)

    assert list(handler.process(TTSInput(text="Do not synthesize.", runtime_config=_candidate_runtime(), response_epoch=40))) == []
    assert calls == []


def test_current_response_epoch_overrides_newer_speculative_turn_for_tts_and_terminal():
    """An audible owner remains allowed to finish before its queued successor.

    The speculative revision tracker may already point at the newer accepted
    speech, but it is not the authority once the lifecycle has claimed a
    response epoch.  This keeps the TTS phrase and its terminal together.
    """

    handler = _handler()
    commits = []
    handler.speculative_turns = type(
        "Tracker",
        (),
        {
            "is_latest_after_reopen_grace": lambda _self, _turn, _revision: False,
            "commit": lambda _self, turn, revision: commits.append((turn, revision)),
        },
    )()
    handler.cancel_scope = None
    handler.api_streaming_supported = False
    handler.api_voice = "clone:default"
    handler.api_response_format = "pcm"
    handler.api_sample_rate = 24000
    handler.blocksize = 4
    handler.text_output_queue = None
    handler._last_tts_response_headers = {}
    requests = []
    handler._process_openai_api = lambda text, _voice, **_kwargs: (
        requests.append(text) or iter([np.array([1, 2, 3, 4], dtype=np.int16)])
    )
    runtime = _candidate_runtime(seed=9)
    runtime.local_pipeline["_response_epoch_is_current"] = lambda epoch: epoch == 81

    item = TTSInput(
        text="Finish the audible answer.",
        runtime_config=runtime,
        turn_id="turn-a",
        turn_revision=1,
        response_epoch=81,
    )
    assert len(list(handler.process(item))) == 1
    assert requests == ["Finish the audible answer."]
    assert commits == [("turn-a", 1)]
    assert list(handler.process(EndOfResponse(
        runtime_config=runtime,
        turn_id="turn-a",
        turn_revision=1,
        response_epoch=81,
    ))) == [AUDIO_RESPONSE_DONE]


def test_stale_response_epoch_rejects_tts_and_terminal_even_if_tracker_is_latest():
    handler = _handler()
    handler.speculative_turns = type(
        "Tracker",
        (),
        {
            "is_latest_after_reopen_grace": lambda _self, _turn, _revision: True,
            "commit": lambda _self, *_args: pytest.fail("stale response must not commit"),
        },
    )()
    handler.cancel_scope = None
    calls = []
    handler._process_openai_api = lambda *_args, **_kwargs: calls.append(True)
    runtime = _candidate_runtime()
    runtime.local_pipeline["_response_epoch_is_current"] = lambda _epoch: False

    assert list(handler.process(TTSInput(
        text="Do not synthesize.",
        runtime_config=runtime,
        turn_id="turn-b",
        turn_revision=2,
        response_epoch=82,
    ))) == []
    assert calls == []
    assert list(handler.process(EndOfResponse(
        runtime_config=runtime,
        turn_id="turn-b",
        turn_revision=2,
        response_epoch=82,
    ))) == []


def test_completed_service_epoch_never_starts_tts_or_emits_terminal_audio_done():
    handler = _handler()
    handler.speculative_turns = None
    handler.cancel_scope = None
    calls = []
    handler._process_openai_api = lambda *_args, **_kwargs: calls.append(True)

    service = RealtimeService()
    conn_id = service.register()
    try:
        runtime = service._state(conn_id).runtime_config
        runtime.local_pipeline.update(_candidate_runtime().local_pipeline)
        owner = service.claim_pending_response(conn_id, turn_id="turn-complete", turn_revision=1)
        service.mark_response_completed(conn_id)

        assert list(handler.process(TTSInput(
            text="Do not synthesize after completion.",
            runtime_config=runtime,
            turn_id="turn-complete",
            turn_revision=1,
            response_epoch=owner.response_epoch,
        ))) == []
        assert calls == []
        assert list(handler.process(EndOfResponse(
            runtime_config=runtime,
            turn_id="turn-complete",
            turn_revision=1,
            response_epoch=owner.response_epoch,
        ))) == []
    finally:
        service.unregister(conn_id)


def test_audio_output_preserves_all_response_identity_fields():
    source = TTSInput(
        text="Identity.",
        cancel_generation=9,
        input_epoch=41,
        response_epoch=42,
        response_id="response-42",
    )
    wrapped = BaseHandler.output_for_queue(object.__new__(BaseHandler), np.array([1], dtype=np.int16), source)

    assert isinstance(wrapped, AudioOutput)
    assert wrapped.cancel_generation == 9
    assert wrapped.input_epoch == 41
    assert wrapped.response_epoch == 42
    assert wrapped.response_id == "response-42"


def test_phrase_requests_remain_single_active_and_ordered():
    handler = _handler()
    handler.cancel_scope = None
    handler.speculative_turns = None
    handler.api_streaming_supported = False
    handler.api_voice = "clone:default"
    handler.api_response_format = "pcm"
    handler.api_sample_rate = 24000
    handler.blocksize = 4
    handler.text_output_queue = None
    handler._last_tts_response_headers = {}
    active = 0
    peak_active = 0
    calls = []

    def synthesize(text, _voice, **_kwargs):
        def stream():
            nonlocal active, peak_active
            active += 1
            peak_active = max(peak_active, active)
            calls.append(text)
            try:
                yield np.array([1, 2, 3, 4], dtype=np.int16)
            finally:
                active -= 1

        return stream()

    handler._process_openai_api = synthesize
    runtime = _candidate_runtime(seed=5)
    assert len(list(handler.process(TTSInput(text="First.", runtime_config=runtime, input_epoch=50, response_epoch=50)))) == 1
    assert len(list(handler.process(TTSInput(text="Second.", runtime_config=runtime, input_epoch=50, response_epoch=50)))) == 1

    assert calls == ["First.", "Second."]
    assert peak_active == 1
    assert active == 0


def test_metric_event_carries_unambiguous_top_level_response_identity():
    handler = _handler()
    handler.text_output_queue = Queue()
    item = TTSInput(
        text="Metric identity.",
        input_epoch=61,
        response_epoch=62,
        response_id="response-62",
    )

    handler._emit_metric("tts", "request_start", item, detail={"legacy": "detail-only"})
    metric = handler.text_output_queue.get_nowait()

    assert metric.input_epoch == 61
    assert metric.response_epoch == 62
    assert metric.response_id == "response-62"
    # The router must not need to infer ownership from this optional payload.
    assert metric.detail["legacy"] == "detail-only"


def test_phrase_failure_retains_snapshot_and_suppresses_later_phrase_request():
    """A failed answer cannot restart mid-response from an edited clone slot."""

    handler = _handler()
    handler.cancel_scope = None
    handler.speculative_turns = None
    handler.api_streaming_supported = False
    handler.api_voice = "clone:default"
    handler.api_response_format = "pcm"
    handler.api_sample_rate = 24000
    handler.blocksize = 4
    handler.text_output_queue = None
    handler._last_tts_response_headers = {}
    original_clone = deepcopy(handler._freeze_candidate_clone(None, None))
    changed_clone = {
        **original_clone,
        "content_revision": 5,
        "content_hash": "b" * 64,
        "ref_audio": base64.b64encode(b"edited-reference-bytes").decode("ascii"),
    }
    calls = []

    def synthesize(_text, _voice, **kwargs):
        calls.append(deepcopy(kwargs["clone_snapshot"]))
        if len(calls) == 1:
            raise RuntimeError("phrase-local provider failure")
        return iter([np.array([1, 2, 3, 4], dtype=np.int16)])

    handler._process_openai_api = synthesize
    runtime = _candidate_runtime(seed=17)
    first = TTSInput(text="First phrase.", runtime_config=runtime, input_epoch=71, response_epoch=72)
    assert list(handler.process(first)) == []
    frozen = handler._response_synthesis_snapshots[("response_epoch", 72)]
    assert frozen.clone_content_hash == "a" * 64

    # Simulate a Voice Studio edit after the failed first phrase. The frozen
    # snapshot remains available for diagnostics until the terminal sentinel,
    # but later phrases are suppressed instead of restarting a partial answer.
    handler._freeze_candidate_clone = lambda _endpoint, _voice: changed_clone
    runtime.local_pipeline["tts_tuning"]["profile_id"] = "quality"
    assert list(handler.process(TTSInput(
        text="Second phrase.", runtime_config=runtime, input_epoch=71, response_epoch=72
    ))) == []

    assert calls == [original_clone]
    assert handler._response_synthesis_snapshots[("response_epoch", 72)] is frozen


def test_snapshot_setup_failure_suppresses_every_later_phrase_in_answer():
    handler = _handler()
    handler.cancel_scope = None
    handler.speculative_turns = None
    handler.text_output_queue = None
    attempts = []

    def fail_resolve(_runtime):
        attempts.append("resolve")
        raise RuntimeError("candidate health unavailable")

    handler._resolve_api_provider = fail_resolve
    runtime = _candidate_runtime(seed=29)
    first = TTSInput(text="First phrase.", runtime_config=runtime, response_epoch=721)
    assert list(handler.process(first)) == []
    assert ("response_epoch", 721) in handler._failed_response_synthesis_keys

    handler._resolve_api_provider = lambda _runtime: (
        "qwen3tts-audiocpp", "http://candidate/v1", "qwen3-tts-1.7b-base-bf16"
    )
    assert list(handler.process(TTSInput(
        text="Must stay suppressed.", runtime_config=runtime, response_epoch=721
    ))) == []
    assert attempts == ["resolve"]


def test_empty_provider_audio_is_terminal_for_remaining_answer_phrases():
    handler = _handler()
    handler.cancel_scope = None
    handler.speculative_turns = None
    handler.api_streaming_supported = False
    handler.api_voice = "clone:default"
    handler.api_response_format = "pcm"
    handler.api_sample_rate = 24000
    handler.blocksize = 4
    handler.text_output_queue = None
    handler._last_tts_response_headers = {}
    calls = []

    def empty_synthesis(text, _voice, **_kwargs):
        calls.append(text)
        return iter(())

    handler._process_openai_api = empty_synthesis
    runtime = _candidate_runtime(seed=31)
    assert list(handler.process(TTSInput(
        text="First phrase.", runtime_config=runtime, response_epoch=722
    ))) == []
    assert ("response_epoch", 722) in handler._failed_response_synthesis_keys
    assert list(handler.process(TTSInput(
        text="Second phrase.", runtime_config=runtime, response_epoch=722
    ))) == []
    assert calls == ["First phrase."]


def test_frozen_candidate_clone_payload_is_forwarded_on_every_phrase():
    handler = _handler()
    handler.cancel_scope = None
    handler.speculative_turns = None
    handler.api_streaming_supported = False
    handler.api_voice = "clone:default"
    handler.api_response_format = "pcm"
    handler.api_sample_rate = 24000
    handler.blocksize = 4
    handler.text_output_queue = None
    handler._last_tts_response_headers = {}
    payloads = []
    handler._process_openai_api = lambda _text, _voice, **kwargs: (
        payloads.append(deepcopy(kwargs["clone_snapshot"])) or iter([np.array([1, 2, 3, 4], dtype=np.int16)])
    )
    runtime = _candidate_runtime(seed=9)

    assert len(list(handler.process(TTSInput(text="One.", runtime_config=runtime, response_epoch=73)))) == 1
    assert len(list(handler.process(TTSInput(text="Two.", runtime_config=runtime, response_epoch=73)))) == 1

    assert len(payloads) == 2
    assert payloads[0] == payloads[1]
    assert payloads[0]["profile_id"] == "stable-clone"
    assert payloads[0]["content_revision"] == 4
    assert payloads[0]["content_hash"] == "a" * 64


def test_openai_candidate_payload_keeps_frozen_clone_material_as_a_private_request_field():
    handler = _handler()
    frozen = handler._freeze_candidate_clone(None, None)

    payload = handler._openai_api_payload(
        "Snapshot payload.",
        "clone:stable-clone",
        "English",
        model="qwen3-tts-1.7b-base-bf16",
        clone_snapshot=deepcopy(frozen),
    )

    assert payload["voice"] == "clone:stable-clone"
    assert payload["clone_snapshot"] == frozen
    assert payload["clone_snapshot"] is not frozen


def test_stale_tts_input_is_rejected_before_speculative_commit():
    handler = _handler()
    commits = []
    handler.speculative_turns = type(
        "Tracker",
        (),
        {
            "is_latest_after_reopen_grace": lambda _self, _turn, _revision: True,
            "commit": lambda _self, turn, revision: commits.append((turn, revision)),
        },
    )()
    handler.cancel_scope = type("Scope", (), {"is_stale": lambda _self, _generation: True})()

    assert list(handler.process(TTSInput(text="Never speak.", turn_id="turn", turn_revision=1, cancel_generation=2))) == []
    assert commits == []
