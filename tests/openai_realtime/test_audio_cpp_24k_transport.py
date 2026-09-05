"""Candidate-only PCM16/24 kHz transport contracts.

The microphone/VAD path remains 16 kHz.  These tests cover only outbound
audio.cpp synthesis, which must retain its model-native 24 kHz source clock.
"""

from __future__ import annotations

import base64

import numpy as np
import pytest
from openai.types.realtime import ResponseCreateEvent
from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams

import speech_to_speech.TTS.qwen3_tts_handler as qwen3_tts_module
from speech_to_speech.api.openai_realtime.websocket_router import _audio_identity, _validated_audio_cpp_tuning
from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.messages import AudioOutput, TTSInput
from speech_to_speech.TTS.qwen3_tts_handler import Qwen3TTSHandler, ResponseSynthesisSnapshot


def _complete_candidate_effective(*, seed: int | None = None) -> dict:
    return {
        "model": "qwen3-tts-1.7b-base-bf16",
        "clone_mode": "full_icl",
        "max_reference_seconds": 20,
        "first_block_frames": 4,
        "steady_block_frames": 12,
        "left_context_frames": 25,
        "text_lookahead": 64,
        "phrase_flush_ms": 500,
        "temperature": 0.7,
        "top_k": 40,
        "top_p": 0.9,
        "repetition_penalty": 1.05,
        "seed": seed,
    }


def _completed_candidate_outcome(request_id: str):
    class OutcomeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"requestId": request_id, "state": "completed", "eos": True}

    return OutcomeResponse()


def _decode_delta(events):
    delta = next(event for event in events if event.type == "response.output_audio.delta")
    return base64.b64decode(delta.delta)


def test_audio_cpp_native_24k_is_not_silently_downsampled(service, conn_id):
    pcm24 = np.arange(1_200, dtype="<i2").tobytes()
    state = service._state(conn_id)
    state.runtime_config.local_pipeline.update(
        {
            "tts_backend": "qwen3tts-audiocpp",
            "audio_output_sample_rate": 24_000,
        }
    )

    events = service.encode_audio_chunk(conn_id, pcm24, source_sample_rate=24_000)

    assert _decode_delta(events) == pcm24
    assert len(_decode_delta(events)) // 2 == 1_200
    assert (len(_decode_delta(events)) // 2) / 24_000 == 0.05


def test_audio_cpp_source_rate_can_be_explicitly_negotiated_downstream(service, conn_id):
    pcm24 = np.zeros(2_400, dtype="<i2").tobytes()

    events = service.audio.encode_audio_chunk(
        conn_id,
        pcm24,
        source_sample_rate=24_000,
        output_sample_rate=16_000,
    )

    # Exactly the same 100 ms duration at the explicitly selected 16 kHz
    # transport; source PCM itself was never mislabeled as 16 kHz.
    assert len(_decode_delta(events)) // 2 == 1_600


def test_response_transport_rate_is_frozen_across_live_config_switch(service, conn_id):
    """A new rate applies to the next response, never a later PCM batch."""
    first = np.arange(1_200, dtype="<i2").tobytes()
    second = np.arange(1_200, 2_400, dtype="<i2").tobytes()
    state = service._state(conn_id)
    state.runtime_config.local_pipeline.update(
        {"tts_backend": "qwen3tts-audiocpp", "audio_output_sample_rate": 24_000}
    )

    assert _decode_delta(service.encode_audio_chunk(conn_id, first, source_sample_rate=24_000)) == first
    assert state.response_output_sample_rate == 24_000

    # The settings acknowledgement is valid for a future response. It must not
    # downsample the 24 kHz PCM still arriving for the current response.
    state.runtime_config.local_pipeline["audio_output_sample_rate"] = 16_000
    assert _decode_delta(service.encode_audio_chunk(conn_id, second, source_sample_rate=24_000)) == second
    assert state.response_output_sample_rate == 24_000

    service.finish_response(conn_id)
    assert state.response_output_sample_rate is None

    # The new response now consumes the updated transport selection.
    next_response = service.encode_audio_chunk(conn_id, second, source_sample_rate=24_000)
    assert len(_decode_delta(next_response)) // 2 == 800
    assert state.response_output_sample_rate == 16_000


def test_pending_response_transport_rate_is_frozen_before_first_pcm(service, conn_id):
    """Changing backends while Gemma/TTS runs cannot retime its first audio."""
    pcm24 = np.arange(1_200, dtype="<i2").tobytes()
    state = service._state(conn_id)
    state.runtime_config.local_pipeline.update(
        {"tts_backend": "qwen3tts-audiocpp", "audio_output_sample_rate": 24_000}
    )

    owner = service.claim_pending_response(conn_id, turn_id="turn_24k", turn_revision=1)
    assert state.response_output_sample_rate == 24_000
    service.set_rendered_playback_ack_supported(conn_id, True)
    lifecycle = service.response_owner_event(conn_id)
    assert lifecycle is not None
    assert lifecycle.response_epoch == owner.response_epoch
    assert lifecycle.output_sample_rate == 24_000

    # A different provider/default may be selected for the future, but the
    # already-owned response must retain its 24 kHz browser contract.
    state.runtime_config.local_pipeline.update(
        {"tts_backend": "openai-api", "audio_output_sample_rate": 16_000}
    )
    events = service.encode_audio_chunk(conn_id, pcm24, source_sample_rate=24_000)
    assert _decode_delta(events) == pcm24
    assert state.response_output_sample_rate == 24_000
    assert service._state(conn_id).response_ownership.active().response_epoch == owner.response_epoch


def test_pre_protocol_cancel_releases_pending_response_transport_rate(service, conn_id):
    state = service._state(conn_id)
    state.runtime_config.local_pipeline["audio_output_sample_rate"] = 24_000
    service.claim_pending_response(conn_id, turn_id="turn_cancel", turn_revision=1)
    assert state.response_output_sample_rate == 24_000

    service.mark_response_cancelled(conn_id, reason="superseded")
    assert state.response_output_sample_rate is None


def test_explicit_response_voice_and_rate_win_at_admission_and_stay_frozen(service, conn_id):
    """A response.create override outranks session defaults before first PCM."""
    pcm24 = np.arange(1_200, dtype="<i2").tobytes()
    state = service._state(conn_id)
    state.runtime_config.local_pipeline["audio_output_sample_rate"] = 16_000
    state.runtime_config.session.audio.output.voice = "clone:session-default"
    response = RealtimeResponseCreateParams(
        audio={"output": {"voice": "clone:response-override", "format": {"type": "audio/pcm", "rate": 24_000}}}
    )

    owner = service.claim_manual_response(conn_id, response=response)
    assert state.response_output_sample_rate == 24_000
    assert state.runtime_config.response_synthesis_configs[owner.response_epoch]["voice"] == "clone:response-override"

    # This acknowledgement belongs to a future answer only.
    state.runtime_config.local_pipeline["audio_output_sample_rate"] = 16_000
    state.runtime_config.session.audio.output.voice = "clone:later-session"
    service.set_rendered_playback_ack_supported(conn_id, True)
    lifecycle = service.response_owner_event(conn_id)
    assert lifecycle is not None and lifecycle.output_sample_rate == 24_000
    assert _decode_delta(service.encode_audio_chunk(conn_id, pcm24, source_sample_rate=24_000)) == pcm24


def test_in_band_response_create_freezes_explicit_voice_and_rate_before_input_append(service, conn_id):
    """The pre-append owner claim receives response.create overrides too."""
    state = service._state(conn_id)
    state.runtime_config.local_pipeline["audio_output_sample_rate"] = 16_000
    state.runtime_config.session.audio.output.voice = "clone:session-default"
    params = RealtimeResponseCreateParams(
        audio={"output": {"voice": "clone:in-band", "format": {"type": "audio/pcm", "rate": 24_000}}},
        input=[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}],
    )

    created = service.response.handle_response_create(
        conn_id,
        ResponseCreateEvent(type="response.create", response=params),
    )
    assert created is not None
    owner = state.response_ownership.active()
    assert owner is not None
    assert state.response_output_sample_rate == 24_000
    assert state.runtime_config.response_synthesis_configs[owner.response_epoch]["voice"] == "clone:in-band"


def test_synthesis_admission_freezes_backend_profile_then_first_phrase_language(service, conn_id):
    """Owner-time backend/profile and first-phrase language form one snapshot."""
    state = service._state(conn_id)
    old_tuning = {
        "provider": "qwen3tts-audiocpp",
        "profile_id": "quality",
        "profile_revision": 7,
        "delivery_mode": "native_incremental_pcm",
        "effective": _complete_candidate_effective(seed=99),
        "overrides": {"temperature": 0.45, "seed": 99},
    }
    state.runtime_config.local_pipeline.update(
        {
            "tts_backend": "qwen3tts-audiocpp",
            "tts_tuning": old_tuning,
            "assistant_language": "fr",
        }
    )
    state.runtime_config.session.audio.output.voice = "clone:old-profile"
    first = service.claim_pending_response(conn_id, turn_id="first", turn_revision=1)

    # Simulate a UI acknowledgement that arrives while Gemma is producing but
    # before its first stable phrase is forwarded to TTS.
    state.runtime_config.local_pipeline.update(
        {"tts_backend": "faster", "tts_tuning": None, "assistant_language": "de"}
    )
    state.runtime_config.session.audio.output.voice = "clone:new-profile"

    handler = object.__new__(Qwen3TTSHandler)
    handler.backend = "openai_api"
    handler.api_backend_model = "faster-default"
    handler.api_streaming_supported = True
    handler.api_candidate_streaming_mode = "native_incremental_pcm"
    handler.api_response_format = "pcm"
    handler.api_sample_rate = 24_000
    handler._audio_cpp_native_lifecycle_epochs = {("http://candidate/v1", "qwen3-tts-1.7b-base-bf16"): "8"}
    handler._audio_cpp_native_lifecycle_instances = {("http://candidate/v1", "qwen3-tts-1.7b-base-bf16"): "candidate-8"}
    handler._resolve_api_provider = lambda runtime: (
        str((getattr(runtime, "local_pipeline", None) or {}).get("tts_backend") or "faster"),
        "http://candidate/v1",
        "qwen3-tts-1.7b-base-bf16",
    )
    handler._freeze_candidate_clone = lambda _endpoint, _voice: {
        "content_hash": "a" * 64,
        "content_revision": 3,
        "ref_audio": "AA==",
        "ref_text": "old reference",
        "reference_excerpts": [],
    }

    old_snapshot = handler._response_synthesis_snapshot(
        TTSInput(text="First phrase.", language_code="de", runtime_config=state.runtime_config, input_epoch=first.input_epoch, response_epoch=first.response_epoch)
    )
    assert old_snapshot.provider == "qwen3tts-audiocpp"
    assert old_snapshot.voice == "clone:old-profile"
    # Language is not known at owner admission; the first authoritative TTS
    # phrase supplies it after Gemma has inferred the current answer.
    assert old_snapshot.language == "German"
    assert old_snapshot.profile_id == "quality"
    assert old_snapshot.profile_revision == 7
    assert old_snapshot.seed == 99
    assert old_snapshot.delivery_native_pcm is True
    assert old_snapshot.delivery_streaming_mode == "native_incremental_pcm"
    assert old_snapshot.delivery_response_format == "pcm"
    assert old_snapshot.delivery_sample_rate == 24_000

    # The candidate can refresh capabilities while the answer is speaking, but
    # phrase two must retain phrase one's route, label, and source PCM clock.
    handler.api_streaming_supported = False
    handler.api_candidate_streaming_mode = "buffered_phrase"
    handler.api_response_format = "wav"
    handler.api_sample_rate = 16_000
    same_response_snapshot = handler._response_synthesis_snapshot(
        TTSInput(text="Second phrase.", language_code="de", runtime_config=state.runtime_config, input_epoch=first.input_epoch, response_epoch=first.response_epoch)
    )
    assert same_response_snapshot is old_snapshot
    assert same_response_snapshot.delivery_native_pcm is True
    assert same_response_snapshot.delivery_streaming_mode == "native_incremental_pcm"
    assert same_response_snapshot.delivery_response_format == "pcm"
    assert same_response_snapshot.delivery_sample_rate == 24_000
    assert Qwen3TTSHandler._snapshot_metric_detail(same_response_snapshot, 1)["streaming_mode"] == "native_incremental_pcm"

    service.mark_response_cancelled(conn_id, reason="test_complete")
    assert first.response_epoch not in state.runtime_config.response_synthesis_configs
    second = service.claim_pending_response(conn_id, turn_id="second", turn_revision=1)
    next_snapshot = handler._create_response_synthesis_snapshot(
        TTSInput(text="Next phrase.", language_code="de", runtime_config=state.runtime_config, input_epoch=second.input_epoch, response_epoch=second.response_epoch)
    )
    assert next_snapshot.provider == "faster"
    assert next_snapshot.voice == "clone:new-profile"
    assert next_snapshot.language == "German"


def test_legacy_16k_output_remains_byte_identical(service, conn_id):
    pcm16 = np.arange(800, dtype="<i2").tobytes()

    events = service.encode_audio_chunk(conn_id, pcm16)

    assert _decode_delta(events) == pcm16


def test_router_never_batches_different_pcm_clocks_and_base_preserves_candidate_envelope():
    candidate = AudioOutput(
        audio=b"\0\0",
        input_epoch=3,
        response_epoch=3,
        response_id="resp_3",
        source_sample_rate=24_000,
    )
    legacy = candidate.model_copy(update={"source_sample_rate": 16_000})

    assert _audio_identity(candidate) == (3, 3, "resp_3", 24_000)
    assert _audio_identity(candidate) != _audio_identity(legacy)
    # BaseHandler must not rebuild a candidate envelope and lose its clock or
    # identity while placing it on the shared output queue.
    assert BaseHandler.output_for_queue(object.__new__(BaseHandler), candidate, object()) is candidate


def test_audio_cpp_phrase_stream_preserves_24k_samples_and_tail(monkeypatch):
    handler = object.__new__(Qwen3TTSHandler)
    handler.api_base_url = "http://127.0.0.1:8890/v1"
    handler.api_model = "qwen3-tts-1.7b-base-bf16"
    handler.api_response_format = "pcm"
    handler.api_sample_rate = 24_000
    handler.blocksize = 512
    handler.cancel_scope = None
    handler._active_response_lock = qwen3_tts_module.Lock()
    handler._active_response = None
    handler._openai_api_headers = lambda: {}
    source = np.arange(1_280, dtype="<i2").tobytes()

    class Response:
        response_headers = {"x-tts-streaming-mode": "buffered-phrase", "x-tts-request-id": "e" * 32}

        def wait_for_headers(self):
            return None

        def iter_bytes(self):
            yield source[:1_024]
            yield source[1_024:]

        def close(self):
            return None

    monkeypatch.setattr(qwen3_tts_module, "CancellableAsyncByteStream", lambda *_args, **_kwargs: Response())
    monkeypatch.setattr(qwen3_tts_module.httpx, "get", lambda *_args, **_kwargs: _completed_candidate_outcome("e" * 32))

    chunks = list(
        handler._stream_openai_api_voice(
            "candidate phrase",
            "clone:candidate",
            progressive_buffered=True,
        )
    )

    assert [chunk.size for chunk in chunks] == [512, 512, 256]
    assert b"".join(chunk.astype("<i2", copy=False).tobytes() for chunk in chunks) == source
    assert sum(chunk.size for chunk in chunks) / 24_000 == len(source) / 2 / 24_000


def test_audio_cpp_tuning_wire_is_response_frozen_not_live_ui_state():
    effective = {
        "model": "qwen3-tts-1.7b-base-bf16",
        "clone_mode": "full_icl",
        "max_reference_seconds": 20,
        "first_block_frames": 4,
        "steady_block_frames": 12,
        "left_context_frames": 25,
        "text_lookahead": 64,
        "phrase_flush_ms": 500,
        "temperature": 0.7,
        "top_k": 40,
        "top_p": 0.9,
        "repetition_penalty": 1.05,
        "seed": None,
    }
    snapshot = ResponseSynthesisSnapshot(
        key=("response_epoch", 7),
        input_epoch=7,
        response_epoch=7,
        response_id="resp_7",
        provider="qwen3tts-audiocpp",
        endpoint="http://127.0.0.1:8890/v1",
        model="qwen3-tts-1.7b-base-bf16",
        model_epoch="candidate-model-7",
        model_instance_id="candidate-instance-7",
        voice="clone:stable",
        reference_fingerprint="frozen",
        clone_content_revision=3,
        clone_content_hash="sha256:frozen",
        frozen_clone={"id": "clone:stable", "content_revision": 3, "content_hash": "sha256:frozen"},
        profile_id="balanced",
        profile_revision=4,
        tuning={
            "effective": effective,
            "resolved": {"model": "wrong-legacy-model", "temperature": 1.5},
            "overrides": {"top_k": 40},
            "scope": "session",  # UI-only: must not cross the API boundary.
        },
        language="Auto",
        seed=1234,
        seed_policy="response_random",
    )

    payload = Qwen3TTSHandler._audio_cpp_request_tuning(snapshot)

    assert payload == {
        "provider": "qwen3tts-audiocpp",
        "scope": "realtime",
        "profile_id": "balanced",
        "profile_revision": 4,
        "effective": effective,
        "overrides": {"top_k": 40, "seed": 1234},
    }
    snapshot.tuning["effective"]["temperature"] = 1.5
    assert payload["effective"]["temperature"] == 0.7


def test_audio_cpp_tuning_wire_does_not_promote_legacy_resolved_metadata():
    snapshot = ResponseSynthesisSnapshot(
        key=("response_epoch", 8), input_epoch=8, response_epoch=8, response_id="resp_8",
        provider="qwen3tts-audiocpp", endpoint=None, model=None, model_epoch=None, model_instance_id=None,
        voice=None, reference_fingerprint=None, clone_content_revision=None,
        clone_content_hash=None, frozen_clone=None, profile_id=None, profile_revision=None,
        tuning={"resolved": {"temperature": 0.6}}, language="Auto", seed=9, seed_policy="response_random",
    )

    with pytest.raises(RuntimeError, match="complete frozen tuning snapshot"):
        Qwen3TTSHandler._audio_cpp_request_tuning(snapshot)


def test_response_snapshot_prefers_explicit_seed_then_effective_then_legacy():
    effective = {"effective": {"seed": 101}, "resolved": {"seed": 202}, "overrides": {"seed": 303}}
    assert Qwen3TTSHandler._configured_seed(effective) == 303
    assert Qwen3TTSHandler._configured_seed({"effective": {"seed": 101}, "resolved": {"seed": 202}}) == 101
    assert Qwen3TTSHandler._configured_seed({"resolved": {"seed": 202}}) == 202
    # An explicit temporary clear means response-random, not profile fallback.
    assert Qwen3TTSHandler._configured_seed({"overrides": {"seed": None}, "effective": {"seed": 101}}) is None
    assert Qwen3TTSHandler._configured_seed({"overrides": {"seed": -1}, "effective": {"seed": 101}}) is None


def test_frozen_effective_snapshot_drives_phrase_queue_not_legacy_resolved():
    settings = {
        "provider": "qwen3tts-audiocpp",
        "effective": {"text_lookahead": 96, "phrase_flush_ms": 700},
        "resolved": {"text_lookahead": 256, "phrase_flush_ms": 2_000},
    }

    assert Qwen3TTSHandler._candidate_phrase_queue_settings_from_tuning(settings) == (96, 0.7)


def _immutable_candidate_tuning(effective):
    return {
        "provider": "qwen3tts-audiocpp",
        "profile_id": "balanced",
        "profile_revision": 2,
        "effective": effective,
        "overrides": {},
    }


def test_immutable_candidate_tuning_rejects_zero_temperature_and_decoder_over_capacity():
    effective = {
        "model": None,
        "clone_mode": "full_icl",
        "max_reference_seconds": 15,
        "first_block_frames": 4,
        "steady_block_frames": 12,
        "left_context_frames": 25,
        "text_lookahead": 64,
        "phrase_flush_ms": 500,
        "temperature": 0.7,
        "top_k": 40,
        "top_p": 0.9,
        "repetition_penalty": 1.0,
        "seed": None,
    }
    valid = _validated_audio_cpp_tuning(_immutable_candidate_tuning(effective))
    assert valid["effective"]["clone_mode"] == "full_icl"

    invalid_temperature = {**effective, "temperature": 0}
    with np.testing.assert_raises_regex(ValueError, "greater than zero"):
        _validated_audio_cpp_tuning(_immutable_candidate_tuning(invalid_temperature))

    invalid_decoder_shape = {**effective, "left_context_frames": 289, "steady_block_frames": 12}
    with np.testing.assert_raises_regex(ValueError, "must not exceed 300"):
        _validated_audio_cpp_tuning(_immutable_candidate_tuning(invalid_decoder_shape))
