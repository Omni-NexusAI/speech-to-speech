import json
from pathlib import Path
from queue import Queue
from threading import Event

import numpy as np
import pytest

import speech_to_speech.TTS.qwen3_tts_handler as qwen3_tts_module
from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.pipeline.messages import TTSInput
from speech_to_speech.TTS.qwen3_tts_handler import Qwen3TTSHandler


def _completed_candidate_outcome(request_id: str):
    class OutcomeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"requestId": request_id, "state": "completed", "eos": True}

    return OutcomeResponse()


def test_local_realtime_config_disables_compaction_for_chat_completions_backend():
    config_path = Path(__file__).resolve().parents[1] / "examples" / "local_gemma_fasterqwen3tts.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert config["llm_backend"] == "chat-completions"
    assert config["responses_api_chat_size"] == 30
    assert config["responses_api_compact_history"] is False
    assert config["responses_api_stream"] is True
    assert config["stream_batch_sentences"] == 1


def test_setup_openai_api_backend_skips_in_process_model_loading(monkeypatch):
    def _setup_mlx(self, *args, **kwargs):
        raise AssertionError("openai-api should not use mlx")

    def _setup_faster(self, *args, **kwargs):
        raise AssertionError("openai-api should not load FasterQwen3TTS in process")

    monkeypatch.setattr(qwen3_tts_module, "platform", "linux")
    monkeypatch.setattr(Qwen3TTSHandler, "_setup_mlx", _setup_mlx)
    monkeypatch.setattr(Qwen3TTSHandler, "_setup_faster", _setup_faster)
    monkeypatch.setattr(Qwen3TTSHandler, "_ensure_openai_api_backend_model", lambda self: None)

    handler = object.__new__(Qwen3TTSHandler)
    handler.setup(Event(), backend="openai-api")

    assert handler.backend == "openai_api"
    assert handler.device == "remote"
    assert handler.api_base_url == "http://127.0.0.1:8881/v1"
    assert handler.api_voice == "clone:16d9bb336799"
    assert handler.api_fallback_voice is None
    assert handler.api_backend_model == "1.7B-Base"
    assert handler._audio_cpp_native_warm_streams == set()
    assert handler._audio_cpp_native_lifecycle_epochs == {}
    assert not hasattr(handler, "model")


def test_audio_cpp_lifecycle_cache_fails_closed_after_legacy_status():
    handler = object.__new__(Qwen3TTSHandler)
    handler._audio_cpp_native_lifecycle_epochs = {}
    handler._audio_cpp_native_lifecycle_instances = {}
    handler._audio_cpp_native_warm_streams = set()
    endpoint = "http://127.0.0.1:8890/v1"
    model = "qwen3-tts-1.7b-base-bf16"
    base_key = (endpoint, model)

    handler._observe_audio_cpp_native_lifecycle_epoch(
        endpoint,
        model,
        {"engineEpoch": 7, "supervisorInstanceId": "a" * 32},
    )
    warm_key = handler._audio_cpp_native_stream_key(endpoint, model)
    handler._audio_cpp_native_warm_streams.add(warm_key)
    assert handler._audio_cpp_native_lifecycle_epochs[base_key] == "7"
    assert handler._audio_cpp_native_lifecycle_instances[base_key] == "a" * 32

    # The same long-lived handler later observes a downgraded/restarting
    # supervisor that has only legacy event metadata. It must not retain and
    # forward the previously cached canonical identity.
    handler._observe_audio_cpp_native_lifecycle_epoch(
        endpoint,
        model,
        {"events": [{"action": "supervisor-started", "at": "legacy"}]},
    )

    assert base_key not in handler._audio_cpp_native_lifecycle_epochs
    assert base_key not in handler._audio_cpp_native_lifecycle_instances
    assert not any(key[:2] == base_key for key in handler._audio_cpp_native_warm_streams)


def test_openai_api_payload_uses_streaming_pcm_and_clone_voice():
    handler = object.__new__(Qwen3TTSHandler)
    handler.api_model = "qwen3-tts"

    payload = handler._openai_api_payload("hello", "clone:16d9bb336799")

    assert payload == {
        "model": "qwen3-tts",
        "input": "hello",
        "voice": "clone:16d9bb336799",
        "response_format": "pcm",
        "stream": True,
        "language": "Auto",
    }


def test_process_openai_api_uses_session_voice_and_yields_audio(monkeypatch):
    handler = object.__new__(Qwen3TTSHandler)
    handler.should_listen = Event()
    handler.cancel_scope = None
    handler.speculative_turns = None
    handler.backend = "openai_api"
    handler.api_voice = "clone:16d9bb336799"
    handler.api_fallback_voice = None
    handler.blocksize = 512
    handler.queue_in = Queue()
    handler._last_tts_response_headers = {
        "x-tts-reference-pairing": "matched-excerpt"
    }

    captured = {}
    metrics = []

    def _process_openai_api(text, voice, **kwargs):
        captured["text"] = text
        captured["voice"] = voice
        captured.update(kwargs)
        yield np.zeros(512, dtype=np.int16)

    handler._process_openai_api = _process_openai_api
    handler._resolve_api_provider = lambda _runtime: ("faster", "http://127.0.0.1:8881/v1", "1.7B-Base")
    handler._emit_metric = lambda stage, status, *_args, **kwargs: metrics.append(
        (stage, status, kwargs)
    )
    monkeypatch.setattr(qwen3_tts_module.console, "print", lambda *args, **kwargs: None)
    runtime_config = RuntimeConfig()
    runtime_config.session.audio.output.voice = "clone:custom"

    outputs = list(handler.process(TTSInput(text="hello", runtime_config=runtime_config)))

    assert len(outputs) == 1
    assert captured == {
        "text": "hello",
        "voice": "clone:custom",
        "language": "Auto",
        "base_url": "http://127.0.0.1:8881/v1",
        "model": "1.7B-Base",
        "generation": None,
        "progressive_buffered": False,
    }
    done = next(kwargs for stage, status, kwargs in metrics if stage == "tts" and status == "done")
    assert "reference_pairing" not in done["detail"]
    assert handler._last_tts_response_headers == {}


def test_process_audio_cpp_native_propagates_profile_and_latency_metrics(monkeypatch):
    handler = object.__new__(Qwen3TTSHandler)
    handler.should_listen = Event()
    handler.cancel_scope = None
    handler.speculative_turns = None
    handler.backend = "openai_api"
    handler.api_voice = "clone:candidate"
    handler.api_fallback_voice = None
    handler.blocksize = 512
    handler.queue_in = Queue()
    handler.api_streaming_supported = True
    handler.api_candidate_streaming_mode = "native_incremental_pcm"
    handler.api_response_format = "pcm"
    captured = {}
    metrics = []

    def process_openai(text, voice, **kwargs):
        captured.update(text=text, voice=voice, **kwargs)
        handler._last_tts_response_headers = {
            "X-TTS-Reference-Source-Seconds": "18.125",
            "X-TTS-Reference-Requested-Limit-Seconds": "12",
            "X-TTS-Reference-Used-Seconds": "18.125",
            "X-TTS-Reference-Limit-Applied": "false",
            "X-TTS-Reference-Pairing": "full",
            "X-TTS-Delivery-Mode": "native-incremental-pcm",
        }
        yield np.zeros(512, dtype=np.int16)

    handler._process_openai_api = process_openai
    handler._resolve_api_provider = lambda _runtime: (
        "qwen3tts-audiocpp",
        "http://127.0.0.1:8890/v1",
        "qwen3-tts-1.7b-base-bf16",
    )
    handler._resolve_api_voice = lambda _runtime, _response: "clone:candidate"
    handler._audio_cpp_native_lifecycle_epochs = {
        ("http://127.0.0.1:8890/v1", "qwen3-tts-1.7b-base-bf16"): "9",
    }
    handler._audio_cpp_native_lifecycle_instances = {
        ("http://127.0.0.1:8890/v1", "qwen3-tts-1.7b-base-bf16"): "a" * 32,
    }
    # This metric-only handler test has no candidate profile service. Clone
    # immutability is covered independently, so keep this transport assertion
    # isolated from live profile I/O.
    monkeypatch.setattr(handler, "_freeze_candidate_clone", lambda *_args: None)
    handler._emit_metric = lambda stage, status, *_args, **kwargs: metrics.append(
        (stage, status, kwargs)
    )
    monkeypatch.setattr(handler, "_model_type", lambda: "base")
    runtime_config = RuntimeConfig()
    runtime_config.local_pipeline.update(
        tts_backend="qwen3tts-audiocpp",
        tts_tuning={
            "provider": "qwen3tts-audiocpp",
            "profile_id": "low-latency",
            "profile_revision": 1,
            "delivery_mode": "native_incremental_pcm",
            "overrides": {"top_k": 32},
            "effective": {
                "model": "qwen3-tts-1.7b-base-bf16",
                "clone_mode": "full_icl",
                "max_reference_seconds": 20,
                "first_block_frames": 4,
                "steady_block_frames": 12,
                "left_context_frames": 25,
                "text_lookahead": 32,
                "phrase_flush_ms": 250,
                "temperature": 0.7,
                "top_k": 32,
                "top_p": 0.9,
                "repetition_penalty": 1.05,
                "seed": None,
            },
        },
    )

    started = qwen3_tts_module.perf_counter()
    outputs = list(
        handler.process(
            TTSInput(
                text="A stable first phrase.",
                runtime_config=runtime_config,
                turn_id="turn-native",
                turn_revision=1,
                speech_stopped_at_s=qwen3_tts_module.perf_counter() - 0.25,
            )
        )
    )

    assert len(outputs) == 1
    assert qwen3_tts_module.perf_counter() - started < 0.1
    assert captured["native_candidate"] is True
    assert captured["progressive_buffered"] is False
    assert captured["tts_tuning"]["profile_id"] == "low-latency"
    assert captured["tts_tuning"]["provider"] == "qwen3tts-audiocpp"
    assert captured["tts_tuning"]["overrides"]["top_k"] == 32
    assert 0 <= captured["tts_tuning"]["overrides"]["seed"] <= 0xFFFFFFFF
    assert captured["tts_tuning"]["effective"]["model"] == "qwen3-tts-1.7b-base-bf16"
    assert captured["tts_tuning"]["effective"]["top_k"] == 32
    assert "resolved" not in captured["tts_tuning"]
    first_phrase = next(
        kwargs for stage, status, kwargs in metrics
        if stage == "gemma" and status == "first_stable_phrase"
    )
    request_start = next(
        kwargs for stage, status, kwargs in metrics
        if stage == "tts" and status == "request_start"
    )
    assert first_phrase["elapsed_ms"] >= 200
    assert request_start["elapsed_ms"] >= first_phrase["elapsed_ms"]
    first_pcm = next(kwargs for stage, status, kwargs in metrics if stage == "tts" and status == "first_audio")
    assert first_pcm["detail"]["mode"] == "native_incremental_pcm"
    assert first_pcm["detail"]["first_pcm_ms"] >= 0
    assert first_pcm["detail"]["end_to_end_ms"] >= request_start["elapsed_ms"]
    done = next(kwargs for stage, status, kwargs in metrics if stage == "tts" and status == "done")
    assert done["detail"]["rtf"] is not None
    expected_reference = {
        "reference_source_seconds": 18.125,
        "reference_requested_limit_seconds": 12.0,
        "reference_used_seconds": 18.125,
        "reference_limit_applied": False,
        "reference_pairing": "full",
        "delivery_mode": "native-incremental-pcm",
    }
    assert {
        key: done["detail"].get(key) for key in expected_reference
    } == expected_reference


def test_candidate_response_metric_detail_is_bounded_and_transcript_free():
    detail = Qwen3TTSHandler._candidate_response_metric_detail(
        {
            "X-TTS-Reference-Source-Seconds": "nan",
            "X-TTS-Reference-Requested-Limit-Seconds": "-1",
            "X-TTS-Reference-Used-Seconds": "8.5",
            "X-TTS-Reference-Truncated": "true",
            "X-TTS-Reference-Pairing": "matched-excerpt",
            "X-TTS-Delivery-Mode": "buffered-fallback",
            "X-TTS-Reference-Transcript": "must never enter metrics",
        }
    )

    assert detail == {
        "reference_used_seconds": 8.5,
        "reference_limit_applied": True,
        "reference_truncated": True,
        "reference_pairing": "matched-excerpt",
        "delivery_mode": "buffered-fallback",
        "reference_used": True,
    }


def test_openai_api_payload_preserves_explicit_multilingual_language():
    handler = object.__new__(Qwen3TTSHandler)
    handler.api_model = "qwen3-tts"

    payload = handler._openai_api_payload("Guten Tag", "clone:16d9bb336799", "German")

    assert payload["input"] == "Guten Tag"
    assert payload["language"] == "German"


def test_audio_cpp_buffered_payload_preserves_clone_and_disables_http_streaming():
    handler = object.__new__(Qwen3TTSHandler)
    handler.api_model = "qwen3-tts-1.7b-base-bf16"
    handler.api_response_format = "pcm"

    payload = handler._openai_api_payload(
        "Capability check.", "clone:776a491528c9", stream=False
    )

    assert payload["voice"] == "clone:776a491528c9"
    assert payload["response_format"] == "pcm"
    assert payload["stream"] is False


def test_audio_cpp_buffered_payload_uses_resident_provider_model():
    handler = object.__new__(Qwen3TTSHandler)
    handler.api_model = "qwen3-tts"
    handler.api_response_format = "pcm"

    payload = handler._openai_api_payload(
        "Capability check.",
        "clone:22fb07ef3a80",
        stream=False,
        model="qwen3-tts-1.7b-base-bf16",
    )

    assert payload["model"] == "qwen3-tts-1.7b-base-bf16"


def test_audio_cpp_does_not_retry_faster_fallback(monkeypatch):
    handler = object.__new__(Qwen3TTSHandler)
    handler.api_fallback_voice = "clone:16d9bb336799"
    handler._api_voice_for_backend = lambda value: value
    calls = []

    def fail_once(*args, **kwargs):
        calls.append(args[1])
        raise RuntimeError("candidate rejected selected clone")
        yield  # pragma: no cover

    handler._stream_openai_api_voice = fail_once

    with pytest.raises(RuntimeError, match="candidate rejected"):
        list(handler._process_openai_api("hello", "clone:776a491528c9", progressive_buffered=True))

    assert calls == ["clone:776a491528c9"]


def test_audio_cpp_buffered_phrase_uses_cancellable_offline_timeout(monkeypatch):
    handler = object.__new__(Qwen3TTSHandler)
    handler.api_base_url = "http://127.0.0.1:8890/v1"
    handler.api_model = "qwen3-tts-1.7b-base-bf16"
    handler.api_response_format = "pcm"
    handler.api_sample_rate = 24000
    handler.blocksize = 512
    handler.cancel_scope = None
    handler._active_response_lock = qwen3_tts_module.Lock()
    handler._active_response = None
    handler._openai_api_headers = lambda: {}
    seen = {}

    class Response:
        response_headers = {"x-tts-request-id": "a" * 32}

        def wait_for_headers(self):
            return None

        def iter_bytes(self):
            yield b"\x00\x00" * 768

        def close(self):
            return None

    def create_response(*_args, **kwargs):
        seen["timeout"] = kwargs["timeout"].read
        seen["payload"] = kwargs["json_body"]
        return Response()

    monkeypatch.setattr(qwen3_tts_module, "CancellableAsyncByteStream", create_response)
    monkeypatch.setattr(qwen3_tts_module.httpx, "get", lambda *_args, **_kwargs: _completed_candidate_outcome("a" * 32))
    chunks = list(handler._stream_openai_api_voice("short", "clone:22fb07ef3a80", progressive_buffered=True))

    assert chunks
    assert seen["timeout"] == 180.0
    assert seen["payload"]["stream"] is False


def test_audio_cpp_native_capability_selects_incremental_pcm(monkeypatch):
    handler = object.__new__(Qwen3TTSHandler)
    handler.audio_cpp_api_base_url = "http://127.0.0.1:8890/v1"
    handler._openai_api_headers = lambda: {}
    runtime_config = RuntimeConfig()
    runtime_config.local_pipeline["tts_backend"] = "qwen3tts-audiocpp"

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "engineEpoch": 9,
                "supervisorInstanceId": "a" * 32,
                "state": "loaded",
                "current": "qwen3-tts-1.7b-base-bf16",
                "loaded_models": ["qwen3-tts-1.7b-base-bf16"],
                "runtime": {
                    "native_incremental_pcm": True,
                    "progressive_phrase_pcm": True,
                    "sample_rate": 24000,
                },
            }

    monkeypatch.setattr(qwen3_tts_module.httpx, "get", lambda *_args, **_kwargs: Response())

    provider = handler._resolve_api_provider(runtime_config)

    assert provider == (
        "qwen3tts-audiocpp",
        "http://127.0.0.1:8890/v1",
        "qwen3-tts-1.7b-base-bf16",
    )
    assert handler.api_streaming_supported is True
    assert handler.api_candidate_streaming_mode == "native_incremental_pcm"
    assert handler.api_response_format == "pcm"
    assert handler._audio_cpp_native_stream_key(
        "http://127.0.0.1:8890/v1", "qwen3-tts-1.7b-base-bf16"
    )[2] == f"{'a' * 32}:9"


def test_audio_cpp_candidate_without_supervisor_instance_id_fails_closed(monkeypatch):
    handler = object.__new__(Qwen3TTSHandler)
    handler.audio_cpp_api_base_url = "http://127.0.0.1:8890/v1"
    handler._openai_api_headers = lambda: {}
    runtime_config = RuntimeConfig()
    runtime_config.local_pipeline["tts_backend"] = "qwen3tts-audiocpp"

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "engineEpoch": 1,
                "state": "loaded",
                "current": "qwen3-tts-1.7b-base-bf16",
                "loaded_models": ["qwen3-tts-1.7b-base-bf16"],
                "runtime": {"native_incremental_pcm": True, "progressive_phrase_pcm": True},
            }

    monkeypatch.setattr(qwen3_tts_module.httpx, "get", lambda *_args, **_kwargs: Response())
    with pytest.raises(RuntimeError, match="supervisor lifecycle identity"):
        handler._resolve_api_provider(runtime_config)


def test_audio_cpp_candidate_legacy_event_lifecycle_fails_closed(monkeypatch):
    """Event history cannot replace the canonical integer per-request guard."""

    handler = object.__new__(Qwen3TTSHandler)
    handler.audio_cpp_api_base_url = "http://127.0.0.1:8890/v1"
    handler._openai_api_headers = lambda: {}
    runtime_config = RuntimeConfig()
    runtime_config.local_pipeline["tts_backend"] = "qwen3tts-audiocpp"

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "supervisorInstanceId": "b" * 32,
                "state": "loaded",
                "current": "qwen3-tts-1.7b-base-bf16",
                "loaded_models": ["qwen3-tts-1.7b-base-bf16"],
                "events": [
                    {"action": "supervisor-started", "at": "legacy"},
                    {
                        "action": "model-ready",
                        "model": "qwen3-tts-1.7b-base-bf16",
                    },
                ],
                "runtime": {
                    "native_incremental_pcm": True,
                    "progressive_phrase_pcm": True,
                },
            }

    monkeypatch.setattr(qwen3_tts_module.httpx, "get", lambda *_args, **_kwargs: Response())

    with pytest.raises(RuntimeError, match="supervisor lifecycle identity"):
        handler._resolve_api_provider(runtime_config)


def test_audio_cpp_native_cold_budget_becomes_normal_after_pcm_per_model(monkeypatch):
    handler = object.__new__(Qwen3TTSHandler)
    handler.api_base_url = "http://127.0.0.1:8881/v1"
    handler.api_response_format = "pcm"
    handler.api_sample_rate = 24000
    handler.blocksize = 512
    handler.cancel_scope = None
    handler._active_response_lock = qwen3_tts_module.Lock()
    handler._active_response = None
    handler._audio_cpp_native_warm_streams = set()
    handler._openai_api_headers = lambda: {}
    observed_budgets = []

    class Response:
        response_headers = {
            "x-tts-streaming-mode": "native-incremental-pcm",
            "x-tts-native-engine-chunk-proof": "two-distinct-sse-delta-events",
            "x-tts-request-id": "b" * 32,
        }

        def wait_for_headers(self):
            return None

        def iter_bytes(self):
            yield b"\x00\x00" * 512
            yield b"\x00\x00" * 512

        def close(self):
            return None

    def create_response(*_args, **kwargs):
        observed_budgets.append(kwargs["timeout"].read)
        return Response()

    monkeypatch.setattr(qwen3_tts_module, "CancellableAsyncByteStream", create_response)
    monkeypatch.setattr(qwen3_tts_module.httpx, "get", lambda *_args, **_kwargs: _completed_candidate_outcome("b" * 32))
    candidate_url = "http://127.0.0.1:8890/v1"

    first = list(
        handler._stream_openai_api_voice(
            "short reply",
            "clone:candidate",
            base_url=candidate_url,
            model="qwen3-tts-0.6b-base-bf16",
            native_candidate=True,
        )
    )
    warm = list(
        handler._stream_openai_api_voice(
            "short reply",
            "clone:candidate",
            base_url=candidate_url,
            model="qwen3-tts-0.6b-base-bf16",
            native_candidate=True,
        )
    )
    different_model = list(
        handler._stream_openai_api_voice(
            "short reply",
            "clone:candidate",
            base_url=candidate_url,
            model="qwen3-tts-1.7b-base-bf16",
            native_candidate=True,
        )
    )

    assert first and warm and different_model
    assert observed_budgets == [45.0, 12.0, 45.0]
    assert handler._audio_cpp_native_warm_streams == {
        (candidate_url, "qwen3-tts-0.6b-base-bf16", "unobserved"),
        (candidate_url, "qwen3-tts-1.7b-base-bf16", "unobserved"),
    }


def test_audio_cpp_native_pcm_is_yielded_before_provider_stream_finishes(monkeypatch):
    handler = object.__new__(Qwen3TTSHandler)
    handler.api_base_url = "http://127.0.0.1:8881/v1"
    handler.api_response_format = "pcm"
    handler.api_sample_rate = 24000
    handler.blocksize = 512
    handler.cancel_scope = None
    handler._active_response_lock = qwen3_tts_module.Lock()
    handler._active_response = None
    handler._audio_cpp_native_warm_streams = set()
    handler._openai_api_headers = lambda: {}
    provider_progress = []

    class Response:
        response_headers = {
            "x-tts-streaming-mode": "native-incremental-pcm",
            "x-tts-native-engine-chunk-proof": "two-distinct-sse-delta-events",
            "x-tts-request-id": "c" * 32,
        }

        def wait_for_headers(self):
            return None

        def iter_bytes(self):
            provider_progress.append("first")
            yield b"\x00\x00" * 768
            provider_progress.append("second")
            yield b"\x00\x00" * 768
            provider_progress.append("complete")

        def close(self):
            provider_progress.append("closed")

    monkeypatch.setattr(
        qwen3_tts_module,
        "CancellableAsyncByteStream",
        lambda *_args, **_kwargs: Response(),
    )
    monkeypatch.setattr(qwen3_tts_module.httpx, "get", lambda *_args, **_kwargs: _completed_candidate_outcome("c" * 32))

    stream = handler._stream_openai_api_voice(
        "The provider is still generating this phrase.",
        "clone:candidate",
        base_url="http://127.0.0.1:8890/v1",
        model="qwen3-tts-0.6b-base-bf16",
        native_candidate=True,
    )

    first = next(stream)
    # Engine incrementality was proven by the supervisor before headers. The
    # first downstream HTTP read is delivered immediately; its byte boundary
    # is not confused with an engine speech.audio.delta boundary.
    assert first.size == 512
    assert provider_progress == ["first"]
    remaining = list(stream)
    assert remaining
    assert provider_progress == ["first", "second", "complete", "closed"]
    assert np.concatenate([first, *remaining]).size == 1536


@pytest.mark.parametrize(
    ("headers", "chunks", "message"),
    [
        ({}, [b"\x00\x00" * 16, b"\x00\x00" * 16], "did not prove"),
        (
            {"x-tts-streaming-mode": "native-incremental-pcm"},
            [b"\x00\x00" * 16],
            "did not prove two engine-owned",
        ),
    ],
)
def test_audio_cpp_native_pcm_requires_supervisor_engine_chunk_proof(monkeypatch, headers, chunks, message):
    handler = object.__new__(Qwen3TTSHandler)
    handler.api_base_url = "http://127.0.0.1:8881/v1"
    handler.api_response_format = "pcm"
    handler.api_sample_rate = 24000
    handler.blocksize = 512
    handler.cancel_scope = None
    handler._active_response_lock = qwen3_tts_module.Lock()
    handler._active_response = None
    handler._audio_cpp_native_warm_streams = set()

    class Response:
        response_headers = headers

        def wait_for_headers(self):
            return None

        def iter_bytes(self):
            yield from chunks

        def close(self):
            return None

    monkeypatch.setattr(
        qwen3_tts_module,
        "CancellableAsyncByteStream",
        lambda *_args, **_kwargs: Response(),
    )

    with pytest.raises(RuntimeError, match=message):
        list(
            handler._stream_openai_api_voice(
                "proof gate",
                "clone:candidate",
                base_url="http://127.0.0.1:8890/v1",
                model="qwen3-tts-0.6b-base-bf16",
                native_candidate=True,
            )
        )
    assert handler._audio_cpp_native_warm_streams == set()


def test_audio_cpp_same_model_reload_lifecycle_becomes_cold_again(monkeypatch):
    handler = object.__new__(Qwen3TTSHandler)
    handler.audio_cpp_api_base_url = "http://127.0.0.1:8890/v1"
    handler.api_base_url = "http://127.0.0.1:8881/v1"
    handler.api_response_format = "pcm"
    handler.api_sample_rate = 24000
    handler.blocksize = 512
    handler.cancel_scope = None
    handler._active_response_lock = qwen3_tts_module.Lock()
    handler._active_response = None
    handler._audio_cpp_native_warm_streams = set()
    handler._audio_cpp_native_lifecycle_epochs = {}
    handler._audio_cpp_native_lifecycle_instances = {}
    handler._openai_api_headers = lambda: {}
    runtime_config = RuntimeConfig()
    runtime_config.local_pipeline["tts_backend"] = "qwen3tts-audiocpp"
    model = "qwen3-tts-0.6b-base-bf16"
    lifecycle_at = {"value": "2026-08-04T16:00:00Z"}
    lifecycle_epoch = {"value": 1}
    lifecycle_instance = {"value": "a" * 32}
    observed_budgets = []

    class StatusResponse:
        def raise_for_status(self):
            return None

        def json(self):
            at = lifecycle_at["value"]
            return {
                "engineEpoch": lifecycle_epoch["value"],
                "supervisorInstanceId": lifecycle_instance["value"],
                "state": "loaded",
                "current": model,
                "loaded_models": [model],
                "runtime": {
                    "native_incremental_pcm": True,
                    "progressive_phrase_pcm": True,
                    "sample_rate": 24000,
                },
                "events": [
                    {"at": at, "action": "supervisor-started", "model": model},
                    {"at": at, "action": "child-start", "model": model},
                    {"at": at, "action": "model-ready", "model": model},
                ],
            }

    class StreamResponse:
        response_headers = {
            "x-tts-streaming-mode": "native-incremental-pcm",
            "x-tts-native-engine-chunk-proof": "two-distinct-sse-delta-events",
            "x-tts-request-id": "d" * 32,
        }

        def wait_for_headers(self):
            return None

        def iter_bytes(self):
            yield b"\x00\x00" * 512
            yield b"\x00\x00" * 512

        def close(self):
            return None

    def create_stream(*_args, **kwargs):
        observed_budgets.append(kwargs["timeout"].read)
        return StreamResponse()

    def get_response(url, *_args, **_kwargs):
        if "/audio/outcomes/" in url:
            return _completed_candidate_outcome("d" * 32)
        return StatusResponse()

    monkeypatch.setattr(qwen3_tts_module.httpx, "get", get_response)
    monkeypatch.setattr(qwen3_tts_module, "CancellableAsyncByteStream", create_stream)

    assert handler._resolve_api_provider(runtime_config)[2] == model
    assert list(
        handler._stream_openai_api_voice(
            "short reply",
            "clone:candidate",
            base_url=handler.audio_cpp_api_base_url,
            model=model,
            native_candidate=True,
        )
    )
    assert list(
        handler._stream_openai_api_voice(
            "short reply",
            "clone:candidate",
            base_url=handler.audio_cpp_api_base_url,
            model=model,
            native_candidate=True,
        )
    )

    lifecycle_at["value"] = "2026-08-04T16:01:00Z"
    lifecycle_epoch["value"] = 2
    assert handler._resolve_api_provider(runtime_config)[2] == model
    assert list(
        handler._stream_openai_api_voice(
            "short reply",
            "clone:candidate",
            base_url=handler.audio_cpp_api_base_url,
            model=model,
            native_candidate=True,
        )
    )

    assert observed_budgets == [45.0, 12.0, 45.0]
    assert len(handler._audio_cpp_native_warm_streams) == 1
    assert handler._audio_cpp_native_stream_key(handler.audio_cpp_api_base_url, model)[2] != "unobserved"


def test_audio_cpp_cold_budget_is_not_used_or_marked_for_other_providers(monkeypatch):
    handler = object.__new__(Qwen3TTSHandler)
    handler.api_base_url = "http://127.0.0.1:8881/v1"
    handler.api_response_format = "pcm"
    handler.api_sample_rate = 24000
    handler.blocksize = 512
    handler.cancel_scope = None
    handler._active_response_lock = qwen3_tts_module.Lock()
    handler._active_response = None
    handler._audio_cpp_native_warm_streams = set()
    observed_budgets = []

    class Response:
        response_headers = {}

        def wait_for_headers(self):
            return None

        def iter_bytes(self):
            yield b"\x00\x00" * 512

        def close(self):
            return None

    def create_response(*_args, **kwargs):
        observed_budgets.append(kwargs["timeout"].read)
        return Response()

    monkeypatch.setattr(qwen3_tts_module, "CancellableAsyncByteStream", create_response)

    assert list(
        handler._stream_openai_api_voice(
            "short reply",
            "clone:faster",
            base_url="http://127.0.0.1:8881/v1",
            model="qwen3-tts",
            native_candidate=False,
        )
    )
    assert observed_budgets == [12.0]
    assert handler._audio_cpp_native_warm_streams == set()


def test_audio_cpp_native_cancellation_does_not_warm_or_replay_buffered(monkeypatch):
    handler = object.__new__(Qwen3TTSHandler)
    handler.api_base_url = "http://127.0.0.1:8881/v1"
    handler.api_response_format = "pcm"
    handler.api_sample_rate = 24000
    handler.blocksize = 512
    handler.cancel_scope = None
    handler._active_response_lock = qwen3_tts_module.Lock()
    handler._active_response = None
    handler._audio_cpp_native_warm_streams = set()

    class Response:
        response_headers = {}
        closed = False

        def wait_for_headers(self):
            raise qwen3_tts_module.StreamCancelled("cancelled")

        def iter_bytes(self):
            raise AssertionError("cancelled response must not emit bytes")

        def close(self):
            self.closed = True

    response = Response()
    monkeypatch.setattr(
        qwen3_tts_module,
        "CancellableAsyncByteStream",
        lambda *_args, **_kwargs: response,
    )

    chunks = list(
        handler._process_openai_api(
            "short reply",
            "clone:candidate",
            base_url="http://127.0.0.1:8890/v1",
            model="qwen3-tts-0.6b-base-bf16",
            native_candidate=True,
        )
    )

    assert chunks == []
    assert response.closed is True
    assert handler._last_streaming_mode == "native_incremental_pcm"
    assert handler._audio_cpp_native_warm_streams == set()


def test_audio_cpp_native_failure_requires_explicit_buffered_mode_selection():
    handler = object.__new__(Qwen3TTSHandler)
    handler.api_fallback_voice = "clone:faster-must-not-run"
    calls = []

    def stream(_text, voice, **kwargs):
        calls.append(
            (
                voice,
                kwargs["progressive_buffered"],
                kwargs.get("tts_tuning"),
                kwargs.get("native_candidate", False),
            )
        )
        raise RuntimeError("native experiment unavailable")
        yield  # pragma: no cover - keep this helper a generator

    handler._stream_openai_api_voice = stream
    tuning = {"provider": "qwen3tts-audiocpp", "profile_id": "balanced", "overrides": {}}

    with pytest.raises(RuntimeError, match="native experiment unavailable"):
        list(
            handler._process_openai_api(
                "hello",
                "clone:candidate-only",
                native_candidate=True,
                tts_tuning=tuning,
            )
        )

    assert calls == [
        ("clone:candidate-only", False, tuning, True),
    ]
    assert handler._last_streaming_mode == "native_failed"


def test_audio_cpp_native_request_never_replays_buffered_after_emitting_pcm():
    handler = object.__new__(Qwen3TTSHandler)
    handler.api_fallback_voice = None
    calls = []

    def stream(_text, _voice, **kwargs):
        calls.append(kwargs["progressive_buffered"])
        yield np.zeros(512, dtype=np.int16)
        raise RuntimeError("stream failed after audible PCM")

    handler._stream_openai_api_voice = stream

    with pytest.raises(RuntimeError, match="after audible PCM"):
        list(handler._process_openai_api("hello", "clone:candidate", native_candidate=True))

    assert calls == [False]


def test_phrase_failure_suppresses_later_phrases_from_the_same_response(monkeypatch):
    handler = object.__new__(Qwen3TTSHandler)
    handler.should_listen = Event()
    handler.cancel_scope = None
    handler.speculative_turns = None
    handler.backend = "openai_api"
    handler.api_voice = "clone:test"
    handler.api_fallback_voice = None
    handler.blocksize = 512
    handler.queue_in = Queue()
    calls = []
    metrics = []

    def fail_phrase(text, _voice, **_kwargs):
        calls.append(text)
        raise RuntimeError("provider stream failed")
        yield  # pragma: no cover - keep this helper a generator

    handler._process_openai_api = fail_phrase
    handler._resolve_api_provider = lambda _runtime: (
        "faster",
        "http://127.0.0.1:8881/v1",
        "1.7B-Base",
    )
    handler._resolve_api_voice = lambda _runtime, _response: "clone:test"
    handler._emit_metric = lambda stage, status, *_args, **kwargs: metrics.append(
        (stage, status, kwargs)
    )
    monkeypatch.setattr(handler, "_model_type", lambda: "base")
    runtime_config = RuntimeConfig()

    first = TTSInput(
        text="First phrase.",
        runtime_config=runtime_config,
        turn_id="turn-failed-tts",
        turn_revision=1,
    )
    second = TTSInput(
        text="Second phrase must not play.",
        runtime_config=runtime_config,
        turn_id="turn-failed-tts",
        turn_revision=1,
    )

    assert list(handler.process(first)) == []
    assert list(handler.process(second)) == []
    assert calls == ["First phrase."]
    assert any(
        stage == "tts" and status == "suppressed_after_failure"
        for stage, status, _kwargs in metrics
    )


def test_audio_cpp_cancelled_before_first_phrase_is_not_reported_done(monkeypatch):
    handler = object.__new__(Qwen3TTSHandler)
    handler.should_listen = Event()
    handler.backend = "openai_api"
    handler.api_voice = "clone:22fb07ef3a80"
    handler.api_fallback_voice = None
    handler.blocksize = 512
    handler.queue_in = Queue()
    handler.speculative_turns = None
    handler.cancel_scope = type("Scope", (), {"is_stale": lambda self, generation: True})()
    handler._resolve_api_provider = lambda _runtime: (_ for _ in ()).throw(
        AssertionError("stale text must not resolve or occupy the external TTS provider")
    )
    handler._resolve_api_voice = lambda _runtime, _response: "clone:22fb07ef3a80"
    handler._process_openai_api = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("stale text must not start synthesis")
    )
    metrics = []
    handler._emit_metric = lambda stage, status, *_args, **kwargs: metrics.append((stage, status, kwargs))
    monkeypatch.setattr(handler, "_model_type", lambda: "base")
    runtime_config = RuntimeConfig()
    runtime_config.local_pipeline["tts_backend"] = "qwen3tts-audiocpp"

    outputs = list(
        handler.process(
            TTSInput(text="This turn was interrupted.", runtime_config=runtime_config, cancel_generation=7)
        )
    )

    assert outputs == []
    assert any(stage == "tts" and status == "cancelled_before_audio" for stage, status, _ in metrics)
    assert not any(stage == "tts" and status == "done" for stage, status, _ in metrics)


@pytest.mark.parametrize(
    ("text", "language"),
    [
        ("\u041f\u0440\u0438\u0432\u0435\u0442", "Russian"),
        ("\u3053\u3093\u306b\u3061\u306f", "Japanese"),
        ("\uc548\ub155\ud558\uc138\uc694", "Korean"),
        ("\u4f60\u597d", "Chinese"),
    ],
)
def test_api_language_uses_unicode_script_detection_as_fallback(text, language):
    assert Qwen3TTSHandler._api_language_name(None, text) == language


def test_openai_api_backend_model_ready_requires_1_7b_loaded():
    handler = object.__new__(Qwen3TTSHandler)
    handler.api_backend_model = "1.7B-Base"

    assert handler._openai_api_backend_model_ready(
        {"current": "1.7B-Base", "state": "loaded", "loaded_models": ["1.7B-Base"]}
    )
    assert not handler._openai_api_backend_model_ready(
        {"current": "0.6B-Base", "state": "loaded", "loaded_models": ["0.6B-Base"]}
    )
    assert not handler._openai_api_backend_model_ready(
        {"current": "1.7B-Base", "state": "unloaded", "loaded_models": []}
    )


def test_groxaxo_provider_uses_voice_studio_loaded_base_model(monkeypatch):
    handler = object.__new__(Qwen3TTSHandler)
    handler.groxaxo_api_base_url = "http://127.0.0.1:8882/v1"
    runtime_config = RuntimeConfig(local_pipeline={"tts_backend": "groxaxo"})

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"state": "loaded", "current": "0.6B-Base", "loaded_models": ["0.6B-Base"]}

    monkeypatch.setattr(qwen3_tts_module.httpx, "get", lambda *args, **kwargs: Response())

    assert handler._resolve_api_provider(runtime_config) == (
        "groxaxo",
        "http://127.0.0.1:8882/v1",
        "0.6B-Base",
    )


def test_groxaxo_provider_rejects_missing_base_model(monkeypatch):
    handler = object.__new__(Qwen3TTSHandler)
    runtime_config = RuntimeConfig(local_pipeline={"tts_backend": "groxaxo"})

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"state": "loaded", "current": "1.7B-CustomVoice", "loaded_models": ["1.7B-CustomVoice"]}

    monkeypatch.setattr(qwen3_tts_module.httpx, "get", lambda *args, **kwargs: Response())

    with pytest.raises(RuntimeError, match="no 0.6B-Base or 1.7B-Base"):
        handler._resolve_api_provider(runtime_config)


def test_tts_runaway_budget_is_bounded_and_scales_with_estimated_speech():
    handler = object.__new__(Qwen3TTSHandler)

    assert handler._runaway_budget_s("short reply") == 12.0
    assert handler._runaway_budget_s("word " * 1000) == 60.0


def test_cancel_active_closes_current_tts_stream():
    handler = object.__new__(Qwen3TTSHandler)
    handler._active_response_lock = qwen3_tts_module.Lock()

    class Response:
        closed = False

        def close(self):
            self.closed = True

    response = Response()
    handler._active_response = response

    handler.cancel_active()

    assert response.closed is True
    assert handler._active_response is None


def test_streaming_tts_runaway_is_aborted_and_closed(monkeypatch):
    handler = object.__new__(Qwen3TTSHandler)
    handler.api_base_url = "http://127.0.0.1:8881/v1"
    handler.api_model = "qwen3-tts"
    handler.api_response_format = "pcm"
    handler.api_sample_rate = 24000
    handler.blocksize = 512
    handler.cancel_scope = None
    handler._active_response_lock = qwen3_tts_module.Lock()
    handler._active_response = None

    class Response:
        closed = False

        def wait_for_headers(self):
            return None

        def iter_bytes(self):
            yield b"\x00\x00"

        def close(self):
            self.closed = True

    response = Response()

    times = iter([0.0, 13.0])
    monkeypatch.setattr(qwen3_tts_module, "perf_counter", lambda: next(times))
    monkeypatch.setattr(qwen3_tts_module, "CancellableAsyncByteStream", lambda *_args, **_kwargs: response)

    with pytest.raises(qwen3_tts_module.TTSRunawayError):
        list(handler._stream_openai_api_voice("short reply", "clone:16d9bb336799"))

    assert response.closed is True


def test_ensure_openai_api_backend_model_switches_to_1_7b(monkeypatch):
    calls = []

    class FakeResponse:
        status_code = 200

        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def get(self, url, headers):
            calls.append(("GET", url, headers, None))
            return FakeResponse(
                {
                    "current": "0.6B-Base",
                    "available": ["0.6B-Base", "1.7B-Base"],
                    "loaded_models": ["0.6B-Base"],
                    "state": "loaded",
                }
            )

        def post(self, url, headers, json):
            calls.append(("POST", url, headers, json))
            return FakeResponse(
                {
                    "current": "1.7B-Base",
                    "available": ["0.6B-Base", "1.7B-Base"],
                    "loaded_models": ["1.7B-Base"],
                    "state": "loaded",
                }
            )

    monkeypatch.setattr(qwen3_tts_module.httpx, "Client", FakeClient)
    handler = object.__new__(Qwen3TTSHandler)
    handler.api_base_url = "http://127.0.0.1:8881/v1"
    handler.api_backend_model = "1.7B-Base"
    handler.api_timeout = None
    handler.api_key = None

    handler._ensure_openai_api_backend_model()

    assert calls[0][0:2] == ("GET", "http://127.0.0.1:8881/v1/backend/models")
    assert calls[1] == (
        "POST",
        "http://127.0.0.1:8881/v1/backend/models/switch",
        {"Content-Type": "application/json"},
        {"model_key": "1.7B-Base"},
    )


def test_ensure_openai_api_backend_model_accepts_fixed_faster_container(monkeypatch):
    calls = []

    class FakeResponse:
        def __init__(self, payload, status_code=200):
            self._payload = payload
            self.status_code = status_code

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def get(self, url, headers):
            calls.append(("GET", url))
            if url.endswith("/backend/models"):
                return FakeResponse({}, status_code=404)
            return FakeResponse({"status": "ok", "backend": "faster-qwen3-tts", "model_loaded": False})

    monkeypatch.setattr(qwen3_tts_module.httpx, "Client", FakeClient)
    handler = object.__new__(Qwen3TTSHandler)
    handler.api_base_url = "http://127.0.0.1:8881/v1"
    handler.api_backend_model = "1.7B-Base"
    handler.api_timeout = None
    handler.api_key = None
    handler.api_response_format = "pcm"

    handler._ensure_openai_api_backend_model()

    assert calls == [
        ("GET", "http://127.0.0.1:8881/v1/backend/models"),
        ("GET", "http://127.0.0.1:8881/health"),
    ]
    assert handler.api_response_format == "wav"


def test_ensure_openai_api_backend_model_uses_fixed_faster_pcm_streaming_when_advertised(monkeypatch):
    class FakeResponse:
        def __init__(self, payload, status_code=200):
            self._payload = payload
            self.status_code = status_code

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, timeout=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def get(self, url, headers=None):
            if url.endswith("/backend/models"):
                return FakeResponse({}, status_code=404)
            return FakeResponse(
                {
                    "backend": "faster-qwen3-tts",
                    "capabilities": {
                        "native_pcm_streaming": True,
                        "stream_response_format": "pcm",
                        "sample_rate": 24000,
                    },
                }
            )

    monkeypatch.setattr(qwen3_tts_module.httpx, "Client", FakeClient)
    handler = object.__new__(Qwen3TTSHandler)
    handler.api_base_url = "http://127.0.0.1:8881/v1"
    handler.api_backend_model = "1.7B-Base"
    handler.api_timeout = None
    handler.api_key = None
    handler.api_response_format = "wav"
    handler.api_sample_rate = 16000
    handler.api_streaming_supported = False

    handler._ensure_openai_api_backend_model()

    assert handler.api_response_format == "pcm"
    assert handler.api_sample_rate == 24000
    assert handler.api_streaming_supported is True


def test_api_voice_for_backend_maps_clone_id_to_profile_name(tmp_path):
    profile_dir = tmp_path / "profiles" / "16d9bb336799"
    profile_dir.mkdir(parents=True)
    (profile_dir / "meta.json").write_text(
        json.dumps({"profile_id": "16d9bb336799", "name": "J.A.R.V.I.S", "task_type": "Base"}),
        encoding="utf-8",
    )
    handler = object.__new__(Qwen3TTSHandler)
    handler.api_voice_library_dir = tmp_path

    assert handler._api_voice_for_backend("clone:16d9bb336799") == "clone:J.A.R.V.I.S"
    assert handler._api_voice_for_backend("clone:missing") == "clone:missing"
