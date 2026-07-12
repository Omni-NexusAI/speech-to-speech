import json
from pathlib import Path
from queue import Queue
from threading import Event

import numpy as np

import speech_to_speech.TTS.qwen3_tts_handler as qwen3_tts_module
from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.pipeline.messages import TTSInput
from speech_to_speech.TTS.qwen3_tts_handler import Qwen3TTSHandler


def test_local_realtime_config_disables_compaction_for_chat_completions_backend():
    config_path = Path(__file__).resolve().parents[1] / "examples" / "local_gemma_fasterqwen3tts.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert config["llm_backend"] == "chat-completions"
    assert config["responses_api_chat_size"] == 30
    assert config["responses_api_compact_history"] is False


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
    assert not hasattr(handler, "model")


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

    captured = {}

    def _process_openai_api(text, voice):
        captured["text"] = text
        captured["voice"] = voice
        yield np.zeros(512, dtype=np.int16)

    handler._process_openai_api = _process_openai_api
    monkeypatch.setattr(qwen3_tts_module.console, "print", lambda *args, **kwargs: None)
    runtime_config = RuntimeConfig()
    runtime_config.session.audio.output.voice = "clone:custom"

    outputs = list(handler.process(TTSInput(text="hello", runtime_config=runtime_config)))

    assert len(outputs) == 1
    assert captured == {"text": "hello", "voice": "clone:custom"}


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
