from __future__ import annotations

from types import SimpleNamespace

import httpx
import numpy as np
import pytest

import speech_to_speech.LLM.base_openai_compatible_language_model as base_llm
from speech_to_speech.api.openai_realtime.runtime_config import ModelEndpointConfig, RuntimeConfig
from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.LLM.chat_completions_language_model import ChatCompletionsApiModelHandler
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.STT.gemma_audio_handler import GemmaAudioSTTHandler


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _ProbeClient:
    calls: list[tuple[str, dict[str, str]]] = []

    def __init__(self, *, headers=None, **kwargs):
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def get(self, url):
        self.calls.append((url, self.headers))
        if url.endswith("/models"):
            return _Response({"data": [{"id": "remote-gemma-audio"}]})
        return _Response({"default_generation_settings": {"n_ctx": 16384}})


def test_remote_validation_uses_only_selected_endpoint_and_redacts_key(monkeypatch):
    _ProbeClient.calls.clear()
    monkeypatch.setattr("speech_to_speech.api.openai_realtime.service.httpx.Client", _ProbeClient)
    service = RealtimeService(context_tokenizer_base_url="http://127.0.0.1:8818/v1")

    endpoint = service.validate_model_endpoint(
        provider="remote",
        base_url="http://10.0.0.25:8080",
        model="gemma-audio-small",
        api_key="secret-key",
    )

    assert endpoint.base_url == "http://10.0.0.25:8080/v1"
    assert endpoint.model == "gemma-audio-small"
    assert endpoint.advertised_model == "remote-gemma-audio"
    assert endpoint.context_window == 16384
    assert all("127.0.0.1:8818" not in url for url, _ in _ProbeClient.calls)
    assert all(headers == {"Authorization": "Bearer secret-key"} for _, headers in _ProbeClient.calls)
    assert endpoint.redacted()["api_key_set"] is True
    assert "api_key" not in endpoint.redacted()


def test_remote_validation_failure_does_not_fall_back_local(monkeypatch):
    class FailingClient(_ProbeClient):
        def get(self, url):
            raise httpx.ConnectError("offline", request=httpx.Request("GET", url))

    monkeypatch.setattr("speech_to_speech.api.openai_realtime.service.httpx.Client", FailingClient)
    service = RealtimeService(context_tokenizer_base_url="http://127.0.0.1:8818/v1")

    with pytest.raises(httpx.ConnectError):
        service.validate_model_endpoint(
            provider="remote",
            base_url="http://10.0.0.30:8080/v1",
            model="remote-model",
        )


def test_speculative_revision_notifies_only_on_newer_revision():
    tracker = SpeculativeTurnTracker()
    observed = []
    tracker.add_revision_listener(lambda turn_id, revision: observed.append((turn_id, revision)))

    tracker.observe("turn", 0)
    tracker.observe("turn", 0)
    tracker.observe("turn", 2)

    assert observed == [("turn", 0), ("turn", 2)]


def test_gemma_uses_conversation_endpoint_and_250ms_settle():
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup()
    runtime = RuntimeConfig(
        model_endpoint=ModelEndpointConfig(
            provider="remote",
            base_url="http://10.0.0.40:8080/v1",
            model="gemma-e4b-audio",
            api_key="remote-key",
        )
    )
    vad_audio = SimpleNamespace(runtime_config=runtime)

    payload = handler._payload(np.zeros(160, dtype=np.float32), vad_audio)

    assert payload["model"] == "gemma-e4b-audio"
    assert handler._model_endpoint(vad_audio) == (
        "http://10.0.0.40:8080/v1",
        "gemma-e4b-audio",
        "remote-key",
    )
    assert handler._headers("remote-key")["Authorization"] == "Bearer remote-key"
    assert handler.final_revision_settle_s == 0.25
    assert handler.timeout.connect == 5.0
    assert handler.timeout.read == 30.0


def test_new_revision_closes_obsolete_direct_audio_transport():
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup()

    class Closable:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    response = Closable()
    client = Closable()
    handler._active_resources = {response, client}
    handler._active_turn = ("turn", 1)

    handler._on_revision_observed("turn", 2)

    assert response.closed
    assert client.closed
    assert handler._active_turn is None


def test_model_endpoint_config_defaults_local():
    config = RuntimeConfig()
    assert config.model_endpoint.provider == "local"


def test_remote_llm_client_has_no_retries_and_local_client_is_not_touched(monkeypatch):
    created = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(base_llm, "OpenAI", FakeOpenAI)
    handler = object.__new__(ChatCompletionsApiModelHandler)
    handler.client = object()
    handler.model_name = "local-model"
    handler.base_url = "http://127.0.0.1:8818/v1"
    handler.api_key = None
    handler.request_timeout = httpx.Timeout(30.0, connect=5.0)
    handler.disable_thinking = True
    handler.reasoning_effort = None
    handler._extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

    local_client, local_model, _ = handler._client_for(RuntimeConfig())
    assert local_client is handler.client
    assert local_model == "local-model"
    assert created == []

    remote = RuntimeConfig(
        model_endpoint=ModelEndpointConfig(
            provider="remote",
            base_url="http://10.0.0.50:8080/v1",
            model="remote-model",
        )
    )
    _, remote_model, _ = handler._client_for(remote)

    assert remote_model == "remote-model"
    assert created[0]["base_url"] == "http://10.0.0.50:8080/v1"
    assert created[0]["api_key"] == "not-needed"
    assert created[0]["max_retries"] == 0
