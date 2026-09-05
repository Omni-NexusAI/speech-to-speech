"""Bounded unit coverage for the explicit direct-audio follow-up diagnostic."""

from __future__ import annotations

import argparse
import asyncio
import base64
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture()
def diagnostic_module(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("validate_direct_audio_followups_test", scripts / "validate_direct_audio_followups.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(tmp_path: Path, *, audio_dir: Path | None = None):
    return argparse.Namespace(
        clips=[tmp_path / "explicit-followup.wav"],
        websocket="ws://127.0.0.1:8765/v1/realtime",
        candidate="http://127.0.0.1:8890",
        gemma="http://127.0.0.1:8081/v1",
        model="selected-local-model",
        voice="clone:diagnostic",
        api_key_env="HFRT_DIAGNOSTIC_GEMMA_KEY",
        audio_dir=audio_dir,
    )


class _HealthResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return self

    def json(self):
        return self.payload


class _HealthClient:
    def __init__(self, calls, payload, **_kwargs):
        self.calls = calls
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def get(self, url):
        self.calls.append(("GET", url))
        return _HealthResponse(self.payload)


class _Socket:
    def __init__(self, events):
        self.events = iter(events)
        self.sent = []

    async def recv(self):
        return json.dumps(next(self.events))

    async def send(self, message):
        self.sent.append(json.loads(message))


class _SocketContext:
    def __init__(self, socket):
        self.socket = socket

    async def __aenter__(self):
        return self.socket

    async def __aexit__(self, *_args):
        return False


def _install_healthy_candidate(monkeypatch, module, calls):
    monkeypatch.setattr(
        module.httpx,
        "Client",
        lambda **kwargs: _HealthClient(calls, {"state": "loaded", "singleResident": True, "activeModel": "candidate-base", "engineEpoch": "7"}, **kwargs),
    )


@pytest.mark.asyncio
async def test_runs_only_selected_wav_and_does_not_persist_remote_secret(diagnostic_module, monkeypatch, tmp_path, capsys):
    module = diagnostic_module
    calls = []
    _install_healthy_candidate(monkeypatch, module, calls)
    selected = _args(tmp_path)
    loaded = []
    monkeypatch.setenv(selected.api_key_env, "not-for-reporting")
    monkeypatch.setattr(module, "load_pcm16_mono_16k", lambda path: loaded.append(path) or b"\0\0" * 8)

    socket = _Socket([
        {"type": "session.created"},
        {"type": "pipeline.config.updated"},
        {"type": "session.updated"},
        {"type": "response.output_audio.delta", "delta": base64.b64encode(b"\0\0" * 8).decode("ascii")},
        {"type": "response.done", "response": {"status": "completed"}},
    ])
    monkeypatch.setattr(module.websockets, "connect", lambda *_args, **_kwargs: _SocketContext(socket))

    async def stream_prompt(_socket, clip):
        assert clip == b"\0\0" * 8

    monkeypatch.setattr(module, "stream_prompt", stream_prompt)
    await module.run(selected)

    assert loaded == [selected.clips[0]]
    assert calls == [("GET", "http://127.0.0.1:8890/health")]
    pipeline_update = socket.sent[0]["config"]
    assert pipeline_update["model_endpoint"] == {
        "provider": "remote",
        "base_url": selected.gemma,
        "model": selected.model,
        "api_key": "not-for-reporting",
    }
    output = capsys.readouterr().out
    assert "not-for-reporting" not in output
    report = json.loads(output)
    assert report["model"] == "candidate-base"
    assert report["turns"][0]["bytes"] == 16


@pytest.mark.asyncio
async def test_audio_export_is_opt_in(diagnostic_module, monkeypatch, tmp_path):
    module = diagnostic_module
    calls = []
    _install_healthy_candidate(monkeypatch, module, calls)
    args = _args(tmp_path)
    monkeypatch.setattr(module, "load_pcm16_mono_16k", lambda _path: b"\0\0" * 4)
    socket = _Socket([
        {"type": "session.created"}, {"type": "pipeline.config.updated"}, {"type": "session.updated"},
        {"type": "response.output_audio.delta", "delta": base64.b64encode(b"\0\0" * 4).decode("ascii")},
        {"type": "response.done", "response": {"status": "completed"}},
    ])
    monkeypatch.setattr(module.websockets, "connect", lambda *_args, **_kwargs: _SocketContext(socket))
    monkeypatch.setattr(module, "stream_prompt", lambda *_args: asyncio.sleep(0))
    monkeypatch.setattr(module, "write_wav", lambda *_args: pytest.fail("audio export must stay opt-in"))

    await module.run(args)


@pytest.mark.asyncio
async def test_rejects_empty_explicit_audio_before_websocket(diagnostic_module, monkeypatch, tmp_path):
    module = diagnostic_module
    calls = []
    _install_healthy_candidate(monkeypatch, module, calls)
    monkeypatch.setattr(module, "load_pcm16_mono_16k", lambda _path: b"")
    monkeypatch.setattr(module.websockets, "connect", lambda *_args, **_kwargs: pytest.fail("empty audio must not connect"))

    with pytest.raises(RuntimeError, match="Every explicit test clip"):
        await module.run(_args(tmp_path))
    assert calls == [("GET", "http://127.0.0.1:8890/health")]


@pytest.mark.asyncio
async def test_rejects_unhealthy_candidate_without_model_mutation(diagnostic_module, monkeypatch, tmp_path):
    module = diagnostic_module
    calls = []
    monkeypatch.setattr(
        module.httpx,
        "Client",
        lambda **kwargs: _HealthClient(calls, {"state": "loading", "singleResident": False}, **kwargs),
    )
    monkeypatch.setattr(module, "load_pcm16_mono_16k", lambda _path: pytest.fail("unhealthy candidate must not read clips"))

    with pytest.raises(RuntimeError, match="already-resident"):
        await module.run(_args(tmp_path))
    assert calls == [("GET", "http://127.0.0.1:8890/health")]


@pytest.mark.asyncio
async def test_fails_when_completed_turn_has_no_audio(diagnostic_module, monkeypatch, tmp_path):
    module = diagnostic_module
    calls = []
    _install_healthy_candidate(monkeypatch, module, calls)
    monkeypatch.setattr(module, "load_pcm16_mono_16k", lambda _path: b"\0\0" * 4)
    socket = _Socket([
        {"type": "session.created"}, {"type": "pipeline.config.updated"}, {"type": "session.updated"},
        {"type": "response.done", "response": {"status": "completed"}},
    ])
    monkeypatch.setattr(module.websockets, "connect", lambda *_args, **_kwargs: _SocketContext(socket))
    monkeypatch.setattr(module, "stream_prompt", lambda *_args: asyncio.sleep(0))

    with pytest.raises(RuntimeError, match="complete with audio"):
        await module.run(_args(tmp_path))


def test_classifies_only_allowlisted_direct_audio_error_text(diagnostic_module):
    assert diagnostic_module._classify_realtime_error({
        "error": {"type": "response_failed", "message": "Direct audio model response timed out."}
    }) == "model_timeout"
    assert diagnostic_module._classify_realtime_error({
        "error": {"type": "response_failed", "message": "provider detail must not be reported"}
    }) == "unspecified"
    assert diagnostic_module._classify_realtime_error({"error": {"code": None}}) == "unspecified"


@pytest.mark.asyncio
async def test_reports_timeout_category_without_provider_message(diagnostic_module, monkeypatch, tmp_path):
    module = diagnostic_module
    calls = []
    _install_healthy_candidate(monkeypatch, module, calls)
    monkeypatch.setattr(module, "load_pcm16_mono_16k", lambda _path: b"\0\0" * 4)
    socket = _Socket([
        {"type": "session.created"}, {"type": "pipeline.config.updated"}, {"type": "session.updated"},
        {"type": "error", "error": {"type": "response_failed", "message": "Direct audio model response timed out."}},
    ])
    monkeypatch.setattr(module.websockets, "connect", lambda *_args, **_kwargs: _SocketContext(socket))
    monkeypatch.setattr(module, "stream_prompt", lambda *_args: asyncio.sleep(0))

    with pytest.raises(RuntimeError, match="Realtime diagnostic failed: model_timeout") as failure:
        await module.run(_args(tmp_path))
    assert "Direct audio model response timed out." not in str(failure.value)
