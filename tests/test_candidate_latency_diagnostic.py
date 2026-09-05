"""Opt-in diagnostics never export aborted or unverifiable candidate audio."""
import importlib.util
from pathlib import Path

import httpx
import pytest


def load_diagnostic():
    path = Path(__file__).parents[1] / "scripts/measure_candidate_tts_paths.py"
    spec = importlib.util.spec_from_file_location("latency_diagnostic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("state", ["completed", "limited", "error", "cancelled", "producing"])
def test_export_requires_complete_matching_outcome(tmp_path, state):
    module = load_diagnostic()
    key = "b" * 32
    calls = []
    pcm = b"\x01\x00" * 480

    def transport(request):
        calls.append((request.method, str(request.url)))
        if request.method == "POST":
            return httpx.Response(200, content=pcm, headers={"X-TTS-Request-Id": key})
        return httpx.Response(200, json={"requestId": key, "state": state, "eos": state == "completed", "input": "excluded"})

    target = tmp_path / "diagnostic.pcm"
    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        if state == "completed":
            result = module.measure(client, "http://studio/api/audio-cpp/audio/speech", {}, audio_path=target, outcome_base="http://candidate")
            assert result["completion_proven"] is True
            assert result["audio_seconds"] == .02
            assert "input" not in result["outcome"]
            assert target.read_bytes() == pcm
        else:
            with pytest.raises(RuntimeError, match="complete diagnostic"):
                module.measure(client, "http://studio/api/audio-cpp/audio/speech", {}, audio_path=target, outcome_base="http://candidate")
            assert not target.exists()
    assert calls == [("POST", "http://studio/api/audio-cpp/audio/speech"), ("GET", f"http://candidate/v1/audio/outcomes/{key}")]


@pytest.mark.parametrize("pcm", [b"", b"\x01"])
def test_empty_or_partial_pcm_is_not_exported(tmp_path, pcm):
    module = load_diagnostic()
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, content=pcm))) as client:
        with pytest.raises(RuntimeError):
            module.measure(client, "http://candidate/v1/audio/speech", {}, audio_path=tmp_path / "out.pcm")
    assert not (tmp_path / "out.pcm").exists()


def test_no_implicit_audio_write_or_completion_claim():
    module = load_diagnostic()
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"\x00\x00" * 24))) as client:
        result = module.measure(client, "http://legacy/v1/audio/speech", {})
    assert result["outcome"] is None
    assert result["completion_proven"] is False


@pytest.mark.parametrize("key", [None, "", "not-an-id"])
def test_candidate_missing_or_invalid_outcome_never_exports(tmp_path, key):
    module = load_diagnostic()
    headers = {} if key is None else {"X-TTS-Request-Id": key}
    target = tmp_path / "unproven.pcm"
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"\x01\x00" * 24, headers=headers))) as client:
        with pytest.raises(RuntimeError, match="outcome identity"):
            module.measure(client, "http://candidate/v1/audio/speech", {}, audio_path=target, outcome_base="http://candidate")
    assert not target.exists()
