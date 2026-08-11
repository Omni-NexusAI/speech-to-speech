from __future__ import annotations

import base64
import importlib.util
import json
import math
import struct
import sys
import wave
from io import BytesIO
from pathlib import Path

import httpx
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "probe_historical_audio_context.py"
_SPEC = importlib.util.spec_from_file_location("probe_historical_audio_context", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
probe = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = probe
_SPEC.loader.exec_module(probe)

_ENCODED_STIMULI = {
    "tone": probe.generated_wav_base64("tone"),
    "noise": probe.generated_wav_base64("noise"),
}


def _wav_samples(encoded: str) -> tuple[int, tuple[int, ...]]:
    raw = base64.b64decode(encoded)
    with wave.open(BytesIO(raw), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    return sample_rate, struct.unpack(f"<{len(frames) // 2}h", frames)


def _stimulus_from_request(body: dict) -> str:
    audio_part = next(
        part
        for message in body["messages"]
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if part.get("type") == "input_audio"
    )
    encoded = audio_part["input_audio"]["data"]
    if encoded == _ENCODED_STIMULI["tone"]:
        return "tone"
    if encoded == _ENCODED_STIMULI["noise"]:
        return "noise"
    raise AssertionError("unexpected synthetic audio")


def _placement_from_request(body: dict) -> str:
    roles = [message["role"] for message in body["messages"]]
    return "historical" if roles == ["system", "user", "assistant", "user"] else "current"


def test_generated_probe_audio_is_deterministic_non_silent_equal_rms_pcm16_wav():
    tone = probe.generated_wav_base64("tone")
    noise = probe.generated_wav_base64("noise")
    tone_rate, tone_samples = _wav_samples(tone)
    noise_rate, noise_samples = _wav_samples(noise)

    assert tone_rate == noise_rate == 16_000
    assert len(tone_samples) == len(noise_samples) == 12_000
    assert tone_samples != noise_samples
    assert any(tone_samples)
    assert any(noise_samples)
    assert noise == probe.generated_wav_base64("noise")
    tone_rms = math.sqrt(sum(value * value for value in tone_samples) / len(tone_samples))
    noise_rms = math.sqrt(sum(value * value for value in noise_samples) / len(noise_samples))
    assert abs(tone_rms - noise_rms) < 1.0

    with pytest.raises(ValueError, match="tone or noise"):
        probe.generated_wav_base64("silence")


def test_probe_payloads_use_current_and_exact_historical_input_audio_shapes():
    current = probe.build_payload("gemma-test", "tone", "current")
    historical = probe.build_payload("gemma-test", "noise", "historical")

    assert [message["role"] for message in current["messages"]] == ["system", "user"]
    assert [part["type"] for part in current["messages"][1]["content"]] == ["input_audio", "text"]
    assert [message["role"] for message in historical["messages"]] == ["system", "user", "assistant", "user"]
    historical_audio = historical["messages"][1]["content"][0]
    assert historical_audio["type"] == "input_audio"
    assert historical_audio["input_audio"]["format"] == "wav"
    assert isinstance(historical_audio["input_audio"]["data"], str)


def test_gate_calibrates_inverted_current_mapping_and_accepts_matching_history():
    requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        stimulus = _stimulus_from_request(body)
        placement = _placement_from_request(body)
        requests.append((stimulus, placement))
        inverted_label = "NOISE" if stimulus == "tone" else "TONE"
        return httpx.Response(200, json={"choices": [{"message": {"content": inverted_label}}]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = probe.run_gate(
            client,
            endpoint="http://model.example/v1",
            model="gemma-test",
            api_key="secret-value",
        )

    expected_sequence = ("tone", "noise", "noise", "tone") * 2
    assert requests == [
        (stimulus, placement)
        for stimulus in expected_sequence
        for placement in ("current", "historical")
    ]
    assert result == {
        "endpoint_reachable": True,
        "request_count": 16,
        "http_success_count": 16,
        "schema_valid_count": 16,
        "current": {"stimulus_consistent": True, "paired_flip": True},
        "historical": {
            "stimulus_consistent": True,
            "paired_flip": True,
            "matches_current_mapping": True,
        },
        "gate_passed": True,
    }
    serialized = json.dumps(result)
    assert "secret-value" not in serialized
    assert "model.example" not in serialized
    assert "gemma-test" not in serialized
    assert "input_audio" not in serialized
    assert "TONE" not in serialized
    assert "NOISE" not in serialized


def test_gate_rejects_fixed_historical_output_even_when_current_audio_flips():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        stimulus = _stimulus_from_request(body)
        placement = _placement_from_request(body)
        label = "NOISE" if placement == "historical" or stimulus == "tone" else "TONE"
        return httpx.Response(200, json={"choices": [{"message": {"content": label}}]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = probe.run_gate(client, endpoint="http://model.example/v1", model="gemma-test")

    assert result["current"] == {"stimulus_consistent": True, "paired_flip": True}
    assert result["historical"] == {
        "stimulus_consistent": True,
        "paired_flip": False,
        "matches_current_mapping": False,
    }
    assert result["gate_passed"] is False


def test_gate_never_exposes_unrecognized_response_content():
    response_sentinel = "response-secret-sentinel"

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": response_sentinel}}]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = probe.run_gate(client, endpoint="http://model.example/v1", model="gemma-test")

    assert result["schema_valid_count"] == 0
    assert result["gate_passed"] is False
    assert response_sentinel not in json.dumps(result)


def test_gate_reduces_http_failure_body_to_content_free_counts():
    response_sentinel = "http-body-secret-sentinel"

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text=response_sentinel)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = probe.run_gate(client, endpoint="http://model.example/v1", model="gemma-test")

    assert result["request_count"] == 16
    assert result["http_success_count"] == 0
    assert result["schema_valid_count"] == 0
    assert result["gate_passed"] is False
    assert response_sentinel not in json.dumps(result)


def test_main_reduces_transport_failure_to_error_class(monkeypatch, capsys):
    transport_sentinel = "transport-secret-sentinel"

    class FailingClient:
        def __enter__(self):
            raise httpx.ConnectError(transport_sentinel)

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(probe.httpx, "Client", lambda **_kwargs: FailingClient())

    exit_code = probe.main(
        [
            "--endpoint",
            "http://endpoint-secret.example/v1",
            "--model",
            "model-secret-value",
            "--api-key-env",
            "MISSING_KEY_ENV",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 2
    assert json.loads(output) == {
        "endpoint_reachable": False,
        "error_class": "ConnectError",
        "gate_passed": False,
    }
    assert transport_sentinel not in output
    assert "endpoint-secret" not in output
    assert "model-secret" not in output
