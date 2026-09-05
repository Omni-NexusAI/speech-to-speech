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

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "probe_user_memory_fallback.py"
_SPEC = importlib.util.spec_from_file_location("probe_user_memory_fallback", _SCRIPT)
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


def _is_primary(body: dict) -> bool:
    return any(
        part.get("type") == "input_audio"
        for message in body["messages"]
        if isinstance(message.get("content"), list)
        for part in message["content"]
    )


def _primary_stimulus(body: dict) -> str:
    audio = next(
        part["input_audio"]["data"]
        for message in body["messages"]
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if part.get("type") == "input_audio"
    )
    if audio == _ENCODED_STIMULI["tone"]:
        return "tone"
    if audio == _ENCODED_STIMULI["noise"]:
        return "noise"
    raise AssertionError("unexpected synthetic audio")


def _persisted_label(body: dict) -> str:
    text = body["messages"][1]["content"]
    if "ALPHA" in text:
        return "ALPHA"
    if "BETA" in text:
        return "BETA"
    raise AssertionError("missing persisted synthetic label")


def _primary_response(label: str) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "content": (
                        f"USER_MEMORY: The user selected {label}.\n"
                        "ASSISTANT_RESPONSE: ACKNOWLEDGED"
                    )
                }
            }
        ]
    }


def test_probe_audio_is_deterministic_non_silent_equal_rms_pcm16_wav():
    tone_rate, tone_samples = _wav_samples(_ENCODED_STIMULI["tone"])
    noise_rate, noise_samples = _wav_samples(_ENCODED_STIMULI["noise"])

    assert tone_rate == noise_rate == 16_000
    assert len(tone_samples) == len(noise_samples) == 12_000
    assert tone_samples != noise_samples
    assert any(tone_samples)
    assert any(noise_samples)
    assert _ENCODED_STIMULI["noise"] == probe.generated_wav_base64("noise")
    tone_rms = math.sqrt(sum(value * value for value in tone_samples) / len(tone_samples))
    noise_rms = math.sqrt(sum(value * value for value in noise_samples) / len(noise_samples))
    assert abs(tone_rms - noise_rms) < 1.0


def test_payloads_preserve_exact_two_request_boundary_with_allowlisted_history():
    primary_payload = probe.build_primary_payload("gemma-test", "tone")
    primary = probe.parse_primary_response(_primary_response("ALPHA"))
    assert primary is not None
    followup_payload = probe.build_followup_payload("gemma-test", primary)

    assert [message["role"] for message in primary_payload["messages"]] == ["system", "user"]
    assert [part["type"] for part in primary_payload["messages"][1]["content"]] == ["input_audio", "text"]
    assert [message["role"] for message in followup_payload["messages"]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert followup_payload["messages"][1]["content"] == "The user selected ALPHA."
    assert followup_payload["messages"][2]["content"] == "ACKNOWLEDGED"
    assert all(not isinstance(message.get("content"), list) for message in followup_payload["messages"])
    assert "USER_MEMORY:" not in followup_payload["messages"][1]["content"]


def test_gate_calibrates_inverted_primary_mapping_and_followup_preserves_it():
    requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if _is_primary(body):
            stimulus = _primary_stimulus(body)
            requests.append((stimulus, "primary"))
            inverted_label = "BETA" if stimulus == "tone" else "ALPHA"
            return httpx.Response(200, json=_primary_response(inverted_label))
        persisted = _persisted_label(body)
        requests.append((persisted.lower(), "followup"))
        return httpx.Response(200, json={"choices": [{"message": {"content": persisted}}]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = probe.run_gate(
            client,
            endpoint="http://endpoint-secret.example/v1",
            model="model-secret-value",
            api_key="api-key-secret-value",
        )

    assert len(requests) == 16
    assert [kind for _, kind in requests] == ["primary", "followup"] * 8
    assert result == {
        "endpoint_reachable": True,
        "trial_count": 8,
        "request_count": 16,
        "primary_http_success_count": 8,
        "primary_schema_valid_count": 8,
        "primary": {"stimulus_consistent": True, "paired_flip": True},
        "followup_request_count": 8,
        "followup_http_success_count": 8,
        "followup_schema_valid_count": 8,
        "followup_preserves_count": 8,
        "followup": {"preserves_primary_mapping": True},
        "gate_passed": True,
    }
    serialized = json.dumps(result)
    for secret in (
        "endpoint-secret",
        "model-secret",
        "api-key-secret",
        "ALPHA",
        "BETA",
        "USER_MEMORY",
        "input_audio",
    ):
        assert secret not in serialized


def test_gate_rejects_fixed_primary_memory():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=(
                _primary_response("ALPHA")
                if _is_primary(json.loads(request.content))
                else {"choices": [{"message": {"content": "ALPHA"}}]}
            ),
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = probe.run_gate(client, endpoint="http://model.example/v1", model="gemma-test")

    assert result["primary"] == {"stimulus_consistent": True, "paired_flip": False}
    assert result["followup_preserves_count"] == 8
    assert result["gate_passed"] is False


def test_gate_rejects_fixed_followup_output():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if _is_primary(body):
            label = "ALPHA" if _primary_stimulus(body) == "tone" else "BETA"
            return httpx.Response(200, json=_primary_response(label))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ALPHA"}}]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = probe.run_gate(client, endpoint="http://model.example/v1", model="gemma-test")

    assert result["primary"]["paired_flip"] is True
    assert result["followup_preserves_count"] == 4
    assert result["followup"]["preserves_primary_mapping"] is False
    assert result["gate_passed"] is False


@pytest.mark.parametrize(
    "content",
    [
        "USER_TRANSCRIPT: secret\nUSER_MEMORY: The user selected ALPHA.\nASSISTANT_RESPONSE: ACKNOWLEDGED",
        "USER_MEMORY: The user selected ALPHA.\nASSISTANT_RESPONSE: ACKNOWLEDGED (TRANSCRIPT: secret)",
        "USER_MEMORY: The user selected ALPHA.\nUSER_MEMORY: secret\nASSISTANT_RESPONSE: ACKNOWLEDGED",
        "USER_MEMORY: The user selected\nALPHA.\nASSISTANT_RESPONSE: ACKNOWLEDGED",
        "USER_MEMORY: The user selected ALPHA.\nASSISTANT_RESPONSE: ACKNOWLEDGED\nRESPONSE: secret",
        "USER_MEMORY: The user selected ALPHA.\nASSISTANT_RESPONSE: arbitrary prose",
    ],
)
def test_primary_parser_rejects_transcript_malformed_markers_and_arbitrary_prose(content):
    assert probe.parse_primary_response({"choices": [{"message": {"content": content}}]}) is None


def test_malformed_primary_never_starts_followup_or_leaks_response():
    response_secret = "malformed-primary-response-secret"
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": f"USER_MEMORY: {response_secret}\nASSISTANT_RESPONSE: ACKNOWLEDGED"
                        }
                    }
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = probe.run_gate(client, endpoint="http://model.example/v1", model="gemma-test")

    assert calls == 8
    assert result["request_count"] == 8
    assert result["primary_schema_valid_count"] == 0
    assert result["followup_request_count"] == 0
    assert result["gate_passed"] is False
    assert response_secret not in json.dumps(result)


def test_primary_and_followup_http_failures_are_content_free():
    primary_secret = "primary-http-body-secret"
    with httpx.Client(
        transport=httpx.MockTransport(lambda _request: httpx.Response(503, text=primary_secret))
    ) as client:
        primary_result = probe.run_gate(client, endpoint="http://model.example/v1", model="gemma-test")
    assert primary_result["primary_http_success_count"] == 0
    assert primary_result["followup_request_count"] == 0
    assert primary_secret not in json.dumps(primary_result)

    followup_secret = "followup-http-body-secret"

    def followup_failure(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if _is_primary(body):
            label = "ALPHA" if _primary_stimulus(body) == "tone" else "BETA"
            return httpx.Response(200, json=_primary_response(label))
        return httpx.Response(503, text=followup_secret)

    with httpx.Client(transport=httpx.MockTransport(followup_failure)) as client:
        followup_result = probe.run_gate(client, endpoint="http://model.example/v1", model="gemma-test")
    assert followup_result["followup_request_count"] == 8
    assert followup_result["followup_http_success_count"] == 0
    assert followup_result["gate_passed"] is False
    assert followup_secret not in json.dumps(followup_result)


def test_main_reduces_transport_failure_to_error_class(monkeypatch, capsys):
    transport_secret = "transport-secret-sentinel"

    class FailingClient:
        def __enter__(self):
            raise httpx.ConnectError(transport_secret)

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
    assert transport_secret not in output
    assert "endpoint-secret" not in output
    assert "model-secret" not in output
