from __future__ import annotations

import base64
import importlib.util
import json
import wave
from io import BytesIO
from pathlib import Path

import httpx

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "probe_historical_audio_context.py"
_SPEC = importlib.util.spec_from_file_location("probe_historical_audio_context", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
probe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(probe)


def test_generated_probe_audio_is_in_memory_pcm16_wav():
    raw = base64.b64decode(probe.generated_wav_base64("tone"))

    with wave.open(BytesIO(raw), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 16_000
        assert wav.getnframes() == 12_000


def test_probe_payload_uses_exact_historical_input_audio_shape():
    payload = probe.build_payload("gemma-test", "silence")

    assert [message["role"] for message in payload["messages"]] == ["system", "user", "assistant", "user"]
    historical = payload["messages"][1]["content"]
    assert historical[0]["type"] == "input_audio"
    assert historical[0]["input_audio"]["format"] == "wav"
    assert isinstance(historical[0]["input_audio"]["data"], str)
    assert payload["messages"][-1]["role"] == "user"


def test_probe_reports_only_bounded_content_free_evidence():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        audio = base64.b64decode(body["messages"][1]["content"][0]["input_audio"]["data"])
        with wave.open(BytesIO(audio), "rb") as wav:
            frames = wav.readframes(wav.getnframes())
        label = "SILENCE" if set(frames) == {0} else "TONE"
        return httpx.Response(200, json={"choices": [{"message": {"content": label}}]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = probe.run_probe(
            client,
            endpoint="http://model.example/v1",
            model="gemma-test",
            kind="tone",
            api_key="secret-value",
        )

    assert result == {
        "kind": "tone",
        "endpoint_reachable": True,
        "http_status": 200,
        "schema_valid": True,
        "semantic_match": True,
    }
    serialized = json.dumps(result)
    assert "secret-value" not in serialized
    assert "input_audio" not in serialized
    assert "TONE" not in serialized
