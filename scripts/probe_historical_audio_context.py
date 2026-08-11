"""Content-free live gate for historical Chat Completions ``input_audio``.

The probe creates a short synthetic WAV entirely in memory, places it in an
earlier user turn, and asks a later user turn to classify it as a tone or
silence. It never writes audio, request messages, transcripts, response text,
or API keys to disk or stdout.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import struct
import wave
from io import BytesIO
from typing import Any

import httpx

_LABEL_RE = re.compile(r"^\s*(TONE|SILENCE)\s*[.!]?\s*$", re.IGNORECASE)


def generated_wav_base64(kind: str, *, sample_rate: int = 16_000, duration_s: float = 0.75) -> str:
    """Return an in-memory mono PCM16 WAV containing a tone or silence."""

    if kind not in {"tone", "silence"}:
        raise ValueError("kind must be tone or silence")
    frame_count = int(sample_rate * duration_s)
    frames = bytearray()
    for index in range(frame_count):
        value = 0.0 if kind == "silence" else 0.2 * math.sin(2.0 * math.pi * 440.0 * index / sample_rate)
        frames.extend(struct.pack("<h", int(value * 32767.0)))
    output = BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(bytes(frames))
    return base64.b64encode(output.getvalue()).decode("ascii")


def build_payload(model: str, kind: str) -> dict[str, Any]:
    """Build the exact historical-audio shape used by the local serializer."""

    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Classify synthetic audio only. When asked about the earlier audio, reply with exactly "
                    "TONE for a steady tone or SILENCE for silence."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": generated_wav_base64(kind), "format": "wav"},
                    }
                ],
            },
            {"role": "assistant", "content": "Received."},
            {
                "role": "user",
                "content": "Classify the audio in the first user message. Reply with exactly TONE or SILENCE.",
            },
        ],
        "stream": False,
        "temperature": 0,
        "max_tokens": 8,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def response_label(payload: Any) -> str | None:
    """Return a recognized label without exposing response content."""

    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    if not isinstance(content, str):
        return None
    match = _LABEL_RE.fullmatch(content)
    return match.group(1).lower() if match else None


def resolve_target(
    client: httpx.Client,
    settings_url: str,
    *,
    endpoint: str | None,
    model: str | None,
) -> tuple[str, str]:
    """Resolve non-secret endpoint/model settings, allowing explicit overrides."""

    if endpoint and model:
        return endpoint.rstrip("/"), model
    response = client.get(settings_url)
    response.raise_for_status()
    payload = response.json()
    settings = payload.get("settings") if isinstance(payload, dict) else None
    if not isinstance(settings, dict):
        raise ValueError("settings_schema_invalid")
    resolved_endpoint = endpoint or settings.get("modelUrl")
    resolved_model = model or settings.get("modelName")
    if not isinstance(resolved_endpoint, str) or not resolved_endpoint.strip():
        raise ValueError("model_endpoint_missing")
    if not isinstance(resolved_model, str) or not resolved_model.strip():
        raise ValueError("model_name_missing")
    return resolved_endpoint.rstrip("/"), resolved_model


def run_probe(
    client: httpx.Client,
    *,
    endpoint: str,
    model: str,
    kind: str,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Run one content-free semantic history check and return bounded evidence."""

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    response = client.post(
        f"{endpoint.rstrip('/')}/chat/completions",
        headers=headers,
        json=build_payload(model, kind),
    )
    result: dict[str, Any] = {
        "kind": kind,
        "endpoint_reachable": True,
        "http_status": response.status_code,
        "schema_valid": False,
        "semantic_match": False,
    }
    if not response.is_success:
        return result
    try:
        label = response_label(response.json())
    except (json.JSONDecodeError, ValueError):
        label = None
    result["schema_valid"] = label is not None
    result["semantic_match"] = label == kind
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings-url", default="http://127.0.0.1:7862/api/ui-settings")
    parser.add_argument("--endpoint", help="Optional explicit OpenAI-compatible /v1 endpoint")
    parser.add_argument("--model", help="Optional explicit model name")
    parser.add_argument("--api-key-env", help="Environment variable containing an optional bearer key")
    parser.add_argument("--expected", choices=("tone", "silence", "both"), default="both")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)

    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    kinds = ("tone", "silence") if args.expected == "both" else (args.expected,)
    try:
        with httpx.Client(timeout=args.timeout) as client:
            endpoint, model = resolve_target(
                client,
                args.settings_url,
                endpoint=args.endpoint,
                model=args.model,
            )
            results = [
                run_probe(client, endpoint=endpoint, model=model, kind=kind, api_key=api_key) for kind in kinds
            ]
    except Exception as exc:  # noqa: BLE001 - bounded diagnostic output is intentional
        print(
            json.dumps(
                {
                    "endpoint_reachable": False,
                    "error_class": type(exc).__name__,
                    "gate_passed": False,
                },
                sort_keys=True,
            )
        )
        return 2

    gate_passed = all(result["semantic_match"] for result in results)
    print(json.dumps({"gate_passed": gate_passed, "results": results}, sort_keys=True))
    return 0 if gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
