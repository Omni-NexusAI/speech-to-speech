"""Content-free live gate for historical Chat Completions ``input_audio``.

The probe compares equal-RMS synthetic tone and noise WAVs in both the current
and historical user position. Current-audio responses calibrate the endpoint's
two-label mapping; historical responses must then follow that same mapping.
Audio, prompts, response text, endpoints, model names, and API keys are never
written to disk or stdout.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import random
import re
import struct
import wave
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Literal

import httpx

Stimulus = Literal["tone", "noise"]
Placement = Literal["current", "historical"]

_LABEL_RE = re.compile(r"^\s*(TONE|NOISE)\s*[.!]?\s*$", re.IGNORECASE)
_STIMULUS_SEQUENCE: tuple[Stimulus, ...] = ("tone", "noise", "noise", "tone") * 2
_PLACEMENTS: tuple[Placement, ...] = ("current", "historical")
_TARGET_RMS = 0.14
_NOISE_SEED = 0x51A7E


@dataclass(frozen=True)
class ProbeObservation:
    """One private observation; ``label`` must never enter public evidence."""

    placement: Placement
    stimulus: Stimulus
    endpoint_reachable: bool
    http_success: bool
    schema_valid: bool
    label: str | None


def generated_wav_base64(
    stimulus: Stimulus,
    *,
    sample_rate: int = 16_000,
    duration_s: float = 0.75,
) -> str:
    """Return an in-memory equal-RMS mono PCM16 tone or seeded-noise WAV."""

    if stimulus not in {"tone", "noise"}:
        raise ValueError("stimulus must be tone or noise")
    frame_count = int(sample_rate * duration_s)
    if frame_count < 2:
        raise ValueError("duration must produce at least two audio frames")

    if stimulus == "tone":
        samples = [math.sin(2.0 * math.pi * 440.0 * index / sample_rate) for index in range(frame_count)]
    else:
        generator = random.Random(_NOISE_SEED)
        samples = [generator.uniform(-1.0, 1.0) for _ in range(frame_count)]

    fade_frames = min(max(1, int(sample_rate * 0.01)), frame_count // 2)
    for index in range(frame_count):
        envelope = min(1.0, index / fade_frames, (frame_count - 1 - index) / fade_frames)
        samples[index] *= max(0.0, envelope)
    source_rms = math.sqrt(sum(value * value for value in samples) / frame_count)
    scale = _TARGET_RMS / source_rms

    frames = bytearray()
    for value in samples:
        pcm_value = max(-32767, min(32767, round(value * scale * 32767.0)))
        frames.extend(struct.pack("<h", pcm_value))

    output = BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(bytes(frames))
    return base64.b64encode(output.getvalue()).decode("ascii")


def build_payload(model: str, stimulus: Stimulus, placement: Placement) -> dict[str, Any]:
    """Build current or historical audio using the production serializer shape."""

    if placement not in _PLACEMENTS:
        raise ValueError("placement must be current or historical")
    audio_part = {
        "type": "input_audio",
        "input_audio": {"data": generated_wav_base64(stimulus), "format": "wav"},
    }
    system_message = {
        "role": "system",
        "content": (
            "This is a deterministic synthetic-audio capability check. Inspect the synthetic audio supplied in the "
            "conversation and reply with exactly TONE for a steady periodic signal or NOISE for broadband noise. "
            "A later CLASSIFY message refers to the immediately preceding historical audio. Do not explain."
        ),
    }
    if placement == "current":
        messages = [
            system_message,
            {"role": "user", "content": [audio_part]},
        ]
    else:
        messages = [
            system_message,
            {"role": "user", "content": [audio_part]},
            {"role": "assistant", "content": "READY"},
            {"role": "user", "content": "CLASSIFY"},
        ]
    return {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": 0,
        "max_tokens": 8,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def response_label(payload: Any) -> str | None:
    """Return a recognized synthetic label without exposing response content."""

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
    stimulus: Stimulus,
    placement: Placement,
    api_key: str | None = None,
) -> ProbeObservation:
    """Run one private semantic observation for later content-free reduction."""

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    response = client.post(
        f"{endpoint.rstrip('/')}/chat/completions",
        headers=headers,
        json=build_payload(model, stimulus, placement),
    )
    label: str | None = None
    if response.is_success:
        try:
            label = response_label(response.json())
        except (json.JSONDecodeError, ValueError):
            label = None
    return ProbeObservation(
        placement=placement,
        stimulus=stimulus,
        endpoint_reachable=True,
        http_success=response.is_success,
        schema_valid=label is not None,
        label=label,
    )


def _placement_mapping(
    observations: list[ProbeObservation], placement: Placement
) -> tuple[dict[Stimulus, str], dict[str, bool]]:
    grouped: dict[Stimulus, list[str | None]] = {"tone": [], "noise": []}
    for observation in observations:
        if observation.placement == placement:
            grouped[observation.stimulus].append(observation.label)

    expected_per_stimulus = _STIMULUS_SEQUENCE.count("tone")
    consistent = all(
        len(labels) == expected_per_stimulus and None not in labels and len(set(labels)) == 1
        for labels in grouped.values()
    )
    mapping: dict[Stimulus, str] = {}
    if consistent:
        mapping = {stimulus: str(labels[0]) for stimulus, labels in grouped.items()}
    paired_flip = consistent and mapping["tone"] != mapping["noise"]
    return mapping, {"stimulus_consistent": consistent, "paired_flip": paired_flip}


def summarize_observations(observations: list[ProbeObservation]) -> dict[str, Any]:
    """Reduce private labels to bounded capability evidence."""

    expected_count = len(_STIMULUS_SEQUENCE) * len(_PLACEMENTS)
    current_mapping, current_status = _placement_mapping(observations, "current")
    historical_mapping, historical_status = _placement_mapping(observations, "historical")
    matches_current_mapping = (
        current_status["stimulus_consistent"]
        and historical_status["stimulus_consistent"]
        and historical_mapping == current_mapping
    )
    endpoint_reachable = len(observations) == expected_count and all(
        observation.endpoint_reachable for observation in observations
    )
    http_success_count = sum(observation.http_success for observation in observations)
    schema_valid_count = sum(observation.schema_valid for observation in observations)
    gate_passed = (
        endpoint_reachable
        and http_success_count == expected_count
        and schema_valid_count == expected_count
        and current_status["paired_flip"]
        and historical_status["paired_flip"]
        and matches_current_mapping
    )
    return {
        "endpoint_reachable": endpoint_reachable,
        "request_count": len(observations),
        "http_success_count": http_success_count,
        "schema_valid_count": schema_valid_count,
        "current": current_status,
        "historical": {
            **historical_status,
            "matches_current_mapping": matches_current_mapping,
        },
        "gate_passed": gate_passed,
    }


def run_gate(
    client: httpx.Client,
    *,
    endpoint: str,
    model: str,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Run paired current/historical ABBA trials and return public evidence."""

    observations = [
        run_probe(
            client,
            endpoint=endpoint,
            model=model,
            stimulus=stimulus,
            placement=placement,
            api_key=api_key,
        )
        for stimulus in _STIMULUS_SEQUENCE
        for placement in _PLACEMENTS
    ]
    return summarize_observations(observations)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings-url", default="http://127.0.0.1:7862/api/ui-settings")
    parser.add_argument("--endpoint", help="Optional explicit OpenAI-compatible /v1 endpoint")
    parser.add_argument("--model", help="Optional explicit model name")
    parser.add_argument("--api-key-env", help="Environment variable containing an optional bearer key")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)

    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    try:
        with httpx.Client(timeout=args.timeout) as client:
            endpoint, model = resolve_target(
                client,
                args.settings_url,
                endpoint=args.endpoint,
                model=args.model,
            )
            result = run_gate(client, endpoint=endpoint, model=model, api_key=api_key)
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

    print(json.dumps(result, sort_keys=True))
    return 0 if result["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
