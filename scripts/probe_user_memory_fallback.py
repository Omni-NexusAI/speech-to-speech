"""Content-free live gate for same-primary ``USER_MEMORY`` continuity.

Each synthetic trial makes exactly one current-audio primary request and, only
after allowlisted memory plus a fixed acknowledgement is returned, one
text-only contextual follow-up. Audio, prompts, memory, responses, endpoints,
model names, and API keys are never written to disk or stdout.
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

_STIMULUS_SEQUENCE: tuple[Stimulus, ...] = ("tone", "noise", "noise", "tone") * 2
_TARGET_RMS = 0.14
_NOISE_SEED = 0x4D454D
_PRIMARY_RESPONSE_RE = re.compile(
    r"\A *USER_MEMORY *: *The user selected +(ALPHA|BETA)\. *\r?\n"
    r"ASSISTANT_RESPONSE *: *ACKNOWLEDGED *\Z",
    re.IGNORECASE,
)
_FOLLOWUP_LABEL_RE = re.compile(r"\A\s*(ALPHA|BETA)\s*[.!]?\s*\Z", re.IGNORECASE)
_TRANSCRIPT_MARKER_RE = re.compile(r"(?i)(?:USER_TRANSCRIPT|USER_SPEECH|TRANSCRIPT|USER)\s*:")


@dataclass(frozen=True)
class PrimaryResult:
    """Validated allowlisted primary output used to construct one follow-up."""

    label: str
    memory_text: str
    assistant_text: str = "ACKNOWLEDGED"


@dataclass(frozen=True)
class TrialObservation:
    """Private result of one primary plus its optional contextual follow-up."""

    stimulus: Stimulus
    primary_reachable: bool
    primary_http_success: bool
    primary_schema_valid: bool
    primary_label: str | None
    followup_attempted: bool
    followup_http_success: bool
    followup_schema_valid: bool
    followup_label: str | None


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
        frames.extend(struct.pack("<h", max(-32767, min(32767, round(value * scale * 32767.0)))))
    output = BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(bytes(frames))
    return base64.b64encode(output.getvalue()).decode("ascii")


def build_primary_payload(model: str, stimulus: Stimulus) -> dict[str, Any]:
    """Build the one permitted current-audio primary request."""

    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "This is a deterministic synthetic-audio memory check. Inspect only the current user audio. "
                    "Use ALPHA for a steady periodic signal and BETA for broadband noise. Omit USER_TRANSCRIPT. "
                    "Return exactly: USER_MEMORY: The user selected <ALPHA or BETA>. on the first line and "
                    "ASSISTANT_RESPONSE: ACKNOWLEDGED on the second line. Add nothing else."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": generated_wav_base64(stimulus), "format": "wav"},
                    },
                    {"type": "text", "text": "Store this synthetic selection and acknowledge it."},
                ],
            },
        ],
        "stream": False,
        "temperature": 0,
        "max_tokens": 48,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def build_followup_payload(model: str, primary: PrimaryResult) -> dict[str, Any]:
    """Build one text-only follow-up from strictly allowlisted history."""

    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Use the retained conversation. Reply with exactly ALPHA or BETA and do not explain.",
            },
            {"role": "user", "content": primary.memory_text},
            {"role": "assistant", "content": primary.assistant_text},
            {"role": "user", "content": "Which synthetic selection did I make?"},
        ],
        "stream": False,
        "temperature": 0,
        "max_tokens": 8,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _response_content(payload: Any) -> str | None:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    return content if isinstance(content, str) else None


def parse_primary_response(payload: Any) -> PrimaryResult | None:
    """Accept only allowlisted memory, a fixed acknowledgement, and no transcript."""

    content = _response_content(payload)
    if content is None or _TRANSCRIPT_MARKER_RE.search(content):
        return None
    match = _PRIMARY_RESPONSE_RE.fullmatch(content)
    if not match:
        return None
    label = match.group(1).lower()
    return PrimaryResult(label=label, memory_text=f"The user selected {label.upper()}.")


def parse_followup_label(payload: Any) -> str | None:
    """Return a recognized label without exposing response content."""

    content = _response_content(payload)
    if content is None:
        return None
    match = _FOLLOWUP_LABEL_RE.fullmatch(content)
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


def run_trial(
    client: httpx.Client,
    *,
    endpoint: str,
    model: str,
    stimulus: Stimulus,
    api_key: str | None = None,
) -> TrialObservation:
    """Run one primary request and at most one validated contextual follow-up."""

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    primary_response = client.post(
        f"{endpoint.rstrip('/')}/chat/completions",
        headers=headers,
        json=build_primary_payload(model, stimulus),
    )
    if not primary_response.is_success:
        return TrialObservation(stimulus, True, False, False, None, False, False, False, None)
    try:
        primary = parse_primary_response(primary_response.json())
    except (json.JSONDecodeError, ValueError):
        primary = None
    if primary is None:
        return TrialObservation(stimulus, True, True, False, None, False, False, False, None)
    followup_response = client.post(
        f"{endpoint.rstrip('/')}/chat/completions",
        headers=headers,
        json=build_followup_payload(model, primary),
    )
    followup_label: str | None = None
    if followup_response.is_success:
        try:
            followup_label = parse_followup_label(followup_response.json())
        except (json.JSONDecodeError, ValueError):
            followup_label = None
    return TrialObservation(
        stimulus,
        True,
        True,
        True,
        primary.label,
        True,
        followup_response.is_success,
        followup_label is not None,
        followup_label,
    )


def _primary_mapping(observations: list[TrialObservation]) -> tuple[dict[Stimulus, str], dict[str, bool]]:
    grouped: dict[Stimulus, list[str | None]] = {"tone": [], "noise": []}
    for observation in observations:
        grouped[observation.stimulus].append(observation.primary_label)
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


def summarize_observations(observations: list[TrialObservation]) -> dict[str, Any]:
    """Reduce private memory and labels to bounded capability evidence."""

    expected_trials = len(_STIMULUS_SEQUENCE)
    _mapping, primary_status = _primary_mapping(observations)
    primary_http_success_count = sum(item.primary_http_success for item in observations)
    primary_schema_valid_count = sum(item.primary_schema_valid for item in observations)
    followup_request_count = sum(item.followup_attempted for item in observations)
    followup_http_success_count = sum(item.followup_http_success for item in observations)
    followup_schema_valid_count = sum(item.followup_schema_valid for item in observations)
    followup_preserves_count = sum(
        item.primary_label is not None and item.followup_label == item.primary_label for item in observations
    )
    endpoint_reachable = len(observations) == expected_trials and all(item.primary_reachable for item in observations)
    preserves_primary_mapping = followup_preserves_count == expected_trials
    gate_passed = (
        endpoint_reachable
        and primary_http_success_count == expected_trials
        and primary_schema_valid_count == expected_trials
        and primary_status["paired_flip"]
        and followup_request_count == expected_trials
        and followup_http_success_count == expected_trials
        and followup_schema_valid_count == expected_trials
        and preserves_primary_mapping
    )
    return {
        "endpoint_reachable": endpoint_reachable,
        "trial_count": len(observations),
        "request_count": len(observations) + followup_request_count,
        "primary_http_success_count": primary_http_success_count,
        "primary_schema_valid_count": primary_schema_valid_count,
        "primary": primary_status,
        "followup_request_count": followup_request_count,
        "followup_http_success_count": followup_http_success_count,
        "followup_schema_valid_count": followup_schema_valid_count,
        "followup_preserves_count": followup_preserves_count,
        "followup": {"preserves_primary_mapping": preserves_primary_mapping},
        "gate_passed": gate_passed,
    }


def run_gate(
    client: httpx.Client,
    *,
    endpoint: str,
    model: str,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Run two ABBA cycles, with exactly one primary and one valid follow-up each."""

    observations = [
        run_trial(client, endpoint=endpoint, model=model, stimulus=stimulus, api_key=api_key)
        for stimulus in _STIMULUS_SEQUENCE
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
                {"endpoint_reachable": False, "error_class": type(exc).__name__, "gate_passed": False},
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0 if result["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
