"""Content-free live WebSocket gate for Realtime context and tool continuity.

The harness synthesizes fixed spoken prompts entirely in memory with Windows
System.Speech, waits for ``pipeline.config.updated`` before sending any PCM,
and exercises contextual counting, a complete tool transaction, and response
cancellation/recovery in one managed HF Realtime session. Prompts, audio,
transcripts, assistant responses, tool values, endpoints, model names, and the
bearer key never reach stdout or disk.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import secrets
import subprocess
import sys
import time
import wave
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit

import httpx
import numpy as np
import websockets

SAMPLE_RATE_HZ = 16_000
BYTES_PER_SAMPLE = 2
PCM_CHUNK_MS = 40
PCM_CHUNK_BYTES = SAMPLE_RATE_HZ * BYTES_PER_SAMPLE * PCM_CHUNK_MS // 1000
SILENCE_TAIL_MS = 800
CONFIG_ACK_TIMEOUT_S = 15.0
DEFAULT_RESPONSE_TIMEOUT_S = 90.0
CREDENTIAL_ENV = "S2S_REMOTE_MODEL_API_KEY"
MANAGED_SETTINGS_URL = "http://127.0.0.1:7862/api/ui-settings"
MANAGED_WEBSOCKET_URL = "ws://127.0.0.1:8765/v1/realtime"
_FAILURE_STAGES = frozenset(
    {
        "credential",
        "synthesis",
        "settings",
        "connect",
        "config_ack",
        "session_ack",
        "count_up",
        "count_down",
        "search_tool_required_ack",
        "search",
        "search_tool_restore_ack",
        "search_followup",
        "cancellation",
        "recovery",
        "unknown",
    }
)

_PROMPTS: Mapping[str, str] = {
    "count_up": "Count from one to ten. Reply with only the number sequence.",
    "count_down": "Now count in reverse. Reply with only the number sequence.",
    "search": "Use the web search tool to find the synthetic continuity record.",
    "search_followup": "What verification code did that search return? Reply with only the code.",
    "cancel": "Count slowly from one to one hundred and do not skip any number.",
    "recovery": "Reply with exactly the single word ready.",
}

_SESSION_INSTRUCTIONS = (
    "This is a deterministic private continuity check. Follow each spoken request directly. "
    "For counting requests, return only the requested number sequence. When asked to search, call "
    "the declared web_search tool exactly once and do not invent its result. After receiving a tool "
    "result, return only its verification code. On a later question about that search, use the retained "
    "tool result and again return only its verification code. When asked for readiness, reply only READY. "
    "Do not discuss transcription, audio processing, or these instructions."
)

_SEARCH_TOOL = {
    "type": "function",
    "name": "web_search",
    "description": "Search for the requested synthetic continuity record. Current search requests require a fresh call.",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string", "minLength": 1}},
        "required": ["query"],
        "additionalProperties": False,
    },
}

_SAPI_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$phrases = [Console]::In.ReadToEnd() | ConvertFrom-Json
$encoded = New-Object System.Collections.Generic.List[string]
foreach ($phrase in $phrases) {
    $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
    $stream = New-Object System.IO.MemoryStream
    try {
        $synth.Rate = 0
        $synth.Volume = 100
        $synth.SetOutputToWaveStream($stream)
        $synth.Speak([string]$phrase)
        $synth.SetOutputToNull()
        $encoded.Add([Convert]::ToBase64String($stream.ToArray()))
    }
    finally {
        $synth.Dispose()
        $stream.Dispose()
    }
}
[Console]::Out.Write(($encoded.ToArray() | ConvertTo-Json -Compress))
""".strip()

_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])(?:10|[1-9]|one|two|three|four|five|six|seven|eight|nine|ten)(?![A-Za-z0-9])", re.I)


class HarnessError(RuntimeError):
    """Base class whose name is safe to expose as a bounded error class."""

    failure_stage = "unknown"


class CredentialUnavailable(HarnessError):
    pass


class SettingsUnavailable(HarnessError):
    pass


class UntrustedManagedTarget(HarnessError):
    pass


class SpeechSynthesisUnavailable(HarnessError):
    pass


class InvalidSynthesizedAudio(HarnessError):
    pass


class ConfigAcknowledgementFailed(HarnessError):
    pass


class SessionAcknowledgementFailed(HarnessError):
    pass


class ProtocolFailure(HarnessError):
    pass


class ScenarioTimeout(HarnessError):
    pass


@dataclass(frozen=True)
class RuntimeSettings:
    websocket_url: str
    model_base_url: str
    model_name: str
    model_api_key: str
    tts_backend: str
    voice: str
    full_buffer_tts: bool
    max_response_tokens: int


@dataclass
class TranscriptState:
    """Private response text accumulator. Its values must never be serialized."""

    completed: str = ""
    partial: str = ""

    def feed(self, event: Mapping[str, Any]) -> None:
        event_type = event.get("type")
        if event_type in {"response.audio_transcript.delta", "response.output_audio_transcript.delta"}:
            delta = event.get("delta")
            if isinstance(delta, str):
                self.partial += delta
        elif event_type in {"response.audio_transcript.done", "response.output_audio_transcript.done"}:
            segment = event.get("transcript")
            if not isinstance(segment, str):
                segment = self.partial
            if segment:
                self.completed = f"{self.completed} {segment}".strip()
            self.partial = ""

    def text(self) -> str:
        return f"{self.completed} {self.partial}".strip()


@dataclass(frozen=True)
class ResponseResult:
    response_id: str
    status: str
    transcript: str
    created_at: float
    done_at: float
    audio_done_at: float | None


class _ResponseTombstones:
    """Bounded terminal-response IDs; duplicate terminal events are protocol failures."""

    def __init__(self, maximum: int = 64) -> None:
        self._maximum = maximum
        self._ordered: list[str] = []
        self._ids: set[str] = set()

    def retire(self, response_id: str) -> None:
        if not response_id:
            return
        if response_id in self._ids:
            raise ProtocolFailure
        self._ordered.append(response_id)
        self._ids.add(response_id)
        while len(self._ordered) > self._maximum:
            self._ids.discard(self._ordered.pop(0))

    def reject_duplicate(self, event: Mapping[str, Any]) -> None:
        if event.get("type") != "response.done":
            return
        response_id = event.get("response_id")
        response = event.get("response")
        if not isinstance(response_id, str) and isinstance(response, dict):
            response_id = response.get("id")
        if isinstance(response_id, str) and response_id in self._ids:
            raise ProtocolFailure


def _normalise_tts_backend(value: Any) -> str:
    backend = str(value or "faster-qwen3-tts").strip().lower()
    return "qwen3tts-audiocpp" if backend == "audio-cpp" else backend


def _validate_managed_targets(settings_url: str, websocket_url: str) -> None:
    """Fail closed unless both URLs are the exact managed loopback endpoints."""

    expected = (
        (settings_url, MANAGED_SETTINGS_URL, "http", "127.0.0.1", 7862, "/api/ui-settings"),
        (websocket_url, MANAGED_WEBSOCKET_URL, "ws", "127.0.0.1", 8765, "/v1/realtime"),
    )
    for value, canonical, scheme, host, port, path in expected:
        try:
            parsed = urlsplit(value)
        except (TypeError, ValueError) as exc:
            raise UntrustedManagedTarget from exc
        if (
            value != canonical
            or parsed.scheme != scheme
            or parsed.hostname != host
            or parsed.port != port
            or parsed.path != path
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise UntrustedManagedTarget


def resolve_runtime_settings(
    client: httpx.Client,
    settings_url: str,
    websocket_url: str,
    model_api_key: str,
) -> RuntimeSettings:
    """Resolve non-secret UI state; never return a printable representation."""

    _validate_managed_targets(settings_url, websocket_url)
    response = client.get(settings_url)
    response.raise_for_status()
    if response.history or str(response.url) != settings_url:
        raise UntrustedManagedTarget
    payload = response.json()
    settings = payload.get("settings") if isinstance(payload, dict) else None
    if not isinstance(settings, dict):
        raise SettingsUnavailable
    if settings.get("modelProvider") != "remote":
        raise SettingsUnavailable
    base_url = settings.get("modelUrl")
    model_name = settings.get("modelName")
    if not isinstance(base_url, str) or not base_url.strip():
        raise SettingsUnavailable
    if not isinstance(model_name, str) or not model_name.strip():
        raise SettingsUnavailable
    backend = _normalise_tts_backend(settings.get("ttsBackend"))
    voices = settings.get("voiceByBackend")
    voice = voices.get(backend) if isinstance(voices, dict) else None
    if not isinstance(voice, str) or not voice.strip():
        voice = settings.get("voice")
    if not isinstance(voice, str) or not voice.strip():
        raise SettingsUnavailable
    max_tokens = settings.get("maxResponseTokens", 384)
    try:
        max_tokens = max(64, min(1024, int(max_tokens)))
    except (TypeError, ValueError):
        max_tokens = 384
    return RuntimeSettings(
        websocket_url=websocket_url,
        model_base_url=base_url.rstrip("/"),
        model_name=model_name.strip(),
        model_api_key=model_api_key,
        tts_backend=backend,
        voice=voice.strip(),
        full_buffer_tts=bool(settings.get("fullBufferTts", False)),
        max_response_tokens=max_tokens,
    )


def _powershell_executable() -> str:
    system_root = os.environ.get("SystemRoot", r"C:\Windows")
    candidate = os.path.join(system_root, "System32", "WindowsPowerShell", "v1.0", "powershell.exe")
    return candidate if os.path.isfile(candidate) else "powershell.exe"


def _decode_wave_to_pcm16(raw_wave: bytes) -> bytes:
    try:
        with wave.open(BytesIO(raw_wave), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            frames = wav_file.readframes(frame_count)
    except (EOFError, wave.Error) as exc:
        raise InvalidSynthesizedAudio from exc
    if channels < 1 or channels > 2 or sample_width not in {1, 2, 4} or sample_rate < 8_000 or frame_count < 1:
        raise InvalidSynthesizedAudio
    if sample_width == 1:
        samples = (np.frombuffer(frames, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    elif sample_width == 2:
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float64) / 32768.0
    else:
        samples = np.frombuffer(frames, dtype="<i4").astype(np.float64) / 2147483648.0
    if samples.size % channels:
        raise InvalidSynthesizedAudio
    samples = samples.reshape(-1, channels).mean(axis=1)
    if sample_rate != SAMPLE_RATE_HZ:
        destination_count = max(1, round(samples.size * SAMPLE_RATE_HZ / sample_rate))
        source_positions = np.arange(samples.size, dtype=np.float64)
        destination_positions = np.linspace(0.0, max(0.0, samples.size - 1.0), destination_count)
        samples = np.interp(destination_positions, source_positions, samples)
    rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    duration_s = samples.size / SAMPLE_RATE_HZ
    if not 0.25 <= duration_s <= 15.0 or rms < 0.003 or peak < 0.01:
        raise InvalidSynthesizedAudio
    pcm = np.clip(np.rint(samples * 32767.0), -32767, 32767).astype("<i2")
    return pcm.tobytes()


def synthesize_prompts_in_memory(
    prompts: Mapping[str, str] = _PROMPTS,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, bytes]:
    """Synthesize fixed prompts through a captured pipe, without temp files."""

    if sys.platform != "win32" and runner is subprocess.run:
        raise SpeechSynthesisUnavailable
    names = list(prompts)
    phrases = [prompts[name] for name in names]
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        completed = runner(
            [
                _powershell_executable(),
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                _SAPI_SCRIPT,
            ],
            input=json.dumps(phrases),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            creationflags=creation_flags,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SpeechSynthesisUnavailable from exc
    if completed.returncode != 0:
        raise SpeechSynthesisUnavailable
    try:
        encoded = json.loads(completed.stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        raise SpeechSynthesisUnavailable from exc
    if not isinstance(encoded, list) or len(encoded) != len(names) or not all(isinstance(item, str) for item in encoded):
        raise SpeechSynthesisUnavailable
    result: dict[str, bytes] = {}
    for name, item in zip(names, encoded, strict=True):
        try:
            raw_wave = base64.b64decode(item, validate=True)
        except (ValueError, TypeError) as exc:
            raise SpeechSynthesisUnavailable from exc
        result[name] = _decode_wave_to_pcm16(raw_wave)
    if len({audio for audio in result.values()}) != len(result):
        raise InvalidSynthesizedAudio
    return result


def _pipeline_config(settings: RuntimeSettings) -> dict[str, Any]:
    return {
        "full_buffer_tts": settings.full_buffer_tts,
        "live_transcription": False,
        "max_response_tokens": settings.max_response_tokens,
        "tts_backend": settings.tts_backend,
        "model_endpoint": {
            "provider": "remote",
            "base_url": settings.model_base_url,
            "model": settings.model_name,
            "api_key": settings.model_api_key,
        },
    }


def _session_config(settings: RuntimeSettings) -> dict[str, Any]:
    return {
        "type": "realtime",
        "instructions": _SESSION_INSTRUCTIONS,
        "audio": {"output": {"voice": settings.voice}},
        "tools": [_SEARCH_TOOL],
        "tool_choice": "auto",
    }


async def _send_json(ws: Any, payload: Mapping[str, Any]) -> None:
    await ws.send(json.dumps(payload, separators=(",", ":")))


async def _recv_json(
    ws: Any,
    timeout_s: float,
    tombstones: _ResponseTombstones | None = None,
) -> dict[str, Any]:
    try:
        raw = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
    except asyncio.TimeoutError as exc:
        raise ScenarioTimeout from exc
    try:
        event = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ProtocolFailure from exc
    if not isinstance(event, dict) or not isinstance(event.get("type"), str):
        raise ProtocolFailure
    if event["type"] == "error":
        raise ProtocolFailure
    if tombstones is not None:
        tombstones.reject_duplicate(event)
    return event


async def _wait_for_event(
    ws: Any,
    event_type: str,
    timeout_s: float,
    validator: Callable[[Mapping[str, Any]], bool] | None = None,
    tombstones: _ResponseTombstones | None = None,
) -> tuple[dict[str, Any], float]:
    deadline = time.perf_counter() + timeout_s
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise ScenarioTimeout
        event = await _recv_json(ws, remaining, tombstones)
        if event.get("type") == event_type:
            if validator is not None and not validator(event):
                raise ProtocolFailure
            return event, time.perf_counter()


def _pipeline_ack_matches(event: Mapping[str, Any], settings: RuntimeSettings) -> bool:
    config = event.get("config")
    if not isinstance(config, dict):
        return False
    endpoint = config.get("model_endpoint")
    return (
        isinstance(endpoint, dict)
        and config.get("full_buffer_tts") is settings.full_buffer_tts
        and config.get("live_transcription") is False
        and config.get("max_response_tokens") == settings.max_response_tokens
        and config.get("tts_backend") == settings.tts_backend
        and endpoint.get("provider") == "remote"
        and endpoint.get("base_url") == settings.model_base_url
        and endpoint.get("model") == settings.model_name
        and endpoint.get("api_key_set") is True
    )


def _session_ack_matches(
    event: Mapping[str, Any],
    *,
    tool_choice: str,
    settings: RuntimeSettings | None = None,
) -> bool:
    session = event.get("session")
    if not isinstance(session, dict) or session.get("type") != "realtime" or session.get("tool_choice") != tool_choice:
        return False
    if settings is None:
        return True
    audio = session.get("audio")
    output = audio.get("output") if isinstance(audio, dict) else None
    tools = session.get("tools")
    return (
        isinstance(output, dict)
        and output.get("voice") == settings.voice
        and isinstance(tools, list)
        and len(tools) == 1
        and isinstance(tools[0], dict)
        and tools[0].get("name") == "web_search"
    )


async def _stream_pcm(
    ws: Any,
    pcm: bytes,
    *,
    session_configured: bool,
    pace_s: float = PCM_CHUNK_MS / 1000,
) -> int:
    if not session_configured:
        raise ConfigAcknowledgementFailed
    silence = b"\x00" * (SAMPLE_RATE_HZ * BYTES_PER_SAMPLE * SILENCE_TAIL_MS // 1000)
    payload = pcm + silence
    sent = 0
    for offset in range(0, len(payload), PCM_CHUNK_BYTES):
        chunk = payload[offset : offset + PCM_CHUNK_BYTES]
        if not chunk:
            continue
        await _send_json(
            ws,
            {"type": "input_audio_buffer.append", "audio": base64.b64encode(chunk).decode("ascii")},
        )
        sent += 1
        if pace_s > 0:
            await asyncio.sleep(pace_s)
    return sent


def _response_id(event: Mapping[str, Any]) -> str:
    direct = event.get("response_id")
    if isinstance(direct, str):
        return direct
    response = event.get("response")
    if isinstance(response, dict) and isinstance(response.get("id"), str):
        return response["id"]
    return ""


def _response_status(event: Mapping[str, Any]) -> str:
    response = event.get("response")
    return str(response.get("status", "")) if isinstance(response, dict) else ""


async def _collect_spoken_response(
    ws: Any,
    timeout_s: float,
    tombstones: _ResponseTombstones,
) -> ResponseResult:
    deadline = time.perf_counter() + timeout_s
    response_id = ""
    created_at = 0.0
    audio_done_at: float | None = None
    transcripts: dict[str, TranscriptState] = {}
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise ScenarioTimeout
        event = await _recv_json(ws, remaining, tombstones)
        event_type = event["type"]
        rid = _response_id(event)
        now = time.perf_counter()
        if event_type == "response.created":
            response_id = rid
            created_at = now
        if rid:
            transcripts.setdefault(rid, TranscriptState()).feed(event)
        if event_type in {"response.audio.done", "response.output_audio.done"} and (
            not response_id or not rid or rid == response_id
        ):
            audio_done_at = now
        if event_type == "response.done" and (not response_id or not rid or rid == response_id):
            response_id = rid or response_id
            transcript = transcripts.get(response_id, TranscriptState()).text()
            tombstones.retire(response_id)
            return ResponseResult(
                response_id=response_id,
                status=_response_status(event),
                transcript=transcript,
                created_at=created_at,
                done_at=now,
                audio_done_at=audio_done_at,
            )


def _number_sequence(text: str) -> list[int]:
    result: list[int] = []
    for match in _NUMBER_RE.finditer(text):
        token = match.group(0).lower()
        result.append(_NUMBER_WORDS.get(token, int(token) if token.isdigit() else 0))
    return result


def _normalise_marker(text: str) -> str:
    return "".join(character for character in text.upper() if character.isalnum())


def _valid_tool_arguments(event: Mapping[str, Any]) -> bool:
    if event.get("name") != "web_search":
        return False
    call_id = event.get("call_id")
    arguments = event.get("arguments")
    if not isinstance(call_id, str) or not call_id.strip() or not isinstance(arguments, str):
        return False
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return False
    return (
        isinstance(parsed, dict)
        and set(parsed) == {"query"}
        and isinstance(parsed.get("query"), str)
        and bool(parsed["query"].strip())
    )


async def _run_tool_turn(
    ws: Any,
    *,
    timeout_s: float,
    tool_code: str,
    tombstones: _ResponseTombstones,
) -> tuple[dict[str, Any], str]:
    deadline = time.perf_counter() + timeout_s
    transcripts: dict[str, TranscriptState] = {}
    tool_call_count = 0
    tool_output_count = 0
    output_ack_count = 0
    response_create_count = 0
    timeline: dict[str, float] = {}
    call_id = ""
    origin_response_id = ""
    origin_done_count = 0
    post_tool_response_id = ""
    post_tool_created_count = 0
    post_tool_done_count = 0
    post_tool_status = ""
    post_tool_text = ""

    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise ScenarioTimeout
        event = await _recv_json(ws, remaining, tombstones)
        event_type = event["type"]
        rid = _response_id(event)
        now = time.perf_counter()
        if rid:
            transcripts.setdefault(rid, TranscriptState()).feed(event)
        if event_type == "response.created":
            if not rid:
                raise ProtocolFailure
            if not origin_response_id:
                if response_create_count:
                    raise ProtocolFailure
                origin_response_id = rid
                timeline["origin_response_created"] = now
            elif response_create_count and not post_tool_response_id:
                if origin_done_count != 1 or rid == origin_response_id:
                    raise ProtocolFailure
                post_tool_response_id = rid
                post_tool_created_count = 1
                timeline["post_response_created"] = now
            else:
                raise ProtocolFailure
        elif event_type == "response.function_call_arguments.done":
            tool_call_count += 1
            timeline.setdefault("tool_call", now)
            if (
                tool_call_count != 1
                or not origin_response_id
                or rid != origin_response_id
                or not _valid_tool_arguments(event)
            ):
                raise ProtocolFailure
            call_id = str(event["call_id"])
            output = json.dumps(
                {"status": "found", "verification_code": tool_code},
                separators=(",", ":"),
            )
            await _send_json(
                ws,
                {
                    "type": "conversation.item.create",
                    "item": {"type": "function_call_output", "call_id": call_id, "output": output},
                },
            )
            tool_output_count += 1
            timeline["tool_output"] = time.perf_counter()
        elif event_type == "conversation.item.created":
            item = event.get("item")
            if (
                isinstance(item, dict)
                and item.get("type") == "function_call_output"
                and item.get("call_id") == call_id
            ):
                output_ack_count += 1
                if output_ack_count != 1 or tool_output_count != 1:
                    raise ProtocolFailure
                timeline["output_ack"] = now
                await _send_json(ws, {"type": "response.create", "response": {"tool_choice": "none"}})
                response_create_count = 1
                timeline["response_create"] = time.perf_counter()
        elif event_type == "response.done":
            if rid == origin_response_id:
                origin_done_count += 1
                if (
                    origin_done_count != 1
                    or post_tool_response_id
                    or _response_status(event) != "completed"
                ):
                    raise ProtocolFailure
                timeline["origin_response_done"] = now
                tombstones.retire(origin_response_id)
            elif post_tool_response_id and rid == post_tool_response_id:
                post_tool_done_count += 1
                if post_tool_done_count != 1 or post_tool_created_count != 1 or origin_done_count != 1:
                    raise ProtocolFailure
                post_tool_status = _response_status(event)
                post_tool_text = transcripts.get(rid, TranscriptState()).text()
                timeline["post_response_done"] = now
                tombstones.retire(post_tool_response_id)
                break
            else:
                raise ProtocolFailure

    tool_ordered_keys = (
        "tool_call",
        "tool_output",
        "output_ack",
        "response_create",
    )
    response_ordered_keys = (
        "origin_response_created",
        "origin_response_done",
        "post_response_created",
        "post_response_done",
    )
    ordering_valid = (
        all(key in timeline for key in (*tool_ordered_keys, *response_ordered_keys))
        and all(
            timeline[left] <= timeline[right]
            for left, right in zip(tool_ordered_keys, tool_ordered_keys[1:])
        )
        and all(
            timeline[left] <= timeline[right]
            for left, right in zip(response_ordered_keys, response_ordered_keys[1:])
        )
        and timeline["response_create"] <= timeline["post_response_created"]
    )
    result_observed = _normalise_marker(post_tool_text) == _normalise_marker(tool_code)
    return (
        {
            "call_count": tool_call_count,
            "output_count": tool_output_count,
            "output_ack_count": output_ack_count,
            "response_create_count": response_create_count,
            "origin_done_count": origin_done_count,
            "post_response_created_count": post_tool_created_count,
            "post_response_done_count": post_tool_done_count,
            "ordering_valid": ordering_valid,
            "post_tool_response_completed": post_tool_status == "completed",
            "post_tool_result_observed": result_observed,
        },
        post_tool_text,
    )


async def _run_cancellation(
    ws: Any,
    timeout_s: float,
    tombstones: _ResponseTombstones,
) -> dict[str, Any]:
    deadline = time.perf_counter() + timeout_s
    requested_count = 0
    audio_done_count = 0
    cancelled_done_count = 0
    timeline: dict[str, float] = {}
    active_response_id = ""
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise ScenarioTimeout
        event = await _recv_json(ws, remaining, tombstones)
        event_type = event["type"]
        rid = _response_id(event)
        now = time.perf_counter()
        if event_type == "response.created" and requested_count == 0:
            active_response_id = rid
            timeline["created"] = now
            await _send_json(ws, {"type": "response.cancel"})
            requested_count = 1
            timeline["cancel_sent"] = time.perf_counter()
        elif event_type in {"response.audio.done", "response.output_audio.done"} and (
            not active_response_id or not rid or rid == active_response_id
        ):
            audio_done_count += 1
            timeline.setdefault("audio_done", now)
        elif event_type == "response.done" and (not active_response_id or not rid or rid == active_response_id):
            if _response_status(event) in {"cancelled", "canceled"}:
                cancelled_done_count += 1
            timeline["response_done"] = now
            tombstones.retire(rid or active_response_id)
            break
    ordering_valid = (
        all(key in timeline for key in ("created", "cancel_sent", "audio_done", "response_done"))
        and timeline["created"] <= timeline["cancel_sent"] <= timeline["audio_done"] <= timeline["response_done"]
    )
    return {
        "requested_count": requested_count,
        "audio_done_count": audio_done_count,
        "cancelled_done_count": cancelled_done_count,
        "ordering_valid": ordering_valid,
    }


async def run_live_gate(
    settings: RuntimeSettings,
    prompt_audio: Mapping[str, bytes],
    *,
    response_timeout_s: float = DEFAULT_RESPONSE_TIMEOUT_S,
    pace_s: float = PCM_CHUNK_MS / 1000,
    connect: Callable[..., Any] = websockets.connect,
    tool_code: str | None = None,
) -> dict[str, Any]:
    """Run all scenarios in one WebSocket; return content-free evidence only."""

    _validate_managed_targets(MANAGED_SETTINGS_URL, settings.websocket_url)
    required_prompts = set(_PROMPTS)
    if set(prompt_audio) != required_prompts or any(not value for value in prompt_audio.values()):
        raise InvalidSynthesizedAudio
    private_tool_code = tool_code or f"VCODE-{secrets.token_hex(6).upper()}"
    started = time.perf_counter()
    config_ack_ms = 0.0
    scenario_timings: dict[str, int] = {}
    input_audio_append_count = 0
    audio_before_ack_count = 0
    session_configured = False
    tombstones = _ResponseTombstones()

    failure_stage = "connect"
    try:
        async with connect(settings.websocket_url, max_size=2**24) as ws:
            await _wait_for_event(
                ws,
                "session.created",
                CONFIG_ACK_TIMEOUT_S,
                tombstones=tombstones,
            )
            failure_stage = "config_ack"
            await _send_json(ws, {"type": "pipeline.config.update", "config": _pipeline_config(settings)})
            config_sent = time.perf_counter()
            try:
                _, config_ack_at = await _wait_for_event(
                    ws,
                    "pipeline.config.updated",
                    CONFIG_ACK_TIMEOUT_S,
                    lambda event: _pipeline_ack_matches(event, settings),
                    tombstones,
                )
            except (ScenarioTimeout, ProtocolFailure) as exc:
                raise ConfigAcknowledgementFailed from exc
            config_ack_ms = (config_ack_at - config_sent) * 1000.0
            failure_stage = "session_ack"
            await _send_json(ws, {"type": "session.update", "session": _session_config(settings)})
            try:
                await _wait_for_event(
                    ws,
                    "session.updated",
                    CONFIG_ACK_TIMEOUT_S,
                    lambda event: _session_ack_matches(event, tool_choice="auto", settings=settings),
                    tombstones,
                )
            except (ScenarioTimeout, ProtocolFailure) as exc:
                raise SessionAcknowledgementFailed from exc
            session_configured = True

            async def spoken_turn(name: str) -> ResponseResult:
                nonlocal input_audio_append_count, audio_before_ack_count
                scenario_start = time.perf_counter()
                if not session_configured:
                    audio_before_ack_count += 1
                input_audio_append_count += await _stream_pcm(
                    ws,
                    prompt_audio[name],
                    session_configured=session_configured,
                    pace_s=pace_s,
                )
                result = await _collect_spoken_response(ws, response_timeout_s, tombstones)
                scenario_timings[name] = round((time.perf_counter() - scenario_start) * 1000)
                return result

            failure_stage = "count_up"
            count_up = await spoken_turn("count_up")
            failure_stage = "count_down"
            count_down = await spoken_turn("count_down")
            count_up_valid = count_up.status == "completed" and _number_sequence(count_up.transcript) == list(range(1, 11))
            count_down_valid = count_down.status == "completed" and _number_sequence(count_down.transcript) == list(range(10, 0, -1))

            failure_stage = "search_tool_required_ack"
            await _send_json(
                ws,
                {"type": "session.update", "session": {"type": "realtime", "tool_choice": "required"}},
            )
            try:
                await _wait_for_event(
                    ws,
                    "session.updated",
                    CONFIG_ACK_TIMEOUT_S,
                    lambda event: _session_ack_matches(event, tool_choice="required"),
                    tombstones,
                )
            except (ScenarioTimeout, ProtocolFailure) as exc:
                raise SessionAcknowledgementFailed from exc

            failure_stage = "search"
            search_start = time.perf_counter()
            input_audio_append_count += await _stream_pcm(
                ws,
                prompt_audio["search"],
                session_configured=session_configured,
                pace_s=pace_s,
            )
            tool_result, _private_post_tool_text = await _run_tool_turn(
                ws,
                timeout_s=response_timeout_s,
                tool_code=private_tool_code,
                tombstones=tombstones,
            )
            scenario_timings["search"] = round((time.perf_counter() - search_start) * 1000)

            failure_stage = "search_tool_restore_ack"
            await _send_json(
                ws,
                {"type": "session.update", "session": {"type": "realtime", "tool_choice": "auto"}},
            )
            try:
                await _wait_for_event(
                    ws,
                    "session.updated",
                    CONFIG_ACK_TIMEOUT_S,
                    lambda event: _session_ack_matches(event, tool_choice="auto"),
                    tombstones,
                )
            except (ScenarioTimeout, ProtocolFailure) as exc:
                raise SessionAcknowledgementFailed from exc

            failure_stage = "search_followup"
            search_followup = await spoken_turn("search_followup")
            contextual_followup_preserved = (
                search_followup.status == "completed"
                and _normalise_marker(search_followup.transcript) == _normalise_marker(private_tool_code)
            )

            failure_stage = "cancellation"
            cancel_start = time.perf_counter()
            input_audio_append_count += await _stream_pcm(
                ws,
                prompt_audio["cancel"],
                session_configured=session_configured,
                pace_s=pace_s,
            )
            cancellation = await _run_cancellation(ws, response_timeout_s, tombstones)
            scenario_timings["cancel"] = round((time.perf_counter() - cancel_start) * 1000)
            failure_stage = "recovery"
            recovery = await spoken_turn("recovery")
            recovery_valid = recovery.status == "completed" and _normalise_marker(recovery.transcript) == "READY"
            cancellation["recovery_valid"] = recovery_valid
    except Exception as exc:
        try:
            exc.failure_stage = failure_stage
        except Exception:
            replacement = ProtocolFailure()
            replacement.failure_stage = failure_stage
            raise replacement from exc
        raise

    context_preserved = count_up_valid and count_down_valid
    tool_gate = (
        tool_result["call_count"] == 1
        and tool_result["output_count"] == 1
        and tool_result["output_ack_count"] == 1
        and tool_result["response_create_count"] == 1
        and tool_result["ordering_valid"]
        and tool_result["post_tool_response_completed"]
        and tool_result["post_tool_result_observed"]
        and contextual_followup_preserved
    )
    cancellation_gate = (
        cancellation["requested_count"] == 1
        and cancellation["audio_done_count"] == 1
        and cancellation["cancelled_done_count"] == 1
        and cancellation["ordering_valid"]
        and cancellation["recovery_valid"]
    )
    gate_passed = (
        session_configured
        and audio_before_ack_count == 0
        and context_preserved
        and tool_gate
        and cancellation_gate
    )
    return {
        "config": {
            "acknowledged": session_configured,
            "audio_before_ack_count": audio_before_ack_count,
            "ack_ms": round(config_ack_ms),
        },
        "input_audio_append_count": input_audio_append_count,
        "context": {
            "turn_count": 2,
            "ascending_valid": count_up_valid,
            "reverse_valid": count_down_valid,
            "preserved": context_preserved,
        },
        "tool": {
            **tool_result,
            "contextual_followup_preserved": contextual_followup_preserved,
            "gate_passed": tool_gate,
        },
        "cancellation": {**cancellation, "gate_passed": cancellation_gate},
        "timing_ms": {
            "scenario_total": round((time.perf_counter() - started) * 1000),
            "scenario_count": len(scenario_timings),
            "scenario_max": max(scenario_timings.values(), default=0),
        },
        "gate_passed": gate_passed,
    }


def _bounded_failure(exc: BaseException, fallback_stage: str = "unknown") -> dict[str, Any]:
    stage = getattr(exc, "failure_stage", fallback_stage)
    if stage == "unknown" and fallback_stage in _FAILURE_STAGES:
        stage = fallback_stage
    if stage not in _FAILURE_STAGES:
        stage = "unknown"
    return {"error_class": type(exc).__name__, "failure_stage": stage, "gate_passed": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings-url", default=MANAGED_SETTINGS_URL)
    parser.add_argument("--websocket-url", default=MANAGED_WEBSOCKET_URL)
    parser.add_argument("--response-timeout", type=float, default=DEFAULT_RESPONSE_TIMEOUT_S)
    args = parser.parse_args(argv)
    try:
        _validate_managed_targets(args.settings_url, args.websocket_url)
    except Exception as exc:  # noqa: BLE001 - only the bounded class reaches stdout
        result = _bounded_failure(exc, "settings")
        print(json.dumps(result, sort_keys=True))
        return 2
    key = os.environ.get(CREDENTIAL_ENV)
    if not key:
        result = _bounded_failure(CredentialUnavailable(), "credential")
        print(json.dumps(result, sort_keys=True))
        return 2
    failure_stage = "synthesis"
    try:
        audio = synthesize_prompts_in_memory()
        failure_stage = "settings"
        with httpx.Client(timeout=15.0) as client:
            settings = resolve_runtime_settings(client, args.settings_url, args.websocket_url, key)
        failure_stage = "connect"
        result = asyncio.run(
            run_live_gate(
                settings,
                audio,
                response_timeout_s=max(5.0, min(180.0, args.response_timeout)),
            )
        )
    except Exception as exc:  # noqa: BLE001 - only the bounded class reaches stdout
        result = _bounded_failure(exc, failure_stage)
        print(json.dumps(result, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0 if result["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
