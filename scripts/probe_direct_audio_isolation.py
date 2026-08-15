"""RAM-only direct-audio isolation matrix with content-free public evidence.

The probe resolves the model selected by the managed UI, synthesizes every
fixed multilingual fixture in one in-memory System.Speech invocation, and
sends one non-streaming Chat Completions request per matrix cell.  It compares
fresh versus deliberately wrong semantic history and a short prompt versus the
production Gemma direct-audio payload builder.

This is direct endpoint evidence, not a live managed-pipeline replay.  The
production arm constructs an isolated in-memory Chat and calls the production
payload serializer without mutating a live session.  Audio, prompts, model
responses, semantic history, endpoints, model names, and credentials are never
written or printed.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import pathlib
import subprocess
import sys
import time
import unicodedata
import wave
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx
import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from speech_to_speech.LLM.chat import Chat, make_assistant_message, make_user_message  # noqa: E402
from speech_to_speech.STT.gemma_audio_handler import (  # noqa: E402
    DIRECT_AUDIO_TEMPERATURE,
    DIRECT_AUDIO_TOP_P,
    GemmaAudioSTTHandler,
)

PromptArm = Literal["short", "full_production"]
HistoryArm = Literal["fresh", "wrong_semantic"]

_MANAGED_SETTINGS_URL = "http://127.0.0.1:7862/api/ui-settings"
_MANAGED_LOCAL_PIPELINE_URL = "http://127.0.0.1:7862/api/local-pipeline"
_LOCAL_MODEL_BASE_URL = "http://127.0.0.1:8818/v1"
_CREDENTIAL_ENV = "S2S_REMOTE_MODEL_API_KEY"
_DEFAULT_TIMEOUT_S = 45.0
_MAX_WAV_BYTES = 2_000_000
_MAX_WAV_SECONDS = 12.0
_MAX_RESPONSE_TOKENS = 64

_SHORT_SYSTEM_PROMPT = (
    "Follow the current spoken request. Reply with only the requested numeric label and no explanation. "
    "The current audio always overrides earlier conversation text."
)
_PRODUCTION_BASE_PROMPT = "You are a local low-latency voice assistant. Answer naturally for speech synthesis."
_PRODUCTION_SESSION_PROMPT = (
    "For this fixed capability check, follow the current spoken request and put only its requested numeric label "
    "in ASSISTANT_RESPONSE."
)


class ProbeError(RuntimeError):
    """Base error whose message never enters public output."""


class SettingsUnavailable(ProbeError):
    pass


class UntrustedManagedTarget(ProbeError):
    pass


class CredentialUnavailable(ProbeError):
    pass


class SpeechSynthesisUnavailable(ProbeError):
    pass


class InvalidFixtureAudio(ProbeError):
    pass


class InvalidRequestShape(ProbeError):
    pass


class ModelRequestFailed(ProbeError):
    pass


class InvalidModelResponse(ProbeError):
    pass


class ProbeProgressError(ProbeError):
    def __init__(
        self,
        *,
        attempted: int,
        completed: int,
        shape_valid: int,
        synthesis_success: int,
        voice_matched: int,
        observations: Sequence[_CellObservation] = (),
    ) -> None:
        super().__init__()
        self.attempted = max(0, int(attempted))
        self.completed = max(0, int(completed))
        self.shape_valid = max(0, int(shape_valid))
        self.synthesis_success = max(0, int(synthesis_success))
        self.voice_matched = max(0, int(voice_matched))
        self.observations = tuple(observations)


@dataclass(frozen=True, repr=False)
class _ManagedTarget:
    endpoint: str = field(repr=False)
    model: str = field(repr=False)
    requires_credential: bool = field(repr=False)


@dataclass(frozen=True, repr=False)
class FixtureDefinition:
    index: int
    culture: str = field(repr=False)
    utterance: str = field(repr=False)
    expected_label: str = field(repr=False)
    wrong_memory: str = field(repr=False)
    wrong_label: str = field(repr=False)


@dataclass(repr=False)
class SynthesizedFixture:
    wav: bytearray = field(repr=False)
    culture_matched: bool


@dataclass(frozen=True, repr=False)
class ProbeCell:
    fixture: FixtureDefinition = field(repr=False)
    prompt_arm: PromptArm
    history_arm: HistoryArm

    @property
    def public_arm(self) -> str:
        return f"{self.history_arm}_{self.prompt_arm}"


@dataclass(frozen=True, repr=False)
class _ShapeObservation:
    valid: bool
    input_audio_parts: int
    text_current_user_parts: int
    history_messages: int
    tool_surface: bool
    streaming: bool


@dataclass(frozen=True, repr=False)
class _CellObservation:
    arm: str
    schema_valid: bool
    exact_label: bool
    elapsed_ms: float
    shape: _ShapeObservation = field(repr=False)


_FIXTURES: tuple[FixtureDefinition, ...] = (
    FixtureDefinition(
        index=0,
        culture="en-US",
        utterance="Reply with only the numeric label seventeen, written as digits.",
        expected_label="17",
        wrong_memory="The user requests numeric label ninety one, written as digits.",
        wrong_label="91",
    ),
    FixtureDefinition(
        index=1,
        culture="es-ES",
        utterance="Responde solo con la etiqueta numérica veintiocho, escrita con dígitos.",
        expected_label="28",
        wrong_memory="El usuario pide la etiqueta numérica ochenta y dos, escrita con dígitos.",
        wrong_label="82",
    ),
    FixtureDefinition(
        index=2,
        culture="de-DE",
        utterance="Antworte nur mit der numerischen Kennzahl neununddreißig, als Ziffern geschrieben.",
        expected_label="39",
        wrong_memory="Der Nutzer verlangt die Kennzahl dreiundsiebzig, als Ziffern geschrieben.",
        wrong_label="73",
    ),
    FixtureDefinition(
        index=3,
        culture="ja-JP",
        utterance="数字の四十六だけを、算用数字で答えてください。",
        expected_label="46",
        wrong_memory="ユーザーは数字の六十四を算用数字で求めています。",
        wrong_label="64",
    ),
)

_CELL_ARMS: tuple[tuple[PromptArm, HistoryArm], ...] = (
    ("short", "fresh"),
    ("short", "wrong_semantic"),
    ("full_production", "wrong_semantic"),
    ("full_production", "fresh"),
)


def probe_cells() -> tuple[ProbeCell, ...]:
    """Return the fixed, bounded request matrix."""

    return tuple(
        ProbeCell(fixture=fixture, prompt_arm=prompt_arm, history_arm=history_arm)
        for fixture in _FIXTURES
        for prompt_arm, history_arm in _CELL_ARMS
    )


_SAPI_SYNTH_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
$request = ConvertFrom-Json ([Console]::In.ReadToEnd())
$catalog = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
  $voices = @($catalog.GetInstalledVoices() | Where-Object { $_.Enabled } | ForEach-Object {
    [PSCustomObject]@{
      name = [string]$_.VoiceInfo.Name
      culture = [string]$_.VoiceInfo.Culture.Name
    }
  })
} finally {
  $catalog.Dispose()
}
$format = [System.Speech.AudioFormat.SpeechAudioFormatInfo]::new(
  16000,
  [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
  [System.Speech.AudioFormat.AudioChannel]::Mono
)
$output = @()
foreach ($item in @($request.items)) {
  $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
  try {
    $matches = @($voices | Where-Object { $_.culture -eq [string]$item.culture })
    $cultureMatched = $matches.Count -gt 0
    if ($cultureMatched) {
      $synth.SelectVoice([string]$matches[0].name)
    }
    $stream = New-Object System.IO.MemoryStream
    try {
      $synth.SetOutputToAudioStream($stream, $format)
      $synth.Speak([string]$item.text)
      $synth.SetOutputToNull()
      $pcm = $stream.ToArray()
      $waveStream = New-Object System.IO.MemoryStream
      $writer = New-Object System.IO.BinaryWriter($waveStream)
      try {
        $writer.Write([System.Text.Encoding]::ASCII.GetBytes('RIFF'))
        $writer.Write([int](36 + $pcm.Length))
        $writer.Write([System.Text.Encoding]::ASCII.GetBytes('WAVE'))
        $writer.Write([System.Text.Encoding]::ASCII.GetBytes('fmt '))
        $writer.Write([int]16)
        $writer.Write([int16]1)
        $writer.Write([int16]1)
        $writer.Write([int]16000)
        $writer.Write([int]32000)
        $writer.Write([int16]2)
        $writer.Write([int16]16)
        $writer.Write([System.Text.Encoding]::ASCII.GetBytes('data'))
        $writer.Write([int]$pcm.Length)
        $writer.Write($pcm)
        $writer.Flush()
        $encodedWave = [Convert]::ToBase64String($waveStream.ToArray())
      } finally {
        $writer.Dispose()
        $waveStream.Dispose()
      }
      $output += [PSCustomObject]@{
        index = [int]$item.index
        wav = $encodedWave
        culture_matched = $cultureMatched
      }
    } finally {
      $stream.Dispose()
    }
  } finally {
    $synth.Dispose()
  }
}
[Console]::Out.Write((ConvertTo-Json -InputObject @($output) -Compress))
"""


def _powershell_executable() -> str:
    system_root = os.environ.get("SystemRoot", r"C:\Windows")
    return str(pathlib.Path(system_root) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe")


@contextmanager
def _serial_sapi_runner():
    if sys.platform != "win32":
        yield
        return
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.argtypes = (ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR)
    kernel32.CreateMutexW.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.ReleaseMutex.argtypes = (wintypes.HANDLE,)
    kernel32.ReleaseMutex.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.CreateMutexW(None, False, "Local\\HFRealtimeDirectIsolationSapi")
    if not handle:
        raise SpeechSynthesisUnavailable
    acquired = False
    try:
        wait_result = kernel32.WaitForSingleObject(handle, 240_000)
        acquired = wait_result in {0x00000000, 0x00000080}
        if not acquired:
            raise SpeechSynthesisUnavailable
        yield
    finally:
        if acquired:
            kernel32.ReleaseMutex(handle)
        kernel32.CloseHandle(handle)


def _validate_wav(raw: bytearray) -> None:
    if not isinstance(raw, bytearray) or not raw or len(raw) > _MAX_WAV_BYTES:
        raise InvalidFixtureAudio
    try:
        with wave.open(io.BytesIO(raw), "rb") as source:
            frame_rate = source.getframerate()
            frame_count = source.getnframes()
            if (
                source.getnchannels() != 1
                or source.getsampwidth() != 2
                or frame_rate != 16_000
                or frame_count <= 0
                or frame_count / frame_rate > _MAX_WAV_SECONDS
            ):
                raise InvalidFixtureAudio
            source.readframes(frame_count)
    except (EOFError, wave.Error) as exc:
        raise InvalidFixtureAudio from exc


def _synthesize_fixed_fixtures(
    fixtures: Sequence[FixtureDefinition],
    *,
    runner: Callable[..., subprocess.CompletedProcess[bytes]],
) -> dict[int, SynthesizedFixture]:
    child_input = json.dumps(
        {
            "items": [
                {"index": fixture.index, "culture": fixture.culture, "text": fixture.utterance} for fixture in fixtures
            ]
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    try:
        completed = runner(
            [
                _powershell_executable(),
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                _SAPI_SYNTH_SCRIPT,
            ],
            input=child_input,
            capture_output=True,
            text=False,
            timeout=180,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SpeechSynthesisUnavailable from exc
    if completed.returncode != 0:
        raise SpeechSynthesisUnavailable
    try:
        decoded = json.loads(completed.stdout.decode("utf-8"))
    except (AttributeError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SpeechSynthesisUnavailable from exc
    if not isinstance(decoded, list) or len(decoded) != len(fixtures):
        raise SpeechSynthesisUnavailable

    result: dict[int, SynthesizedFixture] = {}
    for item in decoded:
        if not isinstance(item, Mapping) or isinstance(item.get("index"), bool):
            raise SpeechSynthesisUnavailable
        try:
            index = int(item["index"])
            raw = bytearray(base64.b64decode(item["wav"], validate=True))
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidFixtureAudio from exc
        _validate_wav(raw)
        if index in result:
            raise InvalidFixtureAudio
        result[index] = SynthesizedFixture(wav=raw, culture_matched=item.get("culture_matched") is True)
    return result


def synthesize_fixed_fixtures_in_memory(
    fixtures: Sequence[FixtureDefinition],
    *,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> dict[int, SynthesizedFixture]:
    """Synthesize every fixed fixture in one child call without disk audio."""

    if tuple(fixtures) != _FIXTURES:
        raise InvalidFixtureAudio
    if sys.platform != "win32" and runner is subprocess.run:
        raise SpeechSynthesisUnavailable
    if runner is subprocess.run:
        with _serial_sapi_runner():
            return _synthesize_fixed_fixtures(fixtures, runner=runner)
    return _synthesize_fixed_fixtures(fixtures, runner=runner)


def _validate_managed_url(value: str, path: str) -> None:
    try:
        parsed = urlsplit(value)
    except (TypeError, ValueError) as exc:
        raise UntrustedManagedTarget from exc
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.port != 7862
        or parsed.path != path
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise UntrustedManagedTarget


def _managed_json(client: httpx.Client, url: str, path: str) -> Mapping[str, Any]:
    _validate_managed_url(url, path)
    try:
        response = client.get(url)
        response.raise_for_status()
        if response.history or str(response.url) != url:
            raise UntrustedManagedTarget
        payload = response.json()
    except UntrustedManagedTarget:
        raise
    except (httpx.HTTPError, TypeError, ValueError) as exc:
        raise SettingsUnavailable from exc
    if not isinstance(payload, Mapping):
        raise SettingsUnavailable
    return payload


def _chat_completions_target(base_url: str) -> str:
    target = base_url.rstrip("/")
    if not target.endswith("/chat/completions"):
        target = f"{target}/chat/completions"
    return target


def _resolve_managed_target(client: httpx.Client) -> _ManagedTarget:
    """Resolve the managed local/remote target before any credential read."""

    settings_payload = _managed_json(client, _MANAGED_SETTINGS_URL, "/api/ui-settings")
    settings = settings_payload.get("settings")
    if not isinstance(settings, Mapping):
        raise SettingsUnavailable
    provider = settings.get("modelProvider")
    if provider == "local":
        local_payload = _managed_json(client, _MANAGED_LOCAL_PIPELINE_URL, "/api/local-pipeline")
        gemma = local_payload.get("gemma")
        if not isinstance(gemma, Mapping):
            raise SettingsUnavailable
        base_url = gemma.get("baseUrl")
        model = gemma.get("model")
        if not isinstance(base_url, str) or not isinstance(model, str) or not model.strip():
            raise SettingsUnavailable
        try:
            parsed = urlsplit(base_url)
        except ValueError as exc:
            raise UntrustedManagedTarget from exc
        if (
            base_url.rstrip("/") != _LOCAL_MODEL_BASE_URL
            or parsed.scheme != "http"
            or parsed.hostname != "127.0.0.1"
            or parsed.port != 8818
            or parsed.path.rstrip("/") != "/v1"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise UntrustedManagedTarget
        return _ManagedTarget(
            endpoint=_chat_completions_target(base_url),
            model=model.strip(),
            requires_credential=False,
        )

    if provider != "remote":
        raise SettingsUnavailable
    base_url = settings.get("modelUrl")
    model = settings.get("modelName")
    if not isinstance(base_url, str) or not base_url.strip() or not isinstance(model, str) or not model.strip():
        raise SettingsUnavailable
    try:
        parsed = urlsplit(base_url)
    except ValueError as exc:
        raise UntrustedManagedTarget from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise UntrustedManagedTarget
    return _ManagedTarget(
        endpoint=_chat_completions_target(base_url),
        model=model.strip(),
        requires_credential=True,
    )


def _history_messages(cell: ProbeCell) -> list[dict[str, Any]]:
    if cell.history_arm == "fresh":
        return []
    return [
        {"role": "user", "content": cell.fixture.wrong_memory},
        {"role": "assistant", "content": cell.fixture.wrong_label},
    ]


def _short_payload(model: str, cell: ProbeCell, encoded_wav: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": _SHORT_SYSTEM_PROMPT},
            *_history_messages(cell),
            {
                "role": "user",
                "content": [{"type": "input_audio", "input_audio": {"data": encoded_wav, "format": "wav"}}],
            },
        ],
        "stream": False,
        "temperature": DIRECT_AUDIO_TEMPERATURE,
        "top_p": DIRECT_AUDIO_TOP_P,
        "max_tokens": _MAX_RESPONSE_TOKENS,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _isolated_production_payload(model: str, cell: ProbeCell, encoded_wav: str) -> dict[str, Any]:
    """Use the production serializer against a private Chat, never the live session."""

    handler = object.__new__(GemmaAudioSTTHandler)
    handler.system_prompt = _PRODUCTION_BASE_PROMPT
    handler.audio_format = "wav"
    handler.stream = False
    handler.gen_kwargs = {}
    handler.base_url = ""
    handler.model_name = model
    handler.api_key = None
    handler._owned_user_context = lambda _vad: None  # type: ignore[method-assign]
    handler._provisional_history_audio = lambda _vad: None  # type: ignore[method-assign]
    handler._conversation_image_urls = lambda _runtime: []  # type: ignore[method-assign]

    chat = Chat(30)
    if cell.history_arm == "wrong_semantic":
        chat.add_item(make_user_message(cell.fixture.wrong_memory))
        chat.add_item(make_assistant_message(cell.fixture.wrong_label))
    session = SimpleNamespace(instructions=_PRODUCTION_SESSION_PROMPT, tools=[], tool_choice=None)
    runtime = SimpleNamespace(
        session=session,
        chat=chat,
        local_pipeline={"max_response_tokens": _MAX_RESPONSE_TOKENS},
        model_endpoint=None,
    )
    vad_audio = SimpleNamespace(runtime_config=runtime, turn_id="isolated_probe", turn_revision=0)
    return handler._payload(np.empty(0, dtype=np.float32), vad_audio, encoded_audio=encoded_wav)


def build_cell_payload(model: str, cell: ProbeCell, encoded_wav: str) -> dict[str, Any]:
    """Injectable deterministic envelope arm used by the direct endpoint probe."""

    if cell.prompt_arm == "short":
        return _short_payload(model, cell, encoded_wav)
    return _isolated_production_payload(model, cell, encoded_wav)


def _shape_observation(
    payload: Mapping[str, Any],
    *,
    model: str,
    cell: ProbeCell,
    encoded_wav: str,
) -> _ShapeObservation:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return _ShapeObservation(False, 0, 0, 0, True, bool(payload.get("stream")))
    roles = [message.get("role") if isinstance(message, Mapping) else None for message in messages]
    expected_roles = ["system", "user"] if cell.history_arm == "fresh" else ["system", "user", "assistant", "user"]
    input_audio_parts = 0
    current_text_parts = 0
    audio_matches = False
    if messages and isinstance(messages[-1], Mapping):
        current_content = messages[-1].get("content")
        if isinstance(current_content, list):
            for part in current_content:
                if not isinstance(part, Mapping):
                    continue
                if part.get("type") == "input_audio":
                    input_audio_parts += 1
                    descriptor = part.get("input_audio")
                    audio_matches = (
                        isinstance(descriptor, Mapping)
                        and descriptor.get("data") == encoded_wav
                        and descriptor.get("format") == "wav"
                    )
                elif part.get("type") in {"input_text", "text"}:
                    current_text_parts += 1
    system_text = messages[0].get("content") if messages and isinstance(messages[0], Mapping) else None
    if cell.prompt_arm == "short":
        prompt_valid = system_text == _SHORT_SYSTEM_PROMPT
    else:
        prompt_valid = (
            isinstance(system_text, str)
            and "USER_MEMORY:" in system_text
            and "ASSISTANT_LANGUAGE:" in system_text
            and "ASSISTANT_RESPONSE:" in system_text
            and _PRODUCTION_SESSION_PROMPT in system_text
        )
    history_valid = True
    if cell.history_arm == "wrong_semantic":
        history_valid = (
            len(messages) == 4
            and isinstance(messages[1], Mapping)
            and isinstance(messages[2], Mapping)
            and messages[1].get("content") == cell.fixture.wrong_memory
            and messages[2].get("content") == cell.fixture.wrong_label
        )
    tool_surface = "tools" in payload or "tool_choice" in payload
    streaming = payload.get("stream") is True
    valid = (
        roles == expected_roles
        and payload.get("model") == model
        and prompt_valid
        and history_valid
        and input_audio_parts == 1
        and current_text_parts == 0
        and audio_matches
        and not tool_surface
        and not streaming
        and payload.get("temperature") == DIRECT_AUDIO_TEMPERATURE
        and payload.get("top_p") == DIRECT_AUDIO_TOP_P
        and payload.get("max_tokens") == _MAX_RESPONSE_TOKENS
        and payload.get("chat_template_kwargs") == {"enable_thinking": False}
    )
    return _ShapeObservation(
        valid=valid,
        input_audio_parts=input_audio_parts,
        text_current_user_parts=current_text_parts,
        history_messages=max(0, len(messages) - 2),
        tool_surface=tool_surface,
        streaming=streaming,
    )


def _request_primary(
    client: httpx.Client,
    target: _ManagedTarget,
    credential: str | None,
    payload: Mapping[str, Any],
    timeout_s: float,
) -> Mapping[str, Any]:
    headers = {"Content-Type": "application/json"}
    if credential is not None:
        headers["Authorization"] = f"Bearer {credential}"
    try:
        response = client.post(
            target.endpoint,
            headers=headers,
            json=dict(payload),
            timeout=timeout_s,
        )
        response.raise_for_status()
        data = response.json()
    except (httpx.HTTPError, TypeError, ValueError) as exc:
        raise ModelRequestFailed from exc
    if not isinstance(data, Mapping):
        raise InvalidModelResponse
    return data


def _response_text(data: Mapping[str, Any]) -> str | None:
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return None
    if not isinstance(message, Mapping) or message.get("tool_calls"):
        return None
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [part.get("text", "") for part in content if isinstance(part, Mapping)]
        return "".join(str(part) for part in parts)
    return None


def _classify_response(data: Mapping[str, Any], expected_label: str) -> tuple[bool, bool]:
    raw = _response_text(data)
    if raw is None:
        return False, False
    visible = GemmaAudioSTTHandler._fallback_response_text(raw)
    normalized = unicodedata.normalize("NFKC", visible).strip()
    return bool(normalized), normalized == expected_label


def _wipe_fixture_buffers(fixtures: Mapping[int, SynthesizedFixture]) -> None:
    for synthesized in fixtures.values():
        if isinstance(synthesized.wav, bytearray):
            synthesized.wav[:] = b"\x00" * len(synthesized.wav)


def _discard_payload(payload: dict[str, Any] | None) -> None:
    if payload is None:
        return
    messages = payload.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("input_audio"), dict):
                    part["input_audio"]["data"] = ""
    payload.clear()


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) * percentile + 0.999999) - 1)))
    return round(ordered[index], 1)


def _timing_metrics(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean_ms": round(sum(values) / len(values), 1) if values else 0.0,
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
        "max_ms": round(max(values), 1) if values else 0.0,
    }


def _empty_arm_metrics() -> dict[str, dict[str, Any]]:
    return {
        f"{history}_{prompt}": {
            "request_success_count": 0,
            "schema_success_count": 0,
            "exact_label_count": 0,
            "timing": _timing_metrics([]),
        }
        for prompt, history in _CELL_ARMS
    }


def _request_shape_metrics(*, shape_valid: int, observations: Sequence[_CellObservation]) -> dict[str, Any]:
    cells = probe_cells()
    return {
        "shape_valid_count": shape_valid,
        "input_audio_part_count": sum(observation.shape.input_audio_parts for observation in observations),
        "text_current_user_part_count": sum(observation.shape.text_current_user_parts for observation in observations),
        "fresh_history_message_count": sum(
            observation.shape.history_messages for observation in observations if observation.arm.startswith("fresh_")
        ),
        "wrong_history_message_count": sum(
            observation.shape.history_messages
            for observation in observations
            if observation.arm.startswith("wrong_semantic_")
        ),
        "short_prompt_cell_count": sum(cell.prompt_arm == "short" for cell in cells),
        "isolated_production_payload_cell_count": sum(cell.prompt_arm == "full_production" for cell in cells),
        "direct_endpoint_cell_count": len(cells),
        "managed_pipeline_replay_count": 0,
        "tool_surface_cell_count": sum(observation.shape.tool_surface for observation in observations),
        "streaming_cell_count": sum(observation.shape.streaming for observation in observations),
        "response_tts_execution_count": 0,
    }


def _summarize(
    observations: Sequence[_CellObservation],
    *,
    synthesis_success: int,
    voice_matched: int,
    attempted: int,
    completed: int,
    shape_valid: int,
    total_ms: float,
) -> dict[str, Any]:
    arms = _empty_arm_metrics()
    for arm in arms:
        selected = [observation for observation in observations if observation.arm == arm]
        arms[arm] = {
            "request_success_count": len(selected),
            "schema_success_count": sum(observation.schema_valid for observation in selected),
            "exact_label_count": sum(observation.exact_label for observation in selected),
            "timing": _timing_metrics([observation.elapsed_ms for observation in selected]),
        }
    cells = probe_cells()
    exact = sum(observation.exact_label for observation in observations)
    schema = sum(observation.schema_valid for observation in observations)
    correction_attempts = sum(cell.history_arm == "wrong_semantic" for cell in cells)
    correction_successes = sum(
        observation.exact_label for observation in observations if observation.arm.startswith("wrong_semantic_")
    )
    passed = (
        synthesis_success == len(_FIXTURES)
        and voice_matched == len(_FIXTURES)
        and attempted == completed == len(cells)
        and shape_valid == len(cells)
        and schema == len(cells)
        and exact == len(cells)
        and correction_successes == correction_attempts
    )
    return {
        "gate_passed": passed,
        "fixture_count": len(_FIXTURES),
        "planned_cell_count": len(cells),
        "synthesis_success_count": synthesis_success,
        "voice_match_count": voice_matched,
        "request_attempted_count": attempted,
        "request_completed_count": completed,
        "request_success_count": len(observations),
        "schema_success_count": schema,
        "exact_label_count": exact,
        "correction_attempt_count": correction_attempts,
        "correction_success_count": correction_successes,
        "one_request_per_cell": attempted == completed == len(cells),
        "timing": {**_timing_metrics([observation.elapsed_ms for observation in observations]), "total_ms": total_ms},
        "request_shape": _request_shape_metrics(shape_valid=shape_valid, observations=observations),
        "arms": arms,
    }


@contextmanager
def _suppress_private_loggers():
    names = ("speech_to_speech.LLM.chat", "httpx", "httpcore")
    previous = {name: logging.getLogger(name).disabled for name in names}
    try:
        for name in names:
            logging.getLogger(name).disabled = True
        yield
    finally:
        for name, disabled in previous.items():
            logging.getLogger(name).disabled = disabled


def run_isolation_probe(
    *,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    client_factory: Callable[..., httpx.Client] = httpx.Client,
    synthesizer: Callable[
        [Sequence[FixtureDefinition]], Mapping[int, SynthesizedFixture]
    ] = synthesize_fixed_fixtures_in_memory,
    requester: Callable[
        [httpx.Client, _ManagedTarget, str | None, Mapping[str, Any], float], Mapping[str, Any]
    ] = _request_primary,
    payload_builder: Callable[[str, ProbeCell, str], dict[str, Any]] = build_cell_payload,
    credential_reader: Callable[[str], str | None] = os.environ.get,
) -> dict[str, Any]:
    """Run the fixed 16-cell direct endpoint matrix with no retries."""

    with _suppress_private_loggers():
        return _run_isolation_probe(
            timeout_s=timeout_s,
            client_factory=client_factory,
            synthesizer=synthesizer,
            requester=requester,
            payload_builder=payload_builder,
            credential_reader=credential_reader,
        )


def _run_isolation_probe(
    *,
    timeout_s: float,
    client_factory: Callable[..., httpx.Client],
    synthesizer: Callable[[Sequence[FixtureDefinition]], Mapping[int, SynthesizedFixture]],
    requester: Callable[[httpx.Client, _ManagedTarget, str | None, Mapping[str, Any], float], Mapping[str, Any]],
    payload_builder: Callable[[str, ProbeCell, str], dict[str, Any]],
    credential_reader: Callable[[str], str | None],
) -> dict[str, Any]:
    started = time.perf_counter()
    bounded_timeout = max(1.0, min(120.0, float(timeout_s)))
    attempted = 0
    completed = 0
    shape_valid = 0
    synthesis_success = 0
    voice_matched = 0
    synthesized: Mapping[int, SynthesizedFixture] = {}
    observations: list[_CellObservation] = []
    try:
        with client_factory(timeout=bounded_timeout, follow_redirects=False) as client:
            target = _resolve_managed_target(client)
            credential = credential_reader(_CREDENTIAL_ENV) if target.requires_credential else None
            if target.requires_credential and not credential:
                raise CredentialUnavailable
            synthesized = synthesizer(_FIXTURES)
            expected_indexes = {fixture.index for fixture in _FIXTURES}
            if set(synthesized) != expected_indexes:
                raise InvalidFixtureAudio
            for fixture in _FIXTURES:
                generated = synthesized[fixture.index]
                _validate_wav(generated.wav)
                synthesis_success += 1
                voice_matched += int(generated.culture_matched)
            if voice_matched != len(_FIXTURES):
                raise SpeechSynthesisUnavailable

            for cell in probe_cells():
                payload: dict[str, Any] | None = None
                data: Mapping[str, Any] | None = None
                encoded_wav = base64.b64encode(synthesized[cell.fixture.index].wav).decode("ascii")
                try:
                    payload = payload_builder(target.model, cell, encoded_wav)
                    shape = _shape_observation(
                        payload,
                        model=target.model,
                        cell=cell,
                        encoded_wav=encoded_wav,
                    )
                    if not shape.valid:
                        raise InvalidRequestShape
                    shape_valid += 1
                    request_started = time.perf_counter()
                    attempted += 1
                    data = requester(client, target, credential, payload, bounded_timeout)
                    completed += 1
                    elapsed_ms = round((time.perf_counter() - request_started) * 1000.0, 1)
                    schema_valid, exact_label = _classify_response(data, cell.fixture.expected_label)
                    observations.append(
                        _CellObservation(
                            arm=cell.public_arm,
                            schema_valid=schema_valid,
                            exact_label=exact_label,
                            elapsed_ms=elapsed_ms,
                            shape=shape,
                        )
                    )
                finally:
                    if isinstance(data, dict):
                        data.clear()
                    _discard_payload(payload)
                    encoded_wav = ""
    except Exception as exc:
        if isinstance(exc, ProbeProgressError):
            raise
        raise ProbeProgressError(
            attempted=attempted,
            completed=completed,
            shape_valid=shape_valid,
            synthesis_success=synthesis_success,
            voice_matched=voice_matched,
            observations=observations,
        ) from exc
    finally:
        _wipe_fixture_buffers(synthesized)

    return _summarize(
        observations,
        synthesis_success=synthesis_success,
        voice_matched=voice_matched,
        attempted=attempted,
        completed=completed,
        shape_valid=shape_valid,
        total_ms=round((time.perf_counter() - started) * 1000.0, 1),
    )


def _failure_report(exc: BaseException, *, total_ms: float) -> dict[str, Any]:
    if isinstance(exc, ProbeProgressError):
        return _summarize(
            exc.observations,
            synthesis_success=exc.synthesis_success,
            voice_matched=exc.voice_matched,
            attempted=exc.attempted,
            completed=exc.completed,
            shape_valid=exc.shape_valid,
            total_ms=round(max(0.0, total_ms), 1),
        )
    return _summarize(
        (),
        synthesis_success=0,
        voice_matched=0,
        attempted=0,
        completed=0,
        shape_valid=0,
        total_ms=round(max(0.0, total_ms), 1),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the RAM-only direct-audio isolation matrix.")
    parser.add_argument("--timeout", type=float, default=_DEFAULT_TIMEOUT_S, help="Per-request timeout in seconds.")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    execution_failed = False
    try:
        report = run_isolation_probe(timeout_s=args.timeout)
    except Exception as exc:  # noqa: BLE001 - exception content is deliberately discarded
        execution_failed = True
        report = _failure_report(exc, total_ms=(time.perf_counter() - started) * 1000.0)
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    if execution_failed:
        return 2
    return 0 if report["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
