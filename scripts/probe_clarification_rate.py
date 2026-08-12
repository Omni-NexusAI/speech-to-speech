"""Privacy-safe 100-turn direct-audio clarification-rate release gate.

The probe deliberately emits only aggregate counts, rates, booleans, timings,
and exception class names.  All utterance text, synthesized WAV data, model
history, model output, endpoint details, model identity, voice identity, and
credentials remain process-local and are never written or publicly printed.

This is a controlled semantic-recoverability gate.  It is not a WER benchmark,
an AEC/room-acoustics test, or a substitute for the managed WebSocket tool and
playback gates.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import logging
import math
import os
import pathlib
import random
import re
import subprocess
import sys
import time
import wave
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit

import httpx
import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from speech_to_speech.LLM.chat import Chat, make_user_audio_message  # noqa: E402
from speech_to_speech.STT.gemma_audio_handler import (  # noqa: E402
    DIRECT_AUDIO_TEMPERATURE,
    DIRECT_AUDIO_TOP_P,
    GemmaAudioSTTHandler,
)

_NORMAL_TURNS = 100
_MEANINGLESS_TURNS = 8
_CATEGORIES = ("clear", "hesitation", "alternate_voice", "moderate_noise", "code_switch")
_CULTURES = ("en-US", "en-GB", "es-ES", "de-DE", "ja-JP")
_CREDENTIAL_ENV = "S2S_REMOTE_MODEL_API_KEY"
_SETTINGS_URL = "http://127.0.0.1:7862/api/ui-settings"
_SNR_DB = 18.0
_RANDOM_SEED = 41021


class ProbeError(RuntimeError):
    """Base class whose type, but never message, may enter public output."""


class CredentialUnavailable(ProbeError):
    pass


class SettingsUnavailable(ProbeError):
    pass


class UntrustedManagedTarget(ProbeError):
    pass


class SpeechSynthesisUnavailable(ProbeError):
    pass


class InvalidSynthesizedAudio(ProbeError):
    pass


class ModelRequestFailed(ProbeError):
    pass


class InvalidModelResponse(ProbeError):
    pass


class ProbeProgressError(ProbeError):
    def __init__(self, *, attempted: int, completed: int, cause: BaseException):
        super().__init__()
        self.attempted = attempted
        self.completed = completed
        self.error_class = type(cause).__name__


@dataclass(frozen=True)
class Scenario:
    index: int
    category: str
    utterance: str
    culture: str
    voice_variant: int
    expected_number: int | None
    noise_seed: int | None = None
    normal: bool = True


@dataclass(frozen=True)
class SynthesizedAudio:
    wav: bytes
    voice_enumeration_available: bool
    culture_matched: bool
    alternate_selected: bool


@dataclass(frozen=True)
class ParsedTurn:
    visible: str
    transcript: str | None
    memory: str | None
    tool_call_count: int


def _operation_text(a: int, operator: str, b: int) -> str:
    word = "plus" if operator == "+" else "minus"
    return f"{a} {word} {b}"


def _code_switch_text(index: int, expression: str) -> tuple[str, str]:
    variants = (
        ("es-ES", f"Por favor, answer this using digits: {expression}. Dime el resultado."),
        ("de-DE", f"Bitte answer this using digits: {expression}. Sag mir das Ergebnis."),
        ("ja-JP", f"数字で answer this question: {expression}. 結果を教えてください。"),
        ("en-US", f"Please responde con números: {expression}. Give me the Ergebnis."),
    )
    return variants[index % len(variants)]


def _monolingual_text(culture: str, a: int, b: int) -> str:
    variants = {
        "en-US": f"Please give the answer in digits: {a} plus {b}.",
        "en-GB": f"Please give the answer in digits: {a} plus {b}.",
        "es-ES": f"Por favor, responde con d\u00edgitos: {a} m\u00e1s {b}.",
        "de-DE": f"Bitte antworte mit Ziffern: {a} plus {b}.",
        "ja-JP": f"{a} \u305f\u3059 {b} \u306e\u7b54\u3048\u3092\u6570\u5b57\u3067\u6559\u3048\u3066\u304f\u3060\u3055\u3044\u3002",
    }
    return variants[culture]


def build_scenarios() -> tuple[list[Scenario], list[Scenario]]:
    normal: list[Scenario] = []
    rng = random.Random(_RANDOM_SEED)
    unique_left_operands = rng.sample(range(100, 1000), _NORMAL_TURNS)
    index = 0
    for category in _CATEGORIES:
        for case_index in range(20):
            a = unique_left_operands[index]
            operator = "+"
            b = 1
            expected = a + b
            expression = _operation_text(a, operator, b)
            culture = "en-US"
            voice_variant = 0
            noise_seed: int | None = None
            if category == "clear":
                utterance = f"Please answer this short question using digits: {expression}."
            elif category == "hesitation":
                utterance = f"Um, could you, uh, answer this using digits: {expression}?"
            elif category == "alternate_voice":
                culture = _CULTURES[case_index % len(_CULTURES)]
                voice_variant = 1
                utterance = _monolingual_text(culture, a, b)
            elif category == "moderate_noise":
                utterance = f"Please answer in digits: {expression}."
                noise_seed = _RANDOM_SEED + case_index
            else:
                culture, utterance = _code_switch_text(case_index, expression)
            normal.append(
                Scenario(
                    index=index,
                    category=category,
                    utterance=utterance,
                    culture=culture,
                    voice_variant=voice_variant,
                    expected_number=expected,
                    noise_seed=noise_seed,
                )
            )
            index += 1

    meaningless_phrases = (
        "zorple navik temba",
        "mivra kelto shan",
        "plovin tarka nesh",
        "vessa drumik po",
        "kelnor zavi trum",
        "shoma pelvik ren",
        "dravin kofta sel",
        "nemba vorsik tal",
    )
    meaningless = [
        Scenario(
            index=_NORMAL_TURNS + offset,
            category="meaningless",
            utterance=phrase,
            culture="en-US",
            voice_variant=0,
            expected_number=None,
            noise_seed=_RANDOM_SEED + 1000 + offset,
            normal=False,
        )
        for offset, phrase in enumerate(meaningless_phrases)
    ]
    assert len(normal) == _NORMAL_TURNS
    assert len(meaningless) == _MEANINGLESS_TURNS
    return normal, meaningless


_SAPI_CATALOG_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$voices = @()
$catalogSynth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
  $voices = @($catalogSynth.GetInstalledVoices() | Where-Object { $_.Enabled } | ForEach-Object {
    [PSCustomObject]@{
      name = [string]$_.VoiceInfo.Name
      culture = [string]$_.VoiceInfo.Culture.Name
    }
  })
} finally {
  try { $catalogSynth.Dispose() } catch {}
}
[Console]::Out.Write((ConvertTo-Json ([PSCustomObject]@{
  available = $true
  voices = $voices
}) -Compress))
"""


_SAPI_SYNTH_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
$request = ConvertFrom-Json ([Console]::In.ReadToEnd())
$items = @($request.items)
$voices = @($request.voices)
$voiceEnumerationAvailable = $request.voice_enumeration_available -eq $true
$output = @()
foreach ($item in @($items)) {
  $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
  try {
    $matches = @($voices | Where-Object {
      $_.culture -eq [string]$item.culture
    })
    $cultureMatched = $false
    $alternate = $false
    if ($matches.Count -gt 0) {
      $slot = [Math]::Min([int]$item.voice_variant, $matches.Count - 1)
      try {
        $synth.SelectVoice($matches[$slot].name)
        $cultureMatched = $true
        $alternate = $slot -gt 0
      } catch {
        $cultureMatched = $false
        $alternate = $false
      }
    }
    $synth.Rate = 0
    $stream = New-Object System.IO.MemoryStream
    try {
      $synth.SetOutputToWaveStream($stream)
      $synth.Speak([string]$item.text)
      $synth.SetOutputToNull()
      $output += [PSCustomObject]@{
        wav = [Convert]::ToBase64String($stream.ToArray())
        voice_enumeration_available = $voiceEnumerationAvailable
        culture_matched = $cultureMatched
        alternate_selected = $alternate
      }
    } finally {
      $stream.Dispose()
    }
  } finally {
    $synth.Dispose()
  }
}
[Console]::Out.Write((ConvertTo-Json $output -Compress))
"""


def _powershell_executable() -> str:
    return str(pathlib.Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe")


def _validate_wav(raw: bytes) -> bytes:
    try:
        with wave.open(io.BytesIO(raw), "rb") as source:
            if source.getnchannels() != 1 or source.getsampwidth() != 2 or source.getnframes() <= 0:
                raise InvalidSynthesizedAudio
            source.readframes(source.getnframes())
    except (wave.Error, EOFError) as exc:
        raise InvalidSynthesizedAudio from exc
    return raw


@contextmanager
def _serial_sapi_runner():
    """Serialize real System.Speech access across concurrent test/probe processes."""

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
    handle = kernel32.CreateMutexW(None, False, "Local\\HFRealtimeClarificationSapi")
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


def synthesize_scenarios_in_memory(
    scenarios: Sequence[Scenario],
    *,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> dict[int, SynthesizedAudio]:
    if sys.platform != "win32" and runner is subprocess.run:
        raise SpeechSynthesisUnavailable
    if runner is subprocess.run:
        with _serial_sapi_runner():
            return _synthesize_scenarios_in_memory(scenarios, runner=runner)
    return _synthesize_scenarios_in_memory(scenarios, runner=runner)


def _synthesize_scenarios_in_memory(
    scenarios: Sequence[Scenario],
    *,
    runner: Callable[..., subprocess.CompletedProcess[bytes]],
) -> dict[int, SynthesizedAudio]:
    common_args = [_powershell_executable(), "-NoLogo", "-NoProfile", "-NonInteractive", "-Command"]
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    voices: list[dict[str, str]] = []
    voice_enumeration_available = False
    try:
        catalog = runner(
            [*common_args, _SAPI_CATALOG_SCRIPT],
            input=b"",
            capture_output=True,
            text=False,
            timeout=30,
            check=False,
            creationflags=creation_flags,
        )
        if catalog.returncode == 0:
            decoded_catalog = json.loads(catalog.stdout.decode("utf-8"))
            raw_voices = decoded_catalog.get("voices") if isinstance(decoded_catalog, dict) else None
            if decoded_catalog.get("available") is True and isinstance(raw_voices, list):
                for voice in raw_voices:
                    if (
                        isinstance(voice, dict)
                        and isinstance(voice.get("name"), str)
                        and isinstance(voice.get("culture"), str)
                    ):
                        voices.append({"name": voice["name"], "culture": voice["culture"]})
                voice_enumeration_available = True
    except (OSError, subprocess.SubprocessError, UnicodeDecodeError, json.JSONDecodeError, AttributeError):
        voices = []
        voice_enumeration_available = False

    child_input = json.dumps(
        {
            "voice_enumeration_available": voice_enumeration_available,
            "voices": voices,
            "items": [
            {
                "text": scenario.utterance,
                "culture": scenario.culture,
                "voice_variant": scenario.voice_variant,
            }
            for scenario in scenarios
            ],
        },
        ensure_ascii=False,
    ).encode("utf-8")
    try:
        completed = runner(
            [*common_args, _SAPI_SYNTH_SCRIPT],
            input=child_input,
            capture_output=True,
            text=False,
            timeout=180,
            check=False,
            creationflags=creation_flags,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SpeechSynthesisUnavailable from exc
    if completed.returncode != 0:
        raise SpeechSynthesisUnavailable
    try:
        encoded = json.loads(completed.stdout.decode("utf-8"))
    except (AttributeError, UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise SpeechSynthesisUnavailable from exc
    if not isinstance(encoded, list) or len(encoded) != len(scenarios):
        raise SpeechSynthesisUnavailable
    result: dict[int, SynthesizedAudio] = {}
    for scenario, item in zip(scenarios, encoded, strict=True):
        if not isinstance(item, dict) or not isinstance(item.get("wav"), str):
            raise SpeechSynthesisUnavailable
        try:
            raw = base64.b64decode(item["wav"], validate=True)
        except (ValueError, TypeError) as exc:
            raise InvalidSynthesizedAudio from exc
        result[scenario.index] = SynthesizedAudio(
            wav=_validate_wav(raw),
            voice_enumeration_available=item.get("voice_enumeration_available") is True,
            culture_matched=item.get("culture_matched") is True,
            alternate_selected=item.get("alternate_selected") is True,
        )
    return result


def add_deterministic_noise(raw_wav: bytes, *, seed: int, snr_db: float = _SNR_DB) -> bytes:
    with wave.open(io.BytesIO(raw_wav), "rb") as source:
        params = source.getparams()
        samples = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float64)
    if params.nchannels != 1 or params.sampwidth != 2 or not samples.size:
        raise InvalidSynthesizedAudio
    signal_rms = float(np.sqrt(np.mean(np.square(samples))))
    if not math.isfinite(signal_rms) or signal_rms <= 1.0:
        raise InvalidSynthesizedAudio
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, samples.size)
    noise_rms = float(np.sqrt(np.mean(np.square(noise))))
    target_rms = signal_rms / (10.0 ** (snr_db / 20.0))
    mixed = np.clip(samples + noise * (target_rms / noise_rms), -32768, 32767).astype("<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setparams(params)
        target.writeframes(mixed.tobytes())
    return output.getvalue()


def _handler(model_name: str) -> GemmaAudioSTTHandler:
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.system_prompt = "You are a concise conversational assistant."
    handler.audio_format = "wav"
    handler.stream = False
    handler.gen_kwargs = {
        "temperature": DIRECT_AUDIO_TEMPERATURE,
        "top_p": DIRECT_AUDIO_TOP_P,
    }
    handler.base_url = ""
    handler.model_name = model_name
    handler.api_key = None
    handler._owned_user_context = lambda _vad: None  # type: ignore[method-assign]
    handler._conversation_image_urls = lambda _runtime: []  # type: ignore[method-assign]
    return handler


def _vad_context(chat: Chat) -> SimpleNamespace:
    session = SimpleNamespace(instructions="", tools=[], tool_choice=None)
    runtime = SimpleNamespace(
        session=session,
        chat=chat,
        local_pipeline={"max_response_tokens": 160},
        model_endpoint=None,
    )
    return SimpleNamespace(runtime_config=runtime)


def build_primary_payload(
    handler: GemmaAudioSTTHandler,
    chat: Chat,
    encoded_wav: str,
) -> dict[str, Any]:
    return handler._payload(np.zeros(1, dtype=np.float32), _vad_context(chat), encoded_audio=encoded_wav)


def parse_primary_response(handler: GemmaAudioSTTHandler, data: Mapping[str, Any]) -> ParsedTurn:
    text, tool_calls = handler._message_text_and_tools(dict(data))
    transcript = handler._validate_transcript(handler._extract_transcript(text))
    memory = None if transcript else handler._validate_user_memory(handler._extract_user_memory(text))
    visible = handler._fallback_response_text(text).strip()
    return ParsedTurn(visible=visible, transcript=transcript, memory=memory, tool_call_count=len(tool_calls))


_GARBLING_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:garbl|transcri|audio (?:was|is)|speech (?:was|is)|could not hear|couldn't hear|cannot hear|can't hear)\w*\b",
        r"\b(?:no (?:te )?entend|transcripci|audio (?:estaba|está)|no puedo oír)\w*\b",
        r"\b(?:nicht verstanden|nicht verstehen|unklar|transkrip|audio war|nicht hören)\w*\b",
        r"(?:聞き取れ|聞こえ|文字起こし|音声|不明瞭|分かりません|わかりません)",
    )
)
_CLARIFICATION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:could|can|would) you (?:please )?(?:repeat|rephrase|say that again)\b",
        r"\bwhat do you mean\b",
        r"\b(?:puedes|podrías) (?:repetir|decirlo de nuevo)\b",
        r"\b(?:kannst|könntest) du (?:das )?(?:wiederholen|anders sagen)\b",
        r"(?:もう一度|言い換えて|どういう意味)",
    )
)


def _contains_expected_number(text: str, expected: int | None) -> bool:
    if expected is None:
        return False
    return re.search(rf"(?<!\d){expected}(?!\d)", text) is not None


def classify_visible_response(text: str, expected: int | None) -> str:
    if any(pattern.search(text) for pattern in _GARBLING_PATTERNS):
        return "garbling_commentary"
    if any(pattern.search(text) for pattern in _CLARIFICATION_PATTERNS):
        return "clarification"
    if text.rstrip().endswith(("?", "？")):
        return "clarification"
    if _contains_expected_number(text, expected):
        return "ordinary"
    return "unclassified"


def _resolve_target(client: httpx.Client) -> tuple[str, str]:
    parsed = urlsplit(_SETTINGS_URL)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.port != 7862
        or parsed.path != "/api/ui-settings"
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise UntrustedManagedTarget
    try:
        response = client.get(_SETTINGS_URL)
        response.raise_for_status()
        if response.history or str(response.url) != _SETTINGS_URL:
            raise UntrustedManagedTarget
        payload = response.json()
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        raise SettingsUnavailable from exc
    settings = payload.get("settings") if isinstance(payload, dict) else None
    if not isinstance(settings, dict) or settings.get("modelProvider") != "remote":
        raise SettingsUnavailable
    base_url = settings.get("modelUrl")
    model = settings.get("modelName")
    if not isinstance(base_url, str) or not base_url.strip() or not isinstance(model, str) or not model.strip():
        raise SettingsUnavailable
    target = base_url.rstrip("/")
    if not target.endswith("/chat/completions"):
        target = f"{target}/chat/completions"
    return target, model


def _request_primary(
    client: httpx.Client,
    target: str,
    credential: str,
    payload: Mapping[str, Any],
    timeout_s: float,
) -> Mapping[str, Any]:
    try:
        response = client.post(
            target,
            headers={"Authorization": f"Bearer {credential}"},
            json=dict(payload),
            timeout=timeout_s,
        )
        response.raise_for_status()
        data = response.json()
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        raise ModelRequestFailed from exc
    if not isinstance(data, dict):
        raise InvalidModelResponse
    return data


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    slot = min(len(ordered) - 1, max(0, math.ceil(percentile * len(ordered)) - 1))
    return round(ordered[slot], 1)


def _empty_category_counts() -> dict[str, dict[str, int]]:
    return {
        category: {"ordinary": 0, "clarification": 0, "garbling_commentary": 0, "unclassified": 0}
        for category in _CATEGORIES
    }


@contextmanager
def _suppress_chat_logger_output():
    chat_logger = logging.getLogger("speech_to_speech.LLM.chat")
    previous_disabled = chat_logger.disabled
    chat_logger.disabled = True
    try:
        yield
    finally:
        chat_logger.disabled = previous_disabled


def run_gate(
    *,
    timeout_s: float = 30.0,
    client_factory: Callable[..., httpx.Client] = httpx.Client,
    synthesizer: Callable[[Sequence[Scenario]], dict[int, SynthesizedAudio]] = synthesize_scenarios_in_memory,
    requester: Callable[[httpx.Client, str, str, Mapping[str, Any], float], Mapping[str, Any]] = _request_primary,
) -> dict[str, Any]:
    with _suppress_chat_logger_output():
        return _run_gate(
            timeout_s=timeout_s,
            client_factory=client_factory,
            synthesizer=synthesizer,
            requester=requester,
        )


def _run_gate(
    *,
    timeout_s: float = 30.0,
    client_factory: Callable[..., httpx.Client] = httpx.Client,
    synthesizer: Callable[[Sequence[Scenario]], dict[int, SynthesizedAudio]] = synthesize_scenarios_in_memory,
    requester: Callable[[httpx.Client, str, str, Mapping[str, Any], float], Mapping[str, Any]] = _request_primary,
) -> dict[str, Any]:
    started = time.perf_counter()
    if _CREDENTIAL_ENV not in os.environ:
        raise CredentialUnavailable
    normal, meaningless = build_scenarios()
    all_scenarios = [*normal, *meaningless]
    category_counts = _empty_category_counts()
    meaningless_counts = {"clarification": 0, "garbling_commentary": 0, "other": 0}
    request_ms: list[float] = []
    requests_attempted = 0
    requests_completed = 0
    transcript_anchors = 0
    memory_anchors = 0
    audio_anchors = 0
    tool_call_responses = 0
    clarification_fingerprints: dict[str, int] = {}
    culture_required = 0
    culture_matched = 0
    alternate_required = 0
    alternate_selected = 0
    chat = Chat(30)

    with client_factory(timeout=timeout_s, follow_redirects=False) as client:
        target, model = _resolve_target(client)
        credential = os.environ.get(_CREDENTIAL_ENV)
        if not credential:
            raise CredentialUnavailable
        synthesis_started = time.perf_counter()
        synthesized = synthesizer(all_scenarios)
        synthesis_ms = round((time.perf_counter() - synthesis_started) * 1000.0, 1)
        if set(synthesized) != {scenario.index for scenario in all_scenarios}:
            raise InvalidSynthesizedAudio
        handler = _handler(model)
        for scenario in all_scenarios:
            try:
                generated = synthesized[scenario.index]
                if scenario.category in {"alternate_voice", "code_switch"}:
                    culture_required += 1
                    culture_matched += int(generated.culture_matched)
                if scenario.category == "alternate_voice":
                    alternate_required += 1
                    alternate_selected += int(generated.alternate_selected)
                wav = generated.wav
                if scenario.noise_seed is not None:
                    wav = add_deterministic_noise(wav, seed=scenario.noise_seed)
                encoded = base64.b64encode(wav).decode("ascii")
                payload = build_primary_payload(handler, chat, encoded)
                request_started = time.perf_counter()
                requests_attempted += 1
                data = requester(client, target, credential, payload, timeout_s)
                requests_completed += 1
                request_ms.append(round((time.perf_counter() - request_started) * 1000.0, 1))
                parsed = parse_primary_response(handler, data)

                user = chat.add_item(make_user_audio_message(encoded))
                if user.id is None:
                    raise InvalidModelResponse
                if parsed.transcript:
                    chat.replace_user_message_text(user.id, parsed.transcript)
                    transcript_anchors += 1
                elif parsed.memory:
                    chat.replace_user_message_text(user.id, parsed.memory)
                    memory_anchors += 1
                else:
                    audio_anchors += 1
                chat.commit_assistant_response(user.id, parsed.visible, [])
                tool_call_responses += parsed.tool_call_count

                classification = classify_visible_response(parsed.visible, scenario.expected_number)
                if scenario.normal:
                    category_counts[scenario.category][classification] += 1
                    if classification == "clarification":
                        normalized_visible = " ".join(re.findall(r"\w+", parsed.visible.casefold(), flags=re.UNICODE))
                        fingerprint = hashlib.sha256(normalized_visible.encode("utf-8")).hexdigest()
                        clarification_fingerprints[fingerprint] = clarification_fingerprints.get(fingerprint, 0) + 1
                elif classification in {"clarification", "garbling_commentary"}:
                    meaningless_counts[classification] += 1
                else:
                    meaningless_counts["other"] += 1
            except Exception as exc:
                raise ProbeProgressError(
                    attempted=requests_attempted,
                    completed=requests_completed,
                    cause=exc,
                ) from exc

    totals = {name: sum(counts[name] for counts in category_counts.values()) for name in next(iter(category_counts.values()))}
    clarification_rate = round(100.0 * totals["clarification"] / _NORMAL_TURNS, 2)
    garbling_rate = round(100.0 * totals["garbling_commentary"] / _NORMAL_TURNS, 2)
    classification_complete = totals["ordinary"] + totals["clarification"] + totals["garbling_commentary"] == _NORMAL_TURNS
    voice_coverage_complete = culture_required == culture_matched and alternate_required == alternate_selected
    repeated_stock_clarifications = sum(count - 1 for count in clarification_fingerprints.values() if count > 1)
    passed = (
        requests_attempted == _NORMAL_TURNS + _MEANINGLESS_TURNS
        and requests_completed == _NORMAL_TURNS + _MEANINGLESS_TURNS
        and totals["clarification"] <= 1
        and totals["garbling_commentary"] == 0
        and repeated_stock_clarifications == 0
        and classification_complete
        and tool_call_responses == 0
        and voice_coverage_complete
    )
    return {
        "gate_passed": passed,
        "normal_turns": _NORMAL_TURNS,
        "meaningless_turns": _MEANINGLESS_TURNS,
        "primary_requests": requests_completed,
        "primary_requests_attempted": requests_attempted,
        "primary_requests_completed": requests_completed,
        "one_primary_per_turn": requests_attempted == requests_completed == _NORMAL_TURNS + _MEANINGLESS_TURNS,
        "ordinary": totals["ordinary"],
        "clarifications": totals["clarification"],
        "clarification_rate_percent": clarification_rate,
        "clarification_target_met": totals["clarification"] <= 1,
        "clarification_maximum_met": totals["clarification"] <= 1,
        "repeated_stock_clarifications": repeated_stock_clarifications,
        "repeated_stock_free": repeated_stock_clarifications == 0,
        "garbling_commentary": totals["garbling_commentary"],
        "garbling_rate_percent": garbling_rate,
        "unclassified": totals["unclassified"],
        "classification_complete": classification_complete,
        "category_counts": category_counts,
        "meaningless_counts": meaningless_counts,
        "transcript_anchors": transcript_anchors,
        "memory_anchors": memory_anchors,
        "audio_anchors": audio_anchors,
        "tool_call_responses": tool_call_responses,
        "culture_voice_required": culture_required,
        "culture_voice_matched": culture_matched,
        "alternate_voice_required": alternate_required,
        "alternate_voice_selected": alternate_selected,
        "voice_coverage_complete": voice_coverage_complete,
        "synthesis_ms": synthesis_ms,
        "request_mean_ms": round(sum(request_ms) / len(request_ms), 1) if request_ms else 0.0,
        "request_p95_ms": _percentile(request_ms, 0.95),
        "request_max_ms": round(max(request_ms), 1) if request_ms else 0.0,
        "total_ms": round((time.perf_counter() - started) * 1000.0, 1),
        "error_class": None,
    }


def _failure(exc: BaseException) -> dict[str, Any]:
    attempted = exc.attempted if isinstance(exc, ProbeProgressError) else 0
    completed = exc.completed if isinstance(exc, ProbeProgressError) else 0
    error_class = exc.error_class if isinstance(exc, ProbeProgressError) else type(exc).__name__
    return {
        "gate_passed": False,
        "normal_turns": _NORMAL_TURNS,
        "meaningless_turns": _MEANINGLESS_TURNS,
        "primary_requests": completed,
        "primary_requests_attempted": attempted,
        "primary_requests_completed": completed,
        "error_class": error_class,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the content-free accepted-turn clarification-rate gate.")
    parser.add_argument("--timeout", type=float, default=30.0, help="Per-request timeout in seconds.")
    args = parser.parse_args(argv)
    try:
        report = run_gate(timeout_s=max(1.0, min(float(args.timeout), 120.0)))
    except Exception as exc:  # noqa: BLE001 - public output is type-only by contract.
        report = _failure(exc)
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0 if report.get("gate_passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
