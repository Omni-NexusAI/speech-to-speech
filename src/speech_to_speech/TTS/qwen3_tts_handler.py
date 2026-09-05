"""
Qwen3 TTS Handler

- On Apple Silicon: Uses mlx-audio with MLX-converted Qwen3-TTS models.
- On CUDA/CPU: Uses faster-qwen3-tts for low-latency streaming.
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import re
import tempfile
import unicodedata
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from secrets import randbits
from sys import platform
from threading import Event, Lock
from time import perf_counter
from typing import Any, Iterator, Optional

import httpx
import numpy as np
import torch
from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.cancellable_http import CancellableAsyncByteStream, StreamCancelled
from speech_to_speech.pipeline.control import SESSION_END, is_control_message
from speech_to_speech.pipeline.events import PipelineMetricEvent
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, PIPELINE_END, AudioOutput, EndOfResponse, TTSInput
from speech_to_speech.pipeline.response_ownership import response_output_allowed
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.utils.mlx_lock import MLXLockContext

logger = logging.getLogger(__name__)


class _NoopConsole:
    """Compatibility shim for callers that previously patched the Rich console."""

    @staticmethod
    def print(*_args: Any, **_kwargs: Any) -> None:
        return None


# Kept for downstream test compatibility. The handler never writes assistant
# content through it, so non-ASCII output cannot poison a Windows console.
console = _NoopConsole()


@dataclass(frozen=True)
class _ProviderRuntimeConfigSnapshot:
    """Minimal admission-time view used by provider resolution.

    Keeping the existing one-argument resolver boundary matters for downstream
    handlers and tests that replace ``_resolve_api_provider``.  The view still
    prevents an admitted response from rereading mutable session settings.
    """

    local_pipeline: dict[str, Any]

DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DEFAULT_MLX_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-6bit"
DEFAULT_REF_TEXT = "I'm confused why some people have super short timelines, yet at the same time are bullish on scaling up reinforcement learning atop LLMs. If we're actually close to a human-like learner, then this whole approach of training on verifiable outcomes."
DEFAULT_FASTER_STREAMING_CHUNK_SIZE = 8
MAX_COALESCED_TTS_CHARS = 420
DEFAULT_MLX_STREAMING_CHUNK_SIZE = 4
DEFAULT_QWEN3_TTS_MAX_NEW_TOKENS = 1536
DEFAULT_OPENAI_API_BASE_URL = "http://127.0.0.1:8881/v1"
DEFAULT_OPENAI_API_VOICE = "clone:16d9bb336799"
DEFAULT_OPENAI_API_BACKEND_MODEL = "1.7B-Base"
DEFAULT_GROXAXO_API_BASE_URL = "http://127.0.0.1:8882/v1"
DEFAULT_AUDIO_CPP_API_BASE_URL = "http://127.0.0.1:8890/v1"
AUDIO_CPP_NATIVE_COLD_FIRST_PCM_BUDGET_S = 45.0
DEFAULT_OPENAI_API_VOICE_LIBRARY_DIR = (
    r"C:\Users\yepyy\Documents\Codex\2026-05-24\files-mentioned-by-the-user-i"
    r"\qwen3-tts-candidate\voice_library_from_original"
)
MIN_QWEN3_TTS_UTTERANCE_TOKENS = 360
VALID_MLX_QUANTIZATION_SUFFIXES = ("bf16", "4bit", "6bit", "8bit")
VALID_FASTER_BACKENDS = ("ggml", "torch", "openai-api")
MLX_STREAMING_TOKENS_PER_SECOND = 12.5
PIPELINE_SR = 16000
ESTIMATED_QWEN3_WORDS_PER_SECOND = 2.6
ESTIMATED_QWEN3_CHARS_PER_SECOND = 14.0
QWEN3_TOKEN_SAFETY_MARGIN = 1.35
QWEN3_BASE_PROMPT_SECONDS = 1.0
QWEN3_PUNCTUATION_PAUSE_SECONDS = 0.5
QWEN3_LANGUAGE_ALIASES = {
    "zh": "chinese",
    "zh-cn": "chinese",
    "zh-hans": "chinese",
    "zh-hant": "chinese",
    "cmn": "chinese",
    "en": "english",
    "en-us": "english",
    "en-gb": "english",
    "ja": "japanese",
    "jp": "japanese",
    "ko": "korean",
    "kr": "korean",
    "de": "german",
    "fr": "french",
    "ru": "russian",
    "pt": "portuguese",
    "pt-br": "portuguese",
    "pt-pt": "portuguese",
    "es": "spanish",
    "it": "italian",
}


class TTSRunawayError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResponseSynthesisSnapshot:
    """Immutable voice settings used by every phrase in one assistant answer.

    The audio.cpp server is phrase-local: each request has a fresh acoustic
    session.  Reusing this snapshot prevents a live profile/clone change or a
    random seed from making later phrases sound like a different speaker.
    """

    key: tuple[Any, ...]
    input_epoch: int | None
    response_epoch: int | None
    response_id: str | None
    provider: str
    endpoint: str | None
    model: str | None
    model_epoch: str | None
    model_instance_id: str | None
    voice: str | None
    reference_fingerprint: str | None
    clone_content_revision: int | None
    clone_content_hash: str | None
    frozen_clone: dict[str, Any] | None
    profile_id: str | None
    profile_revision: int | None
    tuning: dict[str, Any] | None
    language: str | None
    seed: int
    seed_policy: str
    # Delivery capability is observed when the response first becomes stable.
    # A health refresh may update the handler for the next answer, never split
    # a phrase-local answer between native and buffered transports.
    delivery_native_pcm: bool = False
    delivery_streaming_mode: str = "buffered_phrase"
    delivery_response_format: str | None = None
    delivery_sample_rate: int | None = None


class Qwen3TTSHandler(BaseHandler[TTSIn, TTSOut]):
    """
    Handles Text-to-Speech using Qwen3-TTS.

    Backend selection:
      - Apple Silicon (Darwin): mlx-audio
      - Other platforms: faster-qwen3-tts

    Supports three generation modes depending on the loaded model:
      - Voice cloning (ref_audio + ref_text)
      - Custom voice (preset speakers)
      - Voice design (instruct prompt)
    """

    def setup(
        self,
        should_listen: Event,
        model_name: str = DEFAULT_MODEL,
        device: str = "cuda",
        dtype: str | torch.dtype = "auto",
        attn_implementation: str = "eager",
        backend: str = "ggml",
        ref_audio: str | Path | None = None,
        ref_text: str = DEFAULT_REF_TEXT,
        language: str = "auto",
        speaker: Optional[str] = "Aiden",
        instruct: Optional[str] = None,
        xvec_only: bool = False,
        parity_mode: bool = False,
        non_streaming_mode: bool | None = True,
        mlx_quantization: Optional[str] = None,
        streaming_chunk_size: int | None = None,
        max_new_tokens: int = DEFAULT_QWEN3_TTS_MAX_NEW_TOKENS,
        blocksize: int = 512,
        gen_kwargs: dict[str, Any] | None = None,
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
        api_base_url: str = DEFAULT_OPENAI_API_BASE_URL,
        api_key: str | None = None,
        api_model: str = "qwen3-tts",
        api_voice: str = DEFAULT_OPENAI_API_VOICE,
        api_fallback_voice: str | None = None,
        api_backend_model: str = DEFAULT_OPENAI_API_BACKEND_MODEL,
        api_voice_library_dir: str | None = None,
        api_sample_rate: int = 24000,
        api_timeout_s: float = 120.0,
        text_output_queue: Any | None = None,
    ) -> None:
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns
        self.should_listen = should_listen
        self.text_output_queue = text_output_queue
        self.requested_device = device
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self.language = self._normalize_language(language)
        self.speaker = speaker
        self.instruct = instruct
        self.xvec_only = xvec_only
        self.parity_mode = parity_mode
        self.non_streaming_mode = non_streaming_mode
        self.faster_backend = self._normalize_faster_backend(backend)
        self.mlx_quantization = self._normalize_mlx_quantization(mlx_quantization)
        self.max_new_tokens = max_new_tokens
        self.blocksize = blocksize
        self.dtype: torch.dtype | None | str = None
        self.gen_kwargs = gen_kwargs or {}
        self._mlx_ref_audio_cache: dict[str, Any] = {}
        self._mlx_temp_ref_audio_files: set[str] = set()
        self.api_base_url = (os.getenv("QWEN3_TTS_API_BASE_URL") or api_base_url).rstrip("/")
        self.api_key = api_key or os.getenv("QWEN3_TTS_API_KEY")
        self.api_model = api_model
        self.api_voice = api_voice
        self.api_fallback_voice = api_fallback_voice
        self.api_backend_model = api_backend_model
        self.api_voice_library_dir = self._resolve_api_voice_library_dir(api_voice_library_dir)
        self.api_response_format = "pcm"
        self.api_streaming_supported = False
        self.api_sample_rate = int(api_sample_rate)
        self.api_timeout = httpx.Timeout(float(api_timeout_s), connect=10.0)
        self.groxaxo_api_base_url = (
            os.getenv("QWEN3_TTS_GROXAXO_BASE_URL") or DEFAULT_GROXAXO_API_BASE_URL
        ).rstrip("/")
        self.audio_cpp_api_base_url = (
            os.getenv("QWEN3_TTS_AUDIO_CPP_BASE_URL") or DEFAULT_AUDIO_CPP_API_BASE_URL
        ).rstrip("/")
        self._active_response: httpx.Response | None = None
        self._active_response_lock = Lock()
        self._audio_cpp_native_warm_streams: set[tuple[str, str, str]] = set()
        self._audio_cpp_native_lifecycle_epochs: dict[tuple[str, str], str] = {}
        # One entry per assistant response, not per phrase.  The dictionary is
        # deliberately handler-local: session/runtime updates remain valid for
        # the next answer without being able to split an answer already in
        # progress.
        self._response_synthesis_snapshots: dict[tuple[Any, ...], ResponseSynthesisSnapshot] = {}
        self._response_phrase_counts: dict[tuple[Any, ...], int] = {}
        self._failed_response_synthesis_keys: set[tuple[Any, ...]] = set()

        if self.faster_backend == "openai-api":
            self.backend = "openai_api"
        else:
            self.backend = "mlx" if platform == "darwin" else "faster_qwen3_tts"
        self.streaming_chunk_size = self._resolve_streaming_chunk_size(streaming_chunk_size)

        if self.backend == "openai_api":
            self.device = "remote"
            self.model_name = model_name
            logger.info(
                "Using deferred OpenAI-compatible Qwen3-TTS API at %s; provider availability is checked per request",
                self.api_base_url,
            )
        elif self.backend == "mlx":
            self.device = "mps"
            self.model_name = self._resolve_mlx_model_name(model_name)
            logger.info(f"Loading Qwen3-TTS model: {self.model_name} via mlx-audio on Apple Silicon")
            if self.non_streaming_mode is not None:
                logger.debug(
                    "qwen3_tts_non_streaming_mode=%s is ignored on Apple Silicon because "
                    "mlx-audio does not expose non_streaming_mode yet.",
                    self.non_streaming_mode,
                )
            model_quantization = self._model_name_quantization_suffix(self.model_name)
            if model_quantization and model_quantization != "bf16":
                logger.info(
                    "Using MLX quantized Qwen3-TTS variant: %s",
                    model_quantization,
                )
            self._setup_mlx(self.model_name)
        else:
            self.device = device
            self.model_name = model_name
            logger.info(
                "Loading Qwen3-TTS model: %s via faster-qwen3-tts (%s backend)",
                self.model_name,
                self.faster_backend,
            )
            self._setup_faster(
                model_name=self.model_name,
                dtype=dtype,
                attn_implementation=attn_implementation,
                backend=self.faster_backend,
            )

        logger.info(
            "Using Qwen3-TTS streaming chunk size %d (~%.0fms audio per chunk) on %s",
            self.streaming_chunk_size,
            self.streaming_chunk_size / MLX_STREAMING_TOKENS_PER_SECOND * 1000,
            self.backend,
        )

        self._initial_speaker = self.speaker
        self._initial_ref_audio = self.ref_audio

        self.warmup()

    def _setup_faster(self, model_name: str, dtype: Any, attn_implementation: str, backend: str) -> None:
        try:
            import torch
        except ImportError as e:
            raise ImportError("torch is required. Install with: pip install torch") from e

        if dtype == "auto":
            self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif isinstance(dtype, str):
            self.dtype = getattr(torch, dtype)
        else:
            self.dtype = dtype

        try:
            from faster_qwen3_tts import FasterQwen3TTS
        except ImportError as e:
            raise ImportError(
                "faster-qwen3-tts is required for Qwen3 TTS on non-macOS platforms. "
                "Install with: pip install 'faster-qwen3-tts[ggml]'"
            ) from e

        self.model = FasterQwen3TTS.from_pretrained(
            model_name,
            device=self.device,
            dtype=self.dtype,
            attn_implementation=attn_implementation,
            backend=backend,
        )
        logger.info("Qwen3-TTS model loaded")

    def _setup_mlx(self, model_name: str) -> None:
        try:
            from mlx_audio.tts.utils import load_model

            self.model = load_model(model_name)
        except ImportError as e:
            message = str(e)
            if any(
                dep in message
                for dep in (
                    "misaki",
                    "spacy",
                    "phonemizer",
                    "espeakng_loader",
                )
            ):
                raise ImportError(
                    "Qwen3-TTS on Apple Silicon requires mlx-audio and its TTS dependencies. "
                    f"Missing dependency: {message}. "
                    "Install with: pip install mlx-audio misaki spacy phonemizer-fork espeakng-loader"
                ) from e
            raise ImportError(
                "mlx-audio is required for Qwen3 TTS on Apple Silicon. Install with: pip install mlx-audio"
            ) from e

        logger.info("MLX Audio Qwen3-TTS model loaded")

    def _normalize_faster_backend(self, backend: Any) -> str:
        value = str(backend or "ggml").strip().lower()
        if value not in VALID_FASTER_BACKENDS:
            raise ValueError(
                f"Unsupported qwen3_tts_backend value {backend!r}. Supported values: {', '.join(VALID_FASTER_BACKENDS)}"
            )
        return value

    def _normalize_mlx_quantization(self, mlx_quantization: Any) -> Optional[str]:
        if mlx_quantization is None:
            return None

        value = str(mlx_quantization).strip().lower()
        if value in ("", "none", "default"):
            return None
        if value not in VALID_MLX_QUANTIZATION_SUFFIXES:
            raise ValueError(
                "Unsupported qwen3_tts_mlx_quantization value "
                f"{mlx_quantization!r}. Supported values: {', '.join(VALID_MLX_QUANTIZATION_SUFFIXES)}"
            )
        return value

    def _apply_mlx_quantization_suffix(self, model_name: str) -> str:
        if self.mlx_quantization is None:
            return model_name

        desired_suffix = f"-{self.mlx_quantization}"
        for suffix in VALID_MLX_QUANTIZATION_SUFFIXES:
            current_suffix = f"-{suffix}"
            if model_name.endswith(current_suffix):
                return model_name[: -len(current_suffix)] + desired_suffix

        return f"{model_name}{desired_suffix}"

    def _model_name_quantization_suffix(self, model_name: str) -> Optional[str]:
        if not model_name:
            return None

        for suffix in VALID_MLX_QUANTIZATION_SUFFIXES:
            if model_name.endswith(f"-{suffix}"):
                return suffix

        return None

    def _resolve_mlx_model_name(self, model_name: str) -> str:
        if not model_name:
            return self._apply_mlx_quantization_suffix(DEFAULT_MLX_MODEL)
        if model_name.startswith("mlx-community/"):
            if self.mlx_quantization is None and self._model_name_quantization_suffix(model_name) is None:
                return f"{model_name}-6bit"
            return self._apply_mlx_quantization_suffix(model_name)
        if model_name.startswith("Qwen/"):
            mapped = model_name.replace("Qwen/", "mlx-community/", 1)
            if self._model_name_quantization_suffix(mapped) is None:
                if self.mlx_quantization is None:
                    return f"{mapped}-6bit"
                mapped = f"{mapped}-bf16"
            return self._apply_mlx_quantization_suffix(mapped)
        return model_name

    def _resolve_streaming_chunk_size(self, streaming_chunk_size: int | None) -> int:
        if streaming_chunk_size is not None:
            return max(1, int(streaming_chunk_size))
        if self.backend == "mlx":
            return DEFAULT_MLX_STREAMING_CHUNK_SIZE
        return DEFAULT_FASTER_STREAMING_CHUNK_SIZE

    def _normalize_language(self, language: str | None) -> str:
        if language is None:
            return "auto"
        normalized = str(language).strip().replace("_", "-").lower()
        if not normalized:
            return "auto"
        return QWEN3_LANGUAGE_ALIASES.get(normalized, normalized)

    def _infer_model_type_from_name(self) -> str:
        name = (self.model_name or "").lower()
        if "voicedesign" in name:
            return "voice_design"
        if "customvoice" in name:
            return "custom_voice"
        return "base"

    def _resolve_audio_path(self, audio: Any) -> Path | None:
        if not isinstance(audio, (str, Path)) or not audio:
            return None

        candidate = Path(audio).expanduser()
        repo_root = Path(__file__).resolve().parents[1]
        search_paths = []

        if candidate.is_absolute():
            search_paths.append(candidate)
        else:
            search_paths.append(Path.cwd() / candidate)
            search_paths.append(repo_root / candidate)

        seen = set()
        for path in search_paths:
            normalized = str(path)
            if normalized in seen:
                continue
            seen.add(normalized)
            if path.exists():
                return path.resolve()

        return None

    def _prepare_mlx_ref_audio(self, ref_audio: Any) -> Any:
        if self.backend != "mlx" or ref_audio is None:
            return ref_audio

        if not isinstance(ref_audio, (str, Path)):
            return ref_audio

        resolved_path = self._resolve_audio_path(ref_audio)
        if resolved_path is None:
            raise FileNotFoundError(
                "Qwen3-TTS on Apple Silicon requires qwen3_tts_ref_audio to point to "
                f"a readable audio file. Got: {ref_audio!r}"
            )

        cache_key = str(resolved_path)
        cached_path = self._mlx_ref_audio_cache.get(cache_key)
        if cached_path and Path(cached_path).exists():
            return cached_path

        try:
            import soundfile as sf
            from scipy.signal import resample_poly

            waveform, sample_rate = sf.read(str(resolved_path), always_2d=False, dtype="float32")
            waveform = np.asarray(waveform, dtype=np.float32)
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
            target_sample_rate = getattr(self.model, "sample_rate", 24000)
            if sample_rate != target_sample_rate:
                gcd = np.gcd(int(sample_rate), int(target_sample_rate))
                waveform = resample_poly(
                    waveform,
                    up=int(target_sample_rate) // gcd,
                    down=int(sample_rate) // gcd,
                )
                sample_rate = target_sample_rate

            with tempfile.NamedTemporaryFile(
                prefix="qwen3_ref_",
                suffix=".wav",
                delete=False,
            ) as temp_file:
                normalized_path = temp_file.name

            sf.write(
                normalized_path,
                waveform,
                sample_rate,
                format="WAV",
                subtype="PCM_16",
            )
        except Exception as e:
            raise RuntimeError(f"Failed to normalize Qwen3-TTS reference audio {resolved_path}: {e}") from e

        self._mlx_ref_audio_cache[cache_key] = normalized_path
        self._mlx_temp_ref_audio_files.add(normalized_path)
        return normalized_path

    def _apply_session_voice_override(
        self,
        model_type: str,
        runtime_config: RuntimeConfig | None = None,
        response: RealtimeResponseCreateParams | None = None,
    ) -> None:
        session_voice: Optional[str] = None
        if response and response.audio and response.audio.output:
            resp_voice = response.audio.output.voice
            session_voice = str(resp_voice) if resp_voice else None
        if not session_voice and runtime_config is not None:
            audio = runtime_config.session.audio
            output = audio.output if audio is not None else None
            sess_voice = output.voice if output is not None else None
            session_voice = str(sess_voice) if sess_voice else None
        if not session_voice:
            return

        if model_type == "custom_voice":
            supported_speakers = self._supported_speakers()
            if supported_speakers is not None:
                speakers_by_lower = {speaker.lower(): speaker for speaker in supported_speakers}
                speaker = speakers_by_lower.get(session_voice.lower())
                if speaker is None:
                    supported_text = ", ".join(sorted(supported_speakers, key=str.lower)) or "unknown"
                    logger.warning(
                        "Ignoring Qwen3-TTS session voice override %r because it is not a supported "
                        "CustomVoice speaker. Supported speakers: %s",
                        session_voice,
                        supported_text,
                    )
                    return
                session_voice = speaker

            self.speaker = session_voice
            self.ref_audio = None
            return

        if self._resolve_audio_path(session_voice) is not None:
            self.ref_audio = session_voice
            return

        logger.warning(
            "Ignoring Qwen3-TTS session voice override because it is not an audio file path: %r",
            session_voice,
        )

    def warmup(self) -> None:
        logger.info(f"Warming up {self.__class__.__name__}")

        if self.backend == "openai_api":
            return

        if self.backend == "faster_qwen3_tts":
            if self.parity_mode:
                logger.info("Qwen3-TTS parity mode enabled: skipping CUDA graph capture warmup")
            else:
                try:
                    self.model._warmup(prefill_len=100)
                except Exception as e:
                    logger.warning(f"CUDA graph capture failed: {e}")

        try:
            for _ in self._warmup_process("Hello, this is a warmup."):
                pass
            logger.info(f"{self.__class__.__name__} warmed up")
        except Exception as e:
            logger.warning(f"Warmup generation failed: {e}")

    def _model_type(self) -> str:
        if self.backend == "openai_api":
            return "openai_api"

        if self.backend == "mlx":
            config = getattr(self.model, "config", None)
            return getattr(config, "tts_model_type", None) or self._infer_model_type_from_name()

        inner = getattr(getattr(self.model, "model", None), "model", None)
        return getattr(inner, "tts_model_type", None) or self._infer_model_type_from_name()

    def _supported_speakers(self) -> list[str] | None:
        for candidate in (
            self.model,
            getattr(self.model, "model", None),
            getattr(getattr(self.model, "model", None), "model", None),
        ):
            get_speakers = getattr(candidate, "get_supported_speakers", None)
            if callable(get_speakers):
                speakers = get_speakers()
                if speakers is None:
                    return None
                return [str(speaker) for speaker in speakers if speaker]
        return None

    def _resolve_speaker(self) -> Optional[str]:
        if self.speaker:
            return self.speaker

        for candidate in (
            self.model,
            getattr(getattr(self.model, "model", None), "model", None),
        ):
            get_speakers = getattr(candidate, "get_supported_speakers", None)
            if callable(get_speakers):
                speakers = list(get_speakers() or [])
                if speakers:
                    return speakers[0]

        return None

    def _to_int16(self, audio: np.ndarray) -> np.ndarray:
        return np.clip(audio * 32768, -32768, 32767).astype(np.int16)

    def _estimate_max_new_tokens(self, text: Optional[str]) -> int:
        text = (text or "").strip()
        chunk_size = max(1, int(getattr(self, "streaming_chunk_size", 1)))
        configured_cap = max(1, int(getattr(self, "max_new_tokens", DEFAULT_QWEN3_TTS_MAX_NEW_TOKENS)))

        if not text:
            return min(configured_cap, MIN_QWEN3_TTS_UTTERANCE_TOKENS)

        word_count = len(re.findall(r"\w+", text, flags=re.UNICODE))
        char_count = len(re.sub(r"\s+", "", text))
        word_seconds = word_count / ESTIMATED_QWEN3_WORDS_PER_SECOND if word_count else 0.0
        char_seconds = char_count / ESTIMATED_QWEN3_CHARS_PER_SECOND if char_count else 0.0
        punctuation_count = sum(unicodedata.category(ch).startswith("P") for ch in text)
        punctuation_seconds = punctuation_count * QWEN3_PUNCTUATION_PAUSE_SECONDS
        estimated_seconds = max(word_seconds, char_seconds) + punctuation_seconds + QWEN3_BASE_PROMPT_SECONDS
        estimated_tokens = math.ceil(estimated_seconds * MLX_STREAMING_TOKENS_PER_SECOND * QWEN3_TOKEN_SAFETY_MARGIN)
        aligned_tokens = max(
            chunk_size,
            math.ceil(estimated_tokens / chunk_size) * chunk_size,
        )
        requested_tokens = max(MIN_QWEN3_TTS_UTTERANCE_TOKENS, aligned_tokens)
        resolved_tokens = min(configured_cap, requested_tokens)

        if resolved_tokens < requested_tokens:
            logger.warning(
                "Qwen3-TTS estimated %d codec tokens for a %d-character utterance, "
                "but max_new_tokens is capped at %d; output may still truncate.",
                requested_tokens,
                len(text),
                configured_cap,
            )

        logger.debug(
            "Qwen3-TTS using max_new_tokens=%d for utterance with %d words and %d chars",
            resolved_tokens,
            word_count,
            char_count,
        )
        return resolved_tokens

    def _warmup_process(self, llm_sentence: str) -> Iterator[bytes | np.ndarray]:
        model_type = self._model_type()
        if self.ref_audio:
            yield from self._process_voice_clone(llm_sentence)
        elif model_type == "custom_voice":
            yield from self._process_custom_voice(llm_sentence)
        elif model_type == "voice_design":
            yield from self._process_voice_design(llm_sentence)
        else:
            raise ValueError(
                "Qwen3-TTS Base model requires ref_audio for voice cloning. "
                "Provide qwen3_tts_ref_audio or use a CustomVoice/VoiceDesign model."
            )

    def _resample_to_pipeline_sr(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if sr == PIPELINE_SR:
            return audio
        from scipy.signal import resample_poly

        gcd = np.gcd(PIPELINE_SR, sr)
        return resample_poly(audio, up=PIPELINE_SR // gcd, down=sr // gcd)

    def _prepare_audio_chunk(self, item: Any) -> tuple[np.ndarray | None, int | None]:
        if isinstance(item, tuple):
            audio_chunk, sr, _timing = item
            return np.asarray(audio_chunk, dtype=np.float32), sr

        audio = getattr(item, "audio", None)
        if audio is None:
            return None, None

        audio_chunk = np.asarray(audio, dtype=np.float32).squeeze()
        sr = getattr(item, "sample_rate", None) or PIPELINE_SR
        return audio_chunk, sr

    def _stream(self, gen: Any, label: str) -> Iterator[bytes | np.ndarray]:
        """Common streaming loop: log TTFA and RTF, yield int16 chunks."""
        cancel_gen = self.cancel_scope.generation if self.cancel_scope else None
        start = perf_counter()
        total_samples = 0
        first_chunk = True
        found_speech = False
        leftover = np.array([], dtype=np.int16)

        for item in gen:
            if cancel_gen is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(cancel_gen):
                logger.info("TTS generation cancelled (interruption)")
                return

            audio_chunk, sr = self._prepare_audio_chunk(item)
            if audio_chunk is None or sr is None or audio_chunk.size == 0:
                continue

            if first_chunk:
                logger.info(f"Qwen3-TTS TTFA: {perf_counter() - start:.2f}s ({label})")
                first_chunk = False

            audio_chunk = self._resample_to_pipeline_sr(audio_chunk, sr)
            audio_chunk = self._to_int16(audio_chunk)

            # Trim the initial silent ramp-up, but keep enough preroll to avoid
            # shaving soft initial phonemes at the start of the utterance.
            if not found_speech:
                threshold = int(32768 * 0.01)
                above = np.abs(audio_chunk) > threshold
                if not np.any(above):
                    continue
                start_idx = max(0, int(np.argmax(above)) - int(PIPELINE_SR * 0.040))
                audio_chunk = audio_chunk[start_idx:]
                found_speech = True

            audio_chunk = np.concatenate([leftover, audio_chunk])

            n = (len(audio_chunk) // self.blocksize) * self.blocksize
            for i in range(0, n, self.blocksize):
                yield audio_chunk[i : i + self.blocksize]
                total_samples += self.blocksize
            leftover = audio_chunk[n:]

        if len(leftover) > 0:
            chunk = np.pad(leftover, (0, self.blocksize - len(leftover)))
            yield chunk
            total_samples += len(leftover)

        generation_time = perf_counter() - start
        audio_duration = total_samples / PIPELINE_SR
        rtf = generation_time / audio_duration if audio_duration > 0 else 0
        logger.info(
            f"Qwen3-TTS generated {audio_duration:.2f}s audio in {generation_time:.2f}s (RTF: {rtf:.2f}, {label})"
        )

    def _resolve_api_voice(
        self, runtime_config: RuntimeConfig | None, response: RealtimeResponseCreateParams | None
    ) -> str:
        if response and response.audio and response.audio.output and response.audio.output.voice:
            return str(response.audio.output.voice)
        if runtime_config is not None:
            audio = runtime_config.session.audio
            output = audio.output if audio is not None else None
            if output is not None and output.voice:
                return str(output.voice)
        return self.api_voice

    def _openai_api_payload(
        self,
        text: str,
        voice: str,
        language: str = "Auto",
        *,
        stream: bool = True,
        model: str | None = None,
        tts_tuning: dict[str, Any] | None = None,
        clone_snapshot: dict[str, Any] | None = None,
        expected_engine_epoch: int | None = None,
        expected_supervisor_instance_id: str | None = None,
        response_format: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "model": model or self.api_model,
            "input": text,
            "voice": voice,
            "response_format": response_format or getattr(self, "api_response_format", "pcm"),
            "stream": stream,
            "language": language,
        }
        if tts_tuning is not None:
            payload["tuning"] = tts_tuning
        if clone_snapshot is not None:
            # The candidate consumes the frozen bytes rather than reopening a
            # mutable clone ID on every phrase. Other providers never receive
            # this candidate-private field.
            payload["clone_snapshot"] = clone_snapshot
        if expected_engine_epoch is not None:
            # Candidate-private admission guard.  It is deliberately omitted
            # for Faster/Groxaxo and stripped by the candidate supervisor
            # before it reaches the audio.cpp engine.
            payload["expected_engine_epoch"] = expected_engine_epoch
        if expected_supervisor_instance_id is not None:
            payload["expected_supervisor_instance_id"] = expected_supervisor_instance_id
        return payload

    def _openai_api_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = getattr(self, "api_key", None)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _openai_api_model_status_url(self) -> str:
        return f"{self.api_base_url}/backend/models"

    def _openai_api_model_switch_url(self) -> str:
        return f"{self.api_base_url}/backend/models/switch"

    def _openai_api_health_url(self) -> str:
        base = self.api_base_url.removesuffix("/v1")
        return f"{base}/health"

    def _openai_api_backend_model_ready(self, status: dict[str, Any], model_key: str | None = None) -> bool:
        expected = model_key or self.api_backend_model
        loaded_models = status.get("loaded_models") or []
        return (
            status.get("current") == expected
            and status.get("state") == "loaded"
            and expected in loaded_models
        )

    def _ensure_openai_api_backend_model(self) -> None:
        if not self.api_backend_model:
            return

        status_url = self._openai_api_model_status_url()
        switch_url = self._openai_api_model_switch_url()
        with httpx.Client(timeout=self.api_timeout) as client:
            status_response = client.get(status_url, headers=self._openai_api_headers())
            if status_response.status_code == 404:
                health_response = client.get(self._openai_api_health_url(), headers=self._openai_api_headers())
                health_response.raise_for_status()
                health = health_response.json()
                if health.get("backend") == "faster-qwen3-tts":
                    capabilities = health.get("capabilities") or {}
                    streaming_pcm = bool(capabilities.get("native_pcm_streaming")) and (
                        capabilities.get("stream_response_format") == "pcm"
                    )
                    self.api_streaming_supported = streaming_pcm
                    self.api_response_format = "pcm" if streaming_pcm else "wav"
                    sample_rate = capabilities.get("sample_rate")
                    if streaming_pcm and isinstance(sample_rate, int) and sample_rate > 0:
                        self.api_sample_rate = sample_rate
                    logger.info(
                        "FasterQwen3TTS fixed %s backend detected; native PCM streaming=%s",
                        self.api_backend_model,
                        streaming_pcm,
                    )
                    return
            status_response.raise_for_status()
            status = status_response.json()
            if self._openai_api_backend_model_ready(status):
                logger.info("Qwen3-TTS API backend model ready: %s", self.api_backend_model)
                return

            available = status.get("available") or []
            if self.api_backend_model not in available:
                raise RuntimeError(
                    "Qwen3-TTS API backend does not list required model "
                    f"{self.api_backend_model!r}; available models: {available!r}"
                )

            logger.info("Switching Qwen3-TTS API backend model to %s", self.api_backend_model)
            switch_response = client.post(
                switch_url,
                headers=self._openai_api_headers(),
                json={"model_key": self.api_backend_model},
            )
            switch_response.raise_for_status()
            final_status = switch_response.json()
            if not self._openai_api_backend_model_ready(final_status):
                status_response = client.get(status_url, headers=self._openai_api_headers())
                status_response.raise_for_status()
                final_status = status_response.json()
            if not self._openai_api_backend_model_ready(final_status):
                raise RuntimeError(
                    "Qwen3-TTS API backend did not become ready for "
                    f"{self.api_backend_model!r}; status: {final_status!r}"
                )

    def _resolve_api_voice_library_dir(self, configured: str | None) -> Path:
        value = configured or os.getenv("VOICE_LIBRARY_DIR") or DEFAULT_OPENAI_API_VOICE_LIBRARY_DIR
        return Path(value).expanduser()

    def _api_voice_for_backend(self, voice: str) -> str:
        if not voice.startswith("clone:"):
            return voice
        profile_id = voice.removeprefix("clone:").strip()
        meta_path = self.api_voice_library_dir / "profiles" / profile_id / "meta.json"
        if not meta_path.exists():
            return voice
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to read Qwen3-TTS voice profile metadata: %s", meta_path)
            return voice
        name = str(data.get("name") or "").strip()
        return f"clone:{name}" if name else voice

    def _resolve_api_provider(
        self,
        runtime_config: RuntimeConfig | None,
        *,
        local_pipeline: dict[str, Any] | None = None,
    ) -> tuple[str, str, str]:
        selected = "faster"
        if local_pipeline is None and runtime_config is not None:
            local_pipeline = runtime_config.local_pipeline
        if isinstance(local_pipeline, dict):
            selected = str(local_pipeline.get("tts_backend") or "faster").lower()
        if selected == "faster":
            base_url = getattr(self, "api_base_url", DEFAULT_OPENAI_API_BASE_URL)
            try:
                response = httpx.get(
                    f"{base_url.removesuffix('/v1')}/health",
                    headers=self._openai_api_headers(),
                    timeout=2.0,
                )
                response.raise_for_status()
                health = response.json()
            except Exception as exc:
                raise RuntimeError(
                    "FasterQwen3TTS is selected but is currently unreachable on port 8881. "
                    "The pipeline remains running; start the service or choose another TTS backend."
                ) from exc
            capabilities = health.get("capabilities") or {}
            if not health.get("model_loaded") or not capabilities.get("clone_only"):
                raise RuntimeError("FasterQwen3TTS is reachable but its Base clone model is not ready.")
            self.api_streaming_supported = bool(capabilities.get("native_pcm_streaming"))
            self.api_response_format = "pcm" if self.api_streaming_supported else "wav"
            sample_rate = capabilities.get("sample_rate")
            if isinstance(sample_rate, int) and sample_rate > 0:
                self.api_sample_rate = sample_rate
            return (
                selected,
                base_url,
                getattr(self, "api_backend_model", DEFAULT_OPENAI_API_BACKEND_MODEL),
            )
        if selected in {"qwen3tts-audiocpp", "audio-cpp"}:
            # The candidate's root health document is the durable supervisor
            # contract.  It carries the monotonic engineEpoch alongside its
            # resident Base model; do not depend on a UI/proxy-only model route.
            status_url = f"{str(self.audio_cpp_api_base_url).removesuffix('/v1')}/health"
            try:
                response = httpx.get(status_url, headers=self._openai_api_headers(), timeout=2.0)
                response.raise_for_status()
                status = response.json()
            except Exception as exc:
                raise RuntimeError(
                    "Qwen3TTS audio.cpp is unreachable on port 8890. Load a compatible Base model in its Voice Studio first."
                ) from exc
            backend = status.get("backend") if isinstance(status.get("backend"), dict) else {}
            current = str(
                status.get("current")
                or status.get("activeModel")
                or backend.get("current_model_key")
                or backend.get("model_id")
                or ""
            )
            loaded = status.get("loaded_models") or backend.get("loaded_models") or []
            runtime = status.get("runtime") or backend.get("runtime") or {}
            loaded_state = str(status.get("state") or runtime.get("state") or "")
            progressive = bool(runtime.get("progressive_phrase_pcm"))
            native_incremental = bool(runtime.get("native_incremental_pcm"))
            lifecycle_epoch = self._audio_cpp_native_lifecycle_epoch(status, current)
            lifecycle_instance = status.get("supervisorInstanceId")
            if (
                lifecycle_epoch is None
                or not isinstance(lifecycle_instance, str)
                or not lifecycle_instance.strip()
            ):
                raise RuntimeError(
                    "Qwen3TTS audio.cpp does not expose its supervisor lifecycle identity. "
                    "Rebuild the isolated candidate before using it for realtime speech."
                )
            if (
                loaded_state != "loaded"
                or current not in loaded
                or not current.endswith("base-bf16")
                or not (native_incremental or progressive)
            ):
                raise RuntimeError(
                    "Qwen3TTS audio.cpp requires one loaded Base BF16 model and a PCM delivery capability. "
                    "Use its Voice Studio to load and validate the selected model first."
                )
            # Reset mutable Faster-derived transport fields on every provider
            # resolve.  A validated native candidate streams the engine's PCM
            # chunks directly; older candidates retain completed phrase PCM as
            # the explicit rollback path.
            self.api_streaming_supported = native_incremental
            self.api_candidate_streaming_mode = (
                "native_incremental_pcm" if native_incremental else "buffered_phrase"
            )
            self.api_response_format = "pcm"
            sample_rate = runtime.get("sample_rate")
            self.api_sample_rate = int(sample_rate) if isinstance(sample_rate, int) and sample_rate > 0 else 24000
            self._observe_audio_cpp_native_lifecycle_epoch(
                self.audio_cpp_api_base_url, current, status
            )
            return "qwen3tts-audiocpp", self.audio_cpp_api_base_url, current
        if selected != "groxaxo":
            raise RuntimeError(f"Unsupported TTS backend selection: {selected!r}")

        groxaxo_api_base_url = getattr(self, "groxaxo_api_base_url", DEFAULT_GROXAXO_API_BASE_URL)
        status_url = f"{groxaxo_api_base_url}/backend/models"
        try:
            response = httpx.get(status_url, headers=self._openai_api_headers(), timeout=2.0)
            response.raise_for_status()
            status = response.json()
        except Exception as exc:
            raise RuntimeError(
                "Groxaxo TTS is selected but is unreachable on port 8882. "
                "Start it and load a Base model in Voice Studio before connecting."
            ) from exc
        current = str(status.get("current") or "")
        loaded = status.get("loaded_models") or []
        if status.get("state") != "loaded" or current not in loaded or not current.endswith("B-Base"):
            raise RuntimeError(
                "Groxaxo TTS is selected but no 0.6B-Base or 1.7B-Base model is loaded in Voice Studio."
            )
        return selected, groxaxo_api_base_url, current

    @staticmethod
    def _api_language_name(language: str | None, text: str = "") -> str:
        if language is not None and str(language).strip().lower() == "auto":
            # Explicit Auto is conversation language policy, not missing
            # metadata. Preserve it for mixed-language/code-switched text so a
            # clone profile's stored reference language cannot silently pin the
            # current response.
            return "Auto"
        if not language:
            if any("\u0400" <= char <= "\u052f" for char in text):
                return "Russian"
            if any("\u3040" <= char <= "\u30ff" for char in text):
                return "Japanese"
            if any("\uac00" <= char <= "\ud7af" for char in text):
                return "Korean"
            if any("\u4e00" <= char <= "\u9fff" for char in text):
                return "Chinese"
            return "Auto"
        normalized = str(language).strip().replace("_", "-").lower()
        mapped = QWEN3_LANGUAGE_ALIASES.get(normalized, normalized)
        supported = {
            "chinese", "english", "japanese", "korean", "german",
            "french", "russian", "portuguese", "spanish", "italian",
        }
        return mapped.title() if mapped in supported else "Auto"

    @staticmethod
    def _provider_auto_language_supported(provider: str) -> bool:
        """Advertise Auto only where its clone path is verified to preserve it."""

        return str(provider).strip().lower() == "qwen3tts-audiocpp"

    @staticmethod
    def _runaway_budget_s(text: str) -> float:
        words = len(re.findall(r"\w+", text, flags=re.UNICODE))
        chars = len(re.sub(r"\s+", "", text))
        estimated = max(words / 2.6 if words else 0.0, chars / 14.0 if chars else 0.0)
        return min(60.0, max(12.0, 3.0 * estimated + 5.0))

    @staticmethod
    def _audio_cpp_native_lifecycle_epoch(
        status: dict[str, Any], model: str
    ) -> str | None:
        # `engineEpoch` is owned by the supervisor and changes even when the
        # same Base model is unloaded/reloaded. Event snapshots cannot provide
        # the exact integer expected by the per-request generation guard, so a
        # legacy candidate without this canonical value must fail closed.
        raw_epoch = status.get("engineEpoch")
        if isinstance(raw_epoch, int) and not isinstance(raw_epoch, bool) and raw_epoch >= 0:
            return str(raw_epoch)
        return None

    def _observe_audio_cpp_native_lifecycle_epoch(
        self, base_url: str, model: str, status: dict[str, Any]
    ) -> None:
        observed = self._audio_cpp_native_lifecycle_epoch(status, model)
        raw_instance = status.get("supervisorInstanceId")
        instance_id = str(raw_instance).strip() if isinstance(raw_instance, str) else None
        if not instance_id:
            instance_id = None
        base_key = (str(base_url).rstrip("/"), str(model))
        epochs = getattr(self, "_audio_cpp_native_lifecycle_epochs", None)
        if epochs is None:
            epochs = {}
            self._audio_cpp_native_lifecycle_epochs = epochs
        previous = epochs.get(base_key)
        epochs[base_key] = observed
        instances = getattr(self, "_audio_cpp_native_lifecycle_instances", None)
        if instances is None:
            instances = {}
            self._audio_cpp_native_lifecycle_instances = instances
        if observed is None or instance_id is None:
            # A supervisor downgrade/restart can briefly expose a legacy status
            # after this handler cached a canonical pair. Clear both halves and
            # every warm-stream key immediately; retaining the old values would
            # let a legacy endpoint ignore expected identity fields and reuse a
            # stale engine lifecycle.
            epochs.pop(base_key, None)
            instances.pop(base_key, None)
            warm_streams = self._audio_cpp_native_warm_streams_state()
            warm_streams.difference_update(
                key for key in tuple(warm_streams) if key[:2] == base_key
            )
            return
        previous_instance = instances.get(base_key)
        instances[base_key] = instance_id
        if previous is not None and (previous != observed or previous_instance != instance_id):
            warm_streams = self._audio_cpp_native_warm_streams_state()
            warm_streams.difference_update(
                key for key in tuple(warm_streams) if key[:2] == base_key
            )

    def _audio_cpp_native_stream_key(
        self, base_url: str | None, model: str | None
    ) -> tuple[str, str, str]:
        base_key = (
            str(base_url or getattr(self, "api_base_url", "")).rstrip("/"),
            str(model or ""),
        )
        epochs = getattr(self, "_audio_cpp_native_lifecycle_epochs", None) or {}
        instances = getattr(self, "_audio_cpp_native_lifecycle_instances", None) or {}
        epoch = epochs.get(base_key, "unobserved")
        instance_id = instances.get(base_key)
        lifecycle = f"{instance_id}:{epoch}" if instance_id else epoch
        return (*base_key, lifecycle)

    def _audio_cpp_native_warm_streams_state(self) -> set[tuple[str, str, str]]:
        warm_streams = getattr(self, "_audio_cpp_native_warm_streams", None)
        if warm_streams is None:
            warm_streams = set()
            self._audio_cpp_native_warm_streams = warm_streams
        return warm_streams

    def _process_openai_api(
        self,
        text: str,
        voice: str,
        *,
        language: str = "Auto",
        base_url: str | None = None,
        model: str | None = None,
        generation: int | None = None,
        progressive_buffered: bool = False,
        native_candidate: bool = False,
        tts_tuning: dict[str, Any] | None = None,
        clone_snapshot: dict[str, Any] | None = None,
        expected_engine_epoch: int | None = None,
        expected_supervisor_instance_id: str | None = None,
        response_format: str | None = None,
        source_sample_rate: int | None = None,
    ) -> Iterator[np.ndarray]:
        private_clone = progressive_buffered or native_candidate
        voices = [voice if private_clone else self._api_voice_for_backend(voice)]
        # audio.cpp owns a private clone library.  Retrying the configured
        # Faster fallback would silently substitute a foreign clone and makes
        # a selected-candidate failure impossible to diagnose.
        if not private_clone and self.api_fallback_voice and self.api_fallback_voice not in voices:
            voices.append(self._api_voice_for_backend(self.api_fallback_voice))
        last_error: Exception | None = None
        for candidate_voice in voices:
            emitted_audio = False
            try:
                stream_kwargs = {
                    "language": language,
                    "base_url": base_url,
                    "model": model,
                    "generation": generation,
                    "progressive_buffered": progressive_buffered,
                }
                # Preserve the established mock/custom-subclass call contract
                # for non-candidate providers.  Candidate tuning is optional
                # and must not become an unexpected None keyword argument.
                if tts_tuning is not None:
                    stream_kwargs["tts_tuning"] = tts_tuning
                if clone_snapshot is not None:
                    stream_kwargs["clone_snapshot"] = clone_snapshot
                if expected_engine_epoch is not None:
                    stream_kwargs["expected_engine_epoch"] = expected_engine_epoch
                if expected_supervisor_instance_id is not None:
                    stream_kwargs["expected_supervisor_instance_id"] = expected_supervisor_instance_id
                if response_format is not None:
                    stream_kwargs["response_format"] = response_format
                if source_sample_rate is not None:
                    stream_kwargs["source_sample_rate"] = source_sample_rate
                if native_candidate:
                    stream_kwargs["native_candidate"] = True
                self._last_streaming_mode = (
                    "native_incremental_pcm"
                    if native_candidate
                    else "buffered_phrase"
                    if progressive_buffered
                    else "provider_default"
                )
                for audio_chunk in self._stream_openai_api_voice(text, candidate_voice, **stream_kwargs):
                    emitted_audio = True
                    yield audio_chunk
                return
            except TTSRunawayError:
                raise
            except httpx.ReadTimeout as exc:
                if native_candidate and not emitted_audio:
                    self._last_streaming_mode = "native_failed"
                    logger.warning(
                        "audio.cpp native PCM timed out before first playback chunk; "
                        "buffered fallback remains an explicit user-selected mode"
                    )
                if native_candidate or progressive_buffered:
                    raise TimeoutError("audio.cpp produced no data before its transport deadline") from exc
                raise TTSRunawayError("TTS stream produced no data before its latency budget expired") from exc
            except Exception as exc:
                if native_candidate and not emitted_audio:
                    self._last_streaming_mode = "native_failed"
                    logger.warning(
                        "audio.cpp native PCM failed before first playback chunk; "
                        "buffered fallback remains an explicit user-selected mode: %s",
                        exc,
                    )
                last_error = exc
                logger.warning("OpenAI-compatible Qwen3-TTS request failed for voice %s: %s", candidate_voice, exc)
        if last_error is not None:
            raise last_error

    def _stream_openai_api_voice(
        self,
        text: str,
        voice: str,
        *,
        language: str = "Auto",
        base_url: str | None = None,
        model: str | None = None,
        generation: int | None = None,
        progressive_buffered: bool = False,
        tts_tuning: dict[str, Any] | None = None,
        clone_snapshot: dict[str, Any] | None = None,
        expected_engine_epoch: int | None = None,
        expected_supervisor_instance_id: str | None = None,
        native_candidate: bool = False,
        response_format: str | None = None,
        source_sample_rate: int | None = None,
    ) -> Iterator[np.ndarray]:
        url = f"{base_url or self.api_base_url}/audio/speech"
        start = perf_counter()
        # audio.cpp exposes model-native PCM16 at 24 kHz.  Retain that source
        # clock for both its incremental relay and completed-phrase fallback;
        # Faster/Groxaxo retain the established 16 kHz pipeline transport.
        # The realtime router later performs an explicit, rate-aware conversion
        # only when a client has negotiated a different output rate.
        preserve_provider_rate = bool(native_candidate or progressive_buffered)
        self._last_tts_outcome = {}
        effective_response_format = response_format or getattr(self, "api_response_format", "pcm")
        provider_sample_rate = (
            int(source_sample_rate)
            if isinstance(source_sample_rate, int) and source_sample_rate > 0
            else int(self.api_sample_rate)
        )
        stream_sample_rate = provider_sample_rate if preserve_provider_rate else PIPELINE_SR
        # Buffered audio.cpp fallback can legitimately exceed Faster's short
        # realtime stream budget. Every transport remains cancellation-owned so
        # Stop/barge-in can release it promptly. A native audio.cpp engine/model
        # receives one bounded cold first-PCM allowance. The first complete PCM
        # sample marks only that endpoint/model warm; later requests retain the
        # normal 12-60 second budget used before this exception was introduced.
        normal_runaway_budget_s = self._runaway_budget_s(text)
        native_stream_key = (
            self._audio_cpp_native_stream_key(base_url, model)
            if native_candidate and not progressive_buffered
            else None
        )
        native_cold_start = bool(
            native_stream_key is not None
            and native_stream_key not in self._audio_cpp_native_warm_streams_state()
        )
        runaway_budget_s = (
            180.0
            if progressive_buffered
            else max(AUDIO_CPP_NATIVE_COLD_FIRST_PCM_BUDGET_S, normal_runaway_budget_s)
            if native_cold_start
            else normal_runaway_budget_s
        )
        # Slow processing and excessive generated audio are separate failures.
        # Keep the established first-header budget, but do not apply its cold
        # allowance or buffered timeout as an audio-length permission.
        wall_budget_s = 180.0 if preserve_provider_rate else runaway_budget_s
        audio_budget_s = normal_runaway_budget_s
        total_samples = 0
        pending_bytes = b""
        pending_samples = np.array([], dtype=np.int16)
        first_chunk = True
        # A native capability advertisement is merely a prerequisite. The
        # candidate supervisor proves incrementality at its engine-owned SSE
        # event boundary before committing response headers. HTTP read
        # boundaries below may legally split or coalesce those engine deltas,
        # so they must never be counted as native-stream evidence.
        native_proven = not native_candidate
        request_timeout = httpx.Timeout(runaway_budget_s, connect=min(5.0, runaway_budget_s))
        response = CancellableAsyncByteStream(
            "POST",
            url,
            headers=self._openai_api_headers(),
            json_body=self._openai_api_payload(
                text,
                voice,
                language,
                stream=not progressive_buffered,
                model=model,
                tts_tuning=tts_tuning,
                clone_snapshot=clone_snapshot,
                expected_engine_epoch=expected_engine_epoch,
                expected_supervisor_instance_id=expected_supervisor_instance_id,
                response_format=effective_response_format,
            ),
            timeout=request_timeout,
        )
        with self._active_response_lock:
            self._active_response = response
        outcome_checked = False
        try:
            response.wait_for_headers()
            self._last_tts_response_headers = dict(getattr(response, "response_headers", {}) or {})
            normalized_headers = {
                str(key).strip().lower(): str(value).strip()
                for key, value in self._last_tts_response_headers.items()
            }
            if expected_engine_epoch is not None or expected_supervisor_instance_id is not None:
                if expected_engine_epoch is None or not expected_supervisor_instance_id:
                    raise RuntimeError("audio.cpp candidate lifecycle expectation is incomplete")
                echoed_epoch = normalized_headers.get("x-tts-engine-epoch")
                echoed_instance = normalized_headers.get("x-tts-supervisor-instance-id")
                if echoed_epoch != str(expected_engine_epoch) or echoed_instance != str(
                    expected_supervisor_instance_id
                ):
                    raise RuntimeError(
                        "audio.cpp candidate lifecycle changed between health admission and synthesis "
                        f"(expected {expected_supervisor_instance_id}:{expected_engine_epoch}, "
                        f"received {echoed_instance or 'missing'}:{echoed_epoch or 'missing'})"
                    )
            if native_candidate:
                streaming_header = next(
                    (
                        str(value).strip().lower()
                        for key, value in normalized_headers.items()
                        if key == "x-tts-streaming-mode"
                    ),
                    "",
                )
                if streaming_header != "native-incremental-pcm":
                    raise RuntimeError(
                        "audio.cpp native PCM response did not prove "
                        "X-TTS-Streaming-Mode: native-incremental-pcm"
                    )
                engine_chunk_proof = normalized_headers.get("x-tts-native-engine-chunk-proof", "")
                if engine_chunk_proof != "two-distinct-sse-delta-events":
                    raise RuntimeError(
                        "audio.cpp native PCM response did not prove two engine-owned "
                        "speech.audio.delta events before headers"
                    )
                native_proven = True
            if effective_response_format != "pcm":
                encoded_parts: list[bytes] = []
                for chunk in response.iter_bytes():
                    if generation is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(generation):
                        response.close()
                        return
                    if perf_counter() - start > wall_budget_s:
                        response.close()
                        if not preserve_provider_rate:
                            raise TTSRunawayError(f"TTS stream exceeded {wall_budget_s:.1f}s budget")
                        raise TimeoutError(f"TTS processing exceeded {wall_budget_s:.1f}s wall-clock budget")
                    encoded_parts.append(chunk)
                if preserve_provider_rate:
                    outcome_checked = True
                    self._check_candidate_outcome(base_url or self.api_base_url, normalized_headers)
                encoded_audio = b"".join(encoded_parts)
                for out in self._stream_encoded_openai_api_audio(
                    encoded_audio,
                    voice,
                    preserve_provider_rate=preserve_provider_rate,
                    source_sample_rate=stream_sample_rate,
                ):
                    total_samples += len(out)
                    yield out
                return
            for chunk in response.iter_bytes():
                if generation is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(generation):
                    response.close()
                    return
                if perf_counter() - start > wall_budget_s:
                    response.close()
                    if not preserve_provider_rate:
                        raise TTSRunawayError(f"TTS stream exceeded {wall_budget_s:.1f}s budget")
                    raise TimeoutError(f"TTS processing exceeded {wall_budget_s:.1f}s wall-clock budget")
                prospective_samples = (total_samples + len(pending_samples) + (len(pending_bytes) + len(chunk)) // 2
                                       if preserve_provider_rate else total_samples)
                if prospective_samples / stream_sample_rate > audio_budget_s:
                    response.close()
                    raise TTSRunawayError(f"TTS output exceeded {audio_budget_s:.1f}s generated-audio limit")
                if not chunk:
                    continue
                chunks_to_decode = [chunk]
                for pcm_chunk in chunks_to_decode:
                    if first_chunk:
                        logger.info("Qwen3-TTS API TTFA: %.2fs (voice=%s)", perf_counter() - start, voice)
                        first_chunk = False
                    pending_bytes += pcm_chunk
                    even = len(pending_bytes) - (len(pending_bytes) % 2)
                    if even <= 0:
                        continue
                    pcm24 = np.frombuffer(pending_bytes[:even], dtype="<i2")
                    pending_bytes = pending_bytes[even:]
                    if native_stream_key is not None and native_proven and pcm24.size:
                        self._audio_cpp_native_warm_streams_state().add(native_stream_key)
                    pcm_out = (
                        pcm24
                        if preserve_provider_rate
                        else self._resample_to_pipeline_sr(pcm24, provider_sample_rate).astype(np.int16)
                    )
                    pending_samples = np.concatenate([pending_samples, pcm_out])
                    n = (len(pending_samples) // self.blocksize) * self.blocksize
                    for i in range(0, n, self.blocksize):
                        out = pending_samples[i : i + self.blocksize]
                        total_samples += len(out)
                        yield out
                    pending_samples = pending_samples[n:]
            if preserve_provider_rate:
                outcome_checked = True
                self._check_candidate_outcome(base_url or self.api_base_url, normalized_headers)
        except StreamCancelled:
            return
        except Exception:
            if preserve_provider_rate and not outcome_checked and not (generation is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(generation)):
                self._check_candidate_outcome(base_url or self.api_base_url,
                                              locals().get("normalized_headers", {}), require_complete=False)
            raise
        finally:
            response.close()
            with self._active_response_lock:
                if self._active_response is response:
                    self._active_response = None
        if len(pending_samples) > 0:
            # Do not extend audio.cpp's model-native PCM tail with synthetic
            # silence.  Keeping exact samples is important for 24 kHz duration
            # and continuity accounting.  Legacy provider output keeps its
            # fixed block compatibility behavior.
            out = pending_samples if preserve_provider_rate else np.pad(
                pending_samples,
                (0, self.blocksize - len(pending_samples)),
            )
            total_samples += len(pending_samples)
            yield out
        generation_time = perf_counter() - start
        audio_duration = total_samples / stream_sample_rate
        rtf = generation_time / audio_duration if audio_duration > 0 else 0
        logger.info("Qwen3-TTS API generated %.2fs audio in %.2fs (RTF: %.2f)", audio_duration, generation_time, rtf)

    def _check_candidate_outcome(self, base_url: str, headers: dict[str, str], *, require_complete: bool = True) -> None:
        """Explain a late raw-PCM failure without another synthesis or fallback."""
        request_id = headers.get("x-tts-request-id", "")
        if not request_id:
            if require_complete:
                raise RuntimeError("audio.cpp did not provide a synthesis completion identity")
            return  # Do not obscure an already raised provider/transport error.
        if not re.fullmatch(r"[0-9a-f]{32}", request_id):
            raise RuntimeError("audio.cpp returned an invalid synthesis outcome ID")
        try:
            result = httpx.get(f"{base_url.rstrip('/')}/audio/outcomes/{request_id}",
                               headers=self._openai_api_headers(), timeout=1.0)
            result.raise_for_status()
            outcome = result.json()
        except Exception as exc:
            if require_complete:
                raise RuntimeError("audio.cpp synthesis completion could not be verified") from exc
            return
        if outcome.get("requestId") != request_id:
            raise RuntimeError("audio.cpp returned a mismatched synthesis outcome")
        self._last_tts_outcome = {"candidate_request_id": request_id}
        for key, field in (("generatedFrames", "generated_codec_frames"), ("generationCap", "generation_cap_frames"),
                           ("promptMs", "engine_prompt_ms"), ("prefillMs", "engine_prefill_ms"),
                           ("talkerMs", "engine_talker_ms"), ("decodeMs", "engine_decode_ms")):
            value = outcome.get(key)
            if not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 600000:
                self._last_tts_outcome[field] = value
        if isinstance(outcome.get("eos"), bool):
            self._last_tts_outcome["engine_eos"] = outcome["eos"]
        status = outcome.get("state")
        if status == "limited":
            raise TTSRunawayError("audio.cpp stopped incomplete synthesis at its generated-audio limit")
        if status in {"error", "cancelled"} or (require_complete and status != "completed"):
            reason = outcome.get("reason")
            if reason not in {"wall-timeout", "native-stream-error", "native-preheader-error", "engine-error", "synthesis-error", "client-cancelled", "transport-closed"}:
                reason = "incomplete-synthesis"
            raise RuntimeError(f"audio.cpp synthesis failed: {reason}")

    def _stream_encoded_openai_api_audio(
        self,
        encoded_audio: bytes,
        voice: str,
        *,
        preserve_provider_rate: bool = False,
        source_sample_rate: int | None = None,
    ) -> Iterator[np.ndarray]:
        try:
            import soundfile as sf

            audio, sr = sf.read(io.BytesIO(encoded_audio), dtype="float32", always_2d=False)
        except Exception as exc:
            raise RuntimeError(f"Failed to decode Qwen3-TTS API audio response for {voice}: {exc}") from exc

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if not preserve_provider_rate:
            audio = self._resample_to_pipeline_sr(audio, int(sr))
        elif source_sample_rate is not None and int(sr) != int(source_sample_rate):
            # A candidate endpoint which advertises a different native PCM rate
            # must be normalized once here, rather than mislabeled downstream.
            gcd = np.gcd(int(sr), int(source_sample_rate))
            from scipy.signal import resample_poly

            audio = resample_poly(
                audio,
                up=int(source_sample_rate) // int(gcd),
                down=int(sr) // int(gcd),
            )
        samples = self._to_int16(audio)
        for i in range(0, len(samples), self.blocksize):
            chunk = samples[i : i + self.blocksize]
            if len(chunk) < self.blocksize and not preserve_provider_rate:
                chunk = np.pad(chunk, (0, self.blocksize - len(chunk)))
            yield chunk

    @staticmethod
    def _candidate_response_metric_detail(headers: dict[str, Any] | None) -> dict[str, Any]:
        """Return bounded, transcript-free diagnostics from candidate headers."""
        normalized = {
            str(key).strip().lower(): str(value).strip()
            for key, value in (headers or {}).items()
        }
        detail: dict[str, Any] = {}
        float_headers = {
            "reference_source_seconds": "x-tts-reference-source-seconds",
            "reference_requested_limit_seconds": "x-tts-reference-requested-limit-seconds",
            "reference_used_seconds": "x-tts-reference-used-seconds",
            "engine_first_pcm_ms": "x-tts-native-preheader-proof-first-pcm-ms",
            "native_proof_ready_ms": "x-tts-native-preheader-proof-ready-ms",
            "native_proof_hold_ms": "x-tts-native-preheader-proof-additional-wait-ms",
        }
        for field, header in float_headers.items():
            raw = normalized.get(header)
            if raw is None:
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value >= 0:
                detail[field] = round(value, 3)

        limit_raw = normalized.get("x-tts-reference-limit-applied")
        if limit_raw is None:
            limit_raw = normalized.get("x-tts-reference-truncated")
        if limit_raw in {"true", "false"}:
            applied = limit_raw == "true"
            detail["reference_limit_applied"] = applied
            # Retain the original metric name for older diagnostics clients.
            detail["reference_truncated"] = applied

        pairing = normalized.get("x-tts-reference-pairing")
        if pairing in {"full", "matched-excerpt"}:
            detail["reference_pairing"] = pairing

        delivery_mode = normalized.get("x-tts-delivery-mode")
        if delivery_mode in {
            "offline-full-decoder",
            "native-incremental-pcm",
            "buffered-fallback",
        }:
            detail["delivery_mode"] = delivery_mode

        if detail.get("reference_used_seconds", 0) > 0:
            detail["reference_used"] = True
        return detail

    @staticmethod
    def _candidate_phrase_queue_settings(current_input: TTSInput) -> tuple[int, float] | None:
        """Return the bounded audio.cpp-only look-ahead target and flush window."""
        runtime_config = current_input.runtime_config
        local_pipeline = getattr(runtime_config, "local_pipeline", None)
        if not isinstance(local_pipeline, dict) or local_pipeline.get("tts_backend") not in {
            "qwen3tts-audiocpp",
            "audio-cpp",
        }:
            return None
        tuning = local_pipeline.get("tts_tuning") if isinstance(local_pipeline, dict) else None
        return Qwen3TTSHandler._candidate_phrase_queue_settings_from_tuning(tuning)

    @staticmethod
    def _candidate_phrase_queue_settings_from_snapshot(snapshot: ResponseSynthesisSnapshot) -> tuple[int, float] | None:
        return Qwen3TTSHandler._candidate_phrase_queue_settings_from_tuning(snapshot.tuning)

    @staticmethod
    def _candidate_phrase_queue_settings_from_tuning(tuning: Any) -> tuple[int, float] | None:
        if not isinstance(tuning, dict) or tuning.get("provider") != "qwen3tts-audiocpp":
            return None
        values = tuning.get("effective")
        if not isinstance(values, dict):
            # Legacy browser payloads expose only phrase-queue fields under
            # ``resolved``. Never prefer them over an immutable effective
            # response snapshot once one is present.
            values = tuning.get("resolved")
        if not isinstance(values, dict):
            return None
        text_lookahead = values.get("text_lookahead")
        phrase_flush_ms = values.get("phrase_flush_ms")
        if (
            isinstance(text_lookahead, bool)
            or not isinstance(text_lookahead, int)
            or not 16 <= text_lookahead <= 512
            or isinstance(phrase_flush_ms, bool)
            or not isinstance(phrase_flush_ms, int)
            or not 50 <= phrase_flush_ms <= 3000
        ):
            return None
        return text_lookahead, phrase_flush_ms / 1000.0

    @staticmethod
    def _has_explicit_phrase_boundary(text: str) -> bool:
        """Return whether *text* already ends at a speakable sentence boundary.

        LLM and direct-audio handlers emit punctuation-complete text as soon as
        it is stable.  Candidate look-ahead must not hold that first phrase just
        to collect more text; the look-ahead/flush policy is only for an
        otherwise incomplete fragment.
        """

        return bool(re.search(r"[.!?\u3002\uff01\uff1f\u2026][\"'\u2019\u201d)\]}]*\s*$", text))

    @staticmethod
    def _tts_backend_for_input(item: TTSInput) -> str:
        runtime_config = item.runtime_config
        local_pipeline = getattr(runtime_config, "local_pipeline", None)
        if not isinstance(local_pipeline, dict):
            return "faster"
        provider = str(local_pipeline.get("tts_backend") or "faster").lower()
        return "qwen3tts-audiocpp" if provider == "audio-cpp" else provider

    def _input_generation_is_stale(self, item: TTSInput) -> bool:
        generation = item.cancel_generation
        cancel_scope = getattr(self, "cancel_scope", None)
        if not response_output_allowed(
            runtime_config=item.runtime_config,
            response_epoch=item.response_epoch,
            turn_id=item.turn_id,
            turn_revision=item.turn_revision,
            speculative_turns=getattr(self, "speculative_turns", None),
            cancel_scope=cancel_scope,
        ):
            return True
        generation_stale = bool(
            generation is not None
            and cancel_scope is not None
            and cancel_scope.is_stale(generation)
        )
        return generation_stale

    @staticmethod
    def _response_snapshot_key(item: TTSInput) -> tuple[Any, ...]:
        """Return the response-scoped identity, with a legacy-safe fallback."""

        if item.response_epoch is not None:
            return ("response_epoch", int(item.response_epoch))
        if item.response_id:
            return ("response_id", str(item.response_id))
        return ("legacy", item.turn_id, item.turn_revision, item.cancel_generation)

    @staticmethod
    def _configured_candidate_tuning(runtime_config: RuntimeConfig | None) -> dict[str, Any] | None:
        local_pipeline = getattr(runtime_config, "local_pipeline", None)
        if not isinstance(local_pipeline, dict) or local_pipeline.get("tts_backend") != "qwen3tts-audiocpp":
            return None
        tuning = local_pipeline.get("tts_tuning")
        if not isinstance(tuning, dict) or tuning.get("provider") != "qwen3tts-audiocpp":
            return None
        return deepcopy(tuning)

    @staticmethod
    def _response_synthesis_config(item: TTSInput) -> dict[str, Any] | None:
        """Return the admission-time TTS settings for this response epoch."""
        runtime_config = item.runtime_config
        snapshots = getattr(runtime_config, "response_synthesis_configs", None)
        if not isinstance(snapshots, dict) or item.response_epoch is None:
            return None
        snapshot = snapshots.get(item.response_epoch)
        return deepcopy(snapshot) if isinstance(snapshot, dict) else None

    @staticmethod
    def _configured_seed(tuning: dict[str, Any] | None) -> int | None:
        if not isinstance(tuning, dict):
            return None
        # Presence matters: an explicit temporary null/blank/-1 asks for one
        # response-random seed and must not fall through to a profile seed.
        # New effective snapshots precede legacy resolved values, while an
        # explicit session override wins over both.
        for values in (tuning.get("overrides"), tuning.get("effective"), tuning.get("resolved")):
            if not isinstance(values, dict) or "seed" not in values:
                continue
            value = values.get("seed")
            if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 0xFFFFFFFF:
                return value
            return None
        return None

    @classmethod
    def _freeze_tuning_seed(cls, tuning: dict[str, Any] | None) -> tuple[dict[str, Any] | None, int, str]:
        """Copy candidate tuning and make its sampler seed response-stable."""

        configured_seed = cls._configured_seed(tuning)
        seed = configured_seed if configured_seed is not None else randbits(32)
        policy = "profile" if configured_seed is not None else "response_random"
        if tuning is None:
            return None, seed, policy
        frozen = deepcopy(tuning)
        overrides = frozen.setdefault("overrides", {})
        if isinstance(overrides, dict):
            overrides["seed"] = seed
        return frozen, seed, policy

    @staticmethod
    def _candidate_clone_content_hash(
        reference_audio: bytes,
        reference_text: str,
        excerpts: list[tuple[bytes, str]],
    ) -> str:
        """Match the candidate's length-delimited Base-clone content hash."""

        digest = sha256()

        def update(value: bytes) -> None:
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)

        update(reference_audio)
        update(reference_text.encode("utf-8"))
        for audio, text in excerpts:
            update(audio)
            update(text.encode("utf-8"))
        return digest.hexdigest()

    def _freeze_candidate_clone(
        self, endpoint: str | None, voice: str | None,
    ) -> dict[str, Any] | None:
        """Capture one candidate-private Base clone before the first phrase.

        Clone IDs name mutable profile slots.  The supervisor serves a full
        clone snapshot (reference audio, matching transcript/excerpts, content
        revision and hash) so later phrase requests use the exact first-phrase
        conditioning even if the user edits that slot in Voice Studio.
        """

        if not endpoint or not voice or not voice.startswith("clone:"):
            return None
        profile_id = voice.removeprefix("clone:").strip()
        if not profile_id:
            raise RuntimeError("audio.cpp selected clone is missing its profile id")
        try:
            response = httpx.get(
                f"{endpoint.rstrip('/')}/voices/profiles/{profile_id}",
                headers=self._openai_api_headers(),
                timeout=5.0,
            )
            response.raise_for_status()
            profile = response.json()
        except Exception as exc:
            raise RuntimeError(
                "audio.cpp could not freeze the selected clone reference; refresh the live clone inventory before speaking."
            ) from exc
        if not isinstance(profile, dict):
            raise RuntimeError("audio.cpp returned an invalid selected clone snapshot")
        content_hash = str(profile.get("content_hash") or "")
        content_revision = profile.get("content_revision")
        ref_audio = str(profile.get("ref_audio") or "")
        ref_text = str(profile.get("ref_text") or "").strip()
        raw_excerpts = profile.get("reference_excerpts") or []
        if (
            str(profile.get("id") or profile.get("profile_id") or "") != profile_id
            or not re.fullmatch(r"[0-9a-f]{64}", content_hash)
            or isinstance(content_revision, bool)
            or not isinstance(content_revision, int)
            or content_revision < 1
            or not ref_audio
            or not ref_text
            or not isinstance(raw_excerpts, list)
        ):
            raise RuntimeError("audio.cpp returned an incomplete selected clone snapshot")
        try:
            import base64

            reference_bytes = base64.b64decode(ref_audio, validate=True)
        except Exception as exc:
            raise RuntimeError("audio.cpp returned invalid frozen clone audio") from exc
        excerpts: list[dict[str, str]] = []
        hash_excerpts: list[tuple[bytes, str]] = []
        for excerpt in raw_excerpts:
            if not isinstance(excerpt, dict):
                raise RuntimeError("audio.cpp returned an invalid frozen clone excerpt")
            encoded = str(excerpt.get("ref_audio") or "")
            text = str(excerpt.get("ref_text") or "").strip()
            if not encoded or not text:
                raise RuntimeError("audio.cpp returned an incomplete frozen clone excerpt")
            try:
                audio = base64.b64decode(encoded, validate=True)
            except Exception as exc:
                raise RuntimeError("audio.cpp returned invalid frozen clone excerpt audio") from exc
            excerpts.append({"ref_audio": encoded, "ref_text": text})
            hash_excerpts.append((audio, text))
        if self._candidate_clone_content_hash(reference_bytes, ref_text, hash_excerpts) != content_hash:
            raise RuntimeError("audio.cpp selected clone content hash did not verify")
        return {
            "profile_id": profile_id,
            "content_revision": content_revision,
            "content_hash": content_hash,
            "ref_audio": ref_audio,
            "ref_text": ref_text,
            "reference_excerpts": excerpts,
        }

    @staticmethod
    def _audio_cpp_complete_effective_tuning(
        effective: Any, profile_revision: int | None
    ) -> dict[str, Any] | None:
        """Return only a complete, already-validated immutable engine policy.

        The WebSocket boundary validates modern snapshots before storing them in
        ``RuntimeConfig``.  This second structural guard keeps a legacy
        ``resolved`` phrase-queue object from being relabelled as an engine
        ``effective`` policy when callers construct ``TTSInput`` directly.
        ``resolved`` remains local queue metadata only.
        """

        if isinstance(profile_revision, bool) or not isinstance(profile_revision, int) or profile_revision < 1:
            return None
        if not isinstance(effective, dict):
            return None
        required = {
            "model",
            "clone_mode",
            "max_reference_seconds",
            "first_block_frames",
            "steady_block_frames",
            "left_context_frames",
            "text_lookahead",
            "phrase_flush_ms",
            "temperature",
            "top_k",
            "top_p",
            "repetition_penalty",
            "seed",
        }
        if set(effective) != required:
            return None
        # Built-in candidate profiles deliberately use a null model to mean
        # "the already-resident Base model".  The handler independently freezes
        # the actual health-derived resident model and lifecycle identity in the
        # response snapshot, so null is not an unpinned model choice.  Any
        # non-null model remains an exact raw Base-model ID.
        if effective.get("model") not in {
            None,
            "qwen3-tts-0.6b-base-bf16",
            "qwen3-tts-1.7b-base-bf16",
        } or effective.get("clone_mode") != "full_icl":
            return None
        integer_bounds = {
            "max_reference_seconds": (1, 30),
            "first_block_frames": (1, 300),
            "steady_block_frames": (1, 300),
            "left_context_frames": (1, 300),
            "text_lookahead": (16, 512),
            "phrase_flush_ms": (50, 3000),
            "top_k": (1, 200),
        }
        for key, (minimum, maximum) in integer_bounds.items():
            value = effective.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                return None
        for key, minimum, maximum in (
            ("temperature", 0.0, 2.0),
            ("top_p", 0.05, 1.0),
            ("repetition_penalty", 0.8, 2.0),
        ):
            value = effective.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not minimum < value <= maximum:
                return None
        seed = effective.get("seed")
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0xFFFFFFFF
        ):
            return None
        if (
            effective["steady_block_frames"] < effective["first_block_frames"]
            or effective["left_context_frames"] + effective["steady_block_frames"] > 300
        ):
            return None
        return deepcopy(effective)

    @staticmethod
    def _audio_cpp_request_tuning(snapshot: ResponseSynthesisSnapshot) -> dict[str, Any] | None:
        """Return an immutable candidate request snapshot, never live UI state.

        ``effective`` is a deep-copied, response-frozen set of supervisor
        fields.  It intentionally replaces the browser's mutable ``resolved``
        diagnostic object; only the supervisor decides which effective keys can
        reach the engine.  The generated response seed remains an ephemeral
        override and is never persisted back to the named profile.
        """

        if snapshot.provider != "qwen3tts-audiocpp":
            return None
        effective = Qwen3TTSHandler._audio_cpp_complete_effective_tuning(
            snapshot.tuning.get("effective") if isinstance(snapshot.tuning, dict) else None,
            snapshot.profile_revision,
        )
        payload: dict[str, Any] = {
            "provider": "qwen3tts-audiocpp",
            # Complete policy is frozen once before the first request, even
            # for a legacy client that needed supervisor-side resolution.
            "scope": "realtime",
            # Always send the response-frozen seed, including when no named
            # tuning profile has been applied.  Separate phrase requests then
            # remain voice-consistent without inventing an engine profile.
            "overrides": {"seed": snapshot.seed},
        }
        if not snapshot.profile_id or effective is None:
            raise RuntimeError("Qwen3TTS audio.cpp response requires a complete frozen tuning snapshot")
        payload["profile_id"] = snapshot.profile_id
        payload["profile_revision"] = snapshot.profile_revision
        payload["effective"] = effective
        tuning = snapshot.tuning
        if not isinstance(tuning, dict):
            raise RuntimeError("Qwen3TTS audio.cpp response tuning snapshot is unavailable")
        overrides = tuning.get("overrides")
        if isinstance(overrides, dict):
            # RuntimeConfig has already validated this bounded map.  Keep the
            # response seed authoritative even if the saved setting changes.
            payload["overrides"].update(deepcopy(overrides))
            payload["overrides"]["seed"] = snapshot.seed
        return payload

    def _create_response_synthesis_snapshot(self, item: TTSInput) -> ResponseSynthesisSnapshot:
        runtime_config = item.runtime_config
        admission = self._response_synthesis_config(item)
        frozen_local_pipeline = admission.get("local_pipeline") if isinstance(admission, dict) else None
        if not isinstance(frozen_local_pipeline, dict):
            frozen_local_pipeline = None
        provider = self.backend
        endpoint: str | None = None
        model: str | None = getattr(self, "api_backend_model", None)
        voice: str | None = None
        if self.backend == "openai_api":
            provider_runtime_config = runtime_config
            if frozen_local_pipeline is not None:
                provider_runtime_config = _ProviderRuntimeConfigSnapshot(
                    local_pipeline=frozen_local_pipeline,
                )
            provider, endpoint, model = self._resolve_api_provider(provider_runtime_config)
            # Resolve the configured default now as well.  Falling back later
            # would re-read a mutable handler setting during a later phrase.
            if isinstance(admission, dict) and "voice" in admission:
                frozen_voice = admission.get("voice")
                voice = str(frozen_voice) if frozen_voice else getattr(self, "api_voice", None)
            else:
                voice = self._resolve_api_voice(runtime_config, item.response) or getattr(self, "api_voice", None)
        else:
            voice = str(getattr(self, "speaker", "") or "") or None

        # Gemma determines assistant language for this answer after ownership
        # admission. Freeze the first authoritative phrase's language, not the
        # mutable runtime value left by the previous response.
        language_code = item.language_code
        api_language = self._api_language_name(language_code, item.text)
        language = (
            api_language
            if provider == "qwen3tts-audiocpp" or api_language != "Auto"
            else None
        )
        if provider == "qwen3tts-audiocpp" and isinstance(frozen_local_pipeline, dict):
            raw_tuning = frozen_local_pipeline.get("tts_tuning")
            tuning = deepcopy(raw_tuning) if isinstance(raw_tuning, dict) and raw_tuning.get("provider") == "qwen3tts-audiocpp" else None
        else:
            tuning = self._configured_candidate_tuning(runtime_config) if provider == "qwen3tts-audiocpp" else None
        profile_revision = None
        if isinstance(tuning, dict):
            candidate_revision = tuning.get("revision", tuning.get("profile_revision"))
            if isinstance(candidate_revision, int) and not isinstance(candidate_revision, bool):
                profile_revision = candidate_revision
        if provider == "qwen3tts-audiocpp":
            profile_id = tuning.get("profile_id") if isinstance(tuning, dict) else None
            effective = self._audio_cpp_complete_effective_tuning(
                tuning.get("effective") if isinstance(tuning, dict) else None,
                profile_revision,
            )
            if isinstance(profile_id, str) and profile_id and effective is not None:
                assert tuning is not None
                tuning["effective"] = effective
            else:
                # Older/nonbrowser clients may omit the revisioned profile.
                # Resolve once at response admission, never once per phrase:
                # a shared selection change cannot split an answer's voice.
                selection = {"provider": "qwen3tts-audiocpp", "scope": "realtime"}
                if isinstance(tuning, dict):
                    if isinstance(tuning.get("profile_id"), str) and tuning["profile_id"]:
                        selection["profile_id"] = tuning["profile_id"]
                    if isinstance(tuning.get("overrides"), dict):
                        selection["overrides"] = deepcopy(tuning["overrides"])
                result = httpx.post(str(endpoint or "").rstrip("/") + "/tuning/resolve",
                                    json=selection, headers=self._openai_api_headers(), timeout=2.0)
                result.raise_for_status()
                resolved_profile = result.json().get("profile")
                if not isinstance(resolved_profile, dict):
                    raise RuntimeError("Candidate tuning resolution did not return a profile")
                profile_revision = resolved_profile.get("revision")
                effective_fields = {key: resolved_profile[key] for key in (
                    "model", "clone_mode", "max_reference_seconds", "first_block_frames", "steady_block_frames",
                    "left_context_frames", "text_lookahead", "phrase_flush_ms", "temperature", "top_k", "top_p",
                    "repetition_penalty", "seed",
                ) if key in resolved_profile}
                effective = self._audio_cpp_complete_effective_tuning(effective_fields, profile_revision)
                if not effective or not isinstance(resolved_profile.get("id"), str) or not resolved_profile["id"]:
                    raise RuntimeError("Candidate tuning resolution was incomplete")
                tuning = {"provider": "qwen3tts-audiocpp", "scope": "realtime",
                          "profile_id": resolved_profile["id"], "profile_revision": profile_revision,
                          "effective": effective, "overrides": selection.get("overrides", {}),
                          "delivery_mode": (tuning or {}).get("delivery_mode", "buffered_phrase")}
        tuning, seed, seed_policy = self._freeze_tuning_seed(tuning)
        local_pipeline = frozen_local_pipeline if isinstance(frozen_local_pipeline, dict) else getattr(runtime_config, "local_pipeline", None)
        model_epoch = None
        model_instance_id = None
        if provider == "qwen3tts-audiocpp":
            lifecycle_key = (str(endpoint or "").rstrip("/"), str(model or ""))
            observed_epoch = (
                getattr(self, "_audio_cpp_native_lifecycle_epochs", None) or {}
            ).get(lifecycle_key)
            observed_instance = (
                getattr(self, "_audio_cpp_native_lifecycle_instances", None) or {}
            ).get(lifecycle_key)
            if observed_epoch is not None:
                model_epoch = observed_epoch
            if observed_instance is not None:
                model_instance_id = observed_instance
        # Faster/Groxaxo do not own this candidate-only supervisor contract.
        # Never fall back to mutable RuntimeConfig metadata for audio.cpp: an
        # old candidate without an instance nonce must fail closed instead of
        # attaching a later phrase to a restarted supervisor.
        if provider != "qwen3tts-audiocpp" and model_epoch is None and isinstance(local_pipeline, dict):
            raw_epoch = local_pipeline.get("tts_model_epoch") or local_pipeline.get("candidate_model_epoch")
            model_epoch = str(raw_epoch) if raw_epoch is not None else None
        if provider == "qwen3tts-audiocpp" and (model_epoch is None or model_instance_id is None):
            raise RuntimeError(
                "Qwen3TTS audio.cpp lifecycle identity is unavailable; reload its candidate build before synthesis."
            )
        frozen_clone = (
            self._freeze_candidate_clone(endpoint, voice)
            if provider == "qwen3tts-audiocpp"
            else None
        )
        reference_identity = "|".join(
            str(value or "")
            for value in (voice, getattr(self, "ref_audio", None), getattr(self, "ref_text", None))
        )
        clone_content_hash = str(frozen_clone["content_hash"]) if frozen_clone else None
        clone_content_revision = int(frozen_clone["content_revision"]) if frozen_clone else None
        reference_fingerprint = (
            clone_content_hash[:16]
            if clone_content_hash
            else sha256(reference_identity.encode("utf-8")).hexdigest()[:16]
            if reference_identity
            else None
        )
        requested_delivery_mode = (
            str(tuning.get("delivery_mode") or "buffered_phrase")
            if isinstance(tuning, dict) and provider == "qwen3tts-audiocpp"
            else "provider_default"
        )
        if requested_delivery_mode not in {
            "provider_default",
            "buffered_phrase",
            "native_incremental_pcm",
        }:
            raise RuntimeError("Qwen3TTS audio.cpp delivery mode is invalid")
        if (
            provider == "qwen3tts-audiocpp"
            and requested_delivery_mode == "native_incremental_pcm"
            and not bool(getattr(self, "api_streaming_supported", False))
        ):
            raise RuntimeError(
                "Qwen3TTS audio.cpp native PCM was explicitly requested but is not currently advertised. "
                "Choose buffered phrase mode or validate the isolated candidate."
            )
        # A healthy engine advertising native output does not make it the
        # default transport.  Delivery mode is frozen with this response and
        # becomes native only after an explicit validated opt-in.
        delivery_native_pcm = (
            provider == "qwen3tts-audiocpp"
            and requested_delivery_mode == "native_incremental_pcm"
            and bool(getattr(self, "api_streaming_supported", False))
        )
        delivery_streaming_mode = (
            requested_delivery_mode
            if provider == "qwen3tts-audiocpp"
            else "provider_default"
        )
        delivery_response_format = (
            str(getattr(self, "api_response_format", "pcm"))
            if self.backend == "openai_api"
            else None
        )
        raw_delivery_rate = getattr(self, "api_sample_rate", None)
        delivery_sample_rate = (
            int(raw_delivery_rate)
            if provider == "qwen3tts-audiocpp"
            and isinstance(raw_delivery_rate, int)
            and raw_delivery_rate > 0
            else None
        )
        return ResponseSynthesisSnapshot(
            key=self._response_snapshot_key(item),
            input_epoch=item.input_epoch,
            response_epoch=item.response_epoch,
            response_id=item.response_id,
            provider=provider,
            endpoint=endpoint,
            model=model,
            model_epoch=model_epoch,
            model_instance_id=model_instance_id,
            voice=voice,
            reference_fingerprint=reference_fingerprint,
            clone_content_revision=clone_content_revision,
            clone_content_hash=clone_content_hash,
            frozen_clone=deepcopy(frozen_clone) if frozen_clone is not None else None,
            profile_id=(str(tuning.get("profile_id")) if isinstance(tuning, dict) and tuning.get("profile_id") else None),
            profile_revision=profile_revision,
            tuning=tuning,
            language=language,
            seed=seed,
            seed_policy=seed_policy,
            delivery_native_pcm=delivery_native_pcm,
            delivery_streaming_mode=delivery_streaming_mode,
            delivery_response_format=delivery_response_format,
            delivery_sample_rate=delivery_sample_rate,
        )

    def _response_synthesis_snapshot(self, item: TTSInput) -> ResponseSynthesisSnapshot:
        snapshots = getattr(self, "_response_synthesis_snapshots", None)
        if snapshots is None:
            snapshots = self._response_synthesis_snapshots = {}
        key = self._response_snapshot_key(item)
        snapshot = snapshots.get(key)
        if snapshot is None:
            snapshot = self._create_response_synthesis_snapshot(item)
            snapshots[key] = snapshot
            phrase_counts = getattr(self, "_response_phrase_counts", None)
            if phrase_counts is None:
                self._response_phrase_counts = {}
            self._response_phrase_counts.setdefault(key, 0)
        return snapshot

    def _response_phrase_index(self, snapshot: ResponseSynthesisSnapshot) -> int:
        return int(getattr(self, "_response_phrase_counts", {}).get(snapshot.key, 0))

    @staticmethod
    def _candidate_expected_engine_lifecycle(
        snapshot: ResponseSynthesisSnapshot,
    ) -> tuple[int, str] | None:
        """Return only the supervisor's canonical integer lifecycle epoch.

        Legacy event snapshots remain useful for cold-start diagnostics but
        cannot be enforced by the supervisor.  The live health contract emits
        a non-negative integer and is the only value forwarded to generation.
        """

        raw_epoch = snapshot.model_epoch
        raw_instance = snapshot.model_instance_id
        if (
            not isinstance(raw_epoch, str)
            or not raw_epoch.isdecimal()
            or not isinstance(raw_instance, str)
            or not raw_instance
        ):
            return None
        value = int(raw_epoch)
        return (value, raw_instance) if value >= 0 else None

    def _start_response_phrase(self, snapshot: ResponseSynthesisSnapshot) -> int:
        counts = getattr(self, "_response_phrase_counts", None)
        if counts is None:
            counts = self._response_phrase_counts = {}
        phrase_index = int(counts.get(snapshot.key, 0))
        counts[snapshot.key] = phrase_index + 1
        return phrase_index

    def _release_response_synthesis_snapshot(self, item: TTSInput | EndOfResponse) -> None:
        key = ("response_epoch", int(item.response_epoch)) if item.response_epoch is not None else (
            ("response_id", str(item.response_id)) if item.response_id else ("legacy", item.turn_id, item.turn_revision, item.cancel_generation)
        )
        getattr(self, "_response_synthesis_snapshots", {}).pop(key, None)
        getattr(self, "_response_phrase_counts", {}).pop(key, None)
        getattr(self, "_failed_response_synthesis_keys", set()).discard(key)

    def _mark_response_synthesis_failed(self, key: tuple[Any, ...]) -> None:
        failures = getattr(self, "_failed_response_synthesis_keys", None)
        if failures is None:
            failures = self._failed_response_synthesis_keys = set()
        failures.add(key)

    @staticmethod
    def _snapshot_metric_detail(snapshot: ResponseSynthesisSnapshot, phrase_index: int) -> dict[str, Any]:
        return {
            "input_epoch": snapshot.input_epoch,
            "response_epoch": snapshot.response_epoch,
            "response_id": snapshot.response_id,
            "profile_id": snapshot.profile_id,
            "profile_revision": snapshot.profile_revision,
            "clone_fingerprint": snapshot.reference_fingerprint,
            "clone_content_revision": snapshot.clone_content_revision,
            "clone_content_hash": snapshot.clone_content_hash,
            "seed": snapshot.seed,
            "seed_policy": snapshot.seed_policy,
            "phrase_index": phrase_index,
            "model_epoch": snapshot.model_epoch,
            "model_instance_id": snapshot.model_instance_id,
            "native_pcm_streaming": snapshot.delivery_native_pcm,
            "streaming_mode": snapshot.delivery_streaming_mode,
            "response_format": snapshot.delivery_response_format,
            "source_sample_rate": snapshot.delivery_sample_rate,
        }

    def _coalesce_pending_tts_input(
        self,
        current_input: TTSInput,
        snapshot: ResponseSynthesisSnapshot | None = None,
    ) -> tuple[str, Optional[str], bool]:
        """Combine compatible text chunks before the next synthesis call.

        Existing providers retain the ready-queue-only 420-character cap.
        audio.cpp may additionally wait for its supervisor-resolved look-ahead
        target, bounded by the profile's flush window. A response/control or
        turn boundary always releases the current phrase immediately.
        """
        if not hasattr(self.queue_in, "mutex") or not hasattr(self.queue_in, "queue"):
            return current_input.text, current_input.language_code, False

        text = current_input.text
        language_code = current_input.language_code

        parts = [text.strip()] if text and text.strip() else []
        saw_end_of_response = False

        phrase_queue = (
            self._candidate_phrase_queue_settings_from_snapshot(snapshot)
            if snapshot is not None and snapshot.provider == "qwen3tts-audiocpp"
            else self._candidate_phrase_queue_settings(current_input)
        )
        is_first_phrase = snapshot is None or self._response_phrase_index(snapshot) == 0
        if phrase_queue and is_first_phrase and self._has_explicit_phrase_boundary(text):
            # Punctuation already made this phrase stable upstream.  Dispatch it
            # immediately so native synthesis can overlap the rest of the LLM
            # response instead of waiting for another phrase or the flush timer.
            return " ".join(parts).strip(), language_code, saw_end_of_response
        target_chars = phrase_queue[0] if phrase_queue else MAX_COALESCED_TTS_CHARS
        deadline = perf_counter() + phrase_queue[1] if phrase_queue else None

        with self.queue_in.mutex:
            while True:
                combined_length = len(" ".join(parts))
                if phrase_queue and combined_length >= target_chars:
                    break
                if not self.queue_in.queue:
                    if deadline is None or not hasattr(self.queue_in, "not_empty"):
                        break
                    if self._input_generation_is_stale(current_input):
                        break
                    remaining = deadline - perf_counter()
                    if remaining <= 0:
                        break
                    # Cancellation does not have to enqueue another phrase, so
                    # wake periodically and re-check the generation while an
                    # incomplete fragment is waiting on its flush deadline.
                    self.queue_in.not_empty.wait(timeout=min(remaining, 0.05))
                    continue
                next_item = self.queue_in.queue[0]
                if is_control_message(next_item, SESSION_END.kind):
                    break
                if isinstance(next_item, bytes) and next_item == PIPELINE_END:
                    break
                if isinstance(next_item, EndOfResponse):
                    saw_end_of_response = True
                    break
                if not isinstance(next_item, TTSInput):
                    break
                if snapshot is None:
                    # Preserve the legacy ready-queue isolation contract for
                    # callers that use this helper directly.
                    if current_input.turn_id != next_item.turn_id or current_input.turn_revision != next_item.turn_revision:
                        break
                    if current_input.cancel_generation != next_item.cancel_generation:
                        break
                    if self._tts_backend_for_input(current_input) != self._tts_backend_for_input(next_item):
                        break
                    if (
                        language_code is not None
                        and next_item.language_code is not None
                        and next_item.language_code != language_code
                    ):
                        break
                else:
                    if self._response_snapshot_key(current_input) != self._response_snapshot_key(next_item):
                        break
                    # A response epoch owns one accepted input.  This should
                    # normally be implied by the response identity, but keep
                    # the check explicit so an upstream ownership bug cannot
                    # combine audio from a newer accepted input into it.
                    if current_input.input_epoch != next_item.input_epoch:
                        break
                    if self._input_generation_is_stale(next_item):
                        break

                candidate = next_item.text.strip()
                # Keep every provider below the established hard request cap.
                # audio.cpp's look-ahead is a target rather than a truncation:
                # consume the complete stable phrase that crosses the target.
                if parts and combined_length + len(candidate) + 1 > MAX_COALESCED_TTS_CHARS:
                    break
                self.queue_in.queue.popleft()
                if candidate:
                    parts.append(candidate)
                if language_code is None:
                    language_code = next_item.language_code

        combined_text = " ".join(parts).strip()
        return combined_text, (snapshot.language if snapshot is not None else None) or language_code, saw_end_of_response

    def process(self, tts_input: TTSIn) -> Iterator[TTSOut]:
        speculative_turns = getattr(self, "speculative_turns", None)
        if isinstance(tts_input, EndOfResponse):
            if not response_output_allowed(
                runtime_config=tts_input.runtime_config,
                response_epoch=tts_input.response_epoch,
                turn_id=tts_input.turn_id,
                turn_revision=tts_input.turn_revision,
                speculative_turns=speculative_turns,
                cancel_scope=getattr(self, "cancel_scope", None),
            ):
                self._release_response_synthesis_snapshot(tts_input)
                return
            self._release_response_synthesis_snapshot(tts_input)
            yield AUDIO_RESPONSE_DONE
            return

        if not response_output_allowed(
            runtime_config=tts_input.runtime_config,
            response_epoch=tts_input.response_epoch,
            turn_id=tts_input.turn_id,
            turn_revision=tts_input.turn_revision,
            speculative_turns=speculative_turns,
            cancel_scope=getattr(self, "cancel_scope", None),
        ):
            logger.debug("Dropping stale TTS input for turn=%s rev=%s", tts_input.turn_id, tts_input.turn_revision)
            return
        runtime_config = tts_input.runtime_config
        response = tts_input.response
        generation = tts_input.cancel_generation
        cancel_scope = getattr(self, "cancel_scope", None)
        if generation is None and cancel_scope is not None:
            generation = getattr(cancel_scope, "generation", None)
        if self._input_generation_is_stale(tts_input):
            self._emit_metric(
                "tts",
                "cancelled_before_audio",
                tts_input,
                detail={"reason": "stale before phrase dispatch"},
            )
            return

        response_key = self._response_snapshot_key(tts_input)
        if response_key in getattr(self, "_failed_response_synthesis_keys", set()):
            self._emit_metric(
                "tts",
                "suppressed_after_failure",
                tts_input,
                detail={"reason": "an earlier phrase in this response failed"},
            )
            return

        try:
            # Provider health, canonical model lifecycle, and clone freezing all
            # happen while the first response snapshot is created. Treat a
            # failure here as terminal for this answer; otherwise a later phrase
            # can retry independently and become audible halfway through it.
            snapshot = self._response_synthesis_snapshot(tts_input)
            coalesced_text, language_code, _saw_end_of_response = self._coalesce_pending_tts_input(
                tts_input, snapshot
            )
            # A candidate response is admitted only with one complete,
            # revisioned tuning snapshot.  Keep this validation inside the
            # response-scoped setup guard so a malformed first phrase marks the
            # whole answer terminal instead of being retried independently by
            # every later phrase.
            tts_request_tuning = self._audio_cpp_request_tuning(snapshot)
        except Exception as exc:
            self._mark_response_synthesis_failed(response_key)
            logger.error("Qwen3-TTS response snapshot setup failed: %s", exc, exc_info=True)
            self._emit_metric(
                "tts",
                "failed",
                tts_input,
                detail={"error": str(exc), "reason": "response snapshot setup failed"},
            )
            return
        if self._input_generation_is_stale(tts_input):
            self._emit_metric(
                "tts",
                "cancelled_before_audio",
                tts_input,
                detail={"reason": "cancelled while waiting for phrase flush"},
            )
            self._release_response_synthesis_snapshot(tts_input)
            return
        # Only commit after the coalescing/flush window has confirmed this is
        # still the live response.  A revision can become stale while it waits.
        if speculative_turns:
            speculative_turns.commit(tts_input.turn_id, tts_input.turn_revision)

        phrase_index = self._start_response_phrase(snapshot)

        text = coalesced_text or "Hello."

        model_type = self._model_type()
        api_voice = snapshot.voice if self.backend == "openai_api" else None
        provider_name = snapshot.provider
        provider_url = snapshot.endpoint
        provider_model = snapshot.model
        requested_language = str(language_code or "Auto")
        api_language = snapshot.language or "Auto"
        language_auto_supported = self._provider_auto_language_supported(provider_name)
        effective_language = snapshot.language
        candidate_native = provider_name == "qwen3tts-audiocpp" and snapshot.delivery_native_pcm
        source_sample_rate = (
            int(snapshot.delivery_sample_rate or 24000)
            if provider_name == "qwen3tts-audiocpp"
            else PIPELINE_SR
        )
        tts_tuning = deepcopy(snapshot.tuning) if snapshot.tuning is not None else None
        if self.backend != "openai_api" and phrase_index == 0:
            self._apply_session_voice_override(model_type, runtime_config, response)

        # Do not print assistant content through Rich here. A Windows CP1252
        # console can reject non-ASCII text and leave Rich's buffer poisoned,
        # blocking every later TTS request in the process.
        logger.info(
            "Qwen3 TTS request chars=%d provider=%s model=%s voice=%s format=%s",
            len(text),
            provider_name,
            provider_model,
            api_voice if self.backend == "openai_api" else getattr(self, "speaker", None),
            snapshot.delivery_response_format if self.backend == "openai_api" else "pcm",
        )
        start_s = perf_counter()
        if self.backend == "openai_api":
            # A failed or mocked request must not inherit pairing metadata from
            # an earlier provider call in the same long-lived handler.
            self._last_tts_response_headers = {}
        intended_streaming_mode = snapshot.delivery_streaming_mode
        first_phrase_turns = getattr(self, "_metric_first_phrase_turns", None)
        if first_phrase_turns is None:
            first_phrase_turns = self._metric_first_phrase_turns = set()
        phrase_key = snapshot.key
        if phrase_key not in first_phrase_turns:
            first_phrase_turns.add(phrase_key)
            if len(first_phrase_turns) > 256:
                first_phrase_turns.pop()
            phrase_elapsed_ms = None
            if tts_input.speech_stopped_at_s is not None:
                phrase_elapsed_ms = max(0.0, (start_s - tts_input.speech_stopped_at_s) * 1000)
            self._emit_metric(
                "gemma",
                "first_stable_phrase",
                tts_input,
                elapsed_ms=phrase_elapsed_ms,
                detail={
                    "chars": len(text),
                    "provider": provider_name,
                    **self._snapshot_metric_detail(snapshot, phrase_index),
                },
            )
        self._emit_metric(
            "tts",
            "request_start",
            tts_input,
            elapsed_ms=(
                max(0.0, (start_s - tts_input.speech_stopped_at_s) * 1000)
                if tts_input.speech_stopped_at_s is not None
                else None
            ),
            detail={
                "backend": provider_name,
                "api_base_url": provider_url if self.backend == "openai_api" else None,
                "model": provider_model,
                "language": api_language,
                "requested_language": requested_language,
                "effective_language": effective_language,
                "language_auto_supported": language_auto_supported,
                "api_response_format": snapshot.delivery_response_format,
                "chars": len(text),
                "tts_profile_id": tts_tuning.get("profile_id") if isinstance(tts_tuning, dict) else None,
                "tts_tuning_provider": tts_tuning.get("provider") if isinstance(tts_tuning, dict) else None,
                "streaming_mode": intended_streaming_mode,
                "source_sample_rate": source_sample_rate,
                **self._snapshot_metric_detail(snapshot, phrase_index),
            },
        )

        try:
            if self.backend == "openai_api":
                process_kwargs: dict[str, Any] = {
                    "language": api_language,
                    "base_url": provider_url,
                    "model": provider_model,
                    "generation": generation,
                    "progressive_buffered": (
                        provider_name == "qwen3tts-audiocpp" and not candidate_native
                    ),
                }
                if candidate_native:
                    process_kwargs["native_candidate"] = True
                if tts_request_tuning is not None:
                    process_kwargs["tts_tuning"] = tts_request_tuning
                if snapshot.frozen_clone is not None:
                    process_kwargs["clone_snapshot"] = deepcopy(snapshot.frozen_clone)
                if provider_name == "qwen3tts-audiocpp":
                    process_kwargs["response_format"] = snapshot.delivery_response_format
                    process_kwargs["source_sample_rate"] = source_sample_rate
                    expected_lifecycle = self._candidate_expected_engine_lifecycle(snapshot)
                    if expected_lifecycle is not None:
                        process_kwargs["expected_engine_epoch"] = expected_lifecycle[0]
                        process_kwargs["expected_supervisor_instance_id"] = expected_lifecycle[1]
                audio_iter = self._process_openai_api(
                    text,
                    api_voice or self.api_voice,
                    **process_kwargs,
                )
            elif self.ref_audio:
                audio_iter = self._process_voice_clone(text)
            elif model_type == "custom_voice":
                audio_iter = self._process_custom_voice(text)
            elif model_type == "voice_design":
                audio_iter = self._process_voice_design(text)
            else:
                raise ValueError(
                    "Qwen3-TTS Base model requires ref_audio for voice cloning. "
                    "Provide qwen3_tts_ref_audio or use a CustomVoice/VoiceDesign model."
                )
            first_audio = True
            audio_samples = 0
            for audio_chunk in audio_iter:
                # The HTTP stream can finish after its cancellation task has
                # detached.  The ownership check here is the final boundary
                # before a stale phrase could enter the playback queue.
                if self._input_generation_is_stale(tts_input):
                    self.cancel_active()
                    self._emit_metric(
                        "tts",
                        "cancelled_before_audio" if first_audio else "cancelled_after_audio",
                        tts_input,
                        elapsed_ms=(perf_counter() - start_s) * 1000,
                        detail={
                            "reason": "stale before playback queue commit",
                            **self._snapshot_metric_detail(snapshot, phrase_index),
                        },
                    )
                    self._release_response_synthesis_snapshot(tts_input)
                    return
                if first_audio:
                    self._log_first_audio_latency(tts_input)
                    first_pcm_ms = (perf_counter() - start_s) * 1000
                    self._emit_metric(
                        "tts",
                        "first_audio",
                        tts_input,
                        elapsed_ms=first_pcm_ms,
                        detail={
                            "backend": provider_name,
                            "model": provider_model,
                            "requested_language": requested_language,
                            "effective_language": effective_language,
                            "language_auto_supported": language_auto_supported,
                            "profile_id": tts_tuning.get("profile_id") if isinstance(tts_tuning, dict) else None,
                            "mode": getattr(self, "_last_streaming_mode", intended_streaming_mode),
                            "first_pcm_ms": round(first_pcm_ms, 3),
                            "end_to_end_ms": (
                                round(max(0.0, (perf_counter() - tts_input.speech_stopped_at_s) * 1000), 3)
                                if tts_input.speech_stopped_at_s is not None
                                else None
                            ),
                            "gpu": None,
                            "source_sample_rate": source_sample_rate,
                            **self._snapshot_metric_detail(snapshot, phrase_index),
                        },
                    )
                    first_audio = False
                audio_samples += int(np.asarray(audio_chunk).size)
                if provider_name == "qwen3tts-audiocpp":
                    # BaseHandler passes this frozen envelope through unchanged;
                    # keeping 24 kHz here makes duration/queue metrics and the
                    # browser worklet use the same model-native PCM clock.
                    yield AudioOutput(
                        audio=audio_chunk,
                        cancel_generation=tts_input.cancel_generation,
                        input_epoch=tts_input.input_epoch,
                        response_epoch=tts_input.response_epoch,
                        response_id=tts_input.response_id,
                        source_sample_rate=source_sample_rate,
                    )
                else:
                    yield audio_chunk
            elapsed_s = perf_counter() - start_s
            audio_s = audio_samples / source_sample_rate
            if audio_samples == 0:
                cancelled = self._input_generation_is_stale(tts_input)
                status = "cancelled_before_audio" if cancelled else "empty_audio"
                message = (
                    "Qwen3-TTS request cancelled before buffered audio was ready"
                    if cancelled
                    else "Qwen3-TTS request completed without audio"
                )
                log = logger.info if cancelled else logger.warning
                log(
                    "%s provider=%s model=%s voice=%s elapsed=%.2fs",
                    message,
                    provider_name,
                    provider_model,
                    api_voice or self.api_voice,
                    elapsed_s,
                )
                self._emit_metric(
                    "tts",
                    status,
                    tts_input,
                    elapsed_ms=elapsed_s * 1000,
                    detail={
                        "backend": provider_name,
                        "model": provider_model,
                        "voice": api_voice or self.api_voice,
                        "requested_language": requested_language,
                        "effective_language": effective_language,
                        "language_auto_supported": language_auto_supported,
                        "reason": "barge-in/stop/replacement" if cancelled else "provider returned no audio",
                        **self._snapshot_metric_detail(snapshot, phrase_index),
                    },
                )
                if not cancelled:
                    # An empty successful transport is still a terminal TTS
                    # failure for this answer. Suppress its later phrases rather
                    # than allowing a voice to begin in the middle of a reply.
                    self._mark_response_synthesis_failed(snapshot.key)
                return
            self._emit_metric(
                "tts",
                "done",
                tts_input,
                elapsed_ms=elapsed_s * 1000,
                detail={
                    "backend": provider_name,
                    "model": provider_model,
                    "requested_language": requested_language,
                    "effective_language": effective_language,
                    "language_auto_supported": language_auto_supported,
                    "profile_id": tts_tuning.get("profile_id") if isinstance(tts_tuning, dict) else None,
                    "profile_revision": None,
                    "mode": getattr(self, "_last_streaming_mode", intended_streaming_mode),
                    "generation_ms": round(elapsed_s * 1000, 3),
                    "audio_duration_ms": round(audio_s * 1000, 3),
                    "source_sample_rate": source_sample_rate,
                    "rtf": round(elapsed_s / audio_s, 3) if audio_s else None,
                    "reference_used": bool(getattr(self, "ref_audio", None)),
                    "reference_truncated": str(
                        getattr(self, "_last_tts_response_headers", {}).get(
                            "x-tts-reference-truncated", ""
                        )
                    ).lower()
                    == "true",
                    "gpu": None,
                    **self._snapshot_metric_detail(snapshot, phrase_index),
                    **(
                        self._candidate_response_metric_detail(
                            getattr(self, "_last_tts_response_headers", {})
                        )
                        if provider_name == "qwen3tts-audiocpp"
                        else {}
                    ),
                    **(getattr(self, "_last_tts_outcome", {}) if provider_name == "qwen3tts-audiocpp" else {}),
                },
            )
        except TTSRunawayError as e:
            self._mark_response_synthesis_failed(snapshot.key)
            logger.error("Qwen3-TTS runaway stream aborted: %s", e)
            self._emit_metric(
                "tts",
                "runaway_aborted",
                tts_input,
                elapsed_ms=(perf_counter() - start_s) * 1000,
                detail={
                    "backend": provider_name,
                    "language": api_language,
                    "requested_language": requested_language,
                    "effective_language": effective_language,
                    "language_auto_supported": language_auto_supported,
                    "error": str(e),
                    **self._snapshot_metric_detail(snapshot, phrase_index),
                    **(getattr(self, "_last_tts_outcome", {}) if provider_name == "qwen3tts-audiocpp" else {}),
                },
            )
            # Keep the immutable snapshot for diagnostics until the response
            # terminal arrives, but suppress all later phrases in this answer.
        except Exception as e:
            self._mark_response_synthesis_failed(snapshot.key)
            logger.error(f"Error during Qwen3-TTS generation: {e}", exc_info=True)
            self._emit_metric(
                "tts",
                "failed",
                tts_input,
                elapsed_ms=(perf_counter() - start_s) * 1000,
                detail={
                    "backend": provider_name,
                    "language": api_language,
                    "requested_language": requested_language,
                    "effective_language": effective_language,
                    "language_auto_supported": language_auto_supported,
                    "error": str(e),
                    **self._snapshot_metric_detail(snapshot, phrase_index),
                    **(getattr(self, "_last_tts_outcome", {}) if provider_name == "qwen3tts-audiocpp" else {}),
                },
            )
            # The terminal/cancelled response clears both this failure marker
            # and its immutable diagnostic snapshot.

    def cancel_active(self) -> None:
        lock = getattr(self, "_active_response_lock", None)
        if lock is None:
            return
        with lock:
            response = getattr(self, "_active_response", None)
            self._active_response = None
        if response is not None:
            try:
                response.close()
            except Exception:
                logger.debug("TTS stream was already closed during cancellation")

    def _emit_metric(
        self,
        stage: str,
        status: str,
        tts_input: TTSInput,
        *,
        elapsed_ms: float | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        text_output_queue = getattr(self, "text_output_queue", None)
        if text_output_queue is not None:
            text_output_queue.put(
                PipelineMetricEvent(
                    stage=stage,
                    status=status,
                    at_s=perf_counter(),
                    elapsed_ms=elapsed_ms,
                    turn_id=tts_input.turn_id,
                    turn_revision=tts_input.turn_revision,
                    input_epoch=tts_input.input_epoch,
                    response_epoch=tts_input.response_epoch,
                    response_id=tts_input.response_id,
                    detail=detail or {},
                )
            )
        elif elapsed_ms is not None:
            logger.info("Pipeline metric %s.%s %.1fms", stage, status, elapsed_ms)

    def _log_first_audio_latency(self, tts_input: TTSInput) -> None:
        if tts_input.speech_stopped_at_s is None:
            return
        latency_s = perf_counter() - tts_input.speech_stopped_at_s
        if latency_s < 0:
            return
        logger.info(
            "Last speech detected to first speech out: %.3fs (turn=%s rev=%s)",
            latency_s,
            tts_input.turn_id,
            tts_input.turn_revision,
        )

    def _mlx_streaming_interval(self) -> float:
        return max(1, self.streaming_chunk_size) / MLX_STREAMING_TOKENS_PER_SECOND

    def _mlx_stream_kwargs(self, max_tokens: int) -> dict[str, Any]:
        return {
            "max_tokens": max_tokens,
            "verbose": False,
            "stream": True,
            "streaming_interval": self._mlx_streaming_interval(),
            **self.gen_kwargs,
        }

    def _stream_mlx_generation(
        self,
        generation_fn: Callable,
        label: str,
        max_tokens: int,
        **generation_kwargs: Any,
    ) -> Iterator[bytes | np.ndarray]:
        with MLXLockContext(handler_name="Qwen3TTS", timeout=10.0) as acquired:
            if not acquired:
                raise TimeoutError("Timed out waiting for MLX lock")
            yield from self._stream(
                generation_fn(
                    **self._mlx_stream_kwargs(max_tokens=max_tokens),
                    **generation_kwargs,
                ),
                label=label,
            )

    def _process_voice_clone(self, text: str) -> Iterator[bytes | np.ndarray]:
        utterance_max_new_tokens = self._estimate_max_new_tokens(text)
        if self.backend == "mlx":
            if self.xvec_only:
                logger.warning("mlx-audio Qwen3-TTS does not support xvec_only; ignoring it")
            if self.parity_mode:
                logger.info("Qwen3-TTS parity mode is CUDA-specific and is ignored on mlx-audio")

            yield from self._stream_mlx_generation(
                self.model.generate,
                label="voice_clone_mlx",
                max_tokens=utterance_max_new_tokens,
                text=text,
                ref_audio=self._prepare_mlx_ref_audio(self.ref_audio),
                ref_text=self.ref_text,
                lang_code=self.language,
            )
            return

        yield from self._stream(
            self.model.generate_voice_clone_streaming(
                text=text,
                language=self.language,
                ref_audio=self.ref_audio,
                ref_text=self.ref_text,
                xvec_only=self.xvec_only,
                chunk_size=self.streaming_chunk_size,
                max_new_tokens=utterance_max_new_tokens,
                parity_mode=self.parity_mode,
                non_streaming_mode=self.non_streaming_mode,
            ),
            label="voice_clone_parity" if self.parity_mode else "voice_clone",
        )

    def _process_custom_voice(self, text: str) -> Iterator[bytes | np.ndarray]:
        utterance_max_new_tokens = self._estimate_max_new_tokens(text)
        speaker = self._resolve_speaker()
        if not speaker:
            raise ValueError(
                "CustomVoice generation requires a speaker. "
                "Set qwen3_tts_speaker or use a voice-clone model with ref_audio."
            )

        if self.backend == "mlx":
            yield from self._stream_mlx_generation(
                self.model.generate_custom_voice,
                label="custom_voice_mlx",
                max_tokens=utterance_max_new_tokens,
                text=text,
                speaker=speaker,
                language=self.language,
                instruct=self.instruct,
            )
            return

        yield from self._stream(
            self.model.generate_custom_voice_streaming(
                text=text,
                speaker=speaker,
                language=self.language,
                instruct=self.instruct,
                chunk_size=self.streaming_chunk_size,
                max_new_tokens=utterance_max_new_tokens,
                non_streaming_mode=self.non_streaming_mode,
            ),
            label="custom_voice",
        )

    def _process_voice_design(self, text: str) -> Iterator[bytes | np.ndarray]:
        utterance_max_new_tokens = self._estimate_max_new_tokens(text)
        if self.backend == "mlx":
            yield from self._stream_mlx_generation(
                self.model.generate_voice_design,
                label="voice_design_mlx",
                max_tokens=utterance_max_new_tokens,
                text=text,
                instruct=self.instruct,
                language=self.language,
            )
            return

        yield from self._stream(
            self.model.generate_voice_design_streaming(
                text=text,
                instruct=self.instruct,
                language=self.language,
                chunk_size=self.streaming_chunk_size,
                max_new_tokens=utterance_max_new_tokens,
                non_streaming_mode=self.non_streaming_mode,
            ),
            label="voice_design",
        )

    def on_session_end(self) -> None:
        self.speaker = self._initial_speaker
        self.ref_audio = self._initial_ref_audio
        getattr(self, "_response_synthesis_snapshots", {}).clear()
        getattr(self, "_response_phrase_counts", {}).clear()
        getattr(self, "_failed_response_synthesis_keys", set()).clear()
        getattr(self, "_metric_first_phrase_turns", set()).clear()
        logger.debug("Qwen3-TTS session state reset")

    def cleanup(self) -> None:
        try:
            if hasattr(self, "model"):
                del self.model
            for path in list(getattr(self, "_mlx_temp_ref_audio_files", set())):
                try:
                    Path(path).unlink(missing_ok=True)
                except Exception:
                    pass
            if self.backend == "mlx":
                try:
                    import mlx.core as mx

                    mx.clear_cache()
                except Exception:
                    pass
            else:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            logger.info("Qwen3-TTS handler cleaned up")
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")
