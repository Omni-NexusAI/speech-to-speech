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
from pathlib import Path
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
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, PIPELINE_END, EndOfResponse, TTSInput
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

DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DEFAULT_MLX_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-6bit"
DEFAULT_REF_TEXT = "I'm confused why some people have super short timelines, yet at the same time are bullish on scaling up reinforcement learning atop LLMs. If we're actually close to a human-like learner, then this whole approach of training on verifiable outcomes."
DEFAULT_FASTER_STREAMING_CHUNK_SIZE = 8
MAX_COALESCED_TTS_CHARS = 420
DEFAULT_MLX_STREAMING_CHUNK_SIZE = 4
DEFAULT_QWEN3_TTS_MAX_NEW_TOKENS = 1536
DEFAULT_OPENAI_API_BASE_URL = "http://127.0.0.1:8881/v1"
DEFAULT_OPENAI_API_VOICE: str | None = None
DEFAULT_OPENAI_API_BACKEND_MODEL = "1.7B-Base"
DEFAULT_GROXAXO_API_BASE_URL = "http://127.0.0.1:8882/v1"
DEFAULT_AUDIO_CPP_API_BASE_URL = "http://127.0.0.1:8890/v1"
AUDIO_CPP_NATIVE_COLD_FIRST_PCM_BUDGET_S = 45.0
DEFAULT_OPENAI_API_VOICE_LIBRARY_DIR = Path.home() / ".speech-to-speech" / "qwen3-tts-voices"
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
        api_voice: str | None = DEFAULT_OPENAI_API_VOICE,
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
        self.api_fallback_voice = api_fallback_voice
        self.api_backend_model = api_backend_model
        self.api_voice_library_dir = self._resolve_api_voice_library_dir(api_voice_library_dir)
        self.api_voice = api_voice or self._selected_api_voice()
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
        rtf = audio_duration / generation_time if generation_time > 0 else 0
        logger.info(
            f"Qwen3-TTS generated {audio_duration:.2f}s audio in {generation_time:.2f}s (RTF: {rtf:.2f}, {label})"
        )

    def _resolve_api_voice(
        self, runtime_config: RuntimeConfig | None, response: RealtimeResponseCreateParams | None
    ) -> str | None:
        if response and response.audio and response.audio.output and response.audio.output.voice:
            return str(response.audio.output.voice)
        if runtime_config is not None:
            audio = runtime_config.session.audio
            output = audio.output if audio is not None else None
            if output is not None and output.voice:
                return str(output.voice)
        return self.api_voice or self._selected_api_voice()

    @staticmethod
    def _require_api_voice(voice: str | None) -> str:
        if voice:
            return voice
        raise RuntimeError(
            "No live Base clone profile is available for the selected TTS backend. "
            "Create or import a profile, select one, or pass an explicit qwen3_tts_api_voice."
        )

    def _openai_api_payload(
        self,
        text: str,
        voice: str,
        language: str = "Auto",
        *,
        stream: bool = True,
        model: str | None = None,
        tts_tuning: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "model": model or self.api_model,
            "input": text,
            "voice": voice,
            "response_format": getattr(self, "api_response_format", "pcm"),
            "stream": stream,
            "language": language,
        }
        if tts_tuning is not None:
            payload["tuning"] = tts_tuning
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

    def _available_api_base_voices(self) -> list[tuple[str, str]]:
        profiles_dir = self.api_voice_library_dir / "profiles"
        voices: list[tuple[str, str]] = []
        for meta_path in profiles_dir.glob("*/meta.json") if profiles_dir.is_dir() else ():
            try:
                profile = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                logger.warning("Skipping unreadable Qwen3-TTS voice profile metadata: %s", meta_path)
                continue
            if profile.get("task_type") != "Base":
                continue
            profile_id = str(profile.get("profile_id") or meta_path.parent.name).strip()
            if not re.fullmatch(r"[A-Za-z0-9_-]{8,80}", profile_id):
                continue
            name = str(profile.get("name") or profile_id).strip()
            voices.append((profile_id, name))
        return sorted(voices, key=lambda item: (item[1].casefold(), item[0].casefold()))

    def _selected_api_voice(self) -> str | None:
        voices = self._available_api_base_voices()
        if not voices:
            return None
        available = {profile_id for profile_id, _name in voices}
        try:
            selected = json.loads(
                (self.api_voice_library_dir / "selected_profile.json").read_text(encoding="utf-8")
            )
            profile_id = str(selected.get("profile_id") or "").strip()
        except (OSError, json.JSONDecodeError, AttributeError):
            profile_id = ""
        if profile_id in available:
            return f"clone:{profile_id}"
        return f"clone:{voices[0][0]}"

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

    def _resolve_api_provider(self, runtime_config: RuntimeConfig | None) -> tuple[str, str, str]:
        selected = "faster"
        if runtime_config is not None:
            selected = str(runtime_config.local_pipeline.get("tts_backend") or "faster").lower()
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
            status_url = f"{self.audio_cpp_api_base_url}/backend/models"
            try:
                response = httpx.get(status_url, headers=self._openai_api_headers(), timeout=2.0)
                response.raise_for_status()
                status = response.json()
            except Exception as exc:
                raise RuntimeError(
                    "Qwen3TTS audio.cpp is unreachable on port 8890. Load a compatible Base model in its Voice Studio first."
                ) from exc
            current = str(status.get("current") or "")
            loaded = status.get("loaded_models") or []
            runtime = status.get("runtime") or {}
            progressive = bool(runtime.get("progressive_phrase_pcm"))
            native_incremental = bool(runtime.get("native_incremental_pcm"))
            if (
                status.get("state") != "loaded"
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
        """Advertise Auto only where the static provider capability declares it."""

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
        events = status.get("events")
        if not isinstance(events, list):
            return None
        supervisor_event: dict[str, Any] | None = None
        model_event: dict[str, Any] | None = None
        for event in events:
            if not isinstance(event, dict):
                continue
            action = str(event.get("action") or "")
            if action == "supervisor-started":
                supervisor_event = event
            if action in {"child-start", "model-ready"} and str(event.get("model") or "") == model:
                model_event = event
        if supervisor_event is None and model_event is None:
            return None
        return json.dumps(
            {"supervisor": supervisor_event, "model": model_event},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    def _observe_audio_cpp_native_lifecycle_epoch(
        self, base_url: str, model: str, status: dict[str, Any]
    ) -> None:
        observed = self._audio_cpp_native_lifecycle_epoch(status, model)
        if observed is None:
            return
        base_key = (str(base_url).rstrip("/"), str(model))
        epochs = getattr(self, "_audio_cpp_native_lifecycle_epochs", None)
        if epochs is None:
            epochs = {}
            self._audio_cpp_native_lifecycle_epochs = epochs
        previous = epochs.get(base_key)
        epochs[base_key] = observed
        if previous is not None and previous != observed:
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
        return (*base_key, epochs.get(base_key, "unobserved"))

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
                    logger.warning(
                        "audio.cpp native PCM timed out before first playback chunk; using explicit buffered fallback"
                    )
                    fallback_kwargs = dict(stream_kwargs)
                    fallback_kwargs["progressive_buffered"] = True
                    fallback_kwargs["native_candidate"] = False
                    self._last_streaming_mode = "buffered_fallback"
                    yield from self._stream_openai_api_voice(text, candidate_voice, **fallback_kwargs)
                    return
                raise TTSRunawayError("TTS stream produced no data before its latency budget expired") from exc
            except Exception as exc:
                if native_candidate and not emitted_audio:
                    logger.warning(
                        "audio.cpp native PCM failed before first playback chunk; using explicit buffered fallback: %s",
                        exc,
                    )
                    fallback_kwargs = dict(stream_kwargs)
                    fallback_kwargs["progressive_buffered"] = True
                    fallback_kwargs["native_candidate"] = False
                    self._last_streaming_mode = "buffered_fallback"
                    yield from self._stream_openai_api_voice(text, candidate_voice, **fallback_kwargs)
                    return
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
        native_candidate: bool = False,
    ) -> Iterator[np.ndarray]:
        url = f"{base_url or self.api_base_url}/audio/speech"
        start = perf_counter()
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
        total_samples = 0
        pending_bytes = b""
        pending_samples = np.array([], dtype=np.int16)
        first_chunk = True
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
            ),
            timeout=request_timeout,
        )
        with self._active_response_lock:
            self._active_response = response
        try:
            response.wait_for_headers()
            self._last_tts_response_headers = dict(getattr(response, "response_headers", {}) or {})
            if getattr(self, "api_response_format", "pcm") != "pcm":
                encoded_parts: list[bytes] = []
                for chunk in response.iter_bytes():
                    if generation is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(generation):
                        response.close()
                        return
                    if perf_counter() - start > runaway_budget_s:
                        response.close()
                        raise TTSRunawayError(f"TTS stream exceeded {runaway_budget_s:.1f}s budget")
                    encoded_parts.append(chunk)
                encoded_audio = b"".join(encoded_parts)
                for out in self._stream_encoded_openai_api_audio(encoded_audio, voice):
                    total_samples += len(out)
                    yield out
                return
            for chunk in response.iter_bytes():
                if generation is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(generation):
                    response.close()
                    return
                if perf_counter() - start > runaway_budget_s or total_samples / PIPELINE_SR > runaway_budget_s:
                    response.close()
                    raise TTSRunawayError(f"TTS stream exceeded {runaway_budget_s:.1f}s budget")
                if not chunk:
                    continue
                if first_chunk:
                    logger.info("Qwen3-TTS API TTFA: %.2fs (voice=%s)", perf_counter() - start, voice)
                    first_chunk = False
                pending_bytes += chunk
                even = len(pending_bytes) - (len(pending_bytes) % 2)
                if even <= 0:
                    continue
                pcm24 = np.frombuffer(pending_bytes[:even], dtype="<i2")
                pending_bytes = pending_bytes[even:]
                if native_stream_key is not None and pcm24.size:
                    self._audio_cpp_native_warm_streams_state().add(native_stream_key)
                pcm16 = self._resample_to_pipeline_sr(pcm24, self.api_sample_rate).astype(np.int16)
                pending_samples = np.concatenate([pending_samples, pcm16])
                n = (len(pending_samples) // self.blocksize) * self.blocksize
                for i in range(0, n, self.blocksize):
                    out = pending_samples[i : i + self.blocksize]
                    total_samples += len(out)
                    yield out
                pending_samples = pending_samples[n:]
        except StreamCancelled:
            return
        finally:
            response.close()
            with self._active_response_lock:
                if self._active_response is response:
                    self._active_response = None
        if len(pending_samples) > 0:
            out = np.pad(pending_samples, (0, self.blocksize - len(pending_samples)))
            total_samples += len(pending_samples)
            yield out
        generation_time = perf_counter() - start
        audio_duration = total_samples / PIPELINE_SR
        rtf = audio_duration / generation_time if generation_time > 0 else 0
        logger.info("Qwen3-TTS API generated %.2fs audio in %.2fs (RTF: %.2f)", audio_duration, generation_time, rtf)

    def _stream_encoded_openai_api_audio(self, encoded_audio: bytes, voice: str) -> Iterator[np.ndarray]:
        try:
            import soundfile as sf

            audio, sr = sf.read(io.BytesIO(encoded_audio), dtype="float32", always_2d=False)
        except Exception as exc:
            raise RuntimeError(f"Failed to decode Qwen3-TTS API audio response for {voice}: {exc}") from exc

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = self._resample_to_pipeline_sr(audio, int(sr))
        samples = self._to_int16(audio)
        for i in range(0, len(samples), self.blocksize):
            chunk = samples[i : i + self.blocksize]
            if len(chunk) < self.blocksize:
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
        if not isinstance(local_pipeline, dict) or local_pipeline.get("tts_backend") != "qwen3tts-audiocpp":
            return None
        tuning = local_pipeline.get("tts_tuning")
        if not isinstance(tuning, dict) or tuning.get("provider") != "qwen3tts-audiocpp":
            return None
        resolved = tuning.get("resolved")
        if not isinstance(resolved, dict):
            return None
        text_lookahead = resolved.get("text_lookahead")
        phrase_flush_ms = resolved.get("phrase_flush_ms")
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
        return bool(
            generation is not None
            and cancel_scope is not None
            and cancel_scope.is_stale(generation)
        )

    def _coalesce_pending_tts_input(self, current_input: TTSInput) -> tuple[str, Optional[str], bool]:
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

        phrase_queue = self._candidate_phrase_queue_settings(current_input)
        if phrase_queue and self._has_explicit_phrase_boundary(text):
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
        return combined_text, language_code, saw_end_of_response

    def process(self, tts_input: TTSIn) -> Iterator[TTSOut]:
        speculative_turns = getattr(self, "speculative_turns", None)
        if isinstance(tts_input, EndOfResponse):
            if speculative_turns and not speculative_turns.is_latest_after_reopen_grace(
                tts_input.turn_id,
                tts_input.turn_revision,
            ):
                return
            yield AUDIO_RESPONSE_DONE
            return

        if speculative_turns and not speculative_turns.is_latest_after_reopen_grace(
            tts_input.turn_id,
            tts_input.turn_revision,
        ):
            logger.debug("Dropping stale TTS input for turn=%s rev=%s", tts_input.turn_id, tts_input.turn_revision)
            return
        if speculative_turns:
            speculative_turns.commit(tts_input.turn_id, tts_input.turn_revision)

        runtime_config = tts_input.runtime_config
        response = tts_input.response
        generation = tts_input.cancel_generation
        cancel_scope = getattr(self, "cancel_scope", None)
        if generation is None and cancel_scope is not None:
            generation = cancel_scope.generation
        if self._input_generation_is_stale(tts_input):
            self._emit_metric(
                "tts",
                "cancelled_before_audio",
                tts_input,
                detail={"reason": "stale before phrase dispatch"},
            )
            return

        coalesced_text, language_code, _saw_end_of_response = self._coalesce_pending_tts_input(tts_input)
        if self._input_generation_is_stale(tts_input):
            self._emit_metric(
                "tts",
                "cancelled_before_audio",
                tts_input,
                detail={"reason": "cancelled while waiting for phrase flush"},
            )
            return

        text = coalesced_text or "Hello."

        model_type = self._model_type()
        api_voice = self._resolve_api_voice(runtime_config, response) if self.backend == "openai_api" else None
        provider_name = self.backend
        provider_url = getattr(self, "api_base_url", None)
        provider_model = getattr(self, "api_backend_model", None)
        if not language_code and runtime_config is not None:
            language_code = runtime_config.local_pipeline.get("assistant_language")
        requested_language = str(language_code or "Auto")
        api_language = self._api_language_name(language_code, text)
        if self.backend == "openai_api":
            provider_name, provider_url, provider_model = self._resolve_api_provider(runtime_config)
            api_voice = self._require_api_voice(api_voice)
        language_auto_supported = self._provider_auto_language_supported(provider_name)
        if self.backend == "openai_api":
            effective_language = (
                api_language
                if api_language != "Auto" or language_auto_supported
                else None
            )
        else:
            effective_language = str(getattr(self, "language", "") or "") or None
        candidate_native = provider_name == "qwen3tts-audiocpp" and bool(
            getattr(self, "api_streaming_supported", False)
        )
        tts_tuning = None
        if provider_name == "qwen3tts-audiocpp" and runtime_config is not None:
            candidate_tuning = runtime_config.local_pipeline.get("tts_tuning")
            if isinstance(candidate_tuning, dict) and candidate_tuning.get("provider") == "qwen3tts-audiocpp":
                tts_tuning = candidate_tuning
        if self.backend != "openai_api":
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
            getattr(self, "api_response_format", None) if self.backend == "openai_api" else "pcm",
        )
        start_s = perf_counter()
        if self.backend == "openai_api":
            # A failed or mocked request must not inherit pairing metadata from
            # an earlier provider call in the same long-lived handler.
            self._last_tts_response_headers = {}
        intended_streaming_mode = (
            getattr(self, "api_candidate_streaming_mode", "buffered_phrase")
            if provider_name == "qwen3tts-audiocpp"
            else "provider_default"
        )
        first_phrase_turns = getattr(self, "_metric_first_phrase_turns", None)
        if first_phrase_turns is None:
            first_phrase_turns = self._metric_first_phrase_turns = set()
        phrase_key = (tts_input.turn_id, tts_input.turn_revision)
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
                detail={"chars": len(text), "provider": provider_name},
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
                "api_response_format": getattr(self, "api_response_format", None),
                "chars": len(text),
                "tts_profile_id": tts_tuning.get("profile_id") if isinstance(tts_tuning, dict) else None,
                "tts_tuning_provider": tts_tuning.get("provider") if isinstance(tts_tuning, dict) else None,
                "streaming_mode": intended_streaming_mode,
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
                if tts_tuning is not None:
                    process_kwargs["tts_tuning"] = tts_tuning
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
                        },
                    )
                    first_audio = False
                audio_samples += int(np.asarray(audio_chunk).size)
                yield audio_chunk
            elapsed_s = perf_counter() - start_s
            audio_s = audio_samples / PIPELINE_SR
            if audio_samples == 0:
                cancelled = (
                    generation is not None
                    and self.cancel_scope is not None
                    and self.cancel_scope.is_stale(generation)
                )
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
                    },
                )
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
                    "rtf": round(elapsed_s / audio_s, 3) if audio_s else None,
                    "reference_used": bool(getattr(self, "ref_audio", None)),
                    "reference_truncated": str(
                        getattr(self, "_last_tts_response_headers", {}).get(
                            "x-tts-reference-truncated", ""
                        )
                    ).lower()
                    == "true",
                    "gpu": None,
                    **(
                        self._candidate_response_metric_detail(
                            getattr(self, "_last_tts_response_headers", {})
                        )
                        if provider_name == "qwen3tts-audiocpp"
                        else {}
                    ),
                },
            )
        except TTSRunawayError as e:
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
                },
            )
        except Exception as e:
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
                },
            )

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
