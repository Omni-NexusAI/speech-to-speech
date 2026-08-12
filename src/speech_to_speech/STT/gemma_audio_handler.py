from __future__ import annotations

import base64
import json
import logging
import os
import re
import threading
import unicodedata
import wave
from collections.abc import Iterator
from io import BytesIO
from time import perf_counter, time
from typing import Any

import httpx
import numpy as np
from openai.types.realtime.conversation_item import RealtimeConversationItemFunctionCall
from openai.types.responses import ResponseFunctionToolCall

from speech_to_speech.LLM.chat import (
    make_user_audio_message,
    make_user_message,
)
from speech_to_speech.LLM.chat_completions_language_model import (
    ChatCompletionsApiModelHandler,
    _to_chat_tool_choice,
    _to_chat_tools,
)
from speech_to_speech.pipeline.cancellable_http import CancellableAsyncSSEStream
from speech_to_speech.pipeline.events import PipelineMetricEvent
from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import DirectAssistantResponse, PartialTranscription
from speech_to_speech.pipeline.model_operations import ModelOperationCoordinator, ModelOperationToken
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler
from speech_to_speech.utils.utils import _generate_id

logger = logging.getLogger(__name__)

DIRECT_AUDIO_TEMPERATURE = 0.1
DIRECT_AUDIO_TOP_P = 0.9


def _response_max_tokens(value: Any, fallback: Any = 384) -> int:
    """Return a bounded spoken-response limit without affecting live previews."""
    try:
        return min(1024, max(64, int(value if value is not None else fallback)))
    except (TypeError, ValueError):
        return 384


_TRANSCRIPT_MARKER = "USER_TRANSCRIPT:"
_MEMORY_MARKER = "USER_MEMORY:"
_RESPONSE_MARKER = "ASSISTANT_RESPONSE:"
_LANGUAGE_MARKER = "ASSISTANT_LANGUAGE:"
_PREAMBLE_MARKER = "ASSISTANT_PREAMBLE:"
_CAMERA_CONTEXT_MARKER = "CAMERA_CONTEXT:"
_PREVIEW_TRANSCRIPT_MARKER = "TRANSCRIPT:"
_FINAL_TRANSCRIPT_RE = re.compile(
    r"(?ims)^\s*(?:USER_TRANSCRIPT|USER_SPEECH|TRANSCRIPT|USER)\s*:\s*(.+?)"
    r"(?=^\s*(?:USER_MEMORY|ASSISTANT_LANGUAGE|CAMERA_CONTEXT|ASSISTANT_PREAMBLE|ASSISTANT_RESPONSE|"
    r"ASSISTANT|RESPONSE)\s*:|\Z)"
)
_FINAL_MEMORY_RE = re.compile(
    r"(?ims)^\s*USER_MEMORY\s*:\s*(.+?)"
    r"(?=^\s*(?:USER_TRANSCRIPT|USER_SPEECH|TRANSCRIPT|USER|ASSISTANT_LANGUAGE|CAMERA_CONTEXT|"
    r"ASSISTANT_PREAMBLE|ASSISTANT_RESPONSE|ASSISTANT|RESPONSE)\s*:|\Z)"
)
_ASSISTANT_RESPONSE_RE = re.compile(
    r"(?ims)^\s*(?:ASSISTANT_RESPONSE|ASSISTANT|RESPONSE)\s*:\s*(.+?)"
    r"(?=^\s*(?:USER_MEMORY|USER_TRANSCRIPT|USER_SPEECH|TRANSCRIPT|USER|ASSISTANT_LANGUAGE|"
    r"CAMERA_CONTEXT|ASSISTANT_PREAMBLE|ASSISTANT_RESPONSE|ASSISTANT|RESPONSE)\s*:|\Z)"
)
_ASSISTANT_LANGUAGE_RE = re.compile(r"(?im)^\s*ASSISTANT_LANGUAGE\s*:\s*([^\r\n]+)")
_CAMERA_CONTEXT_RE = re.compile(r"(?im)^\s*CAMERA_CONTEXT\s*:\s*([^\r\n]+)")
_ASSISTANT_PREAMBLE_RE = re.compile(
    r"(?ims)^\s*ASSISTANT_PREAMBLE\s*:\s*(.+?)"
    r"(?=^\s*(?:USER_MEMORY|USER_TRANSCRIPT|ASSISTANT_LANGUAGE|CAMERA_CONTEXT|ASSISTANT_RESPONSE|"
    r"ASSISTANT|RESPONSE)\s*:|\Z)"
)
_MEMORY_CONTROL_MARKER_RE = re.compile(
    r"(?i)(?:USER_MEMORY|USER_TRANSCRIPT|USER_SPEECH|TRANSCRIPT|USER|ASSISTANT_LANGUAGE|"
    r"CAMERA_CONTEXT|ASSISTANT_PREAMBLE|ASSISTANT_RESPONSE|ASSISTANT|RESPONSE)\s*:"
)
_OUTPUT_MARKER_RE = re.compile(r"(?im)^\s*(?:ASSISTANT_PREAMBLE|ASSISTANT_RESPONSE|ASSISTANT|RESPONSE)\s*:")
_CONTROL_LINE_RE = re.compile(
    r"(?im)^\s*(?:USER_MEMORY|USER_TRANSCRIPT|USER_SPEECH|TRANSCRIPT|USER|ASSISTANT_LANGUAGE|"
    r"CAMERA_CONTEXT|ASSISTANT_PREAMBLE|ASSISTANT_RESPONSE|ASSISTANT|RESPONSE)\s*:[^\r\n]*(?:\r?\n|$)"
)
_TRAILING_RESPONSE_CONTROL_RE = re.compile(
    r"(?im)^\s*(?:USER_MEMORY|USER_TRANSCRIPT|USER_SPEECH|TRANSCRIPT|USER|ASSISTANT_LANGUAGE|"
    r"CAMERA_CONTEXT|ASSISTANT_PREAMBLE|ASSISTANT_RESPONSE|ASSISTANT|RESPONSE)\s*:"
)
_MAX_USER_MEMORY_CHARS = 600
_SENTENCE_RE = re.compile(r"(.+?[.!?](?:\s+|$))", re.DOTALL)


def _failure_sentinel_key(value: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", value).casefold().split())
    start = 0
    end = len(normalized)
    while start < end and unicodedata.category(normalized[start])[0] in {"P", "Z"}:
        start += 1
    while end > start and unicodedata.category(normalized[end - 1])[0] in {"P", "Z"}:
        end -= 1
    return normalized[start:end]


_TRANSCRIPT_FAILURE_SENTINELS = frozenset(
    _failure_sentinel_key(value)
    for value in {
        "inaudible",
        "unintelligible",
        "garbled",
        "audio unclear",
        "unclear audio",
        "audio unintelligible",
        "unintelligible audio",
        "audio garbled",
        "garbled audio",
        "audio was unclear",
        "audio was unintelligible",
        "audio was garbled",
        "no speech",
        "no speech detected",
        "no intelligible speech",
        "no intelligible speech detected",
        "could not understand audio",
        "could not understand the audio",
        "unable to understand audio",
        "unable to understand the audio",
        "could not transcribe audio",
        "could not transcribe the audio",
        "transcription failed",
        "transcription unavailable",
        "无法听清",
        "无法理解音频",
        "音频不清晰",
        "音频无法辨认",
        "未检测到语音",
        "未检测到可理解的语音",
        "无法转录音频",
        "转录失败",
        "聞き取れません",
        "音声が不明瞭です",
        "音声を理解できません",
        "音声を認識できません",
        "発話が検出されませんでした",
        "文字起こしに失敗しました",
        "알아들을 수 없습니다",
        "오디오가 불명확합니다",
        "음성을 이해할 수 없습니다",
        "음성이 감지되지 않았습니다",
        "알아들을 수 있는 음성이 감지되지 않았습니다",
        "전사에 실패했습니다",
        "unverständlich",
        "audio unverständlich",
        "unklares audio",
        "keine sprache erkannt",
        "keine verständliche sprache erkannt",
        "audio konnte nicht verstanden werden",
        "transkription fehlgeschlagen",
        "incompréhensible",
        "audio incompréhensible",
        "audio peu clair",
        "aucune parole détectée",
        "aucune parole intelligible détectée",
        "impossible de comprendre l'audio",
        "échec de la transcription",
        "неразборчиво",
        "неразборчивый звук",
        "аудио неразборчиво",
        "речь не обнаружена",
        "разборчивая речь не обнаружена",
        "не удалось понять аудио",
        "не удалось расшифровать аудио",
        "ошибка транскрипции",
        "inaudível",
        "ininteligível",
        "áudio ininteligível",
        "áudio pouco claro",
        "nenhuma fala detectada",
        "nenhuma fala inteligível detectada",
        "não foi possível entender o áudio",
        "falha na transcrição",
        "ininteligible",
        "audio ininteligible",
        "audio poco claro",
        "no se detectó habla",
        "no se detectó habla inteligible",
        "no se pudo entender el audio",
        "no se pudo transcribir el audio",
        "falló la transcripción",
        "inudibile",
        "incomprensibile",
        "audio incomprensibile",
        "audio poco chiaro",
        "nessun parlato rilevato",
        "nessun parlato intelligibile rilevato",
        "impossibile comprendere l'audio",
        "trascrizione non riuscita",
    }
)

_ASSISTANT_LANGUAGE_ALIASES = {
    "auto": "Auto",
    "automatic": "Auto",
    "mixed": "Auto",
    "mixed language": "Auto",
    "multilingual": "Auto",
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-hans": "Chinese",
    "zh-hant": "Chinese",
    "cmn": "Chinese",
    "chinese": "Chinese",
    "中文": "Chinese",
    "汉语": "Chinese",
    "漢語": "Chinese",
    "普通话": "Chinese",
    "en": "English",
    "en-us": "English",
    "en-gb": "English",
    "english": "English",
    "ja": "Japanese",
    "jp": "Japanese",
    "japanese": "Japanese",
    "日本語": "Japanese",
    "ko": "Korean",
    "kr": "Korean",
    "korean": "Korean",
    "한국어": "Korean",
    "de": "German",
    "german": "German",
    "deutsch": "German",
    "fr": "French",
    "french": "French",
    "français": "French",
    "francais": "French",
    "ru": "Russian",
    "russian": "Russian",
    "русский": "Russian",
    "pt": "Portuguese",
    "pt-br": "Portuguese",
    "pt-pt": "Portuguese",
    "portuguese": "Portuguese",
    "português": "Portuguese",
    "portugues": "Portuguese",
    "es": "Spanish",
    "spanish": "Spanish",
    "español": "Spanish",
    "espanol": "Spanish",
    "castellano": "Spanish",
    "it": "Italian",
    "italian": "Italian",
    "italiano": "Italian",
}


class GemmaAudioSTTHandler(BaseSTTHandler):
    """STT-slot adapter that sends completed audio turns directly to Gemma."""

    def setup(
        self,
        model_name: str = "gemma-4-12b-it-qat",
        base_url: str = "http://127.0.0.1:8818/v1",
        api_key: str | None = None,
        stream: bool = True,
        timeout_s: float = 30.0,
        revision_settle_s: float = 0.25,
        format: str = "wav",
        prompt: str = "",
        system_prompt: str = "You are a local low-latency voice assistant. Answer naturally for speech synthesis.",
        gen_kwargs: dict[str, Any] | None = None,
        text_output_queue: Any | None = None,
        cancel_scope: Any | None = None,
        model_operations: ModelOperationCoordinator | None = None,
    ) -> None:
        self.model_name = model_name
        self.base_url = (os.getenv("GEMMA_AUDIO_BASE_URL") or base_url).rstrip("/")
        self.api_key = api_key or os.getenv("GEMMA_API_KEY") or os.getenv("LLAMA_CPP_API_KEY")
        self.stream = stream
        self.timeout = httpx.Timeout(float(timeout_s), connect=min(5.0, float(timeout_s)))
        self.final_revision_settle_s = max(0.0, float(revision_settle_s))
        self.audio_format = format
        # Compatibility option for older configs. It must never be included in
        # user multimodal content, where Gemma can mistake it for user speech.
        self.prompt = prompt
        self.system_prompt = system_prompt
        self.gen_kwargs = gen_kwargs or {}
        self.sample_rate = 16000
        self.text_output_queue: Any | None = text_output_queue
        self.cancel_scope = cancel_scope
        self.model_operations = model_operations
        self._preview_transcripts: dict[tuple[str | None, int | None], str] = {}
        # One semantic anchor per session/turn. Revisions replace its cumulative
        # WAV in place instead of creating multiple user turns.
        self._accepted_user_items: dict[tuple[str, str], tuple[str, int]] = {}
        self._accepted_user_lock = threading.Lock()
        self._active_resources: set[Any] = set()
        self._active_turn: tuple[str | None, int | None] | None = None
        self._active_response_lock = threading.Lock()
        logger.info("Gemma audio direct mode configured for %s at %s", self.model_name, self.base_url)

    def on_speculative_turns_attached(self) -> None:
        tracker = getattr(self, "speculative_turns", None)
        if tracker is not None:
            tracker.add_revision_listener(self._on_revision_observed)

    def _on_revision_observed(self, turn_id: str, revision: int) -> None:
        with self._active_response_lock:
            active = self._active_turn
        if active is not None and active[0] == turn_id and active[1] is not None and revision > active[1]:
            logger.info(
                "Cancelling superseded Gemma audio request turn=%s old_rev=%s new_rev=%s",
                turn_id,
                active[1],
                revision,
            )
            if self.model_operations is not None:
                self.model_operations.cancel_and_wait("audio_revision", 2.0)
            else:
                self.cancel_active()

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        generation = self.cancel_scope.generation if self.cancel_scope is not None else None
        operation: ModelOperationToken | None = None
        coordinator = self.model_operations
        runtime_config = getattr(vad_audio, "runtime_config", None)
        progressive = vad_audio.mode == "progressive"
        if progressive and not self._live_preview_enabled(runtime_config):
            return
        if coordinator is not None:
            local_pipeline = getattr(runtime_config, "local_pipeline", None) or {}
            session_id = str(local_pipeline.get("_session_id") or "") or None
            operation = coordinator.acquire(
                kind="transcription_preview" if progressive else "direct_audio",
                session_id=session_id,
                turn_id=vad_audio.turn_id,
                turn_revision=vad_audio.turn_revision,
                cancel_generation=generation,
                drop_if_busy=progressive,
                stale=lambda: self._request_is_stale(vad_audio, generation),
            )
            if operation is None:
                if progressive:
                    self._emit_metric(vad_audio, "transcription", "live_dropped", detail={"reason": "model_busy"})
                return
            coordinator.bind_cancel(operation, self.cancel_active)
        try:
            if operation is not None and coordinator is not None and not coordinator.is_current(operation):
                return
            if progressive:
                yield from self._iter_progressive_transcriptions(vad_audio)
                return
            yield from self._process_direct(vad_audio, generation)
        finally:
            if operation is not None and coordinator is not None:
                coordinator.release(operation)

    def _process_direct(self, vad_audio: STTIn, generation: int | None) -> Iterator[STTOut]:
        start_s = perf_counter()
        runtime_config = getattr(vad_audio, "runtime_config", None)
        if getattr(runtime_config, "local_pipeline", None) is not None:
            # Language belongs to the current direct-audio answer. Do not let a
            # previous turn's TTS language silently influence this one.
            runtime_config.local_pipeline.pop("assistant_language", None)
        audio = self._as_float32_mono(vad_audio.audio)
        duration_s = len(audio) / self.sample_rate if self.sample_rate else 0.0
        absolute_audio = np.abs(audio)
        rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64))) if audio.size else 0.0
        peak = float(np.max(absolute_audio)) if audio.size else 0.0
        clipping_fraction = float(np.mean(absolute_audio >= 0.999)) if audio.size else 0.0
        input_detail = {
            "audio_s": round(duration_s, 3),
            "rms": round(rms, 6),
            "peak": round(peak, 6),
            "near_silence": rms < 0.005,
            "clipping": clipping_fraction > 0.001,
            "clipping_fraction": round(clipping_fraction, 6),
            "revision_count": max(1, int(getattr(vad_audio, "turn_revision", 0) or 0) + 1),
        }
        logger.info(
            "Gemma audio direct request start turn=%s rev=%s audio=%.3fs",
            vad_audio.turn_id,
            vad_audio.turn_revision,
            duration_s,
        )
        full_buffer_tts = self._full_buffer_tts(getattr(vad_audio, "runtime_config", None))
        self._emit_metric(
            vad_audio,
            "gemma",
            "request_start",
            detail={**input_detail, "full_buffer_tts": full_buffer_tts},
        )
        self._emit_metric(vad_audio, "transcription", "captured", detail={**input_detail, "mode": "final"})
        first = True
        failed_status: str | None = None
        terminal_error: str | None = None
        try:
            for response in self._iter_direct_responses(audio, vad_audio, generation=generation):
                if first and (response.text or response.tools):
                    self._emit_metric(vad_audio, "gemma", "first_token", elapsed_ms=(perf_counter() - start_s) * 1000)
                    first = False
                yield response
        except httpx.ReadTimeout:
            failed_status = "timeout"
            terminal_error = "Direct audio model response timed out."
        except Exception as exc:
            failed_status = "failed"
            terminal_error = f"Direct audio model request failed: {type(exc).__name__}"
            logger.exception(
                "Gemma audio direct request failed turn=%s rev=%s",
                vad_audio.turn_id,
                vad_audio.turn_revision,
            )
        finally:
            total_s = perf_counter() - start_s
            logger.info(
                "Gemma audio direct request done turn=%s rev=%s total=%.3fs",
                vad_audio.turn_id,
                vad_audio.turn_revision,
                total_s,
            )
            status = "cancelled" if self._request_is_stale(vad_audio, generation) else failed_status or "complete"
            self._emit_metric(vad_audio, "gemma", status, elapsed_ms=total_s * 1000)
            if self._owns_user_context(vad_audio):
                self._emit_history_metric(
                    vad_audio,
                    (
                        "user_retained_after_cancel"
                        if status == "cancelled"
                        else "user_retained_after_failure"
                        if terminal_error
                        else "user_retained_without_response"
                    ),
                    input_kind="input_audio",
                    committed=True,
                )
                self._finish_user_context(vad_audio)
        if terminal_error and not self._request_is_stale(vad_audio, generation):
            yield self._direct(
                vad_audio,
                "",
                is_final=True,
                error=terminal_error,
                generation=generation,
            )

    def _request_is_stale(self, vad_audio: STTIn, generation: int | None) -> bool:
        if generation is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(generation):
            return True
        tracker = getattr(self, "speculative_turns", None)
        return tracker is not None and not tracker.is_latest(vad_audio.turn_id, vad_audio.turn_revision)

    def _emit_metric(
        self,
        vad_audio: STTIn,
        stage: str,
        status: str,
        *,
        elapsed_ms: float | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        if self.text_output_queue is not None:
            self.text_output_queue.put(
                PipelineMetricEvent(
                    stage=stage,
                    status=status,
                    at_s=time(),
                    elapsed_ms=elapsed_ms,
                    turn_id=vad_audio.turn_id,
                    turn_revision=vad_audio.turn_revision,
                    detail=detail or {},
                )
            )
        elif elapsed_ms is not None:
            logger.info("Pipeline metric %s.%s %.1fms", stage, status, elapsed_ms)

    def _as_float32_mono(self, audio: Any) -> np.ndarray:
        arr = np.asarray(audio, dtype=np.float32).squeeze()
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        return arr

    def _wav_bytes(self, audio: np.ndarray) -> bytes:
        pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
        out = BytesIO()
        with wave.open(out, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(pcm16.tobytes())
        return out.getvalue()

    @staticmethod
    def _turn_revision(vad_audio: STTIn) -> int:
        revision = getattr(vad_audio, "turn_revision", None)
        return revision if isinstance(revision, int) else 0

    @classmethod
    def _turn_key(cls, vad_audio: STTIn) -> tuple[str, str]:
        runtime_config = getattr(vad_audio, "runtime_config", None)
        local_pipeline = getattr(runtime_config, "local_pipeline", None) or {}
        session_id = str(local_pipeline.get("_session_id") or "").strip()
        chat = cls._conversation_chat(vad_audio)
        session_key = session_id or (f"chat-{id(chat)}" if chat is not None else f"runtime-{id(runtime_config)}")
        turn_id = getattr(vad_audio, "turn_id", None)
        return session_key, str(turn_id) if turn_id is not None else f"anonymous-{id(vad_audio)}"

    @staticmethod
    def _conversation_chat(vad_audio: STTIn) -> Any | None:
        runtime_config = getattr(vad_audio, "runtime_config", None)
        return getattr(runtime_config, "chat", None)

    def _emit_history_metric(self, vad_audio: STTIn, status: str, *, input_kind: str, committed: bool) -> None:
        chat = self._conversation_chat(vad_audio)
        stats = chat.stats() if chat is not None and callable(getattr(chat, "stats", None)) else {}
        self._emit_metric(
            vad_audio,
            "history",
            status,
            detail={
                "input_kind": input_kind,
                "committed": committed,
                "turns": int(stats.get("turns", 0)),
                "items": int(stats.get("items", 0)),
                "pending_tool_calls": int(stats.get("pending_tool_calls", 0)),
                "trim_count": int(stats.get("trim_count", 0)),
            },
        )

    def _owned_user_context(self, vad_audio: STTIn) -> tuple[str, int] | None:
        with self._accepted_user_lock:
            return self._accepted_user_items.get(self._turn_key(vad_audio))

    def _owns_user_context(self, vad_audio: STTIn) -> bool:
        owned = self._owned_user_context(vad_audio)
        return owned is not None and owned[1] == self._turn_revision(vad_audio)

    def _commit_accepted_audio(self, vad_audio: STTIn, encoded_audio: str) -> str | None:
        """Persist one accepted turn before generation using its original mono WAV."""

        chat = self._conversation_chat(vad_audio)
        if chat is None:
            return None
        key = self._turn_key(vad_audio)
        revision = self._turn_revision(vad_audio)
        with self._accepted_user_lock:
            existing = self._accepted_user_items.get(key)
            if existing is not None:
                item_id, owned_revision = existing
                if revision < owned_revision:
                    return None
                if revision == owned_revision:
                    return item_id
                if chat.replace_user_message_audio(item_id, encoded_audio):
                    self._accepted_user_items[key] = (item_id, revision)
                    self._emit_history_metric(
                        vad_audio,
                        "user_superseded",
                        input_kind="input_audio",
                        committed=True,
                    )
                    return item_id
                self._accepted_user_items.pop(key, None)
            item = chat.add_item(make_user_audio_message(encoded_audio))
            assert item.id is not None
            self._accepted_user_items[key] = (item.id, revision)
        self._emit_history_metric(vad_audio, "user_committed", input_kind="input_audio", committed=True)
        return item.id

    def _ensure_user_context(
        self,
        vad_audio: STTIn,
        transcript: str | None,
        user_memory: str | None = None,
    ) -> str | None:
        """Return the accepted user item, preferring transcript over semantic memory."""

        chat = self._conversation_chat(vad_audio)
        if chat is None:
            return None
        transcript = self._validate_transcript(transcript)
        user_memory = None if transcript else self._validate_user_memory(user_memory)
        semantic_text = transcript or user_memory
        semantic_kind = "transcript" if transcript else "semantic_memory" if user_memory else "input_audio"
        key = self._turn_key(vad_audio)
        revision = self._turn_revision(vad_audio)
        upgraded = False
        with self._accepted_user_lock:
            owned = self._accepted_user_items.get(key)
            if owned is None:
                if not semantic_text:
                    return None
                item = chat.add_item(make_user_message(semantic_text))
                assert item.id is not None
                item_id = item.id
                self._accepted_user_items[key] = (item_id, revision)
                committed_new = True
            else:
                item_id, owned_revision = owned
                if revision < owned_revision:
                    return None
                if revision > owned_revision:
                    self._accepted_user_items[key] = (item_id, revision)
                committed_new = False
            if semantic_text:
                # Ownership validation and semantic replacement are one
                # transaction. Otherwise rev0 can pass the check, rev1 can
                # replace the cumulative WAV, and rev0 can then overwrite the
                # newer anchor with stale metadata.
                upgraded = chat.replace_user_message_text(item_id, semantic_text)
        if committed_new:
            self._emit_history_metric(vad_audio, "user_committed", input_kind=semantic_kind, committed=True)
            return item_id
        if upgraded:
            self._emit_history_metric(vad_audio, "user_upgraded", input_kind=semantic_kind, committed=True)
        return item_id

    def _finish_user_context(self, vad_audio: STTIn) -> None:
        key = self._turn_key(vad_audio)
        revision = self._turn_revision(vad_audio)
        with self._accepted_user_lock:
            owned = self._accepted_user_items.get(key)
            if owned is not None and owned[1] == revision:
                self._accepted_user_items.pop(key, None)

    def _model_endpoint(self, vad_audio: STTIn | None) -> tuple[str, str, str | None]:
        runtime_config = getattr(vad_audio, "runtime_config", None) if vad_audio is not None else None
        endpoint = getattr(runtime_config, "model_endpoint", None)
        if endpoint is None:
            return self.base_url, self.model_name, self.api_key
        return endpoint.base_url.rstrip("/"), endpoint.model, endpoint.api_key

    def _payload(
        self,
        audio: np.ndarray,
        vad_audio: STTIn | None = None,
        *,
        encoded_audio: str | None = None,
    ) -> dict[str, Any]:
        encoded = encoded_audio or base64.b64encode(self._wav_bytes(audio)).decode("ascii")
        if vad_audio is None:
            vad_audio = type("VadAudioShim", (), {"runtime_config": None})()
        runtime_config = getattr(vad_audio, "runtime_config", None)
        session = runtime_config.session if runtime_config is not None else None
        session_instructions = str(getattr(session, "instructions", "") or "").strip()
        system_parts = [self.system_prompt]
        if session_instructions:
            system_parts.append(session_instructions)
        semantic_instructions = (
            "Treat accepted user input as an ordinary semantic user message. Infer its likely intent from the whole "
            "accepted turn and answer naturally or call an appropriate provided tool. It may use any language, accent, "
            "or code-switching. Follow the language or languages naturally used in the current utterance unless the user "
            "or session instructions request another response language. Stay focused on the user's intended topic unless "
            "the user explicitly asks about system internals. "
            "For every meaningful accepted turn, begin with USER_MEMORY as one short, affirmative, content-faithful "
            "semantic paraphrase of what the user means. Preserve names, numbers, negation, ordinary references, and the "
            "language of the request; resolve references from conversation context without adding facts. Omit USER_MEMORY "
            "only when no meaningful intent is recoverable; in that rare case, ask one short, context-specific question "
            "using natural wording that varies with the conversation. USER_MEMORY is hidden session context, not an answer or a tool "
            "result, and must not gate, alter, or replace the assistant response or tool call. Begin meaningful turns with:\n"
            "USER_MEMORY: <short affirmative semantic paraphrase>\n"
            "Then continue with:\n"
            "ASSISTANT_LANGUAGE: <single language name for a monolingual spoken answer; Auto for a mixed-language "
            "or otherwise unspecified spoken answer>\n"
        )
        if self._camera_tool_available(vad_audio):
            semantic_instructions += (
                "CAMERA_CONTEXT: <current, historical, or none>\n"
                "Use current when the user asks what is visible now, asks you to look again, or asks what changed; this "
                "requires a new camera_snapshot before any result-dependent answer. Use historical only for a question "
                "explicitly about a prior observation. Use none when no camera context is involved. Then continue with:\n"
            )
        semantic_instructions += (
            "ASSISTANT_RESPONSE: <your spoken answer>\n"
            "When a provided tool is needed, call it in the same response and never fabricate its result. Before the "
            "function call, provide one brief, natural acknowledgement whose wording fits the specific request and "
            "varies with the conversation; do not reuse a stock phrase. Put it in ASSISTANT_PREAMBLE: "
            "<acknowledgement>; plain ASSISTANT_RESPONSE text is also accepted for "
            "compatibility. Do not emit a result-dependent ASSISTANT_RESPONSE until the tool result is available. "
            "Ask a brief, content-focused follow-up only when the request itself lacks a detail needed to complete it. "
            "Resolve pronouns, references, and requests such as 'do that in reverse' from the retained conversation and "
            "completed tool results, then continue the conversation naturally. "
            "Do not wrap plain-text responses in JSON or Markdown."
        )
        system_parts.append(semantic_instructions)
        user_content: list[dict[str, Any]] = []
        for image_url in self._conversation_image_urls(runtime_config):
            user_content.append({"type": "image_url", "image_url": {"url": image_url}})
        user_content.append({"type": "input_audio", "input_audio": {"data": encoded, "format": self.audio_format}})

        history: list[dict[str, Any]] = []
        chat = getattr(runtime_config, "chat", None)
        if chat is not None and callable(getattr(chat, "copy", None)):
            history_chat = chat.copy()
            owned = self._owned_user_context(vad_audio)
            if owned is not None:
                # The live user message below is the only representation of the
                # current semantic turn, even while a newer cumulative revision
                # replaces the persistent anchor.
                history_chat.remove_user_message(owned[0])
            history = [
                message
                for message in ChatCompletionsApiModelHandler._chat_messages(history_chat)
                if message.get("role") != "system"
            ]

        _, model_name, _ = self._model_endpoint(vad_audio)
        payload: dict[str, Any] = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "\n\n".join(system_parts)},
                *history,
                {"role": "user", "content": user_content},
            ],
            # Keep the transport streaming even in full-buffer mode so Stop can
            # close an in-flight llama.cpp request. Buffering is applied locally.
            "stream": self.stream,
            **self.gen_kwargs,
        }
        # Direct-audio semantics are intentionally low-variance. Keep release
        # probes and production on the same sampling contract.
        payload["temperature"] = DIRECT_AUDIO_TEMPERATURE
        payload["top_p"] = DIRECT_AUDIO_TOP_P
        local_pipeline = getattr(runtime_config, "local_pipeline", None) or {}
        payload["max_tokens"] = _response_max_tokens(local_pipeline.get("max_response_tokens"), payload.get("max_tokens"))
        chat_template_kwargs = dict(payload.get("chat_template_kwargs") or {})
        chat_template_kwargs.setdefault("enable_thinking", False)
        payload["chat_template_kwargs"] = chat_template_kwargs
        tools = getattr(session, "tools", None) if session is not None else None
        if tools:
            chat_tools = _to_chat_tools(tools)
            if chat_tools:
                payload["tools"] = chat_tools
                tool_choice = getattr(session, "tool_choice", None)
                if tool_choice is not None:
                    payload["tool_choice"] = _to_chat_tool_choice(tool_choice)
        return payload

    def _transcription_payload(self, audio: np.ndarray, vad_audio: STTIn | None = None) -> dict[str, Any]:
        encoded = base64.b64encode(self._wav_bytes(audio)).decode("ascii")
        _, model_name, _ = self._model_endpoint(vad_audio)
        return {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Transcribe the user's speech accurately. Return exactly one line beginning with "
                        "TRANSCRIPT: followed by the cumulative user speech. Do not answer the user, add "
                        "commentary, Markdown, labels, or quotation marks."
                    ),
                },
                {
                    "role": "user",
                    "content": [{"type": "input_audio", "input_audio": {"data": encoded, "format": self.audio_format}}],
                },
            ],
            "stream": True,
            "temperature": 0,
            "max_tokens": 96,
            "chat_template_kwargs": {"enable_thinking": False},
        }

    def _iter_progressive_transcriptions(self, vad_audio: STTIn) -> Iterator[PartialTranscription]:
        audio = self._as_float32_mono(vad_audio.audio)
        key = (vad_audio.turn_id, vad_audio.turn_revision)
        start_s = perf_counter()
        self._emit_metric(
            vad_audio, "transcription", "live_start", detail={"audio_s": round(len(audio) / self.sample_rate, 3)}
        )
        raw = ""
        first = True
        try:
            base_url, _, api_key = self._model_endpoint(vad_audio)
            response = self._stream_request(
                f"{base_url}/chat/completions",
                self._transcription_payload(audio, vad_audio),
                api_key=api_key,
            )
            self._track_active(response, vad_audio)
            try:
                response.wait_for_headers()
                for line in response.iter_lines():
                    if not line:
                        continue
                    if line.startswith("data:"):
                        line = line[len("data:") :].strip()
                    if line == "[DONE]":
                        break
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    choices = data.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content") or choices[0].get("text")
                    if not content:
                        continue
                    if first:
                        self._emit_metric(
                            vad_audio, "transcription", "live_first_text", elapsed_ms=(perf_counter() - start_s) * 1000
                        )
                        first = False
                    raw += str(content)
                    transcript = self._extract_preview_transcript(raw)
                    if transcript and transcript != self._preview_transcripts.get(key):
                        self._preview_transcripts[key] = transcript
                        yield PartialTranscription(
                            text=transcript,
                            turn_id=vad_audio.turn_id,
                            turn_revision=vad_audio.turn_revision,
                        )
            finally:
                self._untrack_active(response)
                response.close()
        except Exception:
            logger.exception("Gemma progressive transcription failed for turn=%s", vad_audio.turn_id)
        finally:
            self._emit_metric(vad_audio, "transcription", "live_done", elapsed_ms=(perf_counter() - start_s) * 1000)

    @staticmethod
    def _full_buffer_tts(runtime_config: Any | None) -> bool:
        local_pipeline = getattr(runtime_config, "local_pipeline", None) or {}
        return bool(local_pipeline.get("full_buffer_tts"))

    @staticmethod
    def _live_preview_enabled(runtime_config: Any | None) -> bool:
        """Preview requests are opt-in because generative audio models are not ASR."""
        local_pipeline = getattr(runtime_config, "local_pipeline", None) or {}
        return bool(local_pipeline.get("live_transcription", False))

    @staticmethod
    def _conversation_image_urls(runtime_config: Any | None) -> list[str]:
        chat = getattr(runtime_config, "chat", None)
        buffer = getattr(chat, "buffer", None) or []
        urls: list[str] = []
        for item in buffer:
            for part in getattr(item, "content", None) or []:
                if getattr(part, "type", None) != "input_image":
                    continue
                image_url = getattr(part, "image_url", None)
                if image_url:
                    urls.append(str(image_url))
        return urls[-2:]

    def _headers(self, api_key: str | None = None) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _stream_request(
        self,
        url: str,
        payload: dict[str, Any],
        *,
        api_key: str | None,
    ) -> CancellableAsyncSSEStream:
        return CancellableAsyncSSEStream(
            "POST",
            url,
            headers=self._headers(api_key),
            json_body=payload,
            timeout=self.timeout,
        )

    def _iter_direct_responses(
        self, audio: np.ndarray, vad_audio: STTIn, *, generation: int | None = None
    ) -> Iterator[DirectAssistantResponse]:
        base_url, _, api_key = self._model_endpoint(vad_audio)
        url = f"{base_url}/chat/completions"
        encoded_audio = base64.b64encode(self._wav_bytes(audio)).decode("ascii")
        try:
            # Snapshot prior history first. Otherwise current audio appears once
            # in history and once as the live user message in the same request.
            payload = self._payload(audio, vad_audio, encoded_audio=encoded_audio)
        finally:
            # VAD admission, not optional metadata or payload serialization, is
            # the accepted-turn boundary.
            self._commit_accepted_audio(vad_audio, encoded_audio)
        response: CancellableAsyncSSEStream | None = None
        try:
            if payload.get("stream", self.stream):
                self._emit_metric(vad_audio, "gemma", "waiting_headers", detail={"operation": "direct_audio"})
                response = self._stream_request(url, payload, api_key=api_key)
                self._track_active(response, vad_audio)
                try:
                    response.wait_for_headers()
                    self._emit_metric(vad_audio, "gemma", "generating", detail={"operation": "direct_audio"})
                    yield from self._consume_stream(response, vad_audio, generation=generation)
                finally:
                    self._untrack_active(response)
                    response.close()
                return
            buffered_payload = dict(payload)
            buffered_payload["stream"] = True
            response = self._stream_request(url, buffered_payload, api_key=api_key)
            self._track_active(response, vad_audio)
            raw = ""
            tool_accum: dict[int, dict[str, str]] = {}
            try:
                response.wait_for_headers()
                for line in response.iter_lines():
                    if not line:
                        continue
                    if line.startswith("data:"):
                        line = line[len("data:") :].strip()
                    if line == "[DONE]":
                        break
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    choices = data.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    self._accumulate_tool_deltas(delta, tool_accum)
                    raw += str(delta.get("content") or choices[0].get("text") or "")
            finally:
                self._untrack_active(response)
                response.close()
            tools = self._tool_calls_from_accum(
                tool_accum,
                chat=self._conversation_chat(vad_audio),
                turn_id=getattr(vad_audio, "turn_id", None),
            )
            text = raw
            yield from self._responses_from_text(text, vad_audio, tools=tools, generation=generation)
        except (httpx.HTTPError, RuntimeError):
            tracker = getattr(self, "speculative_turns", None)
            superseded = tracker is not None and not tracker.is_latest(
                vad_audio.turn_id,
                vad_audio.turn_revision,
            )
            cancelled = (
                generation is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(generation)
            )
            if superseded or cancelled:
                self._emit_metric(vad_audio, "gemma", "cancelled", detail={"operation": "direct_audio"})
                logger.info(
                    "Gemma audio transport closed for superseded request turn=%s rev=%s",
                    vad_audio.turn_id,
                    vad_audio.turn_revision,
                )
                return
            raise
        finally:
            if response is not None:
                response.close()

    def _consume_stream(
        self, response: CancellableAsyncSSEStream, vad_audio: STTIn, *, generation: int | None = None
    ) -> Iterator[DirectAssistantResponse]:
        raw_text = ""
        tool_accum: dict[int, dict[str, str]] = {}
        assistant_started = False
        pending_response = ""
        assistant_response_closed = False
        transcript_value: str | None = None
        language_code: str | None = None
        camera_current = False
        suppress_response_stream = False
        full_buffer_tts = self._full_buffer_tts(getattr(vad_audio, "runtime_config", None))
        for line in response.iter_lines():
            if generation is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(generation):
                logger.info("Gemma audio request cancelled for turn=%s", vad_audio.turn_id)
                response.close()
                return
            if not line:
                continue
            if line.startswith("data:"):
                line = line[len("data:") :].strip()
            if line == "[DONE]":
                break
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("Ignoring non-JSON Gemma stream event (chars=%d)", len(line))
                continue
            choices = data.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            self._accumulate_tool_deltas(delta, tool_accum)
            content = delta.get("content") or choices[0].get("text")
            if not content:
                continue
            raw_text += str(content)
            if not assistant_started and _RESPONSE_MARKER in raw_text:
                before, after = raw_text.split(_RESPONSE_MARKER, 1)
                transcript = self._extract_transcript(before)
                transcript_value = transcript
                language_code = self._effective_assistant_language(before)
                camera_available = self._camera_tool_available(vad_audio)
                camera_context = self._extract_camera_context(before)
                camera_current = camera_available and camera_context == "current"
                # Hold camera-capable responses until the complete control
                # envelope is available. A duplicate or conflicting marker may
                # arrive later and must not retroactively invalidate prose
                # already sent to TTS.
                suppress_response_stream = camera_available
                if transcript:
                    # Transcript metadata only upgrades the already persisted
                    # accepted-audio item; it is not the admission boundary.
                    user_committed = self._ensure_user_context(vad_audio, transcript) is not None
                    yield self._direct(
                        vad_audio,
                        "",
                        transcript=transcript,
                        is_final=False,
                        context_committed=user_committed,
                        transcript_finalized=True,
                        language_code=language_code,
                        generation=generation,
                    )
                assistant_started = True
                pending_response = after
            elif assistant_started and not assistant_response_closed:
                pending_response += str(content)
            if assistant_started and not assistant_response_closed:
                pending_response, assistant_response_closed = self._visible_response_prefix(pending_response)
            if assistant_started and not full_buffer_tts and not suppress_response_stream:
                chunks, pending_response = self._pop_sentence_chunks(pending_response)
                for chunk in chunks:
                    yield self._direct(
                        vad_audio,
                        chunk,
                        is_final=False,
                        language_code=language_code,
                        generation=generation,
                    )
        tools = self._tool_calls_from_accum(
            tool_accum,
            chat=self._conversation_chat(vad_audio),
            turn_id=getattr(vad_audio, "turn_id", None),
        )
        tools, camera_context, _ = self._enforce_camera_context(vad_audio, raw_text, tools)
        camera_current = camera_context == "current"
        if generation is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(generation):
            return
        preamble = self._tool_preamble(raw_text, tools) if tools else None
        final_text = (
            preamble
            if tools and preamble
            else ""
            if camera_current
            else pending_response.strip()
            if assistant_started
            else self._fallback_response_text(raw_text)
        )
        transcript = self._validate_transcript(transcript_value or self._extract_transcript(raw_text))
        user_memory = None if transcript else self._extract_user_memory(raw_text)
        self._emit_metric(
            vad_audio,
            "transcription",
            "transcript_available" if transcript else "transcript_unavailable",
            detail={"mode": "final", "source": "primary"},
        )
        full_response = preamble or ("" if camera_current else self._fallback_response_text(raw_text))
        language_code = language_code or self._effective_assistant_language(raw_text)
        committed = self._commit_context(
            vad_audio,
            transcript,
            full_response,
            tools,
            user_memory=user_memory,
        )
        self._finish_user_context(vad_audio)
        self._preview_transcripts.pop((vad_audio.turn_id, vad_audio.turn_revision), None)
        if final_text:
            logger.info("Gemma audio response ready (%d characters)", len(final_text))
        yield self._direct(
            vad_audio,
            final_text,
            transcript=transcript,
            tools=tools,
            is_final=True,
            context_committed=committed,
            language_code=language_code,
            generation=generation,
        )

    def _responses_from_text(
        self,
        text: str,
        vad_audio: STTIn,
        *,
        tools: list[ResponseFunctionToolCall] | None = None,
        generation: int | None = None,
    ) -> Iterator[DirectAssistantResponse]:
        transcript = self._extract_transcript(text)
        user_memory = None if transcript else self._extract_user_memory(text)
        tools = tools or []
        tools, camera_context, _ = self._enforce_camera_context(vad_audio, text, tools)
        language_code = self._effective_assistant_language(text)
        preamble = self._tool_preamble(text, tools) if tools else None
        self._emit_metric(
            vad_audio,
            "transcription",
            "transcript_available" if transcript else "transcript_unavailable",
            detail={"mode": "final", "source": "primary"},
        )
        response_text = preamble or ("" if camera_context == "current" else self._fallback_response_text(text))
        if response_text:
            logger.info("Gemma audio response ready (%d characters)", len(response_text))
        committed = self._commit_context(
            vad_audio,
            transcript,
            response_text,
            tools,
            user_memory=user_memory,
        )
        self._finish_user_context(vad_audio)
        self._preview_transcripts.pop((vad_audio.turn_id, vad_audio.turn_revision), None)
        yield self._direct(
            vad_audio,
            response_text,
            transcript=transcript,
            tools=tools,
            is_final=True,
            context_committed=committed,
            language_code=language_code,
            generation=generation,
        )

    def _direct(
        self,
        vad_audio: STTIn,
        text: str,
        *,
        transcript: str | None = None,
        tools: list[ResponseFunctionToolCall] | None = None,
        is_final: bool,
        context_committed: bool = False,
        transcript_finalized: bool = False,
        language_code: str | None = None,
        generation: int | None = None,
        error: str | None = None,
    ) -> DirectAssistantResponse:
        runtime_config = getattr(vad_audio, "runtime_config", None)
        if language_code and runtime_config is not None:
            runtime_config.local_pipeline["assistant_language"] = language_code
        return DirectAssistantResponse(
            text=text,
            transcript=transcript,
            is_final=is_final,
            tools=tools or [],
            language_code=language_code,
            turn_id=vad_audio.turn_id,
            turn_revision=vad_audio.turn_revision,
            speech_stopped_at_s=vad_audio.created_at_s,
            runtime_config=getattr(vad_audio, "runtime_config", None),
            context_committed=context_committed,
            transcript_finalized=transcript_finalized,
            cancel_generation=generation,
            error=error,
        )

    def _commit_context(
        self,
        vad_audio: STTIn,
        transcript: str | None,
        assistant_text: str,
        tools: list[ResponseFunctionToolCall],
        *,
        user_memory: str | None = None,
    ) -> bool:
        chat = self._conversation_chat(vad_audio)
        if chat is None:
            return False
        transcript = self._validate_transcript(transcript)
        user_memory = None if transcript else self._validate_user_memory(user_memory)
        user_item_id = self._ensure_user_context(vad_audio, transcript, user_memory)
        if user_item_id is None:
            self._emit_history_metric(vad_audio, "response_uncommitted", input_kind="missing", committed=False)
            return False
        function_calls = [
            RealtimeConversationItemFunctionCall(
                type="function_call",
                name=tool.name,
                arguments=tool.arguments,
                call_id=tool.call_id,
                id=tool.id,
                status=tool.status,
            )
            for tool in tools
        ]
        key = self._turn_key(vad_audio)
        revision = self._turn_revision(vad_audio)
        with self._accepted_user_lock:
            owned = self._accepted_user_items.get(key)
            if owned != (user_item_id, revision):
                self._emit_history_metric(vad_audio, "response_uncommitted", input_kind="missing", committed=False)
                return False
            # Keep ownership validation and assistant/tool persistence in one
            # transaction. A newer cumulative revision cannot claim the anchor
            # between these two operations and receive a stale response.
            chat.commit_assistant_response(user_item_id, assistant_text, function_calls)
            # Keep unresolved function-call pairs intact; the normal tool
            # follow-up path trims after its final assistant response commits.
            if not tools:
                chat.trim_if_needed(None)
        self._emit_history_metric(
            vad_audio,
            "response_committed",
            input_kind="transcript" if transcript else "semantic_memory" if user_memory else "input_audio",
            committed=True,
        )
        return True

    @staticmethod
    def _extract_transcript(text: str) -> str | None:
        match = _FINAL_TRANSCRIPT_RE.search(text)
        if not match:
            return None
        output = _OUTPUT_MARKER_RE.search(text)
        if output is not None and match.start() > output.start():
            return None
        return GemmaAudioSTTHandler._validate_transcript(match.group(1))

    @staticmethod
    def _validate_transcript(value: str | None) -> str | None:
        if not value:
            return None
        transcript = str(value).strip().strip('"')
        if not transcript or "\n" in transcript or len(transcript) > 1200:
            return None
        failure_key = _failure_sentinel_key(transcript)
        if failure_key in _TRANSCRIPT_FAILURE_SENTINELS:
            return None
        return transcript

    @staticmethod
    def _extract_user_memory(text: str) -> str | None:
        match = _FINAL_MEMORY_RE.search(text)
        if not match:
            return None
        output = _OUTPUT_MARKER_RE.search(text)
        if output is not None and match.start() > output.start():
            return None
        return GemmaAudioSTTHandler._validate_user_memory(match.group(1))

    @staticmethod
    def _validate_user_memory(value: str | None) -> str | None:
        if not value:
            return None
        raw_memory = str(value).strip().strip('"')
        if (
            not raw_memory
            or len(raw_memory) > _MAX_USER_MEMORY_CHARS
            or "\n" in raw_memory
            or "\r" in raw_memory
            or any(ord(character) < 32 for character in raw_memory)
        ):
            return None
        memory = " ".join(raw_memory.split())
        failure_key = _failure_sentinel_key(memory)
        if failure_key in _TRANSCRIPT_FAILURE_SENTINELS or _MEMORY_CONTROL_MARKER_RE.search(memory):
            return None
        return memory

    def on_session_end(self) -> None:
        super().on_session_end()
        with self._accepted_user_lock:
            self._accepted_user_items.clear()
        self._preview_transcripts.clear()

    @staticmethod
    def _extract_preview_transcript(text: str) -> str | None:
        """Return only a transcript-only preview with the required marker.

        Preview calls are intentionally fail-closed: assistant prose is worse
        than an absent provisional bubble, because the bubble represents what
        the user said rather than a model answer.
        """
        if _PREVIEW_TRANSCRIPT_MARKER not in text:
            return None
        prefix, transcript = text.split(_PREVIEW_TRANSCRIPT_MARKER, 1)
        if (
            prefix.strip()
            or _TRANSCRIPT_MARKER in transcript
            or _MEMORY_MARKER in transcript
            or _RESPONSE_MARKER in transcript
            or _CAMERA_CONTEXT_MARKER in transcript
        ):
            return None
        transcript = transcript.strip().strip('"')
        return GemmaAudioSTTHandler._validate_transcript(transcript)

    @staticmethod
    def _extract_assistant_language(text: str) -> str | None:
        matches = _ASSISTANT_LANGUAGE_RE.findall(text)
        if len(matches) != 1:
            return None
        value = matches[0].strip().strip('"')
        return value if value and len(value) <= 40 else None

    @staticmethod
    def _canonical_assistant_language(value: str | None) -> str:
        if not value:
            return "Auto"
        key = _failure_sentinel_key(unicodedata.normalize("NFKC", str(value)).replace("_", "-"))
        return _ASSISTANT_LANGUAGE_ALIASES.get(key, "Auto")

    @staticmethod
    def _effective_assistant_language(text: str) -> str:
        """Return one turn-scoped TTS language without an implicit English pin.

        A monolingual answer may name its language explicitly. Mixed-language
        or otherwise unspecified output remains ``Auto`` so the selected Qwen3
        clone backend can interpret the actual text rather than inheriting a
        previous turn or the clone reference metadata.
        """

        return GemmaAudioSTTHandler._canonical_assistant_language(
            GemmaAudioSTTHandler._extract_assistant_language(text)
        )

    @staticmethod
    def _camera_tool_available(vad_audio: STTIn) -> bool:
        runtime_config = getattr(vad_audio, "runtime_config", None)
        session = getattr(runtime_config, "session", None)
        for tool in getattr(session, "tools", None) or []:
            if isinstance(tool, dict):
                name = tool.get("name")
                if not name and isinstance(tool.get("function"), dict):
                    name = tool["function"].get("name")
            else:
                name = getattr(tool, "name", None)
                if not name:
                    name = getattr(getattr(tool, "function", None), "name", None)
            if name == "camera_snapshot":
                return True
        return False

    @staticmethod
    def _extract_camera_context(text: str) -> str | None:
        matches = _CAMERA_CONTEXT_RE.findall(text)
        if len(matches) != 1:
            return None
        value = _failure_sentinel_key(matches[0].strip().strip('"'))
        return value if value in {"current", "historical", "none"} else None

    def _enforce_camera_context(
        self,
        vad_audio: STTIn,
        text: str,
        tools: list[ResponseFunctionToolCall],
    ) -> tuple[list[ResponseFunctionToolCall], str | None, bool]:
        """Enforce a model-declared live-view requirement without inferring from content.

        Missing or malformed classification is privacy fail-closed: it never
        causes a capture. Metrics contain only the enum/call identity.
        """

        if not self._camera_tool_available(vad_audio):
            return tools, None, False
        camera_context = self._extract_camera_context(text)
        if camera_context is None:
            self._emit_metric(
                vad_audio,
                "camera",
                "freshness_unclassified",
                detail={"camera_context": "unclassified", "enforced": False},
            )
            return tools, None, False
        if camera_context != "current":
            self._emit_metric(
                vad_audio,
                "camera",
                "freshness_classified",
                detail={"camera_context": camera_context, "enforced": False},
            )
            return tools, camera_context, False

        first_camera = next((tool for tool in tools if tool.name == "camera_snapshot"), None)
        if first_camera is None:
            synthetic_index = len(tools)
            injected = self._tool_calls_from_accum(
                {synthetic_index: {"name": "camera_snapshot", "args": "{}", "id": ""}},
                chat=self._conversation_chat(vad_audio),
                turn_id=getattr(vad_audio, "turn_id", None),
            )[0]
            tools = [*tools, injected]
            first_camera = injected
            enforced = True
        else:
            # One current observation needs exactly one frame. Preserve all
            # non-camera calls and the first model-provided camera call.
            retained: list[ResponseFunctionToolCall] = []
            camera_retained = False
            for tool in tools:
                if tool.name != "camera_snapshot":
                    retained.append(tool)
                elif not camera_retained:
                    retained.append(tool)
                    camera_retained = True
            tools = retained
            enforced = False
        self._emit_metric(
            vad_audio,
            "camera",
            "freshness_enforced" if enforced else "freshness_satisfied",
            detail={
                "camera_context": camera_context,
                "enforced": enforced,
                "call_id": first_camera.call_id,
            },
        )
        return tools, camera_context, enforced

    @staticmethod
    def _extract_assistant_preamble(text: str) -> str | None:
        match = _ASSISTANT_PREAMBLE_RE.search(text)
        if not match:
            return None
        value = " ".join(match.group(1).strip().split())
        if not value or len(value) > 280:
            return None
        return value

    @classmethod
    def _tool_preamble(cls, text: str, tools: list[ResponseFunctionToolCall]) -> str | None:
        """Return only an explicit model-provided native-tool lead-in."""
        if not tools:
            return None
        return cls._extract_assistant_preamble(text)

    @staticmethod
    def _fallback_response_text(text: str) -> str:
        assistant = _ASSISTANT_RESPONSE_RE.search(text)
        if assistant:
            return assistant.group(1).strip()
        return _CONTROL_LINE_RE.sub("", text).strip()

    @staticmethod
    def _visible_response_prefix(text: str) -> tuple[str, bool]:
        """Return response prose before a trailing protocol-control line."""

        match = _TRAILING_RESPONSE_CONTROL_RE.search(text)
        if not match:
            return text, False
        return text[: match.start()].rstrip(), True

    @staticmethod
    def _pop_sentence_chunks(text: str) -> tuple[list[str], str]:
        chunks: list[str] = []
        pos = 0
        for match in _SENTENCE_RE.finditer(text):
            chunk = match.group(1).strip()
            if chunk:
                chunks.append(chunk)
            pos = match.end()
        return chunks, text[pos:]

    @staticmethod
    def _accumulate_tool_deltas(delta: dict[str, Any], tool_accum: dict[int, dict[str, str]]) -> None:
        for tc in delta.get("tool_calls") or []:
            index = int(tc.get("index") or 0)
            entry = tool_accum.setdefault(index, {"name": "", "args": "", "id": ""})
            if tc.get("id"):
                entry["id"] = str(tc["id"])
            fn = tc.get("function") or {}
            if fn.get("name"):
                entry["name"] = str(fn["name"])
            if fn.get("arguments"):
                entry["args"] += str(fn["arguments"])

    @staticmethod
    def _tool_calls_from_accum(
        tool_accum: dict[int, dict[str, str]],
        *,
        chat: Any | None = None,
        turn_id: str | None = None,
    ) -> list[ResponseFunctionToolCall]:
        tools: list[ResponseFunctionToolCall] = []
        used_call_ids: set[str] = set()
        safe_turn_id = re.sub(r"[^A-Za-z0-9_-]+", "_", str(turn_id or "turn")).strip("_")[:48] or "turn"
        for index in sorted(tool_accum):
            entry = tool_accum[index]
            if not entry["name"]:
                continue
            raw_call_id = entry["id"].strip()
            fallback_call_id = f"call_{safe_turn_id}_{index}"
            if raw_call_id and re.fullmatch(r"call_[A-Za-z0-9_-]+", raw_call_id):
                call_id = raw_call_id
                source = "native"
            else:
                call_id = fallback_call_id
                source = "canonical"
            if chat is not None and callable(getattr(chat, "canonical_call_id", None)):
                canonical = chat.canonical_call_id(call_id, used_call_ids)
                if source == "native" and canonical != call_id:
                    call_id = fallback_call_id
                    canonical = chat.canonical_call_id(call_id, used_call_ids)
                    source = "canonical_reused"
            else:
                canonical = call_id
                suffix = 1
                while canonical in used_call_ids:
                    canonical = f"{call_id}_{suffix}"
                    suffix += 1
            if canonical != call_id:
                source = f"{source}_deduplicated"
            call_id = canonical
            used_call_ids.add(call_id)
            logger.info(
                "Direct audio tool call prepared (stage=adapter name=%s call_id=%s source=%s)",
                entry["name"],
                call_id,
                source,
            )
            tools.append(
                ResponseFunctionToolCall(
                    type="function_call",
                    name=entry["name"],
                    arguments=entry["args"],
                    call_id=call_id,
                    id=_generate_id("fc"),
                    status="completed",
                )
            )
        return tools

    def cancel_active(self) -> None:
        with self._active_response_lock:
            resources = tuple(self._active_resources)
            self._active_resources.clear()
            self._active_turn = None
        for resource in resources:
            try:
                resource.close()
            except Exception:
                logger.debug("Gemma audio transport was already closed during cancellation")

    def _track_active(self, resource: Any, vad_audio: STTIn) -> None:
        with self._active_response_lock:
            self._active_resources.add(resource)
            self._active_turn = (vad_audio.turn_id, vad_audio.turn_revision)

    def _untrack_active(self, resource: Any) -> None:
        with self._active_response_lock:
            self._active_resources.discard(resource)
            if not self._active_resources:
                self._active_turn = None

    def _message_text_and_tools(self, data: dict[str, Any]) -> tuple[str, list[ResponseFunctionToolCall]]:
        choices = data.get("choices") or []
        if not choices:
            return "", []
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        else:
            text = str(content or "")
        tool_accum: dict[int, dict[str, str]] = {}
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function") or {}
            tool_accum[len(tool_accum)] = {
                "name": str(fn.get("name") or ""),
                "args": str(fn.get("arguments") or "{}"),
                "id": str(tc.get("id") or ""),
            }
        return text, self._tool_calls_from_accum(tool_accum)
