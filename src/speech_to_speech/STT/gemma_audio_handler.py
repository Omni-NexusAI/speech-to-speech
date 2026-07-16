from __future__ import annotations

import base64
import json
import logging
import os
import re
import threading
import wave
from collections.abc import Iterator
from io import BytesIO
from time import perf_counter, time
from typing import Any

import httpx
import numpy as np
from openai.types.realtime.conversation_item import RealtimeConversationItemFunctionCall
from openai.types.responses import ResponseFunctionToolCall

from speech_to_speech.LLM.chat import make_assistant_message, make_user_message
from speech_to_speech.LLM.chat_completions_language_model import (
    ChatCompletionsApiModelHandler,
    _to_chat_tool_choice,
    _to_chat_tools,
)
from speech_to_speech.pipeline.events import PipelineMetricEvent
from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import DirectAssistantResponse, PartialTranscription
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler
from speech_to_speech.utils.utils import _generate_id

logger = logging.getLogger(__name__)
_TRANSCRIPT_MARKER = "USER_TRANSCRIPT:"
_RESPONSE_MARKER = "ASSISTANT_RESPONSE:"
_LANGUAGE_MARKER = "ASSISTANT_LANGUAGE:"
_PREVIEW_TRANSCRIPT_MARKER = "TRANSCRIPT:"
_FINAL_TRANSCRIPT_RE = re.compile(
    r"(?ims)^\s*(?:USER_TRANSCRIPT|USER_SPEECH|TRANSCRIPT|USER)\s*:\s*(.+?)"
    r"(?=^\s*(?:ASSISTANT_LANGUAGE|ASSISTANT_RESPONSE|ASSISTANT|RESPONSE)\s*:|\Z)"
)
_ASSISTANT_RESPONSE_RE = re.compile(
    r"(?ims)^\s*(?:ASSISTANT_RESPONSE|ASSISTANT|RESPONSE)\s*:\s*(.+)\Z"
)
_ASSISTANT_LANGUAGE_RE = re.compile(r"(?im)^\s*ASSISTANT_LANGUAGE\s*:\s*([^\r\n]+)")
_SENTENCE_RE = re.compile(r"(.+?[.!?](?:\s+|$))", re.DOTALL)
_TRANSCRIPT_CONTROL_TEXT = (
    "listen to the attached user audio",
    "respond directly as a concise voice assistant",
    "do not include a transcript",
    "audio content appears to be non-verbal",
    "audio content appears to be unintelligible",
    "user_transcript:",
    "assistant_response:",
)


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
        self._preview_transcripts: dict[tuple[str | None, int | None], str] = {}
        self._committed_user_turns: set[tuple[str | None, int | None]] = set()
        self._active_response: httpx.Response | None = None
        self._active_client: httpx.Client | None = None
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
            self.cancel_active()

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        if vad_audio.mode == "progressive":
            if not self._live_preview_enabled(getattr(vad_audio, "runtime_config", None)):
                return
            yield from self._iter_progressive_transcriptions(vad_audio)
            return
        start_s = perf_counter()
        audio = self._as_float32_mono(vad_audio.audio)
        duration_s = len(audio) / self.sample_rate if self.sample_rate else 0.0
        logger.info("Gemma audio direct request start turn=%s rev=%s audio=%.3fs", vad_audio.turn_id, vad_audio.turn_revision, duration_s)
        full_buffer_tts = self._full_buffer_tts(getattr(vad_audio, "runtime_config", None))
        self._emit_metric(
            vad_audio,
            "gemma",
            "request_start",
            detail={"audio_s": round(duration_s, 3), "full_buffer_tts": full_buffer_tts},
        )
        self._emit_metric(vad_audio, "transcription", "active", detail={"mode": "final"})
        first = True
        generation = self.cancel_scope.generation if self.cancel_scope is not None else None
        try:
            for response in self._iter_direct_responses(audio, vad_audio, generation=generation):
                if first and (response.text or response.tools):
                    self._emit_metric(vad_audio, "gemma", "first_token", elapsed_ms=(perf_counter() - start_s) * 1000)
                    first = False
                yield response
        finally:
            total_s = perf_counter() - start_s
            logger.info("Gemma audio direct request done turn=%s rev=%s total=%.3fs", vad_audio.turn_id, vad_audio.turn_revision, total_s)
            self._emit_metric(vad_audio, "gemma", "done", elapsed_ms=total_s * 1000)

    def _emit_metric(self, vad_audio: STTIn, stage: str, status: str, *, elapsed_ms: float | None = None, detail: dict[str, Any] | None = None) -> None:
        if self.text_output_queue is not None:
            self.text_output_queue.put(PipelineMetricEvent(stage=stage, status=status, at_s=time(), elapsed_ms=elapsed_ms, turn_id=vad_audio.turn_id, turn_revision=vad_audio.turn_revision, detail=detail or {}))
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

    def _model_endpoint(self, vad_audio: STTIn | None) -> tuple[str, str, str | None]:
        runtime_config = getattr(vad_audio, "runtime_config", None) if vad_audio is not None else None
        endpoint = getattr(runtime_config, "model_endpoint", None)
        if endpoint is None:
            return self.base_url, self.model_name, self.api_key
        return endpoint.base_url.rstrip("/"), endpoint.model, endpoint.api_key

    def _payload(self, audio: np.ndarray, vad_audio: STTIn | None = None) -> dict[str, Any]:
        encoded = base64.b64encode(self._wav_bytes(audio)).decode("ascii")
        if vad_audio is None:
            vad_audio = type("VadAudioShim", (), {"runtime_config": None})()
        runtime_config = getattr(vad_audio, "runtime_config", None)
        session = runtime_config.session if runtime_config is not None else None
        session_instructions = str(getattr(session, "instructions", "") or "").strip()
        system_parts = [self.system_prompt]
        if session_instructions:
            system_parts.append(session_instructions)
        system_parts.append(
            "Transcribe the user's audio accurately. When no tool is needed, respond in this exact plain-text "
            "shape:\nUSER_TRANSCRIPT: <short transcript of what the user said>\n"
            "ASSISTANT_LANGUAGE: <language name for the assistant response, or Auto>\n"
            "ASSISTANT_RESPONSE: <your spoken answer>\n"
            "When a provided tool is needed, call it in the same response and never fabricate its result. A short "
            "context-specific spoken acknowledgement is allowed before the function call. You may place "
            "USER_TRANSCRIPT and ASSISTANT_LANGUAGE in the "
            "tool-call message content, but do not emit a result-dependent ASSISTANT_RESPONSE until the tool result "
            "is available. Do not wrap plain-text responses in JSON or Markdown."
        )
        user_content: list[dict[str, Any]] = []
        for image_url in self._conversation_image_urls(runtime_config):
            user_content.append({"type": "image_url", "image_url": {"url": image_url}})
        user_content.append({"type": "input_audio", "input_audio": {"data": encoded, "format": self.audio_format}})

        history: list[dict[str, Any]] = []
        chat = getattr(runtime_config, "chat", None)
        if chat is not None and callable(getattr(chat, "copy", None)):
            history = [
                message
                for message in ChatCompletionsApiModelHandler._chat_messages(chat.copy())
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
                    "content": [
                        {"type": "input_audio", "input_audio": {"data": encoded, "format": self.audio_format}}
                    ],
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
        self._emit_metric(vad_audio, "transcription", "live_start", detail={"audio_s": round(len(audio) / self.sample_rate, 3)})
        raw = ""
        first = True
        try:
            base_url, _, api_key = self._model_endpoint(vad_audio)
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream(
                    "POST",
                    f"{base_url}/chat/completions",
                    headers=self._headers(api_key),
                    json=self._transcription_payload(audio, vad_audio),
                ) as response:
                    response.raise_for_status()
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
                            self._emit_metric(vad_audio, "transcription", "live_first_text", elapsed_ms=(perf_counter() - start_s) * 1000)
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
        except Exception:
            logger.exception("Gemma progressive transcription failed for turn=%s", vad_audio.turn_id)
        finally:
            self._emit_metric(vad_audio, "transcription", "live_done", elapsed_ms=(perf_counter() - start_s) * 1000)

    def _transcribe_once(self, vad_audio: STTIn) -> str | None:
        base_url, _, api_key = self._model_endpoint(vad_audio)
        payload = self._transcription_payload(self._as_float32_mono(vad_audio.audio), vad_audio)
        payload["stream"] = False
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{base_url}/chat/completions",
                    headers=self._headers(api_key),
                    json=payload,
                )
                response.raise_for_status()
            text, _ = self._message_text_and_tools(response.json())
            return self._extract_preview_transcript(text)
        except Exception:
            logger.exception("Gemma final transcription fallback failed for turn=%s", vad_audio.turn_id)
            return None

    def _final_transcript(self, vad_audio: STTIn, transcript: str | None) -> str | None:
        """Resolve the mandatory final transcript without trusting assistant text.

        The primary direct-audio response is preferred. A validated live preview
        can satisfy the final row when enabled; otherwise one transcript-only
        request is made. The fallback is deliberately isolated from TTS.
        """
        if transcript:
            resolved = self._validate_transcript(transcript)
            source = "primary"
        else:
            resolved = None
            source = "fallback"
        if not resolved:
            self._emit_metric(vad_audio, "transcription", "fallback_start", detail={"mode": "final"})
            resolved = self._validate_transcript(self._transcribe_once(vad_audio))
        self._emit_metric(
            vad_audio,
            "transcription",
            "completed" if resolved else "failed",
            detail={"mode": "final", "source": source},
        )
        return resolved

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

    def _iter_direct_responses(
        self, audio: np.ndarray, vad_audio: STTIn, *, generation: int | None = None
    ) -> Iterator[DirectAssistantResponse]:
        base_url, _, api_key = self._model_endpoint(vad_audio)
        url = f"{base_url}/chat/completions"
        payload = self._payload(audio, vad_audio)
        client = httpx.Client(timeout=self.timeout)
        with self._active_response_lock:
            self._active_client = client
            self._active_turn = (vad_audio.turn_id, vad_audio.turn_revision)
        try:
            if payload.get("stream", self.stream):
                with client.stream("POST", url, headers=self._headers(api_key), json=payload) as response:
                    with self._active_response_lock:
                        self._active_response = response
                    response.raise_for_status()
                    try:
                        yield from self._consume_stream(response, vad_audio, generation=generation)
                    finally:
                        with self._active_response_lock:
                            if self._active_response is response:
                                self._active_response = None
                    return
            response = client.post(url, headers=self._headers(api_key), json=payload)
            response.raise_for_status()
            text, tools = self._message_text_and_tools(response.json())
            yield from self._responses_from_text(text, vad_audio, tools=tools)
        except (httpx.HTTPError, RuntimeError):
            tracker = getattr(self, "speculative_turns", None)
            superseded = tracker is not None and not tracker.is_latest(
                vad_audio.turn_id,
                vad_audio.turn_revision,
            )
            cancelled = generation is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(generation)
            if superseded or cancelled:
                logger.info(
                    "Gemma audio transport closed for superseded request turn=%s rev=%s",
                    vad_audio.turn_id,
                    vad_audio.turn_revision,
                )
                return
            raise
        finally:
            with self._active_response_lock:
                if self._active_client is client:
                    self._active_client = None
                    self._active_turn = None
            client.close()

    def _consume_stream(
        self, response: httpx.Response, vad_audio: STTIn, *, generation: int | None = None
    ) -> Iterator[DirectAssistantResponse]:
        raw_text = ""
        tool_accum: dict[int, dict[str, str]] = {}
        assistant_started = False
        pending_response = ""
        transcript_value: str | None = None
        language_code: str | None = None
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
                logger.debug("Ignoring non-JSON Gemma stream line: %r", line)
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
                language_code = self._extract_assistant_language(before)
                if transcript:
                    # Establish the user transcript before assistant chunks are
                    # forwarded, so the Realtime UI and conversation chronology
                    # cannot render the assistant first.
                    user_committed = self._commit_user_context(vad_audio, transcript)
                    self._committed_user_turns.add((vad_audio.turn_id, vad_audio.turn_revision))
                    yield self._direct(
                        vad_audio,
                        "",
                        transcript=transcript,
                        is_final=False,
                        context_committed=user_committed,
                        transcript_finalized=True,
                    )
                assistant_started = True
                pending_response = after
            elif assistant_started:
                pending_response += str(content)
            if assistant_started and not full_buffer_tts:
                chunks, pending_response = self._pop_sentence_chunks(pending_response)
                for chunk in chunks:
                    yield self._direct(vad_audio, chunk, is_final=False, language_code=language_code)
        tools = self._tool_calls_from_accum(tool_accum)
        if generation is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(generation):
            return
        final_text = pending_response.strip() if assistant_started else self._fallback_response_text(raw_text)
        # Final-only mode must not make a second generative request just to
        # populate a user bubble: that can hallucinate speech and pollute chat.
        transcript = self._final_transcript(vad_audio, transcript_value or self._extract_transcript(raw_text))
        full_response = self._fallback_response_text(raw_text)
        language_code = language_code or self._extract_assistant_language(raw_text)
        if not transcript:
            self._emit_metric(vad_audio, "transcription", "failed", detail={"mode": "final", "source": "fallback"})
            # A valid native function call is still actionable when Gemma omits
            # transcript metadata. Do not fabricate user history or speak a canned
            # retry; expose the tool and let the normal call-ID transaction continue.
            committed = self._commit_context(vad_audio, None, full_response if tools else "", tools)
            yield self._direct(
                vad_audio,
                final_text if tools else "",
                tools=tools,
                is_final=True,
                context_committed=committed,
                language_code=language_code,
            )
            self._preview_transcripts.pop((vad_audio.turn_id, vad_audio.turn_revision), None)
            return
        user_key = (vad_audio.turn_id, vad_audio.turn_revision)
        committed = self._commit_context(
            vad_audio,
            transcript,
            full_response,
            tools,
            include_user=user_key not in self._committed_user_turns,
        )
        self._committed_user_turns.discard(user_key)
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
        )

    def _responses_from_text(self, text: str, vad_audio: STTIn, *, tools: list[ResponseFunctionToolCall] | None = None) -> Iterator[DirectAssistantResponse]:
        transcript = self._final_transcript(vad_audio, self._extract_transcript(text))
        tools = tools or []
        language_code = self._extract_assistant_language(text)
        if not transcript:
            self._emit_metric(vad_audio, "transcription", "failed", detail={"mode": "final", "source": "fallback"})
            response_text = self._fallback_response_text(text) if tools else ""
            committed = self._commit_context(vad_audio, None, response_text, tools)
            yield self._direct(
                vad_audio,
                response_text,
                tools=tools,
                is_final=True,
                context_committed=committed,
                language_code=language_code,
            )
            self._preview_transcripts.pop((vad_audio.turn_id, vad_audio.turn_revision), None)
            return
        response_text = self._fallback_response_text(text)
        if response_text:
            logger.info("Gemma audio response ready (%d characters)", len(response_text))
        committed = self._commit_context(vad_audio, transcript, response_text, tools)
        self._preview_transcripts.pop((vad_audio.turn_id, vad_audio.turn_revision), None)
        yield self._direct(
            vad_audio,
            response_text,
            transcript=transcript,
            tools=tools,
            is_final=True,
            context_committed=committed,
            language_code=language_code,
        )

    def _direct(self, vad_audio: STTIn, text: str, *, transcript: str | None = None, tools: list[ResponseFunctionToolCall] | None = None, is_final: bool, context_committed: bool = False, transcript_finalized: bool = False, language_code: str | None = None) -> DirectAssistantResponse:
        runtime_config = getattr(vad_audio, "runtime_config", None)
        if language_code and runtime_config is not None:
            runtime_config.local_pipeline["assistant_language"] = language_code
        return DirectAssistantResponse(text=text, transcript=transcript, is_final=is_final, tools=tools or [], language_code=language_code, turn_id=vad_audio.turn_id, turn_revision=vad_audio.turn_revision, speech_stopped_at_s=vad_audio.created_at_s, runtime_config=getattr(vad_audio, "runtime_config", None), context_committed=context_committed, transcript_finalized=transcript_finalized)

    def _commit_context(
        self,
        vad_audio: STTIn,
        transcript: str | None,
        assistant_text: str,
        tools: list[ResponseFunctionToolCall],
        *,
        include_user: bool = True,
    ) -> bool:
        runtime_config = getattr(vad_audio, "runtime_config", None)
        chat = getattr(runtime_config, "chat", None)
        if chat is None:
            return False
        committed = False
        if include_user and transcript:
            chat.add_item(make_user_message(transcript))
            committed = True
        if assistant_text:
            chat.add_item(make_assistant_message(assistant_text))
            committed = True
        for tool in tools:
            chat.add_item(
                RealtimeConversationItemFunctionCall(
                    type="function_call",
                    name=tool.name,
                    arguments=tool.arguments,
                    call_id=tool.call_id,
                    id=tool.id,
                    status=tool.status,
                )
            )
            committed = True
        # Keep unresolved function-call pairs intact; the normal tool follow-up
        # path trims after its final assistant response is committed.
        if not tools:
            chat.trim_if_needed(None)
        return committed

    @staticmethod
    def _commit_user_context(vad_audio: STTIn, transcript: str) -> bool:
        runtime_config = getattr(vad_audio, "runtime_config", None)
        chat = getattr(runtime_config, "chat", None)
        if chat is None:
            return False
        chat.add_item(make_user_message(transcript))
        return True

    @staticmethod
    def _extract_transcript(text: str) -> str | None:
        match = _FINAL_TRANSCRIPT_RE.search(text)
        if not match:
            return None
        return GemmaAudioSTTHandler._validate_transcript(match.group(1))

    @staticmethod
    def _validate_transcript(value: str | None) -> str | None:
        if not value:
            return None
        transcript = str(value).strip().strip('"')
        if not transcript or "\n" in transcript or len(transcript) > 1200:
            return None
        normalized = " ".join(transcript.lower().split())
        if transcript.startswith("[") and transcript.endswith("]"):
            return None
        if any(control in normalized for control in _TRANSCRIPT_CONTROL_TEXT):
            return None
        if normalized.startswith(("system:", "assistant:", "response:", "user:")):
            return None
        return transcript

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
        if prefix.strip() or _TRANSCRIPT_MARKER in transcript or _RESPONSE_MARKER in transcript:
            return None
        transcript = transcript.strip().strip('"')
        return GemmaAudioSTTHandler._validate_transcript(transcript)

    @staticmethod
    def _extract_assistant_language(text: str) -> str | None:
        match = _ASSISTANT_LANGUAGE_RE.search(text)
        if not match:
            return None
        value = match.group(1).strip().strip('"')
        return value if value and len(value) <= 40 else None

    @staticmethod
    def _fallback_response_text(text: str) -> str:
        assistant = _ASSISTANT_RESPONSE_RE.search(text)
        if assistant:
            return assistant.group(1).strip()
        if _FINAL_TRANSCRIPT_RE.search(text):
            return ""
        return text.strip()

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
    def _tool_calls_from_accum(tool_accum: dict[int, dict[str, str]]) -> list[ResponseFunctionToolCall]:
        tools: list[ResponseFunctionToolCall] = []
        used_call_ids: set[str] = set()
        for index in sorted(tool_accum):
            entry = tool_accum[index]
            if not entry["name"]:
                continue
            raw_call_id = entry["id"].strip()
            if not raw_call_id:
                call_id = _generate_id("call")
                source = "generated"
            elif raw_call_id.startswith("call_"):
                call_id = raw_call_id
                source = "native"
            else:
                # llama.cpp may return opaque tool IDs while the Realtime chat
                # contract requires call_* IDs. Normalize only at this adapter
                # boundary, then use the same value for every later transaction.
                call_id = f"call_{raw_call_id}"
                source = "normalized"
            if call_id in used_call_ids:
                base_call_id = call_id
                suffix = index
                while call_id in used_call_ids:
                    call_id = f"{base_call_id}_{suffix}"
                    suffix += 1
                source = f"{source}_deduplicated"
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
                    arguments=entry["args"] or "{}",
                    call_id=call_id,
                    id=_generate_id("fc"),
                    status="completed",
                )
            )
        return tools

    def cancel_active(self) -> None:
        with self._active_response_lock:
            response = self._active_response
            client = self._active_client
            self._active_response = None
            self._active_client = None
            self._active_turn = None
        for resource in (response, client):
            if resource is not None:
                try:
                    resource.close()
                except Exception:
                    logger.debug("Gemma audio transport was already closed during cancellation")

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
            tool_accum[len(tool_accum)] = {"name": str(fn.get("name") or ""), "args": str(fn.get("arguments") or "{}"), "id": str(tc.get("id") or "")}
        return text, self._tool_calls_from_accum(tool_accum)
