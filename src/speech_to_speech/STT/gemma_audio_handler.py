from __future__ import annotations

import base64
import json
import logging
import os
import re
import wave
from collections.abc import Iterator
from io import BytesIO
from time import perf_counter, time
from typing import Any

import httpx
import numpy as np
from openai.types.realtime.conversation_item import RealtimeConversationItemFunctionCall
from openai.types.responses import ResponseFunctionToolCall
from rich.console import Console

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
console = Console()
_TRANSCRIPT_MARKER = "USER_TRANSCRIPT:"
_RESPONSE_MARKER = "ASSISTANT_RESPONSE:"
_PREVIEW_TRANSCRIPT_MARKER = "TRANSCRIPT:"
_SENTENCE_RE = re.compile(r"(.+?[.!?](?:\s+|$))", re.DOTALL)


class GemmaAudioSTTHandler(BaseSTTHandler):
    """STT-slot adapter that sends completed audio turns directly to Gemma."""

    def setup(
        self,
        model_name: str = "gemma-4-12b-it-qat",
        base_url: str = "http://127.0.0.1:8818/v1",
        api_key: str | None = None,
        stream: bool = True,
        timeout_s: float = 120.0,
        format: str = "wav",
        prompt: str = "Listen to the attached user audio and respond directly as a concise voice assistant.",
        system_prompt: str = "You are a local low-latency voice assistant. Answer naturally for speech synthesis.",
        gen_kwargs: dict[str, Any] | None = None,
        text_output_queue: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.base_url = (os.getenv("GEMMA_AUDIO_BASE_URL") or base_url).rstrip("/")
        self.api_key = api_key or os.getenv("GEMMA_API_KEY") or os.getenv("LLAMA_CPP_API_KEY")
        self.stream = stream
        self.timeout = httpx.Timeout(float(timeout_s), connect=10.0)
        self.audio_format = format
        self.prompt = prompt
        self.system_prompt = system_prompt
        self.gen_kwargs = gen_kwargs or {}
        self.sample_rate = 16000
        self.text_output_queue: Any | None = text_output_queue
        self._preview_transcripts: dict[tuple[str | None, int | None], str] = {}
        logger.info("Gemma audio direct mode configured for %s at %s", self.model_name, self.base_url)

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
        first = True
        try:
            for response in self._iter_direct_responses(audio, vad_audio):
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
            "ASSISTANT_RESPONSE: <your spoken answer>\n"
            "When a provided tool is needed, call it in the same response and never fabricate its result. A short "
            "spoken acknowledgement is allowed before the function call. You may place USER_TRANSCRIPT in the "
            "tool-call message content, but do not emit a result-dependent ASSISTANT_RESPONSE until the tool result "
            "is available. Do not wrap plain-text responses in JSON or Markdown."
        )
        user_content: list[dict[str, Any]] = [{"type": "text", "text": self.prompt}]
        for image_url in self._conversation_image_urls(runtime_config):
            user_content.append({"type": "image_url", "image_url": {"url": image_url}})
        user_content.append({"type": "input_audio", "input_audio": {"data": encoded, "format": self.audio_format}})

        full_buffer_tts = self._full_buffer_tts(runtime_config)
        history: list[dict[str, Any]] = []
        chat = getattr(runtime_config, "chat", None)
        if chat is not None and callable(getattr(chat, "copy", None)):
            history = [
                message
                for message in ChatCompletionsApiModelHandler._chat_messages(chat.copy())
                if message.get("role") != "system"
            ]

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "\n\n".join(system_parts)},
                *history,
                {"role": "user", "content": user_content},
            ],
            "stream": self.stream and not full_buffer_tts,
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

    def _transcription_payload(self, audio: np.ndarray) -> dict[str, Any]:
        encoded = base64.b64encode(self._wav_bytes(audio)).decode("ascii")
        return {
            "model": self.model_name,
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
        self._emit_metric(vad_audio, "gemma_preview", "request_start", detail={"audio_s": round(len(audio) / self.sample_rate, 3)})
        raw = ""
        first = True
        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    json=self._transcription_payload(audio),
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
                            self._emit_metric(vad_audio, "gemma_preview", "first_token", elapsed_ms=(perf_counter() - start_s) * 1000)
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
            self._emit_metric(vad_audio, "gemma_preview", "done", elapsed_ms=(perf_counter() - start_s) * 1000)

    def _transcribe_once(self, vad_audio: STTIn) -> str | None:
        payload = self._transcription_payload(self._as_float32_mono(vad_audio.audio))
        payload["stream"] = False
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                )
                response.raise_for_status()
            text, _ = self._message_text_and_tools(response.json())
            return self._extract_preview_transcript(text)
        except Exception:
            logger.exception("Gemma final transcription fallback failed for turn=%s", vad_audio.turn_id)
            return None

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
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _iter_direct_responses(self, audio: np.ndarray, vad_audio: STTIn) -> Iterator[DirectAssistantResponse]:
        url = f"{self.base_url}/chat/completions"
        payload = self._payload(audio, vad_audio)
        with httpx.Client(timeout=self.timeout) as client:
            if payload.get("stream", self.stream):
                with client.stream("POST", url, headers=self._headers(), json=payload) as response:
                    response.raise_for_status()
                    yield from self._consume_stream(response, vad_audio)
                    return
            response = client.post(url, headers=self._headers(), json=payload)
            response.raise_for_status()
            text, tools = self._message_text_and_tools(response.json())
            yield from self._responses_from_text(text, vad_audio, tools=tools)

    def _consume_stream(self, response: httpx.Response, vad_audio: STTIn) -> Iterator[DirectAssistantResponse]:
        raw_text = ""
        tool_accum: dict[int, dict[str, str]] = {}
        assistant_started = False
        pending_response = ""
        transcript_value: str | None = None
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
                if transcript:
                    # Establish the user transcript before assistant chunks are
                    # forwarded, so the Realtime UI and conversation chronology
                    # cannot render the assistant first.
                    yield self._direct(vad_audio, "", transcript=transcript, is_final=False)
                assistant_started = True
                pending_response = after
            elif assistant_started:
                pending_response += str(content)
            if assistant_started:
                chunks, pending_response = self._pop_sentence_chunks(pending_response)
                for chunk in chunks:
                    yield self._direct(vad_audio, chunk, is_final=False)
        tools = self._tool_calls_from_accum(tool_accum)
        final_text = pending_response.strip() if assistant_started else self._fallback_response_text(raw_text)
        # Final-only mode must not make a second generative request just to
        # populate a user bubble: that can hallucinate speech and pollute chat.
        transcript = transcript_value or self._extract_transcript(raw_text)
        full_response = self._fallback_response_text(raw_text)
        committed = self._commit_context(vad_audio, transcript, full_response, tools)
        self._preview_transcripts.pop((vad_audio.turn_id, vad_audio.turn_revision), None)
        if final_text:
            console.print(f"[yellow]GEMMA AUDIO: {final_text}")
        yield self._direct(
            vad_audio,
            final_text,
            transcript=transcript,
            tools=tools,
            is_final=True,
            context_committed=committed,
        )

    def _responses_from_text(self, text: str, vad_audio: STTIn, *, tools: list[ResponseFunctionToolCall] | None = None) -> Iterator[DirectAssistantResponse]:
        transcript = self._extract_transcript(text)
        response_text = self._fallback_response_text(text)
        if response_text:
            console.print(f"[yellow]GEMMA AUDIO: {response_text}")
        tools = tools or []
        committed = self._commit_context(vad_audio, transcript, response_text, tools)
        self._preview_transcripts.pop((vad_audio.turn_id, vad_audio.turn_revision), None)
        yield self._direct(
            vad_audio,
            response_text,
            transcript=transcript,
            tools=tools,
            is_final=True,
            context_committed=committed,
        )

    def _direct(self, vad_audio: STTIn, text: str, *, transcript: str | None = None, tools: list[ResponseFunctionToolCall] | None = None, is_final: bool, context_committed: bool = False) -> DirectAssistantResponse:
        return DirectAssistantResponse(text=text, transcript=transcript, is_final=is_final, tools=tools or [], language_code=None, turn_id=vad_audio.turn_id, turn_revision=vad_audio.turn_revision, speech_stopped_at_s=vad_audio.created_at_s, runtime_config=getattr(vad_audio, "runtime_config", None), context_committed=context_committed)

    def _commit_context(
        self,
        vad_audio: STTIn,
        transcript: str | None,
        assistant_text: str,
        tools: list[ResponseFunctionToolCall],
    ) -> bool:
        runtime_config = getattr(vad_audio, "runtime_config", None)
        chat = getattr(runtime_config, "chat", None)
        if chat is None:
            return False
        before = chat.stats()
        # Fail closed when the audio model omits its transcript metadata. A
        # fabricated placeholder is not user speech and must not enter context.
        # Function calls remain valid partners for their later tool outputs.
        if transcript:
            chat.add_item(make_user_message(transcript))
        if assistant_text:
            chat.add_item(make_assistant_message(assistant_text))
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
        # Keep unresolved function-call pairs intact; the normal tool follow-up
        # path trims after its final assistant response is committed.
        if not tools:
            chat.trim_if_needed(None)
        stats = chat.stats()
        self._emit_metric(
            vad_audio,
            "context",
            "trimmed" if stats["trim_count"] > before["trim_count"] else "committed",
            detail={**stats, "policy": "visible_trim"},
        )
        return True

    @staticmethod
    def _extract_transcript(text: str) -> str | None:
        if _TRANSCRIPT_MARKER not in text:
            return None
        after = text.split(_TRANSCRIPT_MARKER, 1)[1]
        if _RESPONSE_MARKER in after:
            after = after.split(_RESPONSE_MARKER, 1)[0]
        transcript = after.strip().strip('"')
        return transcript or None

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
        if not transcript or "\n" in transcript:
            return None
        return transcript

    @staticmethod
    def _fallback_response_text(text: str) -> str:
        if _RESPONSE_MARKER in text:
            return text.split(_RESPONSE_MARKER, 1)[1].strip()
        if _TRANSCRIPT_MARKER in text:
            return text.split(_TRANSCRIPT_MARKER, 1)[0].strip()
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
        for index in sorted(tool_accum):
            entry = tool_accum[index]
            if not entry["name"]:
                continue
            tools.append(ResponseFunctionToolCall(type="function_call", name=entry["name"], arguments=entry["args"] or "{}", call_id=_generate_id("call"), id=_generate_id("fc"), status="completed"))
        return tools

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
