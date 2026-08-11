import base64
import json
import logging
from queue import Queue
from threading import Event, Thread
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from openai.types.realtime.conversation_item import (
    RealtimeConversationItemFunctionCall,
    RealtimeConversationItemFunctionCallOutput,
)
from openai.types.responses import ResponseFunctionToolCall

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.LLM.base_openai_compatible_language_model import (
    BaseOpenAICompatibleHandler,
    TextDelta,
    _GenState,
    _Turn,
)
from speech_to_speech.LLM.chat import Chat, make_assistant_message, make_user_message
from speech_to_speech.LLM.chat_completions_language_model import ChatCompletionsApiModelHandler
from speech_to_speech.pipeline.events import TranscriptionCompletedEvent
from speech_to_speech.pipeline.messages import (
    DirectAssistantRequest,
    DirectAssistantResponse,
    EndOfResponse,
    LLMResponseChunk,
)
from speech_to_speech.STT.gemma_audio_handler import GemmaAudioSTTHandler
from speech_to_speech.STT.transcription_notifier import TranscriptionNotifier


class _PassThroughHandler(BaseOpenAICompatibleHandler):
    def warmup(self):
        return None

    def _build_compaction_generate_fn(self):
        return lambda system, user: ""

    def _serialize(self, active_chat):
        return []

    def _request(self, api_input, optional_kwargs):
        return None

    def _iter_stream_events(self, api_response):
        yield from ()

    def _iter_response_events(self, api_response):
        yield from ()

    def _build_optional_kwargs(self, req_tools, req_tool_choice):
        return {}


@pytest.mark.parametrize(
    ("consumer", "mode"),
    [
        ("_consume_streaming", "streaming"),
        ("_consume_nonstreaming", "nonstreaming"),
    ],
)
def test_shared_llm_generation_logs_are_content_free(caplog, consumer, mode):
    assistant_secret = "ASSISTANT_PRIVATE_SENTINEL"
    argument_secret = "TOOL_ARGUMENT_PRIVATE_SENTINEL"
    handler = object.__new__(_PassThroughHandler)
    handler.speculative_turns = None
    handler.cancel_scope = None
    state = _GenState(
        tools=[
            ResponseFunctionToolCall(
                type="function_call",
                name="private_tool_name",
                arguments=json.dumps({"query": argument_secret}),
                call_id="call_private_log_test",
                id="fc_private_log_test",
                status="completed",
            )
        ]
    )
    turn = _Turn(
        language_code=None,
        gen=None,
        runtime_config=RuntimeConfig(chat=Chat(30)),
        response=None,
        turn_id="turn_private_log_test",
        turn_revision=0,
        speech_stopped_at_s=None,
        wants_audio=False,
    )

    caplog.set_level(logging.DEBUG, logger="speech_to_speech.LLM.base_openai_compatible_language_model")
    list(getattr(handler, consumer)(iter([TextDelta(text=assistant_secret)]), state, turn))

    assert assistant_secret not in caplog.text
    assert argument_secret not in caplog.text
    assert "private_tool_name" not in caplog.text
    assert (
        f"LLM generation summary mode={mode} assistant_chars={len(assistant_secret)} tool_calls=1"
        in caplog.text
    )


def _encoded_silence(handler: GemmaAudioSTTHandler, samples: int = 1600) -> str:
    return base64.b64encode(handler._wav_bytes(np.zeros(samples, dtype=np.float32))).decode("ascii")


def test_gemma_audio_payload_uses_input_audio_wav_base64():
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=False)

    payload = handler._payload(np.zeros(1600, dtype=np.float32))

    assert payload["model"] == "gemma-test"
    assert payload["stream"] is False
    content = payload["messages"][1]["content"]
    assert [part["type"] for part in content] == ["input_audio"]
    assert content[0]["input_audio"]["format"] == "wav"
    assert isinstance(content[0]["input_audio"]["data"], str)


def test_gemma_audio_full_buffer_keeps_cancellable_streaming_transport():
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=True)
    vad_audio = SimpleNamespace(
        runtime_config=SimpleNamespace(
            session=None,
            chat=SimpleNamespace(buffer=[]),
            local_pipeline={"full_buffer_tts": True},
        )
    )

    payload = handler._payload(np.zeros(1600, dtype=np.float32), vad_audio)

    assert payload["stream"] is True


def test_gemma_audio_payload_includes_recent_camera_images():
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=False)
    image_url = "data:image/jpeg;base64,abc123"
    vad_audio = SimpleNamespace(
        runtime_config=SimpleNamespace(
            session=None,
            chat=SimpleNamespace(
                buffer=[
                    SimpleNamespace(
                        content=[SimpleNamespace(type="input_image", image_url=image_url)],
                    )
                ],
            ),
        )
    )

    payload = handler._payload(np.zeros(1600, dtype=np.float32), vad_audio)

    content = payload["messages"][1]["content"]
    assert content[0] == {"type": "image_url", "image_url": {"url": image_url}}
    assert content[1]["type"] == "input_audio"


def test_gemma_audio_payload_includes_instructions_history_tools_and_disables_thinking():
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=True)
    chat = Chat(30)
    chat.add_item(make_user_message("Earlier question"))
    chat.add_item(make_assistant_message("Earlier answer"))
    session = SimpleNamespace(
        instructions="Always answer as TEST ROLE.",
        tools=[
            {
                "type": "function",
                "name": "web_search",
                "description": "Search the web",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            }
        ],
        tool_choice="auto",
    )
    vad_audio = SimpleNamespace(
        runtime_config=SimpleNamespace(session=session, chat=chat, local_pipeline={})
    )

    payload = handler._payload(np.zeros(1600, dtype=np.float32), vad_audio)

    system_prompt = payload["messages"][0]["content"]
    assert "Always answer as TEST ROLE." in system_prompt
    assert "any language, accent, or code-switching" in system_prompt
    assert "Follow the language or languages naturally used in the current utterance" in system_prompt
    assert "Do not mention transcription, audio quality, garbling" in system_prompt
    assert "unless the user explicitly asks about that topic" in system_prompt
    assert "English by default" not in system_prompt
    assert "semantic intent is genuinely unclear" not in system_prompt
    assert "varies with the conversation" in system_prompt
    assert "Otherwise begin with:\nUSER_MEMORY:" in system_prompt
    assert "\nor USER_MEMORY:" not in system_prompt
    assert "Let me check that" not in system_prompt
    assert "Could you say that another way" not in system_prompt
    assert payload["messages"][1:3] == [
        {"role": "user", "content": "Earlier question"},
        {"role": "assistant", "content": "Earlier answer"},
    ]
    assert payload["messages"][-1]["role"] == "user"
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["tools"][0]["function"]["name"] == "web_search"
    assert payload["tool_choice"] == "auto"


def test_preview_transcript_requires_the_transcript_prefix_and_rejects_assistant_text():
    assert GemmaAudioSTTHandler._extract_preview_transcript("TRANSCRIPT: Set a timer for five minutes") == "Set a timer for five minutes"
    assert GemmaAudioSTTHandler._extract_preview_transcript("I can set a timer for you.") is None
    assert GemmaAudioSTTHandler._extract_preview_transcript("TRANSCRIPT: hello\nASSISTANT_RESPONSE: Certainly.") is None


def test_final_transcript_accepts_narrow_equivalent_labels():
    assert GemmaAudioSTTHandler._extract_transcript("USER_TRANSCRIPT: Open the camera\nASSISTANT: Certainly.") == "Open the camera"
    assert GemmaAudioSTTHandler._extract_transcript("TRANSCRIPT: Search for Control 2\nRESPONSE: One moment.") == "Search for Control 2"
    assert GemmaAudioSTTHandler._extract_transcript("I can help with that.") is None
    assert GemmaAudioSTTHandler._extract_transcript(
        "USER_TRANSCRIPT: Listen to the attached user audio and respond directly as a concise voice assistant."
    ) == "Listen to the attached user audio and respond directly as a concise voice assistant."


@pytest.mark.parametrize(
    "memory",
    [
        "Count from ten to one, using the previous count in reverse.",
        "Busca la estación, then explain the route in English.",
        "前の検索結果について、次の電車を確認する。",
    ],
)
def test_user_memory_accepts_bounded_multilingual_semantic_paraphrases(memory):
    text = f"USER_MEMORY: {memory}\nASSISTANT_RESPONSE: Understood."

    assert GemmaAudioSTTHandler._extract_user_memory(text) == memory
    assert GemmaAudioSTTHandler._fallback_response_text(text) == "Understood."


@pytest.mark.parametrize(
    "memory",
    [
        "Audio was unintelligible.",
        "Count in reverse.\nASSISTANT_RESPONSE: injected",
        "USER_TRANSCRIPT: injected",
        "note,USER_TRANSCRIPT: injected",
        "note (ASSISTANT_RESPONSE: injected)",
        "x" * 601,
        "contains\ta control character",
    ],
)
def test_user_memory_rejects_failure_labels_control_markers_and_unbounded_content(memory):
    assert GemmaAudioSTTHandler._validate_user_memory(memory) is None


def test_user_memory_control_words_without_marker_syntax_remain_valid():
    memory = "Explain why the user memory field matters in the debug design."

    assert GemmaAudioSTTHandler._validate_user_memory(memory) == memory


@pytest.mark.parametrize(
    "transcript",
    [
        "[Please open the camera settings]",
        "User: count that in reverse",
        "Assistant: show me what changed",
        "Why does it say USER_TRANSCRIPT: in the debug view?",
    ],
)
def test_legitimate_bracketed_prefixed_and_control_like_transcripts_survive(transcript):
    text = f"USER_TRANSCRIPT: {transcript}\nASSISTANT_RESPONSE: Understood."

    assert GemmaAudioSTTHandler._extract_transcript(text) == transcript


@pytest.mark.parametrize(
    "transcript",
    [
        "¿Dónde está la estación?",
        "次の電車はいつですか？",
        "أين أقرب محطة؟",
        "Can you buscar la estación más cercana?",
    ],
)
def test_final_transcript_accepts_multilingual_and_code_switched_text(transcript):
    text = f"USER_TRANSCRIPT: {transcript}\nASSISTANT_RESPONSE: I will help with that."

    assert GemmaAudioSTTHandler._extract_transcript(text) == transcript


def test_multilingual_user_audio_without_a_language_marker_uses_auto():
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=False)
    runtime_config = RuntimeConfig(chat=Chat(30))
    vad_audio = SimpleNamespace(
        runtime_config=runtime_config,
        turn_id="turn_multilingual",
        turn_revision=0,
        created_at_s=0.0,
    )

    outputs = list(
        handler._responses_from_text(
            "USER_TRANSCRIPT: ¿Dónde está la estación?\nASSISTANT_RESPONSE: The station is two blocks ahead.",
            vad_audio,
        )
    )

    assert outputs[-1].transcript == "¿Dónde está la estación?"
    assert outputs[-1].text == "The station is two blocks ahead."
    assert outputs[-1].language_code == "Auto"
    assert runtime_config.local_pipeline["assistant_language"] == "Auto"


@pytest.mark.parametrize(
    ("response", "expected_language"),
    [
        (
            "USER_TRANSCRIPT: ¿Dónde está la estación?\n"
            "ASSISTANT_LANGUAGE: Spanish\n"
            "ASSISTANT_RESPONSE: La estación está a dos cuadras.",
            "Spanish",
        ),
        (
            "USER_TRANSCRIPT: Can you buscar la estación?\n"
            "ASSISTANT_LANGUAGE: Auto\n"
            "ASSISTANT_RESPONSE: Sí, it is two blocks ahead.",
            "Auto",
        ),
    ],
)
def test_direct_audio_preserves_monolingual_or_auto_response_language(response, expected_language):
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=False)
    runtime_config = RuntimeConfig(chat=Chat(30))
    vad_audio = SimpleNamespace(
        runtime_config=runtime_config,
        turn_id="turn_response_language",
        turn_revision=0,
        created_at_s=0.0,
    )

    outputs = list(handler._responses_from_text(response, vad_audio))

    assert outputs[-1].language_code == expected_language
    assert runtime_config.local_pipeline["assistant_language"] == expected_language


@pytest.mark.parametrize(
    "sentinel",
    [
        "[inaudible]",
        "Unintelligible.",
        "Audio was garbled.",
        "No intelligible speech detected.",
        "Transcription unavailable.",
    ],
)
def test_failure_only_transcript_sentinels_are_rejected_exactly(sentinel):
    text = f"USER_TRANSCRIPT: {sentinel}\nASSISTANT_RESPONSE: I will answer naturally."

    assert GemmaAudioSTTHandler._extract_transcript(text) is None


def test_transcript_failure_words_inside_real_user_content_are_preserved():
    transcript = "Why does the recording sound garbled and unintelligible?"
    text = f"USER_TRANSCRIPT: {transcript}\nASSISTANT_RESPONSE: I will explain."

    assert GemmaAudioSTTHandler._extract_transcript(text) == transcript


def test_failure_only_transcript_does_not_gate_response_or_tools():
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=False)
    runtime_config = RuntimeConfig(chat=Chat(30))
    vad_audio = SimpleNamespace(
        runtime_config=runtime_config,
        turn_id="turn_failure_sentinel",
        turn_revision=0,
        created_at_s=0.0,
    )
    tool = ResponseFunctionToolCall(
        type="function_call",
        name="web_search",
        arguments='{"query":"current weather"}',
        call_id="call_failure_sentinel",
        id="fc_failure_sentinel",
        status="completed",
    )

    outputs = list(
        handler._responses_from_text(
            "USER_TRANSCRIPT: Audio was unintelligible.\n"
            "ASSISTANT_LANGUAGE: Auto\n"
            "ASSISTANT_PREAMBLE: I'll check the current forecast.",
            vad_audio,
            tools=[tool],
        )
    )

    assert outputs[-1].transcript is None
    assert outputs[-1].text == "I'll check the current forecast."
    assert outputs[-1].tools == [tool]
    assert outputs[-1].language_code == "Auto"
    assert runtime_config.local_pipeline["assistant_language"] == "Auto"


def test_direct_audio_resets_a_previous_turn_language_before_generation():
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=False)
    runtime_config = RuntimeConfig(chat=Chat(30))
    runtime_config.local_pipeline["assistant_language"] = "Spanish"
    vad_audio = SimpleNamespace(
        audio=np.zeros(1600, dtype=np.float32),
        mode="final",
        runtime_config=runtime_config,
        turn_id="turn_language_reset",
        turn_revision=0,
        created_at_s=0.0,
    )

    def responses(_audio, received_vad_audio, *, generation=None):
        assert "assistant_language" not in received_vad_audio.runtime_config.local_pipeline
        yield DirectAssistantResponse(text="Hello.", is_final=True)

    handler._iter_direct_responses = responses
    list(handler.process(vad_audio))

    assert "assistant_language" not in runtime_config.local_pipeline


def test_accepted_turn_metrics_are_content_free_input_measurements():
    metrics = Queue()
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(
        model_name="gemma-test",
        base_url="http://127.0.0.1:8818/v1",
        stream=False,
        text_output_queue=metrics,
    )
    runtime_config = RuntimeConfig(chat=Chat(30))
    vad_audio = SimpleNamespace(
        audio=np.asarray([0.0, 0.5, 1.0, -1.0], dtype=np.float32),
        mode="final",
        runtime_config=runtime_config,
        turn_id="turn_content_free_metrics",
        turn_revision=1,
        created_at_s=0.0,
    )
    handler._iter_direct_responses = lambda *_args, **_kwargs: iter(
        [DirectAssistantResponse(text="A private semantic answer.", is_final=True)]
    )

    list(handler.process(vad_audio))

    events = []
    while not metrics.empty():
        events.append(metrics.get_nowait())
    request_start = next(event for event in events if event.stage == "gemma" and event.status == "request_start")
    captured = next(event for event in events if event.stage == "transcription" and event.status == "captured")
    for detail in (request_start.detail, captured.detail):
        assert detail["rms"] == 0.75
        assert detail["peak"] == 1.0
        assert detail["near_silence"] is False
        assert detail["clipping"] is True
        assert detail["clipping_fraction"] == 0.5
        assert detail["revision_count"] == 2
        assert "transcript" not in detail
        assert "text" not in detail
        assert "A private semantic answer" not in json.dumps(detail)


def test_final_transcript_has_no_second_request_fallback():
    assert not hasattr(GemmaAudioSTTHandler, "_transcribe_once")
    assert not hasattr(GemmaAudioSTTHandler, "_final_transcript")


def test_optional_tool_preamble_is_extracted_without_becoming_response_metadata():
    text = (
        "USER_TRANSCRIPT: Check today's weather\n"
        "ASSISTANT_LANGUAGE: English\n"
        "ASSISTANT_PREAMBLE: I'll check the latest forecast."
    )

    assert GemmaAudioSTTHandler._extract_transcript(text) == "Check today's weather"
    assert GemmaAudioSTTHandler._extract_assistant_preamble(text) == "I'll check the latest forecast."
    assert GemmaAudioSTTHandler._fallback_response_text(text) == ""


def test_tool_preamble_is_never_synthesized_or_rewritten():
    camera = ResponseFunctionToolCall(
        type="function_call", name="camera_snapshot", arguments="{}", call_id="call_camera", id="fc_camera", status="completed"
    )
    search = ResponseFunctionToolCall(
        type="function_call", name="web_search", arguments="{}", call_id="call_search", id="fc_search", status="completed"
    )
    assert GemmaAudioSTTHandler._tool_preamble("", [camera]) is None
    assert GemmaAudioSTTHandler._tool_preamble("", [search]) is None
    assert GemmaAudioSTTHandler._tool_preamble("ASSISTANT_RESPONSE: A substantive answer.", [search]) is None
    assert (
        GemmaAudioSTTHandler._tool_preamble(
            "ASSISTANT_PREAMBLE: I'll check the latest information.",
            [search],
        )
        == "I'll check the latest information."
    )


def test_tool_preamble_is_spoken_and_committed_before_the_function_call():
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=False)
    chat = Chat(30)
    vad_audio = SimpleNamespace(
        runtime_config=RuntimeConfig(chat=chat),
        turn_id="turn_tool_preamble",
        turn_revision=0,
        created_at_s=0.0,
    )
    tool = ResponseFunctionToolCall(
        type="function_call",
        name="web_search",
        arguments='{"query":"weather"}',
        call_id="call_weather",
        id="fc_weather",
        status="completed",
    )
    text = (
        "USER_TRANSCRIPT: Check today's weather\n"
        "ASSISTANT_LANGUAGE: English\n"
        "ASSISTANT_PREAMBLE: I'll check the latest forecast."
    )

    outputs = list(handler._responses_from_text(text, vad_audio, tools=[tool]))

    assert outputs[0].text == "I'll check the latest forecast."
    assert outputs[0].tools == [tool]
    assert [item.type for item in chat.buffer] == ["message", "message", "function_call"]
    assert chat.stats()["pending_tool_calls"] == 1


def test_semantic_memory_preserves_tool_preamble_and_call_order_without_transcript():
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=False)
    chat = Chat(30)
    vad_audio = SimpleNamespace(
        runtime_config=RuntimeConfig(chat=chat),
        turn_id="turn_tool_preamble_no_transcript",
        turn_revision=0,
        created_at_s=0.0,
    )
    tool = ResponseFunctionToolCall(
        type="function_call",
        name="camera_snapshot",
        arguments="{}",
        call_id="call_camera_preamble",
        id="fc_camera_preamble",
        status="completed",
    )
    handler._commit_accepted_audio(vad_audio, _encoded_silence(handler))

    outputs = list(
        handler._responses_from_text(
            "USER_MEMORY: Inspect the current camera view more closely.\n"
            "ASSISTANT_LANGUAGE: English\n"
            "ASSISTANT_PREAMBLE: Let me take a closer look.",
            vad_audio,
            tools=[tool],
        )
    )

    assert outputs[0].transcript is None
    assert outputs[0].text == "Let me take a closer look."
    assert [item.type for item in chat.buffer] == ["message", "message", "function_call"]
    assert chat.buffer[0].content[0].type == "input_text"
    assert chat.buffer[0].content[0].text == "Inspect the current camera view more closely."
    assert chat.stats()["pending_tool_calls"] == 1


def test_direct_transport_timeout_reaches_failed_end_of_response():
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=True)
    handler._iter_direct_responses = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        httpx.ReadTimeout("timed out")
    )
    vad_audio = SimpleNamespace(
        audio=np.zeros(1600, dtype=np.float32),
        mode="final",
        runtime_config=RuntimeConfig(chat=Chat(30)),
        turn_id="turn_timeout",
        turn_revision=0,
        created_at_s=0.0,
    )

    direct = list(handler.process(vad_audio))
    notifier = object.__new__(TranscriptionNotifier)
    notifier.setup(text_output_queue=None, runtime_config=None)
    request = list(notifier.process(direct[0]))[0]
    outputs = list(object.__new__(_PassThroughHandler).process(request))

    assert direct[0].error == "Direct audio model response timed out."
    assert len(outputs) == 1
    assert isinstance(outputs[0], EndOfResponse)
    assert outputs[0].error == direct[0].error


def test_unicode_assistant_text_is_preserved_without_console_output():
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=False)
    chat = Chat(30)
    vad_audio = SimpleNamespace(
        runtime_config=RuntimeConfig(chat=chat),
        turn_id="turn_ru",
        turn_revision=0,
        created_at_s=0.0,
    )

    outputs = list(handler._responses_from_text("USER: Say it in Russian\nASSISTANT: Привет, как дела?", vad_audio))

    assert outputs[0].text == "Привет, как дела?"
    assert outputs[0].transcript == "Say it in Russian"
    assert chat.stats()["turns"] == 1


def test_direct_audio_generation_reaches_tts_pass_through_messages():
    notifier = object.__new__(TranscriptionNotifier)
    notifier.setup(text_output_queue=None, runtime_config=None)
    response = DirectAssistantResponse(
        text="Hello.",
        transcript="Hello",
        is_final=True,
        cancel_generation=17,
    )
    request = next(notifier.process(response))
    handler = object.__new__(_PassThroughHandler)
    outputs = list(handler.process(request))

    assert request.cancel_generation == 17
    assert outputs[0].cancel_generation == 17
    assert outputs[1].cancel_generation == 17


def test_direct_tool_call_is_committed_before_browser_output():
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=False)
    chat = Chat(30)
    runtime_config = SimpleNamespace(chat=chat)
    vad_audio = SimpleNamespace(
        runtime_config=runtime_config,
        turn_id="turn_1",
        turn_revision=0,
    )
    tool = ResponseFunctionToolCall(
        type="function_call",
        name="camera_snapshot",
        arguments="{}",
        call_id="call_camera",
        id="fc_camera",
        status="completed",
    )

    committed = handler._commit_context(vad_audio, "What can you see?", "", [tool])

    assert committed is True
    assert chat.stats()["pending_tool_calls"] == 1
    chat.add_item(
        RealtimeConversationItemFunctionCallOutput(
            type="function_call_output",
            call_id="call_camera",
            output="Snapshot captured.",
        )
    )
    assert [item.type for item in chat.buffer] == ["message", "function_call", "function_call_output"]
    assert chat.buffer[-2].call_id == "call_camera"
    assert chat.buffer[-1].call_id == "call_camera"
    assert chat.stats()["pending_tool_calls"] == 0


def test_direct_tool_call_without_transcript_is_persisted_without_fabricated_user_text():
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=False)
    chat = Chat(30)
    vad_audio = SimpleNamespace(runtime_config=SimpleNamespace(chat=chat), turn_id="turn_1", turn_revision=0)
    tool = ResponseFunctionToolCall(
        type="function_call",
        name="web_search",
        arguments='{"query":"local test"}',
        call_id="call_search",
        id="fc_search",
        status="completed",
    )

    handler._commit_accepted_audio(vad_audio, _encoded_silence(handler))
    assert handler._commit_context(vad_audio, None, "", [tool]) is True
    assert chat.stats()["pending_tool_calls"] == 1
    assert [item.type for item in chat.buffer] == ["message", "function_call"]
    assert chat.buffer[0].content[0].type == "input_audio"


def test_streamed_tool_call_normalizes_opaque_llama_cpp_id_and_commits():
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=True)
    accum = {}
    handler._accumulate_tool_deltas(
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "UNM0K7ZOZpEN5uS0vGTo1G1UnSDH8Vki",
                    "function": {"name": "web_search", "arguments": '{"query":"Control 2"}'},
                }
            ]
        },
        accum,
    )

    tools = handler._tool_calls_from_accum(accum)

    assert tools[0].call_id == "call_turn_0"
    chat = Chat(30)
    vad_audio = SimpleNamespace(runtime_config=SimpleNamespace(chat=chat), turn_id="turn_tool", turn_revision=0)
    assert handler._commit_context(vad_audio, "Search for Control 2", "", tools) is True
    assert chat.stats()["pending_tool_calls"] == 1


def test_buffered_tool_call_preserves_prefixed_id():
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=False)

    _, tools = handler._message_text_and_tools(
        {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_camera_1",
                                "function": {"name": "camera_snapshot", "arguments": "{}"},
                            }
                        ],
                    }
                }
            ]
        }
    )

    assert tools[0].call_id == "call_camera_1"


def test_tool_call_without_model_id_generates_realtime_id():
    tools = GemmaAudioSTTHandler._tool_calls_from_accum(
        {0: {"name": "web_search", "args": '{"query":"local"}', "id": ""}}
    )

    assert tools[0].call_id == "call_turn_0"


def test_parallel_opaque_tool_ids_are_normalized_and_unique():
    tools = GemmaAudioSTTHandler._tool_calls_from_accum(
        {
            0: {"name": "web_search", "args": '{"query":"one"}', "id": "opaque"},
            1: {"name": "camera_snapshot", "args": "{}", "id": "opaque"},
        }
    )

    assert [tool.call_id for tool in tools] == ["call_turn_0", "call_turn_1"]


def test_malformed_tool_arguments_stay_raw_for_browser_and_explicit_in_history():
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=False)
    chat = Chat(30)
    tools = handler._tool_calls_from_accum(
        {0: {"name": "web_search", "args": '{"query":', "id": "call_bad_args"}},
        chat=chat,
        turn_id="bad_args",
    )
    vad_audio = SimpleNamespace(runtime_config=SimpleNamespace(chat=chat), turn_id="bad_args", turn_revision=0)
    handler._commit_accepted_audio(vad_audio, _encoded_silence(handler))

    assert tools[0].arguments == '{"query":'
    assert handler._commit_context(vad_audio, None, "I'll check.", tools) is True
    serialized = ChatCompletionsApiModelHandler._chat_messages(chat)
    assert serialized[-1]["tool_calls"][0]["function"]["arguments"] == json.dumps(
        {"error": "invalid_tool_arguments"}
    )


def test_reused_native_tool_id_gets_turn_scoped_deterministic_id():
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=False)
    chat = Chat(30)
    first = RealtimeConversationItemFunctionCall(
        type="function_call", name="web_search", arguments="{}", call_id="call_reused", id="fc_reused"
    )
    chat.add_item(first)
    chat.add_item(
        RealtimeConversationItemFunctionCallOutput(
            type="function_call_output", call_id="call_reused", output="done"
        )
    )

    tools = handler._tool_calls_from_accum(
        {0: {"name": "camera_snapshot", "args": "{}", "id": "call_reused"}},
        chat=chat,
        turn_id="camera-follow-up",
    )

    assert tools[0].call_id == "call_camera-follow-up_0"


def test_tool_result_and_continuation_preserve_exact_atomic_order():
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=False)
    chat = Chat(30)
    runtime_config = RuntimeConfig(chat=chat)
    vad_audio = SimpleNamespace(runtime_config=runtime_config, turn_id="search", turn_revision=0, created_at_s=0.0)
    handler._commit_accepted_audio(vad_audio, _encoded_silence(handler))
    tool = ResponseFunctionToolCall(
        type="function_call",
        name="web_search",
        arguments='{"query":"local"}',
        call_id="call_search_order",
        id="fc_search_order",
        status="completed",
    )

    outputs = list(
        handler._responses_from_text(
            "USER_MEMORY: Busca la información local y conserva el contexto en español.\n"
            "ASSISTANT_LANGUAGE: Spanish\n"
            "ASSISTANT_PREAMBLE: Buscaré eso.",
            vad_audio,
            tools=[tool],
        )
    )
    assert outputs[-1].context_committed is True
    assert runtime_config.local_pipeline["assistant_language"] == "Spanish"
    chat.add_item(
        RealtimeConversationItemFunctionCallOutput(
            type="function_call_output", call_id="call_search_order", output='{"answer":"found"}'
        )
    )
    chat.add_item(make_assistant_message("Encontré el resultado."))

    assert [item.type for item in chat.buffer] == [
        "message",
        "message",
        "function_call",
        "function_call_output",
        "message",
    ]
    follow_up = SimpleNamespace(runtime_config=runtime_config, turn_id="search_follow_up", turn_revision=0)
    payload = handler._payload(np.zeros(1600, dtype=np.float32), follow_up)
    assert [message["role"] for message in payload["messages"]] == [
        "system",
        "user",
        "assistant",
        "assistant",
        "tool",
        "assistant",
        "user",
    ]
    assert payload["messages"][1]["content"] == "Busca la información local y conserva el contexto en español."
    assert payload["messages"][3]["tool_calls"][0]["id"] == "call_search_order"
    assert payload["messages"][4]["tool_call_id"] == "call_search_order"
    assert payload["messages"][5]["content"] == "Encontré el resultado."
    assert runtime_config.local_pipeline["assistant_language"] == "Spanish"


def test_missing_transcript_preserves_assistant_output_without_polluting_context():
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=False)
    chat = Chat(30)
    vad_audio = SimpleNamespace(
        runtime_config=RuntimeConfig(chat=chat),
        turn_id="turn_bad_audio",
        turn_revision=0,
        created_at_s=0.0,
    )

    handler._commit_accepted_audio(vad_audio, _encoded_silence(handler))
    outputs = list(handler._responses_from_text("ASSISTANT_RESPONSE: I heard you.", vad_audio))

    assert outputs[0].transcript is None
    assert outputs[0].text == "I heard you."
    assert outputs[0].tools == []
    assert [item.type for item in chat.buffer] == ["message", "message"]
    assert chat.buffer[0].content[0].type == "input_audio"


def test_missing_transcript_emits_display_only_user_audio_and_one_tts_sequence():
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=False)
    chat = Chat(30)
    vad_audio = SimpleNamespace(
        runtime_config=RuntimeConfig(chat=chat),
        turn_id="turn_audio",
        turn_revision=0,
        created_at_s=0.0,
    )
    handler._commit_accepted_audio(vad_audio, _encoded_silence(handler))
    direct = list(handler._responses_from_text("ASSISTANT_RESPONSE: Ready.", vad_audio))[0]
    queue = Queue()
    notifier = object.__new__(TranscriptionNotifier)
    notifier.setup(text_output_queue=queue, runtime_config=None, should_listen=Event())

    requests = list(notifier.process(direct))
    completed = queue.get_nowait()
    outputs = list(object.__new__(_PassThroughHandler).process(requests[0]))

    assert completed.transcript == "[User audio]"
    assert completed.display_only is True
    assert completed.direct_audio_completed is True
    assert [item.text for item in outputs if isinstance(item, LLMResponseChunk)] == ["Ready."]
    assert outputs[-1].tag == "end_of_response"
    assert [item.type for item in chat.buffer] == ["message", "message"]
    assert chat.buffer[0].content[0].type == "input_audio"


class _FakeSSEStream:
    def __init__(self, lines):
        self._lines = lines
        self.closed = False

    def wait_for_headers(self):
        return None

    def iter_lines(self):
        yield from self._lines

    def close(self):
        self.closed = True


def _sse_text(text: str) -> list[str]:
    return [f"data: {json.dumps({'choices': [{'delta': {'content': text}}]})}", "data: [DONE]"]


def test_primary_payload_contains_current_audio_once_before_history_anchor_is_committed():
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=True)
    chat = Chat(30)
    payloads = []

    def stream_request(_url, payload, *, api_key):
        payloads.append(payload)
        return _FakeSSEStream(_sse_text("ASSISTANT_RESPONSE: Ready."))

    handler._stream_request = stream_request
    vad_audio = SimpleNamespace(
        audio=np.zeros(1600, dtype=np.float32),
        mode="final",
        runtime_config=RuntimeConfig(chat=chat),
        turn_id="turn_once",
        turn_revision=0,
        created_at_s=0.0,
    )

    list(handler.process(vad_audio))

    assert [message["role"] for message in payloads[0]["messages"]] == ["system", "user"]
    assert [part["type"] for part in payloads[0]["messages"][-1]["content"]] == ["input_audio"]
    assert [item.type for item in chat.buffer] == ["message", "message"]


def test_streamed_primary_user_memory_upgrades_existing_audio_anchor_without_ui_transcript():
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=True)
    chat = Chat(30)
    handler._stream_request = lambda *_args, **_kwargs: _FakeSSEStream(
        _sse_text(
            "USER_MEMORY: Count from one to ten.\n"
            "ASSISTANT_LANGUAGE: English\n"
            "ASSISTANT_RESPONSE: One, two, three."
        )
    )
    vad_audio = SimpleNamespace(
        audio=np.zeros(1600, dtype=np.float32),
        mode="final",
        runtime_config=RuntimeConfig(chat=chat),
        turn_id="turn_memory_streamed",
        turn_revision=0,
        created_at_s=0.0,
    )

    outputs = list(handler.process(vad_audio))

    assert outputs[-1].transcript is None
    assert "".join(output.text for output in outputs) == "One, two, three."
    assert outputs[-1].context_committed is True
    assert [item.type for item in chat.buffer] == ["message", "message"]
    assert chat.buffer[0].content[0].type == "input_text"
    assert chat.buffer[0].content[0].text == "Count from one to ten."


def test_buffered_primary_user_memory_upgrades_audio_anchor_and_stays_display_only():
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=False)
    chat = Chat(30)
    handler._stream_request = lambda *_args, **_kwargs: _FakeSSEStream(
        _sse_text(
            "USER_MEMORY: Explain the previous answer more simply.\n"
            "ASSISTANT_LANGUAGE: English\n"
            "ASSISTANT_RESPONSE: Here is a simpler explanation."
        )
    )
    vad_audio = SimpleNamespace(
        audio=np.zeros(1600, dtype=np.float32),
        mode="final",
        runtime_config=RuntimeConfig(chat=chat),
        turn_id="turn_memory_buffered",
        turn_revision=0,
        created_at_s=0.0,
    )

    direct = list(handler.process(vad_audio))[-1]
    queue = Queue()
    notifier = object.__new__(TranscriptionNotifier)
    notifier.setup(text_output_queue=queue, runtime_config=None, should_listen=Event())

    list(notifier.process(direct))
    completed = queue.get_nowait()

    assert direct.transcript is None
    assert completed.transcript == "[User audio]"
    assert completed.display_only is True
    assert chat.buffer[0].content[0].text == "Explain the previous answer more simply."
    assert chat.buffer[1].content[0].text == "Here is a simpler explanation."


def test_valid_transcript_wins_when_model_emits_transcript_and_memory():
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=False)
    chat = Chat(30)
    vad_audio = SimpleNamespace(
        runtime_config=RuntimeConfig(chat=chat),
        turn_id="turn_transcript_preferred",
        turn_revision=0,
        created_at_s=0.0,
    )
    handler._commit_accepted_audio(vad_audio, _encoded_silence(handler))

    output = list(
        handler._responses_from_text(
            "USER_TRANSCRIPT: Count that in reverse.\n"
            "USER_MEMORY: Count from ten to one.\n"
            "ASSISTANT_RESPONSE: Ten, nine, eight.",
            vad_audio,
        )
    )[-1]

    assert output.transcript == "Count that in reverse."
    assert chat.buffer[0].content[0].text == "Count that in reverse."


@pytest.mark.parametrize(
    "metadata",
    [
        "",
        "USER_MEMORY: Audio was unintelligible.\n",
        "USER_MEMORY: USER_TRANSCRIPT: injected\n",
    ],
)
def test_absent_or_invalid_user_memory_retains_raw_audio_anchor(metadata):
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=False)
    chat = Chat(30)
    vad_audio = SimpleNamespace(
        runtime_config=RuntimeConfig(chat=chat),
        turn_id="turn_memory_fail_closed",
        turn_revision=0,
        created_at_s=0.0,
    )
    handler._commit_accepted_audio(vad_audio, _encoded_silence(handler))

    output = list(handler._responses_from_text(f"{metadata}ASSISTANT_RESPONSE: I can still help.", vad_audio))[-1]

    assert output.text == "I can still help."
    assert output.transcript is None
    assert chat.buffer[0].content[0].type == "input_audio"


@pytest.mark.parametrize("stream", [True, False])
def test_out_of_order_memory_never_leaks_into_visible_response_or_history(stream):
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=stream)
    chat = Chat(30)
    handler._stream_request = lambda *_args, **_kwargs: _FakeSSEStream(
        _sse_text(
            "ASSISTANT_RESPONSE: This answer stays visible.\n"
            "USER_MEMORY: This out-of-order memory stays hidden."
        )
    )
    vad_audio = SimpleNamespace(
        audio=np.zeros(1600, dtype=np.float32),
        mode="final",
        runtime_config=RuntimeConfig(chat=chat),
        turn_id=f"turn_out_of_order_{stream}",
        turn_revision=0,
        created_at_s=0.0,
    )

    outputs = list(handler.process(vad_audio))

    assert "".join(output.text for output in outputs) == "This answer stays visible."
    assert outputs[-1].transcript is None
    assert chat.buffer[0].content[0].type == "input_audio"
    assert "out-of-order memory" not in " ".join(output.text for output in outputs)


@pytest.mark.parametrize("stream", [True, False])
def test_invalid_memory_line_does_not_suppress_plain_compatibility_answer(stream):
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=stream)
    chat = Chat(30)
    handler._stream_request = lambda *_args, **_kwargs: _FakeSSEStream(
        _sse_text("USER_MEMORY: Audio was unintelligible.\nA plain compatibility answer remains visible.")
    )
    vad_audio = SimpleNamespace(
        audio=np.zeros(1600, dtype=np.float32),
        mode="final",
        runtime_config=RuntimeConfig(chat=chat),
        turn_id=f"turn_invalid_memory_plain_{stream}",
        turn_revision=0,
        created_at_s=0.0,
    )

    outputs = list(handler.process(vad_audio))

    assert "".join(output.text for output in outputs) == "A plain compatibility answer remains visible."
    assert chat.buffer[0].content[0].type == "input_audio"


def test_streamed_split_trailing_control_marker_is_never_spoken():
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=True)
    chat = Chat(30)
    chunks = [
        "ASSISTANT_RESPONSE: This sentence is complete.\nUSER_MEM",
        "ORY: hidden semantic text",
    ]
    lines = [f"data: {json.dumps({'choices': [{'delta': {'content': chunk}}]})}" for chunk in chunks]
    lines.append("data: [DONE]")
    handler._stream_request = lambda *_args, **_kwargs: _FakeSSEStream(lines)
    vad_audio = SimpleNamespace(
        audio=np.zeros(1600, dtype=np.float32),
        mode="final",
        runtime_config=RuntimeConfig(chat=chat),
        turn_id="turn_split_trailing_control",
        turn_revision=0,
        created_at_s=0.0,
    )

    outputs = list(handler.process(vad_audio))

    assert "".join(output.text for output in outputs) == "This sentence is complete."
    assert chat.buffer[0].content[0].type == "input_audio"


def test_newer_revision_supersedes_one_stable_audio_anchor_without_payload_duplication():
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=True)
    chat = Chat(30)
    runtime_config = RuntimeConfig(chat=chat)
    runtime_config.local_pipeline["_session_id"] = "session_revision"
    rev0 = SimpleNamespace(runtime_config=runtime_config, turn_id="turn_1", turn_revision=0, created_at_s=0.0)
    rev1 = SimpleNamespace(runtime_config=runtime_config, turn_id="turn_1", turn_revision=1, created_at_s=0.0)
    old_audio = _encoded_silence(handler, 800)
    new_samples = np.full(2400, 0.25, dtype=np.float32)
    new_audio = base64.b64encode(handler._wav_bytes(new_samples)).decode("ascii")

    old_item_id = handler._commit_accepted_audio(rev0, old_audio)
    payload = handler._payload(new_samples, rev1, encoded_audio=new_audio)

    assert [message["role"] for message in payload["messages"]] == ["system", "user"]
    assert payload["messages"][-1]["content"] == [
        {"type": "input_audio", "input_audio": {"data": new_audio, "format": "wav"}}
    ]
    assert handler._commit_accepted_audio(rev1, new_audio) == old_item_id
    assert chat.stats()["turns"] == 1
    assert chat.buffer[0].id == old_item_id
    assert chat.buffer[0].content[0].audio == new_audio


def test_stale_revision_cleanup_cannot_clear_new_owner_and_new_revision_completes_once():
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=False)
    chat = Chat(30)
    runtime_config = RuntimeConfig(chat=chat)
    runtime_config.local_pipeline["_session_id"] = "session_revision_complete"
    rev0 = SimpleNamespace(runtime_config=runtime_config, turn_id="turn_1", turn_revision=0, created_at_s=0.0)
    rev1 = SimpleNamespace(runtime_config=runtime_config, turn_id="turn_1", turn_revision=1, created_at_s=0.0)

    handler._commit_accepted_audio(rev0, _encoded_silence(handler, 800))
    handler._commit_accepted_audio(rev1, _encoded_silence(handler, 2400))
    handler._finish_user_context(rev0)

    assert handler._owns_user_context(rev1) is True
    assert handler._commit_context(rev0, "stale transcript", "stale answer", []) is False
    outputs = list(
        handler._responses_from_text(
            "USER_TRANSCRIPT: Count in reverse.\nASSISTANT_RESPONSE: Ten, nine, eight.",
            rev1,
        )
    )
    assert outputs[-1].context_committed is True
    assert [item.type for item in chat.buffer] == ["message", "message"]
    assert chat.buffer[0].content[0].text == "Count in reverse."
    assert chat.buffer[1].content[0].text == "Ten, nine, eight."
    assert handler._owned_user_context(rev1) is None


def test_paused_stale_transcript_cannot_overwrite_newer_cumulative_audio(monkeypatch):
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=False)
    chat = Chat(30)
    runtime_config = RuntimeConfig(chat=chat)
    runtime_config.local_pipeline["_session_id"] = "session_revision_race"
    rev0 = SimpleNamespace(runtime_config=runtime_config, turn_id="turn_1", turn_revision=0)
    rev1 = SimpleNamespace(runtime_config=runtime_config, turn_id="turn_1", turn_revision=1)
    old_audio = _encoded_silence(handler, 800)
    new_audio = _encoded_silence(handler, 2400)
    anchor_id = handler._commit_accepted_audio(rev0, old_audio)
    original_turn_key = handler._turn_key
    rev0_paused = Event()
    resume_rev0 = Event()

    def paused_turn_key(vad_audio):
        if vad_audio is rev0 and not rev0_paused.is_set():
            rev0_paused.set()
            assert resume_rev0.wait(timeout=2.0)
        return original_turn_key(vad_audio)

    monkeypatch.setattr(handler, "_turn_key", paused_turn_key)
    stale_result = []
    stale_thread = Thread(
        target=lambda: stale_result.append(handler._ensure_user_context(rev0, "stale rev0 transcript")),
        daemon=True,
    )
    stale_thread.start()
    assert rev0_paused.wait(timeout=2.0)

    assert handler._commit_accepted_audio(rev1, new_audio) == anchor_id
    resume_rev0.set()
    stale_thread.join(timeout=2.0)

    assert not stale_thread.is_alive()
    assert stale_result == [None]
    assert chat.buffer[0].content[0].type == "input_audio"
    assert chat.buffer[0].content[0].audio == new_audio
    assert handler._ensure_user_context(rev1, "final rev1 transcript") == anchor_id
    assert chat.buffer[0].content[0].type == "input_text"
    assert chat.buffer[0].content[0].text == "final rev1 transcript"


def test_stale_semantic_memory_cannot_overwrite_newer_cumulative_audio_or_memory():
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=False)
    chat = Chat(30)
    runtime_config = RuntimeConfig(chat=chat)
    runtime_config.local_pipeline["_session_id"] = "session_memory_revision"
    rev0 = SimpleNamespace(runtime_config=runtime_config, turn_id="turn_1", turn_revision=0)
    rev1 = SimpleNamespace(runtime_config=runtime_config, turn_id="turn_1", turn_revision=1)
    newer_audio = _encoded_silence(handler, 2400)

    anchor_id = handler._commit_accepted_audio(rev0, _encoded_silence(handler, 800))
    assert handler._commit_accepted_audio(rev1, newer_audio) == anchor_id

    assert handler._ensure_user_context(rev0, None, "Stale earlier intent.") is None
    assert chat.buffer[0].content[0].type == "input_audio"
    assert chat.buffer[0].content[0].audio == newer_audio

    assert handler._ensure_user_context(rev1, None, "Use the complete revised request.") == anchor_id
    assert chat.buffer[0].content[0].type == "input_text"
    assert chat.buffer[0].content[0].text == "Use the complete revised request."
    assert handler._ensure_user_context(rev0, None, "Still stale earlier intent.") is None
    assert chat.buffer[0].content[0].text == "Use the complete revised request."


def test_response_commit_and_revision_supersession_share_one_ownership_transaction(monkeypatch):
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=False)
    chat = Chat(30)
    runtime_config = RuntimeConfig(chat=chat)
    runtime_config.local_pipeline["_session_id"] = "session_response_commit_race"
    rev0 = SimpleNamespace(runtime_config=runtime_config, turn_id="turn_1", turn_revision=0)
    rev1 = SimpleNamespace(runtime_config=runtime_config, turn_id="turn_1", turn_revision=1)
    handler._commit_accepted_audio(rev0, _encoded_silence(handler, 800))
    stale_tool = ResponseFunctionToolCall(
        type="function_call",
        name="web_search",
        arguments='{"query":"stale"}',
        call_id="call_stale_revision",
        id="fc_stale_revision",
        status="completed",
    )
    original_ensure = handler._ensure_user_context
    ensure_complete = Event()
    resume_rev0 = Event()
    rev1_complete = Event()

    def paused_ensure(vad_audio, transcript, user_memory=None):
        result = original_ensure(vad_audio, transcript, user_memory)
        if vad_audio is rev0 and not ensure_complete.is_set():
            ensure_complete.set()
            assert resume_rev0.wait(timeout=2.0)
        return result

    monkeypatch.setattr(handler, "_ensure_user_context", paused_ensure)
    rev0_result = []
    rev0_thread = Thread(
        target=lambda: rev0_result.append(
            handler._commit_context(
                rev0,
                None,
                "The first revision completed.",
                [stale_tool],
                user_memory="Remember the first revision.",
            )
        ),
        daemon=True,
    )
    rev0_thread.start()
    assert ensure_complete.wait(timeout=2.0)

    def supersede():
        handler._commit_accepted_audio(rev1, _encoded_silence(handler, 2400))
        rev1_complete.set()

    rev1_thread = Thread(target=supersede, daemon=True)
    rev1_thread.start()
    assert rev1_complete.wait(timeout=2.0) is True

    resume_rev0.set()
    rev0_thread.join(timeout=2.0)
    rev1_thread.join(timeout=2.0)

    assert rev0_result == [False]
    assert rev1_complete.is_set()
    assert [item.type for item in chat.buffer] == ["message"]
    assert chat.buffer[0].content[0].type == "input_audio"
    item_count = len(chat.buffer)
    assert (
        handler._commit_context(
            rev0,
            None,
            "This stale response must not append.",
            [],
            user_memory="Stale first-revision memory.",
        )
        is False
    )
    assert len(chat.buffer) == item_count


def test_cross_session_reused_turn_id_has_distinct_ownership_and_stale_cleanup_is_scoped():
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=False)
    chat_old = Chat(30)
    chat_new = Chat(30)
    runtime_old = RuntimeConfig(chat=chat_old)
    runtime_new = RuntimeConfig(chat=chat_new)
    runtime_old.local_pipeline["_session_id"] = "session_old"
    runtime_new.local_pipeline["_session_id"] = "session_new"
    old_turn = SimpleNamespace(runtime_config=runtime_old, turn_id="turn_1", turn_revision=0)
    new_turn = SimpleNamespace(runtime_config=runtime_new, turn_id="turn_1", turn_revision=0)

    old_item_id = handler._commit_accepted_audio(old_turn, _encoded_silence(handler, 800))
    new_item_id = handler._commit_accepted_audio(new_turn, _encoded_silence(handler, 1600))
    handler._finish_user_context(old_turn)

    assert old_item_id != new_item_id
    assert handler._owns_user_context(new_turn) is True
    assert handler._ensure_user_context(new_turn, None, "Remember the new session request.") == new_item_id
    assert chat_old.buffer[0].content[0].type == "input_audio"
    assert chat_new.buffer[0].content[0].text == "Remember the new session request."
    handler.on_session_end()
    assert handler._owned_user_context(new_turn) is None


def test_transcriptless_count_turn_uses_semantic_memory_for_reverse_follow_up():
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=True)
    chat = Chat(30)
    payloads = []

    def stream_request(_url, payload, *, api_key):
        payloads.append(payload)
        if len(payloads) == 1:
            return _FakeSSEStream(
                _sse_text(
                    "USER_MEMORY: Count from one to ten.\n"
                    "ASSISTANT_RESPONSE: One, two, three, four, five, six, seven, eight, nine, ten."
                )
            )
        return _FakeSSEStream(
            _sse_text("USER_TRANSCRIPT: Count in reverse.\nASSISTANT_RESPONSE: Ten, nine, eight, seven.")
        )

    handler._stream_request = stream_request
    runtime_config = RuntimeConfig(chat=chat)
    for turn_id in ("count_forward", "count_reverse"):
        list(
            handler.process(
                SimpleNamespace(
                    audio=np.zeros(1600, dtype=np.float32),
                    mode="final",
                    runtime_config=runtime_config,
                    turn_id=turn_id,
                    turn_revision=0,
                    created_at_s=0.0,
                )
            )
        )

    second = payloads[1]["messages"]
    assert [message["role"] for message in second] == ["system", "user", "assistant", "user"]
    assert second[1]["content"] == "Count from one to ten."
    assert second[2]["content"].startswith("One, two, three")
    assert second[3]["content"][0]["type"] == "input_audio"


def test_valid_primary_transcript_replaces_session_audio_anchor():
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=True)
    chat = Chat(30)
    handler._stream_request = lambda *_args, **_kwargs: _FakeSSEStream(
        _sse_text("USER_TRANSCRIPT: Count from one to ten.\nASSISTANT_RESPONSE: One, two, three.")
    )
    vad_audio = SimpleNamespace(
        audio=np.zeros(1600, dtype=np.float32),
        mode="final",
        runtime_config=RuntimeConfig(chat=chat),
        turn_id="turn_replace",
        turn_revision=0,
        created_at_s=0.0,
    )

    list(handler.process(vad_audio))

    assert chat.buffer[0].content[0].type == "input_text"
    assert chat.buffer[0].content[0].text == "Count from one to ten."


def test_cancelled_primary_retains_only_accepted_user_audio():
    class _CancelledScope:
        generation = 7

        @staticmethod
        def is_stale(_generation):
            return True

    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(
        model_name="gemma-test",
        base_url="http://127.0.0.1:8818/v1",
        stream=True,
        cancel_scope=_CancelledScope(),
    )
    chat = Chat(30)
    handler._stream_request = lambda *_args, **_kwargs: _FakeSSEStream(
        _sse_text(
            "USER_MEMORY: This memory belongs only to the cancelled response.\n"
            "ASSISTANT_RESPONSE: This partial answer must not persist."
        )
    )
    vad_audio = SimpleNamespace(
        audio=np.zeros(1600, dtype=np.float32),
        mode="final",
        runtime_config=RuntimeConfig(chat=chat),
        turn_id="turn_cancelled",
        turn_revision=0,
        created_at_s=0.0,
    )

    assert list(handler.process(vad_audio)) == []
    assert [item.type for item in chat.buffer] == ["message"]
    assert chat.buffer[0].content[0].type == "input_audio"


def test_payload_serialization_failure_still_retains_accepted_user_audio():
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=True)
    chat = Chat(30)

    def fail_payload(*_args, **_kwargs):
        raise RuntimeError("serialization failed")

    handler._payload = fail_payload
    vad_audio = SimpleNamespace(
        audio=np.zeros(1600, dtype=np.float32),
        mode="final",
        runtime_config=RuntimeConfig(chat=chat),
        turn_id="turn_payload_failure",
        turn_revision=0,
        created_at_s=0.0,
    )

    outputs = list(handler.process(vad_audio))

    assert outputs[-1].error == "Direct audio model request failed: RuntimeError"
    assert [item.type for item in chat.buffer] == ["message"]
    assert chat.buffer[0].content[0].type == "input_audio"


def test_non_json_sse_debug_log_never_contains_stream_content(caplog):
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=True)
    chat = Chat(30)
    vad_audio = SimpleNamespace(
        runtime_config=RuntimeConfig(chat=chat),
        turn_id="turn_private_log",
        turn_revision=0,
        created_at_s=0.0,
    )
    handler._commit_accepted_audio(vad_audio, _encoded_silence(handler))
    secret = "PRIVATE TRANSCRIPT CONTENT MUST NOT BE LOGGED"
    memory_secret = "PRIVATE SEMANTIC MEMORY MUST NOT BE LOGGED"
    stream = _FakeSSEStream(
        [
            secret,
            *_sse_text(f"USER_MEMORY: {memory_secret}\nASSISTANT_RESPONSE: Ready."),
        ]
    )
    caplog.set_level(logging.DEBUG, logger="speech_to_speech.STT.gemma_audio_handler")

    list(handler._consume_stream(stream, vad_audio))

    assert secret not in caplog.text
    assert memory_secret not in caplog.text
    assert "Ignoring non-JSON Gemma stream event (chars=" in caplog.text


def test_legacy_notifier_does_not_duplicate_direct_transcript_when_context_is_committed():
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=False)
    chat = Chat(30)
    runtime_config = RuntimeConfig(chat=chat)
    vad_audio = SimpleNamespace(
        runtime_config=runtime_config,
        turn_id="legacy_direct",
        turn_revision=0,
        created_at_s=0.0,
    )
    handler._commit_accepted_audio(vad_audio, _encoded_silence(handler))
    direct = list(
        handler._responses_from_text(
            "USER_TRANSCRIPT: Keep this once.\nASSISTANT_RESPONSE: Done.",
            vad_audio,
        )
    )[-1]
    notifier = object.__new__(TranscriptionNotifier)
    notifier.setup(runtime_config=runtime_config)
    turns_before = chat.stats()["turns"]

    requests = list(notifier.process(direct))

    assert len(requests) == 1
    assert chat.stats()["turns"] == turns_before == 1
    assert [item.content[0].text for item in chat.buffer] == ["Keep this once.", "Done."]


def test_streamed_mixed_response_without_language_marker_uses_auto():
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=True)
    runtime_config = RuntimeConfig(chat=Chat(30))
    handler._stream_request = lambda *_args, **_kwargs: _FakeSSEStream(
        _sse_text(
            "USER_TRANSCRIPT: Can you buscar la estación?\n"
            "ASSISTANT_RESPONSE: Sí, it is two blocks ahead."
        )
    )
    vad_audio = SimpleNamespace(
        audio=np.zeros(1600, dtype=np.float32),
        mode="final",
        runtime_config=runtime_config,
        turn_id="turn_streamed_auto",
        turn_revision=0,
        created_at_s=0.0,
    )

    outputs = list(handler.process(vad_audio))

    spoken = [output for output in outputs if output.text]
    assert "".join(output.text for output in spoken) == "Sí, it is two blocks ahead."
    assert all(output.language_code == "Auto" for output in spoken)
    assert runtime_config.local_pipeline["assistant_language"] == "Auto"


def test_one_hundred_accepted_turns_make_one_primary_request_each_with_optional_transcripts():
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=True)
    chat = Chat(30)
    calls = []

    def stream_request(_url, payload, *, api_key):
        turn = len(calls) + 1
        calls.append(payload)
        prefix = (
            f"USER_MEMORY: semantic request {turn}\n"
            if turn in {9, 37, 73, 96}
            else f"USER_TRANSCRIPT: request {turn}\n"
        )
        return _FakeSSEStream(
            _sse_text(f"{prefix}ASSISTANT_LANGUAGE: English\nASSISTANT_RESPONSE: answer {turn}.")
        )

    handler._stream_request = stream_request
    all_outputs = []
    for turn in range(1, 101):
        vad_audio = SimpleNamespace(
            audio=np.zeros(1600, dtype=np.float32),
            mode="final",
            runtime_config=RuntimeConfig(chat=chat),
            turn_id=f"turn_{turn}",
            turn_revision=0,
            created_at_s=0.0,
        )
        outputs = list(handler.process(vad_audio))
        all_outputs.append(outputs)
        assert "".join(output.text for output in outputs) == f"answer {turn}."

    assert len(calls) == 100
    assert all(all_outputs[index - 1][-1].transcript is None for index in {9, 37, 73, 96})
    assert all(output[-1].is_final is True for output in all_outputs)
    assert chat.stats()["turns"] == 30
    assert len(chat.buffer) == 60
    assert all(chat.buffer[index].role == "user" for index in range(0, len(chat.buffer), 2))
    assert all(chat.buffer[index].role == "assistant" for index in range(1, len(chat.buffer), 2))


@pytest.mark.parametrize(
    "audio",
    [
        np.zeros(1600, dtype=np.float32),
        np.tile(np.array([0.01, -0.01], dtype=np.float32), 800),
    ],
)
def test_every_vad_accepted_audio_fixture_reaches_gemma_and_empty_output_closes(audio):
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=True)
    requests = []
    handler._stream_request = lambda _url, payload, *, api_key: (
        requests.append(payload) or _FakeSSEStream(["data: [DONE]"])
    )
    vad_audio = SimpleNamespace(
        audio=audio,
        mode="final",
        runtime_config=RuntimeConfig(chat=Chat(30)),
        turn_id="turn_accepted",
        turn_revision=0,
        created_at_s=0.0,
    )

    outputs = list(handler.process(vad_audio))

    assert len(requests) == 1
    assert len(outputs) == 1
    assert outputs[0].text == ""
    assert outputs[0].tools == []
    assert outputs[0].is_final is True

def test_direct_response_records_language_for_post_tool_tts():
    handler = object.__new__(GemmaAudioSTTHandler)
    runtime_config = RuntimeConfig()
    vad_audio = SimpleNamespace(
        runtime_config=runtime_config,
        turn_id="turn_language",
        turn_revision=0,
        created_at_s=0.0,
    )

    response = handler._direct(vad_audio, "Ich suche danach.", is_final=True, language_code="German")

    assert response.language_code == "German"
    assert runtime_config.local_pipeline["assistant_language"] == "German"


def test_direct_assistant_response_passes_through_transcription_notifier_and_llm():
    notifier = object.__new__(TranscriptionNotifier)
    notifier.setup(text_output_queue=None, runtime_config=None, should_listen=Event())

    direct = DirectAssistantResponse(text="Hello from Gemma", turn_id="turn_1", turn_revision=0)
    requests = list(notifier.process(direct))

    assert requests == [
        DirectAssistantRequest(text="Hello from Gemma", turn_id="turn_1", turn_revision=0, runtime_config=None)
    ]

    llm = object.__new__(_PassThroughHandler)
    outputs = list(llm.process(requests[0]))

    assert isinstance(outputs[0], LLMResponseChunk)
    assert outputs[0].text == "Hello from Gemma"
    assert outputs[-1].tag == "end_of_response"




def test_final_direct_response_marks_transcription_as_already_answered():
    from queue import Queue

    queue = Queue()
    notifier = object.__new__(TranscriptionNotifier)
    notifier.setup(text_output_queue=queue, runtime_config=None, should_listen=Event())

    list(
        notifier.process(
            DirectAssistantResponse(
                text="Hello from Gemma",
                transcript="Hello there",
                is_final=True,
                turn_id="turn_1",
                turn_revision=0,
            )
        )
    )

    completed = queue.get_nowait()
    assert completed.transcript == "Hello there"
    assert completed.direct_audio_completed is True


def test_early_finalized_direct_transcript_is_persistent_and_not_duplicated():
    from queue import Empty, Queue

    queue = Queue()
    notifier = object.__new__(TranscriptionNotifier)
    notifier.setup(text_output_queue=queue, runtime_config=None, should_listen=Event())
    early = DirectAssistantResponse(
        text="",
        transcript="Search for Control 2.",
        is_final=False,
        transcript_finalized=True,
        turn_id="turn_early",
        turn_revision=0,
    )
    final = DirectAssistantResponse(
        text="I will look that up.",
        transcript="Search for Control 2.",
        is_final=True,
        turn_id="turn_early",
        turn_revision=0,
    )

    list(notifier.process(early))
    event = queue.get_nowait()
    assert isinstance(event, TranscriptionCompletedEvent)
    assert event.transcript == "Search for Control 2."
    assert event.direct_audio_completed is True

    list(notifier.process(final))
    with pytest.raises(Empty):
        queue.get_nowait()
