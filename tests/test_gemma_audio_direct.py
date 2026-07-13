from threading import Event
from types import SimpleNamespace

import numpy as np
from openai.types.realtime.conversation_item import RealtimeConversationItemFunctionCallOutput
from openai.types.responses import ResponseFunctionToolCall

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.LLM.base_openai_compatible_language_model import BaseOpenAICompatibleHandler
from speech_to_speech.LLM.chat import Chat, make_assistant_message, make_user_message
from speech_to_speech.pipeline.messages import DirectAssistantRequest, DirectAssistantResponse, LLMResponseChunk
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


def test_gemma_audio_payload_uses_input_audio_wav_base64():
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=False)

    payload = handler._payload(np.zeros(1600, dtype=np.float32))

    assert payload["model"] == "gemma-test"
    assert payload["stream"] is False
    content = payload["messages"][1]["content"]
    assert content[1]["type"] == "input_audio"
    assert content[1]["input_audio"]["format"] == "wav"
    assert isinstance(content[1]["input_audio"]["data"], str)


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
    assert content[1] == {"type": "image_url", "image_url": {"url": image_url}}
    assert content[2]["type"] == "input_audio"


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

    assert "Always answer as TEST ROLE." in payload["messages"][0]["content"]
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
    assert GemmaAudioSTTHandler._extract_transcript("USER: Open the camera\nASSISTANT: Certainly.") == "Open the camera"
    assert GemmaAudioSTTHandler._extract_transcript("TRANSCRIPT: Search for Control 2\nRESPONSE: One moment.") == "Search for Control 2"
    assert GemmaAudioSTTHandler._extract_transcript("I can help with that.") is None


def test_missing_primary_transcript_uses_one_transcript_only_fallback():
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=True)
    calls = []
    handler._transcribe_once = lambda vad: calls.append(vad.turn_id) or "What did the search find?"
    vad_audio = SimpleNamespace(turn_id="turn_2", turn_revision=0)

    assert handler._final_transcript(vad_audio, None) == "What did the search find?"
    assert calls == ["turn_2"]


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


def test_direct_tool_call_without_transcript_keeps_only_the_function_call_partner():
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

    assert handler._commit_context(vad_audio, None, "", [tool]) is True
    assert chat.stats()["pending_tool_calls"] == 1
    assert chat.buffer == []

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
