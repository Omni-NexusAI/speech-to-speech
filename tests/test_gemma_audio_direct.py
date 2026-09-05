import json
from queue import Queue
from threading import Event, Thread
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from openai.types.realtime.conversation_item import RealtimeConversationItemFunctionCallOutput
from openai.types.responses import ResponseFunctionToolCall

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.LLM.base_openai_compatible_language_model import BaseOpenAICompatibleHandler
from speech_to_speech.LLM.chat import Chat, make_assistant_message, make_user_message
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.events import TranscriptionCompletedEvent
from speech_to_speech.pipeline.messages import (
    DirectAssistantRequest,
    DirectAssistantResponse,
    EndOfResponse,
    LLMResponseChunk,
)
from speech_to_speech.pipeline.model_operations import ModelOperationCoordinator
from speech_to_speech.pipeline.response_ownership import ResponseOwnershipTracker
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
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
    assert GemmaAudioSTTHandler._extract_transcript("USER_TRANSCRIPT: Open the camera\nASSISTANT: Certainly.") == "Open the camera"
    assert GemmaAudioSTTHandler._extract_transcript("TRANSCRIPT: Search for Control 2\nRESPONSE: One moment.") == "Search for Control 2"
    assert GemmaAudioSTTHandler._extract_transcript("I can help with that.") is None
    assert GemmaAudioSTTHandler._extract_transcript(
        "USER_TRANSCRIPT: Listen to the attached user audio and respond directly as a concise voice assistant."
    ) is None


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


def test_tool_preamble_has_a_safe_fallback_when_the_model_omits_it():
    camera = ResponseFunctionToolCall(
        type="function_call", name="camera_snapshot", arguments="{}", call_id="call_camera", id="fc_camera", status="completed"
    )
    search = ResponseFunctionToolCall(
        type="function_call", name="web_search", arguments="{}", call_id="call_search", id="fc_search", status="completed"
    )
    camera_fallback = GemmaAudioSTTHandler._tool_preamble("", [camera])
    search_fallback = GemmaAudioSTTHandler._tool_preamble("", [search])
    assert camera_fallback in {
        "I'll take a closer look.",
        "Let me see what you're showing me.",
        "I'll check the camera view.",
    }
    assert search_fallback in {
        "I'll look that up.",
        "I'll check the latest information.",
        "I'll find that for you.",
    }
    assert GemmaAudioSTTHandler._tool_preamble("", [camera]) == camera_fallback
    assert GemmaAudioSTTHandler._tool_preamble("", [search]) == search_fallback


def test_direct_gemma_current_response_epoch_overrides_newer_speculative_turn():
    """A retained audible owner must still deliver its terminal output."""

    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_new", 0)
    runtime = RuntimeConfig()
    runtime.local_pipeline["_response_epoch_is_current"] = lambda epoch: epoch == 17
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.cancel_scope = None
    handler.speculative_turns = tracker
    vad_audio = SimpleNamespace(
        runtime_config=runtime,
        turn_id="turn_old",
        turn_revision=0,
        response_epoch=17,
    )

    assert handler._request_is_stale(vad_audio, generation=None) is False


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
    assert [item.type for item in chat.buffer] == ["message", "message"]
    assert chat.stats()["pending_tool_calls"] == 1


def test_tool_preamble_without_transcript_is_still_retained_for_tool_continuity():
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

    outputs = list(
        handler._responses_from_text(
            "ASSISTANT_PREAMBLE: Let me take a closer look.",
            vad_audio,
            tools=[tool],
        )
    )

    assert outputs[0].transcript is None
    assert outputs[0].text == "Let me take a closer look."
    assert [item.type for item in chat.buffer] == ["message"]
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

    assert handler._commit_context(vad_audio, None, "", [tool]) is True
    assert chat.stats()["pending_tool_calls"] == 1
    assert chat.buffer == []
    pending = chat._pending_tool_calls["call_search"]
    assert chat._assistant_exchange_ids[pending.id].startswith("direct:turn_1:0:")


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

    assert tools[0].call_id == "call_UNM0K7ZOZpEN5uS0vGTo1G1UnSDH8Vki"
    chat = Chat(30)
    vad_audio = SimpleNamespace(runtime_config=SimpleNamespace(chat=chat), turn_id="turn_tool", turn_revision=0)
    assert handler._commit_context(vad_audio, "Search for Control 2", "", tools) is True
    assert chat.stats()["pending_tool_calls"] == 1


def test_streamed_assistant_response_tool_preamble_is_not_spoken_twice():
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=True)
    chat = Chat(30)
    runtime = RuntimeConfig(chat=chat)
    vad_audio = SimpleNamespace(
        audio=np.zeros(1600, dtype=np.float32),
        mode="final",
        runtime_config=runtime,
        turn_id="turn_streamed_tool_preamble",
        turn_revision=0,
        input_epoch=1,
        response_epoch=1,
        response_id="resp_streamed_tool_preamble",
        created_at_s=0.0,
    )
    stream = _FakeSSEStream(
        [
            'data: {"choices":[{"delta":{"content":"ASSISTANT_RESPONSE: I will check that now."}}]}',
            (
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
                '"id":"search_once","function":{"name":"web_search",'
                '"arguments":"{\\"query\\":\\"status\\"}"}}]}}]}'
            ),
            "data: [DONE]",
        ]
    )

    outputs = list(handler._consume_stream(stream, vad_audio))

    assert [output.text for output in outputs] == ["I will check that now.", ""]
    assert outputs[-1].is_final is True
    assert [tool.name for tool in outputs[-1].tools] == ["web_search"]
    assert "".join(output.text for output in outputs) == "I will check that now."


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

    assert tools[0].call_id.startswith("call_")


def test_parallel_opaque_tool_ids_are_normalized_and_unique():
    tools = GemmaAudioSTTHandler._tool_calls_from_accum(
        {
            0: {"name": "web_search", "args": '{"query":"one"}', "id": "opaque"},
            1: {"name": "camera_snapshot", "args": "{}", "id": "opaque"},
        }
    )

    assert [tool.call_id for tool in tools] == ["call_opaque", "call_opaque_1"]


def test_missing_transcript_preserves_truthful_assistant_only_context():
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=False)
    chat = Chat(30)
    vad_audio = SimpleNamespace(
        runtime_config=RuntimeConfig(chat=chat),
        turn_id="turn_bad_audio",
        turn_revision=0,
        created_at_s=0.0,
    )

    outputs = list(handler._responses_from_text("ASSISTANT_RESPONSE: I heard you.", vad_audio))

    assert outputs[0].transcript is None
    assert outputs[0].text == "I heard you."
    assert outputs[0].tools == []
    assert len(chat.buffer) == 1
    assert chat.buffer[0].content[0].text == "I heard you."


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
    assert len(chat.buffer) == 1
    assert chat.buffer[0].content[0].text == "Ready."


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


def test_twelve_accepted_turns_make_one_primary_request_each_when_turn_nine_has_no_transcript():
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=True)
    chat = Chat(30)
    calls = []

    def stream_request(_url, payload, *, api_key):
        turn = len(calls) + 1
        calls.append(payload)
        prefix = "" if turn == 9 else f"USER_TRANSCRIPT: request {turn}\n"
        return _FakeSSEStream(
            _sse_text(f"{prefix}ASSISTANT_LANGUAGE: English\nASSISTANT_RESPONSE: answer {turn}.")
        )

    handler._stream_request = stream_request
    all_outputs = []
    for turn in range(1, 13):
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

    turn_nine_final = all_outputs[8][-1]
    assert len(calls) == 12
    assert turn_nine_final.transcript is None
    assert all_outputs[9][-1].is_final is True
    assert chat.stats()["turns"] == 11


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


def test_stale_direct_stream_event_cannot_commit_tool_or_emit_response():
    """Supersession between SSE events must detach Gemma before tool/history output."""

    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=True)
    chat = Chat(30)
    runtime = RuntimeConfig(chat=chat)
    admitted = {"value": True}
    runtime.local_pipeline["_response_epoch_admits_output"] = lambda _epoch: admitted["value"]
    vad_audio = SimpleNamespace(
        runtime_config=runtime,
        turn_id="turn_stale_stream",
        turn_revision=0,
        input_epoch=3,
        response_epoch=9,
        response_id="resp_stale_stream",
        created_at_s=0.0,
    )

    class Stream:
        def iter_lines(self):
            # This event creates no history itself. The simulated newer speech
            # arrives before Gemma's next event, which carries a tool call.
            yield 'data: {"choices":[{"delta":{"content":"draft"}}]}'
            admitted["value"] = False
            yield (
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
                '"id":"old_tool","function":{"name":"web_search","arguments":"{}"}}]}}]}'
            )

        def close(self):
            return None

    assert list(handler._consume_stream(Stream(), vad_audio)) == []
    assert chat.buffer == []


def test_next_direct_request_keeps_assistant_only_answer_without_fabricated_transcript():
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=False)
    chat = Chat(30)
    runtime = RuntimeConfig(chat=chat)
    first = SimpleNamespace(runtime_config=runtime, turn_id="turn_first", turn_revision=0, created_at_s=0.0)
    list(handler._responses_from_text("ASSISTANT_RESPONSE: One, two, three, four, five.", first))

    second = SimpleNamespace(runtime_config=runtime)
    payload = handler._payload(np.zeros(1600, dtype=np.float32), second)

    assert payload["messages"][1:-1] == [{"role": "assistant", "content": "One, two, three, four, five."}]
    assistant = chat.buffer[0]
    assert chat._assistant_exchange_ids[assistant.id].startswith("direct:turn_first:0:")


def test_stale_direct_final_cannot_commit_history_or_terminal_output():
    """The final direct path rechecks ownership before history and its terminal."""

    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=False)
    chat = Chat(30)
    runtime = RuntimeConfig(chat=chat)
    runtime.local_pipeline["_response_epoch_admits_output"] = lambda _epoch: False
    vad_audio = SimpleNamespace(
        runtime_config=runtime,
        turn_id="turn_stale_final",
        turn_revision=0,
        input_epoch=4,
        response_epoch=10,
        response_id="resp_stale_final",
        created_at_s=0.0,
    )

    text = "USER_TRANSCRIPT: old request\nASSISTANT_RESPONSE: old response."
    assert list(handler._responses_from_text(text, vad_audio)) == []
    assert chat.buffer == []


def test_direct_history_commit_is_atomic_with_vad_invalidation_and_rollback():
    """A VAD invalidation cannot slip between admission and a Chat transaction.

    The direct worker intentionally holds the tracker guard while it appends a
    user message, assistant answer, and function call. A concurrent VAD
    invalidation must wait until that short transaction completes, then restore
    the owner checkpoint so no stale item survives into the successor context.
    """

    class BlockingChat(Chat):
        def __init__(self):
            super().__init__(30)
            self.item_entered = Event()
            self.allow_transaction = Event()

        def add_item(self, item):
            if not self.item_entered.is_set():
                self.item_entered.set()
                assert self.allow_transaction.wait(1)
            return super().add_item(item)

    tracker = ResponseOwnershipTracker()
    owner = tracker.claim_pending(turn_id="turn_atomic", turn_revision=0)
    chat = BlockingChat()
    checkpoint = chat.checkpoint()
    runtime = RuntimeConfig(chat=chat)
    runtime.local_pipeline["_response_epoch_admits_output"] = tracker.admits_output
    runtime.local_pipeline["_response_epoch_history_transaction"] = tracker.history_transaction
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(model_name="gemma-test", base_url="http://127.0.0.1:8818/v1", stream=False)
    vad_audio = SimpleNamespace(
        runtime_config=runtime,
        turn_id="turn_atomic",
        turn_revision=0,
        input_epoch=owner.input_epoch,
        response_epoch=owner.response_epoch,
    )
    tool = ResponseFunctionToolCall(
        type="function_call",
        name="web_search",
        arguments="{}",
        call_id="call_atomic",
        id="fc_atomic",
        status="completed",
    )
    committed: list[bool] = []
    worker = Thread(
        target=lambda: committed.append(handler._commit_context(vad_audio, "latest user", "old answer", [tool])),
        daemon=True,
    )
    worker.start()
    assert chat.item_entered.wait(1)

    invalidation_attempted = Event()
    invalidation_finished = Event()

    def invalidate_and_rollback() -> None:
        invalidation_attempted.set()
        tracker.input_started(reason="newer_speech")
        chat.rollback(checkpoint)
        invalidation_finished.set()

    invalidator = Thread(target=invalidate_and_rollback, daemon=True)
    invalidator.start()
    assert invalidation_attempted.wait(1)
    # The invalidator cannot mark stale or roll back while the response's
    # atomic Chat transaction is in progress.
    assert not invalidation_finished.wait(0.05)
    chat.allow_transaction.set()
    worker.join(1)
    invalidator.join(1)

    assert committed == [True]
    assert invalidation_finished.is_set()
    # User, assistant, and function-call entries all roll back together; no
    # stale fragment can become the final input's history.
    assert chat.buffer == []
    assert chat.stats()["pending_tool_calls"] == 0


def test_progressive_transcription_uses_captured_generation_and_stops_after_cancellation():
    """A live preview must not leak text after its captured generation is stale."""

    scope = CancelScope()
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(
        model_name="gemma-test",
        base_url="http://127.0.0.1:8818/v1",
        stream=True,
        cancel_scope=scope,
    )
    stream = _FakeSSEStream(
        [
            'data: {"choices":[{"delta":{"content":"TRANSCRIPT: first preview"}}]}',
            'data: {"choices":[{"delta":{"content":" must not leak"}}]}',
        ]
    )
    handler._stream_request = lambda _url, _payload, *, api_key: stream
    seen_generations: list[int | None] = []
    admitted = handler._direct_output_admitted

    def capture_generation(vad_audio, generation=None):
        seen_generations.append(generation)
        return admitted(vad_audio, generation)

    handler._direct_output_admitted = capture_generation

    def cancel_after_first_preview():
        yield stream._lines[0]
        scope.cancel()
        yield stream._lines[1]

    stream.iter_lines = cancel_after_first_preview
    vad_audio = SimpleNamespace(
        audio=np.zeros(1600, dtype=np.float32),
        mode="progressive",
        runtime_config=RuntimeConfig(local_pipeline={"live_transcription": True}),
        turn_id="turn_live_cancel",
        turn_revision=0,
        created_at_s=0.0,
    )

    previews = list(handler.process(vad_audio))

    assert [preview.text for preview in previews] == ["first preview"]
    assert seen_generations and all(generation == 0 for generation in seen_generations)
    assert stream.closed is True


def test_queued_direct_audio_refreshes_cancel_generation_after_promotion():
    """A queued successor binds to the generation current after its wait."""

    scope = CancelScope()
    acquired_generations: list[int | None] = []
    processed_generations: list[int | None] = []

    class Coordinator:
        def acquire(self, **kwargs):
            acquired_generations.append(kwargs["cancel_generation"])
            return SimpleNamespace(operation_id=1)

        def bind_cancel(self, _operation, _cancel):
            return True

        def is_current(self, _operation):
            return True

        def release(self, _operation):
            return None

    def wait_until_promoted(_response_epoch):
        scope.cancel()
        return True

    runtime = RuntimeConfig(
        local_pipeline={
            "_session_id": "session-queued",
            "_wait_response_epoch_current": wait_until_promoted,
            "_response_epoch_admits_output": lambda _epoch: True,
        }
    )
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(
        model_name="gemma-test",
        base_url="http://127.0.0.1:8818/v1",
        stream=True,
        cancel_scope=scope,
        model_operations=Coordinator(),
    )

    def process_direct(_vad_audio, generation):
        processed_generations.append(generation)
        yield DirectAssistantResponse(text="current", is_final=True)

    handler._process_direct = process_direct
    vad_audio = SimpleNamespace(
        audio=np.zeros(1600, dtype=np.float32),
        mode="final",
        runtime_config=runtime,
        turn_id="turn-queued",
        turn_revision=0,
        input_epoch=2,
        response_epoch=2,
        response_id="resp-queued",
        created_at_s=0.0,
    )

    assert [item.text for item in handler.process(vad_audio)] == ["current"]
    assert scope.generation == 1
    assert acquired_generations == [1]
    assert processed_generations == [1]


def test_direct_gemma_pre_transport_cancel_closes_late_registered_stream():
    """Cancellation between coordinator acquire and transport creation is latched."""

    coordinator = ModelOperationCoordinator()
    handler = object.__new__(GemmaAudioSTTHandler)
    handler.setup(
        model_name="gemma-test",
        base_url="http://127.0.0.1:8818/v1",
        stream=True,
        model_operations=coordinator,
    )
    construction_entered = Event()
    allow_construction = Event()
    headers_called = Event()

    class LateStream(_FakeSSEStream):
        def wait_for_headers(self):
            headers_called.set()

    stream = LateStream([])

    def delayed_stream_request(_url, _payload, *, api_key):
        del api_key
        construction_entered.set()
        assert allow_construction.wait(1.0)
        return stream

    handler._stream_request = delayed_stream_request
    runtime = RuntimeConfig(
        local_pipeline={
            "_session_id": "session-pre-transport",
            "_response_epoch_admits_output": lambda _epoch: True,
        }
    )
    vad_audio = SimpleNamespace(
        audio=np.zeros(1600, dtype=np.float32),
        mode="final",
        runtime_config=runtime,
        turn_id="turn-pre-transport",
        turn_revision=0,
        input_epoch=1,
        response_epoch=1,
        response_id=None,
        created_at_s=0.0,
    )
    outputs: list[object] = []
    worker = Thread(target=lambda: outputs.extend(handler.process(vad_audio)), daemon=True)
    worker.start()
    assert construction_entered.wait(1.0)
    token = coordinator.active_token()
    assert token is not None
    assert coordinator.request_cancel_token(token, "newer_speech") is True
    allow_construction.set()
    worker.join(1.0)

    assert not worker.is_alive()
    assert outputs == []
    assert stream.closed is True
    assert not headers_called.is_set()
    assert coordinator.active_token() is None
