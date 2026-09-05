from types import SimpleNamespace
import threading

from openai.types.realtime.conversation_item import (
    RealtimeConversationItemFunctionCall,
    RealtimeConversationItemFunctionCallOutput,
)

from speech_to_speech.LLM.chat import Chat, make_assistant_message, make_user_message
from speech_to_speech.LLM.direct_history_compaction import DirectHistoryMaintenance, normalize_history_compaction
from speech_to_speech.pipeline.model_operations import ModelOperationCoordinator


def _runtime(chat, *, context=10000, provider="remote"):
    endpoint = SimpleNamespace(
        base_url="http://remote.example/v1", model="chosen-model", api_key="secret", context_window=context,
        provider=provider, model_copy=lambda deep=True: endpoint,
    )
    return SimpleNamespace(
        chat=chat,
        model_endpoint=endpoint,
        local_pipeline={"_session_id": "s", "history_compaction": {"enabled": True, "trigger_ratio": .2, "target_ratio": .15, "recent_turns": 2}},
    )


def _chat():
    chat = Chat(2)
    chat.enable_token_managed_history()
    for i in range(5):
        chat.add_item(make_user_message(f"question {i}"))
        chat.add_item(make_assistant_message(f"answer {i}"))
    return chat


def test_policy_is_bounded_and_uses_documented_defaults():
    assert normalize_history_compaction({}) == {"enabled": True, "trigger_ratio": .7, "target_ratio": .5, "recent_turns": 6}
    assert normalize_history_compaction({"trigger_ratio": 9, "target_ratio": 8, "recent_turns": 99}) == {
        "enabled": True, "trigger_ratio": .9, "target_ratio": .85, "recent_turns": 12
    }


def test_target_failure_leaves_original_history_intact(monkeypatch):
    chat = _chat()
    before = list(chat.buffer)
    runtime = _runtime(chat, context=10000)

    class Stream:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs
        def iter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":"{\\"user_summary\\":\\"' + ('x' * 5000) + '\\",\\"assistant_summary\\":\\"y\\"}"}}]}'
            yield "data: [DONE]"
        def close(self):
            pass

    monkeypatch.setattr("speech_to_speech.LLM.direct_history_compaction.CancellableAsyncSSEStream", Stream)
    DirectHistoryMaintenance(ModelOperationCoordinator())._run(runtime, normalize_history_compaction(runtime.local_pipeline["history_compaction"]))
    assert chat.buffer == before
    assert chat.memory_summary() is None


def test_selected_endpoint_and_cancellable_transport_are_used(monkeypatch):
    chat = _chat()
    runtime = _runtime(chat, context=10000)
    captured = {}

    class Stream:
        def __init__(self, method, url, **kwargs):
            captured.update(method=method, url=url, body=kwargs["json_body"], headers=kwargs["headers"])
        def iter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":"{\\"user_summary\\":\\"u\\",\\"assistant_summary\\":\\"a\\"}"}}]}'
            yield "data: [DONE]"
        def close(self):
            captured["closed"] = True

    monkeypatch.setattr("speech_to_speech.LLM.direct_history_compaction.CancellableAsyncSSEStream", Stream)
    DirectHistoryMaintenance(ModelOperationCoordinator())._run(runtime, normalize_history_compaction(runtime.local_pipeline["history_compaction"]))
    assert captured["url"] == "http://remote.example/v1/chat/completions"
    assert captured["body"]["model"] == "chosen-model"
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["closed"]


def test_speech_preemption_cancels_the_owned_summary_transport(monkeypatch):
    chat = _chat()
    runtime = _runtime(chat, context=10000)
    opened = threading.Event()
    closed = threading.Event()

    class Stream:
        def __init__(self, *args, **kwargs):
            pass
        def iter_lines(self):
            opened.set()
            closed.wait(1)
            yield "data: [DONE]"
        def close(self):
            closed.set()

    coordinator = ModelOperationCoordinator()
    monkeypatch.setattr("speech_to_speech.LLM.direct_history_compaction.CancellableAsyncSSEStream", Stream)
    worker = threading.Thread(target=DirectHistoryMaintenance(coordinator)._run, args=(runtime, normalize_history_compaction(runtime.local_pipeline["history_compaction"])))
    worker.start()
    assert opened.wait(.5)
    result = coordinator.cancel_and_wait("newer_speech", .5)
    worker.join(.5)
    assert result.released and closed.is_set() and not worker.is_alive()


def test_budget_names_actual_serialized_fields_and_estimated_multimodal_reserves():
    runtime = _runtime(_chat(), context=10000)
    runtime.local_pipeline["_history_compaction_budget"] = {
        "history_serialized_chars": 400,
        "system_serialized_chars": 200,
        "tools_serialized_chars": 80,
        "response_allowance_tokens": 384,
        "media_reserve_tokens_estimate": 1024,
        "image_reserve_tokens_estimate": 256,
    }
    budget = DirectHistoryMaintenance._budget(runtime)
    assert budget["response_allowance_tokens"] == 384
    assert budget["media_reserve_source"] == "direct_audio_estimate"
    assert budget["estimated_request_tokens"] >= 384 + 1024 + 256


def test_endpoint_change_after_summary_leaves_history_intact(monkeypatch):
    chat = _chat()
    before = list(chat.buffer)
    runtime = _runtime(chat, context=10000)
    runtime.local_pipeline["history_compaction"]["target_ratio"] = .9
    live_endpoint = runtime.model_endpoint
    live_endpoint.model_copy = lambda deep=True: SimpleNamespace(
        base_url=live_endpoint.base_url,
        model=live_endpoint.model,
        api_key=live_endpoint.api_key,
        context_window=live_endpoint.context_window,
        provider=live_endpoint.provider,
    )

    class Stream:
        def __init__(self, *args, **kwargs):
            pass
        def iter_lines(self):
            runtime.model_endpoint.model = "newly-selected-model"
            yield 'data: {"choices":[{"delta":{"content":"{\\"user_summary\\":\\"u\\",\\"assistant_summary\\":\\"a\\"}"}}]}'
            yield "data: [DONE]"
        def close(self):
            pass

    monkeypatch.setattr("speech_to_speech.LLM.direct_history_compaction.CancellableAsyncSSEStream", Stream)
    DirectHistoryMaintenance(ModelOperationCoordinator())._run(runtime, normalize_history_compaction(runtime.local_pipeline["history_compaction"]))
    assert chat.buffer == before
    assert runtime.local_pipeline["_history_compaction_telemetry"]["last_failure"] == "endpoint_changed"


def test_truncated_or_error_summary_stream_cannot_splice_history(monkeypatch):
    chat = _chat()
    before = list(chat.buffer)
    runtime = _runtime(chat, context=10000)

    class Stream:
        def __init__(self, *args, **kwargs):
            pass
        def iter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":"{\\"user_summary\\":\\"u\\",\\"assistant_summary\\":\\"a\\"}"}}]}'
        def close(self):
            pass

    monkeypatch.setattr("speech_to_speech.LLM.direct_history_compaction.CancellableAsyncSSEStream", Stream)
    DirectHistoryMaintenance(ModelOperationCoordinator())._run(runtime, normalize_history_compaction(runtime.local_pipeline["history_compaction"]))
    assert chat.buffer == before
    assert runtime.local_pipeline["_history_compaction_telemetry"]["last_failure"] == "summary_stream_truncated"


def test_summary_request_is_bounded_before_transport(monkeypatch):
    chat = _chat()
    runtime = _runtime(chat, context=256)
    opened = False

    class Stream:
        def __init__(self, *args, **kwargs):
            nonlocal opened
            opened = True
        def iter_lines(self):
            return iter(())
        def close(self):
            pass

    monkeypatch.setattr("speech_to_speech.LLM.direct_history_compaction.CancellableAsyncSSEStream", Stream)
    DirectHistoryMaintenance(ModelOperationCoordinator())._run(runtime, normalize_history_compaction(runtime.local_pipeline["history_compaction"]))
    assert opened is False
    assert runtime.local_pipeline["_history_compaction_telemetry"]["last_failure"] == "summary_request_too_large"


def test_assistant_only_turns_compact_without_fabricated_user_history():
    chat = Chat(2)
    chat.enable_token_managed_history()
    for number in range(8):
        item = make_assistant_message(f"answer {number}")
        chat.add_item(item)
        chat.mark_assistant_only_exchange(item.id, f"audio-{number}")

    _snapshot, markers, _revision = chat.memory_snapshot(6)

    assert len(markers) == 2
    assert all(item.role == "assistant" for item in chat.buffer if item.id in markers)


def test_mixed_history_keeps_a_tool_turn_and_its_terminal_answer_together():
    chat = Chat(2)
    chat.enable_token_managed_history()

    user = make_user_message("typed request")
    typed_answer = make_assistant_message("typed answer")
    chat.add_item(user)
    chat.add_item(typed_answer)

    preamble = make_assistant_message("I will check that.")
    chat.add_item(preamble)
    chat.mark_assistant_only_exchange(preamble.id, "audio-2")
    call = RealtimeConversationItemFunctionCall(
        type="function_call", id="fc_audio_2", call_id="call_audio_2", name="lookup", arguments="{}"
    )
    chat.add_item(call)
    chat.mark_direct_exchange_item(call.id, "audio-2")
    chat.add_item(
        RealtimeConversationItemFunctionCallOutput(
            type="function_call_output", id="fco_audio_2", call_id="call_audio_2", output='{"ok":true}'
        )
    )
    terminal = make_assistant_message("The result is ready.")
    chat.add_item(terminal)
    chat.mark_assistant_only_exchange(terminal.id, "audio-2")

    later_audio = make_assistant_message("A later transcript-less answer.")
    chat.add_item(later_audio)
    chat.mark_assistant_only_exchange(later_audio.id, "audio-3")

    _snapshot, markers, _revision = chat.memory_snapshot(2)

    # The two most recent exchanges are the complete tool exchange and the
    # next direct answer.  No preamble/tool/final fragment can be compacted
    # independently, and no user wording has been fabricated.
    assert user.id in markers and typed_answer.id in markers
    assert preamble.id not in markers
    assert call.id not in markers
    assert terminal.id not in markers
    assert later_audio.id not in markers


def test_transcriptless_tool_turn_without_preamble_starts_its_own_exchange():
    chat = Chat(2)
    chat.enable_token_managed_history()
    user = make_user_message("typed request")
    typed_answer = make_assistant_message("typed answer")
    chat.add_item(user)
    chat.add_item(typed_answer)

    call = RealtimeConversationItemFunctionCall(
        type="function_call", id="fc_audio_no_preamble", call_id="call_audio_no_preamble", name="lookup", arguments="{}"
    )
    chat.add_item(call)
    chat.mark_direct_exchange_item(call.id, "audio-no-preamble")
    output = RealtimeConversationItemFunctionCallOutput(
        type="function_call_output", id="fco_audio_no_preamble", call_id="call_audio_no_preamble", output='{"ok":true}'
    )
    chat.add_item(output)
    terminal = make_assistant_message("Completed without a spoken preamble.")
    chat.add_item(terminal)
    chat.mark_assistant_only_exchange(terminal.id, "audio-no-preamble")
    later_audio = make_assistant_message("Another direct answer.")
    chat.add_item(later_audio)
    chat.mark_assistant_only_exchange(later_audio.id, "audio-later")

    _snapshot, markers, _revision = chat.memory_snapshot(2)

    assert user.id in markers and typed_answer.id in markers
    assert call.id not in markers and output.id not in markers
    assert terminal.id not in markers and later_audio.id not in markers


def test_detached_cancel_remains_sticky_when_late_stream_finishes(monkeypatch):
    chat = _chat()
    before = list(chat.buffer)
    runtime = _runtime(chat, context=10000)
    opened = threading.Event()
    close_requested = threading.Event()
    release_late_stream = threading.Event()

    class Stream:
        def __init__(self, *args, **kwargs):
            pass
        def iter_lines(self):
            opened.set()
            release_late_stream.wait(1)
            yield 'data: {"choices":[{"delta":{"content":"{\\"user_summary\\":\\"u\\",\\"assistant_summary\\":\\"a\\"}"}}]}'
            yield "data: [DONE]"
        def close(self):
            close_requested.set()

    coordinator = ModelOperationCoordinator()
    monkeypatch.setattr("speech_to_speech.LLM.direct_history_compaction.CancellableAsyncSSEStream", Stream)
    worker = threading.Thread(
        target=DirectHistoryMaintenance(coordinator)._run,
        args=(runtime, normalize_history_compaction(runtime.local_pipeline["history_compaction"])),
    )
    worker.start()
    assert opened.wait(.5)
    coordinator.cancel_and_wait("newer_speech", .01)
    assert close_requested.is_set()
    release_late_stream.set()
    worker.join(.5)
    assert chat.buffer == before
    assert runtime.local_pipeline["_history_compaction_telemetry"]["status"] == "cancelled"


def test_close_immediately_before_apply_keeps_original_history(monkeypatch):
    chat = _chat()
    before = list(chat.buffer)
    runtime = _runtime(chat, context=10000)
    runtime.local_pipeline["history_compaction"]["target_ratio"] = .9
    runtime.history_maintenance_lock = threading.RLock()
    original_projection = chat.projected_direct_history_serialized_chars

    def close_before_projection(*args, **kwargs):
        result = original_projection(*args, **kwargs)
        chat.close()
        return result

    class Stream:
        def __init__(self, *args, **kwargs):
            pass
        def iter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":"{\\"user_summary\\":\\"u\\",\\"assistant_summary\\":\\"a\\"}"}}]}'
            yield "data: [DONE]"
        def close(self):
            pass

    monkeypatch.setattr(chat, "projected_direct_history_serialized_chars", close_before_projection)
    monkeypatch.setattr("speech_to_speech.LLM.direct_history_compaction.CancellableAsyncSSEStream", Stream)
    DirectHistoryMaintenance(ModelOperationCoordinator())._run(runtime, normalize_history_compaction(runtime.local_pipeline["history_compaction"]))
    assert chat.buffer == before
    assert runtime.local_pipeline["_history_compaction_telemetry"]["status"] == "stale"
