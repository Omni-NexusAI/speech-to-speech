from speech_to_speech.LLM.native_tool_diagnostics import NativeToolStreamDiagnostics
from speech_to_speech.LLM.voice_prompt import NATIVE_TOOL_OUTPUT_CONTRACT, build_voice_system_prompt


def _choice(*, content=None, tool_calls=None, finish_reason=None):
    delta = {}
    if content is not None:
        delta["content"] = content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    return {"delta": delta, "finish_reason": finish_reason}


def test_shared_contract_requires_native_calls_without_printed_syntax():
    prompt = build_voice_system_prompt("Be concise.")

    assert prompt.count(NATIVE_TOOL_OUTPUT_CONTRACT.rstrip()) == 1
    assert "optionally give one brief natural preamble" in prompt
    assert "use only native tool_calls" in prompt
    assert "With required, emit a native call" in prompt
    assert "printed call-like prose is non-executable" in prompt
    assert "ASSISTANT_PREAMBLE" not in prompt
    assert "Tool turns may be silent" in prompt
    assert "always speak first" not in prompt
    assert "Sure, here's my best <emotion>." not in prompt


def test_required_tool_stream_reports_native_fragments_and_completed_call():
    diagnostics = NativeToolStreamDiagnostics(tool_choice="required")
    diagnostics.observe_choice(
        _choice(
            content="ASSISTANT_PREAMBLE: I'll check.",
            tool_calls=[{"index": 0, "id": "opaque", "function": {"name": "lookup", "arguments": ""}}],
        )
    )
    diagnostics.observe_choice(
        _choice(tool_calls=[{"index": 0, "function": {"arguments": '{"query":"current"}'}}])
    )
    diagnostics.observe_choice(_choice(finish_reason="tool_calls"))

    detail = diagnostics.finalize(
        {0: {"name": "lookup", "args": '{"query":"current"}', "id": "opaque"}},
        completed_call_count=1,
    )

    assert detail == {
        "finish_reason_category": "tool_calls",
        "assistant_text_length": len("ASSISTANT_PREAMBLE: I'll check."),
        "native_tool_fragment_count": 2,
        "completed_call_count": 1,
        "malformed_call_category": "none",
    }


def test_auto_plain_answer_has_no_native_tool_activity():
    diagnostics = NativeToolStreamDiagnostics(tool_choice="auto")
    diagnostics.observe_choice(_choice(content="A normal answer."))
    diagnostics.observe_choice(_choice(finish_reason="stop"))

    detail = diagnostics.finalize({}, completed_call_count=0)

    assert detail == {
        "finish_reason_category": "stop",
        "assistant_text_length": len("A normal answer."),
        "native_tool_fragment_count": 0,
        "completed_call_count": 0,
        "malformed_call_category": "none",
    }


def test_printed_prose_call_is_counted_as_text_and_never_as_a_native_call():
    fake_call = 'lookup({"query":"current"})'
    diagnostics = NativeToolStreamDiagnostics(tool_choice="auto")
    diagnostics.observe_choice(_choice(content=fake_call))
    diagnostics.observe_choice(_choice(finish_reason="stop"))

    detail = diagnostics.finalize({}, completed_call_count=0)

    assert detail["assistant_text_length"] == len(fake_call)
    assert detail["native_tool_fragment_count"] == 0
    assert detail["completed_call_count"] == 0
    assert detail["malformed_call_category"] == "none"
    assert fake_call not in detail.values()


def test_malformed_native_arguments_are_classified_without_content():
    malformed_arguments = '{"query":'
    diagnostics = NativeToolStreamDiagnostics(tool_choice="required")
    diagnostics.observe_choice(
        _choice(
            tool_calls=[
                {
                    "index": 0,
                    "id": "opaque",
                    "function": {"name": "lookup", "arguments": malformed_arguments},
                }
            ],
            finish_reason="tool_calls",
        )
    )

    detail = diagnostics.finalize(
        {0: {"name": "lookup", "args": malformed_arguments, "id": "opaque"}},
        completed_call_count=1,
    )

    assert detail["malformed_call_category"] == "native_invalid_arguments_json"
    assert malformed_arguments not in detail.values()
    assert "lookup" not in detail.values()


def test_invalid_native_index_is_classified_without_retaining_its_value():
    secret_index = "PRIVATE_MODEL_CONTROLLED_INDEX"
    diagnostics = NativeToolStreamDiagnostics(tool_choice="auto")
    diagnostics.observe_choice(
        _choice(
            tool_calls=[
                {
                    "index": secret_index,
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ]
        )
    )

    detail = diagnostics.finalize({}, completed_call_count=0)

    assert detail["malformed_call_category"] == "native_invalid_index"
    assert secret_index not in repr(diagnostics)
    assert secret_index not in repr(vars(diagnostics))
    assert secret_index not in detail.values()


def test_tool_result_continuation_with_tools_disabled_is_plain_text_only():
    diagnostics = NativeToolStreamDiagnostics(tool_choice="none")
    diagnostics.observe_choice(_choice(content="The result is ready."))
    diagnostics.observe_choice(_choice(finish_reason="stop"))

    detail = diagnostics.finalize({}, completed_call_count=0)

    assert detail == {
        "finish_reason_category": "stop",
        "assistant_text_length": len("The result is ready."),
        "native_tool_fragment_count": 0,
        "completed_call_count": 0,
        "malformed_call_category": "none",
    }


def test_required_choice_without_native_call_is_content_free_failure_category():
    diagnostics = NativeToolStreamDiagnostics(tool_choice="required")
    diagnostics.observe_choice(_choice(content="I will call it."))
    diagnostics.observe_choice(_choice(finish_reason="stop"))

    detail = diagnostics.finalize({}, completed_call_count=0)

    assert detail["malformed_call_category"] == "required_without_native_call"
    assert "I will call it." not in detail.values()


def test_named_required_choice_is_reduced_to_a_category_immediately():
    secret_tool_name = "SECRET_TOOL_NAME"
    diagnostics = NativeToolStreamDiagnostics(
        tool_choice={"type": "function", "function": {"name": secret_tool_name}}
    )

    detail = diagnostics.finalize({}, completed_call_count=0)

    assert detail["malformed_call_category"] == "required_without_native_call"
    assert secret_tool_name not in repr(diagnostics)
    assert secret_tool_name not in repr(vars(diagnostics))
