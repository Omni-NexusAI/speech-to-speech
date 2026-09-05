from speech_to_speech.LLM.language_model import LanguageModelHandler, StreamContext
from speech_to_speech.LLM.tool_call.function_tool import FunctionTool
from speech_to_speech.LLM.tool_call.tool_prompt import END_CODE, ENTER_CODE, build_block_regex, build_tool_system_prompt
from speech_to_speech.LLM.voice_prompt import (
    NATIVE_TOOL_OUTPUT_CONTRACT,
    VOICE_INPUT_TOOL_POLICY,
    VOICE_SYSTEM_PROMPT,
    build_voice_system_prompt,
)


def test_voice_prompt_is_short_and_keeps_persona_in_session_prompt():
    prompt = build_voice_system_prompt("Be concise.")

    assert len(VOICE_SYSTEM_PROMPT.split()) < 300
    assert len(prompt.split()) < 310
    assert "The session prompt defines persona" in prompt
    assert "Match the user's intent" not in prompt


def test_voice_prompt_reuses_semantic_input_and_search_policy_without_old_shortcuts():
    prompt = build_voice_system_prompt("Be concise.")

    assert prompt.count(VOICE_INPUT_TOOL_POLICY.rstrip()) == 1
    assert "Speech is the default response channel." in prompt
    assert "accepted turns as semantic input" in prompt
    assert "current/external/visual facts" in prompt
    assert "stable or historical questions" in prompt
    assert "today/latest/now/changed/since" in prompt
    assert "absent/undated/conflicting evidence" in prompt
    assert "resolved prior entity and applicable absolute date" in prompt
    assert "not persona/system-prompt topics" in prompt
    assert "one distinct narrower refinement" in prompt
    assert "never duplicate or broaden the search" in prompt
    assert "retrieved_at_utc is retrieval" in prompt
    assert "Report only returned dates/sources" in prompt
    assert prompt.count(NATIVE_TOOL_OUTPUT_CONTRACT.rstrip()) == 1
    assert "Treat transcripts as noisy." not in prompt
    assert "Use at most one tool" not in prompt
    assert "If unsure whether a tool is needed, just speak." not in prompt
    assert "Reachy/Richie/Richy" not in prompt


def test_voice_prompt_keeps_native_tool_preambles_optional_and_natural():
    prompt = build_voice_system_prompt("Be concise.")

    assert "Tool turns may be silent" in prompt
    assert "use one brief natural acknowledgement only when helpful" in prompt
    assert "ASSISTANT_PREAMBLE" not in prompt
    assert "Expression/background tool preambles are optional" in prompt
    assert "use stock wording" in prompt
    assert "Sure, here's my best <emotion>." not in prompt
    assert "Before tools, speak briefly" not in prompt
    assert "speak again only for user-facing result information" in prompt
    assert "Use motion/dance/emotion tools sparingly" in prompt


def test_local_tool_prompt_forbids_multiple_tool_calls():
    prompt = build_tool_system_prompt(
        [
            FunctionTool(
                type="function",
                name="dance",
                description="Dance once.",
                parameters={"type": "object", "properties": {}},
            )
        ]
    )

    assert "Only one tool call may appear in a response." in prompt
    assert "Multiple tool calls can live" not in prompt


def test_local_tool_prompt_allows_spoken_lead_in_before_code_block():
    prompt = build_tool_system_prompt(
        [
            FunctionTool(
                type="function",
                name="camera",
                description="Look through the camera.",
                parameters={"type": "object", "properties": {}},
            )
        ]
    )

    assert "one brief natural sentence before the tool call" in prompt
    assert "always speak first" in prompt
    assert "Sure, here's my best <emotion>." in prompt
    assert "Sure, here's my best sadness." not in prompt
    assert "fitting empathetic sentence" in prompt
    assert "do not claim tool results before a tool result is available" in prompt
    assert "Omit optional args instead of placeholder values" in prompt


def test_local_tool_parser_flushes_lead_in_before_tool_even_with_large_sentence_batch():
    handler = object.__new__(LanguageModelHandler)
    ctx = StreamContext(
        function_tools=[
            FunctionTool(
                type="function",
                name="dance",
                description="Dance once.",
                parameters={"type": "object", "properties": {}},
            )
        ],
        block_regex=build_block_regex(),
        enter_code=ENTER_CODE,
        end_code=END_CODE,
    )
    text = f"Here we go. {ENTER_CODE}dance(){END_CODE}"

    chunks, tools, remaining = handler._process_printable_text(text, None, [], ctx)

    assert [chunk.text for chunk in chunks] == ["Here we go.", ""]
    assert chunks[0].tools == []
    assert [tool.name for tool in chunks[1].tools] == ["dance"]
    assert [tool.name for tool in tools] == ["dance"]
    assert remaining == ""


def test_local_tool_parser_flushes_pending_batch_before_tool_with_empty_before_text():
    handler = object.__new__(LanguageModelHandler)
    ctx = StreamContext(
        function_tools=[
            FunctionTool(
                type="function",
                name="dance",
                description="Dance once.",
                parameters={"type": "object", "properties": {}},
            )
        ],
        block_regex=build_block_regex(),
        enter_code=ENTER_CODE,
        end_code=END_CODE,
        sentence_batch=["Queued lead-in."],
    )
    text = f"{ENTER_CODE}dance(){END_CODE}"

    chunks, tools, remaining = handler._process_printable_text(text, None, [], ctx)

    assert [chunk.text for chunk in chunks] == ["Queued lead-in.", ""]
    assert chunks[0].tools == []
    assert [tool.name for tool in chunks[1].tools] == ["dance"]
    assert [tool.name for tool in tools] == ["dance"]
    assert remaining == ""


def test_local_tool_parser_skips_duplicate_tool_blocks_and_preserves_trailing_text():
    handler = object.__new__(LanguageModelHandler)
    ctx = StreamContext(
        function_tools=[
            FunctionTool(
                type="function",
                name="dance",
                description="Dance once.",
                parameters={"type": "object", "properties": {}},
            )
        ],
        block_regex=build_block_regex(),
        enter_code=ENTER_CODE,
        end_code=END_CODE,
    )
    text = f"Watch this. {ENTER_CODE}dance(){END_CODE} Watch this. {ENTER_CODE}dance(){END_CODE}"

    chunks, tools, remaining = handler._process_printable_text(text, None, [], ctx)

    assert [chunk.text for chunk in chunks] == ["Watch this.", ""]
    assert [tool.name for tool in chunks[1].tools] == ["dance"]
    assert [tool.name for tool in tools] == ["dance"]
    assert remaining.strip() == "Watch this."
