import json
import logging

from openai.types.responses import ResponseFunctionToolCall

from speech_to_speech.LLM.language_model import BaseLanguageModelHandler, StreamContext
from speech_to_speech.LLM.utils import remove_unspeechable


def test_legacy_llm_generation_logs_are_content_free(caplog) -> None:
    assistant_secret = "LEGACY_ASSISTANT_PRIVATE_SENTINEL"
    argument_secret = "LEGACY_TOOL_ARGUMENT_PRIVATE_SENTINEL"
    ctx = StreamContext(
        generated_text=assistant_secret,
        tools=[
            ResponseFunctionToolCall(
                type="function_call",
                name="legacy_private_tool_name",
                arguments=json.dumps({"query": argument_secret}),
                call_id="call_legacy_private_log_test",
                id="fc_legacy_private_log_test",
                status="completed",
            )
        ],
    )

    caplog.set_level(logging.DEBUG, logger="speech_to_speech.LLM.language_model")
    BaseLanguageModelHandler._log_generation_summary(ctx)

    assert assistant_secret not in caplog.text
    assert argument_secret not in caplog.text
    assert "legacy_private_tool_name" not in caplog.text
    assert (
        f"Legacy LLM generation summary assistant_chars={len(assistant_secret)} tool_calls=1"
        in caplog.text
    )


def test_remove_unspeechable_normalizes_smart_apostrophes() -> None:
    assert remove_unspeechable("I’ll reply if here’s the plan.") == "I'll reply if here's the plan."


def test_remove_unspeechable_keeps_text_and_drops_emoji() -> None:
    assert remove_unspeechable("Hello 👋 lobster 🦞") == "Hello  lobster "
