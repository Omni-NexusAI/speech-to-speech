"""Content-free diagnostics for native Chat Completions tool-call streams."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
from typing import Any

_FINISH_REASON_CATEGORIES = {
    "stop": "stop",
    "tool_calls": "tool_calls",
    "function_call": "tool_calls",
    "length": "length",
    "content_filter": "content_filter",
    "cancelled": "cancelled",
    "canceled": "cancelled",
    "error": "error",
}


def _finish_reason_category(value: Any) -> str:
    if value is None:
        return "missing"
    normalized = str(value).strip().casefold()
    if not normalized:
        return "missing"
    return _FINISH_REASON_CATEGORIES.get(normalized, "other")


def _tool_choice_category(value: Any) -> str:
    if isinstance(value, str):
        normalized = value.strip().casefold()
        return normalized if normalized in {"auto", "none", "required"} else "other"
    if isinstance(value, Mapping):
        return "required" if value.get("type") == "function" else "other"
    return "other"


def _assistant_text_length(value: Any) -> int:
    if isinstance(value, str):
        return len(value)
    if isinstance(value, list):
        return sum(
            len(part.get("text", ""))
            for part in value
            if isinstance(part, Mapping) and isinstance(part.get("text"), str)
        )
    return 0


@dataclass
class NativeToolStreamDiagnostics:
    """Collect only bounded shapes and counts; never retain model or tool content."""

    tool_choice: InitVar[Any] = "auto"
    finish_reason_category: str = "missing"
    assistant_text_length: int = 0
    native_tool_fragment_count: int = 0
    completed_call_count: int = 0
    malformed_call_category: str = "none"
    _fragment_shape_error: bool = False
    _fragment_index_error: bool = False
    _tool_choice_category: str = field(init=False, default="other", repr=False)

    def __post_init__(self, tool_choice: Any) -> None:
        self._tool_choice_category = _tool_choice_category(tool_choice)

    @property
    def native_calls_allowed(self) -> bool:
        """Whether this response-scoped choice permits any native call."""

        return (
            self._tool_choice_category != "none"
            and not self._fragment_shape_error
            and not self._fragment_index_error
        )

    def observe_malformed_stream_shape(self) -> None:
        """Record a malformed top-level/container shape without retaining it."""

        self._fragment_shape_error = True

    def observe_choice(self, choice: Any) -> None:
        """Observe one streamed ``choices[0]`` object without retaining its content."""

        if not isinstance(choice, Mapping):
            self._fragment_shape_error = True
            return

        finish_reason = choice.get("finish_reason")
        if finish_reason is not None:
            self.finish_reason_category = _finish_reason_category(finish_reason)

        delta = choice.get("delta") or {}
        if not isinstance(delta, Mapping):
            self._fragment_shape_error = True
            return

        content = delta.get("content")
        if content is None:
            content = choice.get("text")
        if content is not None and not isinstance(content, (str, list)):
            self._fragment_shape_error = True
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, Mapping) or (
                    "text" in part and not isinstance(part.get("text"), str)
                ):
                    self._fragment_shape_error = True
        self.assistant_text_length += _assistant_text_length(content)

        fragments = delta.get("tool_calls")
        if fragments is None:
            return
        if not isinstance(fragments, list):
            self._fragment_shape_error = True
            return

        self.native_tool_fragment_count += len(fragments)
        for fragment in fragments:
            if not isinstance(fragment, Mapping):
                self._fragment_shape_error = True
                continue
            index = fragment.get("index", 0)
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                self._fragment_index_error = True
            function = fragment.get("function")
            if function is not None and not isinstance(function, Mapping):
                self._fragment_shape_error = True

    def finalize(self, tool_accum: Any, *, completed_call_count: int) -> dict[str, int | str]:
        """Return the fixed content-free metric shape after native call conversion."""

        try:
            self.completed_call_count = max(0, int(completed_call_count))
        except (TypeError, ValueError):
            self.completed_call_count = 0
            self._fragment_shape_error = True

        self.malformed_call_category = self._classify_malformed(tool_accum)
        return self.metric_detail()

    def _classify_malformed(self, tool_accum: Any) -> str:
        if self._fragment_index_error:
            return "native_invalid_index"
        if self._fragment_shape_error:
            return "native_fragment_shape"
        if not isinstance(tool_accum, Mapping):
            return "native_accumulator_shape"

        named_call_count = 0
        for entry in tool_accum.values():
            if not isinstance(entry, Mapping):
                return "native_accumulator_entry_shape"
            name = entry.get("name")
            if not isinstance(name, str) or not name.strip():
                return "native_missing_name"
            named_call_count += 1

            arguments = entry.get("args")
            if not isinstance(arguments, str):
                return "native_arguments_type"
            if not arguments.strip():
                return "native_missing_arguments"
            try:
                parsed_arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return "native_invalid_arguments_json"
            if not isinstance(parsed_arguments, dict):
                return "native_arguments_not_object"

        if self._tool_choice_category == "none" and named_call_count:
            return "native_call_disallowed"
        if named_call_count != self.completed_call_count:
            return "native_completion_count_mismatch"
        if self._tool_choice_category == "required" and not self.completed_call_count:
            return "required_without_native_call"
        if self.finish_reason_category == "tool_calls" and not self.completed_call_count:
            return "tool_finish_without_completed_call"
        if self.native_tool_fragment_count and not self.completed_call_count:
            return "fragments_without_completed_call"
        return "none"

    def metric_detail(self) -> dict[str, int | str]:
        """Expose exactly the five approved content-free diagnostic fields."""

        return {
            "finish_reason_category": self.finish_reason_category,
            "assistant_text_length": self.assistant_text_length,
            "native_tool_fragment_count": self.native_tool_fragment_count,
            "completed_call_count": self.completed_call_count,
            "malformed_call_category": self.malformed_call_category,
        }
