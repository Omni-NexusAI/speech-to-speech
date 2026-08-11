from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from typing import Any, Literal, Union

from openai.types.realtime import ConversationItem
from openai.types.realtime.conversation_item import (
    RealtimeConversationItemAssistantMessage,
    RealtimeConversationItemFunctionCall,
    RealtimeConversationItemFunctionCallOutput,
    RealtimeConversationItemSystemMessage,
    RealtimeConversationItemUserMessage,
)
from openai.types.realtime.realtime_conversation_item_assistant_message import (
    Content as AssistantContent,
)
from openai.types.realtime.realtime_conversation_item_system_message import Content as SystemContent
from openai.types.realtime.realtime_conversation_item_user_message import Content as UserContent
from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams
from openai.types.responses.response_input_image_param import ResponseInputImageParam
from openai.types.responses.response_input_message_content_list_param import (
    ResponseInputMessageContentListParam,
)
from openai.types.responses.response_input_param import (
    FunctionCallOutput,
    ResponseFunctionToolCallParam,
    ResponseInputItemParam,
    ResponseInputParam,
    ResponseOutputMessageParam,
)
from openai.types.responses.response_input_param import (
    Message as ResponseMessage,
)
from openai.types.responses.response_input_text_param import ResponseInputTextParam
from openai.types.responses.response_output_text_param import ResponseOutputTextParam
from pydantic import BaseModel

from speech_to_speech.utils.utils import _generate_id

logger = logging.getLogger(__name__)


class ChatItemError(Exception):
    """Raised when a conversation item fails validation in :meth:`Chat.add_item`."""


class CompactionResult(BaseModel):
    """Output of a :data:`CompactFn` summarization run."""

    user_summary: str
    assistant_summary: str


def _ensure_id(value: str | None, prefix: str) -> str:
    if value is None:
        return _generate_id(prefix)
    if not value.startswith(f"{prefix}_"):
        raise ChatItemError(f"ID must start with '{prefix}_', got {value!r}")
    return value


def _tool_arguments_envelope(value: Any) -> dict[str, Any]:
    """Return structured arguments or a content-free invalid marker."""

    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except (json.JSONDecodeError, TypeError):
        return {"error": "invalid_tool_arguments"}
    return parsed if isinstance(parsed, dict) else {"error": "invalid_tool_arguments"}


SupportedItem = Union[
    RealtimeConversationItemSystemMessage,
    RealtimeConversationItemUserMessage,
    RealtimeConversationItemAssistantMessage,
    RealtimeConversationItemFunctionCall,
    RealtimeConversationItemFunctionCallOutput,
]


CompactFn = Callable[[ResponseInputParam], CompactionResult]


class Chat:
    """Manages conversation history with bounded size to avoid OOM issues.

    The buffer stores ``ConversationItem`` objects (user messages, assistant
    messages, function calls, function call outputs).  System messages are
    stored separately in ``init_chat_message`` and never placed in the buffer.

    History bounding is decided per ``add_item`` call via the ``compactor``
    argument:

    - ``compactor=None``: when the user-turn count exceeds ``size`` the oldest
      complete turn is evicted in place. Synchronous, lossy, no LLM involvement.
    - ``compactor=<fn>``: when ``size`` is exceeded, ``fn`` is invoked in a
      background thread to summarize older turns into a single user/assistant
      pair (with pending function calls preserved). Single-flight: while a
      compaction is running, additional triggers are silently bypassed.
    """

    def __init__(self, size: int) -> None:
        self.size = size
        self.init_chat_message: RealtimeConversationItemSystemMessage | None = None
        # ``size`` is the number of user turns to keep.  When exceeded the
        # oldest complete turn (everything up to the next user message)
        # is evicted -- or, with a compactor, summarized in the background.
        self.buffer: list[SupportedItem] = []
        self._pending_tool_calls: dict[str, RealtimeConversationItemFunctionCall] = {}
        # IDs remain reserved for the session even after their turns are
        # trimmed, preventing a later provider response from aliasing an old
        # browser tool transaction.
        self._seen_call_ids: set[str] = set()
        self._tool_call_assistant_ids: dict[str, str | None] = {}
        self._user_turn_count: int = 0
        self._trim_count: int = 0

        # All state mutations and serializations go through _lock. Public methods
        # acquire it once; internal callers that already hold it use the
        # ``_locked`` helpers, so no reentry is needed (regular Lock is safe).
        self._lock = threading.Lock()
        self._compact_in_flight: bool = False
        self._compact_thread: threading.Thread | None = None
        self._shutdown = threading.Event()
        self._gen_counter = 0

    # ── Internal mutators (caller holds _lock) ─────────────────

    def _evict_oldest_turn(self) -> None:
        """Remove items from the front until the next user message boundary."""
        if not self.buffer:
            return
        removed = [self.buffer.pop(0)]
        first = removed[0]
        if isinstance(first, RealtimeConversationItemUserMessage):
            self._user_turn_count -= 1
        while self.buffer and not isinstance(self.buffer[0], RealtimeConversationItemUserMessage):
            removed.append(self.buffer.pop(0))
        for item in removed:
            if isinstance(item, RealtimeConversationItemFunctionCall) and item.call_id:
                self._tool_call_assistant_ids.pop(item.call_id, None)
        self._trim_count += 1

    def _has_call_id_in_buffer(self, call_id: str) -> bool:
        for entry in self.buffer:
            if isinstance(entry, RealtimeConversationItemFunctionCall) and entry.call_id == call_id:
                return True
        return False

    def _mark_call_completed(
        self, call_id: str, status: Literal["completed", "incomplete", "in_progress"] | None = None
    ) -> None:
        """Set ``status`` to ``"completed"`` on the matching function_call."""
        for entry in self.buffer:
            if isinstance(entry, RealtimeConversationItemFunctionCall) and entry.call_id == call_id:
                entry.status = "completed" if status is None else status
                return

    def _assistant_has_completed_call(self, assistant_id: str) -> bool:
        """Whether one completed call/result pair follows this preamble."""

        in_response = False
        call_ids: set[str] = set()
        for item in self.buffer:
            if isinstance(item, RealtimeConversationItemAssistantMessage):
                if in_response:
                    break
                in_response = item.id == assistant_id
                continue
            if not in_response:
                continue
            if isinstance(item, RealtimeConversationItemUserMessage):
                break
            if isinstance(item, RealtimeConversationItemFunctionCall) and item.call_id:
                call_ids.add(item.call_id)
            elif isinstance(item, RealtimeConversationItemFunctionCallOutput) and item.call_id in call_ids:
                return True
        return False

    def append_tool_output(self, call_id: str, output_item: RealtimeConversationItemFunctionCallOutput) -> None:
        """Append a ``function_call_output``, re-injecting its ``function_call`` if evicted.

        Also marks the paired ``function_call`` as ``"completed"`` if its
        status was ``None``.

        Raises :class:`ChatItemError` if *call_id* is unknown.
        """
        with self._lock:
            self._append_tool_output_locked(call_id, output_item)

    def discard_pending_tool_calls(self, call_ids: set[str]) -> None:
        """Drop unresolved calls while retaining the accepted user/completed pairs."""
        if not call_ids:
            return
        with self._lock:
            pending_to_discard = call_ids & set(self._pending_tool_calls)
            candidate_assistant_ids = {
                self._tool_call_assistant_ids.get(call_id)
                for call_id in pending_to_discard
            }
            self._pending_tool_calls = {
                call_id: call
                for call_id, call in self._pending_tool_calls.items()
                if call_id not in pending_to_discard
            }
            self.buffer = [
                item
                for item in self.buffer
                if not (
                    isinstance(item, RealtimeConversationItemFunctionCall)
                    and item.call_id in pending_to_discard
                )
            ]
            for call_id in pending_to_discard:
                self._tool_call_assistant_ids.pop(call_id, None)
            for assistant_id in candidate_assistant_ids - {None}:
                remaining_associated = {
                    call_id
                    for call_id, mapped_assistant_id in self._tool_call_assistant_ids.items()
                    if mapped_assistant_id == assistant_id
                }
                has_pending = any(call_id in self._pending_tool_calls for call_id in remaining_associated)
                has_completed = any(
                    isinstance(item, RealtimeConversationItemFunctionCall)
                    and item.call_id in remaining_associated
                    for item in self.buffer
                ) or self._assistant_has_completed_call(assistant_id)
                if not has_pending and not has_completed:
                    self.buffer = [
                        item
                        for item in self.buffer
                        if not (isinstance(item, RealtimeConversationItemAssistantMessage) and item.id == assistant_id)
                    ]

    def _append_tool_output_locked(self, call_id: str, output_item: RealtimeConversationItemFunctionCallOutput) -> None:
        """Body of :meth:`append_tool_output`. Caller must hold ``_lock``."""
        if self._has_call_id_in_buffer(call_id):
            self._pending_tool_calls.pop(call_id, None)
            self._tool_call_assistant_ids.pop(call_id, None)
            self._mark_call_completed(call_id, output_item.status)
            self.buffer.append(output_item)
            return

        if call_id in self._pending_tool_calls:
            logger.info("Re-injecting evicted function_call for call_id=%s", call_id)
            fc = self._pending_tool_calls.pop(call_id)
            self._tool_call_assistant_ids.pop(call_id, None)
            fc.status = "completed" if output_item.status is None else output_item.status
            self.buffer.append(fc)
            self.buffer.append(output_item)
            return

        raise ChatItemError(f"No function_call with call_id '{call_id}' found in conversation history.")

    def init_chat(self, message: RealtimeConversationItemSystemMessage) -> None:
        with self._lock:
            self.init_chat_message = message

    def add_item(self, item: SupportedItem) -> SupportedItem:
        """Validate and route a conversation item into the chat buffer.

        Does not enforce the soft size limit — call :meth:`trim_if_needed`
        explicitly after each successful generation to evict or compact old
        turns. A hard upper bound at ``2 * size`` is enforced inline as a
        runaway-client safety net: if the user-turn count exceeds it, the
        oldest complete turn is evicted (lossy, no compaction).

        Raises :class:`ChatItemError` if the item fails validation.
        """
        with self._lock:
            if isinstance(item, RealtimeConversationItemSystemMessage):
                item.id = _ensure_id(item.id, "sys")
                self.init_chat_message = item
                logger.debug("Set system message via conversation item")

            elif isinstance(item, RealtimeConversationItemUserMessage):
                item.id = _ensure_id(item.id, "msg")
                item.content = [
                    p
                    for p in item.content
                    if (p.type == "input_text" and p.text)
                    or (p.type == "input_image" and p.image_url)
                    or (p.type == "input_audio" and p.audio)
                ]
                if not item.content:
                    raise ChatItemError(
                        "Message has no supported content. Supported modalities: input_text, input_image, input_audio."
                    )
                self.buffer.append(item)
                self._user_turn_count += 1
                logger.debug("Added user message to chat (%d parts)", len(item.content))

            elif isinstance(item, RealtimeConversationItemAssistantMessage):
                item.id = _ensure_id(item.id, "msg")
                item.content = [p for p in item.content if p.type == "output_text" and p.text]
                if not item.content:
                    return item
                self.buffer.append(item)
                logger.debug("Added assistant message to chat (%d parts)", len(item.content))

            elif isinstance(item, RealtimeConversationItemFunctionCall):
                item.id = _ensure_id(item.id, "fc")
                item.call_id = _ensure_id(item.call_id, "call")
                if item.call_id in self._seen_call_ids:
                    raise ChatItemError(f"Duplicate function call_id {item.call_id!r} in conversation history.")
                self._seen_call_ids.add(item.call_id)
                self._pending_tool_calls[item.call_id] = item
                logger.debug("Added function_call to chat (call_id=%s)", item.call_id)

            elif isinstance(item, RealtimeConversationItemFunctionCallOutput):
                item.id = _ensure_id(item.id, "fco")
                self._append_tool_output_locked(item.call_id, item)
                logger.debug("Added function_call_output to chat (call_id=%s)", item.call_id)

            else:
                raise ChatItemError(f"Unsupported item type: {getattr(item, 'type', None)}")

            if self.size > 0 and self._user_turn_count > 2 * self.size:
                logger.warning(
                    "Chat buffer exceeded hard cap (%d > 2 * size=%d); evicting oldest turn",
                    self._user_turn_count,
                    self.size,
                )
                while self._user_turn_count > 2 * self.size:
                    self._evict_oldest_turn()

            return item

    def commit_assistant_response(
        self,
        user_item_id: str,
        assistant_text: str,
        function_calls: list[RealtimeConversationItemFunctionCall],
    ) -> bool:
        """Atomically append a completed response for an accepted user turn.

        The user item is persisted before model generation. Assistant prose and
        tool calls are validated here before any response-side mutation, so a
        cancelled or invalid response cannot leave partial assistant history.
        Tool outputs are appended later through :meth:`append_tool_output`.
        """

        with self._lock:
            if not any(
                isinstance(item, RealtimeConversationItemUserMessage) and item.id == user_item_id
                for item in self.buffer
            ):
                raise ChatItemError(f"Accepted user item {user_item_id!r} is not present in conversation history.")

            assistant: RealtimeConversationItemAssistantMessage | None = None
            if assistant_text:
                assistant = make_assistant_message(assistant_text)
                assistant.id = _ensure_id(assistant.id, "msg")

            prepared_calls: list[RealtimeConversationItemFunctionCall] = []
            response_call_ids: set[str] = set()
            for function_call in function_calls:
                function_call.id = _ensure_id(function_call.id, "fc")
                function_call.call_id = _ensure_id(function_call.call_id, "call")
                if function_call.call_id in self._seen_call_ids or function_call.call_id in response_call_ids:
                    raise ChatItemError(
                        f"Duplicate function call_id {function_call.call_id!r} in conversation history."
                    )
                response_call_ids.add(function_call.call_id)
                prepared_calls.append(function_call)

            if assistant is not None:
                self.buffer.append(assistant)
            for function_call in prepared_calls:
                assert function_call.call_id is not None
                self._seen_call_ids.add(function_call.call_id)
                self._pending_tool_calls[function_call.call_id] = function_call
                self._tool_call_assistant_ids[function_call.call_id] = assistant.id if assistant is not None else None
                self.buffer.append(function_call)
            return assistant is not None or bool(prepared_calls)

    def canonical_call_id(self, value: str | None, additional_used: set[str] | None = None) -> str:
        """Select a Realtime-shaped call ID unique across this session."""

        with self._lock:
            base = value if value and value.startswith("call_") else f"call_{value}" if value else _generate_id("call")
            used = self._seen_call_ids | set(additional_used or ())
            if base not in used:
                return base
            suffix = 1
            candidate = f"{base}_{suffix}"
            while candidate in used:
                suffix += 1
                candidate = f"{base}_{suffix}"
            return candidate

    def trim_if_needed(self, compactor: CompactFn | None = None) -> None:
        """Enforce the size limit after a generation completes. Fires when
        ``user_turn_count > size``.

        - ``compactor=None``: synchronous eviction of the oldest complete turn.
        - ``compactor=<fn>``: launch a background compaction (single-flight).

        Call once after each successful generation, not inside :meth:`add_item`.
        """
        with self._lock:
            if self._user_turn_count <= self.size:
                return
            if compactor is not None:
                self._maybe_trigger_compaction(compactor)
            else:
                while self._user_turn_count > self.size:
                    self._evict_oldest_turn()

    def replace_user_message_text(self, item_id: str, text: str) -> bool:
        """Replace the text content of an existing user message.

        Used by speculative turn revisions: the conversation turn remains the
        same, but the STT transcript is superseded by a transcription of a
        longer raw-audio buffer.
        """

        with self._lock:
            for item in self.buffer:
                if not isinstance(item, RealtimeConversationItemUserMessage) or item.id != item_id:
                    continue
                item.content = [UserContent(type="input_text", text=text)]
                logger.debug("Replaced speculative user message %s", item_id)
                return True
        return False

    def replace_user_message_audio(self, item_id: str, audio: str) -> bool:
        """Replace one accepted user's content with a cumulative mono-WAV anchor."""

        with self._lock:
            for item in self.buffer:
                if not isinstance(item, RealtimeConversationItemUserMessage) or item.id != item_id:
                    continue
                item.content = [UserContent(type="input_audio", audio=audio)]
                logger.debug("Replaced accepted user audio %s", item_id)
                return True
        return False

    def remove_user_message(self, item_id: str) -> bool:
        """Remove an existing user message from the bounded chat buffer."""

        with self._lock:
            for index, item in enumerate(self.buffer):
                if not isinstance(item, RealtimeConversationItemUserMessage) or item.id != item_id:
                    continue
                del self.buffer[index]
                self._user_turn_count -= 1
                logger.debug("Removed speculative user message %s", item_id)
                return True
        return False

    def to_responses_api_chat(self, items: list[SupportedItem] | None = None) -> ResponseInputParam:
        """Serialize the chat (system prompt + buffer) for the OpenAI Responses API.

        If *items* is provided, serialize that slice instead of the live buffer
        (used by the compaction snapshot).
        """
        with self._lock:
            return self._to_responses_api_chat_locked(items if items is not None else self.buffer)

    def _to_responses_api_chat_locked(self, items: list[SupportedItem]) -> ResponseInputParam:
        """Body of :meth:`to_responses_api_chat`. Caller must hold ``_lock``."""
        buffer_items = list(items)
        result: list[ResponseInputItemParam] = []
        if self.init_chat_message:
            result.append(
                ResponseMessage(
                    content=[
                        ResponseInputTextParam(text=p.text or "A helpful AI assistant.", type="input_text")
                        for p in self.init_chat_message.content
                    ],
                    role="system",
                    type="message",
                )
            )
        for item in buffer_items:
            assert item.id is not None and item.id != "", f"item.id is {item.id}"
            if isinstance(item, RealtimeConversationItemUserMessage):
                content: ResponseInputMessageContentListParam = []
                for user_part in item.content:
                    if user_part.type == "input_text" and user_part.text is not None:
                        content.append(ResponseInputTextParam(text=user_part.text or "", type="input_text"))
                    elif user_part.type == "input_image" and user_part.image_url is not None:
                        img = ResponseInputImageParam(type="input_image", detail=user_part.detail or "auto")
                        if user_part.image_url is not None:
                            img["image_url"] = user_part.image_url
                        content.append(img)
                    elif user_part.type == "input_audio" and user_part.audio:
                        # The installed Responses API message-content contract
                        # does not include input_audio. Never omit the semantic
                        # user turn and leave its assistant/tool items orphaned.
                        raise ChatItemError(
                            "Responses API serialization does not support retained input_audio history; "
                            "use a provider with historical audio support or wait for a validated transcript."
                        )
                if content:
                    result.append(ResponseMessage(content=content, role="user", type="message"))
            elif isinstance(item, RealtimeConversationItemAssistantMessage):
                assistant_content: list[ResponseOutputTextParam] = []
                for assistant_part in item.content:
                    if assistant_part.type == "output_text" and assistant_part.text is not None:
                        assistant_content.append(
                            ResponseOutputTextParam(text=assistant_part.text, type="output_text", annotations=[])
                        )
                if assistant_content:
                    result.append(
                        ResponseOutputMessageParam(
                            id=item.id,
                            content=assistant_content,
                            role="assistant",
                            status=item.status or "completed",
                            type="message",
                        )
                    )
            elif isinstance(item, RealtimeConversationItemFunctionCall) and item.call_id is not None:
                assert item.call_id is not None and item.call_id != ""
                function_call = ResponseFunctionToolCallParam(
                    arguments=item.arguments,
                    call_id=item.call_id,
                    name=item.name,
                    type="function_call",
                    id=item.id,
                )
                if item.id is not None:
                    function_call["id"] = item.id
                if item.status is not None:
                    function_call["status"] = item.status
                result.append(function_call)
            elif isinstance(item, RealtimeConversationItemFunctionCallOutput):
                function_call_output = FunctionCallOutput(
                    call_id=item.call_id,
                    output=item.output,
                    type="function_call_output",
                )
                if item.id is not None:
                    function_call_output["id"] = item.id
                if item.status is not None:
                    function_call_output["status"] = item.status
                result.append(function_call_output)
        return result

    def to_transformers_chat(self) -> list[dict[str, Any]]:
        """Serialize the full chat for HuggingFace transformers ``apply_chat_template``.

        User messages with only text produce a plain string ``content`` value.
        User messages containing images or session-only WAV history keep
        ``content`` as a multimodal list.
        """
        with self._lock:
            messages: list[TransformersChatMessage] = []
            if self.init_chat_message:
                text = " ".join(p.text for p in self.init_chat_message.content if p.text)
                messages.append(TransformersSystemMessage(content=text))
            for item in self.buffer:
                if isinstance(item, RealtimeConversationItemUserMessage):
                    has_multimodal = any(p.type in {"input_image", "input_audio"} for p in item.content)
                    if has_multimodal:
                        content: list[dict[str, Any]] = []
                        for part in item.content:
                            if part.type == "input_text" and part.text:
                                content.append({"type": "input_text", "text": part.text})
                            elif part.type == "input_image" and part.image_url:
                                content.append(part.model_dump(exclude_none=True))
                            elif part.type == "input_audio" and part.audio:
                                content.append(
                                    {
                                        "type": "input_audio",
                                        "input_audio": {"data": part.audio, "format": "wav"},
                                    }
                                )
                        messages.append(TransformersUserMessage(content=content))
                    else:
                        text = " ".join(p.text for p in item.content if p.type == "input_text" and p.text)
                        messages.append(TransformersUserMessage(content=text))
                elif isinstance(item, RealtimeConversationItemAssistantMessage):
                    text = " ".join(p.text for p in item.content if p.text)
                    messages.append(TransformersAssistantMessage(content=text))
                elif isinstance(item, RealtimeConversationItemFunctionCall):
                    assert item.call_id is not None and item.call_id != ""
                    args = _tool_arguments_envelope(item.arguments)
                    messages.append(
                        TransformersFunctionCallMessage(
                            tool_calls=[
                                TransformersToolCall(
                                    id=item.call_id,
                                    function=TransformersToolCallFunction(name=item.name, arguments=args),
                                )
                            ]
                        )
                    )
                elif isinstance(item, RealtimeConversationItemFunctionCallOutput):
                    name = ""
                    for prev in reversed(messages):
                        if isinstance(prev, TransformersFunctionCallMessage):
                            for tc in prev.tool_calls:
                                if tc.id == item.call_id:
                                    name = tc.function.name
                                    break
                            if name:
                                break
                    messages.append(
                        TransformersToolMessage(
                            tool_call_id=item.call_id,
                            name=name,
                            content=item.output,
                        )
                    )
            return [m.model_dump() for m in messages]

    def copy(self) -> Chat:
        """Return a shallow snapshot safe for concurrent read access."""
        with self._lock:
            clone = Chat(self.size)
            clone.init_chat_message = self.init_chat_message
            clone.buffer = list(self.buffer)
            clone._pending_tool_calls = dict(self._pending_tool_calls)
            clone._seen_call_ids = set(self._seen_call_ids)
            clone._tool_call_assistant_ids = dict(self._tool_call_assistant_ids)
            clone._user_turn_count = self._user_turn_count
            clone._trim_count = self._trim_count
            return clone

    def history_token_text(self) -> str:
        """Return retained history as compact text for the server tokenizer.

        System instructions and raw image data are intentionally excluded: the
        diagnostics counter represents conversation history and therefore starts
        at zero for a new session.
        """
        with self._lock:
            entries: list[dict[str, Any]] = []
            for item in self.buffer:
                if isinstance(item, RealtimeConversationItemUserMessage):
                    content = [
                        part.text
                        if part.type == "input_text"
                        else "<image>"
                        if part.type == "input_image"
                        else "<audio:wav>"
                        for part in item.content
                        if (part.type == "input_text" and part.text)
                        or part.type == "input_image"
                        or (part.type == "input_audio" and part.audio)
                    ]
                    entries.append({"role": "user", "content": content})
                elif isinstance(item, RealtimeConversationItemAssistantMessage):
                    entries.append(
                        {"role": "assistant", "content": [part.text for part in item.content if part.text]}
                    )
                elif isinstance(item, RealtimeConversationItemFunctionCall):
                    entries.append(
                        {
                            "role": "tool_call",
                            "name": item.name,
                            "arguments": _tool_arguments_envelope(item.arguments),
                            "call_id": item.call_id,
                        }
                    )
                elif isinstance(item, RealtimeConversationItemFunctionCallOutput):
                    entries.append(
                        {
                            "role": "tool_output",
                            "output": f"<tool_output:chars={len(item.output)}>",
                            "call_id": item.call_id,
                        }
                    )
            buffered_calls = {
                item.call_id for item in self.buffer if isinstance(item, RealtimeConversationItemFunctionCall)
            }
            for call_id, item in self._pending_tool_calls.items():
                if call_id not in buffered_calls:
                    entries.append(
                        {
                            "role": "tool_call",
                            "name": item.name,
                            "arguments": _tool_arguments_envelope(item.arguments),
                            "call_id": call_id,
                        }
                    )
            return json.dumps(entries, ensure_ascii=False, separators=(",", ":")) if entries else ""

    def stats(self) -> dict[str, int]:
        """Return content-free counters for local diagnostics."""
        with self._lock:
            return {
                "turns": self._user_turn_count,
                "items": len(self.buffer),
                "pending_tool_calls": len(self._pending_tool_calls),
                "limit": self.size,
                "trim_count": self._trim_count,
            }

    def reset(self) -> None:
        """Clear all conversation state. Cancels any in-flight compaction splice."""
        with self._lock:
            self._gen_counter += 1
            self._compact_in_flight = False
            self.buffer = []
            self.init_chat_message = None
            self._pending_tool_calls = {}
            self._seen_call_ids = set()
            self._tool_call_assistant_ids = {}
            self._user_turn_count = 0
            self._trim_count = 0

    def close(self) -> None:
        """Permanently shut down the chat. In-flight compaction splice is suppressed.

        The compaction worker (a daemon thread) is not joined: it may be blocked
        in an LLM call. Process exit reaps it.
        """
        self._shutdown.set()
        with self._lock:
            self._gen_counter += 1
            self._compact_in_flight = False

    def image_message_ids(self) -> set[str]:
        """IDs of user messages currently carrying ``input_image`` content."""
        with self._lock:
            return {
                item.id
                for item in self.buffer
                if isinstance(item, RealtimeConversationItemUserMessage)
                and item.id is not None
                and any(p.type == "input_image" for p in item.content)
            }

    def strip_images(self, only_ids: set[str] | None = None) -> None:
        """Remove image content parts from user messages in the buffer.

        Called after appending the assistant response so images don't persist
        across turns. With *only_ids*, strip only those message IDs — the images
        the just-completed response actually consumed (captured before the
        request was sent). This leaves intact an image a fast client injected
        mid-generation for the *next* turn, which the current response never saw.
        Without *only_ids*, every image is stripped.

        An image-only message is a transient request attachment rather than a
        durable empty semantic turn. Remove that item after its image is retired
        and keep the user-turn counter aligned with the actual buffer.
        """
        with self._lock:
            retained: list[SupportedItem] = []
            for item in self.buffer:
                if isinstance(item, RealtimeConversationItemUserMessage):
                    if only_ids is not None and item.id not in only_ids:
                        retained.append(item)
                        continue
                    item.content = [p for p in item.content if p.type != "input_image"]
                    if not item.content:
                        self._user_turn_count = max(0, self._user_turn_count - 1)
                        continue
                retained.append(item)
            self.buffer = retained

    # ── Compaction internals ──────────────────────────────────

    def _snapshot_for_compaction(
        self,
    ) -> tuple[ResponseInputParam, set[str], int]:
        """Compute the snapshot of items eligible for compaction.

        Caller must hold ``_lock``. Returns
        ``(serialized_snapshot, marker_ids, n_turns)``. ``marker_ids``
        identifies the buffer items that may be removed when the splice runs.
        Always leaves the most recent user turn untouched (it may be in-flight).
        Returns an empty result if there are fewer than 2 compactable turns.
        """
        n_turns = max(0, self._user_turn_count - 1)
        if n_turns < 2:
            return [], set(), n_turns

        # Slice up to (but not including) the (n_turns + 1)-th user message.
        user_seen = 0
        end_idx = len(self.buffer)
        for i, entry in enumerate(self.buffer):
            if isinstance(entry, RealtimeConversationItemUserMessage):
                user_seen += 1
                if user_seen == n_turns + 1:
                    end_idx = i
                    break

        items_to_compact = self.buffer[:end_idx]
        marker_ids = {entry.id for entry in items_to_compact if entry.id is not None}
        snapshot = self._to_responses_api_chat_locked(items=items_to_compact)
        # Strip image parts so the summarizer doesn't have to handle them.
        for raw in snapshot:
            if not isinstance(raw, dict) or raw.get("role") != "user":
                continue
            msg: dict[str, Any] = raw  # type: ignore[assignment]
            content = msg.get("content")
            if isinstance(content, list):
                msg["content"] = [c for c in content if not (isinstance(c, dict) and c.get("type") == "input_image")]
        return snapshot, marker_ids, n_turns

    def _maybe_trigger_compaction(self, compactor: CompactFn) -> None:
        """Start a background compaction worker. Bypass silently if one is running.

        Caller must hold ``_lock``.
        """
        if self._shutdown.is_set() or self._compact_in_flight:
            return
        snapshot, marker_ids, n_turns = self._snapshot_for_compaction()
        if n_turns < 2 or not marker_ids:
            return
        gen = self._gen_counter
        self._compact_in_flight = True
        thread = threading.Thread(
            target=self._compact_worker,
            args=(compactor, snapshot, marker_ids, gen),
            daemon=True,
            name="chat-compact",
        )
        self._compact_thread = thread
        logger.info(
            "Chat compaction triggered: compacting %d turn(s) (%d item(s)), buffer size=%d",
            n_turns,
            len(marker_ids),
            len(self.buffer),
        )
        thread.start()

    def _compact_worker(
        self,
        compactor: CompactFn,
        snapshot: ResponseInputParam,
        marker_ids: set[str],
        gen: int,
    ) -> None:
        """Worker thread entry point."""
        try:
            if self._shutdown.is_set() or self._gen_counter != gen:
                return
            try:
                result = compactor(snapshot)
            except Exception:
                logger.exception("Chat compaction failed; chat unchanged")
                return
            if not isinstance(result, CompactionResult):
                logger.error("Compactor must return a CompactionResult, got %r", type(result).__name__)
                return
            if self._shutdown.is_set() or self._gen_counter != gen:
                return
            self._apply_compaction(result, marker_ids, gen)
        finally:
            # Don't clobber the flag if reset/close has advanced the gen.
            with self._lock:
                if self._gen_counter == gen:
                    self._compact_in_flight = False

    def _apply_compaction(
        self,
        result: CompactionResult,
        marker_ids: set[str],
        gen: int,
    ) -> None:
        """Splice the summary in front of items not consumed by compaction.

        FC/FCO pairing is left entirely to :meth:`add_item` / :meth:`append_tool_output`.
        Compaction only drops items; it never inserts an FC into the buffer.
        Pending FCs (no FCO yet) stay in ``_pending_tool_calls`` and will be
        appended adjacent to their FCO when it arrives.
        """
        with self._lock:
            if self._shutdown.is_set() or self._gen_counter != gen:
                return
            # Keep FC if its FCO is outside the compacted range -- otherwise
            # the FCO in `remaining` would be orphaned.
            fco_call_ids_in_range = {
                x.call_id
                for x in self.buffer
                if isinstance(x, RealtimeConversationItemFunctionCallOutput) and x.id in marker_ids
            }
            fc_ids_to_keep = {
                x.id
                for x in self.buffer
                if x.id in marker_ids
                and isinstance(x, RealtimeConversationItemFunctionCall)
                and x.call_id not in fco_call_ids_in_range
            }
            drop_ids = marker_ids - fc_ids_to_keep
            remaining = [x for x in self.buffer if x.id not in drop_ids]

            user_msg = make_user_message(result.user_summary)
            user_msg.id = _generate_id("msg")
            asst_msg = make_assistant_message(result.assistant_summary)
            asst_msg.id = _generate_id("msg")

            self.buffer = [user_msg, asst_msg, *remaining]
            self._user_turn_count = sum(1 for x in self.buffer if isinstance(x, RealtimeConversationItemUserMessage))
            logger.info(
                "Chat compaction applied: buffer now %d item(s), %d user turn(s)",
                len(self.buffer),
                self._user_turn_count,
            )


# ---------------------------------------------------------------------------
# Transformers chat message models
# ---------------------------------------------------------------------------


class TransformersToolCallFunction(BaseModel):
    name: str
    arguments: dict[str, Any]


class TransformersToolCall(BaseModel):
    type: Literal["function"] = "function"
    id: str
    function: TransformersToolCallFunction


class TransformersSystemMessage(BaseModel):
    role: Literal["system"] = "system"
    content: str


class TransformersUserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: str | list[dict[str, Any]]


class TransformersAssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class TransformersFunctionCallMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    tool_calls: list[TransformersToolCall]


class TransformersToolMessage(BaseModel):
    role: Literal["tool"] = "tool"
    tool_call_id: str
    name: str
    content: str


TransformersChatMessage = Union[
    TransformersSystemMessage,
    TransformersUserMessage,
    TransformersAssistantMessage,
    TransformersFunctionCallMessage,
    TransformersToolMessage,
]


# ---------------------------------------------------------------------------
# Factory helpers -- hide verbose constructors behind simple calls
# ---------------------------------------------------------------------------


def make_user_message(text: str) -> RealtimeConversationItemUserMessage:
    return RealtimeConversationItemUserMessage(
        type="message",
        role="user",
        content=[UserContent(type="input_text", text=text)],
    )


def make_user_audio_message(wav_base64: str) -> RealtimeConversationItemUserMessage:
    """Create a session-only accepted user turn backed by the original mono WAV."""

    return RealtimeConversationItemUserMessage(
        type="message",
        role="user",
        content=[UserContent(type="input_audio", audio=wav_base64)],
    )


def make_assistant_message(text: str) -> RealtimeConversationItemAssistantMessage:
    return RealtimeConversationItemAssistantMessage(
        type="message",
        role="assistant",
        content=[AssistantContent(type="output_text", text=text)],
    )


def make_system_message(text: str) -> RealtimeConversationItemSystemMessage:
    return RealtimeConversationItemSystemMessage(
        type="message",
        role="system",
        content=[SystemContent(type="input_text", text=text)],
    )


def add_supported_item(chat: Chat, item: ConversationItem) -> None:
    """Narrow a protocol ``ConversationItem`` to a :data:`SupportedItem` and add it to *chat*.

    Raises :class:`ChatItemError` on validation failure or unsupported type. Shared
    by the conversation handler (in-band item injection) and the language-model
    handlers (seeding an out-of-band response's throwaway chat from ``response.input``).
    """
    # call_id on function_call items must be client-supplied: it is referenced later by
    # function_call_output items, so we cannot silently generate one here.
    if isinstance(item, RealtimeConversationItemFunctionCall) and (
        item.call_id is None or not item.call_id.startswith("call_")
    ):
        raise ChatItemError("function_call item is missing a call_id. The call_id should start with 'call_'.")

    if isinstance(
        item,
        (
            RealtimeConversationItemSystemMessage,
            RealtimeConversationItemUserMessage,
            RealtimeConversationItemAssistantMessage,
            RealtimeConversationItemFunctionCall,
            RealtimeConversationItemFunctionCallOutput,
        ),
    ):
        chat.add_item(item)
        return

    raise ChatItemError(f"Unsupported item type: {getattr(item, 'type', None)}")


def build_active_chat(original_chat: Chat, response: RealtimeResponseCreateParams | None) -> Chat:
    """Build the chat an *out-of-band* response generates against (caller ensures out-of-band).

    Mirrors the OpenAI realtime semantics for ``input``:

    - ``input is None`` -> a read-only **copy of the default conversation** (the
      out-of-band response reads history but never commits back).
    - ``input == []`` -> a **fresh, empty chat** (context cleared; only the
      system prompt, added later by the handler, will be present).
    - ``input == [...]`` -> a **fresh chat seeded** with those items.

    Raises :class:`ChatItemError` if an ``input`` item fails validation.
    """
    if response is not None and response.input is not None:
        fresh = Chat(original_chat.size)
        for item in response.input:
            add_supported_item(fresh, item)
        return fresh
    return original_chat.copy()
