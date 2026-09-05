"""Idle, token-budgeted maintenance for direct-audio conversation memory."""

from __future__ import annotations

import json
import math
import threading
from contextlib import nullcontext
from hashlib import sha256
from copy import deepcopy
from typing import Any

from speech_to_speech.LLM.compaction_prompt import COMPACTION_SYSTEM_PROMPT, _extract_json, _render_transcript
from speech_to_speech.pipeline.cancellable_http import CancellableAsyncSSEStream, StreamCancelled
from speech_to_speech.pipeline.model_operations import ModelOperationCoordinator


def normalize_history_compaction(value: Any) -> dict[str, Any]:
    """Validate the small, session-facing compaction policy."""
    value = value if isinstance(value, dict) else {}
    trigger = min(0.90, max(0.20, float(value.get("trigger_ratio", 0.70))))
    target = float(value.get("target_ratio", 0.50))
    return {
        "enabled": bool(value.get("enabled", True)),
        "trigger_ratio": trigger,
        "target_ratio": min(trigger - 0.05, max(0.10, target)),
        "recent_turns": min(12, max(1, int(value.get("recent_turns", 6)))),
    }


def _token_estimate(characters: int) -> int:
    """Conservative, explicitly labelled fallback until a provider tokenizer exists."""
    return max(0, math.ceil(max(0, characters) / 4))


def _endpoint_identity(endpoint: Any) -> tuple[str | None, str | None, str | None]:
    return (
        getattr(endpoint, "provider", None),
        str(getattr(endpoint, "base_url", "")).rstrip("/"),
        getattr(endpoint, "model", None),
    )


def _session_fingerprint(runtime_config: Any) -> str:
    """Compare mutable prompt/tool inputs without retaining their contents."""
    session = getattr(runtime_config, "session", None)
    material = {
        "instructions": str(getattr(session, "instructions", "") or ""),
        "tools": getattr(session, "tools", None) or [],
        "tool_choice": getattr(session, "tool_choice", None),
    }
    try:
        encoded = json.dumps(material, default=str, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        encoded = repr(material)
    return sha256(encoded.encode("utf-8")).hexdigest()


class DirectHistoryMaintenance:
    """Single-flight summary worker that never competes with a response."""

    def __init__(self, coordinator: ModelOperationCoordinator) -> None:
        self._coordinator = coordinator
        self._lock = threading.Lock()
        self._running = False

    @staticmethod
    def _telemetry(runtime_config: Any, *, status: str, failure: str | None = None, budget: dict[str, Any] | None = None) -> None:
        local = getattr(runtime_config, "local_pipeline", None)
        if not isinstance(local, dict):
            return
        previous = local.get("_history_compaction_telemetry")
        detail: dict[str, Any] = dict(previous) if isinstance(previous, dict) else {}
        detail["status"] = status
        detail["last_failure"] = failure
        if budget is not None:
            # Numbers and source labels only: diagnostics must never retain
            # prompts, transcripts, tool schemas, audio, or endpoint secrets.
            detail["budget"] = deepcopy(budget)
        local["_history_compaction_telemetry"] = detail

    @staticmethod
    def _budget(runtime_config: Any, *, history_chars: int | None = None, memory_summary: str | None = None) -> dict[str, Any]:
        """Estimate a direct request from real serialized fields and named reserves."""
        local = getattr(runtime_config, "local_pipeline", {}) or {}
        captured = local.get("_history_compaction_budget") if isinstance(local, dict) else None
        captured = captured if isinstance(captured, dict) else {}
        session = getattr(runtime_config, "session", None)
        fallback_instructions = str(getattr(session, "instructions", "") or "")
        fallback_tools = getattr(session, "tools", None) or []
        try:
            fallback_tools_chars = len(json.dumps(fallback_tools, default=str, ensure_ascii=False, separators=(",", ":")))
        except (TypeError, ValueError):
            fallback_tools_chars = 0
        if history_chars is None:
            chat = getattr(runtime_config, "chat", None)
            history_chars = len(chat.history_token_text()) if chat is not None else 0
            if memory_summary is None and chat is not None:
                memory_summary = chat.memory_summary()
        memory_block = (
            "Conversation memory (compressed; not a verbatim user quote): " + memory_summary
            if memory_summary
            else ""
        )
        captured_system = int(captured.get("system_serialized_chars", len(fallback_instructions)))
        captured_memory = int(captured.get("memory_serialized_chars", 0))
        values = {
            "history_serialized_chars": max(0, int(history_chars)),
            "system_serialized_chars": max(0, captured_system - captured_memory) + len(memory_block),
            "tools_serialized_chars": max(0, int(captured.get("tools_serialized_chars", fallback_tools_chars))),
            "response_allowance_tokens": max(1, int(captured.get("response_allowance_tokens", local.get("max_response_tokens", 384)))),
            "media_reserve_tokens_estimate": max(0, int(captured.get("media_reserve_tokens_estimate", 1024))),
            "image_reserve_tokens_estimate": max(0, int(captured.get("image_reserve_tokens_estimate", 0))),
        }
        values.update({
            "history_tokens_estimate": _token_estimate(values["history_serialized_chars"]),
            "system_tokens_estimate": _token_estimate(values["system_serialized_chars"]),
            "tools_tokens_estimate": _token_estimate(values["tools_serialized_chars"]),
            "token_source": "serialized_char_estimate",
            "media_reserve_source": "direct_audio_estimate",
        })
        values["estimated_request_tokens"] = (
            values["history_tokens_estimate"]
            + values["system_tokens_estimate"]
            + values["tools_tokens_estimate"]
            + values["response_allowance_tokens"]
            + values["media_reserve_tokens_estimate"]
            + values["image_reserve_tokens_estimate"]
        )
        return values

    def schedule(self, runtime_config: Any) -> bool:
        local = getattr(runtime_config, "local_pipeline", {}) or {}
        policy = normalize_history_compaction(local.get("history_compaction"))
        endpoint = getattr(runtime_config, "model_endpoint", None)
        context = getattr(endpoint, "context_window", None)
        if not policy["enabled"]:
            self._telemetry(runtime_config, status="disabled")
            return False
        if not isinstance(context, int) or context <= 0:
            self._telemetry(runtime_config, status="unavailable_context", failure="context_window_unknown")
            return False
        budget = self._budget(runtime_config)
        budget["context_window"] = context
        budget["trigger_tokens"] = int(context * policy["trigger_ratio"])
        budget["target_tokens"] = int(context * policy["target_ratio"])
        if budget["estimated_request_tokens"] < budget["trigger_tokens"]:
            self._telemetry(runtime_config, status="not_needed", budget=budget)
            return False
        with self._lock:
            if self._running:
                self._telemetry(runtime_config, status="deferred_busy", failure="model_operation_active", budget=budget)
                return False
            self._running = True
        self._telemetry(runtime_config, status="scheduled", budget=budget)
        threading.Thread(target=self._run, args=(runtime_config, policy), daemon=True, name="direct-history-compact").start()
        return True

    def _run(self, runtime_config: Any, policy: dict[str, Any]) -> None:
        response: CancellableAsyncSSEStream | None = None
        token = None
        cancelled = threading.Event()
        try:
            chat = runtime_config.chat
            snapshot, marker_ids, revision = chat.memory_snapshot(policy["recent_turns"])
            if not snapshot or not marker_ids:
                self._telemetry(runtime_config, status="retention_protected", failure="recent_or_unresolved_history")
                return
            endpoint = runtime_config.model_endpoint.model_copy(deep=True)
            if not isinstance(getattr(endpoint, "context_window", None), int) or endpoint.context_window <= 0:
                self._telemetry(runtime_config, status="unavailable_context", failure="context_window_unknown")
                return
            source_identity = _endpoint_identity(endpoint)
            source_session_id = str(runtime_config.local_pipeline.get("_session_id") or "")
            source_session_fingerprint = _session_fingerprint(runtime_config)
            token = self._coordinator.acquire(
                kind="history_compaction",
                session_id=str(runtime_config.local_pipeline.get("_session_id") or "") or None,
                turn_id=None,
                turn_revision=None,
                cancel_generation=None,
                drop_if_busy=True,
                stale=lambda: bool(runtime_config.local_pipeline.get("_history_closed")),
            )
            if token is None:
                self._telemetry(runtime_config, status="deferred_busy", failure="model_operation_active")
                return
            self._telemetry(runtime_config, status="running")
            transcript = _render_transcript(snapshot)
            previous_memory = chat.memory_summary()
            if previous_memory:
                transcript = "[Earlier compressed memory]\n" + previous_memory + "\n\n[New settled history]\n" + transcript
            summary_input_tokens = _token_estimate(
                len(COMPACTION_SYSTEM_PROMPT) + len("Summarize this conversation as memory. Return only JSON.\n") + len(transcript)
            )
            if summary_input_tokens + 512 > endpoint.context_window:
                self._telemetry(runtime_config, status="failed", failure="summary_request_too_large")
                return
            body = {
                "model": endpoint.model,
                "stream": True,
                "messages": [
                    {"role": "system", "content": COMPACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": "Summarize this conversation as memory. Return only JSON.\n" + transcript},
                ],
                "max_tokens": 512,
            }
            headers = {"Content-Type": "application/json"}
            if endpoint.api_key:
                headers["Authorization"] = f"Bearer {endpoint.api_key}"
            response = CancellableAsyncSSEStream(
                "POST", endpoint.base_url.rstrip("/") + "/chat/completions",
                json_body=body, headers=headers, timeout=20.0,
            )
            def close_transport() -> None:
                # Coordinator detachment after its two-second bound must not
                # turn an old cancellation into permission for a late stream
                # to splice history.
                cancelled.set()
                response.close()

            self._coordinator.bind_cancel(token, close_transport)
            raw_parts: list[str] = []
            saw_done = False
            saw_provider_error = False
            for line in response.iter_lines():
                if cancelled.is_set() or self._coordinator.cancellation_requested(token):
                    self._telemetry(runtime_config, status="cancelled", failure="speech_preempted")
                    return
                if line.startswith("data:"):
                    line = line[5:].strip()
                if not line:
                    continue
                if line == "[DONE]":
                    saw_done = True
                    continue
                try:
                    data = json.loads(line)
                except ValueError:
                    continue
                if data.get("error"):
                    saw_provider_error = True
                    continue
                choices = data.get("choices") or []
                if choices:
                    raw_parts.append(str((choices[0].get("delta") or {}).get("content") or ""))
            if saw_provider_error:
                self._telemetry(runtime_config, status="failed", failure="summary_provider_error")
                return
            if not saw_done:
                self._telemetry(runtime_config, status="failed", failure="summary_stream_truncated")
                return
            data = _extract_json("".join(raw_parts))
            user_summary = data.get("user_summary")
            assistant_summary = data.get("assistant_summary")
            if not isinstance(user_summary, str) or not isinstance(assistant_summary, str):
                self._telemetry(runtime_config, status="failed", failure="invalid_summary")
                return
            summary = "User context: " + user_summary.strip() + "\nAssistant context: " + assistant_summary.strip()
            max_summary_chars = min(6000, max(512, int(endpoint.context_window * policy["target_ratio"] * 2)))
            if summary == "User context: \nAssistant context:" or len(summary) > max_summary_chars:
                self._telemetry(runtime_config, status="failed", failure="invalid_summary")
                return
            projected_history_chars = chat.projected_direct_history_serialized_chars(
                summary, marker_ids, expected_revision=revision,
            )
            if projected_history_chars is None:
                self._telemetry(runtime_config, status="stale", failure="history_revision_or_tools_changed")
                return
            budget = self._budget(runtime_config, history_chars=projected_history_chars, memory_summary=summary)
            budget["context_window"] = endpoint.context_window
            budget["target_tokens"] = int(endpoint.context_window * policy["target_ratio"])
            if budget["estimated_request_tokens"] > budget["target_tokens"]:
                self._telemetry(runtime_config, status="failed", failure="target_not_met", budget=budget)
                return
            maintenance_lock = getattr(runtime_config, "history_maintenance_lock", None)
            with (maintenance_lock if maintenance_lock is not None else nullcontext()):
                if _endpoint_identity(getattr(runtime_config, "model_endpoint", None)) != source_identity:
                    self._telemetry(runtime_config, status="stale", failure="endpoint_changed", budget=budget)
                    return
                if (
                    str(runtime_config.local_pipeline.get("_session_id") or "") != source_session_id
                    or _session_fingerprint(runtime_config) != source_session_fingerprint
                    or runtime_config.local_pipeline.get("_history_closed")
                ):
                    self._telemetry(runtime_config, status="stale", failure="session_inputs_changed", budget=budget)
                    return
                if cancelled.is_set() or self._coordinator.cancellation_requested(token):
                    self._telemetry(runtime_config, status="cancelled", failure="speech_preempted", budget=budget)
                    return
                admission = runtime_config.local_pipeline.get("_history_maintenance_admissible")
                if callable(admission) and not admission():
                    self._telemetry(runtime_config, status="stale", failure="response_or_session_changed", budget=budget)
                    return
                if chat.apply_memory_summary(summary, marker_ids, expected_revision=revision):
                    self._telemetry(runtime_config, status="completed", budget=budget)
                else:
                    self._telemetry(runtime_config, status="stale", failure="history_revision_or_tools_changed", budget=budget)
        except StreamCancelled:
            self._telemetry(runtime_config, status="cancelled", failure="speech_preempted")
        except Exception:
            # Preserve original history and never expose provider text as telemetry.
            self._telemetry(runtime_config, status="failed", failure="maintenance_request_failed")
        finally:
            if response is not None:
                response.close()
            if token is not None:
                self._coordinator.release(token)
            with self._lock:
                self._running = False
