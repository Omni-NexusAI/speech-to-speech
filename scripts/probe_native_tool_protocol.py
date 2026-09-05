"""Content-free native Chat Completions tool-protocol release gate.

The probe resolves the exact model selected by the managed UI, sends only
fixed source fixtures, and emits only bounded protocol shapes, counts, timings,
and pass/fail state. Prompts, assistant output, tool names/arguments/results,
endpoint/model identity, and credentials remain process-local and are never
written or printed. Printed call-like prose is observed as assistant text only;
it is never parsed, recovered, or executed.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import httpx

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from speech_to_speech.LLM.native_tool_diagnostics import NativeToolStreamDiagnostics  # noqa: E402
from speech_to_speech.LLM.voice_prompt import build_voice_system_prompt  # noqa: E402

_MANAGED_SETTINGS_URL = "http://127.0.0.1:7862/api/ui-settings"
_MANAGED_LOCAL_PIPELINE_URL = "http://127.0.0.1:7862/api/local-pipeline"
_LOCAL_MODEL_BASE_URL = "http://127.0.0.1:8818/v1"
_CREDENTIAL_ENV = "S2S_REMOTE_MODEL_API_KEY"
_DEFAULT_TIMEOUT_S = 45.0

_FIXED_SESSION_PROMPT = "Follow the fixed protocol-check request exactly and keep ordinary answers brief."
_FIXED_TOOL_NAME = "protocol_probe_lookup"
_FIXED_ARGUMENT_KEY = "record_id"
_FIXED_ARGUMENT_VALUE = "fixed-protocol-record"
_FIXED_TOOL_RESULT = '{"status":"available"}'
_FIXED_REQUIRED_REQUEST = "Retrieve the fixed protocol record with the available function. Call it silently."
_FIXED_AUTO_TOOL_REQUEST = "Use the available function to retrieve the fixed protocol record. Call it silently."
_FIXED_ORDINARY_REQUEST = "Reply with one brief ordinary sentence without using a function."
_FIXED_PROSE_SHAPED_RESPONSE = 'protocol_probe_lookup({"record_id":"fixed-protocol-record"})'
_FIXED_MALFORMED_ARGUMENTS = '{"record_id":'
_FIXED_CALL_ID = "call_fixed_protocol_probe"

_FIXED_TOOL = {
    "type": "function",
    "function": {
        "name": _FIXED_TOOL_NAME,
        "description": "Return the fixed, non-sensitive protocol-check record.",
        "parameters": {
            "type": "object",
            "properties": {
                _FIXED_ARGUMENT_KEY: {
                    "type": "string",
                    "enum": [_FIXED_ARGUMENT_VALUE],
                }
            },
            "required": [_FIXED_ARGUMENT_KEY],
            "additionalProperties": False,
        },
    },
}


class ProbeError(RuntimeError):
    """Base error whose message is never included in public output."""


class SettingsUnavailable(ProbeError):
    pass


class UntrustedManagedTarget(ProbeError):
    pass


class CredentialUnavailable(ProbeError):
    pass


class ModelRequestFailed(ProbeError):
    pass


class InvalidModelResponse(ProbeError):
    pass


class ProbeProgressError(ProbeError):
    def __init__(self, *, attempted: int, completed: int):
        super().__init__()
        self.attempted = max(0, int(attempted))
        self.completed = max(0, int(completed))


@dataclass(frozen=True, repr=False)
class _ManagedTarget:
    endpoint: str = field(repr=False)
    model: str = field(repr=False)
    requires_credential: bool = field(repr=False)


@dataclass(frozen=True, repr=False)
class _ProtocolObservation:
    detail: Mapping[str, int | str] = field(repr=False)
    expected_call_valid: bool = field(repr=False, default=False)


def _validate_managed_url(value: str, canonical: str, path: str) -> None:
    try:
        parsed = urlsplit(value)
    except (TypeError, ValueError) as exc:
        raise UntrustedManagedTarget from exc
    if (
        value != canonical
        or parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.port != 7862
        or parsed.path != path
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise UntrustedManagedTarget


def _managed_json(client: httpx.Client, url: str, path: str) -> Mapping[str, Any]:
    _validate_managed_url(url, url, path)
    try:
        response = client.get(url)
        response.raise_for_status()
        if response.history or str(response.url) != url:
            raise UntrustedManagedTarget
        payload = response.json()
    except UntrustedManagedTarget:
        raise
    except (httpx.HTTPError, TypeError, ValueError) as exc:
        raise SettingsUnavailable from exc
    if not isinstance(payload, Mapping):
        raise SettingsUnavailable
    return payload


def _chat_completions_target(base_url: str) -> str:
    target = base_url.rstrip("/")
    if not target.endswith("/chat/completions"):
        target = f"{target}/chat/completions"
    return target


def _resolve_managed_target(client: httpx.Client) -> _ManagedTarget:
    """Resolve the managed local or remote model without reading a credential."""

    settings_payload = _managed_json(client, _MANAGED_SETTINGS_URL, "/api/ui-settings")
    settings = settings_payload.get("settings")
    if not isinstance(settings, Mapping):
        raise SettingsUnavailable
    provider = settings.get("modelProvider")

    if provider == "local":
        local_payload = _managed_json(client, _MANAGED_LOCAL_PIPELINE_URL, "/api/local-pipeline")
        gemma = local_payload.get("gemma")
        if not isinstance(gemma, Mapping):
            raise SettingsUnavailable
        base_url = gemma.get("baseUrl")
        model = gemma.get("model")
        if not isinstance(base_url, str) or not isinstance(model, str) or not model.strip():
            raise SettingsUnavailable
        try:
            parsed = urlsplit(base_url)
        except ValueError as exc:
            raise UntrustedManagedTarget from exc
        if (
            base_url.rstrip("/") != _LOCAL_MODEL_BASE_URL
            or parsed.scheme != "http"
            or parsed.hostname != "127.0.0.1"
            or parsed.port != 8818
            or parsed.path.rstrip("/") != "/v1"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise UntrustedManagedTarget
        return _ManagedTarget(
            endpoint=_chat_completions_target(base_url),
            model=model.strip(),
            requires_credential=False,
        )

    if provider != "remote":
        raise SettingsUnavailable
    base_url = settings.get("modelUrl")
    model = settings.get("modelName")
    if not isinstance(base_url, str) or not base_url.strip() or not isinstance(model, str) or not model.strip():
        raise SettingsUnavailable
    try:
        parsed = urlsplit(base_url)
    except ValueError as exc:
        raise UntrustedManagedTarget from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise UntrustedManagedTarget
    return _ManagedTarget(
        endpoint=_chat_completions_target(base_url),
        model=model.strip(),
        requires_credential=True,
    )


def _base_payload(model: str, user_text: str, tool_choice: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": build_voice_system_prompt(_FIXED_SESSION_PROMPT)},
            {"role": "user", "content": user_text},
        ],
        "tools": [_FIXED_TOOL],
        "tool_choice": tool_choice,
        "stream": True,
        "temperature": 0,
        "max_tokens": 96,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _continuation_payload(model: str) -> dict[str, Any]:
    arguments = json.dumps({_FIXED_ARGUMENT_KEY: _FIXED_ARGUMENT_VALUE}, separators=(",", ":"))
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": build_voice_system_prompt(_FIXED_SESSION_PROMPT)},
            {"role": "user", "content": _FIXED_REQUIRED_REQUEST},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": _FIXED_CALL_ID,
                        "type": "function",
                        "function": {"name": _FIXED_TOOL_NAME, "arguments": arguments},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": _FIXED_CALL_ID, "content": _FIXED_TOOL_RESULT},
        ],
        "tools": [_FIXED_TOOL],
        "tool_choice": "none",
        "stream": True,
        "temperature": 0,
        "max_tokens": 96,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _normalize_choice(choice: Any) -> Mapping[str, Any]:
    if not isinstance(choice, Mapping):
        raise InvalidModelResponse
    delta = choice.get("delta")
    if isinstance(delta, Mapping):
        return choice
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise InvalidModelResponse
    return {
        "delta": {
            "content": message.get("content"),
            "tool_calls": message.get("tool_calls"),
        },
        "finish_reason": choice.get("finish_reason"),
    }


def _decoded_events(lines: Iterable[str | bytes]) -> Iterable[Mapping[str, Any]]:
    for raw_line in lines:
        if isinstance(raw_line, bytes):
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise InvalidModelResponse from exc
        elif isinstance(raw_line, str):
            line = raw_line
        else:
            raise InvalidModelResponse
        line = line.strip()
        if not line or line.startswith(":"):
            continue
        if line.startswith("data:"):
            line = line[5:].strip()
        if line == "[DONE]":
            break
        try:
            payload = json.loads(line)
        except (json.JSONDecodeError, TypeError) as exc:
            raise InvalidModelResponse from exc
        if not isinstance(payload, Mapping):
            raise InvalidModelResponse
        yield payload


def _accumulate_tool_fragments(choice: Mapping[str, Any], tool_accum: dict[int, dict[str, str]]) -> None:
    delta = choice.get("delta")
    if not isinstance(delta, Mapping):
        return
    fragments = delta.get("tool_calls")
    if fragments is None:
        return
    if not isinstance(fragments, list):
        return
    for fragment in fragments:
        if not isinstance(fragment, Mapping):
            continue
        try:
            index = int(fragment.get("index") or 0)
        except (TypeError, ValueError) as exc:
            raise InvalidModelResponse from exc
        if index < 0:
            raise InvalidModelResponse
        entry = tool_accum.setdefault(index, {"name": "", "args": "", "id": ""})
        call_id = fragment.get("id")
        if call_id:
            entry["id"] = str(call_id)
        function = fragment.get("function")
        if not isinstance(function, Mapping):
            continue
        name = function.get("name")
        if name:
            entry["name"] = str(name)
        arguments = function.get("arguments")
        if arguments:
            entry["args"] += str(arguments)


def _expected_call_is_valid(tool_accum: Mapping[int, Mapping[str, str]], completed_call_count: int) -> bool:
    if completed_call_count != 1 or len(tool_accum) != 1:
        return False
    entry = next(iter(tool_accum.values()))
    if entry.get("name") != _FIXED_TOOL_NAME:
        return False
    try:
        arguments = json.loads(entry.get("args", ""))
    except (json.JSONDecodeError, TypeError):
        return False
    return arguments == {_FIXED_ARGUMENT_KEY: _FIXED_ARGUMENT_VALUE}


def _observe_events(events: Iterable[Mapping[str, Any]], *, tool_choice: str) -> _ProtocolObservation:
    diagnostics = NativeToolStreamDiagnostics(tool_choice=tool_choice)
    tool_accum: dict[int, dict[str, str]] = {}
    choice_seen = False
    for event in events:
        choices = event.get("choices")
        if choices == []:
            continue
        if not isinstance(choices, list) or not choices:
            raise InvalidModelResponse
        choice = _normalize_choice(choices[0])
        choice_seen = True
        diagnostics.observe_choice(choice)
        _accumulate_tool_fragments(choice, tool_accum)
    if not choice_seen:
        raise InvalidModelResponse
    completed_call_count = sum(
        1 for entry in tool_accum.values() if isinstance(entry.get("name"), str) and entry["name"].strip()
    )
    detail = diagnostics.finalize(tool_accum, completed_call_count=completed_call_count)
    return _ProtocolObservation(
        detail=detail,
        expected_call_valid=_expected_call_is_valid(tool_accum, completed_call_count),
    )


def _request_stream(
    client: httpx.Client,
    target: _ManagedTarget,
    credential: str | None,
    payload: Mapping[str, Any],
    timeout_s: float,
) -> _ProtocolObservation:
    headers = {"Content-Type": "application/json"}
    if credential is not None:
        headers["Authorization"] = f"Bearer {credential}"
    try:
        with client.stream(
            "POST",
            target.endpoint,
            headers=headers,
            json=dict(payload),
            timeout=timeout_s,
        ) as response:
            response.raise_for_status()
            return _observe_events(_decoded_events(response.iter_lines()), tool_choice=str(payload.get("tool_choice")))
    except ProbeError:
        raise
    except httpx.HTTPError as exc:
        raise ModelRequestFailed from exc
    except (TypeError, ValueError) as exc:
        raise InvalidModelResponse from exc


def _length_bucket(length: int) -> str:
    if length <= 0:
        return "zero"
    if length <= 64:
        return "1_64"
    if length <= 256:
        return "65_256"
    return "257_plus"


def _public_detail(observation: _ProtocolObservation, *, elapsed_ms: int, passed: bool) -> dict[str, Any]:
    detail = dict(observation.detail)
    length = int(detail.get("assistant_text_length", 0))
    detail["assistant_text_length_bucket"] = _length_bucket(length)
    detail["elapsed_ms"] = max(0, int(elapsed_ms))
    detail["passed"] = bool(passed)
    return detail


def _deterministic_classification_cases() -> dict[str, dict[str, Any]]:
    prose = NativeToolStreamDiagnostics(tool_choice="auto")
    prose.observe_choice(
        {"delta": {"content": _FIXED_PROSE_SHAPED_RESPONSE}, "finish_reason": "stop"}
    )
    prose_observation = _ProtocolObservation(detail=prose.finalize({}, completed_call_count=0))
    prose_passed = (
        prose_observation.detail["native_tool_fragment_count"] == 0
        and prose_observation.detail["completed_call_count"] == 0
        and prose_observation.detail["malformed_call_category"] == "none"
        and prose_observation.detail["assistant_text_length"] == len(_FIXED_PROSE_SHAPED_RESPONSE)
    )

    malformed = NativeToolStreamDiagnostics(tool_choice="required")
    malformed_choice = {
        "delta": {
            "tool_calls": [
                {
                    "index": 0,
                    "function": {"name": _FIXED_TOOL_NAME, "arguments": _FIXED_MALFORMED_ARGUMENTS},
                }
            ]
        },
        "finish_reason": "tool_calls",
    }
    malformed.observe_choice(malformed_choice)
    malformed_observation = _ProtocolObservation(
        detail=malformed.finalize(
            {0: {"name": _FIXED_TOOL_NAME, "args": _FIXED_MALFORMED_ARGUMENTS, "id": ""}},
            completed_call_count=1,
        )
    )
    malformed_passed = malformed_observation.detail["malformed_call_category"] == "native_invalid_arguments_json"

    return {
        "deterministic_prose_shaped": _public_detail(prose_observation, elapsed_ms=0, passed=prose_passed),
        "deterministic_malformed_native": _public_detail(
            malformed_observation,
            elapsed_ms=0,
            passed=malformed_passed,
        ),
    }


def _live_case_passed(category: str, observation: _ProtocolObservation) -> bool:
    detail = observation.detail
    if detail.get("malformed_call_category") != "none":
        return False
    if category in {"required_native_tool", "auto_tool_selection"}:
        return (
            detail.get("finish_reason_category") == "tool_calls"
            and detail.get("assistant_text_length") == 0
            and int(detail.get("native_tool_fragment_count", 0)) >= 1
            and detail.get("completed_call_count") == 1
            and observation.expected_call_valid
        )
    return (
        detail.get("finish_reason_category") == "stop"
        and detail.get("assistant_text_length", 0) > 0
        and detail.get("native_tool_fragment_count") == 0
        and detail.get("completed_call_count") == 0
    )


def run_live_probe(
    *,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    client_factory: Callable[..., httpx.Client] = httpx.Client,
    requester: Callable[
        [httpx.Client, _ManagedTarget, str | None, Mapping[str, Any], float], _ProtocolObservation
    ] = _request_stream,
    credential_reader: Callable[[str], str | None] = os.environ.get,
) -> dict[str, Any]:
    """Run four one-shot live requests plus two deterministic parser cases."""

    started = time.perf_counter()
    attempted = 0
    completed = 0
    public_cases = _deterministic_classification_cases()
    bounded_timeout = max(1.0, min(120.0, float(timeout_s)))
    try:
        with client_factory(timeout=bounded_timeout, follow_redirects=False) as client:
            target = _resolve_managed_target(client)
            credential = credential_reader(_CREDENTIAL_ENV) if target.requires_credential else None
            if target.requires_credential and not credential:
                raise CredentialUnavailable
            cases = (
                (
                    "required_native_tool",
                    _base_payload(target.model, _FIXED_REQUIRED_REQUEST, "required"),
                ),
                (
                    "auto_tool_selection",
                    _base_payload(target.model, _FIXED_AUTO_TOOL_REQUEST, "auto"),
                ),
                (
                    "ordinary_auto_no_call",
                    _base_payload(target.model, _FIXED_ORDINARY_REQUEST, "auto"),
                ),
                ("tool_result_continuation", _continuation_payload(target.model)),
            )
            for category, payload in cases:
                attempted += 1
                request_started = time.perf_counter()
                observation = requester(client, target, credential, payload, bounded_timeout)
                completed += 1
                elapsed_ms = round((time.perf_counter() - request_started) * 1000.0)
                public_cases[category] = _public_detail(
                    observation,
                    elapsed_ms=elapsed_ms,
                    passed=_live_case_passed(category, observation),
                )
    except ProbeProgressError:
        raise
    except Exception as exc:  # noqa: BLE001 - public output receives counts only
        raise ProbeProgressError(attempted=attempted, completed=completed) from exc

    gate_passed = all(case["passed"] for case in public_cases.values())
    return {
        "model_request_attempted_count": attempted,
        "model_request_completed_count": completed,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0),
        "gate_passed": gate_passed,
        **public_cases,
    }


def _bounded_failure(exc: BaseException, *, elapsed_ms: int) -> dict[str, int | bool]:
    attempted = exc.attempted if isinstance(exc, ProbeProgressError) else 0
    completed = exc.completed if isinstance(exc, ProbeProgressError) else 0
    return {
        "model_request_attempted_count": attempted,
        "model_request_completed_count": completed,
        "elapsed_ms": max(0, int(elapsed_ms)),
        "gate_passed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the content-free native tool protocol gate.")
    parser.add_argument("--timeout", type=float, default=_DEFAULT_TIMEOUT_S)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    try:
        result = run_live_probe(timeout_s=args.timeout)
    except Exception as exc:  # noqa: BLE001 - exception content is deliberately discarded
        result = _bounded_failure(exc, elapsed_ms=round((time.perf_counter() - started) * 1000.0))
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
