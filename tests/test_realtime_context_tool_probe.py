from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import math
import struct
import subprocess
import sys
import wave
from collections import deque
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "probe_realtime_context_tools.py"
_SPEC = importlib.util.spec_from_file_location("probe_realtime_context_tools", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
probe = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = probe
_SPEC.loader.exec_module(probe)

_SECRET_ENDPOINT = "http://endpoint-secret.example/v1"
_SECRET_MODEL = "model-secret-value"
_SECRET_KEY = "api-key-secret-value"
_SECRET_TOOL_CODE = "VCODE-SECRET-7391"


class FakeWebSocket:
    def __init__(self, events: list[dict]):
        self.events = deque(json.dumps(event) for event in events)
        self.sent: list[dict] = []

    async def send(self, raw: str):
        self.sent.append(json.loads(raw))

    async def recv(self):
        if not self.events:
            await asyncio.Future()
        return self.events.popleft()


class FakeConnection:
    def __init__(self, ws: FakeWebSocket):
        self.ws = ws

    async def __aenter__(self):
        return self.ws

    async def __aexit__(self, *_args):
        return False


def _settings() -> probe.RuntimeSettings:
    return probe.RuntimeSettings(
        websocket_url=probe.MANAGED_WEBSOCKET_URL,
        model_base_url=_SECRET_ENDPOINT,
        model_name=_SECRET_MODEL,
        model_api_key=_SECRET_KEY,
        tts_backend="qwen3tts-audiocpp",
        voice="clone:voice-secret-value",
        full_buffer_tts=False,
        max_response_tokens=384,
    )


def _prompt_audio() -> dict[str, bytes]:
    return {name: struct.pack("<160h", *([index + 1] * 160)) for index, name in enumerate(probe._PROMPTS)}


def _response_events(response_id: str, transcript: str, *, status: str = "completed") -> list[dict]:
    return [
        {"type": "input_audio_buffer.speech_started"},
        {"type": "input_audio_buffer.speech_stopped"},
        {"type": "response.created", "response": {"id": response_id, "status": "in_progress"}},
        {
            "type": "response.output_audio_transcript.done",
            "response_id": response_id,
            "transcript": transcript,
        },
        {"type": "response.output_audio.done", "response_id": response_id},
        {"type": "response.done", "response": {"id": response_id, "status": status}},
    ]


def _pipeline_ack() -> dict:
    settings = _settings()
    return {
        "type": "pipeline.config.updated",
        "config": {
            "full_buffer_tts": settings.full_buffer_tts,
            "live_transcription": False,
            "max_response_tokens": settings.max_response_tokens,
            "tts_backend": settings.tts_backend,
            "model_endpoint": {
                "provider": "remote",
                "base_url": settings.model_base_url,
                "model": settings.model_name,
                "api_key_set": True,
            },
        },
    }


def _session_ack(tool_choice: str, *, complete: bool = False) -> dict:
    session = {"type": "realtime", "tool_choice": tool_choice}
    if complete:
        session.update(
            {
                "audio": {"output": {"voice": _settings().voice}},
                "tools": [probe._SEARCH_TOOL],
            }
        )
    return {"type": "session.updated", "session": session}


def _success_events(*, reverse_text: str = "10, 9, 8, 7, 6, 5, 4, 3, 2, 1", tool_args: str = '{"query":"continuity"}') -> list[dict]:
    events = [
        {"type": "session.created", "session": {"id": "session-private"}},
        {"type": "pipeline.runtime", "runtime": {"status": "ready"}},
        _pipeline_ack(),
        _session_ack("auto", complete=True),
    ]
    events.extend(_response_events("response-count-up", "1, 2, 3, 4, 5, 6, 7, 8, 9, 10"))
    events.extend(_response_events("response-count-down", reverse_text))
    events.append(_session_ack("required"))
    events.extend(
        [
            {"type": "input_audio_buffer.speech_started"},
            {"type": "input_audio_buffer.speech_stopped"},
            {"type": "response.created", "response": {"id": "response-tool-origin", "status": "in_progress"}},
            {
                "type": "response.function_call_arguments.done",
                "response_id": "response-tool-origin",
                "call_id": "call-private-1",
                "name": "web_search",
                "arguments": tool_args,
            },
            {
                "type": "conversation.item.created",
                "item": {"type": "function_call_output", "call_id": "call-private-1"},
            },
            {"type": "response.done", "response": {"id": "response-tool-origin", "status": "completed"}},
            {"type": "response.created", "response": {"id": "response-tool-followup", "status": "in_progress"}},
            {
                "type": "response.output_audio_transcript.done",
                "response_id": "response-tool-followup",
                "transcript": _SECRET_TOOL_CODE,
            },
            {"type": "response.output_audio.done", "response_id": "response-tool-followup"},
            {"type": "response.done", "response": {"id": "response-tool-followup", "status": "completed"}},
        ]
    )
    events.append(_session_ack("auto"))
    events.extend(_response_events("response-search-context", _SECRET_TOOL_CODE))
    events.extend(
        [
            {"type": "input_audio_buffer.speech_started"},
            {"type": "input_audio_buffer.speech_stopped"},
            {"type": "response.created", "response": {"id": "response-cancel", "status": "in_progress"}},
            {"type": "response.output_audio.done", "response_id": "response-cancel"},
            {
                "type": "response.done",
                "response": {
                    "id": "response-cancel",
                    "status": "cancelled",
                    "status_details": {"reason": "client_cancelled"},
                },
            },
        ]
    )
    events.extend(_response_events("response-recovery", "READY"))
    return events


def _run(events: list[dict]) -> tuple[dict, FakeWebSocket]:
    ws = FakeWebSocket(events)
    result = asyncio.run(
        probe.run_live_gate(
            _settings(),
            _prompt_audio(),
            response_timeout_s=0.25,
            pace_s=0,
            connect=lambda *_args, **_kwargs: FakeConnection(ws),
            tool_code=_SECRET_TOOL_CODE,
        )
    )
    return result, ws


def _wave_base64(frequency: float) -> str:
    samples = [round(math.sin(2 * math.pi * frequency * index / 16_000) * 8_000) for index in range(8_000)]
    output = BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return base64.b64encode(output.getvalue()).decode("ascii")


def test_success_gate_proves_ack_context_tool_order_and_cancel_recovery_without_leaks():
    result, ws = _run(_success_events())

    assert result["gate_passed"] is True
    assert result["config"]["acknowledged"] is True
    assert result["config"]["audio_before_ack_count"] == 0
    assert result["context"] == {
        "turn_count": 2,
        "ascending_valid": True,
        "reverse_valid": True,
        "preserved": True,
    }
    assert result["tool"]["call_count"] == 1
    assert result["tool"]["output_count"] == 1
    assert result["tool"]["output_ack_count"] == 1
    assert result["tool"]["response_create_count"] == 1
    assert result["tool"]["origin_done_count"] == 1
    assert result["tool"]["post_response_created_count"] == 1
    assert result["tool"]["post_response_done_count"] == 1
    assert result["tool"]["ordering_valid"] is True
    assert result["tool"]["contextual_followup_preserved"] is True
    assert result["cancellation"]["cancelled_done_count"] == 1
    assert result["cancellation"]["audio_done_count"] == 1
    assert result["cancellation"]["ordering_valid"] is True
    assert result["cancellation"]["recovery_valid"] is True

    sent_types = [event["type"] for event in ws.sent]
    assert sent_types.index("pipeline.config.update") < sent_types.index("input_audio_buffer.append")
    output_index = next(
        index
        for index, event in enumerate(ws.sent)
        if event["type"] == "conversation.item.create"
        and event["item"]["type"] == "function_call_output"
    )
    create_indices = [index for index, event in enumerate(ws.sent) if event["type"] == "response.create"]
    assert len(create_indices) == 1
    assert output_index < create_indices[0]
    assert ws.sent[create_indices[0]]["response"] == {"tool_choice": "none"}
    tool_choices = [
        event["session"].get("tool_choice")
        for event in ws.sent
        if event["type"] == "session.update"
    ]
    assert tool_choices == ["auto", "required", "auto"]
    assert sum(event["type"] == "response.cancel" for event in ws.sent) == 1

    serialized = json.dumps(result)
    for secret in (
        _SECRET_ENDPOINT,
        _SECRET_MODEL,
        _SECRET_KEY,
        _SECRET_TOOL_CODE,
        "voice-secret-value",
        "Count from one",
        "response-count-up",
        "call-private-1",
    ):
        assert secret not in serialized


def test_wrong_reverse_sequence_fails_context_gate_without_changing_tool_order():
    result, _ws = _run(_success_events(reverse_text="1, 2, 3, 4, 5, 6, 7, 8, 9, 10"))
    assert result["context"]["ascending_valid"] is True
    assert result["context"]["reverse_valid"] is False
    assert result["context"]["preserved"] is False
    assert result["tool"]["gate_passed"] is True
    assert result["gate_passed"] is False


@pytest.mark.parametrize(
    "bad_arguments",
    [
        "not-json",
        "[]",
        "{}",
        '{"query":7}',
        '{"query":"ok","secret":"must-not-execute"}',
    ],
)
def test_invalid_tool_arguments_fail_before_function_output_or_followup(bad_arguments):
    ws = FakeWebSocket(_success_events(tool_args=bad_arguments))
    with pytest.raises(probe.ProtocolFailure):
        asyncio.run(
            probe.run_live_gate(
                _settings(),
                _prompt_audio(),
                response_timeout_s=0.25,
                pace_s=0,
                connect=lambda *_args, **_kwargs: FakeConnection(ws),
                tool_code=_SECRET_TOOL_CODE,
            )
        )
    assert not any(
        event["type"] == "conversation.item.create"
        and event.get("item", {}).get("type") == "function_call_output"
        for event in ws.sent
    )
    assert not any(event["type"] == "response.create" for event in ws.sent)


def test_pre_ack_server_error_fails_closed_before_any_audio_and_hides_error_content():
    server_secret = "server-error-secret-value"
    ws = FakeWebSocket(
        [
            {"type": "session.created", "session": {"id": "private"}},
            {"type": "error", "error": {"type": "private_failure", "message": server_secret}},
        ]
    )
    with pytest.raises(probe.ConfigAcknowledgementFailed) as captured:
        asyncio.run(
            probe.run_live_gate(
                _settings(),
                _prompt_audio(),
                response_timeout_s=0.05,
                pace_s=0,
                connect=lambda *_args, **_kwargs: FakeConnection(ws),
            )
        )
    assert not any(event["type"] == "input_audio_buffer.append" for event in ws.sent)
    bounded = json.dumps(probe._bounded_failure(captured.value))
    assert json.loads(bounded)["failure_stage"] == "config_ack"
    assert server_secret not in bounded
    assert _SECRET_ENDPOINT not in bounded
    assert _SECRET_KEY not in bounded


@pytest.mark.parametrize(
    ("event_index", "replacement"),
    [
        (2, {"type": "pipeline.config.updated", "config": {"tts_backend": "groxaxo"}}),
        (3, _session_ack("required", complete=True)),
        (16, _session_ack("auto")),
        (27, _session_ack("required")),
    ],
)
def test_mismatched_or_stale_ack_values_fail_closed(event_index, replacement):
    events = _success_events()
    assert events[event_index]["type"] in {"pipeline.config.updated", "session.updated"}
    events[event_index] = replacement
    ws = FakeWebSocket(events)
    with pytest.raises((probe.ConfigAcknowledgementFailed, probe.SessionAcknowledgementFailed)):
        asyncio.run(
            probe.run_live_gate(
                _settings(),
                _prompt_audio(),
                response_timeout_s=0.25,
                pace_s=0,
                connect=lambda *_args, **_kwargs: FakeConnection(ws),
                tool_code=_SECRET_TOOL_CODE,
            )
        )


def test_untrusted_websocket_fails_before_connect_or_secret_transmission():
    connect_count = 0

    def connect(*_args, **_kwargs):
        nonlocal connect_count
        connect_count += 1
        raise AssertionError("must not connect")

    with pytest.raises(probe.UntrustedManagedTarget):
        asyncio.run(
            probe.run_live_gate(
                replace(_settings(), websocket_url="wss://collector.invalid/v1/realtime"),
                _prompt_audio(),
                response_timeout_s=0.25,
                pace_s=0,
                connect=connect,
            )
        )
    assert connect_count == 0


def test_untrusted_cli_target_fails_before_credential_or_synthesis(monkeypatch, capsys):
    monkeypatch.setenv(probe.CREDENTIAL_ENV, _SECRET_KEY)
    monkeypatch.setattr(
        probe,
        "synthesize_prompts_in_memory",
        lambda: (_ for _ in ()).throw(AssertionError("must not synthesize")),
    )
    exit_code = probe.main(["--settings-url", "http://collector.invalid/api/ui-settings"])
    output = capsys.readouterr().out
    assert exit_code == 2
    assert json.loads(output) == {
        "error_class": "UntrustedManagedTarget",
        "failure_stage": "settings",
        "gate_passed": False,
    }
    assert _SECRET_KEY not in output
    assert "collector" not in output


def test_redirected_settings_response_is_rejected_before_payload_use():
    class RedirectedResponse:
        history = [object()]
        url = probe.MANAGED_SETTINGS_URL

        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            raise AssertionError("redirected payload must not be read")

    class Client:
        @staticmethod
        def get(_url):
            return RedirectedResponse()

    with pytest.raises(probe.UntrustedManagedTarget):
        probe.resolve_runtime_settings(
            Client(),
            probe.MANAGED_SETTINGS_URL,
            probe.MANAGED_WEBSOCKET_URL,
            _SECRET_KEY,
        )


def test_duplicate_tool_output_ack_is_rejected_before_second_response_create():
    events = _success_events()
    ack_index = next(
        index
        for index, event in enumerate(events)
        if event["type"] == "conversation.item.created"
    )
    events.insert(ack_index + 1, dict(events[ack_index]))
    ws = FakeWebSocket(events)
    with pytest.raises(probe.ProtocolFailure):
        asyncio.run(
            probe.run_live_gate(
                _settings(),
                _prompt_audio(),
                response_timeout_s=0.25,
                pace_s=0,
                connect=lambda *_args, **_kwargs: FakeConnection(ws),
                tool_code=_SECRET_TOOL_CODE,
            )
        )
    assert sum(event["type"] == "response.create" for event in ws.sent) == 1


def test_post_tool_response_before_origin_done_is_rejected():
    events = _success_events()
    origin_done_index = next(
        index
        for index, event in enumerate(events)
        if event.get("type") == "response.done"
        and event.get("response", {}).get("id") == "response-tool-origin"
    )
    del events[origin_done_index]
    ws = FakeWebSocket(events)
    with pytest.raises(probe.ProtocolFailure):
        asyncio.run(
            probe.run_live_gate(
                _settings(),
                _prompt_audio(),
                response_timeout_s=0.25,
                pace_s=0,
                connect=lambda *_args, **_kwargs: FakeConnection(ws),
                tool_code=_SECRET_TOOL_CODE,
            )
        )


def test_origin_done_before_late_tool_arguments_preserves_exact_tool_transaction():
    events = _success_events()
    origin_done_index = next(
        index
        for index, event in enumerate(events)
        if event.get("type") == "response.done"
        and event.get("response", {}).get("id") == "response-tool-origin"
    )
    origin_done = events.pop(origin_done_index)
    tool_call_index = next(
        index
        for index, event in enumerate(events)
        if event.get("type") == "response.function_call_arguments.done"
    )
    events.insert(tool_call_index, origin_done)
    result, ws = _run(events)
    assert result["gate_passed"] is True
    assert result["tool"]["ordering_valid"] is True
    assert result["tool"]["origin_done_count"] == 1
    assert result["tool"]["call_count"] == 1
    assert result["tool"]["output_ack_count"] == 1
    assert sum(event["type"] == "response.create" for event in ws.sent) == 1


def test_duplicate_cancel_audio_done_fails_single_terminal_delivery_gate():
    events = _success_events()
    cancel_audio_index = next(
        index
        for index, event in enumerate(events)
        if event.get("type") == "response.output_audio.done"
        and event.get("response_id") == "response-cancel"
    )
    events.insert(cancel_audio_index + 1, dict(events[cancel_audio_index]))
    result, _ws = _run(events)
    assert result["cancellation"]["audio_done_count"] == 2
    assert result["cancellation"]["gate_passed"] is False
    assert result["gate_passed"] is False


def test_duplicate_completed_post_tool_response_is_rejected_on_next_receive():
    events = _success_events()
    post_done_index = next(
        index
        for index, event in enumerate(events)
        if event.get("type") == "response.done"
        and event.get("response", {}).get("id") == "response-tool-followup"
    )
    events.insert(post_done_index + 1, dict(events[post_done_index]))
    ws = FakeWebSocket(events)
    with pytest.raises(probe.SessionAcknowledgementFailed) as captured:
        asyncio.run(
            probe.run_live_gate(
                _settings(),
                _prompt_audio(),
                response_timeout_s=0.25,
                pace_s=0,
                connect=lambda *_args, **_kwargs: FakeConnection(ws),
                tool_code=_SECRET_TOOL_CODE,
            )
        )
    assert probe._bounded_failure(captured.value) == {
        "error_class": "SessionAcknowledgementFailed",
        "failure_stage": "search_tool_restore_ack",
        "gate_passed": False,
    }


def test_sapi_runner_uses_stdin_and_captured_memory_without_prompt_or_audio_in_argv():
    captured: dict = {}
    encoded = [_wave_base64(300.0 + index * 40.0) for index in range(len(probe._PROMPTS))]

    def runner(args, **kwargs):
        captured["args"] = args
        captured["input"] = kwargs["input"]
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        return subprocess.CompletedProcess(args, 0, stdout=json.dumps(encoded), stderr="private-stderr")

    audio = probe.synthesize_prompts_in_memory(runner=runner)

    assert set(audio) == set(probe._PROMPTS)
    assert all(value and isinstance(value, bytes) for value in audio.values())
    argv = " ".join(captured["args"])
    assert all(phrase not in argv for phrase in probe._PROMPTS.values())
    assert all(phrase in captured["input"] for phrase in probe._PROMPTS.values())
    assert "private-stderr" not in json.dumps({name: len(value) for name, value in audio.items()})


def test_main_without_exact_credential_env_is_content_free(monkeypatch, capsys):
    monkeypatch.delenv(probe.CREDENTIAL_ENV, raising=False)
    exit_code = probe.main([])
    output = capsys.readouterr().out
    assert exit_code == 2
    assert json.loads(output) == {
        "error_class": "CredentialUnavailable",
        "failure_stage": "credential",
        "gate_passed": False,
    }
    assert "S2S_REMOTE_MODEL_API_KEY" not in output


def test_bounded_failure_never_serializes_exception_message():
    secret = "transport-message-secret"
    output = json.dumps(probe._bounded_failure(probe.ProtocolFailure(secret)))
    assert json.loads(output) == {
        "error_class": "ProtocolFailure",
        "failure_stage": "unknown",
        "gate_passed": False,
    }
    assert secret not in output


def test_missing_required_search_call_reports_safe_search_stage_without_content():
    events = _success_events()
    tool_index = next(
        index for index, event in enumerate(events) if event["type"] == "response.function_call_arguments.done"
    )
    events[tool_index] = {
        "type": "response.done",
        "response": {"id": "response-tool-origin", "status": "completed"},
    }
    ws = FakeWebSocket(events[: tool_index + 1])
    with pytest.raises(probe.ScenarioTimeout) as captured:
        asyncio.run(
            probe.run_live_gate(
                _settings(),
                _prompt_audio(),
                response_timeout_s=0.01,
                pace_s=0,
                connect=lambda *_args, **_kwargs: FakeConnection(ws),
                tool_code=_SECRET_TOOL_CODE,
            )
        )
    bounded = probe._bounded_failure(captured.value)
    assert bounded == {
        "error_class": "ScenarioTimeout",
        "failure_stage": "search",
        "gate_passed": False,
    }
    assert _SECRET_TOOL_CODE not in json.dumps(bounded)
