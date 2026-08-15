from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "probe_native_tool_protocol.py"
SPEC = importlib.util.spec_from_file_location("native_tool_protocol_probe", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


class _Response:
    def __init__(self, payload, *, url, history=None):
        self._payload = payload
        self.url = url
        self.history = list(history or [])

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _Client:
    def __init__(self, *, provider="local", local_base=None, redirect=False):
        self.provider = provider
        self.local_base = local_base or probe._LOCAL_MODEL_BASE_URL
        self.redirect = redirect
        self.get_calls = []
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.closed = True

    def get(self, url):
        self.get_calls.append(url)
        if url == probe._MANAGED_SETTINGS_URL:
            if self.provider == "remote":
                payload = {
                    "settings": {
                        "modelProvider": "remote",
                        "modelUrl": "https://private.invalid/v1",
                        "modelName": "private-remote-model",
                    }
                }
            else:
                payload = {"settings": {"modelProvider": self.provider}}
            return _Response(
                payload,
                url=url,
                history=[object()] if self.redirect else [],
            )
        if url == probe._MANAGED_LOCAL_PIPELINE_URL:
            return _Response(
                {
                    "gemma": {
                        "baseUrl": self.local_base,
                        "model": "private-local-model",
                    }
                },
                url=url,
            )
        raise AssertionError("unexpected managed request")


class _ClientFactory:
    def __init__(self, client):
        self.client = client
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return self.client


def _detail(*, tools: bool) -> dict[str, int | str]:
    return {
        "finish_reason_category": "tool_calls" if tools else "stop",
        "assistant_text_length": 0 if tools else 18,
        "native_tool_fragment_count": 1 if tools else 0,
        "completed_call_count": 1 if tools else 0,
        "malformed_call_category": "none",
    }


def _successful_requester(call_log):
    def request(_client, target, credential, payload, timeout_s):
        index = len(call_log)
        call_log.append((target, credential, payload, timeout_s))
        tools = index < 2
        return probe._ProtocolObservation(
            detail=_detail(tools=tools),
            expected_call_valid=tools,
        )

    return request


def test_managed_local_probe_uses_exact_loopback_route_without_reading_a_key():
    client = _Client()
    factory = _ClientFactory(client)
    calls = []

    def forbidden_credential_read(_name):
        raise AssertionError("local routing must not read a remote credential")

    result = probe.run_live_probe(
        client_factory=factory,
        requester=_successful_requester(calls),
        credential_reader=forbidden_credential_read,
    )

    assert factory.kwargs == {"timeout": 45.0, "follow_redirects": False}
    assert client.get_calls == [probe._MANAGED_SETTINGS_URL, probe._MANAGED_LOCAL_PIPELINE_URL]
    assert len(calls) == 4
    assert all(call[0].endpoint == "http://127.0.0.1:8818/v1/chat/completions" for call in calls)
    assert all(call[1] is None for call in calls)
    assert result["model_request_attempted_count"] == 4
    assert result["model_request_completed_count"] == 4
    assert result["gate_passed"] is True


def test_remote_probe_reads_key_only_after_exact_managed_settings_handoff():
    client = _Client(provider="remote")
    calls = []
    credential_reads = []

    def credential_reader(name):
        credential_reads.append(name)
        return "PRIVATE_REMOTE_CREDENTIAL"

    result = probe.run_live_probe(
        client_factory=_ClientFactory(client),
        requester=_successful_requester(calls),
        credential_reader=credential_reader,
    )

    assert client.get_calls == [probe._MANAGED_SETTINGS_URL]
    assert credential_reads == [probe._CREDENTIAL_ENV]
    assert all(call[0].endpoint == "https://private.invalid/v1/chat/completions" for call in calls)
    assert all(call[1] == "PRIVATE_REMOTE_CREDENTIAL" for call in calls)
    rendered = json.dumps(result, sort_keys=True)
    assert "PRIVATE_REMOTE_CREDENTIAL" not in rendered
    assert "private.invalid" not in rendered
    assert "private-remote-model" not in rendered


def test_redirected_settings_are_rejected_before_credential_read_or_model_request():
    client = _Client(provider="remote", redirect=True)
    credential_reads = []
    model_requests = []

    with pytest.raises(probe.ProbeProgressError) as caught:
        probe.run_live_probe(
            client_factory=_ClientFactory(client),
            requester=lambda *args: model_requests.append(args),
            credential_reader=lambda name: credential_reads.append(name),
        )

    assert caught.value.attempted == 0
    assert caught.value.completed == 0
    assert credential_reads == []
    assert model_requests == []


@pytest.mark.parametrize(
    "unsafe_base",
    [
        "http://localhost:8818/v1",
        "http://127.0.0.1:8819/v1",
        "http://127.0.0.1:8818/v1?private=1",
        "http://name:secret@127.0.0.1:8818/v1",
        "http://192.0.2.10:8818/v1",
    ],
)
def test_managed_local_descriptor_rejects_every_noncanonical_model_route(unsafe_base):
    client = _Client(local_base=unsafe_base)

    with pytest.raises(probe.UntrustedManagedTarget):
        probe._resolve_managed_target(client)


def test_live_payload_order_covers_required_auto_plain_and_tool_disabled_continuation():
    calls = []

    probe.run_live_probe(
        client_factory=_ClientFactory(_Client()),
        requester=_successful_requester(calls),
        credential_reader=lambda _name: None,
    )

    payloads = [call[2] for call in calls]
    assert [payload["tool_choice"] for payload in payloads] == ["required", "auto", "auto", "none"]
    assert all(payload["stream"] is True for payload in payloads)
    assert all("Call it silently." in payloads[index]["messages"][-1]["content"] for index in (0, 1))
    assert [message["role"] for message in payloads[3]["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert payloads[3]["tool_choice"] == "none"
    assert len(payloads[3]["messages"][2]["tool_calls"]) == 1


def test_live_tool_case_rejects_spoken_text_even_when_native_call_is_valid():
    observation = probe._ProtocolObservation(
        detail={**_detail(tools=True), "assistant_text_length": 12},
        expected_call_valid=True,
    )

    assert probe._live_case_passed("required_native_tool", observation) is False
    assert probe._live_case_passed("auto_tool_selection", observation) is False


def test_stream_parser_counts_native_fragments_and_validates_only_completed_native_call():
    first = {
        "choices": [
            {
                "delta": {
                    "content": "brief preamble",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "private-call-id",
                            "function": {
                                "name": probe._FIXED_TOOL_NAME,
                                "arguments": '{"record_id":"fixed-',
                            },
                        }
                    ],
                },
                "finish_reason": None,
            }
        ]
    }
    second = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {"arguments": 'protocol-record"}'},
                        }
                    ]
                },
                "finish_reason": None,
            }
        ]
    }
    final = {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}
    lines = [
        f"data: {json.dumps(first)}",
        f"data: {json.dumps(second)}",
        f"data: {json.dumps(final)}",
        "data: [DONE]",
    ]

    observation = probe._observe_events(probe._decoded_events(lines), tool_choice="required")

    assert observation.expected_call_valid is True
    assert observation.detail == {
        "finish_reason_category": "tool_calls",
        "assistant_text_length": len("brief preamble"),
        "native_tool_fragment_count": 2,
        "completed_call_count": 1,
        "malformed_call_category": "none",
    }


def test_deterministic_prose_shape_remains_text_and_malformed_native_args_are_classified():
    cases = probe._deterministic_classification_cases()

    prose = cases["deterministic_prose_shaped"]
    assert prose["assistant_text_length"] == len(probe._FIXED_PROSE_SHAPED_RESPONSE)
    assert prose["native_tool_fragment_count"] == 0
    assert prose["completed_call_count"] == 0
    assert prose["malformed_call_category"] == "none"
    assert prose["passed"] is True

    malformed = cases["deterministic_malformed_native"]
    assert malformed["native_tool_fragment_count"] == 1
    assert malformed["completed_call_count"] == 1
    assert malformed["malformed_call_category"] == "native_invalid_arguments_json"
    assert malformed["passed"] is True


def test_public_success_shape_contains_only_bounded_metrics_and_no_private_fixture_values():
    calls = []
    result = probe.run_live_probe(
        client_factory=_ClientFactory(_Client()),
        requester=_successful_requester(calls),
        credential_reader=lambda _name: None,
    )
    scenario_keys = {
        "deterministic_prose_shaped",
        "deterministic_malformed_native",
        "required_native_tool",
        "auto_tool_selection",
        "ordinary_auto_no_call",
        "tool_result_continuation",
    }
    assert set(result) == {
        "model_request_attempted_count",
        "model_request_completed_count",
        "elapsed_ms",
        "gate_passed",
        *scenario_keys,
    }
    detail_keys = {
        "finish_reason_category",
        "assistant_text_length",
        "assistant_text_length_bucket",
        "native_tool_fragment_count",
        "completed_call_count",
        "malformed_call_category",
        "elapsed_ms",
        "passed",
    }
    assert all(set(result[key]) == detail_keys for key in scenario_keys)

    rendered = json.dumps(result, sort_keys=True)
    forbidden = (
        probe._FIXED_SESSION_PROMPT,
        probe._FIXED_TOOL_NAME,
        probe._FIXED_ARGUMENT_KEY,
        probe._FIXED_ARGUMENT_VALUE,
        probe._FIXED_TOOL_RESULT,
        probe._FIXED_REQUIRED_REQUEST,
        probe._FIXED_AUTO_TOOL_REQUEST,
        probe._FIXED_ORDINARY_REQUEST,
        probe._FIXED_PROSE_SHAPED_RESPONSE,
        probe._FIXED_MALFORMED_ARGUMENTS,
        probe._LOCAL_MODEL_BASE_URL,
        "private-local-model",
    )
    assert all(value not in rendered for value in forbidden)


def test_model_failure_is_one_shot_and_public_failure_discards_exception_content():
    calls = []

    def failing_requester(*_args):
        calls.append(1)
        raise RuntimeError("PRIVATE_RESPONSE PRIVATE_ENDPOINT PRIVATE_KEY")

    with pytest.raises(probe.ProbeProgressError) as caught:
        probe.run_live_probe(
            client_factory=_ClientFactory(_Client()),
            requester=failing_requester,
            credential_reader=lambda _name: None,
        )

    assert calls == [1]
    assert caught.value.attempted == 1
    assert caught.value.completed == 0
    public = probe._bounded_failure(caught.value, elapsed_ms=7)
    assert public == {
        "model_request_attempted_count": 1,
        "model_request_completed_count": 0,
        "elapsed_ms": 7,
        "gate_passed": False,
    }
    assert "PRIVATE" not in json.dumps(public)


def test_main_emits_one_json_object_with_empty_stderr_and_no_exception_message(monkeypatch, capsys):
    def fail_probe(**_kwargs):
        raise RuntimeError("PRIVATE_PROMPT PRIVATE_RESPONSE PRIVATE_TOOL PRIVATE_KEY")

    monkeypatch.setattr(probe, "run_live_probe", fail_probe)

    exit_code = probe.main([])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.err == ""
    lines = captured.out.splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["gate_passed"] is False
    assert payload["model_request_attempted_count"] == 0
    assert "PRIVATE" not in captured.out
