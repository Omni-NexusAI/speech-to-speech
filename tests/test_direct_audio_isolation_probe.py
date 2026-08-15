from __future__ import annotations

import base64
import copy
import importlib.util
import io
import json
import pathlib
import struct
import subprocess
import sys
import wave

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "probe_direct_audio_isolation.py"
SPEC = importlib.util.spec_from_file_location("direct_audio_isolation_probe", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def _wav_bytes(seed: int) -> bytes:
    output = io.BytesIO()
    samples = [((index * (seed + 3)) % 2000) - 1000 for index in range(1600)]
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return output.getvalue()


def _synthesized_fixtures(*, matched: bool = True):
    return {
        fixture.index: probe.SynthesizedFixture(
            wav=bytearray(_wav_bytes(fixture.index)),
            culture_matched=matched,
        )
        for fixture in probe._FIXTURES
    }


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
                        "modelUrl": "https://private-endpoint.invalid/v1",
                        "modelName": "private-remote-model",
                    }
                }
            else:
                payload = {"settings": {"modelProvider": self.provider}}
            return _Response(payload, url=url, history=[object()] if self.redirect else [])
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


def _provider_with_capture(captured, *, matched=True):
    def provider(fixtures):
        assert tuple(fixtures) == probe._FIXTURES
        generated = _synthesized_fixtures(matched=matched)
        captured.update(generated)
        return generated

    return provider


def _response_for(cell, label):
    if cell.prompt_arm == "full_production":
        content = f"USER_MEMORY: private response memory\nASSISTANT_LANGUAGE: Auto\nASSISTANT_RESPONSE: {label}"
    else:
        content = label
    return {"choices": [{"message": {"content": content}}]}


def _successful_requester(calls, *, wrong_history_fails=False, credential_log=None):
    cells = probe.probe_cells()

    def requester(_client, target, credential, payload, timeout_s):
        cell = cells[len(calls)]
        calls.append((target, credential, copy.deepcopy(payload), timeout_s))
        if credential_log is not None:
            credential_log.append(credential)
        label = (
            cell.fixture.wrong_label
            if wrong_history_fails and cell.history_arm == "wrong_semantic"
            else cell.fixture.expected_label
        )
        return _response_for(cell, label)

    return requester


def _audio_parts(payload):
    return [
        part
        for message in payload["messages"]
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if part.get("type") == "input_audio"
    ]


def test_sapi_synthesizer_uses_one_in_memory_child_call_for_the_fixed_matrix():
    calls = []

    def runner(args, **kwargs):
        calls.append((args, kwargs))
        child = json.loads(kwargs["input"].decode("utf-8"))
        assert child == {
            "items": [
                {"index": fixture.index, "culture": fixture.culture, "text": fixture.utterance}
                for fixture in probe._FIXTURES
            ]
        }
        output = [
            {
                "index": fixture.index,
                "wav": base64.b64encode(_wav_bytes(fixture.index)).decode("ascii"),
                "culture_matched": True,
            }
            for fixture in probe._FIXTURES
        ]
        return subprocess.CompletedProcess(args, 0, stdout=json.dumps(output).encode("utf-8"), stderr=b"")

    result = probe.synthesize_fixed_fixtures_in_memory(probe._FIXTURES, runner=runner)

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[-2] == "-Command"
    assert args[-1] == probe._SAPI_SYNTH_SCRIPT
    assert "SetOutputToAudioStream" in args[-1]
    assert "SetOutputToWaveStream($stream, $format)" not in args[-1]
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is False
    assert set(result) == {fixture.index for fixture in probe._FIXTURES}
    assert all(item.culture_matched for item in result.values())
    assert all(item.wav.startswith(b"RIFF") for item in result.values())

    called = []
    with pytest.raises(probe.InvalidFixtureAudio):
        probe.synthesize_fixed_fixtures_in_memory(probe._FIXTURES[:-1], runner=lambda *a, **k: called.append((a, k)))
    assert called == []


def test_local_matrix_uses_exact_managed_route_and_never_reads_remote_credential():
    client = _Client()
    factory = _ClientFactory(client)
    calls = []
    buffers = {}

    def forbidden_credential_read(_name):
        raise AssertionError("local target must not read a remote credential")

    result = probe.run_isolation_probe(
        client_factory=factory,
        synthesizer=_provider_with_capture(buffers),
        requester=_successful_requester(calls),
        credential_reader=forbidden_credential_read,
    )

    assert factory.kwargs == {"timeout": 45.0, "follow_redirects": False}
    assert client.get_calls == [probe._MANAGED_SETTINGS_URL, probe._MANAGED_LOCAL_PIPELINE_URL]
    assert len(calls) == len(probe.probe_cells()) == 16
    assert all(call[0].endpoint == "http://127.0.0.1:8818/v1/chat/completions" for call in calls)
    assert all(call[1] is None for call in calls)
    assert all(not any(item.wav) for item in buffers.values())
    assert result["gate_passed"] is True


def test_remote_matrix_reads_key_only_after_redirect_rejecting_managed_handoff():
    client = _Client(provider="remote")
    calls = []
    credential_reads = []

    result = probe.run_isolation_probe(
        client_factory=_ClientFactory(client),
        synthesizer=lambda _fixtures: _synthesized_fixtures(),
        requester=_successful_requester(calls),
        credential_reader=lambda name: credential_reads.append(name) or "PRIVATE_REMOTE_KEY",
    )

    assert client.get_calls == [probe._MANAGED_SETTINGS_URL]
    assert credential_reads == [probe._CREDENTIAL_ENV]
    assert all(call[0].endpoint == "https://private-endpoint.invalid/v1/chat/completions" for call in calls)
    assert all(call[1] == "PRIVATE_REMOTE_KEY" for call in calls)
    rendered = json.dumps(result, sort_keys=True)
    assert "PRIVATE_REMOTE_KEY" not in rendered
    assert "private-endpoint" not in rendered
    assert "private-remote-model" not in rendered


def test_redirected_managed_settings_fail_before_key_synthesis_or_model_request():
    credential_reads = []
    synthesis_calls = []
    request_calls = []

    with pytest.raises(probe.ProbeProgressError) as caught:
        probe.run_isolation_probe(
            client_factory=_ClientFactory(_Client(provider="remote", redirect=True)),
            synthesizer=lambda fixtures: synthesis_calls.append(fixtures),
            requester=lambda *args: request_calls.append(args),
            credential_reader=lambda name: credential_reads.append(name),
        )

    assert caught.value.attempted == 0
    assert credential_reads == []
    assert synthesis_calls == []
    assert request_calls == []


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
def test_local_target_rejects_every_noncanonical_model_route(unsafe_base):
    with pytest.raises(probe.UntrustedManagedTarget):
        probe._resolve_managed_target(_Client(local_base=unsafe_base))


def test_matrix_uses_one_audio_only_request_per_cell_and_reports_only_aggregate_evidence():
    calls = []
    buffers = {}
    result = probe.run_isolation_probe(
        client_factory=_ClientFactory(_Client()),
        synthesizer=_provider_with_capture(buffers),
        requester=_successful_requester(calls),
        credential_reader=lambda _name: None,
    )

    cells = probe.probe_cells()
    assert len(calls) == len(cells) == 16
    for cell, (_target, _credential, payload, _timeout) in zip(cells, calls, strict=True):
        expected_roles = ["system", "user"] if cell.history_arm == "fresh" else ["system", "user", "assistant", "user"]
        assert [message["role"] for message in payload["messages"]] == expected_roles
        assert len(_audio_parts(payload)) == 1
        assert _audio_parts(payload)[0]["input_audio"]["format"] == "wav"
        assert payload["stream"] is False
        assert "tools" not in payload
        assert "tool_choice" not in payload
        if cell.prompt_arm == "full_production":
            system = payload["messages"][0]["content"]
            assert "USER_MEMORY:" in system
            assert "ASSISTANT_RESPONSE:" in system
        else:
            assert payload["messages"][0]["content"] == probe._SHORT_SYSTEM_PROMPT

    assert result == {
        "gate_passed": True,
        "fixture_count": 4,
        "planned_cell_count": 16,
        "synthesis_success_count": 4,
        "voice_match_count": 4,
        "request_attempted_count": 16,
        "request_completed_count": 16,
        "request_success_count": 16,
        "schema_success_count": 16,
        "exact_label_count": 16,
        "correction_attempt_count": 8,
        "correction_success_count": 8,
        "one_request_per_cell": True,
        "timing": result["timing"],
        "request_shape": {
            "shape_valid_count": 16,
            "input_audio_part_count": 16,
            "text_current_user_part_count": 0,
            "fresh_history_message_count": 0,
            "wrong_history_message_count": 16,
            "short_prompt_cell_count": 8,
            "isolated_production_payload_cell_count": 8,
            "direct_endpoint_cell_count": 16,
            "managed_pipeline_replay_count": 0,
            "tool_surface_cell_count": 0,
            "streaming_cell_count": 0,
            "response_tts_execution_count": 0,
        },
        "arms": result["arms"],
    }
    assert set(result["arms"]) == {
        "fresh_short",
        "wrong_semantic_short",
        "wrong_semantic_full_production",
        "fresh_full_production",
    }
    assert all(arm["request_success_count"] == 4 for arm in result["arms"].values())
    assert all(arm["exact_label_count"] == 4 for arm in result["arms"].values())
    assert all(not any(item.wav) for item in buffers.values())

    rendered = json.dumps(result, sort_keys=True)
    forbidden = [
        probe._SHORT_SYSTEM_PROMPT,
        probe._PRODUCTION_BASE_PROMPT,
        probe._PRODUCTION_SESSION_PROMPT,
        probe._LOCAL_MODEL_BASE_URL,
        "private-local-model",
        "private response memory",
        *[fixture.utterance for fixture in probe._FIXTURES],
        *[fixture.wrong_memory for fixture in probe._FIXTURES],
    ]
    assert all(value not in rendered for value in forbidden)
    assert all(f'"{fixture.expected_label}"' not in rendered for fixture in probe._FIXTURES)
    assert all(f'"{fixture.wrong_label}"' not in rendered for fixture in probe._FIXTURES)


def test_wrong_history_failures_are_isolated_as_correction_metrics():
    calls = []
    result = probe.run_isolation_probe(
        client_factory=_ClientFactory(_Client()),
        synthesizer=lambda _fixtures: _synthesized_fixtures(),
        requester=_successful_requester(calls, wrong_history_fails=True),
        credential_reader=lambda _name: None,
    )

    assert result["gate_passed"] is False
    assert result["exact_label_count"] == 8
    assert result["correction_attempt_count"] == 8
    assert result["correction_success_count"] == 0
    assert result["arms"]["fresh_short"]["exact_label_count"] == 4
    assert result["arms"]["fresh_full_production"]["exact_label_count"] == 4
    assert result["arms"]["wrong_semantic_short"]["exact_label_count"] == 0
    assert result["arms"]["wrong_semantic_full_production"]["exact_label_count"] == 0


def test_unrecognized_response_content_is_discarded_and_never_enters_report():
    response_secret = "PRIVATE_RESPONSE_CONTENT_MUST_NOT_APPEAR"
    returned = []

    def requester(*_args):
        data = {"choices": [{"message": {"content": response_secret}}]}
        returned.append(data)
        return data

    result = probe.run_isolation_probe(
        client_factory=_ClientFactory(_Client()),
        synthesizer=lambda _fixtures: _synthesized_fixtures(),
        requester=requester,
        credential_reader=lambda _name: None,
    )

    assert result["gate_passed"] is False
    assert result["schema_success_count"] == 16
    assert result["exact_label_count"] == 0
    assert response_secret not in json.dumps(result)
    assert returned and all(data == {} for data in returned)


def test_request_failure_is_one_shot_private_and_wipes_all_audio_buffers():
    calls = []
    buffers = {}

    def requester(*_args):
        calls.append(1)
        raise RuntimeError("PRIVATE_ENDPOINT PRIVATE_RESPONSE PRIVATE_KEY")

    with pytest.raises(probe.ProbeProgressError) as caught:
        probe.run_isolation_probe(
            client_factory=_ClientFactory(_Client()),
            synthesizer=_provider_with_capture(buffers),
            requester=requester,
            credential_reader=lambda _name: None,
        )

    assert calls == [1]
    assert caught.value.attempted == 1
    assert caught.value.completed == 0
    assert caught.value.shape_valid == 1
    assert all(not any(item.wav) for item in buffers.values())
    public = probe._failure_report(caught.value, total_ms=7.0)
    assert public["request_attempted_count"] == 1
    assert public["request_completed_count"] == 0
    assert public["request_shape"]["shape_valid_count"] == 1
    assert "PRIVATE" not in json.dumps(public)


def test_invalid_injected_payload_is_rejected_before_any_model_request():
    request_calls = []
    buffers = {}

    def invalid_builder(model, cell, encoded_wav):
        payload = probe.build_cell_payload(model, cell, encoded_wav)
        payload["tool_choice"] = "auto"
        return payload

    with pytest.raises(probe.ProbeProgressError) as caught:
        probe.run_isolation_probe(
            client_factory=_ClientFactory(_Client()),
            synthesizer=_provider_with_capture(buffers),
            requester=lambda *args: request_calls.append(args),
            payload_builder=invalid_builder,
            credential_reader=lambda _name: None,
        )

    assert caught.value.attempted == 0
    assert caught.value.shape_valid == 0
    assert request_calls == []
    assert all(not any(item.wav) for item in buffers.values())


def test_main_emits_one_content_free_json_object_on_failure(monkeypatch, capsys):
    monkeypatch.setattr(
        probe,
        "run_isolation_probe",
        lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("PRIVATE_PROMPT PRIVATE_AUDIO PRIVATE_RESPONSE PRIVATE_ENDPOINT PRIVATE_MODEL PRIVATE_KEY")
        ),
    )

    exit_code = probe.main([])
    captured = capsys.readouterr()

    assert exit_code != 0
    assert captured.err == ""
    assert len(captured.out.splitlines()) == 1
    payload = json.loads(captured.out)
    assert payload["gate_passed"] is False
    assert payload["request_attempted_count"] == 0
    assert payload["request_completed_count"] == 0
    assert "PRIVATE" not in captured.out
