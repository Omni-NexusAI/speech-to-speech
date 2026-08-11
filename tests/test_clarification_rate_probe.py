from __future__ import annotations

import base64
import importlib.util
import io
import json
import logging
import math
import pathlib
import struct
import subprocess
import sys
import wave

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "probe_clarification_rate.py"
SPEC = importlib.util.spec_from_file_location("clarification_rate_probe", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def _wav(samples: list[int] | None = None) -> bytes:
    values = samples or [int(7000 * math.sin(2 * math.pi * 220 * index / 16000)) for index in range(320)]
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16000)
        target.writeframes(struct.pack(f"<{len(values)}h", *values))
    return output.getvalue()


class _Response:
    def __init__(self, data, *, history=None, url=None):
        self._data = data
        self.history = list(history or [])
        self.url = url or probe._SETTINGS_URL

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


class _Client:
    def __init__(self, *args, **kwargs):
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def get(self, _target):
        return _Response(
            {
                "settings": {
                    "modelProvider": "remote",
                    "modelUrl": "http://private.invalid/v1",
                    "modelName": "private-model",
                }
            }
        )


def _synthesized(scenarios):
    raw = _wav()
    return {
        scenario.index: probe.SynthesizedAudio(
            wav=raw,
            voice_enumeration_available=True,
            culture_matched=True,
            alternate_selected=True,
        )
        for scenario in scenarios
    }


def _response(text: str):
    return {"choices": [{"message": {"content": text}}]}


def test_corpus_has_exact_normal_cohorts_and_separate_meaningless_audio():
    normal, meaningless = probe.build_scenarios()

    assert len(normal) == 100
    assert len(meaningless) == 8
    assert {scenario.category for scenario in normal} == set(probe._CATEGORIES)
    assert all(sum(s.category == category for s in normal) == 20 for category in probe._CATEGORIES)
    assert all(s.normal for s in normal)
    assert all(not s.normal for s in meaningless)
    assert len({s.index for s in [*normal, *meaningless]}) == 108
    assert len({s.expected_number for s in normal}) == 100
    assert len({s.utterance for s in normal}) == 100


def test_multilingual_fixtures_survive_utf8_json_round_trip_by_codepoint():
    normal, _ = probe.build_scenarios()
    fixtures = [scenario.utterance for scenario in normal if scenario.category == "code_switch"]
    encoded = json.dumps(fixtures, ensure_ascii=False).encode("utf-8")
    decoded = json.loads(encoded.decode("utf-8"))

    assert decoded == fixtures
    assert any(any(ord(character) > 0x3000 for character in fixture) for fixture in fixtures)


def test_sapi_synthesis_uses_stdin_and_keeps_prompts_and_voices_out_of_argv():
    normal, _ = probe.build_scenarios()
    scenarios = normal[:3]
    captured = {"commands": []}
    raw = base64.b64encode(_wav()).decode("ascii")

    def runner(argv, **kwargs):
        captured["commands"].append(" ".join(argv))
        if "GetInstalledVoices" in captured["commands"][-1]:
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout=json.dumps({"available": True, "voices": []}).encode("utf-8"),
                stderr=b"",
            )
        captured["argv"] = list(argv)
        captured["input"] = kwargs["input"]
        payload = [
            {
                "wav": raw,
                "voice_enumeration_available": True,
                "culture_matched": True,
                "alternate_selected": index == 1,
            }
            for index, _scenario in enumerate(scenarios)
        ]
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps(payload).encode("utf-8"),
            stderr="private child failure text".encode("utf-8"),
        )

    result = probe.synthesize_scenarios_in_memory(scenarios, runner=runner)

    command = " ".join(captured["argv"])
    assert len(captured["commands"]) == 2
    assert "GetInstalledVoices" in captured["commands"][0]
    assert "GetInstalledVoices" not in captured["commands"][1]
    assert all(scenario.utterance not in command for scenario in scenarios)
    assert all(scenario.culture not in command for scenario in scenarios)
    assert isinstance(captured["input"], bytes)
    assert all(scenario.utterance in captured["input"].decode("utf-8") for scenario in scenarios)
    assert "InputEncoding" in command
    assert "UTF8Encoding" in command
    assert len(result) == len(scenarios)
    assert all(item.wav.startswith(b"RIFF") for item in result.values())


def test_sapi_contract_uses_disposable_catalog_and_fresh_synth_per_item():
    normal, _ = probe.build_scenarios()
    scenarios = [
        next(item for item in normal if item.category == "alternate_voice"),
        next(item for item in normal if item.category == "clear"),
    ]
    raw = base64.b64encode(_wav()).decode("ascii")

    commands = []

    def runner(argv, **kwargs):
        command = " ".join(argv)
        commands.append(command)
        if "GetInstalledVoices" in command:
            assert "$catalogSynth.Dispose()" in command
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout=json.dumps(
                    {"available": True, "voices": [{"name": "fixture", "culture": scenarios[0].culture}]}
                ).encode("utf-8"),
                stderr=b"",
            )
        assert "GetInstalledVoices" not in command
        loop = command.index("foreach ($item in @($items))")
        per_item = command.index("$synth = New-Object", loop)
        per_item_dispose = command.index("$synth.Dispose()", per_item)
        assert loop < per_item < per_item_dispose
        assert "$synth.SelectVoice($matches[$slot].name)" in command
        request = json.loads(kwargs["input"].decode("utf-8"))
        assert request["voice_enumeration_available"] is True
        assert request["voices"] == [{"name": "fixture", "culture": scenarios[0].culture}]
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps(
                [
                    {
                        "wav": raw,
                        "voice_enumeration_available": True,
                        "culture_matched": True,
                        "alternate_selected": False,
                    },
                    {
                        "wav": raw,
                        "voice_enumeration_available": True,
                        "culture_matched": False,
                        "alternate_selected": False,
                    },
                ]
            ).encode("utf-8"),
            stderr=b"",
        )

    result = probe.synthesize_scenarios_in_memory(scenarios, runner=runner)

    assert len(commands) == 2
    assert result[scenarios[0].index].culture_matched is True
    assert result[scenarios[1].index].culture_matched is False
    assert result[scenarios[1].index].alternate_selected is False


def test_failed_catalog_child_uses_clean_default_synthesis_child():
    normal, _ = probe.build_scenarios()
    scenario = normal[0]
    raw = base64.b64encode(_wav()).decode("ascii")
    calls = 0

    def runner(argv, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            assert "GetInstalledVoices" in " ".join(argv)
            return subprocess.CompletedProcess(argv, 1, stdout=b"", stderr=b"private catalog failure")
        assert "GetInstalledVoices" not in " ".join(argv)
        request = json.loads(kwargs["input"].decode("utf-8"))
        assert request["voice_enumeration_available"] is False
        assert request["voices"] == []
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps(
                [
                    {
                        "wav": raw,
                        "voice_enumeration_available": False,
                        "culture_matched": False,
                        "alternate_selected": False,
                    }
                ]
            ).encode("utf-8"),
            stderr=b"",
        )

    result = probe.synthesize_scenarios_in_memory([scenario], runner=runner)

    assert calls == 2
    assert result[scenario.index].wav.startswith(b"RIFF")
    assert result[scenario.index].voice_enumeration_available is False
    assert result[scenario.index].culture_matched is False
    assert result[scenario.index].alternate_selected is False


@pytest.mark.skipif(sys.platform != "win32", reason="Windows System.Speech contract")
def test_real_sapi_runner_accepts_multilingual_utf8_stdin_in_memory():
    normal, _ = probe.build_scenarios()
    scenario = next(
        item
        for item in normal
        if item.category == "code_switch" and any(ord(character) > 0x3000 for character in item.utterance)
    )

    result = probe.synthesize_scenarios_in_memory([scenario])

    assert set(result) == {scenario.index}
    assert result[scenario.index].wav.startswith(b"RIFF")
    if not result[scenario.index].voice_enumeration_available:
        assert result[scenario.index].culture_matched is False
        assert result[scenario.index].alternate_selected is False


def test_noise_is_seeded_in_memory_and_keeps_wave_shape():
    raw = _wav()
    first = probe.add_deterministic_noise(raw, seed=101)
    repeated = probe.add_deterministic_noise(raw, seed=101)
    different = probe.add_deterministic_noise(raw, seed=102)

    assert first == repeated
    assert first != different
    with wave.open(io.BytesIO(first), "rb") as result, wave.open(io.BytesIO(raw), "rb") as source:
        assert result.getparams() == source.getparams()
        assert result.getnframes() == source.getnframes()


def test_classifier_prioritizes_garbling_and_clarification_over_answer_recovery():
    assert probe.classify_visible_response("The result is 11.", 11) == "ordinary"
    assert probe.classify_visible_response("I could not hear that; perhaps 11.", 11) == "garbling_commentary"
    assert probe.classify_visible_response("Could you repeat that?", 11) == "clarification"
    assert probe.classify_visible_response("Could you repeat that? I think it is 11.", 11) == "clarification"
    assert probe.classify_visible_response("I can help with several things.", 11) == "unclassified"


def test_production_parser_prefers_transcript_then_memory_and_recovers_visible_response():
    handler = probe._handler("private-model")
    both = probe.parse_primary_response(
        handler,
        _response("USER_TRANSCRIPT: exact text\nUSER_MEMORY: semantic memory\nASSISTANT_RESPONSE: 11"),
    )
    memory = probe.parse_primary_response(
        handler,
        _response("USER_MEMORY: semantic memory\nASSISTANT_RESPONSE: 6"),
    )
    absent = probe.parse_primary_response(handler, _response("8"))

    assert both.transcript == "exact text"
    assert both.memory is None
    assert both.visible == "11"
    assert memory.transcript is None
    assert memory.memory == "semantic memory"
    assert memory.visible == "6"
    assert absent.transcript is None
    assert absent.memory is None
    assert absent.visible == "8"


def test_primary_payload_uses_one_current_audio_and_serializes_prior_semantic_turn():
    handler = probe._handler("private-model")
    chat = probe.Chat(30)
    first_audio = base64.b64encode(_wav()).decode("ascii")
    first = probe.build_primary_payload(handler, chat, first_audio)

    assert [message["role"] for message in first["messages"]] == ["system", "user"]
    assert sum(
        part.get("type") == "input_audio"
        for message in first["messages"]
        for part in message.get("content", [])
        if isinstance(part, dict)
    ) == 1

    user = chat.add_item(probe.make_user_audio_message(first_audio))
    assert user.id is not None
    assert chat.replace_user_message_text(user.id, "durable semantic memory")
    chat.commit_assistant_response(user.id, "11", [])
    second_audio = base64.b64encode(_wav([1000, -1000] * 1600)).decode("ascii")
    second = probe.build_primary_payload(handler, chat, second_audio)

    assert [message["role"] for message in second["messages"]] == ["system", "user", "assistant", "user"]
    current_audio_parts = [
        part
        for part in second["messages"][-1]["content"]
        if isinstance(part, dict) and part.get("type") == "input_audio"
    ]
    assert len(current_audio_parts) == 1
    assert current_audio_parts[0]["input_audio"]["data"] == second_audio


def test_gate_counts_one_primary_per_turn_and_history_anchor_kinds(monkeypatch):
    monkeypatch.setenv(probe._CREDENTIAL_ENV, "private-key")
    normal, meaningless = probe.build_scenarios()
    scenarios = [*normal, *meaningless]
    calls = 0

    def requester(_client, _target, _credential, _payload, _timeout):
        nonlocal calls
        scenario = scenarios[calls]
        calls += 1
        if not scenario.normal:
            return _response("Could you repeat that?")
        answer = str(scenario.expected_number)
        if scenario.index % 3 == 0:
            return _response(f"USER_TRANSCRIPT: accepted utterance {scenario.index}\nASSISTANT_RESPONSE: {answer}")
        if scenario.index % 3 == 1:
            return _response(f"USER_MEMORY: semantic intent {scenario.index}\nASSISTANT_RESPONSE: {answer}")
        return _response(answer)

    report = probe.run_gate(
        client_factory=_Client,
        synthesizer=_synthesized,
        requester=requester,
    )

    assert calls == 108
    assert report["primary_requests"] == 108
    assert report["primary_requests_attempted"] == 108
    assert report["primary_requests_completed"] == 108
    assert report["one_primary_per_turn"] is True
    assert report["ordinary"] == 100
    assert report["clarifications"] == 0
    assert report["garbling_commentary"] == 0
    assert report["unclassified"] == 0
    assert report["transcript_anchors"] == 34
    assert report["memory_anchors"] == 33
    assert report["audio_anchors"] == 41
    assert report["meaningless_counts"]["clarification"] == 8
    assert report["gate_passed"] is True
    public_output = json.dumps(report, sort_keys=True)
    assert all(scenario.utterance not in public_output for scenario in scenarios)
    assert "private-key" not in public_output
    assert "private-model" not in public_output
    assert "private.invalid" not in public_output


def test_unclassified_normal_response_fails_gate_instead_of_becoming_ordinary(monkeypatch):
    monkeypatch.setenv(probe._CREDENTIAL_ENV, "private-key")
    normal, meaningless = probe.build_scenarios()
    scenarios = [*normal, *meaningless]
    calls = 0

    def requester(_client, _target, _credential, _payload, _timeout):
        nonlocal calls
        scenario = scenarios[calls]
        calls += 1
        if scenario.index == 5:
            return _response("A substantive but non-matching answer.")
        if scenario.expected_number is not None:
            return _response(str(scenario.expected_number))
        return _response("Could you repeat that?")

    report = probe.run_gate(client_factory=_Client, synthesizer=_synthesized, requester=requester)

    assert report["unclassified"] == 1
    assert report["classification_complete"] is False
    assert report["gate_passed"] is False


def test_public_failure_output_contains_only_error_class_not_exception_content(monkeypatch, capsys):
    secret_values = (
        "private prompt",
        "private response",
        "private endpoint",
        "private model",
        "private voice",
        "private key",
    )

    def fail(**_kwargs):
        raise RuntimeError(" ".join(secret_values))

    monkeypatch.setattr(probe, "run_gate", fail)
    assert probe.main([]) == 1
    output = capsys.readouterr().out
    data = json.loads(output)

    assert data["error_class"] == "RuntimeError"
    assert all(value not in output for value in secret_values)
    assert set(data) == {
        "gate_passed",
        "normal_turns",
        "meaningless_turns",
        "primary_requests",
        "primary_requests_attempted",
        "primary_requests_completed",
        "error_class",
    }


def test_main_emits_one_json_object_and_scopes_chat_logger_suppression(monkeypatch, capsys):
    report = {"gate_passed": True, "normal_turns": 100, "primary_requests": 108}
    chat_logger = logging.getLogger("speech_to_speech.LLM.chat")
    previous_disabled = chat_logger.disabled
    previous_propagate = chat_logger.propagate
    handler = logging.StreamHandler()
    chat_logger.disabled = False
    chat_logger.propagate = False
    chat_logger.addHandler(handler)

    def fake_run_gate(**_kwargs):
        chat_logger.warning("Chat buffer exceeded hard cap")
        return report

    monkeypatch.setattr(probe, "_run_gate", fake_run_gate)
    try:
        assert probe.main([]) == 0
        assert chat_logger.disabled is False
    finally:
        chat_logger.removeHandler(handler)
        chat_logger.disabled = previous_disabled
        chat_logger.propagate = previous_propagate

    captured = capsys.readouterr()
    expected = json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n"
    assert captured.out == expected
    assert captured.err == ""
    assert json.loads(captured.out) == report


def test_missing_exact_credential_environment_never_starts_synthesis(monkeypatch):
    monkeypatch.delenv(probe._CREDENTIAL_ENV, raising=False)
    called = False

    def synthesizer(_scenarios):
        nonlocal called
        called = True
        return {}

    with pytest.raises(probe.CredentialUnavailable):
        probe.run_gate(client_factory=_Client, synthesizer=synthesizer)
    assert called is False


def test_settings_redirect_is_rejected_before_synthesis_or_model_request(monkeypatch):
    monkeypatch.setenv(probe._CREDENTIAL_ENV, "private-key")
    synthesized = False
    requested = False

    class RedirectingClient(_Client):
        def get(self, _target):
            return _Response(
                {"settings": {"modelProvider": "remote", "modelUrl": "http://private.invalid/v1", "modelName": "private-model"}},
                history=[object()],
            )

    def synthesizer(_scenarios):
        nonlocal synthesized
        synthesized = True
        return {}

    def requester(*_args):
        nonlocal requested
        requested = True
        return {}

    with pytest.raises(probe.UntrustedManagedTarget):
        probe.run_gate(client_factory=RedirectingClient, synthesizer=synthesizer, requester=requester)
    assert synthesized is False
    assert requested is False


def test_partial_transport_failure_reports_exact_attempted_and_completed_counts(monkeypatch):
    monkeypatch.setenv(probe._CREDENTIAL_ENV, "private-key")
    normal, meaningless = probe.build_scenarios()
    scenarios = [*normal, *meaningless]
    calls = 0

    def requester(_client, _target, _credential, _payload, _timeout):
        nonlocal calls
        calls += 1
        if calls == 7:
            raise RuntimeError("private transport detail")
        return _response(str(scenarios[calls - 1].expected_number))

    with pytest.raises(probe.ProbeProgressError) as caught:
        probe.run_gate(client_factory=_Client, synthesizer=_synthesized, requester=requester)

    report = probe._failure(caught.value)
    encoded = json.dumps(report, sort_keys=True)
    assert report["primary_requests_attempted"] == 7
    assert report["primary_requests_completed"] == 6
    assert report["primary_requests"] == 6
    assert report["error_class"] == "RuntimeError"
    assert "private transport detail" not in encoded
