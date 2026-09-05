"""Unit coverage for the bounded non-Gemma HFRT TTS-handler harness."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture()
def harness_module(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("validate_hfrt_tts_handler_path_test", scripts / "validate_hfrt_tts_handler_path.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(tmp_path: Path, *, audio_dir: Path | None = None):
    return argparse.Namespace(
        candidate="http://127.0.0.1:8890",
        voice="clone:fixed-candidate",
        profile="balanced",
        profile_revision=7,
        seed=321,
        language="English",
        delivery="buffered_phrase",
        text="Count slowly from one to five. Then count backwards from five to one.",
        audio_dir=audio_dir,
    )


def _effective(module):
    return {
        "model": None,
        "clone_mode": "full_icl",
        "max_reference_seconds": 20,
        "first_block_frames": 4,
        "steady_block_frames": 12,
        "left_context_frames": 25,
        "text_lookahead": 64,
        "phrase_flush_ms": 500,
        "temperature": 0.7,
        "top_k": 40,
        "top_p": 0.9,
        "repetition_penalty": 1.05,
        "seed": 0,
    }


def _snapshot(*, model="qwen3-tts-1.7b-base-bf16", epoch="11", instance="candidate-11"):
    return SimpleNamespace(
        model=model,
        model_epoch=epoch,
        model_instance_id=instance,
        profile_id="balanced",
        profile_revision=7,
        seed=321,
        language="English",
        delivery_streaming_mode="buffered_phrase",
        delivery_sample_rate=24000,
        clone_content_revision=3,
        clone_content_hash="b" * 64,
    )


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return self

    def json(self):
        return self.payload


class _Client:
    def __init__(self, responses, calls, **_kwargs):
        self.responses = responses
        self.calls = calls

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def get(self, url):
        self.calls.append(("GET", url))
        return _Response(self.responses[url])


def _install_candidate(monkeypatch, module, *, health=None):
    calls = []
    health = health or {
        "state": "loaded", "singleResident": True,
        "activeModel": "qwen3-tts-1.7b-base-bf16", "engineEpoch": 11,
        "supervisorInstanceId": "candidate-11",
    }
    profile = {"revision": 7, **_effective(module)}
    responses = {
        "http://127.0.0.1:8890/health": health,
        "http://127.0.0.1:8890/v1/tuning/profiles": {"profiles": {"balanced": profile}},
    }
    monkeypatch.setattr(module.httpx, "Client", lambda **kwargs: _Client(responses, calls, **kwargs))
    return calls


def test_runs_one_fixed_ttsinput_through_process_without_mutation(harness_module, monkeypatch, tmp_path):
    module = harness_module
    calls = _install_candidate(monkeypatch, module)
    seen = {}
    audio_dir = tmp_path / "export"
    replace_calls = []
    original_replace = module.os.replace

    def record_replace(source, destination):
        replace_calls.append((source, destination))
        return original_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", record_replace)

    class Handler:
        def __init__(self, *_args, **kwargs):
            seen["handler_init"] = kwargs
            self._last_tts_outcome = {"candidate_request_id": "a" * 32, "engine_eos": True}
            self.metrics = kwargs["setup_kwargs"]["text_output_queue"]

        def _response_synthesis_snapshot(self, item):
            seen["snapshot_item"] = item
            return _snapshot()

        def process(self, item):
            seen["process_item"] = item
            self.metrics.put(SimpleNamespace(
                stage="tts", status="done", elapsed_ms=12.5,
                detail={
                    "generation_ms": 12.5, "source_sample_rate": 24000,
                    "engine_eos": True, "candidate_request_id": "a" * 32,
                },
            ))
            yield module.AudioOutput(audio=np.array([1, 2, 3], dtype=np.int16), source_sample_rate=24000)

    monkeypatch.setattr(module, "Qwen3TTSHandler", Handler)
    report = module.run(_args(tmp_path, audio_dir=audio_dir))

    item = seen["process_item"]
    assert item is seen["snapshot_item"]
    assert item.text == _args(tmp_path).text
    assert item.runtime_config.local_pipeline["tts_backend"] == "qwen3tts-audiocpp"
    assert item.runtime_config.response_synthesis_configs[1]["voice"] == "clone:fixed-candidate"
    assert seen["handler_init"]["setup_kwargs"]["backend"] == "openai-api"
    assert calls == [
        ("GET", "http://127.0.0.1:8890/health"),
        ("GET", "http://127.0.0.1:8890/v1/tuning/profiles"),
    ]
    assert report["audio"] == {"samples": 3, "source_sample_rate": 24000, "duration_ms": 0.125, "exported": True}
    assert report["outcome"]["candidate_request_id"] == "a" * 32
    assert report["outcome"]["completion_proven"] is True
    assert report["identity"]["snapshot_engine_epoch"] == "11"
    final_audio = audio_dir / "tts-handler-acceptance.pcm"
    assert final_audio.read_bytes() == np.array([1, 2, 3], dtype="<i2").tobytes()
    assert len(replace_calls) == 1
    assert replace_calls[0][1] == final_audio
    assert replace_calls[0][0].parent == audio_dir
    assert not replace_calls[0][0].exists()


def test_rejects_nonresident_candidate_before_handler_construction(harness_module, monkeypatch, tmp_path):
    module = harness_module
    calls = _install_candidate(monkeypatch, module, health={"state": "loading", "singleResident": False})
    monkeypatch.setattr(module, "Qwen3TTSHandler", lambda: pytest.fail("unhealthy candidate must not construct handler"))

    with pytest.raises(RuntimeError, match="candidate_unavailable"):
        module.run(_args(tmp_path))
    assert calls == [("GET", "http://127.0.0.1:8890/health"), ("GET", "http://127.0.0.1:8890/v1/tuning/profiles")]


def test_fails_closed_when_handler_does_not_emit_completed_audio(harness_module, monkeypatch, tmp_path):
    module = harness_module
    _install_candidate(monkeypatch, module)

    class FailedHandler:
        def __init__(self, *_args, **kwargs):
            self._last_tts_outcome = {}
            self.metrics = kwargs["setup_kwargs"]["text_output_queue"]

        def _response_synthesis_snapshot(self, _item):
            return _snapshot()

        def process(self, item):
            self.metrics.put(SimpleNamespace(
                stage="tts", status="failed", elapsed_ms=1,
                detail={"reason": "engine-error", "error": "Bearer must-not-leak"},
            ))
            return iter(())

    monkeypatch.setattr(module, "Qwen3TTSHandler", FailedHandler)

    audio_dir = tmp_path / "failed-export"
    with pytest.raises(RuntimeError, match="engine_error") as error:
        module.run(_args(tmp_path, audio_dir=audio_dir))
    assert "must-not-leak" not in str(error.value)
    assert not (audio_dir / "tts-handler-acceptance.pcm").exists()
    assert list(audio_dir.iterdir()) == []


def test_rejects_existing_export_before_any_candidate_dispatch(harness_module, monkeypatch, tmp_path):
    module = harness_module
    audio_dir = tmp_path / "existing-export"
    audio_dir.mkdir()
    final_audio = audio_dir / "tts-handler-acceptance.pcm"
    final_audio.write_bytes(b"prior-success")
    monkeypatch.setattr(module, "_resident_profile", lambda _args: pytest.fail("existing export must prevent dispatch"))

    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        module.run(_args(tmp_path, audio_dir=audio_dir))
    assert final_audio.read_bytes() == b"prior-success"


def test_clears_inherited_api_key_before_snapshot_headers(harness_module, monkeypatch, tmp_path):
    module = harness_module
    _install_candidate(monkeypatch, module)
    monkeypatch.setenv("QWEN3_TTS_API_KEY", "must-not-leak")
    monkeypatch.setenv("QWEN3_TTS_API_BASE_URL", "http://unrelated.invalid/v1")
    seen = {}
    production_handler = module.Qwen3TTSHandler

    class HeaderCheckingHandler(production_handler):
        def _response_synthesis_snapshot(self, _item):
            seen["headers"] = self._openai_api_headers()
            seen["api_base_url"] = self.api_base_url
            return _snapshot()

        def process(self, _item):
            self._last_tts_outcome = {"candidate_request_id": "a" * 32, "engine_eos": True}
            self.text_output_queue.put(SimpleNamespace(
                stage="tts", status="done", elapsed_ms=1,
                detail={"candidate_request_id": "a" * 32, "engine_eos": True},
            ))
            yield module.AudioOutput(audio=np.array([1], dtype=np.int16), source_sample_rate=24000)

    monkeypatch.setattr(module, "Qwen3TTSHandler", HeaderCheckingHandler)
    module.run(_args(tmp_path))

    assert "Authorization" not in seen["headers"]
    assert seen["api_base_url"] == "http://127.0.0.1:8890/v1"


@pytest.mark.parametrize(
    "snapshot_kwargs",
    [
        {"model": "qwen3-tts-0.6b-base-bf16"},
        {"epoch": "12"},
        {"instance": "candidate-reloaded"},
    ],
)
def test_rejects_snapshot_identity_changed_after_admission(harness_module, monkeypatch, tmp_path, snapshot_kwargs):
    module = harness_module
    _install_candidate(monkeypatch, module)

    class RacedHandler:
        def __init__(self, *_args, **_kwargs):
            pass

        def _response_synthesis_snapshot(self, _item):
            return _snapshot(**snapshot_kwargs)

        def process(self, _item):
            pytest.fail("raced snapshot must not dispatch synthesis")

    monkeypatch.setattr(module, "Qwen3TTSHandler", RacedHandler)
    with pytest.raises(RuntimeError, match="lifecycle_changed"):
        module.run(_args(tmp_path))


@pytest.mark.parametrize(
    ("outcome", "detail"),
    [
        ({"candidate_request_id": "a" * 32, "engine_eos": False}, {"candidate_request_id": "a" * 32, "engine_eos": False}),
        ({"candidate_request_id": "a" * 32, "engine_eos": True}, {"candidate_request_id": "b" * 32, "engine_eos": True}),
        ({"candidate_request_id": "not-a-valid-id", "engine_eos": True}, {"candidate_request_id": "not-a-valid-id", "engine_eos": True}),
    ],
)
def test_requires_correlated_completed_engine_outcome(harness_module, monkeypatch, tmp_path, outcome, detail):
    module = harness_module
    _install_candidate(monkeypatch, module)

    class IncompleteOutcomeHandler:
        def __init__(self, *_args, **kwargs):
            self._last_tts_outcome = outcome
            self.metrics = kwargs["setup_kwargs"]["text_output_queue"]

        def _response_synthesis_snapshot(self, _item):
            return _snapshot()

        def process(self, _item):
            self.metrics.put(SimpleNamespace(stage="tts", status="done", elapsed_ms=1, detail=detail))
            yield module.AudioOutput(audio=np.array([1], dtype=np.int16), source_sample_rate=24000)

    monkeypatch.setattr(module, "Qwen3TTSHandler", IncompleteOutcomeHandler)
    with pytest.raises(RuntimeError, match="completion_unverified"):
        module.run(_args(tmp_path))


@pytest.mark.parametrize(
    ("provider_error", "expected_code"),
    [
        ("audio.cpp synthesis failed: wall-timeout", "wall_timeout"),
        ("audio.cpp synthesis failed: engine-error", "engine_error"),
        ("transport rejected Authorization: secret-value", "unspecified"),
    ],
)
def test_generic_handler_failure_is_allowlisted_without_raw_provider_text(
    harness_module, monkeypatch, tmp_path, provider_error, expected_code,
):
    module = harness_module
    _install_candidate(monkeypatch, module)

    class RaisingHandler:
        def __init__(self, *_args, **_kwargs):
            pass

        def _response_synthesis_snapshot(self, _item):
            return _snapshot()

        def process(self, _item):
            raise RuntimeError(provider_error)

    monkeypatch.setattr(module, "Qwen3TTSHandler", RaisingHandler)
    with pytest.raises(RuntimeError, match=expected_code) as error:
        module.run(_args(tmp_path))
    assert provider_error not in str(error.value)


@pytest.mark.parametrize(
    ("message", "expected_code"),
    [
        ("Selected candidate profile/revision is not currently available", "profile_unavailable"),
        ("Selected candidate profile has no complete effective tuning snapshot", "profile_incomplete"),
        ("HFRT TTS handler did not complete audio: {'detail': {'reason': 'stale_before_dispatch'}}", "lifecycle_changed"),
        ("HFRT TTS handler did not complete audio: {'detail': {'reason': 'native_stream_error'}}", "completion_unverified"),
    ],
)
def test_failure_classifier_covers_harness_preflight_and_sanitized_metric_reasons(
    harness_module, message, expected_code,
):
    assert harness_module._safe_failure_code(RuntimeError(message)) == expected_code
