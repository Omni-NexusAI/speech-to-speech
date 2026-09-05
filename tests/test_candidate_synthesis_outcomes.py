"""Content-free late failure lookup and shared generated-audio bounds."""
import asyncio
import base64
import json
import importlib.util
from pathlib import Path

import httpx
import pytest

def load_supervisor(monkeypatch, tmp_path):
    config = tmp_path / "models.json"
    config.write_text(json.dumps({"models": [{"id": "qwen3-tts-1.7b-base-bf16"}]}))
    monkeypatch.setenv("AUDIO_CPP_CONFIG_TEMPLATE", str(config))
    monkeypatch.setenv("VOICE_LIBRARY_DIR", str(tmp_path / "voices"))
    path = Path(__file__).parents[1] / "integrations/audio-cpp/supervisor.py"
    spec = importlib.util.spec_from_file_location("outcome_test_supervisor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_outcomes_are_bounded_expiring_content_free_and_terminal(monkeypatch, tmp_path):
    supervisor = load_supervisor(monkeypatch, tmp_path)
    clock = [1000.0]
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: clock[0])
    payload = {"request_id": "private words", "input": "private words", "ref_audio": "secret"}
    key = supervisor._begin_request_outcome(payload, "model", "buffered-fallback")
    supervisor._finish_request_outcome(key, "limited", "max_generated_frames", input="not permitted")
    supervisor._finish_request_outcome(key, "completed")
    assert supervisor.REQUEST_OUTCOMES[key]["state"] == "limited"
    assert "private words" not in json.dumps(supervisor.REQUEST_OUTCOMES)
    assert "secret" not in json.dumps(supervisor.REQUEST_OUTCOMES)
    assert len(key) == 32
    for _ in range(260):
        supervisor._begin_request_outcome({}, "model", "buffered-fallback")
    assert len(supervisor.REQUEST_OUTCOMES) == supervisor.OUTCOME_CAPACITY
    clock[0] += 601
    supervisor._prune_request_outcomes()
    assert not supervisor.REQUEST_OUTCOMES


def test_outcome_http_contract(monkeypatch, tmp_path):
    supervisor = load_supervisor(monkeypatch, tmp_path)
    key = supervisor._begin_request_outcome({}, "model", "native-incremental-pcm")

    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=supervisor.app), base_url="http://test") as client:
            result = await client.get(f"/v1/audio/outcomes/{key}")
            assert result.status_code == 200
            assert result.headers["cache-control"] == "no-store"
            assert "_updated" not in result.json()
            assert (await client.post(f"/v1/audio/outcomes/{key}")).status_code == 405
            assert (await client.get("/v1/audio/outcomes/not-an-id")).status_code == 422
            assert (await client.get("/v1/audio/outcomes/" + "0" * 32)).status_code == 404
    asyncio.run(check())


def test_live_bounds_are_shared_and_not_offline_quality_controls(monkeypatch, tmp_path):
    supervisor = load_supervisor(monkeypatch, tmp_path)
    payload = {"input": "Count from one to five."}
    buffered = supervisor._request_audio_limit_frames(payload, "buffered-fallback")
    native = supervisor._request_audio_limit_frames(payload, "native-incremental-pcm")
    assert buffered == native == 150
    assert supervisor._request_audio_limit_frames(payload, supervisor.OFFLINE_FULL_DECODE_MODE) is None
    assert supervisor._request_audio_limit_frames({"input": "word " * 300}, "buffered-fallback") == 750
    engine_payload = {"options": {"qwen3_tts.max_generated_frames": 100000, "other": 4}}
    supervisor._install_request_limit(payload, engine_payload, "buffered-fallback")
    assert engine_payload["options"] == {"qwen3_tts.max_generated_frames": 150, "other": 4}


@pytest.mark.parametrize("terminal,expected", [("eos", "completed"), ("max_generated_frames", "limited"), ("error", "error"), ("missing", "error"), ("duplicate", "error"), ("pcm-after-eos", "error")])
def test_native_late_outcome_exact_pcm_and_clean_recovery(monkeypatch, tmp_path, terminal, expected):
    supervisor = load_supervisor(monkeypatch, tmp_path)
    sent = []
    references = []

    def prepare(*args):
        reference = tmp_path / f"reference-{len(references)}.wav"
        reference.write_bytes(b"temporary")
        references.append(reference)
        return reference, {"source_seconds": 1, "used_seconds": 1, "requested_limit_seconds": None,
                           "limit_applied": False, "truncated": False, "pairing": "full"}, "reference"

    class EngineResponse:
        headers = {}
        is_error = False
        async def aclose(self):
            pass
        async def aiter_lines(self):
            yield 'data: {"type":"speech.decode_mode","mode":"native-incremental-pcm"}'
            for _ in range(2):
                yield "data: " + json.dumps({"type": "speech.audio.delta", "audio": base64.b64encode(b"\x01\x00" * 1920).decode()})
            if terminal == "error":
                yield 'data: {"type":"error","error":{"message":"failure"}}'
            elif terminal != "missing":
                event = "data: " + json.dumps({"type": "speech.generation", "generated_frames": 2,
                                              "generation_cap": 150, "termination": terminal if terminal == "max_generated_frames" else "eos"})
                yield event
                if terminal == "duplicate":
                    yield event
                if terminal == "pcm-after-eos":
                    yield "data: " + json.dumps({"type": "speech.audio.delta", "audio": base64.b64encode(b"\x01\x00" * 1920).decode()})
            yield 'data: {"type":"speech.audio.done"}'
            yield "data: [DONE]"

    class EngineClient:
        def __init__(self, **kwargs):
            pass
        def build_request(self, *args, **kwargs):
            sent.append(kwargs["json"])
        async def send(self, *args, **kwargs):
            return EngineResponse()
        async def aclose(self):
            pass

    monkeypatch.setattr(supervisor.httpx, "AsyncClient", EngineClient)
    monkeypatch.setattr(supervisor, "_prepare_reference_pair", prepare)
    monkeypatch.setattr(supervisor, "_revalidate_generation_admission", lambda *a, **k: None)

    async def check():
        nonlocal terminal
        response = await supervisor._native_clone_pcm_response({"input": "test", "ref_audio": "test"}, "model")
        pcm = []
        try:
            async for chunk in response.body_iterator:
                pcm.append(chunk)
        except supervisor.HTTPException:
            assert expected != "completed"
        assert len(b"".join(pcm)) == 2 * 1920 * 2
        key = response.headers["x-tts-request-id"]
        assert supervisor.REQUEST_OUTCOMES[key]["state"] == expected
        assert not supervisor.generation_lock.locked()
        assert all(not item.exists() for item in references)
        assert sent[0]["options"]["qwen3_tts.max_generated_frames"] == 150
        assert not any(key.startswith("_") for key in sent[0])
        terminal = "eos"
        recovered = await supervisor._native_clone_pcm_response({"input": "next", "ref_audio": "test"}, "model")
        async for _ in recovered.body_iterator:
            pass
        assert supervisor.REQUEST_OUTCOMES[recovered.headers["x-tts-request-id"]]["state"] == "completed"
    asyncio.run(check())


def test_deadline_applies_even_without_new_chunks(monkeypatch, tmp_path):
    supervisor = load_supervisor(monkeypatch, tmp_path)
    async def delayed():
        await asyncio.sleep(10)
        yield "late"
    async def check():
        with pytest.raises(TimeoutError):
            async for _ in supervisor._deadline_lines(delayed(), supervisor.time.monotonic() + 0.01):
                pass
    asyncio.run(check())


def test_buffered_cap_exception_is_not_a_completed_clip(monkeypatch, tmp_path):
    supervisor = load_supervisor(monkeypatch, tmp_path)
    reference = tmp_path / "bounded-reference.wav"
    reference.write_bytes(b"temporary")
    monkeypatch.setattr(supervisor, "_apply_engine_tuning", lambda _: {"delivery_mode": "buffered-fallback"})
    monkeypatch.setattr(supervisor, "_native_request_diagnostics", lambda *a: {})
    monkeypatch.setattr(supervisor, "_prepare_reference_pair", lambda *a: (reference, {}, "paired text"))
    monkeypatch.setattr(supervisor, "_render_master_wav", lambda *a: pytest.fail("exported aborted output"))

    async def engine(*args, **kwargs):
        return httpx.Response(500, json={"error": {"message": "qwen3_tts.max_generated_frames_exhausted"}})

    monkeypatch.setattr(supervisor, "_post_engine_with_disconnect", engine)
    async def check():
        with pytest.raises(supervisor.HTTPException) as exc:
            await supervisor._voice_clone_response({"input": "Count backwards.", "ref_audio": "reference", "response_format": "pcm"})
        assert exc.value.status_code == 422
        key = exc.value.headers["X-TTS-Request-Id"]
        assert supervisor.REQUEST_OUTCOMES[key]["state"] == "limited"
        assert not reference.exists()
    asyncio.run(check())


@pytest.mark.parametrize("field,value", [("prompt_ms", True), ("prefill_ms", float("nan")), ("decode_ms", -1), ("talker_ms", 999999)])
def test_stage_telemetry_rejects_unmeasured_numbers(monkeypatch, tmp_path, field, value):
    supervisor = load_supervisor(monkeypatch, tmp_path)
    with pytest.raises(supervisor.HTTPException):
        supervisor._generation_event_fields({"generated_frames": 2, "generation_cap": 150, "termination": "eos", field: value})


def test_buffered_eos_at_exact_frame_cap_is_complete(monkeypatch, tmp_path):
    supervisor = load_supervisor(monkeypatch, tmp_path)
    reference = tmp_path / "exact-cap-reference.wav"
    reference.write_bytes(b"temporary")
    monkeypatch.setattr(supervisor, "_apply_engine_tuning", lambda _: {"delivery_mode": "buffered-fallback"})
    monkeypatch.setattr(supervisor, "_native_request_diagnostics", lambda *a: {})
    monkeypatch.setattr(supervisor, "_prepare_reference_pair", lambda *a: (reference, {
        "source_seconds": 1, "used_seconds": 1, "requested_limit_seconds": None,
        "limit_applied": False, "truncated": False, "pairing": "full"}, "paired text"))
    monkeypatch.setattr(supervisor, "_render_master_wav", lambda *a: (b"\x01\x00" * 288000, "audio/pcm", "pcm", {
        "durationSeconds": 12.0, "codec": "pcm_s16le", "container": "raw", "quality": "lossless",
        "sampleRate": 24000, "bitsPerSample": 16, "channels": 1}))

    async def engine(*args, **kwargs):
        return httpx.Response(200, content=b"completed master", headers={
            supervisor.ENGINE_DECODE_MODE_HEADER: supervisor.OFFLINE_FULL_DECODE_MODE,
            "X-AudioCPP-Termination": "eos", "X-AudioCPP-Generated-Frames": "150",
            "X-AudioCPP-Generation-Cap": "150"})

    monkeypatch.setattr(supervisor, "_post_engine_with_disconnect", engine)

    async def check():
        response = await supervisor._voice_clone_response({"input": "Count backwards.", "ref_audio": "reference", "response_format": "pcm"})
        assert response.status_code == 200
        outcome = supervisor.REQUEST_OUTCOMES[response.headers["x-tts-request-id"]]
        assert outcome["state"] == "completed" and outcome["eos"] is True
        assert outcome["generatedFrames"] == outcome["generationCap"] == 150
        assert not reference.exists()
    asyncio.run(check())
