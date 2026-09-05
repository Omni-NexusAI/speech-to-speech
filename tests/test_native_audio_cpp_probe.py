"""Offline contracts for the read-only native candidate probe harness."""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
from base64 import b64encode
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "probe_audio_cpp_native_pcm.py"
SPEC = importlib.util.spec_from_file_location("native_probe", SCRIPT)
assert SPEC and SPEC.loader
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


class _Handler(BaseHTTPRequestHandler):
    requests = 0
    payloads = []
    stream = [b"\x01\x00\x02\x00", b"\x03\x00\x04\x00"]
    content_type = "audio/pcm"

    def log_message(self, _format, *_args):  # pragma: no cover - test noise
        return

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            body = {"nativeIncrementalPcm": True, "backend": {"model_id": "model-a", "loaded_models": ["model-a"], "runtime": {"native_incremental_pcm": True, "mem_saver": False}}}
        elif self.path == "/control/status":
            body = {"activeModel": "model-a", "state": "loaded", "singleResident": True, "engineEpoch": 4, "supervisorInstanceId": "a" * 32}
        elif self.path == "/v1/models":
            body = {"data": [{"id": "model-a"}]}
        else:
            self.send_error(404)
            return
        encoded = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_POST(self):  # noqa: N802
        assert self.path == "/v1/audio/speech"
        _Handler.requests += 1
        _Handler.payloads.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        self.send_response(200)
        self.send_header("Content-Type", self.content_type)
        self.send_header("X-TTS-Streaming-Mode", "native-incremental-pcm")
        self.send_header("X-TTS-Native-Engine-Chunk-Proof", "two-distinct-sse-delta-events")
        self.end_headers()
        if self.content_type == "text/event-stream":
            for index, chunk in enumerate(self.stream):
                self.wfile.write(b"data: " + json.dumps({"audio": b64encode(chunk).decode(), "event_id": f"event-{index}"}).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            for chunk in self.stream:
                self.wfile.write(chunk)


@pytest.fixture
def candidate_server():
    _Handler.requests = 0
    _Handler.payloads = []
    _Handler.stream = [b"\x01\x00\x02\x00", b"\x03\x00\x04\x00"]
    _Handler.content_type = "audio/pcm"
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()


def _args(endpoint, **updates):
    values = dict(endpoint=endpoint, model=None, voice="clone:stable", text="first", follow_up_text="second", transport="raw", abort_after_chunks=None, timeout=3.0, read_size=4)
    values.update(updates)
    return type("Args", (), values)()


def test_probe_reads_identity_and_reports_pcm_timing_without_audio(candidate_server):
    report = probe.run_probe(_args(candidate_server))
    assert report["result"] == "passed"
    assert report["identity"]["active_model"] == "model-a"
    assert report["primary"]["sample_count"] == 4
    assert report["primary"]["alignment_ok"] is True
    assert report["primary"]["duplicate_sse_event_ids"] == []
    # Only aggregate facts are retained; raw/base64 PCM must never leak into
    # the report even though it contains an ``audio_seconds`` metric.
    assert "AQAC" not in json.dumps(report)
    assert _Handler.payloads[0]["expected_engine_epoch"] == 4
    assert _Handler.payloads[0]["expected_supervisor_instance_id"] == "a" * 32


def test_abort_closes_first_request_and_requires_clean_follow_up(candidate_server):
    report = probe.run_probe(_args(candidate_server, abort_after_chunks=1))
    assert report["result"] == "passed"
    assert report["primary"]["aborted"] is True
    assert report["follow_up"]["chunks"] == 2
    assert _Handler.requests == 2
    assert all(payload["expected_engine_epoch"] == 4 for payload in _Handler.payloads)
    assert all(payload["expected_supervisor_instance_id"] == "a" * 32 for payload in _Handler.payloads)


def test_sse_pcm_is_decoded_and_alignment_is_enforced(candidate_server):
    _Handler.content_type = "text/event-stream"
    report = probe.run_probe(_args(candidate_server, transport="sse"))
    assert report["result"] == "passed"
    assert report["primary"]["pcm_bytes"] == 8


def test_raw_repeated_silence_is_not_an_exact_once_failure_but_misalignment_is(candidate_server):
    _Handler.stream = [b"\x01\x00", b"\x01\x00"]
    repeated_silence = probe.run_probe(_args(candidate_server, read_size=2))
    assert repeated_silence["result"] == "passed"

    _Handler.stream = [b"\x01"]
    misaligned = probe.run_probe(_args(candidate_server))
    assert misaligned["result"] == "failed"
    assert misaligned["primary"]["alignment_ok"] is False


def test_duplicate_sse_event_identity_fails_exact_once_contract(candidate_server):
    _Handler.content_type = "text/event-stream"
    # Reuse the same explicit event ID by replacing the event generator input.
    original = _Handler.do_POST

    def duplicate_event(self):
        if self.path != "/v1/audio/speech":
            return original(self)
        _Handler.requests += 1
        _Handler.payloads.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("X-TTS-Streaming-Mode", "native-incremental-pcm")
        self.send_header("X-TTS-Native-Engine-Chunk-Proof", "two-distinct-sse-delta-events")
        self.end_headers()
        for chunk in _Handler.stream:
            self.wfile.write(b"data: " + json.dumps({"audio": b64encode(chunk).decode(), "event_id": "same"}).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")

    _Handler.do_POST = duplicate_event
    try:
        report = probe.run_probe(_args(candidate_server, transport="sse"))
    finally:
        _Handler.do_POST = original
    assert report["result"] == "failed"
    assert report["primary"]["duplicate_sse_event_ids"] == ["same"]


def test_refuses_model_mismatch_without_posting(candidate_server):
    with pytest.raises(probe.ProbeError, match="never switches"):
        probe.run_probe(_args(candidate_server, model="other-model"))
    assert _Handler.requests == 0
