"""Read-only acceptance probe for an already-running audio.cpp candidate.

The probe never calls a candidate control endpoint.  It reads the current
health, supervisor, and model identity, then makes one native PCM synthesis
request against that *already resident* model.  It records timing and framing
facts without writing audio bytes to stdout or disk.

Examples (only run after a model and Base clone are already selected):

    python scripts/probe_audio_cpp_native_pcm.py --voice clone:example
    python scripts/probe_audio_cpp_native_pcm.py --voice clone:example --abort-after-chunks 2

The optional abort closes the client response after the requested number of
chunks and then sends a short clean follow-up request.  This validates recovery
without loading, switching, unloading, or otherwise managing the candidate.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PCM_SAMPLE_RATE = 24_000
PCM_BYTES_PER_SAMPLE = 2
DEFAULT_ENDPOINT = "http://127.0.0.1:8890"


class ProbeError(RuntimeError):
    """A request or native transport contract failed."""


@dataclass
class StreamFacts:
    status_code: int | None = None
    content_type: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    first_byte_ms: float | None = None
    first_chunk_ms: float | None = None
    elapsed_ms: float | None = None
    chunks: int = 0
    pcm_bytes: int = 0
    chunk_bytes: list[int] = field(default_factory=list)
    cadence_ms: list[float] = field(default_factory=list)
    alignment_ok: bool = True
    sse_event_ids: list[str] = field(default_factory=list)
    duplicate_sse_event_ids: list[str] = field(default_factory=list)
    aborted: bool = False
    error: str | None = None

    def compact(self) -> dict[str, Any]:
        audio_seconds = self.pcm_bytes / (PCM_SAMPLE_RATE * PCM_BYTES_PER_SAMPLE)
        cadence = self.cadence_ms
        return {
            "status": self.status_code,
            "content_type": self.content_type,
            "first_byte_ms": _round_ms(self.first_byte_ms),
            "first_pcm_ms": _round_ms(self.first_chunk_ms),
            "elapsed_ms": _round_ms(self.elapsed_ms),
            "chunks": self.chunks,
            "pcm_bytes": self.pcm_bytes,
            "sample_count": self.pcm_bytes // PCM_BYTES_PER_SAMPLE,
            "audio_seconds": round(audio_seconds, 4),
            "rtf": round((self.elapsed_ms / 1000) / audio_seconds, 4) if audio_seconds and self.elapsed_ms else None,
            "chunk_bytes": self.chunk_bytes,
            "cadence_ms": {
                "mean": _round_ms(sum(cadence) / len(cadence)) if cadence else None,
                "max": _round_ms(max(cadence)) if cadence else None,
            },
            "alignment_ok": self.alignment_ok,
            "sse_event_ids": self.sse_event_ids,
            "duplicate_sse_event_ids": self.duplicate_sse_event_ids,
            "aborted": self.aborted,
            "error": self.error,
            "streaming_mode": self.headers.get("x-tts-streaming-mode"),
            "decoder_mode": self.headers.get("x-tts-decoder-mode"),
            "engine_chunk_proof": self.headers.get("x-tts-native-engine-chunk-proof"),
            "request_id": self.headers.get("x-tts-request-id"),
        }


def _round_ms(value: float | None) -> float | None:
    return round(value, 3) if value is not None else None


def _json_request(url: str, *, timeout: float) -> dict[str, Any]:
    try:
        with urlopen(Request(url, headers={"Accept": "application/json"}), timeout=timeout) as response:
            body = response.read()
    except (HTTPError, URLError, TimeoutError) as exc:
        raise ProbeError(f"GET {url} failed: {exc}") from exc
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ProbeError(f"GET {url} did not return JSON") from exc
    if not isinstance(data, dict):
        raise ProbeError(f"GET {url} returned a non-object JSON value")
    return data


def read_candidate_identity(endpoint: str, *, timeout: float) -> dict[str, Any]:
    """Fetch uncached current identity without changing candidate state."""
    base = endpoint.rstrip("/")
    health = _json_request(f"{base}/health", timeout=timeout)
    control = _json_request(f"{base}/control/status", timeout=timeout)
    models = _json_request(f"{base}/v1/models", timeout=timeout)
    backend = health.get("backend") if isinstance(health.get("backend"), dict) else {}
    runtime = backend.get("runtime") if isinstance(backend.get("runtime"), dict) else {}
    active_model = str(control.get("activeModel") or backend.get("model_id") or "")
    loaded = list(backend.get("loaded_models") or [])
    if not active_model or control.get("state") != "loaded" or active_model not in loaded:
        raise ProbeError("candidate has no confirmed resident model; refusing to issue synthesis")
    if control.get("singleResident") is not True:
        raise ProbeError("candidate does not attest to the single-resident-model contract")
    if runtime.get("native_incremental_pcm") is not True or health.get("nativeIncrementalPcm") is not True:
        raise ProbeError("candidate does not advertise native incremental PCM")
    advertised = models.get("data") if isinstance(models.get("data"), list) else []
    if active_model not in {str(item.get("id")) for item in advertised if isinstance(item, dict)}:
        raise ProbeError("resident model is absent from the fresh model inventory")
    engine_epoch = control.get("engineEpoch")
    supervisor_instance_id = control.get("supervisorInstanceId")
    if isinstance(engine_epoch, bool) or not isinstance(engine_epoch, int) or engine_epoch < 0:
        raise ProbeError("candidate did not provide a valid current engine epoch")
    if not isinstance(supervisor_instance_id, str) or not re.fullmatch(r"[0-9a-f]{32}", supervisor_instance_id):
        raise ProbeError("candidate did not provide a valid current supervisor instance identity")
    return {
        "active_model": active_model,
        "engine_epoch": engine_epoch,
        "supervisor_instance_id": supervisor_instance_id,
        "single_resident": True,
        "native_incremental_pcm": True,
        "mem_saver": runtime.get("mem_saver"),
        "cuda_graphs_disabled": runtime.get("cuda_graphs_disabled"),
        "gpu": runtime.get("gpu") or control.get("gpu"),
    }


def native_payload(
    *,
    model: str,
    voice: str,
    text: str,
    transport: str,
    expected_engine_epoch: int,
    expected_supervisor_instance_id: str,
) -> dict[str, Any]:
    """Create the constrained Base-clone request used by this acceptance probe."""
    payload: dict[str, Any] = {
        "model": model,
        "input": text,
        "voice": voice,
        "task_type": "Base",
        "response_format": "pcm",
        "stream": True,
        # These are read from one fresh identity snapshot.  The supervisor
        # rejects any concurrent reload/restart rather than letting this probe
        # silently test a different resident engine.
        "expected_engine_epoch": expected_engine_epoch,
        "expected_supervisor_instance_id": expected_supervisor_instance_id,
    }
    # The candidate's verified public path is raw audio.  SSE remains useful
    # for protocol-compatible engines and is decoded only when explicitly set.
    payload["stream_format"] = "audio" if transport == "raw" else "sse"
    return payload


def _iter_sse_pcm(lines: Iterable[bytes]) -> Iterable[tuple[bytes, str | None]]:
    """Decode common OpenAI-style SSE audio fields without accepting text bytes."""
    for line in lines:
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        value = line[5:].strip()
        if value == b"[DONE]":
            return
        try:
            event = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ProbeError("SSE data was not JSON") from exc
        if not isinstance(event, dict):
            continue
        encoded = event.get("audio") or event.get("delta") or event.get("audio_data")
        if not encoded:
            continue
        if not isinstance(encoded, str):
            raise ProbeError("SSE audio payload was not base64 text")
        try:
            event_id = event.get("event_id") or event.get("id")
            yield base64.b64decode(encoded, validate=True), str(event_id) if event_id is not None else None
        except ValueError as exc:
            raise ProbeError("SSE audio payload was not valid base64") from exc


def probe_stream(
    endpoint: str,
    payload: dict[str, Any],
    *,
    timeout: float,
    read_size: int = 4096,
    abort_after_chunks: int | None = None,
) -> StreamFacts:
    """Read native PCM/SSE once, retaining only transport-provable facts."""
    if read_size < PCM_BYTES_PER_SAMPLE:
        raise ValueError("read_size must be at least two bytes")
    if abort_after_chunks is not None and abort_after_chunks < 1:
        raise ValueError("abort_after_chunks must be positive")
    facts = StreamFacts()
    started = time.monotonic()
    request = Request(
        endpoint.rstrip("/") + "/v1/audio/speech",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "audio/pcm, text/event-stream"},
        method="POST",
    )
    previous_at: float | None = None
    seen_sse_event_ids: set[str] = set()
    try:
        with urlopen(request, timeout=timeout) as response:
            facts.status_code = response.status
            facts.content_type = response.headers.get_content_type()
            facts.headers = {key.lower(): value for key, value in response.headers.items()}
            is_sse = facts.content_type == "text/event-stream"
            chunks: Iterable[tuple[bytes, str | None]]
            chunks = _iter_sse_pcm(response) if is_sse else ((chunk, None) for chunk in iter(lambda: response.read(read_size), b""))
            for pcm, event_id in chunks:
                if not pcm:
                    continue
                now = time.monotonic()
                facts.first_byte_ms = facts.first_byte_ms if facts.first_byte_ms is not None else (now - started) * 1000
                facts.first_chunk_ms = facts.first_chunk_ms if facts.first_chunk_ms is not None else (now - started) * 1000
                if previous_at is not None:
                    facts.cadence_ms.append((now - previous_at) * 1000)
                previous_at = now
                facts.chunks += 1
                facts.pcm_bytes += len(pcm)
                facts.chunk_bytes.append(len(pcm))
                facts.alignment_ok = facts.alignment_ok and len(pcm) % PCM_BYTES_PER_SAMPLE == 0
                # Raw reads are TCP/client buffer boundaries: identical samples
                # (especially silence) are legitimate and cannot establish
                # duplicate server framing.  SSE has optional event identity,
                # which is the only exact-once fact this probe can verify.
                if event_id:
                    facts.sse_event_ids.append(event_id)
                    if event_id in seen_sse_event_ids:
                        facts.duplicate_sse_event_ids.append(event_id)
                    seen_sse_event_ids.add(event_id)
                if abort_after_chunks is not None and facts.chunks >= abort_after_chunks:
                    facts.aborted = True
                    break
            # Breaking the iterator exits this context and closes the transport,
            # which is the intentional cancellation operation.  No control API
            # or model lifecycle call is made.
    except HTTPError as exc:
        facts.status_code = exc.code
        facts.error = f"HTTP {exc.code}: {exc.read(512).decode('utf-8', errors='replace')}"
    except (URLError, TimeoutError, OSError, ProbeError) as exc:
        facts.error = str(exc)
    finally:
        facts.elapsed_ms = (time.monotonic() - started) * 1000
    return facts


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    identity = read_candidate_identity(args.endpoint, timeout=args.timeout)
    if args.model and args.model != identity["active_model"]:
        raise ProbeError("--model differs from the resident model; this probe never switches models")
    primary = probe_stream(
        args.endpoint,
        native_payload(
            model=identity["active_model"], voice=args.voice, text=args.text, transport=args.transport,
            expected_engine_epoch=identity["engine_epoch"], expected_supervisor_instance_id=identity["supervisor_instance_id"],
        ),
        timeout=args.timeout,
        read_size=args.read_size,
        abort_after_chunks=args.abort_after_chunks,
    )
    report: dict[str, Any] = {"identity": identity, "primary": primary.compact()}
    if primary.error:
        report["result"] = "failed"
        return report
    if primary.headers.get("x-tts-streaming-mode") != "native-incremental-pcm":
        report.update(result="failed", error="upstream did not prove native-incremental-pcm")
        return report
    if primary.headers.get("x-tts-native-engine-chunk-proof") != "two-distinct-sse-delta-events":
        report.update(result="failed", error="upstream did not prove two engine-owned PCM delta events")
        return report
    if primary.pcm_bytes == 0 or not primary.alignment_ok or primary.duplicate_sse_event_ids:
        report.update(result="failed", error="PCM alignment or exact-once chunk contract failed")
        return report
    if args.abort_after_chunks is not None:
        follow_up = probe_stream(
            args.endpoint,
            native_payload(
                model=identity["active_model"], voice=args.voice, text=args.follow_up_text, transport=args.transport,
                expected_engine_epoch=identity["engine_epoch"], expected_supervisor_instance_id=identity["supervisor_instance_id"],
            ),
            timeout=args.timeout,
            read_size=args.read_size,
        )
        report["follow_up"] = follow_up.compact()
        if follow_up.error or follow_up.chunks == 0 or not follow_up.alignment_ok:
            report.update(result="failed", error="clean follow-up request failed after cancellation")
            return report
    report["result"] = "passed"
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="Candidate base URL; default is the isolated candidate.")
    parser.add_argument("--voice", required=True, help="Existing Base clone ID, e.g. clone:your-profile. This script never creates profiles.")
    parser.add_argument("--model", help="Must equal the current resident model. Omit to read it from /health.")
    parser.add_argument("--text", default="Native PCM acceptance probe.", help="Short first request text.")
    parser.add_argument("--follow-up-text", default="Recovery probe.", help="Short request used after an intentional abort.")
    parser.add_argument("--transport", choices=("raw", "sse"), default="raw", help="Native candidate uses raw PCM; SSE is protocol-test support.")
    parser.add_argument("--abort-after-chunks", type=int, help="Close the first request after N PCM chunks, then verify one clean follow-up.")
    parser.add_argument("--timeout", type=float, default=90.0, help="Per-request HTTP timeout in seconds.")
    parser.add_argument("--read-size", type=int, default=4096, help="Maximum raw read size; does not re-chunk the server stream.")
    parser.add_argument("--report", type=Path, help="Optional JSON report file. Audio bytes are never written.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run_probe(args)
    except (ProbeError, ValueError) as exc:
        report = {"result": "failed", "error": str(exc)}
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.write_text(encoded, encoding="utf-8")
    sys.stdout.write(encoded)
    return 0 if report.get("result") == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
