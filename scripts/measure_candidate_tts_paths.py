"""Explicit, resident-model-only latency diagnostic; never changes saved settings.

Without --execute only nonsecret identity is returned. --execute warms each
configuration, then interleaves three repetitions. No audio is saved unless
--audio-dir is explicitly supplied for this diagnostic run. HTTP read chunks
are transport cadence, not engine chunk boundaries or listening evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import time
from pathlib import Path

import httpx


def inter_request_cooldown_seconds(value: str) -> float:
    """Parse the bounded pause without permitting non-finite sleep values."""
    try:
        seconds = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("inter-request cooldown must be a number from 0 to 60") from exc
    if not math.isfinite(seconds) or not 0 <= seconds <= 60:
        raise argparse.ArgumentTypeError("inter-request cooldown must be finite and between 0 and 60 seconds")
    return seconds


def gpu_snapshot() -> dict:
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,utilization.gpu,temperature.gpu",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=True)
        free, utilization, temperature = [int(part.strip()) for part in result.stdout.splitlines()[0].split(",")]
        return {"free_mib": free, "utilization_percent": utilization, "temperature_c": temperature}
    except Exception:
        return {"unavailable": True}


def settled_gpu_snapshot(max_temperature: int, *, snapshot=gpu_snapshot, sleep=time.sleep) -> dict:
    """Give the prior diagnostic's utilization sample at most five seconds to age out.

    Memory/temperature/telemetry failures stop immediately. Persistent utilization
    still blocks; this never relaxes a threshold or changes another workload.
    """
    for attempt in range(6):
        gpu = snapshot()
        if (gpu.get("unavailable") or gpu["free_mib"] < 2048
                or gpu["temperature_c"] > max_temperature
                or gpu["utilization_percent"] <= 90 or attempt == 5):
            return gpu
        sleep(1)
    return gpu


def measure(client, url, payload, *, audio_path=None, outcome_base=None):
    started = time.perf_counter()
    first = None
    bytes_received = 0
    last = None
    gaps = []
    parts = [] if audio_path else None
    with client.stream("POST", url, json=payload) as response:
        response.raise_for_status()
        headers_ms = (time.perf_counter() - started) * 1000
        headers = {key: value for key, value in response.headers.items() if key.startswith(("x-tts-", "x-audiocpp-"))}
        for chunk in response.iter_bytes():
            if not chunk:
                continue
            now = time.perf_counter()
            if first is None:
                first = now
            if last is not None:
                gaps.append((now - last) * 1000)
            last = now
            bytes_received += len(chunk)
            if parts is not None:
                parts.append(chunk)
    elapsed = time.perf_counter() - started
    if bytes_received % 2:
        raise RuntimeError("Incomplete PCM16 sample")
    if not bytes_received:
        raise RuntimeError("Empty audio is not a successful diagnostic")
    outcome = None
    request_id = headers.get("x-tts-request-id")
    if outcome_base:
        if not isinstance(request_id, str) or not re.fullmatch(r"[0-9a-f]{32}", request_id):
            raise RuntimeError("Invalid candidate outcome identity")
        response = client.get(outcome_base.rstrip("/") + "/v1/audio/outcomes/" + request_id)
        response.raise_for_status()
        outcome = response.json()
        if outcome.get("requestId") != request_id or outcome.get("state") != "completed":
            raise RuntimeError("Candidate did not confirm complete diagnostic audio")
        outcome = {key: outcome[key] for key in (
            "requestId", "state", "reason", "generatedFrames", "generationCap", "eos",
            "promptMs", "prefillMs", "talkerMs", "decodeMs", "generationMs", "audioDurationMs",
        ) if key in outcome}
    if parts is not None:
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        # Raw PCM deliberately stays byte-identical to the requested transport.
        audio_path.write_bytes(b"".join(parts))
    duration = bytes_received / 48000
    return {"headers_ms": round(headers_ms, 3), "first_transport_pcm_ms": round((first - started) * 1000, 3) if first else None,
            "elapsed_ms": round(elapsed * 1000, 3), "audio_seconds": duration,
            "rtf_including_transport": elapsed / duration if duration else None,
            "max_transport_gap_ms": max(gaps, default=0), "bytes": bytes_received,
            "headers": headers, "outcome": outcome,
            "completion_proven": outcome is not None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default="http://127.0.0.1:8890")
    parser.add_argument("--studio", default="http://127.0.0.1:8891")
    parser.add_argument("--voice", required=True)
    parser.add_argument("--profile", default="balanced")
    parser.add_argument("--text", default="Count slowly from one to five. Then count backwards from five to one.")
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--max-gpu-temperature", type=int, default=83)
    parser.add_argument("--inter-request-cooldown-seconds", type=inter_request_cooldown_seconds, default=0)
    args = parser.parse_args()
    if not 0 <= args.seed <= 4294967295:
        parser.error("seed must be uint32")
    base = args.candidate.rstrip("/")
    with httpx.Client(timeout=180, trust_env=False) as client:
        health = client.get(base + "/health").raise_for_status().json()
        profiles = client.get(base + "/v1/tuning/profiles").raise_for_status().json()
        clone = client.get(base + "/v1/voices/profiles/" + args.voice.removeprefix("clone:")).raise_for_status().json()
        profile = profiles["profiles"][args.profile]
        tuning_fields = {"model", "clone_mode", "max_reference_seconds", "first_block_frames", "steady_block_frames",
                         "left_context_frames", "text_lookahead", "phrase_flush_ms", "temperature", "top_k", "top_p", "repetition_penalty", "seed"}
        frozen_clone = {key: clone[key] for key in ("content_hash", "content_revision", "ref_audio", "ref_text", "reference_excerpts") if key in clone}
        frozen_clone["profile_id"] = args.voice.removeprefix("clone:")
        identity = {key: health.get(key) for key in ("activeModel", "engineEpoch", "supervisorInstanceId", "state", "singleResident")}
        report = {"identity": identity, "profile": profile, "gpu": gpu_snapshot(),
                  "clone_fingerprint": clone.get("content_hash", clone.get("contentHash")),
                  "text_sha256": hashlib.sha256(args.text.encode()).hexdigest(), "seed": args.seed,
                  "inter_request_cooldown_seconds": args.inter_request_cooldown_seconds,
                  "listening_verified": False, "phrase_queue_tuning": "Not measured by an HTTP speech request; requires a live Studio/HFRT turn",
                  "runs": []}
        if not args.execute:
            print(json.dumps(report, indent=2))
            return
        if identity["state"] != "loaded" or not identity["singleResident"]:
            raise RuntimeError("No confirmed single resident model; diagnostic never loads one")
        native_profiles = [("native-baseline", {}), ("native-smaller-first", {"first_block_frames": max(1, profile["first_block_frames"] // 2)})]
        configurations = [(label, True, overrides) for label, overrides in native_profiles] + [("buffered-reference", False, {})]
        prior_self_request_completed = False
        for repetition in range(4):  # first pass is warmup, kept separately.
            for label, native, overrides in configurations:
                for path_name, url in (("candidate", base + "/v1/audio/speech"),
                                       ("studio-same-origin-proxy", args.studio.rstrip("/") + "/api/audio-cpp/audio/speech")):
                    # A first request must not inherit a stale observation from
                    # another workload.  Only this diagnostic's completed prior
                    # request may receive the bounded utilization settle window.
                    # An optional pace starts only after that completed request,
                    # outside its measurement interval and before fresh admission.
                    if prior_self_request_completed and args.inter_request_cooldown_seconds:
                        time.sleep(args.inter_request_cooldown_seconds)
                    gpu = (
                        settled_gpu_snapshot(args.max_gpu_temperature)
                        if prior_self_request_completed
                        else gpu_snapshot()
                    )
                    if gpu.get("unavailable") or gpu["free_mib"] < 2048 or gpu["utilization_percent"] > 90 or gpu["temperature_c"] > args.max_gpu_temperature:
                        report.update(blocked="GPU contention or temperature; performance inconclusive", gpu=gpu)
                        print(json.dumps(report, indent=2))
                        return
                    current = client.get(base + "/health").raise_for_status().json()
                    if any(current.get(key) != value for key, value in identity.items()):
                        raise RuntimeError("Candidate residency changed during diagnostic; stopped")
                    payload = {"model": identity["activeModel"], "voice": args.voice, "input": args.text,
                               "stream": native, "response_format": "pcm", "language": "English",
                               "expected_engine_epoch": identity["engineEpoch"],
                               "expected_supervisor_instance_id": identity["supervisorInstanceId"],
                               "clone_snapshot": frozen_clone,
                               "tuning": {"provider": "qwen3tts-audiocpp", "profile_id": args.profile,
                                          "profile_revision": profile["revision"],
                                          "effective": {key: profile[key] for key in tuning_fields if key in profile},
                                          "scope": "realtime", "overrides": {"seed": args.seed, **overrides}}}
                    output = args.audio_dir / f"{repetition}-{label}-{path_name}.pcm" if args.audio_dir else None
                    facts = measure(client, url, payload, audio_path=output, outcome_base=base)
                    prior_self_request_completed = True
                    report["runs"].append({"warmup": repetition == 0, "repetition": repetition,
                                           "configuration": label, "path": path_name, "gpu": gpu, **facts})
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
