"""Behavioral checks for opt-in native PCM wiring and rollback boundaries."""
from __future__ import annotations

import asyncio
import base64
import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

ROOT = Path(__file__).parents[3]
INTEGRATION = Path(__file__).parents[1]
UI = INTEGRATION / "gradio_voice_studio.py"
SUPERVISOR = INTEGRATION / "supervisor.py"
SERVER = ROOT / "web" / "hf-realtime-voice" / "server.py"

gradio_stub = types.ModuleType("gradio")
gradio_stub.Blocks = object
gradio_stub.themes = types.SimpleNamespace(Base=object)
sys.modules.setdefault("gradio", gradio_stub)

ui_spec = importlib.util.spec_from_file_location("candidate_gradio_native_test", UI)
assert ui_spec and ui_spec.loader
studio = importlib.util.module_from_spec(ui_spec)
ui_spec.loader.exec_module(studio)

supervisor_spec = importlib.util.spec_from_file_location("candidate_supervisor_native_test", SUPERVISOR)
assert supervisor_spec and supervisor_spec.loader
supervisor = importlib.util.module_from_spec(supervisor_spec)
supervisor_spec.loader.exec_module(supervisor)


class _FakeResponse:
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.headers = {
            "x-tts-sample-rate": "24000",
            "x-tts-streaming-mode": "native-incremental-pcm",
        }
        self.exited = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.exited = True

    def raise_for_status(self) -> None:
        return None

    def iter_raw(self):
        yield from self.chunks


class _FakeClient:
    response: _FakeResponse
    request: tuple[str, str, dict] | None = None

    def __init__(self, **_kwargs) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def stream(self, method: str, url: str, json: dict):
        type(self).request = (method, url, json)
        return type(self).response


class _CancelAfterFirstChunk:
    def __init__(self) -> None:
        self.cancelled = False

    def is_set(self) -> bool:
        return self.cancelled


class CandidateNativeWiringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_client = studio.httpx.Client
        self.original_native = studio.NATIVE_INCREMENTAL_PCM_ENABLED
        self.original_supervisor_native = supervisor.NATIVE_INCREMENTAL_PCM_ENABLED
        studio.httpx.Client = _FakeClient
        studio.NATIVE_INCREMENTAL_PCM_ENABLED = True

    def tearDown(self) -> None:
        studio.httpx.Client = self.original_client
        studio.NATIVE_INCREMENTAL_PCM_ENABLED = self.original_native
        supervisor.NATIVE_INCREMENTAL_PCM_ENABLED = self.original_supervisor_native

    def test_python_stream_helper_uses_raw_native_contract_and_reports_chunks(self) -> None:
        _FakeClient.response = _FakeResponse([b"\x01\x00", b"\x02\x00"])
        observed: list[bytes] = []
        wav, extension, timing = studio.request_tts_streaming(
            "http://candidate",
            {"input": "hello", "voice": "clone:test", "response_format": "wav", "stream": False},
            30,
            on_chunk=observed.append,
        )
        self.assertEqual(_FakeClient.request[0:2], ("POST", "http://candidate/v1/audio/speech"))
        self.assertTrue(_FakeClient.request[2]["stream"])
        self.assertEqual(_FakeClient.request[2]["response_format"], "pcm")
        self.assertEqual(observed, [b"\x01\x00", b"\x02\x00"])
        self.assertEqual(extension, "wav")
        self.assertTrue(wav.startswith(b"RIFF"))
        self.assertEqual(timing["chunk_count"], 2)
        self.assertEqual(timing["pcm_bytes"], 4)
        self.assertEqual(timing["streaming_mode"], "native-incremental-pcm")

    def test_python_stream_cancellation_closes_response_before_late_pcm(self) -> None:
        _FakeClient.response = _FakeResponse([b"\x01\x00", b"\x02\x00"])
        cancel = _CancelAfterFirstChunk()
        observed: list[bytes] = []

        def receive(block: bytes) -> None:
            observed.append(block)
            cancel.cancelled = True

        with self.assertRaises(studio.NativeStreamingCancelled):
            studio.request_tts_streaming(
                "http://candidate", {"input": "cancel", "voice": "clone:test"}, 30,
                cancel_event=cancel, on_chunk=receive,
            )
        self.assertEqual(observed, [b"\x01\x00"])
        self.assertTrue(_FakeClient.response.exited)

    def test_seed_is_uint32_and_xvector_remains_unsupported(self) -> None:
        payload = {"seed": 7}
        supervisor._normalize_gradio_seed(payload)
        self.assertEqual(payload["seed"], 7)
        with self.assertRaises(HTTPException) as error:
            asyncio.run(supervisor._voice_clone_response({"x_vector_only_mode": True}))
        self.assertEqual(error.exception.status_code, 422)
        with self.assertRaises(HTTPException) as imported:
            supervisor._write_candidate_profile(
                "xvector-test",
                {"ref_audio": "AA==", "ref_text": "reference", "x_vector_only_mode": True},
            )
        self.assertEqual(imported.exception.status_code, 422)

    def test_browser_proxy_and_widget_preserve_streaming_and_cancellation(self) -> None:
        server = SERVER.read_text(encoding="utf-8")
        ui = UI.read_text(encoding="utf-8")
        self.assertIn("upstream = await http.send(upstream_request, stream=True)", server)
        self.assertIn("async for chunk in upstream.aiter_raw()", server)
        self.assertIn("await upstream.aclose()", server)
        self.assertIn("response.body.getReader()", ui)
        # Streaming transport is frozen with the response snapshot so a
        # mid-answer UI change cannot split one answer across delivery modes.
        self.assertIn("stream: responseSnapshot.nativeStreaming", ui)
        self.assertIn("tuning: responseSnapshot.tuning", ui)
        self.assertIn('session_tuning.get("profile_id") == tuning_profile_id', ui)
        self.assertIn("reader.cancel()", ui)
        self.assertIn("cancelScheduledPlayback()", ui)
        self.assertIn("activeLlmRequest", ui)
        self.assertIn("activeTtsRequest", ui)
        self.assertIn("Native incremental PCM (experimental)", ui)
        self.assertIn("Buffered phrase PCM (rollback fallback)", ui)
        self.assertIn("def default_playback_mode() -> str:", ui)
        default_fn = ui.split("def default_playback_mode() -> str:", 1)[1].split(
            "DEFAULT_PLAYBACK_MODE =", 1
        )[0]
        self.assertIn("return BUFFERED_PLAYBACK_MODE", default_fn)
        self.assertIn("value=DEFAULT_PLAYBACK_MODE", ui)

    def test_native_enablement_is_candidate_only_and_default_is_buffered(self) -> None:
        candidate = (INTEGRATION / "compose.candidate.yml").read_text(encoding="utf-8")
        native = (INTEGRATION / "compose.native.yml").read_text(encoding="utf-8")
        self.assertIn('AUDIO_CPP_NATIVE_INCREMENTAL_PCM: "false"', candidate)
        self.assertIn('AUDIO_CPP_NATIVE_INCREMENTAL_PCM: "true"', native)
        self.assertIn('AUDIO_CPP_NATIVE_LOAD_WARMUP: "true"', native)
        self.assertIn('AUDIO_CPP_NATIVE_LOAD_WARMUP_TIMEOUT_SECONDS: "180"', native)
        self.assertIn('AUDIO_CPP_TALKER_PREFIX_CACHE_SLOTS: "4"', native)
        self.assertNotIn("GGML_CUDA_DISABLE_GRAPHS", native)
        self.assertIn("no supported runtime CUDA-graph", native)
        self.assertIn("exact clone/language/control/reference KV prefix", native)
        self.assertIn("phrase-specific suffix", native)
        self.assertIn("generated speech state remain fresh", native)
        self.assertIn("engine consumes one", native)
        self.assertIn("${AUDIO_CPP_IMAGE:-local/audio-cpp-qwen3-tts-voice-studio:native-development}", candidate)
        self.assertNotIn("image:", native)
        self.assertIn("  audio-cpp-candidate:", native)
        self.assertNotIn("  qwen3-tts-faster:", native)
        self.assertNotIn("  groxaxo:", native.lower())

    def test_cuda_architecture_is_explicit_for_gpu_invisible_docker_builds(self) -> None:
        # CMake initializes CMAKE_CUDA_ARCHITECTURES from CUDAARCHS only on
        # the first configure.  Docker builder stages have no visible host
        # GPU, so each canonical engine build must pin sm_86 before invoking
        # the upstream build script rather than inheriting nvcc's default.
        expected_arg = "ARG AUDIO_CPP_CUDA_ARCHITECTURES=86"
        expected_env = 'CUDAARCHS="$AUDIO_CPP_CUDA_ARCHITECTURES" bash scripts/build_linux.sh'
        for name in ("Dockerfile", "Dockerfile.overlay"):
            source = (INTEGRATION / name).read_text(encoding="utf-8")
            with self.subTest(name=name):
                self.assertIn(expected_arg, source)
                self.assertIn(expected_env, source)
                self.assertIn("--target audiocpp_server", source)

    def test_cuda_graph_status_is_uncontrolled_and_never_inferred_from_environment(self) -> None:
        originals = {
            "state": dict(supervisor.state),
            "reconcile": supervisor._reconcile_engine,
            "ready": supervisor._engine_ready,
            "guard_policy": supervisor._guard_policy,
            "enforced_policy": supervisor._enforced_gpu_policy,
        }

        async def ready() -> bool:
            return True

        try:
            supervisor.state.update(state="loaded", activeModel="qwen3-tts-1.7b-base-bf16")
            supervisor._reconcile_engine = lambda: None
            supervisor._engine_ready = ready
            supervisor._guard_policy = lambda *_args: (0, {})
            supervisor._enforced_gpu_policy = lambda: {}
            observed = []
            for requested_value in ("1", "0"):
                # This is a historical engine-side environment knob, but the
                # pinned engine provides no runtime graph control or evidence
                # from which the supervisor may derive a status claim.
                with patch.dict(os.environ, {"GGML_CUDA_DISABLE_GRAPHS": requested_value}):
                    observed.append(asyncio.run(supervisor.health())["backend"]["runtime"])
            for runtime in observed:
                self.assertIsNone(runtime["cuda_graphs_disabled"])
                self.assertFalse(runtime["cuda_graphs_control_supported"])
                self.assertEqual(runtime["cuda_graphs_mode"], "engine-default-uncontrolled")
        finally:
            supervisor.state.clear()
            supervisor.state.update(originals["state"])
            supervisor._reconcile_engine = originals["reconcile"]
            supervisor._engine_ready = originals["ready"]
            supervisor._guard_policy = originals["guard_policy"]
            supervisor._enforced_gpu_policy = originals["enforced_policy"]

    def test_native_engine_config_enables_only_bounded_immutable_prefix_slots(self) -> None:
        original_native = supervisor.NATIVE_INCREMENTAL_PCM_ENABLED
        original_slots = supervisor.TALKER_PREFIX_CACHE_SLOTS
        original_models = supervisor._models
        original_template = supervisor._template
        original_active_config = supervisor.ACTIVE_CONFIG
        original_popen = supervisor.subprocess.Popen
        original_engine = supervisor.engine
        captured: dict[str, object] = {}

        class FakeProcess:
            returncode = None

            def poll(self):
                return None

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                supervisor.NATIVE_INCREMENTAL_PCM_ENABLED = True
                supervisor.TALKER_PREFIX_CACHE_SLOTS = 4
                supervisor._models = lambda: {
                    "qwen3-tts-1.7b-base-bf16": {"session_options": {}}
                }
                supervisor._template = lambda: {"models": [], "port": 0}
                supervisor.ACTIVE_CONFIG = Path(temp_dir) / "active.json"
                supervisor.subprocess.Popen = (
                    lambda args: captured.update(args=args) or FakeProcess()
                )

                supervisor._start_engine("qwen3-tts-1.7b-base-bf16")
                config = __import__("json").loads(
                    supervisor.ACTIVE_CONFIG.read_text(encoding="utf-8")
                )
                options = config["models"][0]["session_options"]
                self.assertEqual(options["qwen3_tts.mem_saver"], "false")
                self.assertEqual(options["qwen3_tts.talker_prefix_cache_slots"], "4")
                self.assertEqual(config["models"][0]["mode"], "streaming")
        finally:
            supervisor.NATIVE_INCREMENTAL_PCM_ENABLED = original_native
            supervisor.TALKER_PREFIX_CACHE_SLOTS = original_slots
            supervisor._models = original_models
            supervisor._template = original_template
            supervisor.ACTIVE_CONFIG = original_active_config
            supervisor.subprocess.Popen = original_popen
            supervisor.engine = original_engine

    def test_native_load_warmup_consumes_private_pcm_before_ready(self) -> None:
        originals = {
            "gpu_guard": supervisor.gpu_guard,
            "inventory": supervisor._candidate_profile_response,
            "profile": supervisor._candidate_profile_payload,
            "resolve": supervisor._resolve_request_tuning,
            "native": supervisor._native_clone_pcm_response,
        }
        captured: dict[str, object] = {}

        async def blocks():
            yield b"\x01\x00"
            yield b"\x02\x00"

        try:
            supervisor.gpu_guard = lambda *_args, **_kwargs: {"ok": True}
            supervisor._candidate_profile_response = lambda: {
                "selectedVoice": "",
                "defaultVoice": "clone:warm-profile",
                "voices": [{"id": "warm-profile"}],
            }
            supervisor._candidate_profile_payload = lambda profile_id, include_audio: {
                "id": profile_id,
                "ref_audio": "UklGRg==",
                "ref_text": "paired transcript",
            }
            supervisor._resolve_request_tuning = lambda payload: {"resolved": True}

            async def fake_native(payload, model_id):
                captured.update(payload=payload, model_id=model_id)
                return types.SimpleNamespace(body_iterator=blocks())

            supervisor._native_clone_pcm_response = fake_native
            result = asyncio.run(supervisor._run_native_load_warmup("qwen3-tts-1.7b-base-bf16"))
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["profileId"], "warm-profile")
            self.assertEqual(captured["model_id"], "qwen3-tts-1.7b-base-bf16")
            payload = captured["payload"]
            self.assertEqual(payload["input"], "Ready.")
            self.assertEqual(payload["tuning"]["overrides"]["seed"], 321)
            self.assertEqual(payload["_tuning_snapshot"], {"resolved": True})
            self.assertTrue(payload["_internal_warmup"])
            self.assertEqual(payload["_engine_epoch"], supervisor.state["engineEpoch"])
        finally:
            supervisor.gpu_guard = originals["gpu_guard"]
            supervisor._candidate_profile_response = originals["inventory"]
            supervisor._candidate_profile_payload = originals["profile"]
            supervisor._resolve_request_tuning = originals["resolve"]
            supervisor._native_clone_pcm_response = originals["native"]

    def test_generation_admission_rejects_stale_epoch_and_allows_internal_warmup(self) -> None:
        original_state = dict(supervisor.state)
        try:
            supervisor.state.update(activeModel="qwen3-tts-1.7b-base-bf16", state="loaded", engineEpoch=7)
            supervisor._revalidate_generation_admission(
                {"_engine_epoch": 7}, "qwen3-tts-1.7b-base-bf16"
            )
            with self.assertRaises(supervisor.HTTPException) as stale:
                supervisor._revalidate_generation_admission(
                    {"_engine_epoch": 6}, "qwen3-tts-1.7b-base-bf16"
                )
            self.assertEqual(stale.exception.status_code, 409)
            supervisor.state["state"] = "warming"
            supervisor._revalidate_generation_admission(
                {"_engine_epoch": 7},
                "qwen3-tts-1.7b-base-bf16",
                internal_warmup=True,
            )
            with self.assertRaises(supervisor.HTTPException):
                supervisor._revalidate_generation_admission(
                    {"_engine_epoch": 7}, "qwen3-tts-1.7b-base-bf16"
                )
        finally:
            supervisor.state.clear()
            supervisor.state.update(original_state)

    def test_native_diagnostics_are_request_correlated_without_phrase_text(self) -> None:
        payload = {
            "request_id": "response-17",
            "voice": "clone:jarvis",
            "_engine_epoch": 9,
            "_frozen_clone_content_hash": "a" * 64,
            "_frozen_clone_content_revision": 4,
            "input": "do not log this phrase",
            "ref_text": "do not log this transcript",
        }
        diagnostics = supervisor._native_request_diagnostics(
            payload,
            {
                "scope": "realtime", "id": "balanced", "revision": 2,
                "effective": {
                    "first_block_frames": 4,
                    "steady_block_frames": 12,
                    "left_context_frames": 25,
                },
            },
            {"seed": 123},
            {"source_seconds": 12, "requested_limit_seconds": 20, "used_seconds": 12,
             "limit_applied": False, "pairing": "full", "truncated": False},
            "qwen3-tts-1.7b-base-bf16",
        )
        self.assertEqual(diagnostics["requestId"], "response-17")
        self.assertEqual(diagnostics["engineEpoch"], 9)
        self.assertEqual(diagnostics["tuningProfileId"], "balanced")
        self.assertEqual(diagnostics["responseSeed"], 123)
        self.assertEqual(diagnostics["cloneId"], "jarvis")
        self.assertNotIn("input", diagnostics)
        self.assertNotIn("ref_text", diagnostics)
        self.assertEqual(
            supervisor._engine_stage_timings(types.SimpleNamespace(headers={
                "X-AudioCPP-Prefill-Ms": "2.1", "X-AudioCPP-Qwen3-Decode-Mode": "native-incremental-pcm",
                "content-type": "audio/pcm",
            })),
            {"X-AudioCPP-Prefill-Ms": "2.1"},
        )

    def test_native_sse_proof_parser_rejects_malformed_or_unproven_events(self) -> None:
        self.assertEqual(
            supervisor._native_sse_event(
                'data: {"type":"speech.decode_mode","mode":"native-incremental-pcm"}'
            ),
            {"type": "speech.decode_mode", "mode": "native-incremental-pcm"},
        )
        self.assertIs(supervisor._native_sse_event("data: [DONE]"), supervisor._NATIVE_SSE_DONE)
        with self.assertRaises(supervisor.HTTPException) as malformed:
            supervisor._native_sse_event("data: not-json")
        self.assertEqual(malformed.exception.status_code, 502)

    def test_native_sse_prefetch_verifies_proof_and_two_engine_pcm_boundaries(self) -> None:
        async def events():
            yield 'data: {"type":"speech.decode_mode","mode":"native-incremental-pcm"}'
            yield 'data: {"type":"speech.audio.delta","audio":"AQACAA=="}'
            yield 'data: {"type":"speech.audio.delta","audio":"AwAEAA=="}'
            yield 'data: [DONE]'

        async def prefetch_then_read_terminal() -> tuple[str, list[tuple[bytes, float]], str]:
            stream = events()
            mode = await supervisor._await_native_decode_mode_proof(stream)
            pcm = await supervisor._prefetch_native_pcm_proof(
                stream, verified_decode_mode=mode
            )
            return mode, pcm, await stream.__anext__()

        mode, pcm, line = asyncio.run(prefetch_then_read_terminal())
        self.assertEqual(mode, "native-incremental-pcm")
        self.assertEqual([chunk for chunk, _received_at in pcm], [b"\x01\x00\x02\x00", b"\x03\x00\x04\x00"])
        self.assertIs(supervisor._native_sse_event(line), supervisor._NATIVE_SSE_DONE)

        async def premature_audio():
            yield 'data: {"type":"speech.audio.delta","audio":"AQACAA=="}'

        with self.assertRaises(supervisor.HTTPException) as error:
            asyncio.run(supervisor._await_native_decode_mode_proof(premature_audio()))
        self.assertEqual(error.exception.status_code, 502)
        self.assertEqual(error.exception.detail["state"], "decoder-mode-proof-missing")

        async def error_before_proof():
            yield 'data: {"type":"error","error":{"message":"prefill failed"}}'
            yield 'data: {"type":"speech.decode_mode","mode":"native-incremental-pcm"}'

        with self.assertRaises(supervisor.HTTPException) as engine_error:
            asyncio.run(supervisor._await_native_decode_mode_proof(error_before_proof()))
        self.assertEqual(engine_error.exception.detail["state"], "native-stream-error")

        async def done_before_proof():
            yield "data: [DONE]"
            yield 'data: {"type":"speech.decode_mode","mode":"native-incremental-pcm"}'

        with self.assertRaises(supervisor.HTTPException) as terminal_error:
            asyncio.run(supervisor._await_native_decode_mode_proof(done_before_proof()))
        self.assertEqual(terminal_error.exception.detail["state"], "decoder-mode-proof-missing")

    def test_native_relay_rejects_post_pcm_error_and_missing_completion(self) -> None:
        class FakeAsyncResponse:
            def __init__(self, lines: list[str]) -> None:
                self.lines = lines
                self.headers = {}
                self.status_code = 200
                self.is_error = False

            async def aiter_lines(self):
                for line in self.lines:
                    yield line

            async def aclose(self) -> None:
                return None

        class FakeAsyncClient:
            lines: list[str] = []

            def __init__(self, **_kwargs) -> None:
                pass

            def build_request(self, method: str, url: str, json: dict):
                return (method, url, json)

            async def send(self, _request, stream: bool = False):
                self.assert_stream = stream
                return FakeAsyncResponse(type(self).lines)

            async def aclose(self) -> None:
                return None

        originals = {
            "client": supervisor.httpx.AsyncClient,
            "prepare": supervisor._prepare_reference_pair,
            "revalidate": supervisor._revalidate_generation_admission,
            "record": supervisor._record_event,
        }
        temporary_reference = INTEGRATION / ".native-relay-test-reference.wav"
        observed_events: list[tuple[str, dict]] = []

        async def consume(lines: list[str]) -> tuple[list[bytes], dict[str, str], BaseException | None]:
            FakeAsyncClient.lines = lines
            chunks: list[bytes] = []
            try:
                response = await supervisor._native_clone_pcm_response(
                    {
                        "input": "test",
                        "ref_audio": "placeholder",
                        "voice": "clone:test",
                        "_engine_epoch": 1,
                    },
                    "qwen3-tts-1.7b-base-bf16",
                )
                headers = dict(response.headers)
                async for chunk in response.body_iterator:
                    chunks.append(chunk)
            except BaseException as exc:  # Proof failures can precede response headers.
                return chunks, {}, exc
            return chunks, headers, None

        try:
            supervisor.httpx.AsyncClient = FakeAsyncClient
            supervisor._prepare_reference_pair = lambda *_args, **_kwargs: (
                temporary_reference,
                {
                    "source_seconds": 1.0, "requested_limit_seconds": None,
                    "used_seconds": 1.0, "limit_applied": False,
                    "pairing": "full", "truncated": False,
                },
                "reference",
            )
            supervisor._revalidate_generation_admission = lambda *_args, **_kwargs: None
            supervisor._record_event = lambda name, **detail: observed_events.append((name, detail))

            pcm = 'data: {"type":"speech.audio.delta","audio":"AQACAA=="}'
            pcm_two = 'data: {"type":"speech.audio.delta","audio":"AwAEAA=="}'
            proof = 'data: {"type":"speech.decode_mode","mode":"native-incremental-pcm"}'
            engine_error = 'data: {"type":"error","error":{"message":"decoder failed"}}'
            chunks, _headers, failure = asyncio.run(consume([proof, pcm, pcm_two, engine_error, "data: [DONE]"]))
            self.assertEqual(chunks, [b"\x01\x00\x02\x00", b"\x03\x00\x04\x00"])
            self.assertIsInstance(failure, supervisor.HTTPException)
            self.assertEqual(failure.detail["state"], "native-stream-error")
            self.assertFalse(any(name == "generation-complete" for name, _ in observed_events))
            self.assertTrue(any(name == "generation-error" for name, _ in observed_events))

            observed_events.clear()
            chunks, _headers, failure = asyncio.run(consume([proof, pcm, pcm_two, "data: [DONE]"]))
            self.assertEqual(chunks, [b"\x01\x00\x02\x00", b"\x03\x00\x04\x00"])
            self.assertIsInstance(failure, supervisor.HTTPException)
            self.assertEqual(failure.detail["state"], "native-stream-incomplete")
            self.assertFalse(any(name == "generation-complete" for name, _ in observed_events))

            observed_events.clear()
            whole_frame = b"\x01\x00" * 1920
            frame_pcm = 'data: {"type":"speech.audio.delta","audio":"' + base64.b64encode(whole_frame).decode() + '"}'
            generation = 'data: {"type":"speech.generation","generated_frames":2,"generation_cap":150,"termination":"eos"}'
            chunks, headers, failure = asyncio.run(
                consume([proof, frame_pcm, frame_pcm, generation, 'data: {"type":"speech.audio.done"}', "", "data: [DONE]", ""])
            )
            self.assertIsNone(failure)
            self.assertEqual(chunks, [whole_frame, whole_frame])
            self.assertTrue(any(name == "generation-complete" for name, _ in observed_events))
            proof_events = [detail for name, detail in observed_events if name == "native-pcm-preheader-proof"]
            self.assertEqual(len(proof_events), 1)
            self.assertEqual(proof_events[0]["preheaderProofChunks"], 2)
            self.assertGreaterEqual(proof_events[0]["preheaderProofReadyMs"], proof_events[0]["preheaderProofFirstPcmMs"])
            self.assertGreaterEqual(proof_events[0]["preheaderProofAdditionalWaitMs"], 0)
            self.assertEqual(headers["x-tts-native-engine-chunk-proof"], "two-distinct-sse-delta-events")
            self.assertEqual(
                headers["x-tts-native-preheader-proof-first-pcm-ms"],
                str(proof_events[0]["preheaderProofFirstPcmMs"]),
            )
            self.assertEqual(
                headers["x-tts-native-preheader-proof-ready-ms"],
                str(proof_events[0]["preheaderProofReadyMs"]),
            )
            self.assertEqual(
                headers["x-tts-native-preheader-proof-additional-wait-ms"],
                str(proof_events[0]["preheaderProofAdditionalWaitMs"]),
            )

            for premature_terminal in (
                [proof, "data: [DONE]", pcm, pcm_two, 'data: {"type":"speech.audio.done"}'],
                [proof, pcm, pcm_two, 'data: {"type":"speech.audio.done"}', "data: [DONE]", "data: [DONE]"],
            ):
                observed_events.clear()
                chunks, _headers, failure = asyncio.run(consume(premature_terminal))
                self.assertIsInstance(failure, supervisor.HTTPException)
                self.assertFalse(any(name == "generation-complete" for name, _ in observed_events))

            observed_events.clear()
            for trailing in (
                proof,
                'data: {"type":"unexpected.after.done"}',
            ):
                chunks, _headers, failure = asyncio.run(
                    consume([proof, pcm, pcm_two, generation, 'data: {"type":"speech.audio.done"}', trailing, "data: [DONE]"])
                )
                self.assertEqual(chunks, [b"\x01\x00\x02\x00", b"\x03\x00\x04\x00"])
                self.assertIsInstance(failure, supervisor.HTTPException)
                self.assertIn("structured event after completion", str(failure.detail))

            async def insufficient_engine_deltas():
                FakeAsyncClient.lines = [proof, pcm, 'data: {"type":"speech.audio.done"}', "data: [DONE]"]
                return await supervisor._native_clone_pcm_response(
                    {
                        "input": "test",
                        "ref_audio": "placeholder",
                        "voice": "clone:test",
                        "_engine_epoch": 1,
                    },
                    "qwen3-tts-1.7b-base-bf16",
                )

            with self.assertRaises(supervisor.HTTPException) as insufficient:
                asyncio.run(insufficient_engine_deltas())
            self.assertEqual(insufficient.exception.detail["state"], "native-stream-not-incremental")
        finally:
            supervisor.httpx.AsyncClient = originals["client"]
            supervisor._prepare_reference_pair = originals["prepare"]
            supervisor._revalidate_generation_admission = originals["revalidate"]
            supervisor._record_event = originals["record"]
            temporary_reference.unlink(missing_ok=True)

    def test_native_preheader_disconnect_cancels_each_stage_and_releases_generation_lock(self) -> None:
        class DisconnectRequest:
            async def is_disconnected(self) -> bool:
                return True

        class BlockingResponse:
            def __init__(self, prefix: list[str]) -> None:
                self.prefix = prefix
                self.headers = {}
                self.status_code = 200
                self.is_error = False
                self.closed = False

            async def aiter_lines(self):
                for line in self.prefix:
                    yield line
                await asyncio.Event().wait()

            async def aclose(self) -> None:
                self.closed = True

        class StageClient:
            block_send = False
            prefix: list[str] = []

            def __init__(self, **_kwargs) -> None:
                self.response = BlockingResponse(type(self).prefix)

            def build_request(self, method: str, url: str, json: dict):
                return (method, url, json)

            async def send(self, _request, stream: bool = False):
                if type(self).block_send:
                    await asyncio.Event().wait()
                return self.response

            async def aclose(self) -> None:
                return None

        originals = {
            "client": supervisor.httpx.AsyncClient,
            "prepare": supervisor._prepare_reference_pair,
            "revalidate": supervisor._revalidate_generation_admission,
            "record": supervisor._record_event,
        }
        temporary_reference = INTEGRATION / ".native-disconnect-test-reference.wav"
        proof = 'data: {"type":"speech.decode_mode","mode":"native-incremental-pcm"}'
        pcm = 'data: {"type":"speech.audio.delta","audio":"AQACAA=="}'

        async def run_stage(block_send: bool, prefix: list[str]) -> None:
            StageClient.block_send = block_send
            StageClient.prefix = prefix
            with self.assertRaises(asyncio.CancelledError):
                await supervisor._native_clone_pcm_response(
                    {
                        "input": "disconnect",
                        "ref_audio": "placeholder",
                        "voice": "clone:test",
                        "_engine_epoch": 1,
                    },
                    "qwen3-tts-1.7b-base-bf16",
                    request=DisconnectRequest(),
                )
            await asyncio.wait_for(supervisor.generation_lock.acquire(), timeout=0.25)
            supervisor.generation_lock.release()

        async def exercise() -> None:
            await run_stage(True, [])
            await run_stage(False, [])
            await run_stage(False, [proof, pcm])

        try:
            supervisor.httpx.AsyncClient = StageClient
            supervisor._prepare_reference_pair = lambda *_args, **_kwargs: (
                temporary_reference,
                {
                    "source_seconds": 1.0, "requested_limit_seconds": None,
                    "used_seconds": 1.0, "limit_applied": False,
                    "pairing": "full", "truncated": False,
                },
                "reference",
            )
            supervisor._revalidate_generation_admission = lambda *_args, **_kwargs: None
            supervisor._record_event = lambda *_args, **_kwargs: None
            asyncio.run(exercise())
        finally:
            supervisor.httpx.AsyncClient = originals["client"]
            supervisor._prepare_reference_pair = originals["prepare"]
            supervisor._revalidate_generation_admission = originals["revalidate"]
            supervisor._record_event = originals["record"]
            temporary_reference.unlink(missing_ok=True)

    def test_native_relay_requires_engine_sse_proof_before_headers_or_pcm(self) -> None:
        source = (Path(__file__).parents[1] / "supervisor.py").read_text(encoding="utf-8")
        self.assertIn('stream_format="sse"', source)
        self.assertIn('event_type == "speech.decode_mode"', source)
        self.assertIn('event_type in {"speech.audio.delta", "speech.audio.done"}', source)
        self.assertIn('"decoder-mode-proof-missing"', source)
        self.assertIn('stream_lines = _deadline_lines(response.aiter_lines(),', source)
        self.assertIn('line = await lines.__anext__()', source)
        self.assertIn('"X-TTS-Streaming-Mode": "native-incremental-pcm"', source)
        self.assertIn('"X-TTS-Decoder-Mode": verified_decode_mode', source)
        self.assertIn('"X-TTS-Native-Engine-Chunk-Proof": "two-distinct-sse-delta-events"', source)

    def test_thin_overlay_rebuilds_and_copies_the_patched_engine(self) -> None:
        overlay = (INTEGRATION / "Dockerfile.overlay").read_text(encoding="utf-8")
        self.assertIn("AS engine-build", overlay)
        self.assertIn("AUDIO_CPP_REF=238ab6a9e321c17de8e120559f57efeedaeb1345", overlay)
        self.assertIn("qwen3-native-pcm-streaming.patch", overlay)
        self.assertIn("--target audiocpp_server", overlay)
        self.assertIn(
            "COPY --from=engine-build /opt/audio.cpp/build/linux-cuda-release/bin/audiocpp_server /usr/local/bin/audiocpp_server",
            overlay,
        )
        self.assertIn('org.opencontainers.image.revision="${AUDIO_CPP_REF}"', overlay)
        self.assertIn('org.opencontainers.image.version="${AUDIO_CPP_OVERLAY_REV}"', overlay)
        self.assertIn('ARG AUDIO_CPP_BASE_IMAGE\n', overlay)
        self.assertNotIn('ARG AUDIO_CPP_BASE_IMAGE=', overlay)
        self.assertIn('io.omninexus.rollback.base="${AUDIO_CPP_BASE_IMAGE_ID}"', overlay)
        self.assertIn("test -d /opt/models/Qwen3-TTS-12Hz-0.6B-Base", overlay)
        self.assertIn("test -d /opt/models/Qwen3-TTS-12Hz-1.7B-Base", overlay)
        self.assertIn("test -f /config/qwen3-tts-base-f16.json", overlay)
        self.assertIn("test -x /usr/local/bin/audio-cpp-candidate", overlay)
        self.assertIn('ENTRYPOINT ["/usr/local/bin/audio-cpp-candidate"]', overlay)

    def test_runtime_overlay_is_hash_pinned_to_the_validated_native_engine(self) -> None:
        overlay = (INTEGRATION / "Dockerfile.runtime-overlay").read_text(encoding="utf-8")
        self.assertIn(
            "AUDIO_CPP_ENGINE_SHA256=28fb9d4a2940b8f0d6f426cfc7da21361438f878804cf6ac59888d5e807bb1a8",
            overlay,
        )
        self.assertIn("sha256sum -c -", overlay)
        self.assertIn("/opt/models/Qwen3-TTS-12Hz-0.6B-Base", overlay)
        self.assertIn("/opt/models/Qwen3-TTS-12Hz-1.7B-Base", overlay)
        self.assertIn("test -f /config/qwen3-tts-base-f16.json", overlay)
        self.assertIn("test -x /usr/local/bin/audio-cpp-candidate", overlay)
        self.assertIn('ARG AUDIO_CPP_BASE_IMAGE\n', overlay)
        self.assertNotIn('ARG AUDIO_CPP_BASE_IMAGE=', overlay)
        self.assertIn('io.omninexus.rollback.base="${AUDIO_CPP_BASE_IMAGE_ID}"', overlay)
        self.assertNotIn("AS engine-build", overlay)
        self.assertNotIn("apt-get", overlay)

    def test_model_inventory_mode_matches_both_engine_configuration_states(self) -> None:
        original_models = supervisor._models
        supervisor._models = lambda: {
            "qwen3-tts-0.6b-base-bf16": {},
            "qwen3-tts-1.7b-base-bf16": {},
        }
        try:
            for enabled, expected in ((False, "offline"), (True, "streaming")):
                with self.subTest(enabled=enabled):
                    supervisor.NATIVE_INCREMENTAL_PCM_ENABLED = enabled
                    inventory = asyncio.run(supervisor.models())
                    self.assertEqual({item["mode"] for item in inventory["data"]}, {expected})
                    self.assertEqual(supervisor._configured_model_mode(), expected)
        finally:
            supervisor._models = original_models

    def test_enforced_load_reserves_residency_plus_unchanged_synthesis_floor(self) -> None:
        original_check_output = supervisor.subprocess.check_output
        original_settings = dict(supervisor.gpu_guard_settings)
        reading = {"free": 7266, "utilization": 0}
        supervisor.subprocess.check_output = lambda *_args, **_kwargs: (
            f"{reading['free']}, {reading['utilization']}"
        )
        try:
            supervisor.gpu_guard_settings.clear()
            supervisor.gpu_guard_settings.update(supervisor._default_gpu_guard_settings(), mode="enforced")
            blocked = supervisor.gpu_guard("qwen3-tts-0.6b-base-bf16", operation="load")
            self.assertFalse(blocked["ok"])
            self.assertEqual(blocked["requiredMiB"], 8048)
            self.assertEqual(blocked["residencyReserveMiB"], 6000)
            self.assertEqual(blocked["postLoadSynthesisReserveMiB"], 2048)
            self.assertIn("6000 MiB measured/rounded model residency", blocked["reason"])

            reading["free"] = 8944
            admitted = supervisor.gpu_guard("qwen3-tts-0.6b-base-bf16", operation="load")
            self.assertTrue(admitted["ok"])
            self.assertEqual(supervisor._guard_policy("qwen3-tts-1.7b-base-bf16", "load")[0], 10500)
            reading["free"] = 10499
            blocked_17b = supervisor.gpu_guard("qwen3-tts-1.7b-base-bf16", operation="load")
            self.assertFalse(blocked_17b["ok"])
            self.assertEqual(blocked_17b["admissionKind"], "existing-total-threshold")
            self.assertIsNone(blocked_17b["residencyReserveMiB"])
            self.assertIn("residency delta has not been measured", blocked_17b["reason"])
            self.assertEqual(supervisor._guard_policy("qwen3-tts-0.6b-base-bf16", "synthesis")[0], 2048)

            supervisor.gpu_guard_settings.update(
                mode="custom", load_min_free_mib=7000, synthesis_min_free_mib=3000
            )
            reading["free"] = 7266
            custom = supervisor.gpu_guard("qwen3-tts-0.6b-base-bf16", operation="load")
            self.assertTrue(custom["ok"])
            self.assertEqual(custom["requiredMiB"], 7000)
            self.assertTrue(custom["customAbsoluteLoadThreshold"])
            self.assertEqual(supervisor._guard_policy("qwen3-tts-0.6b-base-bf16", "synthesis")[0], 3000)

            supervisor.gpu_guard_settings["mode"] = "disabled"
            reading["free"] = 100
            bypassed = supervisor.gpu_guard("qwen3-tts-0.6b-base-bf16", operation="load")
            self.assertTrue(bypassed["ok"])
            self.assertTrue(bypassed["bypassed"])
        finally:
            supervisor.subprocess.check_output = original_check_output
            supervisor.gpu_guard_settings.clear()
            supervisor.gpu_guard_settings.update(original_settings)

    def test_voice_studio_probe_uses_health_and_clone_adapter_contracts(self) -> None:
        server = SERVER.read_text(encoding="utf-8")
        probe = server.split('async def voice_studio_test', 1)[1].split('async def _probe_tts_backend', 1)[0]
        self.assertIn('current_model = backend.get("model_id") or backend.get("current_model_key")', probe)
        self.assertIn('current_model != req.model_id', probe)
        self.assertIn('"voice": f"clone:{req.profile_id}"', probe)
        self.assertIn('"response_format": "wav"', probe)
        self.assertNotIn('runtime.get("current")', probe)


if __name__ == "__main__":
    unittest.main()
