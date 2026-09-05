"""Backend contracts for true offline clone synthesis and completed-master export."""
from __future__ import annotations

import asyncio
import base64
import copy
import importlib.util
import io
import unittest
import wave
from pathlib import Path

from fastapi import HTTPException

SUPERVISOR = Path(__file__).parents[1] / "supervisor.py"
SPEC = importlib.util.spec_from_file_location("candidate_supervisor_offline_output_test", SUPERVISOR)
assert SPEC and SPEC.loader
supervisor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(supervisor)


def wav_bytes(*, seconds: int = 1, rate: int = 24000, channels: int = 1) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(b"\x01\x00" * seconds * rate * channels)
    return output.getvalue()


class _FakeResponse:
    def __init__(self, content: bytes, decode_mode: str | None = None) -> None:
        self.content = content
        self.status_code = 200
        self.is_error = False
        self.headers = {"content-type": "audio/wav", "X-AudioCPP-Termination": "eos",
                        "X-AudioCPP-Generated-Frames": "13", "X-AudioCPP-Generation-Cap": "150"}
        if decode_mode:
            self.headers[supervisor.ENGINE_DECODE_MODE_HEADER] = decode_mode


class _FakeAsyncClient:
    request: dict | None = None
    decode_mode = supervisor.OFFLINE_FULL_DECODE_MODE
    output = wav_bytes()

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def post(self, _url: str, json: dict) -> _FakeResponse:
        type(self).request = copy.deepcopy(json)
        return _FakeResponse(type(self).output, type(self).decode_mode)


class _JsonRequest:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    async def json(self) -> dict:
        return copy.deepcopy(self.payload)

    async def is_disconnected(self) -> bool:
        return False


class OfflineOutputContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_client = supervisor.httpx.AsyncClient
        self.original_read = supervisor._read_tuning_profiles
        self.original_assert = supervisor._assert_synthesis_ready
        self.original_voice_response = supervisor._voice_clone_response
        self.original_state = dict(supervisor.state)
        self.document = supervisor._default_tuning_profiles()
        supervisor.httpx.AsyncClient = _FakeAsyncClient
        supervisor._read_tuning_profiles = lambda: self.document
        supervisor.state.update(activeModel="qwen3-tts-1.7b-base-bf16", state="loaded")
        _FakeAsyncClient.request = None
        _FakeAsyncClient.decode_mode = supervisor.OFFLINE_FULL_DECODE_MODE
        self.reference = base64.b64encode(wav_bytes()).decode("ascii")

    def tearDown(self) -> None:
        supervisor.httpx.AsyncClient = self.original_client
        supervisor._read_tuning_profiles = self.original_read
        supervisor._assert_synthesis_ready = self.original_assert
        supervisor._voice_clone_response = self.original_voice_response
        supervisor.state.clear()
        supervisor.state.update(self.original_state)

    def payload(self, response_format: str = "wav") -> dict:
        payload = {
            "input": "One complete offline utterance.",
            "ref_audio": self.reference,
            "ref_text": "Exact reference transcript.",
            "response_format": response_format,
            "stream": True,
            "seed": 19,
            "_private_marker": "must-not-leak",
            "tuning": {
                "provider": supervisor.TUNING_PROVIDER,
                "scope": "voice-studio",
                "profile_id": "low-latency",
                "overrides": {"temperature": 1.4, "seed": 7},
            },
        }
        payload["_tuning_snapshot"] = supervisor._resolve_request_tuning(
            payload, force_offline_full=True,
        )
        return payload

    def test_voice_clone_endpoint_forces_offline_policy_for_pcm_too(self) -> None:
        captured: dict = {}

        async def ready(_payload: dict) -> str:
            return "qwen3-tts-1.7b-base-bf16"

        async def respond(payload: dict, *, request=None):
            captured.update(payload)
            self.assertIsNotNone(request)
            return supervisor.Response(content=b"ok")

        supervisor._assert_synthesis_ready = ready
        supervisor._voice_clone_response = respond
        asyncio.run(supervisor.voice_clone(_JsonRequest({
            "response_format": "pcm",
            "stream": True,
            "tuning": {"profile_id": "low-latency"},
        })))
        snapshot = captured["_tuning_snapshot"]
        self.assertEqual(snapshot["delivery_mode"], supervisor.OFFLINE_FULL_DECODE_MODE)
        self.assertEqual(snapshot["policy"], "offline-full-quality")

    def test_speech_endpoint_keeps_nonstreaming_pcm_as_buffered_fallback(self) -> None:
        captured: dict = {}

        async def ready(_payload: dict) -> str:
            return "qwen3-tts-1.7b-base-bf16"

        async def respond(payload: dict, *, request=None):
            captured.update(payload)
            self.assertIsNotNone(request)
            return supervisor.Response(content=b"ok")

        supervisor._assert_synthesis_ready = ready
        supervisor._voice_clone_response = respond
        asyncio.run(supervisor.speech(_JsonRequest({
            "input": "buffer this phrase",
            "ref_audio": self.reference,
            "ref_text": "Exact reference transcript.",
            "response_format": "pcm",
            "stream": False,
            "tuning": {"profile_id": "low-latency"},
        })))
        snapshot = captured["_tuning_snapshot"]
        self.assertEqual(snapshot["delivery_mode"], "buffered-fallback")
        self.assertEqual(snapshot["policy"], "profile-streaming")

    def test_buffered_disconnect_cancels_upstream_and_releases_generation_lock(self) -> None:
        class BlockingClient:
            started = asyncio.Event()
            cancelled = False

            async def post(self, _url: str, json: dict):
                del json
                type(self).started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    type(self).cancelled = True
                    raise

        class DisconnectedRequest:
            async def is_disconnected(self) -> bool:
                return BlockingClient.started.is_set()

        async def scenario() -> None:
            async def abandoned_request() -> None:
                async with supervisor.generation_lock:
                    await supervisor._post_engine_with_disconnect(
                        BlockingClient(),
                        {"input": "abandoned"},
                        request=DisconnectedRequest(),
                    )

            task = asyncio.create_task(abandoned_request())
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=1.0)
            self.assertTrue(BlockingClient.cancelled)
            await asyncio.wait_for(supervisor.generation_lock.acquire(), timeout=0.25)
            supervisor.generation_lock.release()

        asyncio.run(scenario())

    def test_offline_request_uses_one_wav_master_and_proven_decoder_mode(self) -> None:
        response = asyncio.run(supervisor._voice_clone_response(self.payload("pcm")))
        upstream = _FakeAsyncClient.request
        assert upstream is not None
        self.assertFalse(upstream["stream"])
        self.assertEqual(upstream["response_format"], "wav")
        self.assertEqual(upstream["options"]["qwen3_tts.decode_mode"], "offline_full")
        self.assertFalse(any(key.startswith("_") for key in upstream))
        # Full quality pins every quality sampler, while an explicit seed
        # remains request identity and must survive.
        self.assertEqual(upstream["temperature"], 0.8)
        self.assertEqual(upstream["seed"], 7)
        self.assertEqual(response.headers["x-tts-seed"], "7")
        self.assertEqual(response.headers["x-tts-delivery-mode"], "offline-full-decoder")
        self.assertEqual(response.headers["x-tts-decoder-mode"], "offline-full-decoder")
        self.assertEqual(response.headers["x-audiocpp-qwen3-decode-mode"], "offline-full-decoder")
        self.assertEqual(response.headers["x-tts-format"], "pcm")
        self.assertEqual(response.headers["x-tts-codec"], "pcm_s16le")
        self.assertEqual(response.headers["x-tts-container"], "raw PCM")
        self.assertEqual(response.headers["x-tts-sample-rate"], "24000")
        self.assertEqual(response.headers["x-tts-bits-per-sample"], "16")
        self.assertEqual(response.headers["x-tts-channels"], "1")

    def test_stream_false_pcm_speech_policy_remains_buffered_fallback(self) -> None:
        payload = self.payload("pcm")
        payload["stream"] = False
        payload["_tuning_snapshot"] = supervisor._resolve_request_tuning(payload)
        response = asyncio.run(supervisor._voice_clone_response(payload))
        upstream = _FakeAsyncClient.request
        assert upstream is not None
        self.assertEqual(
            upstream["options"]["qwen3_tts.decode_mode"],
            "offline_full",
        )
        self.assertEqual(response.headers["x-tts-delivery-mode"], "buffered-fallback")
        self.assertEqual(response.headers["x-tts-decoder-mode"], "offline-full-decoder")
        self.assertEqual(
            response.headers["x-audiocpp-qwen3-decode-mode"],
            "offline-full-decoder",
        )

    def test_buffered_pcm_rejects_native_decoder_proof(self) -> None:
        payload = self.payload("pcm")
        payload["stream"] = False
        payload["_tuning_snapshot"] = supervisor._resolve_request_tuning(payload)
        _FakeAsyncClient.decode_mode = "native-incremental-pcm"
        with self.assertRaises(HTTPException) as error:
            asyncio.run(supervisor._voice_clone_response(payload))
        self.assertEqual(error.exception.status_code, 502)
        self.assertEqual(error.exception.detail["state"], "decoder-mode-mismatch")

    def test_proven_offline_decoder_mismatch_is_a_hard_error(self) -> None:
        _FakeAsyncClient.decode_mode = "native-incremental-pcm"
        with self.assertRaises(HTTPException) as error:
            asyncio.run(supervisor._voice_clone_response(self.payload("wav")))
        self.assertEqual(error.exception.status_code, 502)
        self.assertEqual(error.exception.detail["state"], "decoder-mode-mismatch")

    def test_missing_offline_decoder_proof_is_a_hard_error(self) -> None:
        _FakeAsyncClient.decode_mode = None
        with self.assertRaises(HTTPException) as error:
            asyncio.run(supervisor._voice_clone_response(self.payload("wav")))
        self.assertEqual(error.exception.status_code, 502)
        self.assertEqual(error.exception.detail["state"], "decoder-mode-proof-missing")

    def test_invalid_master_and_unknown_format_fail_without_fallback(self) -> None:
        _FakeAsyncClient.output = b"not-a-wave"
        try:
            with self.assertRaises(HTTPException) as invalid_master:
                asyncio.run(supervisor._voice_clone_response(self.payload("wav")))
            self.assertEqual(invalid_master.exception.status_code, 502)
            first_request = copy.deepcopy(_FakeAsyncClient.request)
            self.assertIsNotNone(first_request)
            with self.assertRaises(HTTPException) as unsupported:
                asyncio.run(supervisor._voice_clone_response(self.payload("webm")))
            self.assertEqual(unsupported.exception.status_code, 400)
            self.assertEqual(_FakeAsyncClient.request, first_request)
        finally:
            _FakeAsyncClient.output = wav_bytes()

    def test_seed_sentinels_and_uint32_forwarding(self) -> None:
        for value in (None, "", "  ", -1, "-1"):
            payload = {"seed": value}
            supervisor._normalize_gradio_seed(payload)
            self.assertNotIn("seed", payload)
        for value in (0, 2**32 - 1, "42", 42.0):
            payload = {"seed": value}
            supervisor._normalize_gradio_seed(payload)
            self.assertEqual(payload["seed"], int(value))
        for value in (-2, 2**32, 1.5, float("inf"), float("nan"), True, "random"):
            with self.subTest(value=value), self.assertRaises(HTTPException):
                supervisor._normalize_gradio_seed({"seed": value})

    def test_full_wav_preserves_direct_uint32_seed_and_omits_random_sentinels(self) -> None:
        for value in (0, 2**32 - 1):
            with self.subTest(value=value):
                payload = self.payload("wav")
                payload["tuning"]["overrides"].pop("seed")
                payload["seed"] = value
                payload["_tuning_snapshot"] = supervisor._resolve_request_tuning(
                    payload, force_offline_full=True,
                )
                response = asyncio.run(supervisor._voice_clone_response(payload))
                self.assertEqual(_FakeAsyncClient.request["seed"], value)
                self.assertEqual(response.headers["x-tts-seed"], str(value))

        for value in (None, "", "  ", -1, "-1"):
            with self.subTest(value=value):
                payload = self.payload("wav")
                payload["tuning"]["overrides"].pop("seed")
                payload["seed"] = value
                payload["_tuning_snapshot"] = supervisor._resolve_request_tuning(
                    payload, force_offline_full=True,
                )
                response = asyncio.run(supervisor._voice_clone_response(payload))
                self.assertNotIn("seed", _FakeAsyncClient.request)
                self.assertNotIn("x-tts-seed", response.headers)

    def test_completed_master_exports_all_six_formats_without_resampling(self) -> None:
        master = wav_bytes(channels=2)
        wav, _, extension, metadata = supervisor._render_master_wav(master, "wav")
        self.assertEqual(wav, master)
        self.assertEqual(extension, "wav")
        self.assertEqual(metadata["channels"], 2)
        pcm, _, extension, _ = supervisor._render_master_wav(master, "pcm")
        self.assertEqual(extension, "pcm")
        self.assertEqual(len(pcm), 24000 * 2 * 2)

        original_run = supervisor.subprocess.run
        commands: list[list[str]] = []

        def encode(command: list[str], **_kwargs):
            commands.append(command)
            Path(command[-1]).write_bytes((Path(command[-1]).suffix + "-encoded").encode())
            return type("Completed", (), {"returncode": 0, "stderr": ""})()

        supervisor.subprocess.run = encode
        try:
            expected_metadata = {
                "flac": ("flac", "FLAC"),
                "mp3": ("mp3", "MP3"),
                "aac": ("aac", "ADTS"),
                "opus": ("opus", "OGG"),
            }
            for fmt in ("flac", "mp3", "aac", "opus"):
                content, _, extension, encoded = supervisor._render_master_wav(master, fmt)
                self.assertTrue(content)
                self.assertEqual(extension, fmt)
                self.assertEqual(encoded["sampleRate"], 24000)
                self.assertEqual(encoded["channels"], 2)
                self.assertEqual(
                    (encoded["codec"], encoded["container"]), expected_metadata[fmt],
                )
        finally:
            supervisor.subprocess.run = original_run

        for command in commands:
            self.assertNotIn("-ar", command)
            self.assertNotIn("-ac", command)
        lossy = [command for command in commands if any(codec in command for codec in ("libmp3lame", "aac", "libopus"))]
        self.assertEqual(len(lossy), 3)
        for command in lossy:
            expected = "256k" if "libopus" in command else "320k"
            self.assertEqual(command[command.index("-b:a") + 1], expected)


if __name__ == "__main__":
    unittest.main()
