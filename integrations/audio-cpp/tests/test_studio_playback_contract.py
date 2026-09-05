"""Focused contracts for Voice Studio mode separation and continuous playback."""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[3]
INTEGRATION = Path(__file__).parents[1]
UI = INTEGRATION / "gradio_voice_studio.py"
WORKLET = ROOT / "web" / "hf-realtime-voice" / "worklets" / "studio-playback.js"
NODE_TEST = Path(__file__).with_name("studio_playback.test.mjs")
SNAPSHOT_NODE_TEST = Path(__file__).with_name("studio_response_snapshot.test.mjs")

gradio_stub = types.ModuleType("gradio")
gradio_stub.Blocks = object
gradio_stub.themes = types.SimpleNamespace(Base=object)
sys.modules.setdefault("gradio", gradio_stub)

ui_spec = importlib.util.spec_from_file_location("candidate_studio_playback_test", UI)
assert ui_spec and ui_spec.loader
studio = importlib.util.module_from_spec(ui_spec)
ui_spec.loader.exec_module(studio)


class StudioPlaybackContractTests(unittest.TestCase):
    def test_full_quality_has_dedicated_offline_policy_and_preserves_format(self) -> None:
        payload = {
            "input": "hello",
            "response_format": "pcm",
            "stream": True,
            "tuning": {"profile_id": "low-latency", "overrides": {"temperature": 1.4}},
        }
        returned = studio.apply_full_wav_quality_policy(payload, "flac")
        self.assertIs(returned, payload)
        self.assertEqual(payload["response_format"], "flac")
        self.assertFalse(payload["stream"])
        self.assertEqual(
            payload["tuning"],
            {
                "provider": "qwen3tts-audiocpp",
                "scope": "voice-studio",
                "profile_id": "quality",
                "overrides": {},
            },
        )
        with self.assertRaises(ValueError):
            studio.apply_full_wav_quality_policy({}, "streaming-wav")

    def test_full_wav_callback_cannot_inherit_streaming_session_state(self) -> None:
        source = UI.read_text(encoding="utf-8")
        callback = source.split("def on_play_generate(", 1)[1].split("# Streaming mode callbacks", 1)[0]
        self.assertIn("apply_full_wav_quality_policy(payload, response_format)", callback)
        self.assertNotIn("apply_session_tuning(", callback)
        self.assertNotIn("request_tts_streaming(", callback)
        self.assertIn("offline-full-decoder", callback)
        self.assertIn("Quality (dedicated Full Quality)", callback)
        wiring = source.split("play_generate_btn.click(", 1)[1].split("# Streaming mode wiring", 1)[0]
        self.assertNotIn("tuning_profile_dropdown", wiring)
        self.assertNotIn("tuning_override_state", wiring)
        self.assertIn("play_response_format = gr.Dropdown", source)
        for output_format in ("wav", "pcm", "flac", "mp3", "aac", "opus"):
            self.assertIn(f'"{output_format}")', source)
        self.assertIn("play_response_format, play_speed", wiring)

    def test_base_clone_preview_is_offline_only_without_streaming_fallback(self) -> None:
        source = UI.read_text(encoding="utf-8")
        callback = source.split("def on_generate_clone(", 1)[1].split("def on_save_clone_profile", 1)[0]
        self.assertIn('apply_full_wav_quality_policy(payload, "wav")', callback)
        self.assertIn("request_tts_voice_clone(", callback)
        self.assertNotIn("request_tts_streaming(", callback)
        self.assertNotIn("fallback", callback.lower())

    def test_active_widget_uses_profile_phrase_policy_and_continuous_worklet(self) -> None:
        source = UI.read_text(encoding="utf-8")
        active_widget = source.rsplit("def _build_streaming_widget_html(", 1)[1].split("# Callback implementations", 1)[0]
        self.assertNotIn("getPhraseSettings", active_widget)
        self.assertNotIn("phrase-min", active_widget)
        self.assertNotIn("phrase-max", active_widget)
        self.assertNotIn("phrase-idle", active_widget)
        self.assertNotIn("createBufferSource", active_widget)
        self.assertIn("findImmediateBoundary", active_widget)
        self.assertIn("findSafePhraseCut", active_widget)
        self.assertIn("phrasePolicyForResponse", active_widget)
        self.assertIn("snapshot.phrasePolicy", active_widget)
        self.assertIn("requestSnapshotForPhraseDispatch", active_widget)
        self.assertIn("ensurePlaybackQueue", active_widget)
        self.assertIn("enqueuePcmForTurn", active_widget)
        self.assertIn("finishPlaybackTurn", active_widget)
        self.assertIn("kind: 'clear'", active_widget)
        self.assertIn("config.playbackStartupMs", active_widget)
        self.assertIn("max(1, first_block_frames) * 80", active_widget)
        self.assertNotIn("first_block_frames + steady_block_frames", active_widget)

    def test_live_profile_or_mode_change_keeps_the_active_widget_and_snapshot(self) -> None:
        source = UI.read_text(encoding="utf-8")
        wiring = source.split("# Streaming mode wiring", 1)[1].split("tuning_context_unlock.change", 1)[0]
        self.assertIn("queue_streaming_widget_next_config", wiring)
        self.assertNotIn("fn=on_update_streaming_widget", wiring)
        self.assertIn("pendingNextResponseConfig", source)
        self.assertIn("function queueNextResponseConfig", source)
        self.assertIn("function applyNextResponseConfig", source)
        self.assertIn("applyNextResponseConfig();", source.split("function sendToLLM", 1)[1].split("function", 1)[0])

    def test_response_snapshot_uses_only_supervisor_frozen_tuning_fields(self) -> None:
        source = UI.read_text(encoding="utf-8")
        snapshot = source.split("function freezeResponseSnapshot", 1)[1].split(
            "function ttsPayloadForPhrase", 1
        )[0]
        self.assertNotIn("'crossfade_samples'", snapshot)
        for field in (
            "model", "clone_mode", "max_reference_seconds", "first_block_frames",
            "steady_block_frames", "left_context_frames", "text_lookahead",
            "phrase_flush_ms", "temperature", "top_k", "top_p",
            "repetition_penalty", "seed",
        ):
            self.assertIn(f"'{field}'", snapshot)
        self.assertIn("clone_snapshot: responseSnapshot.cloneSnapshot", source)
        self.assertIn("expected_engine_epoch: responseSnapshot.engineEpoch", source)
        self.assertIn("expected_supervisor_instance_id: responseSnapshot.supervisorInstanceId", source)

    def test_phrase_provider_failure_is_terminal_for_the_active_response(self) -> None:
        source = UI.read_text(encoding="utf-8")
        helper = source.split("function terminatePhrasesAfterProviderFailure", 1)[1].split(
            "function synthesizeSpeech", 1
        )[0]
        self.assertIn("state.ttsFailureTerminal = true", helper)
        self.assertIn("state.phraseQueue = []", helper)
        self.assertIn("state.phraseText = ''", helper)
        self.assertIn("state.activeLlmRequest.abort()", helper)
        self.assertIn("finishPlaybackTurn(turnId, responseSnapshot)", helper)
        phrase_pump = source.split("function pumpProgressiveSpeech", 1)[1].split(
            "function synthesizeSpeech", 1
        )[0]
        self.assertIn("if (state.ttsFailureTerminal) return", phrase_pump)
        self.assertIn("if (!state.ttsFailureTerminal) pumpProgressiveSpeech()", phrase_pump)

    def test_worklet_is_packaged_by_both_candidate_images(self) -> None:
        self.assertTrue(WORKLET.is_file())
        dockerfile = (INTEGRATION / "Dockerfile").read_text(encoding="utf-8")
        overlay = (INTEGRATION / "Dockerfile.overlay").read_text(encoding="utf-8")
        self.assertIn("COPY web/hf-realtime-voice/ .", dockerfile)
        self.assertIn("COPY web/hf-realtime-voice/ /opt/voice-studio/", overlay)
        self.assertIn(studio.STUDIO_PLAYBACK_WORKLET_URL.split("?", 1)[0], UI.read_text(encoding="utf-8"))

    def test_streaming_controls_are_disabled_for_full_wav(self) -> None:
        source = UI.read_text(encoding="utf-8")
        callback = source.split("def on_play_mode_change", 1)[1].split("def on_s_voice_change", 1)[0]
        self.assertIn("is_streaming = mode != FULL_WAV_PLAYBACK_MODE", callback)
        self.assertIn("gr.update(interactive=is_native)", callback)
        self.assertIn("gr.update(interactive=is_streaming)", callback)
        self.assertIn("Full Quality is independent", callback)
        self.assertIn("All streaming and tuning-profile controls are locked", callback)
        self.assertIn("tuning_context_unlock", source)
        self.assertIn("first_block_frames", source.split("def resolve_tuning_override", 1)[1])
        self.assertIn("max_reference_seconds", source.split("def resolve_tuning_override", 1)[1])
        self.assertIn('profile.get("left_context_frames", 25)', source)
        self.assertIn('value=25', source)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for AudioWorklet behavior tests")
    def test_audio_worklet_behavior(self) -> None:
        completed = subprocess.run(
            [shutil.which("node"), str(NODE_TEST)],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("studio playback worklet tests passed", completed.stdout)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for Voice Studio browser tests")
    def test_response_snapshot_payload_behavior(self) -> None:
        completed = subprocess.run(
            [shutil.which("node"), str(SNAPSHOT_NODE_TEST)],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("studio response snapshot payload tests passed", completed.stdout)
        self.assertIn("studio provider failure terminal tests passed", completed.stdout)


if __name__ == "__main__":
    unittest.main()
