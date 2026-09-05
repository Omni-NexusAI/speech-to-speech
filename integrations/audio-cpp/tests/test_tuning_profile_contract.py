"""Focused behavioral contracts for candidate-owned tuning admission."""
from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path

from fastapi import HTTPException

SUPERVISOR = Path(__file__).parents[1] / "supervisor.py"
SPEC = importlib.util.spec_from_file_location("candidate_supervisor_tuning_test", SUPERVISOR)
assert SPEC and SPEC.loader
supervisor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(supervisor)
UI = Path(__file__).parents[1] / "gradio_voice_studio.py"
UI_SPEC = importlib.util.spec_from_file_location("candidate_gradio_tuning_test", UI)
assert UI_SPEC and UI_SPEC.loader
studio = importlib.util.module_from_spec(UI_SPEC)
gradio_stub = types.ModuleType("gradio")
gradio_stub.Blocks = object
gradio_stub.themes = types.SimpleNamespace(Base=object)
sys.modules.setdefault("gradio", gradio_stub)
UI_SPEC.loader.exec_module(studio)


class TuningProfileContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = supervisor._default_tuning_profiles()
        self.document["profiles"]["balanced"].update(
            temperature=0.8,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.05,
            text_lookahead=64,
            phrase_flush_ms=450,
            seed=None,
        )
        self.original_read = supervisor._read_tuning_profiles
        self.original_write = supervisor._write_tuning_profiles
        self.original_state = dict(supervisor.state)
        supervisor._read_tuning_profiles = lambda: self.document
        supervisor._write_tuning_profiles = lambda document: document
        supervisor.state["activeModel"] = "qwen3-tts-1.7b-base-bf16"

    def tearDown(self) -> None:
        supervisor._read_tuning_profiles = self.original_read
        supervisor._write_tuning_profiles = self.original_write
        supervisor.state.clear()
        supervisor.state.update(self.original_state)

    def test_duplicate_create_never_overwrites_builtin_or_custom(self) -> None:
        with self.assertRaises(HTTPException) as error:
            asyncio.run(supervisor.create_tuning_profile({"id": "balanced"}))
        self.assertEqual(error.exception.status_code, 409)

    def test_save_as_persists_complete_effective_schema_and_one_scope_atomically(self) -> None:
        saved = asyncio.run(supervisor.create_tuning_profile({
            "id": "my-realtime-test",
            "name": "My Realtime Test",
            "clone_from": "balanced",
            "values": {
                "model": "qwen3-tts-1.7b-base-bf16",
                "clone_mode": "full_icl",
                "max_reference_seconds": 17,
                "first_block_frames": 3,
                "steady_block_frames": 10,
                "left_context_frames": 72,
                "text_lookahead": 51,
                "phrase_flush_ms": 410,
                "temperature": 0.72,
                "top_k": 37,
                "top_p": 0.88,
                "repetition_penalty": 1.07,
                "seed": 123,
            },
            "scope": "realtime",
            "select": True,
        }))
        profile = saved["profile"]
        self.assertEqual(saved["created_profile_id"], "my-realtime-test")
        self.assertEqual(saved["selected_profile_id"], "my-realtime-test")
        self.assertEqual(saved["selected_scope"], "realtime")
        self.assertEqual(profile["revision"], 1)
        self.assertEqual(profile["seed"], 123)
        self.assertEqual(profile["text_lookahead"], 51)
        self.assertEqual(
            saved["selections"]["realtime"][supervisor.TUNING_PROVIDER],
            "my-realtime-test",
        )
        self.assertEqual(
            saved["selections"]["voice-studio"][supervisor.TUNING_PROVIDER],
            "balanced",
        )

    def test_builtins_are_immutable_and_custom_updates_require_matching_revision(self) -> None:
        with self.assertRaises(HTTPException) as immutable:
            asyncio.run(supervisor.patch_tuning_profile("balanced", {"revision": 2, "temperature": 0.7}))
        self.assertEqual(immutable.exception.status_code, 409)

        asyncio.run(supervisor.create_tuning_profile({"id": "custom"}))
        with self.assertRaises(HTTPException) as missing:
            asyncio.run(supervisor.patch_tuning_profile("custom", {"temperature": 0.7}))
        self.assertEqual(missing.exception.status_code, 422)
        with self.assertRaises(HTTPException) as stale:
            asyncio.run(supervisor.patch_tuning_profile("custom", {"revision": 99, "temperature": 0.7}))
        self.assertEqual(stale.exception.status_code, 409)
        updated = asyncio.run(supervisor.patch_tuning_profile(
            "custom", {"revision": 1, "temperature": 0.7, "seed": "42"},
        ))
        self.assertEqual(updated["profiles"]["custom"]["revision"], 2)
        self.assertEqual(updated["profiles"]["custom"]["seed"], 42)

    def test_clone_accepts_complete_values_and_can_select_only_voice_studio(self) -> None:
        cloned = asyncio.run(supervisor.clone_tuning_profile("quality", {
            "id": "studio-clone",
            "name": "Studio Clone",
            "values": {
                "model": "qwen3-tts-1.7b-base-bf16",
                "clone_mode": "full_icl",
                "max_reference_seconds": 24,
                "first_block_frames": 5,
                "steady_block_frames": 14,
                "left_context_frames": 72,
                "text_lookahead": 96,
                "phrase_flush_ms": 700,
                "temperature": 0.75,
                "top_k": 45,
                "top_p": 0.92,
                "repetition_penalty": 1.08,
                "seed": None,
            },
            "scope": "voice-studio",
            "select": True,
        }))
        self.assertEqual(cloned["profile"]["text_lookahead"], 96)
        self.assertEqual(cloned["selected_profile_id"], "studio-clone")
        self.assertEqual(
            cloned["selections"]["voice-studio"][supervisor.TUNING_PROVIDER],
            "studio-clone",
        )
        self.assertEqual(
            cloned["selections"]["realtime"][supervisor.TUNING_PROVIDER],
            "balanced",
        )

    def test_temperature_must_be_greater_than_zero(self) -> None:
        with self.assertRaises(HTTPException) as zero:
            asyncio.run(supervisor.create_tuning_profile({
                "id": "zero-temperature",
                "values": {"temperature": 0},
            }))
        self.assertEqual(zero.exception.status_code, 422)

    def test_selection_is_provider_scoped(self) -> None:
        with self.assertRaises(HTTPException) as error:
            asyncio.run(supervisor.select_tuning_profile({"provider": "faster", "profile_id": "balanced"}))
        self.assertEqual(error.exception.status_code, 422)

    def test_import_rejects_duplicate_custom_without_changing_scoped_selections(self) -> None:
        asyncio.run(supervisor.create_tuning_profile({
            "id": "existing-custom",
            "name": "Keep this profile",
            "scope": "realtime",
            "select": True,
        }))
        asyncio.run(supervisor.select_tuning_profile({
            "scope": "voice-studio",
            "profile_id": "quality",
        }))
        before_studio = self.document["selections"]["voice-studio"][supervisor.TUNING_PROVIDER]
        before_realtime = self.document["selections"]["realtime"][supervisor.TUNING_PROVIDER]
        imported = dict(self.document["profiles"]["balanced"])
        imported.update(id="existing-custom", name="Must not overwrite")

        with self.assertRaises(HTTPException) as conflict:
            asyncio.run(supervisor.import_tuning_profiles({
                "schema": "qwen3tts-audiocpp.tuning/v1",
                "document": {"profiles": {"existing-custom": imported}},
            }))

        self.assertEqual(conflict.exception.status_code, 409)
        self.assertEqual(conflict.exception.detail["state"], "profile-import-conflict")
        self.assertEqual(self.document["profiles"]["existing-custom"]["name"], "Keep this profile")
        self.assertEqual(
            self.document["selections"]["voice-studio"][supervisor.TUNING_PROVIDER],
            before_studio,
        )
        self.assertEqual(
            self.document["selections"]["realtime"][supervisor.TUNING_PROVIDER],
            before_realtime,
        )

    def test_override_snapshot_validates_and_forwards_only_engine_sampler_fields(self) -> None:
        snapshot = supervisor._resolve_request_tuning({
            "tuning": {
                "provider": "qwen3tts-audiocpp",
                "profile_id": "balanced",
                "overrides": {"temperature": 0.6, "top_k": 31, "phrase_flush_ms": 900},
            }
        })
        self.assertEqual(snapshot["engine_fields"], {"temperature": 0.6, "top_k": 31, "top_p": 0.9, "repetition_penalty": 1.05})
        payload = {"_tuning_snapshot": snapshot, "tuning": {"profile_id": "balanced"}, "input": "hello"}
        supervisor._apply_engine_tuning(payload)
        self.assertNotIn("_tuning_snapshot", payload)
        self.assertNotIn("tuning", payload)
        self.assertEqual(payload["temperature"], 0.6)
        self.assertEqual(payload["top_k"], 31)
        self.assertNotIn("phrase_flush_ms", payload)

    def test_model_mismatch_and_invalid_override_are_rejected_without_switch(self) -> None:
        with self.assertRaises(HTTPException) as mismatch:
            supervisor._resolve_request_tuning({"tuning": {"profile_id": "balanced", "overrides": {"model": "qwen3-tts-0.6b-base-bf16"}}})
        self.assertEqual(mismatch.exception.status_code, 409)
        self.assertEqual(supervisor.state["activeModel"], "qwen3-tts-1.7b-base-bf16")
        with self.assertRaises(HTTPException) as invalid:
            supervisor._resolve_request_tuning({"tuning": {"profile_id": "balanced", "overrides": {"top_k": 201}}})
        self.assertEqual(invalid.exception.status_code, 422)

    def test_builtins_are_complete_and_native_frame_controls_are_truthfully_inactive(self) -> None:
        for profile in self.document["profiles"].values():
            for field in ("model", "text_lookahead", "phrase_flush_ms", "temperature", "top_k", "top_p", "repetition_penalty", "seed"):
                self.assertIn(field, profile)
            self.assertEqual(
                profile["left_context_frames"],
                supervisor.MODEL_REQUIRED_LEFT_CONTEXT_FRAMES,
            )
        self.assertEqual(supervisor.MODEL_REQUIRED_LEFT_CONTEXT_FRAMES, 25)
        resolved = asyncio.run(supervisor.resolve_tuning_profile({"provider": "qwen3tts-audiocpp", "profile_id": "balanced"}))
        if not supervisor.NATIVE_INCREMENTAL_PCM_ENABLED:
            self.assertIn("first_block_frames", resolved["inactiveFields"])
            self.assertNotIn("first_block_frames", resolved["effectiveFields"])

    def test_unknown_override_and_persisted_profile_model_mismatch_are_rejected(self) -> None:
        with self.assertRaises(HTTPException) as unknown:
            asyncio.run(supervisor.resolve_tuning_profile({"provider": "qwen3tts-audiocpp", "profile_id": "balanced", "overrides": {"not_a_knob": 1}}))
        self.assertEqual(unknown.exception.status_code, 422)
        self.document["profiles"]["balanced"]["model"] = "qwen3-tts-0.6b-base-bf16"
        with self.assertRaises(HTTPException) as mismatch:
            asyncio.run(supervisor.resolve_tuning_profile({"provider": "qwen3tts-audiocpp", "profile_id": "balanced"}))
        self.assertEqual(mismatch.exception.status_code, 409)
        self.assertEqual(supervisor.state["activeModel"], "qwen3-tts-1.7b-base-bf16")

    def test_session_override_is_attached_only_to_candidate_playground_payload(self) -> None:
        payload = {"input": "hello"}
        returned = studio.apply_session_tuning(payload, {
            "provider": "qwen3tts-audiocpp",
            "profile_id": "balanced",
            "overrides": {"temperature": 0.6},
        })
        self.assertIs(returned, payload)
        self.assertEqual(payload["tuning"]["profile_id"], "balanced")
        self.assertEqual(payload["tuning"]["overrides"], {"temperature": 0.6})
        foreign = {"input": "hello"}
        studio.apply_session_tuning(foreign, {"provider": "faster", "profile_id": "balanced", "overrides": {}})
        self.assertNotIn("tuning", foreign)

    def test_reference_temp_files_and_editor_hydration_have_explicit_contracts(self) -> None:
        source = SUPERVISOR.read_text(encoding="utf-8")
        buffered = source.split("async def _voice_clone_response", 1)[1].split("def _native_clone_pcm_response", 1)[0]
        native = source.split("async def _native_clone_pcm_response", 1)[1].split('@app.post("/v1/voice-studio', 1)[0]
        self.assertIn("finally:\n        reference.unlink(missing_ok=True)", buffered)
        self.assertIn("await close_upstream()", native)
        self.assertIn("reference.unlink(missing_ok=True)", native)
        ui_source = UI.read_text(encoding="utf-8")
        self.assertIn("def tuning_profile_form", ui_source)
        self.assertIn("tuning_profile_dropdown.change(fn=tuning_profile_form", ui_source)
        self.assertIn("demo.load(fn=tuning_profile_form", ui_source)


if __name__ == "__main__":
    unittest.main()
