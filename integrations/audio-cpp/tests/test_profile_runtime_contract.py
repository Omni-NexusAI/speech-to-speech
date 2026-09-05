"""Focused contracts for canonical live clones and surface-scoped tuning state."""
from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

from fastapi import HTTPException

INTEGRATION = Path(__file__).parents[1]
sys.path.insert(0, str(INTEGRATION))

gradio_stub = types.ModuleType("gradio")
gradio_stub.Blocks = object
gradio_stub.themes = types.SimpleNamespace(Base=object)
sys.modules.setdefault("gradio", gradio_stub)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


supervisor = _load("candidate_supervisor_profile_runtime_test", INTEGRATION / "supervisor.py")
studio = _load("candidate_gradio_profile_runtime_test", INTEGRATION / "gradio_voice_studio.py")


class ProfileRuntimeContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="audio-cpp-profile-runtime-")
        self.library = Path(self.temporary.name)
        (self.library / "profiles").mkdir(parents=True)
        self.original_library = supervisor.VOICE_LIBRARY_DIR
        self.original_tuning_path = supervisor.TUNING_PROFILES_PATH
        supervisor.VOICE_LIBRARY_DIR = self.library
        supervisor.TUNING_PROFILES_PATH = self.library / "tts_profiles.json"

    def tearDown(self) -> None:
        supervisor.VOICE_LIBRARY_DIR = self.original_library
        supervisor.TUNING_PROFILES_PATH = self.original_tuning_path
        self.temporary.cleanup()

    def _legacy_profile(self, profile_id: str = "legacy-voice") -> Path:
        root = self.library / "profiles" / profile_id
        root.mkdir(parents=True)
        reference = b"RIFFlegacy-reference-audio"
        (root / "ref_audio.wav").write_bytes(reference)
        (root / "meta.json").write_text(json.dumps({
            "profile_id": "stale-metadata-id",
            "name": "Legacy voice",
            "task_type": "Base",
            "language": "Auto",
            "ref_text": "A real reference transcript.",
            "ref_audio_filename": "ref_audio.wav",
        }), encoding="utf-8")
        return root

    def test_supervisor_and_gradio_share_live_normalized_inventory(self) -> None:
        root = self._legacy_profile()
        missing = self.library / "profiles" / "missing-audio"
        missing.mkdir()
        (missing / "meta.json").write_text('{"task_type":"Base"}', encoding="utf-8")
        hidden = self.library / "profiles" / ".cache"
        hidden.mkdir()
        (hidden / "ref_audio.wav").write_bytes(b"hidden")
        (hidden / "meta.json").write_text('{"task_type":"Base"}', encoding="utf-8")

        api_ids = [item["id"] for item in supervisor._voice_profiles()]
        studio_ids = [item.profile_id for item in studio.list_profiles(self.library)]
        self.assertEqual(api_ids, ["legacy-voice"])
        self.assertEqual(studio_ids, api_ids)
        payload = supervisor._candidate_profile_response()["voices"][0]
        self.assertEqual(payload["id"], "legacy-voice")
        self.assertEqual(payload["voice"], "clone:legacy-voice")
        self.assertEqual(payload["provider"], "qwen3tts-audiocpp")

        normalized = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        for field in ("created_at", "origin", "voice", "instructions", "provider"):
            self.assertIn(field, normalized)
        self.assertEqual(normalized["profile_id"], "legacy-voice")
        self.assertEqual((root / "ref_audio.wav").read_bytes(), b"RIFFlegacy-reference-audio")

        (root / "ref_audio.wav").unlink()
        self.assertEqual(supervisor._voice_profiles(), [])
        self.assertEqual(studio.list_profiles(self.library), [])

    def test_frozen_clone_snapshot_survives_same_id_profile_edit_between_phrases(self) -> None:
        original_audio = b"RIFF-original-reference"
        original = supervisor._write_candidate_profile(
            "mutable-clone",
            {
                "name": "Mutable clone",
                "ref_audio": base64.b64encode(original_audio).decode("ascii"),
                "ref_text": "Original conditioning transcript.",
            },
        )
        inventory = supervisor._voice_profiles()
        self.assertEqual(inventory[0]["content_revision"], original["content_revision"])
        self.assertEqual(inventory[0]["content_hash"], original["content_hash"])
        frozen = supervisor._candidate_profile_payload("mutable-clone", include_audio=True)
        frozen_snapshot = {
            key: frozen[key]
            for key in ("profile_id", "content_revision", "content_hash", "ref_audio", "ref_text", "reference_excerpts")
        }

        edited = supervisor._write_candidate_profile(
            "mutable-clone",
            {
                "name": "Mutable clone edited",
                "ref_audio": base64.b64encode(b"RIFF-edited-reference").decode("ascii"),
                "ref_text": "Edited conditioning transcript.",
            },
        )
        self.assertGreater(edited["content_revision"], original["content_revision"])
        self.assertNotEqual(edited["content_hash"], original["content_hash"])

        request = {"voice": "clone:mutable-clone", "clone_snapshot": frozen_snapshot}
        supervisor._apply_clone_profile(request)

        self.assertEqual(request["ref_audio"], frozen_snapshot["ref_audio"])
        self.assertEqual(request["ref_text"], "Original conditioning transcript.")
        self.assertEqual(request["_frozen_clone_content_hash"], original["content_hash"])
        self.assertEqual(request["_frozen_clone_content_revision"], original["content_revision"])
        self.assertNotIn("clone_snapshot", request)

    def test_tuning_definitions_are_shared_but_selections_are_surface_scoped(self) -> None:
        migrated = supervisor._normalize_tuning_document({
            "version": 1,
            "selected": {supervisor.TUNING_PROVIDER: "quality"},
            "profiles": {},
        })
        self.assertEqual(migrated["selections"]["voice-studio"][supervisor.TUNING_PROVIDER], "quality")
        self.assertEqual(migrated["selections"]["realtime"][supervisor.TUNING_PROVIDER], "quality")
        supervisor._write_tuning_profiles(migrated)

        asyncio.run(supervisor.select_tuning_profile({
            "provider": supervisor.TUNING_PROVIDER,
            "scope": "voice-studio",
            "profile_id": "low-latency",
        }))
        document = supervisor._read_tuning_profiles()
        self.assertEqual(document["selections"]["voice-studio"][supervisor.TUNING_PROVIDER], "low-latency")
        self.assertEqual(document["selections"]["realtime"][supervisor.TUNING_PROVIDER], "quality")
        self.assertEqual(document["selected"][supervisor.TUNING_PROVIDER], "quality")

        asyncio.run(supervisor.select_tuning_profile({
            "provider": supervisor.TUNING_PROVIDER,
            "scope": "realtime",
            "profile_id": "balanced",
        }))
        document = supervisor._read_tuning_profiles()
        self.assertEqual(document["selections"]["voice-studio"][supervisor.TUNING_PROVIDER], "low-latency")
        self.assertEqual(document["selections"]["realtime"][supervisor.TUNING_PROVIDER], "balanced")
        self.assertEqual(document["selected"][supervisor.TUNING_PROVIDER], "balanced")
        with self.assertRaises(HTTPException) as invalid:
            asyncio.run(supervisor.select_tuning_profile({"scope": "unknown", "profile_id": "balanced"}))
        self.assertEqual(invalid.exception.status_code, 422)

    def test_disabled_guard_and_buffered_ui_are_process_defaults(self) -> None:
        self.assertEqual(supervisor._default_gpu_guard_settings()["mode"], "disabled")
        original = studio.NATIVE_INCREMENTAL_PCM_ENABLED
        try:
            studio.NATIVE_INCREMENTAL_PCM_ENABLED = True
            self.assertEqual(studio.default_playback_mode(), studio.BUFFERED_PLAYBACK_MODE)
            studio.NATIVE_INCREMENTAL_PCM_ENABLED = False
            self.assertEqual(studio.default_playback_mode(), studio.BUFFERED_PLAYBACK_MODE)
        finally:
            studio.NATIVE_INCREMENTAL_PCM_ENABLED = original
        dockerfile = (INTEGRATION / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("profile_library.py /opt/profile_library.py", dockerfile)
        self.assertIn("profile_library.py /opt/voice-studio/profile_library.py", dockerfile)


if __name__ == "__main__":
    unittest.main()
