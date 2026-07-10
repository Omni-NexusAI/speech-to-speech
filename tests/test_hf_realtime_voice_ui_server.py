import importlib.util
import json
import sys
from pathlib import Path


def _load_ui_server_module():
    ui_dir = Path(__file__).resolve().parents[1] / "web" / "hf-realtime-voice"
    sys.path.insert(0, str(ui_dir))
    try:
        spec = importlib.util.spec_from_file_location("hf_realtime_voice_server", ui_dir / "server.py")
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module
    finally:
        try:
            sys.path.remove(str(ui_dir))
        except ValueError:
            pass


def test_load_base_clone_profiles_filters_to_base_profiles(tmp_path):
    server = _load_ui_server_module()
    profiles = tmp_path / "profiles"
    base = profiles / "16d9bb336799"
    custom = profiles / "not-base"
    base.mkdir(parents=True)
    custom.mkdir()
    (base / "meta.json").write_text(
        json.dumps(
            {
                "profile_id": "16d9bb336799",
                "name": "J.A.R.V.I.S",
                "task_type": "Base",
                "created_at": "2026-05-28T05:29:38Z",
                "ref_text": "Systems are now fully operational.",
            }
        ),
        encoding="utf-8",
    )
    (custom / "meta.json").write_text(
        json.dumps({"profile_id": "custom", "name": "Vivian", "task_type": "CustomVoice"}),
        encoding="utf-8",
    )

    voices = server._load_base_clone_profiles(tmp_path)

    assert voices == [
        {
            "id": "16d9bb336799",
            "voice": "clone:16d9bb336799",
            "name": "J.A.R.V.I.S",
            "task_type": "Base",
            "created_at": "2026-05-28T05:29:38Z",
            "ref_text": "Systems are now fully operational.",
        }
    ]


def test_load_base_clone_profiles_handles_missing_library(tmp_path):
    server = _load_ui_server_module()

    assert server._load_base_clone_profiles(tmp_path / "missing") == []


def test_local_ui_identity_keeps_upstream_credit_and_shows_local_provider_slots():
    ui_dir = Path(__file__).resolve().parents[1] / "web" / "hf-realtime-voice"
    html = (ui_dir / "index.html").read_text(encoding="utf-8")
    main_js = (ui_dir / "main.js").read_text(encoding="utf-8")

    assert "Built by" in html
    assert "Modified by" in html
    assert "Omni-NexusAI" in html
    assert "https://github.com/Omni-NexusAI/speech-to-speech" in html
    assert 'id="local-provider-gemma"' in html
    assert 'id="local-provider-tts"' in html
    assert 'set("local-provider-gemma"' in main_js
    assert 'set("local-provider-tts"' in main_js
