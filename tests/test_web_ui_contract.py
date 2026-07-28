from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN_JS = (ROOT / "web" / "hf-realtime-voice" / "main.js").read_text(encoding="utf-8")
INDEX_HTML = (ROOT / "web" / "hf-realtime-voice" / "index.html").read_text(encoding="utf-8")
CLIENT_JS = (ROOT / "web" / "hf-realtime-voice" / "ws" / "s2s-ws-client.js").read_text(encoding="utf-8")


def test_camera_capability_depends_on_enabled_state_not_stream_readiness():
    assert "if (toolsEnabled.camera_snapshot) defs.push(TOOL_DEFS.camera_snapshot);" in MAIN_JS
    assert "toolsEnabled.camera_snapshot && cameraStream" not in MAIN_JS
    assert "toolCamSwitch.checked = toolsEnabled.camera_snapshot;" in MAIN_JS
    permission_handler = MAIN_JS.split('status.addEventListener("change"', 1)[1].split("});", 1)[0]
    assert "pushToolsToSession();" in permission_handler


def test_adaptive_v2_is_the_migrated_default_and_ui_contract_is_current():
    assert 'echoGuardVersion: "s2s.ws.echoGuardVersion"' in MAIN_JS
    assert 'localStorage.setItem(STORAGE_KEYS.echoGuardVersion, "2")' in MAIN_JS
    assert 'const EXPECTED_UI_API_VERSION = 10;' in MAIN_JS
    assert 'src="main.js?v=10-adaptive-v2"' in INDEX_HTML
    assert '"./ws/s2s-ws-client.js?v=10-adaptive-v2"' in MAIN_JS
    assert 'new URL("mic-capture.js?v=10-adaptive-v2", base)' in CLIENT_JS
