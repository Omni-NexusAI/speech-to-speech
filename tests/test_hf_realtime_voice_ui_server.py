import asyncio
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


def test_tts_backend_status_reports_faster_and_voice_studio_model(monkeypatch):
    server = _load_ui_server_module()

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url):
            if url.endswith("/health"):
                return Response(
                    {
                        "model_loaded": True,
                        "capabilities": {"native_pcm_streaming": True, "clone_only": True},
                    }
                )
            return Response(
                {"state": "loaded", "current": "0.6B-Base", "loaded_models": ["0.6B-Base"]}
            )

    monkeypatch.setattr(server.httpx, "AsyncClient", lambda **kwargs: Client())

    faster = asyncio.run(server._probe_tts_backend("faster", server.TTS_BACKENDS["faster"]))
    groxaxo = asyncio.run(server._probe_tts_backend("groxaxo", server.TTS_BACKENDS["groxaxo"]))

    assert faster["ready"] is True
    assert faster["currentModel"] == "1.7B-Base"
    assert groxaxo["ready"] is True
    assert groxaxo["currentModel"] == "0.6B-Base"


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


def test_live_transcript_toggle_only_controls_the_floating_user_bubble():
    ui_dir = Path(__file__).resolve().parents[1] / "web" / "hf-realtime-voice"
    main_js = (ui_dir / "main.js").read_text(encoding="utf-8")
    chat_js = (ui_dir / "ui" / "chat.js").read_text(encoding="utf-8")

    assert "showUserBubble: settings.liveTranscript" in main_js
    assert "if (options.showUserBubble && (d.partial" in chat_js
    assert "this._pendingUserHist || this._appendHistMsg" in chat_js
    assert "Speech could not be transcribed." not in main_js
    assert "discardPendingUserTurn" not in chat_js


def test_tool_output_is_acknowledged_before_one_post_tool_response():
    ui_dir = Path(__file__).resolve().parents[1] / "web" / "hf-realtime-voice"
    main_js = (ui_dir / "main.js").read_text(encoding="utf-8")
    client_js = (ui_dir / "ws" / "s2s-ws-client.js").read_text(encoding="utf-8")

    assert main_js.index("sessionClient.sendToolOutput") < main_js.index("sessionClient.requestToolResponse")
    assert main_js.index("sessionClient.requestToolResponse") < main_js.index("await outputAck")
    assert "requestToolResponse(opts = {})" in client_js
    assert 'this._send({ type: "response.create" })' in client_js


def test_local_ui_teardown_isolates_closed_client_events():
    ui_dir = Path(__file__).resolve().parents[1] / "web" / "hf-realtime-voice"
    main_js = (ui_dir / "main.js").read_text(encoding="utf-8")
    client_js = (ui_dir / "ws" / "s2s-ws-client.js").read_text(encoding="utf-8")

    assert "if (client !== c) return;" in main_js
    assert "const closingClient = client;\n  client = null;" in main_js
    assert "runTool(c, name, args, callId)" in main_js
    assert 'if (this._closed && status !== "closed") return;' in client_js
    assert 'this._captureNode?.port.postMessage({ kind: "enable", value: false });' in client_js
    assert 'this._playbackNode?.port.postMessage({ kind: "clear" });' in client_js


def test_diagnostics_use_transcription_and_dynamic_context_tokens():
    ui_dir = Path(__file__).resolve().parents[1] / "web" / "hf-realtime-voice"
    main_js = (ui_dir / "main.js").read_text(encoding="utf-8")

    assert '"transcription"' in main_js
    assert '"gemma_preview"' not in main_js
    assert "history_tokens" in main_js
    assert "contextWindow" in main_js


def test_remote_model_connection_test_redacts_key(monkeypatch):
    server = _load_ui_server_module()
    calls = []

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class Client:
        def __init__(self, **kwargs):
            calls.append(("client", kwargs))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url):
            calls.append(("get", url))
            if url.endswith("/models"):
                return Response({"data": [{"id": "remote-audio-model"}]})
            return Response({"default_generation_settings": {"n_ctx": 32768}})

    monkeypatch.setattr(server.httpx, "AsyncClient", Client)
    result = asyncio.run(
        server.test_model_endpoint(
            server.ModelTestRequest(
                provider="remote",
                base_url="http://10.0.0.60:8080",
                model="configured-model",
                api_key="secret",
            )
        )
    )

    assert result["model"] == "remote-audio-model"
    assert result["context_window"] == 32768
    assert result["api_key_set"] is True
    assert "api_key" not in result
    assert calls[0][1]["headers"] == {"Authorization": "Bearer secret"}
    assert all("127.0.0.1:8818" not in str(call) for call in calls)


def test_settings_show_model_provider_before_stacked_tts_and_voice():
    ui_dir = Path(__file__).resolve().parents[1] / "web" / "hf-realtime-voice"
    html = (ui_dir / "index.html").read_text(encoding="utf-8")
    client_js = (ui_dir / "ws" / "s2s-ws-client.js").read_text(encoding="utf-8")

    assert html.index('id="model-provider"') < html.index('id="tts-backend"') < html.index('id="voice"')
    assert 'id="model-api-key" type="password"' in html
    assert client_js.index("this.updateLocalPipeline") < client_js.index("this._sendSessionUpdate")
