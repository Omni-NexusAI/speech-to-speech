"""Focused browser/launcher contracts for local Realtime response ownership.

These intentionally inspect the vendored browser sources instead of opening a
model or Docker service.  The runtime owns cancellation; the browser owns the
audible acknowledgement and must reject obsolete output independently.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLIENT = (ROOT / "web" / "hf-realtime-voice" / "ws" / "s2s-ws-client.js").read_text(encoding="utf-8")
WORKLET = (ROOT / "web" / "hf-realtime-voice" / "worklets" / "audio-playback.js").read_text(encoding="utf-8")
MAIN = (ROOT / "web" / "hf-realtime-voice" / "main.js").read_text(encoding="utf-8")
SERVER = (ROOT / "web" / "hf-realtime-voice" / "server.py").read_text(encoding="utf-8")
LAUNCHER = (ROOT / "scripts" / "local_realtime.ps1").read_text(encoding="utf-8")
TTS_HANDLER = (ROOT / "src" / "speech_to_speech" / "TTS" / "qwen3_tts_handler.py").read_text(encoding="utf-8")


def test_pipeline_response_binds_epoch_before_epochless_pcm_and_ack_is_idempotent():
    assert 'case "pipeline.response"' in CLIENT
    assert "_bindPlaybackResponseEpoch(responseId, responseEpoch)" in CLIENT
    assert "this._playbackByResponse.get(rid)?.responseEpoch" in CLIENT
    assert "_responseEpoch(event) ?? responsePlayback?.responseEpoch ?? null" in CLIENT
    assert 'type: "pipeline.playback.started"' in CLIENT
    assert "response_epoch: snapshot.responseEpoch" in CLIENT
    assert "if (!snapshot.playbackAcked)" in CLIENT


def test_stale_epoch_rejection_and_worklet_epoch_generation_transport_are_explicit():
    assert "_isStaleResponseEpoch(epoch)" in CLIENT
    assert "if (this._isStaleResponseEpoch(responseEpoch)) return;" in CLIENT
    assert "responseEpoch: snapshot.responseEpoch" in CLIENT
    assert "responseEpoch: this._activeResponseEpoch" in WORKLET
    assert "generation_mismatch" in WORKLET
    assert "this._queue.length = 0;" in WORKLET


def test_playback_keeps_pcm_clock_and_adaptive_reservoir_is_bounded():
    assert "MAX_PLAYBACK_PRIME_MS = 2_000" in CLIENT
    assert "PLAYBACK_CONTINUITY_ADAPTIVE" in CLIENT
    assert "PLAYBACK_CONTINUITY_FAST_START" in CLIENT
    assert "Math.min(MAX_PLAYBACK_PRIME_MS" in CLIENT
    assert "provider_unsustainable" in CLIENT
    assert "Provider cannot sustain realtime" in CLIENT
    # The worklet only interpolates the fixed input PCM into the AudioContext;
    # it has no playback-rate/time-stretch control and preserves FIFO order.
    assert "this._stepRatio = this._inputRate / sampleRate" in WORKLET
    assert "playbackRate" not in WORKLET
    assert "this._queue.shift();" in WORKLET


def test_audio_cpp_model_pcm_and_hfrt_transport_rates_are_labeled_truthfully():
    # Faster/Groxaxo retain their 16 kHz browser transport. The isolated
    # candidate carries its model-native PCM16/24 kHz clock end-to-end after
    # server acknowledgement; the worklet only interpolates to AudioContext.
    assert "const DEFAULT_OUTPUT_SAMPLE_RATE = 16000;" in CLIENT
    assert "config.audio_output_sample_rate" in CLIENT
    assert "inputRate: this._playbackSampleRate" in CLIENT
    assert "PIPELINE_SR = 16000" in TTS_HANDLER
    assert "preserve_provider_rate" in TTS_HANDLER
    assert "audio.cpp candidate acknowledges its model-native 24 kHz clock" in CLIENT
    assert "this._stepRatio = this._inputRate / sampleRate" in WORKLET
    page = (ROOT / "web" / "hf-realtime-voice" / "index.html").read_text(encoding="utf-8")
    assert "audio.cpp: PCM16 / 24 kHz" in page
    assert "Faster/Groxaxo: PCM16 / 16 kHz" in page
    assert "server-acknowledged transport" in MAIN


def test_startup_identity_is_uncached_latest_wins_and_unavailable_is_explicit():
    assert "let localIdentityRequest = 0;" in MAIN
    assert "const request = ++localIdentityRequest;" in MAIN
    assert "cache: \"no-store\"" in MAIN
    assert "request !== localIdentityRequest" in MAIN
    assert "mode: \"unavailable\"" in MAIN
    assert "refreshLocalIdentity({ retry: true })" in MAIN
    assert "const MAX_LOCAL_IDENTITY_RETRIES = 2;" in MAIN
    assert "localIdentityRetryAttempts < MAX_LOCAL_IDENTITY_RETRIES" in MAIN
    assert "refreshLocalIdentity({ retry: true, resetRetry: true })" in MAIN
    assert "document.body.classList.remove(\"booting\");" in MAIN
    assert "requestAnimationFrame(() => {\n  document.body.classList.remove(\"booting\");" not in MAIN
    assert '"playbackContinuity"' in SERVER
    assert "selected_provider = _canonical_tts_provider" in SERVER
    assert '"container"' not in SERVER.split("async def local_pipeline", 1)[1].split("async def _faster_model_inventory", 1)[0]


def test_healthy_local_pipeline_identity_renders_before_slow_backend_inventory():
    """A stalled clone/backend inventory must not preserve static Checking labels."""
    pipeline_await = MAIN.index("const pipelineResult = await pipelinePromise.then(")
    first_render = MAIN.index("renderLocalPipeline();", pipeline_await)
    backend_await = MAIN.index("const backendResult = await backendPromise.then(")
    assert pipeline_await < first_render < backend_await
    assert "until the richer backend inventory arrives" in MAIN


def test_launcher_reports_verified_url_without_opening_unless_requested():
    assert '$FrontendUrl = "http://127.0.0.1:7862"' in LAUNCHER
    assert 'Write-Host "HF Realtime Voice UI: $FrontendUrl"' in LAUNCHER
    assert 'if ($Open) { Open-FrontendIfReady }' in LAUNCHER
    helper = LAUNCHER.split("function Open-FrontendIfReady", 1)[1].split("function Get-Process-Info", 1)[0]
    assert "Endpoint-Ok $Specs.frontend.Health" in helper
    assert "Start-Process $FrontendUrl" in helper
