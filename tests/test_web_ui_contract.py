from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN_JS = (ROOT / "web" / "hf-realtime-voice" / "main.js").read_text(encoding="utf-8")
INDEX_HTML = (ROOT / "web" / "hf-realtime-voice" / "index.html").read_text(encoding="utf-8")
CLIENT_JS = (ROOT / "web" / "hf-realtime-voice" / "ws" / "s2s-ws-client.js").read_text(encoding="utf-8")
MIC_CAPTURE_JS = (ROOT / "web" / "hf-realtime-voice" / "worklets" / "mic-capture.js").read_text(encoding="utf-8")
PLAYBACK_JS = (ROOT / "web" / "hf-realtime-voice" / "worklets" / "audio-playback.js").read_text(encoding="utf-8")


def test_camera_capability_depends_on_enabled_state_not_stream_readiness():
    assert "if (toolsEnabled.camera_snapshot) defs.push(TOOL_DEFS.camera_snapshot);" in MAIN_JS
    assert "toolsEnabled.camera_snapshot && cameraStream" not in MAIN_JS
    assert "toolCamSwitch.checked = toolsEnabled.camera_snapshot;" in MAIN_JS
    permission_handler = MAIN_JS.split('status.addEventListener("change"', 1)[1].split("});", 1)[0]
    assert "pushToolsToSession();" in permission_handler


def test_native_v3_is_the_migrated_default_and_echo_ui_is_truthful():
    assert 'echoGuardVersion: "s2s.ws.echoGuardVersion"' in MAIN_JS
    assert 'localStorage.setItem(STORAGE_KEYS.echoGuardVersion, "3")' in MAIN_JS
    assert 'value="native" selected>Native browser AEC (default)' in INDEX_HTML
    assert 'Adaptive (AEC3 reference cancellation)' in INDEX_HTML
    assert 'value="off"' not in INDEX_HTML
    assert 'requested === "strict" ? "strict" : "native"' in MIC_CAPTURE_JS
    assert 'const EXPECTED_UI_API_VERSION = 21;' in MAIN_JS
    assert 'src="main.js?v=30-history-outcomes"' in INDEX_HTML
    assert '"./ws/s2s-ws-client.js?v=17-synthesis-outcomes"' in MAIN_JS
    assert "loadAec3Worklet(ctx)" in CLIENT_JS
    assert 'new URL("mic-capture.js?v=12-aec3-fallback", base)' in CLIENT_JS
    assert 'new URL("audio-playback.js?v=16-epoch-continuity", base)' in CLIENT_JS
    assert "new AudioWorkletNode(ctx, aec3.processorName" in CLIENT_JS


def test_audio_cpp_context_help_matches_the_pinned_25_frame_engine_contract():
    """Diagnostics must not steer users toward the obsolete 72-frame baseline."""

    assert "model-required 25 frames" in INDEX_HTML
    assert "model-required 72 frames" not in INDEX_HTML


def test_aec3_calibration_is_device_pair_scoped_and_persisted():
    server = (ROOT / "web" / "hf-realtime-voice" / "server.py").read_text(encoding="utf-8")
    for control_id in (
        "diagnostics-echo-device-pair",
        "diagnostics-echo-output-latency",
        "diagnostics-echo-save",
        "diagnostics-echo-use-measured",
    ):
        assert f'id="{control_id}"' in INDEX_HTML
    for key in (
        "delayMs",
        "suppressionStrength",
        "leakageThreshold",
        "doubleTalkSensitivity",
    ):
        assert f'data-echo-calibration-key="{key}"' in INDEX_HTML
    assert "echoCalibrations: settings.echoCalibrations" in MAIN_JS
    assert "this._echoDevicePair" in CLIENT_JS
    assert "microphoneId" in CLIENT_JS and "outputId" in CLIENT_JS
    assert '"echoCalibrations",' in server


def test_voice_studio_management_and_candidate_validation_are_explicit():
    assert 'id="profile-create"' in INDEX_HTML
    assert 'id="profile-import"' in INDEX_HTML
    assert 'id="faster-model-load"' in INDEX_HTML
    assert 'id="validate-audio-cpp"' in INDEX_HTML
    assert 'id="audio-cpp-model"' not in INDEX_HTML
    assert 'api/tts/backends/${encodeURIComponent(settings.ttsBackend)}/profiles/${encodeURIComponent(profile.id)}/select' in MAIN_JS
    assert 'api/tts/backends/${encodeURIComponent(backend)}/voices' in MAIN_JS
    assert 'api/tts/backends/${encodeURIComponent(settings.ttsBackend)}/validate' in MAIN_JS
    assert "let voiceInventoryRequest = 0;" in MAIN_JS
    assert "Loading live Base clone profiles for ${backend}" in MAIN_JS
    assert "request !== voiceInventoryRequest || backend !== settings.ttsBackend" in MAIN_JS
    assert 'Manage selected backend clone profiles' in INDEX_HTML
    assert 'Selected backend model status' in INDEX_HTML
    assert 'Validate selected TTS backend' in INDEX_HTML
    assert '"audio-cpp"' in MAIN_JS
    assert '"qwen3tts-audiocpp"' in MAIN_JS


def test_streaming_default_and_optional_non_streaming_dispatch_are_preserved():
    assert 'fullBufferTts: localStorage.getItem(STORAGE_KEYS.fullBufferTts) === "1"' in MAIN_JS
    assert 'full_buffer_tts: settings.fullBufferTts' in MAIN_JS
    assert "Streaming is the default." in INDEX_HTML
    assert "Non-streaming assistant dispatch" in INDEX_HTML


def test_audio_cpp_settings_proxy_and_candidate_persistence_are_explicit():
    SERVER = (ROOT / "web" / "hf-realtime-voice" / "server.py").read_text(encoding="utf-8")
    STUDIO = (ROOT / "integrations" / "audio-cpp" / "gradio_voice_studio.py").read_text(encoding="utf-8")
    assert '@app.get("/api/audio-cpp/settings")' in SERVER
    assert '@app.put("/api/audio-cpp/settings")' in SERVER
    assert "voice-studio/settings" in SERVER
    assert "Saved to candidate storage. API key stays browser-only." in STUDIO


def test_hf_ui_preserves_nonsecret_settings_across_browser_environment_reset():
    SERVER = (ROOT / "web" / "hf-realtime-voice" / "server.py").read_text(encoding="utf-8")
    assert '@app.get("/api/ui-settings")' in SERVER
    assert '@app.put("/api/ui-settings")' in SERVER
    assert "PUBLIC_UI_SETTING_KEYS" in SERVER
    assert "modelApiKey" not in SERVER.split("PUBLIC_UI_SETTING_KEYS", 1)[1].split("DEFAULT_QWEN3", 1)[0]
    assert "restorePersistentSettings" in MAIN_JS
    assert 'fetch("/api/ui-settings"' in MAIN_JS


def test_model_api_key_is_page_session_only_and_legacy_storage_is_tombstoned_without_reading():
    storage_block = MAIN_JS.split("const STORAGE_KEYS =", 1)[1].split("};", 1)[0]
    assert "modelApiKey" not in storage_block
    assert 'const LEGACY_MODEL_API_KEY_STORAGE_KEY = "s2s.ws.modelApiKey";' in MAIN_JS
    load_settings = MAIN_JS.split("function loadSettings()", 1)[1].split("function loadGateThreshold()", 1)[0]
    assert "localStorage.removeItem(LEGACY_MODEL_API_KEY_STORAGE_KEY);" in load_settings
    assert "localStorage.getItem(LEGACY_MODEL_API_KEY_STORAGE_KEY)" not in load_settings
    assert 'modelApiKey: ""' in load_settings
    save_settings = MAIN_JS.split("function saveSettings", 1)[1].split("function loadEchoCalibrations", 1)[0]
    assert "modelApiKey" not in save_settings
    restore = MAIN_JS.split("async function restorePersistentSettings", 1)[1].split("function loadTools", 1)[0]
    assert "modelApiKey: settings.modelApiKey" in restore


def test_managed_settings_restore_precedes_initial_backend_inventory_fetch():
    initialize = MAIN_JS.split("async function initializeApp()", 1)[1].split('setState("idle")', 1)[0]
    assert initialize.index("await restorePersistentSettings()") < initialize.index("await fetchConfig()")
    assert "await refreshCandidateTuningProfiles()" in initialize
    assert "await fetchVoiceProfiles()" in initialize
    open_settings = MAIN_JS.split("function openSettings()", 1)[1].split("function renderVoiceOptions()", 1)[0]
    assert "fetchVoiceProfiles()" in open_settings
    assert "refreshTtsBackends()" in open_settings
    assert "renderSelectedTtsBackendStatus()" in open_settings
    assert "payload.backend && payload.backend !== backend" in MAIN_JS


def test_voice_change_waits_for_persistence_before_refreshing_clone_validation():
    selection = MAIN_JS.split("async function selectVoiceProfile()", 1)[1].split("/** dB position", 1)[0]
    assert "const saved = await saveSettings(settings);" in selection
    assert selection.index("await saveSettings(settings)") < selection.index("await refreshTtsBackends()")
    assert "validation?.speech && validation.voice === profile?.voice" in selection


def test_backend_change_persists_selected_inventory_before_validation_refresh():
    change = MAIN_JS.split('inputTtsBackend.addEventListener("change"', 1)[1].split("});", 1)[0]
    assert change.index("clearVoiceProfileOptions(") < change.index("await Promise.all(")
    assert change.index("fetchVoiceProfiles()") < change.index("await saveSettings(settings)")
    assert change.index("await saveSettings(settings)") < change.index("await refreshTtsBackends()")
    assert "renderSelectedTtsBackendStatus()" in MAIN_JS


def test_audio_cpp_profile_refresh_is_used_for_backend_changes_and_session_payloads():
    change = MAIN_JS.split('inputTtsBackend.addEventListener("change"', 1)[1].split("});", 1)[0]
    assert "refreshCandidateTuningProfiles().catch" in change
    assert "async function resolveCandidateTuning" in MAIN_JS
    assert "function resolvedCandidatePhraseQueue" in MAIN_JS
    assert "payload.resolved = phraseQueue" in MAIN_JS
    assert "let realtimeTuningByBackend = {};" in MAIN_JS
    assert "ttsProfileByBackend" in MAIN_JS
    assert "ttsDeliveryMode" in MAIN_JS
    form_settings = MAIN_JS.split("function readSettingsFromForm()", 1)[1].split("function syncModelProviderUi()", 1)[0]
    assert "activeTtsTuning(backend)" in form_settings
    assert 'playbackContinuity: settings.playbackContinuity === "fast-start"' in form_settings
    assert 'ttsDeliveryMode: settings.ttsDeliveryMode === "native_incremental_pcm"' in form_settings
    pipeline = MAIN_JS.split("pipelineConfig:", 1)[1].split("...(audioContext", 1)[0]
    assert "...(normalizeTtsProvider(settings.ttsBackend) === AUDIO_CPP_PROVIDER" in pipeline
    assert "{ tts_tuning: initialTtsTuning }" in pipeline
    assert "client.updateLocalPipeline(" in MAIN_JS
    assert "tts_backend: settings.ttsBackend" in MAIN_JS
    assert "tts_tuning: tuning" in MAIN_JS
    active = MAIN_JS.split("function activeTtsTuning", 1)[1].split("function updateRealtimeAudioSummary", 1)[0]
    assert "if (normalizeTtsProvider(backend) !== AUDIO_CPP_PROVIDER) return null" in active
    # The WebSocket now sends a complete immutable response-input snapshot
    # directly; REST-only `scope` and mutable `resolved` never cross it.
    assert "profile_revision: revision" in active
    assert "effective," in active
    assert "overrides: { ...state.overrides }" in active
    assert 'delivery_mode: settings.ttsDeliveryMode === "native_incremental_pcm"' in active
    assert "scope:" not in active


def test_audio_cpp_start_requires_and_freezes_a_complete_tuning_snapshot():
    required = MAIN_JS.split("function requiredCandidateTtsTuning", 1)[1].split(
        "/** Browser-only playback snapshot", 1
    )[0]
    assert "const tuning = activeTtsTuning(backend)" in required
    assert "complete resolved tuning profile" in required
    assert "throw new Error" in required

    preflight = MAIN_JS.split("async function assertTtsBackendReady()", 1)[1].split(
        "async function assertModelEndpointReady", 1
    )[0]
    assert "return requiredCandidateTtsTuning(settings.ttsBackend);" in preflight

    start = MAIN_JS.split("async function doStart", 1)[1].split("c.addEventListener(\"queue\"", 1)[0]
    assert "let initialTtsTuning = null;" in start
    assert "initialTtsTuning = await awaitStartPreflight" in start
    assert "{ tts_tuning: initialTtsTuning }" in start
    assert "activeTtsTuning(settings.ttsBackend)" not in start


def test_initial_pipeline_configuration_is_acknowledged_or_fails_closed():
    assert "PIPELINE_CONFIG_ACK_TIMEOUT_MS = 15_000" in CLIENT_JS
    assert "Promise.all([audioReady, wsReady, configReady])" in CLIENT_JS
    assert 'case "pipeline.config.updated"' in CLIENT_JS
    assert "this._resolveInitialConfig()" in CLIENT_JS
    assert "if (!this._sessionConfigured) return" in CLIENT_JS
    pre_config_error = CLIENT_JS.split('case "error":', 1)[1].split(
        'if (err?.type === "conversation_already_has_active_response"', 1
    )[0]
    assert "if (!this._sessionConfigured)" in pre_config_error
    assert "this._rejectInitialConfig(failure)" in pre_config_error
    assert "await this.close()" in pre_config_error


def test_fatal_startup_error_remains_visible_after_pipeline_cleanup():
    fatal_handler = MAIN_JS.split("function onFatalError(err) {", 1)[1].split(
        "async function initializeApp()", 1
    )[0]
    assert "void teardown().then(showError, showError)" in fatal_handler
    assert "showError();" in fatal_handler


def test_candidate_rest_scope_never_enters_websocket_tuning():
    rest = MAIN_JS.split("function candidateRestTuningPayload", 1)[1].split(
        "function activeTtsTuning", 1
    )[0]
    session = MAIN_JS.split("function activeTtsTuning", 1)[1].split(
        "function updateRealtimeAudioSummary", 1
    )[0]
    assert 'scope: "realtime"' in rest
    assert "profile_revision: revision" in session
    assert "scope:" not in session
    preflight = MAIN_JS.split("async function assertTtsBackendReady", 1)[1].split(
        "async function assertModelEndpointReady", 1
    )[0]
    assert "await refreshCandidateTuningProfiles();" in preflight
    assert ".catch(" not in preflight


def test_voice_inventory_is_live_backend_scoped_and_never_leaves_stale_ids_visible():
    fetcher = MAIN_JS.split("async function fetchVoiceProfiles()", 1)[1].split(
        "async function refreshFasterModelInventory", 1
    )[0]
    assert 'fetch(`api/tts/backends/${encodeURIComponent(backend)}/voices`, { cache: "no-store" })' in fetcher
    assert "clearVoiceProfileOptions(backend)" in fetcher
    assert 'placeholder.value = ""' in MAIN_JS
    assert "request !== voiceInventoryRequest || backend !== settings.ttsBackend" in fetcher


def test_realtime_audio_follows_live_grids_and_is_collapsed_by_default():
    graph = INDEX_HTML.index('id="diagnostics-graph"')
    waterfall = INDEX_HTML.index('id="diagnostics-waterfall"')
    audio = INDEX_HTML.index('id="diagnostics-audio-details"')
    assert graph < waterfall < audio
    opening_tag = INDEX_HTML[audio:INDEX_HTML.index(">", audio)]
    assert " open" not in opening_tag
    assert 'id="diagnostics-audio-summary"' in INDEX_HTML
    assert INDEX_HTML.index('id="diagnostics-tuning-advanced"', audio) > audio


def test_realtime_audio_diagnostics_exposes_shared_safe_tuning_schema():
    for control_id in (
        "diagnostics-tts-profile",
        "diagnostics-tts-profile-name",
        "diagnostics-tts-profile-save-as",
        "tuning-model",
        "tuning-max-reference-seconds",
        "tuning-first-block-frames",
        "tuning-steady-block-frames",
        "tuning-left-context-frames",
        "tuning-text-lookahead",
        "tuning-phrase-flush-ms",
        "tuning-temperature",
        "tuning-top-k",
        "tuning-top-p",
        "tuning-repetition-penalty",
        "tuning-seed",
        "tuning-context-unlock",
    ):
        assert f'id="{control_id}"' in INDEX_HTML
    assert "audio.cpp: PCM16 / 24 kHz · Faster/Groxaxo: PCM16 / 16 kHz" in INDEX_HTML
    assert 'value="Full ICL"' in INDEX_HTML
    assert "Matched reference limit" in INDEX_HTML
    assert "These sampler values control speech-token variation and prosody. They do not change the chat LLM." in INDEX_HTML
    assert "Latency and phrase dispatch" in INDEX_HTML
    assert "Conditioning and TTS sampling" in INDEX_HTML
    assert "Expert safety" in INDEX_HTML
    assert 'min="0.05" max="2"' in INDEX_HTML
    assert 'fetch("/api/audio-cpp/tuning/resolve"' in MAIN_JS
    assert 'fetch("/api/audio-cpp/tuning/selection"' in MAIN_JS
    assert 'fetch("/api/audio-cpp/tuning/profiles"' in MAIN_JS
    assert 'scope: "realtime"' in MAIN_JS
    assert 'clone_from: diagnosticsTtsProfile.value' in MAIN_JS
    assert 'select: true' in MAIN_JS
    assert "applyCandidateTuningToLiveSession" in MAIN_JS
    assert "client.updateLocalPipeline" in MAIN_JS


def test_realtime_tuning_distinguishes_named_temporary_and_effective_values():
    for control_id in (
        "diagnostics-tuning-named-values",
        "diagnostics-tuning-override-values",
        "diagnostics-tuning-effective-values",
    ):
        assert f'id="{control_id}"' in INDEX_HTML
    assert "function effectiveCandidateTuning" in MAIN_JS
    assert "Temporary overrides are active for this page session only." in MAIN_JS
    assert "Saved as a new Realtime profile; Voice Studio selection remains unchanged." in MAIN_JS
    assert "syncDecoderContextUnlock" in MAIN_JS
    assert 'candidateInactiveTuningFields.has("left_context_frames")' in MAIN_JS


def test_native_pcm_metrics_cover_first_phrase_pcm_playback_rtf_and_end_to_end():
    assert 'm.status === "first_stable_phrase"' in MAIN_JS
    assert 'm.stage === "tts" && m.status === "first_audio"' in MAIN_JS
    assert 'm.stage === "playback" && m.status === "first_audio"' in MAIN_JS
    assert "done.rtf" in MAIN_JS
    assert "detail?.end_to_end_ms" in MAIN_JS
    assert "gpu.freeMiB" in MAIN_JS
    assert "gpu.utilizationPercent" in MAIN_JS
    assert 'this._diagnostic("started", {' in PLAYBACK_JS
    assert "first_playback_ms" in CLIENT_JS
    assert "end_to_end_ms" in CLIENT_JS


def test_audio_cpp_adaptive_cold_prime_uses_only_the_first_codec_block():
    playback_config = MAIN_JS.split("function activePlaybackConfig", 1)[1].split(
        "function updateRealtimeAudioSummary", 1
    )[0]
    assert "firstBlockFrames * 80" in playback_config
    assert "firstBlockFrames +" not in playback_config
    snapshot = CLIENT_JS.split("_playbackSnapshot", 1)[1].split("_onPlaybackEvent", 1)[0]
    assert "reprimeMs" in snapshot
    assert "steady-state target" in snapshot


def test_realtime_audio_diagnostics_show_paired_reference_truth_without_transcript_content():
    assert "done.delivery_mode" in MAIN_JS
    assert "done.reference_source_seconds" in MAIN_JS
    assert "done.reference_requested_limit_seconds" in MAIN_JS
    assert "done.reference_used_seconds" in MAIN_JS
    assert "done.reference_limit_applied" in MAIN_JS
    assert "done.reference_pairing" in MAIN_JS
    assert "reference_transcript" not in MAIN_JS


def test_audio_cpp_proxy_is_incremental_and_candidate_labels_are_truthful():
    server = (ROOT / "web" / "hf-realtime-voice" / "server.py").read_text(encoding="utf-8")
    assert "StreamingResponse(" in server
    assert "async for chunk in upstream.aiter_raw()" in server
    assert "await request.is_disconnected()" in server
    assert 'result["nativeIncrementalPcm"]' in server
    assert 'result["bufferedFallback"] = not result["nativeIncrementalPcm"]' in server
    assert 'selectedTts?.nativeStreaming ? "native PCM" : "buffered fallback"' in MAIN_JS


def test_history_compaction_uses_only_the_confirmed_bounded_config_schema():
    server = (ROOT / "web" / "hf-realtime-voice" / "server.py").read_text(encoding="utf-8")
    for control_id in (
        "history-compaction-enabled",
        "history-compaction-trigger",
        "history-compaction-target",
        "history-compaction-recent-turns",
        "diagnostics-history-compaction-summary",
        "diagnostics-history-compaction-status",
    ):
        assert f'id="{control_id}"' in INDEX_HTML
    assert "history_compaction: normalizeHistoryCompaction(settings.historyCompaction)" in MAIN_JS
    assert "client.updateLocalPipeline({ history_compaction: settings.historyCompaction })" in MAIN_JS
    assert "config?.history_compaction" in MAIN_JS
    assert 'import { formatHistoryCompactionDiagnostics } from "./ui/history-compaction-diagnostics.js"' in MAIN_JS
    assert "formatHistoryCompactionDiagnostics({ configured, acknowledged, contextDetail })" in MAIN_JS
    assert (ROOT / "tests" / "history_compaction_diagnostics.mjs").is_file()
    assert '"historyCompaction"' in server
    assert "max(1, min(12, int(recent_turns)))" in server
