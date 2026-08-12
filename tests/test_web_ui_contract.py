from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN_JS = (ROOT / "web" / "hf-realtime-voice" / "main.js").read_text(encoding="utf-8")
INDEX_HTML = (ROOT / "web" / "hf-realtime-voice" / "index.html").read_text(encoding="utf-8")
CLIENT_JS = (ROOT / "web" / "hf-realtime-voice" / "ws" / "s2s-ws-client.js").read_text(encoding="utf-8")
WEB_SEARCH_JS = (ROOT / "web" / "hf-realtime-voice" / "tools" / "web-search.js").read_text(encoding="utf-8")
CHAT_JS = (ROOT / "web" / "hf-realtime-voice" / "ui" / "chat.js").read_text(encoding="utf-8")
MIC_CAPTURE_JS = (ROOT / "web" / "hf-realtime-voice" / "worklets" / "mic-capture.js").read_text(encoding="utf-8")
PLAYBACK_JS = (ROOT / "web" / "hf-realtime-voice" / "worklets" / "audio-playback.js").read_text(encoding="utf-8")
REALTIME_README = (ROOT / "web" / "hf-realtime-voice" / "README.md").read_text(encoding="utf-8")
REALTIME_CONTEXT = (ROOT / "web" / "hf-realtime-voice" / "CONTEXT.md").read_text(encoding="utf-8")
REALTIME_SERVER = (ROOT / "web" / "hf-realtime-voice" / "server.py").read_text(encoding="utf-8")
REALTIME_AGENTS = (ROOT / "web" / "hf-realtime-voice" / "AGENTS.md").read_text(encoding="utf-8")
QWEN_HANDLER = (ROOT / "src" / "speech_to_speech" / "TTS" / "qwen3_tts_handler.py").read_text(
    encoding="utf-8"
)
QWEN_ARGUMENTS = (
    ROOT / "src" / "speech_to_speech" / "arguments_classes" / "qwen3_tts_arguments.py"
).read_text(encoding="utf-8")
ROOT_AGENTS = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
LOCAL_REALTIME_GUIDE = (ROOT / "docs" / "local-gemma-fasterqwen3tts.md").read_text(encoding="utf-8")
LOCAL_REALTIME_EXAMPLE = (ROOT / "examples" / "local_gemma_fasterqwen3tts.json").read_text(encoding="utf-8")
HOSTING_ADR = (
    ROOT / "web" / "hf-realtime-voice" / "docs" / "adr" / "0001-docker-space-with-search-proxy.md"
).read_text(encoding="utf-8")
GEMMA_LAUNCHER = (ROOT / "scripts" / "launch_gemma_4_12b_16k.ps1").read_text(
    encoding="utf-8"
)


def test_realtime_docs_describe_current_direct_audio_and_acknowledgement_contracts():
    assert "Gemma direct audio" in REALTIME_README
    assert "pipeline.config.updated" in REALTIME_README
    assert "Base clone profiles" in REALTIME_README
    assert "VOICE_LIBRARY_DIR" in REALTIME_README
    assert "Gemma direct audio" in REALTIME_CONTEXT
    assert "Configuration acknowledgement" in REALTIME_CONTEXT
    assert "historical scope" in HOSTING_ADR

    for stale_claim in (
        "Same load balancer, same",
        "Aiden, Ryan, Dylan",
        "nvidia/parakeet-tdt-1.1b",
        "google/gemma-4-31B-it",
        "front-end, audio pipeline, and s2s handshake are untouched",
    ):
        assert stale_claim not in REALTIME_README
        assert stale_claim not in REALTIME_CONTEXT
        assert stale_claim not in HOSTING_ADR


def test_realtime_voice_default_is_live_inventory_driven_and_has_no_privileged_clone():
    retired_profile_id = "16d9" + "bb336799"
    retired_profile_name = "J.A.R." + "V.I.S"
    product_surfaces = (
        MAIN_JS,
        INDEX_HTML,
        REALTIME_SERVER,
        QWEN_HANDLER,
        QWEN_ARGUMENTS,
        ROOT_AGENTS,
        LOCAL_REALTIME_GUIDE,
        LOCAL_REALTIME_EXAMPLE,
    )
    for surface in product_surfaces:
        assert retired_profile_id not in surface
        assert retired_profile_name not in surface

    assert 'const DEFAULT_VOICE = "";' in MAIN_JS
    assert 'DEFAULT_OPENAI_API_VOICE: str | None = None' in QWEN_HANDLER
    assert 'default=None' in QWEN_ARGUMENTS.split("qwen3_tts_api_voice", 1)[1].split(")", 1)[0]
    assert "selected_profile.json" in QWEN_HANDLER
    assert "selected_profile.json" in REALTIME_SERVER
    privileged_delete_message = "configured J.A.R." + "V.I.S default profile cannot be deleted"
    assert privileged_delete_message not in REALTIME_SERVER
    assert 'profile.id === DEFAULT_VOICE.replace("clone:", "")' not in MAIN_JS
    assert "profileDeleteBtn.disabled = !profileLibraryWritable || !profile" in MAIN_JS
    assert "profileCreateBtn.disabled = !profileLibraryWritable" in MAIN_JS


def test_gemma_launcher_has_no_workstation_path_defaults():
    for variable in (
        "LLAMA_CPP_DIR",
        "GEMMA_MODEL_PATH",
        "GEMMA_DRAFT_MODEL_PATH",
        "GEMMA_MMPROJ_PATH",
    ):
        assert f"$env:{variable}" in GEMMA_LAUNCHER
    for parameter in ("$LlamaCppDir", "$ModelPath", "$DraftModelPath", "$MmprojPath"):
        assert parameter in GEMMA_LAUNCHER
    assert "C:\\llama.cpp" not in GEMMA_LAUNCHER
    assert "D:\\LMStudio\\Models" not in GEMMA_LAUNCHER


def test_remote_model_api_key_wording_matches_browser_local_storage_contract():
    normalized = " ".join(REALTIME_README.split())
    assert "browser/device's `localStorage`" in normalized
    assert "excluded from UI-server persistence payloads" in normalized
    assert "browser-session state" not in normalized
    assert "browser-local API key" in MAIN_JS
    assert "page-session-only API key" not in MAIN_JS
    assert "declared provider Auto capability" in REALTIME_AGENTS
    assert "verified provider Auto support" not in REALTIME_AGENTS


def test_camera_capability_depends_on_enabled_state_not_stream_readiness():
    assert "if (toolsEnabled.camera_snapshot) defs.push(TOOL_DEFS.camera_snapshot);" in MAIN_JS
    assert "toolsEnabled.camera_snapshot && cameraStream" not in MAIN_JS
    assert "toolCamSwitch.checked = toolsEnabled.camera_snapshot;" in MAIN_JS
    permission_handler = MAIN_JS.split('status.addEventListener("change"', 1)[1].split("});", 1)[0]
    assert "pushToolsToSession();" in permission_handler


def test_browser_rejects_invalid_tool_arguments_without_executing_them():
    executor = MAIN_JS.split("async function runTool", 1)[1].split("function renderTtsBackendOptions", 1)[0]
    validation = executor.index("const validation = prepared.validation")
    invalid_result = executor.index("result.output = prepared.displayArguments")
    search = executor.index('name === "web_search"')
    camera = executor.index('name === "camera_snapshot"')
    tool_listener = MAIN_JS.split('c.addEventListener("toolcall"', 1)[1].split('c.addEventListener("error"', 1)[0]

    assert validation < invalid_result < search < camera
    assert tool_listener.index("prepareToolArgumentsForBrowser") < tool_listener.index("chat.onToolCall")
    assert "chat.onToolCall(name, args" not in tool_listener
    assert "chat.onToolResult(name, args" not in tool_listener
    assert 'type: "invalid_tool_arguments"' in CLIENT_JS
    assert 'error_class: failure.errorClass' in CLIENT_JS
    assert 'return { path: `${path}.*`, errorClass: "unexpected_property" }' in CLIENT_JS
    assert 'args = typeof event.arguments === "string" ? event.arguments : null' in CLIENT_JS
    assert 'JSON.parse(argsJson || "{}")' not in MAIN_JS
    assert 'additionalProperties: false' in WEB_SEARCH_JS
    assert 'pattern: "\\\\S"' in WEB_SEARCH_JS


def test_web_search_has_canonical_freshness_schema_and_truthful_structured_results():
    assert "WEB_SEARCH_ARGUMENT_SCHEMA" in MAIN_JS
    assert 'enum: ["auto", "web", "news"]' in WEB_SEARCH_JS
    assert 'enum: ["none", "day", "week", "month", "year"]' in WEB_SEARCH_JS
    assert 'required: ["query"]' in WEB_SEARCH_JS
    assert 'maxLength: 500' in WEB_SEARCH_JS
    assert 'cache: "no-store"' in MAIN_JS
    assert 'json?.type !== "web_search_result"' in MAIN_JS
    assert "output: JSON.stringify(json)" in MAIN_JS
    assert "Google search result from" not in MAIN_JS
    assert "new Date().toISOString().slice(0, 10)" not in MAIN_JS
    for field in (
        "requested_mode",
        "effective_mode",
        "freshness",
        "retrieved_at_utc",
        "recency_filter_applied",
        "fallback_applied",
        "date",
        "source",
        "position",
    ):
        assert f'"{field}"' in REALTIME_SERVER
    assert '"day": "qdr:d"' in REALTIME_SERVER
    assert '"week": "qdr:w"' in REALTIME_SERVER
    assert '"month": "qdr:m"' in REALTIME_SERVER
    assert '"year": "qdr:y"' in REALTIME_SERVER
    assert '"news": "https://google.serper.dev/news"' in REALTIME_SERVER
    assert '"web": "https://google.serper.dev/search"' in REALTIME_SERVER


def test_search_refinement_policy_is_shared_and_terminal_override_is_response_scoped():
    assert "one distinct narrower refinement" in MAIN_JS
    assert "class SearchTurnPolicy" in WEB_SEARCH_JS
    assert "isDistinctNarrowerSearch" in WEB_SEARCH_JS
    assert 'return { accepted: false, args, terminal: true, reason: "limit_reached" }' in WEB_SEARCH_JS
    assert 'return { accepted: false, args, terminal: true, reason: "not_narrower" }' in WEB_SEARCH_JS
    assert 'result.responseToolChoice = decision.terminal ? "none" : "auto"' in MAIN_JS
    assert "if (!decision.terminal) result.responseTools = [TOOL_DEFS.web_search]" in MAIN_JS
    assert 'responseToolChoice: /** @type {const} */ ("none")' in MAIN_JS
    assert "...(opts.tools ? { tools: opts.tools } : {})" in CLIENT_JS
    assert "...(opts.toolChoice ? { tool_choice: opts.toolChoice } : {})" in CLIENT_JS
    assert 'session: { type: "realtime", tools, tool_choice: tools.length ? "auto" : "none" }' in CLIENT_JS
    assert "beginAcceptedSearchTurn(turnState.itemId)" in MAIN_JS
    assert MAIN_JS.count("resetSearchTurnPolicy();") >= 3
    assert 'itemId: typeof event.item_id === "string" ? event.item_id : ""' in CLIENT_JS
    search_branch = MAIN_JS.split('} else if (name === "web_search") {', 1)[1].split(
        '} else if (name === "camera_snapshot") {', 1
    )[0]
    assert "query:" not in search_branch
    assert 'stage: "search"' in search_branch
    assert "result_count" not in search_branch


def test_terminal_search_keeps_exact_tool_output_image_create_order():
    executor = MAIN_JS.split("async function runTool", 1)[1].split("function renderTtsBackendOptions", 1)[0]
    assert executor.index("sessionClient.sendToolOutput") < executor.index("sessionClient.requestToolResponse")
    assert executor.index("sessionClient.requestToolResponse") < executor.index("await outputAck")
    request = CLIENT_JS.split("requestToolResponse(opts = {})", 1)[1].split("_responseActive()", 1)[0]
    assert request.index("if (opts.image) this.sendUserImage(opts.image)") < request.index(
        'type: "response.create"'
    )
    assert request.count('type: "response.create"') == 1


def test_missing_tool_call_identity_is_visible_but_never_executable():
    assert 'new CustomEvent("tool-protocol-error", { detail: { code } })' in CLIENT_JS
    dispatch = CLIENT_JS.split('case "response.function_call_arguments.done"', 1)[1].split("break;", 1)[0]
    assert dispatch.index("if (name && callId.trim())") < dispatch.index('new CustomEvent("toolcall"')
    assert 'const code = name ? "missing_call_id" : "missing_tool_name"' in dispatch
    assert 'c.addEventListener("tool-protocol-error"' in MAIN_JS
    assert "chat.onToolProtocolFailure(code)" in MAIN_JS
    assert 'type: "invalid_tool_call"' in CHAT_JS


def test_debug_logging_is_content_free_for_user_and_assistant_transcripts():
    assert "event.delta ?? event.transcript" in CLIENT_JS
    assert "chars=${String(event.delta ?? event.transcript ?? \"\").length}" in CLIENT_JS
    assert "${event.delta ?? event.transcript ?? \"\"}" not in CLIENT_JS
    assert "chars=${String(d.text || \"\").length}" in CHAT_JS
    assert "text=${JSON.stringify(d.text)}" not in CHAT_JS


def test_every_camera_call_has_distinct_visible_content_free_lifecycle():
    assert "let cameraCaptureGeneration = 0;" in MAIN_JS
    assert "capture_generation: lifecycle.captureGeneration" in MAIN_JS
    assert "requested_at_ms: lifecycle.requestedAtMs" in MAIN_JS
    assert 'stage: "camera"' in MAIN_JS
    assert "argsJson" not in MAIN_JS.split("function cameraLifecycleDetail", 1)[1].split("function beginToolLifecycle", 1)[0]
    assert 'return `camera:${lifecycle.captureGeneration}:${callId || "missing-call-id"}`' in CHAT_JS
    assert "Capture #${lifecycle.captureGeneration}" in CHAT_JS
    assert "this._updateToolLifecycle(existing, lifecycle)" in CHAT_JS
    assert "this._appendHistImage(image, existing)" in CHAT_JS
    assert "The camera is not available right now." in MAIN_JS
    assert "call camera_snapshot again" in MAIN_JS


def test_native_v3_is_the_migrated_default_and_echo_ui_is_truthful():
    assert 'echoGuardVersion: "s2s.ws.echoGuardVersion"' in MAIN_JS
    assert 'localStorage.setItem(STORAGE_KEYS.echoGuardVersion, "3")' in MAIN_JS
    assert 'value="native" selected>Native browser AEC (default)' in INDEX_HTML
    assert 'Adaptive (AEC3 reference cancellation)' in INDEX_HTML
    assert 'value="off"' not in INDEX_HTML
    assert 'requested === "strict" ? "strict" : "native"' in MIC_CAPTURE_JS
    assert 'const EXPECTED_UI_API_VERSION = 21;' in MAIN_JS
    assert 'const EXPECTED_BACKEND_API_VERSION = 7;' in MAIN_JS
    assert "Do not reuse a stock " in MAIN_JS
    assert "Let me check that" not in MAIN_JS
    assert 'src="main.js?v=32-search-freshness"' in INDEX_HTML
    assert '"./ws/s2s-ws-client.js?v=19-search-freshness"' in MAIN_JS
    assert '"./tools/web-search.js?v=1-search-freshness"' in MAIN_JS
    assert '"./ui/chat.js?v=3-tool-privacy"' in MAIN_JS
    assert "loadAec3Worklet(ctx)" in CLIENT_JS
    assert 'new URL("mic-capture.js?v=12-aec3-fallback", base)' in CLIENT_JS
    assert 'new URL("audio-playback.js?v=16-adaptive-safe-start", base)' in CLIENT_JS
    assert "new AudioWorkletNode(ctx, aec3.processorName" in CLIENT_JS


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


def test_audio_cpp_settings_proxy_is_explicit():
    SERVER = (ROOT / "web" / "hf-realtime-voice" / "server.py").read_text(encoding="utf-8")
    assert '@app.get("/api/audio-cpp/settings")' in SERVER
    assert '@app.put("/api/audio-cpp/settings")' in SERVER
    assert "voice-studio/settings" in SERVER


def test_hf_ui_preserves_nonsecret_settings_across_browser_environment_reset():
    SERVER = (ROOT / "web" / "hf-realtime-voice" / "server.py").read_text(encoding="utf-8")
    assert '@app.get("/api/ui-settings")' in SERVER
    assert '@app.put("/api/ui-settings")' in SERVER
    assert "PUBLIC_UI_SETTING_KEYS" in SERVER
    assert "modelApiKey" not in SERVER.split("PUBLIC_UI_SETTING_KEYS", 1)[1].split("DEFAULT_QWEN3", 1)[0]
    assert "restorePersistentSettings" in MAIN_JS
    assert 'fetch("/api/ui-settings"' in MAIN_JS


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
    assert "renderSelectedTtsBackendStatus()" in change


def test_audio_cpp_profile_refresh_is_used_for_backend_changes_and_session_payloads():
    change = MAIN_JS.split('inputTtsBackend.addEventListener("change"', 1)[1].split("});", 1)[0]
    assert "refreshCandidateTuningProfiles().catch" in change
    assert "async function resolveCandidateTuning" in MAIN_JS
    assert "function resolvedCandidatePhraseQueue" in MAIN_JS
    assert "payload.resolved = phraseQueue" in MAIN_JS
    assert "let realtimeTuningByBackend = {};" in MAIN_JS
    assert "ttsProfileByBackend" in MAIN_JS
    form_settings = MAIN_JS.split("function readSettingsFromForm()", 1)[1].split("function syncModelProviderUi()", 1)[0]
    assert "activeTtsTuning(backend)" in form_settings
    pipeline = MAIN_JS.split("pipelineConfig:", 1)[1].split("...(audioContext", 1)[0]
    assert "...(activeTtsTuning(settings.ttsBackend)" in pipeline
    assert "client.updateLocalPipeline(" in MAIN_JS
    assert "activePlaybackConfig(settings.ttsBackend)" in MAIN_JS
    assert "tts_backend: settings.ttsBackend" in MAIN_JS
    assert "tts_tuning: tuning" in MAIN_JS
    active = MAIN_JS.split("function activeTtsTuning", 1)[1].split("function updateRealtimeAudioSummary", 1)[0]
    assert "if (normalizeTtsProvider(backend) !== AUDIO_CPP_PROVIDER) return null" in active
    assert "const { scope: _restScope, ...sessionTuning } = payload" in active
    assert "scope" not in active.split("return sessionTuning", 1)[1]


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
    assert "return sessionTuning" in session
    assert "_restScope" in session
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
    unreachable_guard = 'if (json?.reachable === false) throw new Error("Voice backend unavailable");'
    assert unreachable_guard in fetcher
    assert fetcher.index(unreachable_guard) < fetcher.index("applyVoiceProfilePayload(json, backend)")
    failure_branch = fetcher.split("} catch (err) {", 1)[1]
    assert 'settings.voice = "";' not in failure_branch
    assert "saved selection retained" in failure_branch
    renderer = MAIN_JS.split("function renderVoiceOptions()", 1)[1].split(
        "function clearVoiceProfileOptions", 1
    )[0]
    assert 'settings.voice = "";' not in renderer
    applier = MAIN_JS.split("function applyVoiceProfilePayload", 1)[1].split(
        "async function selectVoiceProfile", 1
    )[0]
    empty_inventory = applier.split("} else {", 1)[1]
    assert 'settings.voice = "";' in empty_inventory
    assert "void saveSettings(settings);" in empty_inventory


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
    assert "Model / Realtime transport" in INDEX_HTML
    assert "24 kHz / 16 kHz PCM16" in INDEX_HTML
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
    assert 'this._diagnostic("started"' in PLAYBACK_JS
    assert "first_playback_ms" in CLIENT_JS
    assert "end_to_end_ms" in CLIENT_JS


def test_native_pcm_adaptive_safe_start_is_acknowledged_bounded_and_sample_exact():
    for field in (
        "profileRevision",
        "model",
        "clone",
        "firstBlockFrames",
        "steadyBlockFrames",
        "outputRate",
    ):
        assert field in MAIN_JS
    assert "resolveAdaptivePlaybackSignature" in CLIENT_JS
    assert "AdaptivePlaybackPolicyStore" in CLIENT_JS
    assert "PLAYBACK_LEARNING_TTL_MS" in CLIENT_JS
    assert "MAX_PLAYBACK_LEARNING_SIGNATURES" in CLIENT_JS
    assert "PLAYBACK_GAP_WINDOW" in CLIENT_JS
    assert "snapshot.policy.targetMs >= snapshot.policy.ceilingMs" in CLIENT_JS
    assert "record.recoveryCleanCount >= 3" in CLIENT_JS
    assert "inputSampleOffset" in CLIENT_JS
    assert "inputSampleCount" in CLIENT_JS
    assert 'this._diagnostic("stream_drained"' in PLAYBACK_JS
    assert 'this._reject("audio", receivedGeneration, "input_sample_mismatch")' in PLAYBACK_JS


def test_realtime_audio_diagnostics_show_paired_reference_truth_without_transcript_content():
    assert "done.delivery_mode" in MAIN_JS
    assert "done.reference_source_seconds" in MAIN_JS
    assert "done.reference_requested_limit_seconds" in MAIN_JS
    assert "done.reference_used_seconds" in MAIN_JS
    assert "done.reference_limit_applied" in MAIN_JS
    assert "done.reference_pairing" in MAIN_JS
    assert "reference_transcript" not in MAIN_JS


def test_realtime_audio_diagnostics_show_requested_effective_language_and_auto_support():
    assert "done.requested_language" in MAIN_JS
    assert "done.effective_language" in MAIN_JS
    assert "done.language_auto_supported" in MAIN_JS
    assert "Auto ${autoLanguageText}" in MAIN_JS


def test_audio_cpp_proxy_is_incremental_and_candidate_labels_are_truthful():
    server = (ROOT / "web" / "hf-realtime-voice" / "server.py").read_text(encoding="utf-8")
    assert "StreamingResponse(" in server
    assert "async for chunk in upstream.aiter_raw()" in server
    assert "await request.is_disconnected()" in server
    assert 'result["nativeIncrementalPcm"]' in server
    assert 'result["bufferedFallback"] = not result["nativeIncrementalPcm"]' in server
    assert 'selectedTts?.nativeStreaming ? "native PCM" : "buffered fallback"' in MAIN_JS
