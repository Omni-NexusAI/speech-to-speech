// @ts-check
/**
 * Minimal voice conversation app, talking to a Hugging Face speech-to-speech
 * backend over **WebSocket** (drop-in alternative to the WebRTC variant).
 *
 * Click the orb -> we ask for the mic, POST a session on the LB, open a
 * WebSocket on the routed compute endpoint, push session.update + mic
 * audio, play back the TTS audio. The orb visually reflects the live
 * state (idle, connecting, listening, user-speaking, processing,
 * ai-speaking).
 *
 * The only meaningful difference vs. the WebRTC main.js is that the
 * client owns its own AudioContext (no `attachOutputTrack`), so we hand
 * it the MediaStream directly.
 *
 * @typedef {"idle" | "connecting" | "queued" | "your-turn" | "listening" | "user-speaking" | "processing" | "ai-speaking" | "error"} AppState
 */

import { S2sWsRealtimeClient } from "./ws/s2s-ws-client.js?v=17-synthesis-outcomes";
import { $, truncateError, DEBUG } from "./ui/dom.js";
import { ChatView } from "./ui/chat.js";
import { StartAttemptController } from "./ui/start-attempt.js";
import { Account } from "./ui/account.js";
import { formatHistoryCompactionDiagnostics } from "./ui/history-compaction-diagnostics.js";

const DEFAULT_VOICE = "clone:16d9bb336799";
const DEFAULT_INSTRUCTIONS =
  "You are a friendly voice assistant. " +
  "Keep replies short, warm, and spoken. Avoid long monologues.";

// Appended to the user's instructions whenever at least one tool is enabled.
// Stops the model from announcing capabilities ("Yes, I can search") and then
// idling for the next turn — it should act immediately in the same response.
const TOOL_USE_HINT =
  " When the user's request calls for one of your tools, briefly acknowledge " +
  "that you are acting before the tool call, using natural wording that fits " +
  "the specific request and varies with the conversation. Do not reuse a stock " +
  "phrase, describe capabilities, or wait for another turn. Call the tool right " +
  "away in the same response.";

const STORAGE_KEYS = {
  // Direct s2s server URL, used only when the deploy has no LOAD_BALANCER_URL
  // (in LB mode the browser never learns the LB address — it POSTs /api/session).
  directUrl: "s2s.ws.directUrl",
  voice: "s2s.ws.voice",
  instructions: "s2s.ws.instructions",
  tools: "s2s.ws.tools",
  searchKey: "s2s.ws.searchKey",
  noiseGate: "s2s.ws.noiseGate",
  echoGuard: "s2s.ws.echoGuard",
  echoGuardVersion: "s2s.ws.echoGuardVersion",
  echoCalibrations: "s2s.ws.echoCalibrations",
  diagnostics: "s2s.ws.diagnostics",
  diagnosticsGeometry: "s2s.ws.diagnosticsGeometry",
  fullBufferTts: "s2s.ws.fullBufferTts",
  liveTranscript: "s2s.ws.liveTranscript",
  maxResponseTokens: "s2s.ws.maxResponseTokens",
  ttsBackend: "s2s.ws.ttsBackend",
  voiceByBackend: "s2s.ws.voiceByBackend",
  ttsProfileByBackend: "s2s.ws.ttsProfileByBackend",
  playbackContinuity: "s2s.ws.playbackContinuity",
  ttsDeliveryMode: "s2s.ws.ttsDeliveryMode",
  historyCompaction: "s2s.ws.historyCompaction",
  modelProvider: "s2s.ws.modelProvider",
  modelUrl: "s2s.ws.modelUrl",
  modelName: "s2s.ws.modelName",
};

// Older builds mistakenly treated the model API key as a local preference.
// Keep the literal separate from STORAGE_KEYS so new code cannot accidentally
// read or write it.  Migration deliberately removes the entry without
// retrieving or logging its value.
const LEGACY_MODEL_API_KEY_STORAGE_KEY = "s2s.ws.modelApiKey";

// ── Noise gate ──────────────────────────────────────────────────────────────
// The Settings cursor sets the gate's open threshold in dBFS. Its leftmost
// position is an OFF detent (gate disabled, pure passthrough); the rest of the
// travel is the active threshold. The cursor shares the meter's dB axis, so the
// handle sits on the level bar — raise it until room noise stops lighting it up.
// The slider range IS the shared axis: the live meter fill and the threshold
// thumb both map across [GATE_OFF_DB, GATE_MAX_DB], so the thumb sits exactly
// where the gate cuts on the same scale as the level bar.
const GATE_OFF_DB = -66; // slider minimum = off / bottom of the meter axis
const GATE_MAX_DB = -3; // slider maximum = most aggressive / top of the meter axis
const GATE_DEFAULT_DB = -50; // first-run default: a gentle gate, enabled

/** @param {number} thresholdDb @returns {import("./ws/s2s-ws-client.js").NoiseGate} */
function gateParams(thresholdDb) {
  return { enabled: thresholdDb > GATE_OFF_DB, thresholdDb };
}

// ── Tools ─────────────────────────────────────────────────────────────────
// Function tools we declare to the backend. The model decides when to call
// one; the executor below runs it and returns the result (see runTool).
/** @type {Record<string, import("./ws/s2s-ws-client.js").ToolDef>} */
const TOOL_DEFS = {
  web_search: {
    type: "function",
    name: "web_search",
    description:
      "Search the web for current or factual information you don't already know " +
      "(news, prices, facts, documentation). Returns the top results with titles, " +
      "snippets and URLs.",
    parameters: {
      type: "object",
      properties: { query: { type: "string", description: "The search query." } },
      required: ["query"],
    },
  },
  camera_snapshot: {
    type: "function",
    name: "camera_snapshot",
    description:
      "Capture the current frame from the user's webcam so you can see what they " +
      "are showing you. Use it whenever the user refers to something visual or " +
      "asks you to look.",
    parameters: { type: "object", properties: {}, required: [] },
  },
};

/** Longest edge of the snapshot sent to the VLM, in px (keeps payload sane). */
const SNAPSHOT_MAX_EDGE = 768;
const SNAPSHOT_QUALITY = 0.7;

function loadSettings() {
  // API keys are page-session-only.  Remove a legacy persisted entry before
  // reading any preferences; do not inspect the value during this migration.
  localStorage.removeItem(LEGACY_MODEL_API_KEY_STORAGE_KEY);
  const storedEchoGuard = localStorage.getItem(STORAGE_KEYS.echoGuard);
  const echoGuardVersion = localStorage.getItem(STORAGE_KEYS.echoGuardVersion);
  // "off" was the historical name for browser-native AEC.  Migrate it
  // truthfully and keep Native as the safe default while AEC3 is unavailable.
  const echoGuard = storedEchoGuard === "strict" ? "strict" :
    storedEchoGuard === "adaptive" ? "adaptive" : "native";
  if (echoGuardVersion !== "3") {
    localStorage.setItem(STORAGE_KEYS.echoGuard, echoGuard);
    localStorage.setItem(STORAGE_KEYS.echoGuardVersion, "3");
  }
  const storedTtsBackend = localStorage.getItem(STORAGE_KEYS.ttsBackend) || "faster";
  let voiceByBackend = {};
  try { voiceByBackend = JSON.parse(localStorage.getItem(STORAGE_KEYS.voiceByBackend) || "{}"); } catch (_) {}
  if (!voiceByBackend || typeof voiceByBackend !== "object") voiceByBackend = {};
  let ttsProfileByBackend = {};
  try { ttsProfileByBackend = JSON.parse(localStorage.getItem(STORAGE_KEYS.ttsProfileByBackend) || "{}"); } catch (_) {}
  if (!ttsProfileByBackend || typeof ttsProfileByBackend !== "object") ttsProfileByBackend = {};
  const normalizedBackend = normalizeTtsProvider(storedTtsBackend);
  voiceByBackend = normalizeTtsProviderMap(voiceByBackend);
  ttsProfileByBackend = normalizeTtsProviderMap(ttsProfileByBackend);
  const storedVoice = voiceByBackend[normalizedBackend] || localStorage.getItem(STORAGE_KEYS.voice) || DEFAULT_VOICE;
  return {
    directUrl: localStorage.getItem(STORAGE_KEYS.directUrl) || "http://127.0.0.1:8765",
    voice: storedVoice,
    voiceByBackend: { faster: DEFAULT_VOICE, ...voiceByBackend, [normalizedBackend]: storedVoice },
    ttsProfileByBackend,
    playbackContinuity: localStorage.getItem(STORAGE_KEYS.playbackContinuity) === "fast-start" ? "fast-start" : "adaptive",
    // Native PCM is experimental until an explicit per-session choice. Never
    // infer it from a healthy candidate capability.
    ttsDeliveryMode: localStorage.getItem(STORAGE_KEYS.ttsDeliveryMode) === "native_incremental_pcm"
      ? "native_incremental_pcm"
      : "buffered_phrase",
    historyCompaction: normalizeHistoryCompaction(loadJsonSetting(STORAGE_KEYS.historyCompaction)),
    instructions: localStorage.getItem(STORAGE_KEYS.instructions) || DEFAULT_INSTRUCTIONS,
    noiseGate: loadGateThreshold(),
    echoGuard,
    echoCalibrations: loadEchoCalibrations(),
    fullBufferTts: localStorage.getItem(STORAGE_KEYS.fullBufferTts) === "1",
    liveTranscript: localStorage.getItem(STORAGE_KEYS.liveTranscript) === "1",
    maxResponseTokens: Math.min(1024, Math.max(64, Number(localStorage.getItem(STORAGE_KEYS.maxResponseTokens)) || 384)),
    // Preserve old local settings while converging on the public provider name.
    ttsBackend: normalizedBackend,
    modelProvider: localStorage.getItem(STORAGE_KEYS.modelProvider) || "local",
    modelUrl: localStorage.getItem(STORAGE_KEYS.modelUrl) || "",
    modelName: localStorage.getItem(STORAGE_KEYS.modelName) || "",
    modelApiKey: "",
  };
}

/** Stored gate threshold (dBFS), clamped to the slider range. Defaults to a
 * gentle enabled gate (GATE_DEFAULT_DB) when the user hasn't set one yet. */
function loadGateThreshold() {
  const stored = localStorage.getItem(STORAGE_KEYS.noiseGate);
  // getItem returns null when unset, and Number(null) === 0 (finite!), so guard
  // the missing/empty case explicitly before coercing — otherwise the default
  // never fires and 0 clamps to the slider max.
  if (stored === null || stored === "") return GATE_DEFAULT_DB;
  const raw = Number(stored);
  if (!Number.isFinite(raw)) return GATE_DEFAULT_DB;
  return Math.min(GATE_MAX_DB, Math.max(GATE_OFF_DB, Math.round(raw)));
}

/** @param {ReturnType<typeof loadSettings>} s */
function saveSettings(s) {
  s.voiceByBackend = { ...(s.voiceByBackend || {}), [s.ttsBackend]: s.voice };
  localStorage.setItem(STORAGE_KEYS.directUrl, s.directUrl);
  localStorage.setItem(STORAGE_KEYS.voice, s.voice);
  localStorage.setItem(STORAGE_KEYS.instructions, s.instructions);
  localStorage.setItem(STORAGE_KEYS.noiseGate, String(s.noiseGate));
  localStorage.setItem(STORAGE_KEYS.echoGuard, s.echoGuard);
  localStorage.setItem(STORAGE_KEYS.echoGuardVersion, "3");
  localStorage.setItem(STORAGE_KEYS.echoCalibrations, JSON.stringify(s.echoCalibrations || {}));
  localStorage.setItem(STORAGE_KEYS.playbackContinuity, s.playbackContinuity === "fast-start" ? "fast-start" : "adaptive");
  localStorage.setItem(STORAGE_KEYS.ttsDeliveryMode, s.ttsDeliveryMode === "native_incremental_pcm" ? "native_incremental_pcm" : "buffered_phrase");
  s.historyCompaction = normalizeHistoryCompaction(s.historyCompaction);
  localStorage.setItem(STORAGE_KEYS.historyCompaction, JSON.stringify(s.historyCompaction));
  localStorage.setItem(STORAGE_KEYS.fullBufferTts, s.fullBufferTts ? "1" : "0");
  localStorage.setItem(STORAGE_KEYS.liveTranscript, s.liveTranscript ? "1" : "0");
  localStorage.setItem(STORAGE_KEYS.maxResponseTokens, String(s.maxResponseTokens));
  localStorage.setItem(STORAGE_KEYS.ttsBackend, s.ttsBackend);
  localStorage.setItem(STORAGE_KEYS.voiceByBackend, JSON.stringify(s.voiceByBackend));
  localStorage.setItem(STORAGE_KEYS.ttsProfileByBackend, JSON.stringify(s.ttsProfileByBackend || {}));
  localStorage.setItem(STORAGE_KEYS.modelProvider, s.modelProvider);
  localStorage.setItem(STORAGE_KEYS.modelUrl, s.modelUrl);
  localStorage.setItem(STORAGE_KEYS.modelName, s.modelName);
  // Browser storage is convenient, but an environment/browser reset clears it.
  // Preserve only non-secret preferences in the managed local UI state. API
  // keys remain browser-only and are intentionally excluded from this payload.
  return fetch("/api/ui-settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(publicSettingsPayload(s)),
  }).then((response) => {
    if (!response.ok) throw new Error(`settings save failed: HTTP ${response.status}`);
    return response.json().then((payload) => ({ ok: true, payload }));
  }).catch((error) => ({ ok: false, error: error instanceof Error ? error.message : String(error) }));
}

function loadEchoCalibrations() {
  try {
    const value = JSON.parse(localStorage.getItem(STORAGE_KEYS.echoCalibrations) || "{}");
    return value && typeof value === "object" && !Array.isArray(value) ? value : {};
  } catch {
    return {};
  }
}

function loadJsonSetting(key) {
  try {
    const value = JSON.parse(localStorage.getItem(key) || "null");
    return value && typeof value === "object" && !Array.isArray(value) ? value : null;
  } catch {
    return null;
  }
}

function normalizeHistoryCompaction(value) {
  const raw = value && typeof value === "object" ? value : {};
  const trigger = Math.max(0.20, Math.min(0.90, Number(raw.trigger_ratio) || 0.70));
  const target = Math.max(0.10, Math.min(trigger - 0.05, Number(raw.target_ratio) || 0.50));
  return {
    enabled: raw.enabled !== false,
    trigger_ratio: trigger,
    target_ratio: target,
    recent_turns: Math.max(1, Math.min(12, Math.round(Number(raw.recent_turns) || 6))),
  };
}

function syncHistoryCompactionControls() {
  const config = normalizeHistoryCompaction(settings.historyCompaction);
  historyCompactionEnabled.checked = config.enabled;
  historyCompactionTrigger.value = config.trigger_ratio.toFixed(2);
  historyCompactionTarget.value = config.target_ratio.toFixed(2);
  historyCompactionRecentTurns.value = String(config.recent_turns);
}

function renderHistoryCompactionDiagnostics(contextDetail = null) {
  const configured = normalizeHistoryCompaction(settings.historyCompaction);
  const acknowledged = acknowledgedHistoryCompaction;
  const rendered = formatHistoryCompactionDiagnostics({ configured, acknowledged, contextDetail });
  if (diagnosticsHistoryCompactionSummary) {
    diagnosticsHistoryCompactionSummary.textContent = rendered.summary;
  }
  if (!diagnosticsHistoryCompactionStatus) return;
  diagnosticsHistoryCompactionStatus.textContent = rendered.status;
}

async function applyHistoryCompactionSettings() {
  settings.historyCompaction = normalizeHistoryCompaction({
    enabled: historyCompactionEnabled.checked,
    trigger_ratio: Number(historyCompactionTrigger.value),
    target_ratio: Number(historyCompactionTarget.value),
    recent_turns: Number(historyCompactionRecentTurns.value),
  });
  syncHistoryCompactionControls();
  acknowledgedHistoryCompaction = null;
  const saved = await saveSettings(settings);
  if (!saved.ok) {
    if (diagnosticsHistoryCompactionStatus) {
      diagnosticsHistoryCompactionStatus.textContent = `Could not persist compaction settings: ${saved.error}`;
    }
    return;
  }
  if (client && LIVE_STATES.has(currentState)) {
    client.updateLocalPipeline({ history_compaction: settings.historyCompaction });
  }
  renderHistoryCompactionDiagnostics();
}

function publicSettingsPayload(s) {
  return {
    directUrl: s.directUrl,
    voice: s.voice,
    voiceByBackend: s.voiceByBackend,
    ttsProfileByBackend: s.ttsProfileByBackend || {},
    playbackContinuity: s.playbackContinuity === "fast-start" ? "fast-start" : "adaptive",
    ttsDeliveryMode: s.ttsDeliveryMode === "native_incremental_pcm" ? "native_incremental_pcm" : "buffered_phrase",
    historyCompaction: normalizeHistoryCompaction(s.historyCompaction),
    instructions: s.instructions,
    noiseGate: s.noiseGate,
    echoGuard: s.echoGuard,
    echoCalibrations: s.echoCalibrations || {},
    fullBufferTts: s.fullBufferTts,
    liveTranscript: s.liveTranscript,
    maxResponseTokens: s.maxResponseTokens,
    ttsBackend: s.ttsBackend,
    modelProvider: s.modelProvider,
    modelUrl: s.modelUrl,
    modelName: s.modelName,
  };
}

async function restorePersistentSettings() {
  try {
    const response = await fetch("/api/ui-settings");
    if (!response.ok) return;
    const payload = await response.json();
    if (!payload?.settings || typeof payload.settings !== "object") return;
    // Preserve any page-session-only API key already present in this browser.
    settings = { ...settings, ...payload.settings, modelApiKey: settings.modelApiKey };
    settings.ttsBackend = normalizeTtsProvider(settings.ttsBackend);
    settings.voiceByBackend = normalizeTtsProviderMap(
      payload.settings.voiceByBackend || settings.voiceByBackend || {},
    );
    settings.ttsProfileByBackend = normalizeTtsProviderMap(
      payload.settings.ttsProfileByBackend || settings.ttsProfileByBackend || {},
    );
    settings.voice = settings.voiceByBackend[settings.ttsBackend] || settings.voice || DEFAULT_VOICE;
    await saveSettings(settings);
    if (settings.modelProvider === "local") void refreshLocalPipeline();
  } catch (_) {
    // The UI remains fully usable from browser-local settings if the managed
    // frontend is temporarily unavailable during startup.
  }
}

/** @returns {{ web_search: boolean, camera_snapshot: boolean }} */
function loadTools() {
  try {
    const raw = JSON.parse(localStorage.getItem(STORAGE_KEYS.tools) || "{}");
    // Both tools default ON (web search still only activates when a key exists).
    // We never call getUserMedia on page load — the camera only actually starts
    // on a user gesture (conversation start), so a default-on flag doesn't
    // silently resume the webcam; an explicit saved `false` is respected.
    return {
      web_search: raw.web_search ?? true,
      camera_snapshot: raw.camera_snapshot ?? true,
    };
  } catch {
    return { web_search: true, camera_snapshot: true };
  }
}

function saveTools() {
  localStorage.setItem(STORAGE_KEYS.tools, JSON.stringify(toolsEnabled));
}

/** @type {Record<AppState, { caption: string; disabled: boolean }>} */
const STATE_VIEWS = {
  idle:            { caption: "Tap to start",  disabled: false },
  connecting:      { caption: "Connecting",    disabled: true  },
  queued:          { caption: "Finding you a spot…", disabled: true },
  "your-turn":     { caption: "You're up! 🎉", disabled: true  },
  listening:       { caption: "",              disabled: false },
  "user-speaking": { caption: "",              disabled: false },
  processing:      { caption: "",              disabled: false },
  "ai-speaking":   { caption: "",              disabled: false },
  error:           { caption: "Tap to retry",  disabled: false },
};

/** @type {Record<AppState, string>} */
const STATE_CLASS = {
  idle: "state-idle",
  connecting: "state-connecting",
  queued: "state-queued",
  "your-turn": "state-your-turn",
  listening: "state-listening",
  "user-speaking": "state-user-speaking",
  processing: "state-processing",
  "ai-speaking": "state-ai-speaking",
  error: "state-error",
};

/** @type {ReadonlySet<AppState>} */
const LIVE_STATES = new Set(["listening", "user-speaking", "processing", "ai-speaking"]);

/** @type {HTMLButtonElement} */
const circleBtn = $("#main-circle");
/** @type {HTMLParagraphElement} */
const circleCaption = $("#circle-caption");
/** @type {HTMLParagraphElement} */
const circleSubcaption = $("#circle-subcaption");
/** @type {HTMLElement} */
const orbWrap = $(".orb-wrap");
/** @type {HTMLButtonElement} */
const micBtn = $("#mic-btn");
/** @type {HTMLButtonElement} */
const stopBtn = $("#stop-btn");
/** @type {HTMLElement} */
const queueActions = $("#queue-actions");
/** @type {HTMLButtonElement} */
const joinQueueBtn = $("#join-queue-btn");
/** @type {HTMLButtonElement} */
const leaveQueueBtn = $("#leave-queue-btn");

/** @type {HTMLButtonElement} */
const settingsBtn = $("#settings-btn");
/** @type {HTMLDialogElement} */
const settingsModal = $("#settings-modal");

/** @type {HTMLButtonElement} */
const aboutBtn = $("#about-btn");
/** @type {HTMLDialogElement} */
const aboutModal = $("#about-modal");
/** @type {HTMLButtonElement} */
const aboutClose = $("#about-close");

/** @type {HTMLButtonElement} */
const diagnosticsBtn = $("#diagnostics-btn");
/** @type {HTMLElement} */
const diagnosticsPanel = $("#diagnostics-panel");
/** @type {HTMLButtonElement} */
const diagnosticsClose = $("#diagnostics-close");
/** @type {HTMLElement} */
const diagnosticsList = $("#diagnostics-list");
/** @type {HTMLElement} */
const diagnosticsSummary = $("#diagnostics-summary");
const diagnosticsGraph = $("#diagnostics-graph");
const diagnosticsWaterfall = $("#diagnostics-waterfall");
const diagnosticsWarning = $("#diagnostics-warning");
const diagnosticsTtsProfile = $("#diagnostics-tts-profile");
const diagnosticsTtsProfileApply = $("#diagnostics-tts-profile-apply");
const diagnosticsTtsProfileName = $("#diagnostics-tts-profile-name");
const diagnosticsTtsProfileSaveAs = $("#diagnostics-tts-profile-save-as");
const diagnosticsAudioSummary = $("#diagnostics-audio-summary");
const diagnosticsAudioStatus = $("#diagnostics-audio-status");
const diagnosticsAudioMetrics = $("#diagnostics-audio-metrics");
const diagnosticsPlaybackContinuity = $("#diagnostics-playback-continuity");
const diagnosticsTtsDeliveryMode = $("#diagnostics-tts-delivery-mode");
const diagnosticsEchoDevicePair = $("#diagnostics-echo-device-pair");
const diagnosticsEchoOutputLatency = $("#diagnostics-echo-output-latency");
const diagnosticsEchoSave = $("#diagnostics-echo-save");
const diagnosticsEchoUseMeasured = $("#diagnostics-echo-use-measured");
const diagnosticsEchoInputs = Array.from(document.querySelectorAll("[data-echo-calibration-key]"));
let latestEchoStatus = null;
const diagnosticsTuningStatus = $("#diagnostics-tuning-status");
const diagnosticsTuningWarnings = $("#diagnostics-tuning-warnings");
const diagnosticsTuningApply = $("#diagnostics-tuning-apply");
const diagnosticsTuningClear = $("#diagnostics-tuning-clear");
const diagnosticsTuningInputs = Array.from(document.querySelectorAll("[data-tuning-key]"));
const diagnosticsTuningNamedValues = $("#diagnostics-tuning-named-values");
const diagnosticsTuningOverrideValues = $("#diagnostics-tuning-override-values");
const diagnosticsTuningEffectiveValues = $("#diagnostics-tuning-effective-values");
const diagnosticsTuningContextUnlock = $("#tuning-context-unlock");
const diagnosticsTuningContext = $("#tuning-left-context-frames");
const diagnosticsContextWarning = $("#diagnostics-context-warning");
const historyCompactionEnabled = $("#history-compaction-enabled");
const historyCompactionTrigger = $("#history-compaction-trigger");
const historyCompactionTarget = $("#history-compaction-target");
const historyCompactionRecentTurns = $("#history-compaction-recent-turns");
const diagnosticsHistoryCompactionSummary = $("#diagnostics-history-compaction-summary");
const diagnosticsHistoryCompactionStatus = $("#diagnostics-history-compaction-status");
let acknowledgedHistoryCompaction = null;
let candidateTuningResolved = null;
let candidateTuningProfileDocument = null;
let candidateInactiveTuningFields = new Set();
const AUDIO_CPP_PROVIDER = "qwen3tts-audiocpp";
/** Page-session-only selections and overrides, isolated by TTS provider. */
let realtimeTuningByBackend = {};

function normalizeTtsProvider(provider) {
  return provider === "audio-cpp" ? AUDIO_CPP_PROVIDER : provider;
}

function normalizeTtsProviderMap(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};
  const normalized = { ...value };
  if (normalized["audio-cpp"] !== undefined && normalized[AUDIO_CPP_PROVIDER] === undefined) {
    normalized[AUDIO_CPP_PROVIDER] = normalized["audio-cpp"];
  }
  delete normalized["audio-cpp"];
  return normalized;
}

function realtimeTuningState(backend = settings.ttsBackend) {
  const current = realtimeTuningByBackend[backend];
  const savedProfile = settings.ttsProfileByBackend?.[backend] || "balanced";
  if (current && typeof current === "object") {
    return {
      profile_id: current.profile_id || savedProfile,
      overrides: { ...(current.overrides || {}) },
    };
  }
  return { profile_id: savedProfile, overrides: {} };
}

function setRealtimeTuningState(backend, profileId, overrides = {}) {
  realtimeTuningByBackend = {
    ...realtimeTuningByBackend,
    [backend]: { profile_id: profileId || "balanced", overrides: { ...overrides } },
  };
  return realtimeTuningByBackend[backend];
}

const SAFE_TUNING_KEYS = [
  "model",
  "max_reference_seconds",
  "first_block_frames",
  "steady_block_frames",
  "left_context_frames",
  "text_lookahead",
  "phrase_flush_ms",
  "temperature",
  "top_k",
  "top_p",
  "repetition_penalty",
  "seed",
];

function effectiveCandidateTuning(resolved = candidateTuningResolved) {
  if (!resolved || typeof resolved !== "object") return {};
  return {
    ...(resolved.profile || {}),
    ...(resolved.effective || {}),
    ...(resolved.temporaryOverrides || {}),
  };
}

function tuningProfileSlug(name) {
  return String(name || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 64);
}

function compactTuningSummary(values, keys = SAFE_TUNING_KEYS) {
  const present = keys.filter((key) => values?.[key] !== undefined && values?.[key] !== null && values?.[key] !== "");
  if (!present.length) return "none";
  const preview = present.slice(0, 4).map((key) => `${key}=${values[key]}`).join(" · ");
  return present.length > 4 ? `${preview} · +${present.length - 4} more` : preview;
}

function syncDecoderContextUnlock() {
  if (!diagnosticsTuningContext || !diagnosticsTuningContextUnlock) return;
  const available = settings.ttsBackend === AUDIO_CPP_PROVIDER
    && !candidateInactiveTuningFields.has("left_context_frames");
  diagnosticsTuningContextUnlock.disabled = !available;
  diagnosticsTuningContext.disabled = !available || !diagnosticsTuningContextUnlock.checked;
  diagnosticsContextWarning?.classList.toggle("active", available && diagnosticsTuningContextUnlock.checked);
}

function resolvedCandidatePhraseQueue(resolved = candidateTuningResolved) {
  if (!resolved || typeof resolved !== "object") return null;
  const effective = effectiveCandidateTuning(resolved);
  const textLookahead = Number(effective.text_lookahead);
  const phraseFlushMs = Number(effective.phrase_flush_ms);
  if (!Number.isFinite(textLookahead) || !Number.isFinite(phraseFlushMs)) return null;
  return { text_lookahead: textLookahead, phrase_flush_ms: phraseFlushMs };
}

function candidateRestTuningPayload(
  overrides = realtimeTuningState(AUDIO_CPP_PROVIDER).overrides,
  resolved = candidateTuningResolved,
  profileId = diagnosticsTtsProfile?.value || realtimeTuningState(AUDIO_CPP_PROVIDER).profile_id,
) {
  const payload = {
    provider: AUDIO_CPP_PROVIDER,
    scope: "realtime",
    profile_id: profileId || "balanced",
    overrides,
  };
  const phraseQueue = resolved?.profile?.id === profileId
    ? resolvedCandidatePhraseQueue(resolved)
    : null;
  if (phraseQueue) payload.resolved = phraseQueue;
  return payload;
}

function activeTtsTuning(backend = settings.ttsBackend) {
  if (normalizeTtsProvider(backend) !== AUDIO_CPP_PROVIDER) return null;
  const state = realtimeTuningState(backend);
  const profile = candidateTuningResolved?.profile || {};
  const revision = Number(profile.revision ?? candidateTuningResolved?.profile_revision);
  const effectiveValues = effectiveCandidateTuning(candidateTuningResolved);
  // WebSocket session config is an immutable response-input snapshot. Do not
  // send the mutable UI diagnostic object (`resolved`) or a partial profile.
  if (!Number.isSafeInteger(revision) || revision <= 0) return null;
  const effective = { clone_mode: "full_icl" };
  for (const key of SAFE_TUNING_KEYS) {
    if (effectiveValues[key] === undefined) return null;
    effective[key] = effectiveValues[key];
  }
  return {
    provider: AUDIO_CPP_PROVIDER,
    profile_id: state.profile_id,
    profile_revision: revision,
    effective,
    overrides: { ...state.overrides },
    delivery_mode: settings.ttsDeliveryMode === "native_incremental_pcm"
      ? "native_incremental_pcm"
      : "buffered_phrase",
  };
}

/**
 * Return the complete immutable audio.cpp tuning snapshot that the WebSocket
 * router requires for each response.  A profile list is not sufficient: the
 * selected profile must have been resolved by the candidate and carry its
 * concrete revision and effective fields.  This keeps an unavailable/stale
 * browser resolve from becoming a late TTS failure after Gemma has answered.
 */
function requiredCandidateTtsTuning(backend = settings.ttsBackend) {
  if (normalizeTtsProvider(backend) !== AUDIO_CPP_PROVIDER) return null;
  const tuning = activeTtsTuning(backend);
  if (!tuning) {
    throw new Error(
      "audio.cpp needs a complete resolved tuning profile before a conversation can start. Refresh the selected TTS backend and try again.",
    );
  }
  return tuning;
}

/** Browser-only playback snapshot paired with a pipeline config update. The
 * backend acknowledgement remains the commit point; this object is never sent
 * over the public Realtime WebSocket schema. */
function activePlaybackConfig(backend = settings.ttsBackend) {
  const provider = normalizeTtsProvider(backend);
  if (provider !== AUDIO_CPP_PROVIDER) {
    return {
      provider,
      nativeStreaming: false,
      profileId: "",
      profileRevision: null,
      resolvedPrimeMs: 0,
      // Faster, Groxaxo, and the buffered provider path retain their
      // immediate zero-reservoir contract. Cadence learning is an explicit
      // audio.cpp-native experiment and must not introduce re-prime pauses in
      // another backend.
      continuityMode: "fast-start",
    };
  }
  const state = realtimeTuningState(provider);
  const profile = candidateTuningResolved?.profile || {};
  const profileRevision = Number(profile.revision ?? candidateTuningResolved?.profile_revision);
  const effective = effectiveCandidateTuning();
  const firstBlockFrames = Number(effective.first_block_frames);
  const resolvedPrimeMs = Number.isFinite(firstBlockFrames)
    ? firstBlockFrames * 80
    : 0;
  const backendStatus = ttsBackendStatuses?.[provider];
  return {
    provider,
    profileId: state.profile_id,
    profileRevision: Number.isSafeInteger(profileRevision) && profileRevision > 0
      ? profileRevision
      : null,
    // Both engine capability and the direct chunk/header probe must agree.
    nativeStreaming: settings.ttsDeliveryMode === "native_incremental_pcm"
      && backendStatus?.nativeStreaming === true
      && candidateTuningResolved?.capabilities?.native_incremental_pcm === true,
    resolvedPrimeMs,
    continuityMode: settings.playbackContinuity === "fast-start" ? "fast-start" : "adaptive",
  };
}

function updateRealtimeAudioSummary() {
  if (!diagnosticsAudioSummary) return;
  const backend = settings.ttsBackend || "no provider";
  const backendStatus = ttsBackendStatuses?.[backend];
  const provider = backendStatus?.displayName || backend;
  const state = realtimeTuningState(backend);
  const profile = backend === AUDIO_CPP_PROVIDER
    ? candidateTuningResolved?.profile?.name || state.profile_id || "Balanced"
    : "provider defaults";
  const latestFirst = [...pipelineMetrics].reverse().find((metric) => metric.stage === "tts" && metric.status === "first_audio");
  const latestTts = [...pipelineMetrics].reverse().find((metric) => metric.stage === "tts" && ["request_start", "done"].includes(metric.status));
  const mode = latestTts?.detail?.mode || latestTts?.detail?.streaming_mode || (
    backend === AUDIO_CPP_PROVIDER
      ? settings.ttsDeliveryMode === "native_incremental_pcm" && backendStatus?.nativeStreaming
        ? "native PCM (experimental)"
        : "buffered phrase (stable default)"
      : "provider managed"
  );
  const firstPcm = latestFirst?.detail?.first_pcm_ms ?? latestFirst?.elapsed_ms;
  diagnosticsAudioSummary.textContent = `${provider} · ${profile} · ${mode} · first PCM ${Number.isFinite(Number(firstPcm)) ? `${Math.round(Number(firstPcm))} ms` : "pending"}`;
}

function setCandidateTuningEnabled(enabled) {
  diagnosticsTtsProfile.disabled = !enabled;
  diagnosticsTtsProfileApply.disabled = !enabled;
  diagnosticsTtsProfileName.disabled = !enabled;
  diagnosticsTtsProfileSaveAs.disabled = !enabled;
  if (diagnosticsPlaybackContinuity) {
    diagnosticsPlaybackContinuity.disabled = !enabled;
    diagnosticsPlaybackContinuity.value = settings.playbackContinuity === "fast-start" ? "fast-start" : "adaptive";
  }
  if (diagnosticsTtsDeliveryMode) {
    diagnosticsTtsDeliveryMode.disabled = !enabled;
    diagnosticsTtsDeliveryMode.value = settings.ttsDeliveryMode === "native_incremental_pcm"
      ? "native_incremental_pcm"
      : "buffered_phrase";
  }
  diagnosticsTuningApply.disabled = !enabled;
  diagnosticsTuningClear.disabled = !enabled;
  if (!enabled) {
    diagnosticsTuningInputs.forEach((input) => { input.disabled = true; });
    candidateInactiveTuningFields = new Set();
  }
  syncDecoderContextUnlock();
}

function tuningOverridesFromControls() {
  const overrides = {};
  for (const input of diagnosticsTuningInputs) {
    if (input.disabled) continue;
    const value = input.value.trim();
    if (value === "") continue;
    overrides[input.dataset.tuningKey] = input.dataset.tuningType === "string" ? value : Number(value);
  }
  return overrides;
}

async function resolveCandidateTuning(
  overrides = realtimeTuningState(AUDIO_CPP_PROVIDER).overrides,
  profileId = diagnosticsTtsProfile?.value || realtimeTuningState(AUDIO_CPP_PROVIDER).profile_id,
) {
  const response = await fetch("/api/audio-cpp/tuning/resolve", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(candidateRestTuningPayload(overrides, null, profileId)),
    cache: "no-store",
  });
  const resolved = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = typeof resolved.detail === "string" ? resolved.detail : JSON.stringify(resolved.detail || resolved);
    throw new Error(detail || "Candidate tuning could not be resolved");
  }
  candidateTuningResolved = resolved;
  setRealtimeTuningState(AUDIO_CPP_PROVIDER, profileId, overrides);
  const effective = effectiveCandidateTuning(resolved);
  const inactive = new Set(resolved.inactiveFields || []);
  candidateInactiveTuningFields = inactive;
  for (const input of diagnosticsTuningInputs) {
    const key = input.dataset.tuningKey;
    input.value = effective[key] ?? "";
    input.disabled = inactive.has(key);
    input.dataset.namedValue = String(resolved.profile?.[key] ?? "");
    input.dataset.effectiveValue = String(effective[key] ?? "");
    input.classList.toggle("has-temporary-override", Object.hasOwn(resolved.temporaryOverrides || {}, key));
    input.title = inactive.has(key)
      ? `${key} is inactive in the running candidate`
      : `Named: ${resolved.profile?.[key] ?? "unset"}; effective: ${effective[key] ?? "unset"}`;
  }
  syncDecoderContextUnlock();
  const firstMs = Number(effective.first_block_frames || 0) * 80;
  const steadyMs = Number(effective.steady_block_frames || 0) * 80;
  const native = !!resolved.capabilities?.native_incremental_pcm;
  const selectedMode = settings.ttsDeliveryMode === "native_incremental_pcm"
    ? native ? "native incremental PCM (experimental)" : "native PCM requested but unavailable"
    : "buffered phrase (stable default)";
  diagnosticsTuningStatus.textContent = `${resolved.profile?.name || resolved.profile?.id || "Profile"} · ${selectedMode} · first/steady ${firstMs}/${steadyMs} ms · audio.cpp PCM16 / 24 kHz server-acknowledged transport.`;
  diagnosticsTuningNamedValues.textContent = `${resolved.profile?.name || profileId} · ${compactTuningSummary(resolved.profile || {})}`;
  diagnosticsTuningOverrideValues.textContent = compactTuningSummary(resolved.temporaryOverrides || {});
  diagnosticsTuningEffectiveValues.textContent = compactTuningSummary(effective);
  diagnosticsTuningWarnings.textContent = (resolved.warnings || []).join(" ");
  updateRealtimeAudioSummary();
  return resolved;
}

function applyCandidateTuningToLiveSession() {
  const tuning = activeTtsTuning();
  if (client && LIVE_STATES.has(currentState) && !tuning) {
    diagnosticsTuningStatus.textContent = "Realtime tuning update withheld: the selected profile is not fully resolved. Refresh it before the next response.";
    return false;
  }
  if (client && LIVE_STATES.has(currentState) && tuning) {
    client.updateLocalPipeline(
      {
        tts_backend: AUDIO_CPP_PROVIDER,
        tts_tuning: tuning,
      },
      activePlaybackConfig(AUDIO_CPP_PROVIDER),
    );
    settings.playbackContinuity = settings.playbackContinuity === "fast-start" ? "fast-start" : "adaptive";
  }
  return true;
}

async function refreshCandidateTuningProfiles() {
  if (settings.ttsBackend !== AUDIO_CPP_PROVIDER) {
    diagnosticsTtsProfile.replaceChildren();
    candidateTuningResolved = null;
    candidateTuningProfileDocument = null;
    setCandidateTuningEnabled(false);
    diagnosticsTuningStatus.textContent = "The selected provider manages its own synthesis settings; audio.cpp controls are not sent.";
    diagnosticsTuningNamedValues.textContent = "not applicable";
    diagnosticsTuningOverrideValues.textContent = "none";
    diagnosticsTuningEffectiveValues.textContent = "provider managed";
    updateRealtimeAudioSummary();
    return;
  }
  const response = await fetch("/api/audio-cpp/tuning/profiles", { cache: "no-store" });
  if (!response.ok) throw new Error("Candidate tuning profiles are unavailable");
  const profileDocument = await response.json();
  candidateTuningProfileDocument = profileDocument;
  diagnosticsTtsProfile.replaceChildren(...Object.values(profileDocument.profiles || {}).map((profile) => {
    const option = window.document.createElement("option"); option.value = profile.id; option.textContent = `${profile.name} (${profile.first_block_frames * 80}/${profile.steady_block_frames * 80} ms)`; return option;
  }));
  const profileIds = new Set(Object.keys(profileDocument.profiles || {}));
  const current = realtimeTuningState(AUDIO_CPP_PROVIDER);
  const canonicalSelected = profileDocument.selections?.realtime?.[AUDIO_CPP_PROVIDER];
  const selectedId = profileIds.has(canonicalSelected)
    ? canonicalSelected
    : profileIds.has(current.profile_id)
      ? current.profile_id
      : profileIds.has("balanced") ? "balanced" : [...profileIds][0] || "";
  diagnosticsTtsProfile.value = selectedId;
  const overrides = current.profile_id === selectedId ? current.overrides : {};
  setRealtimeTuningState(AUDIO_CPP_PROVIDER, selectedId, overrides);
  settings.ttsProfileByBackend = { ...(settings.ttsProfileByBackend || {}), [AUDIO_CPP_PROVIDER]: selectedId };
  setCandidateTuningEnabled(true);
  await resolveCandidateTuning(overrides, selectedId);
}

diagnosticsTtsProfileApply?.addEventListener("click", async () => {
  try {
    const profileId = diagnosticsTtsProfile.value;
    setRealtimeTuningState(AUDIO_CPP_PROVIDER, profileId, {});
    settings.ttsProfileByBackend = { ...(settings.ttsProfileByBackend || {}), [AUDIO_CPP_PROVIDER]: profileId };
    await resolveCandidateTuning({}, profileId);
    const selected = await fetch("/api/audio-cpp/tuning/selection", {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ provider: AUDIO_CPP_PROVIDER, scope: "realtime", profile_id: profileId }),
    });
    if (!selected.ok) {
      const detail = await selected.json().catch(() => ({}));
      throw new Error(detail.detail || "Candidate Realtime profile selection could not be saved");
    }
    const saved = await saveSettings(settings);
    if (!saved.ok) throw new Error(saved.error || "Realtime profile selection could not be saved");
    applyCandidateTuningToLiveSession();
    diagnosticsTuningStatus.textContent += " Saved for Realtime only and applies from the next assistant response; an active answer keeps its frozen profile. Voice Studio keeps its own active selection.";
  } catch (error) {
    diagnosticsTuningStatus.textContent = `Profile apply failed: ${error instanceof Error ? error.message : String(error)}`;
  }
});

diagnosticsTtsProfileSaveAs?.addEventListener("click", async () => {
  try {
    const name = diagnosticsTtsProfileName.value.trim();
    const id = tuningProfileSlug(name);
    if (!name || !id) throw new Error("Enter a profile name containing letters or numbers");
    if (!candidateTuningResolved) await resolveCandidateTuning();
    const effective = effectiveCandidateTuning();
    const values = Object.fromEntries(
      SAFE_TUNING_KEYS
        .filter((key) => effective[key] !== undefined)
        .map((key) => [key, effective[key]]),
    );
    const response = await fetch("/api/audio-cpp/tuning/profiles", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        id,
        name,
        clone_from: diagnosticsTtsProfile.value,
        values,
        scope: "realtime",
        select: true,
      }),
    });
    const created = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = typeof created.detail === "string" ? created.detail : JSON.stringify(created.detail || created);
      throw new Error(detail || "Profile could not be created");
    }
    const createdId = created.profile?.id || created.created?.id || id;
    setRealtimeTuningState(AUDIO_CPP_PROVIDER, createdId, {});
    settings.ttsProfileByBackend = { ...(settings.ttsProfileByBackend || {}), [AUDIO_CPP_PROVIDER]: createdId };
    const saved = await saveSettings(settings);
    if (!saved.ok) throw new Error(saved.error || "Created profile selection could not be saved");
    diagnosticsTtsProfileName.value = "";
    await refreshCandidateTuningProfiles();
    applyCandidateTuningToLiveSession();
    diagnosticsTuningStatus.textContent += " Saved as a new Realtime profile; Voice Studio selection remains unchanged.";
  } catch (error) {
    diagnosticsTuningStatus.textContent = `Save As failed: ${error instanceof Error ? error.message : String(error)}`;
  }
});

diagnosticsTtsProfile?.addEventListener("change", () => {
  const profileId = diagnosticsTtsProfile.value;
  setRealtimeTuningState(AUDIO_CPP_PROVIDER, profileId, {});
  void resolveCandidateTuning({}, profileId).catch((error) => {
    diagnosticsTuningStatus.textContent = `Profile resolve failed: ${error instanceof Error ? error.message : String(error)}`;
  });
});

diagnosticsPlaybackContinuity?.addEventListener("change", async () => {
  settings.playbackContinuity = diagnosticsPlaybackContinuity.value === "fast-start" ? "fast-start" : "adaptive";
  const saved = await saveSettings(settings);
  diagnosticsTuningStatus.textContent = saved.ok
    ? "Playback continuity saved. It applies from the next assistant response; current audio keeps its existing queue."
    : `Playback continuity is browser-only until settings save succeeds: ${saved.error}`;
  applyCandidateTuningToLiveSession();
});

diagnosticsTtsDeliveryMode?.addEventListener("change", async () => {
  settings.ttsDeliveryMode = diagnosticsTtsDeliveryMode.value === "native_incremental_pcm"
    ? "native_incremental_pcm"
    : "buffered_phrase";
  await saveSettings(settings);
  applyCandidateTuningToLiveSession();
  updateRealtimeAudioSummary();
});

for (const control of [historyCompactionEnabled, historyCompactionTrigger, historyCompactionTarget, historyCompactionRecentTurns]) {
  control?.addEventListener("change", applyHistoryCompactionSettings);
}

diagnosticsTuningContextUnlock?.addEventListener("change", syncDecoderContextUnlock);

diagnosticsTuningApply?.addEventListener("click", async () => {
  try {
    const overrides = tuningOverridesFromControls();
    const profileId = diagnosticsTtsProfile.value;
    await resolveCandidateTuning(overrides, profileId);
    setRealtimeTuningState(AUDIO_CPP_PROVIDER, profileId, overrides);
    applyCandidateTuningToLiveSession();
    diagnosticsTuningStatus.textContent += " Temporary overrides are active for this page session only.";
  } catch (error) {
    diagnosticsTuningStatus.textContent = `Override rejected: ${error instanceof Error ? error.message : String(error)}`;
  }
});

diagnosticsTuningClear?.addEventListener("click", async () => {
  try {
    const profileId = diagnosticsTtsProfile.value;
    setRealtimeTuningState(AUDIO_CPP_PROVIDER, profileId, {});
    await resolveCandidateTuning({}, profileId);
    applyCandidateTuningToLiveSession();
    diagnosticsTuningStatus.textContent += " Temporary overrides cleared.";
  } catch (error) {
    diagnosticsTuningStatus.textContent = `Could not clear overrides: ${error instanceof Error ? error.message : String(error)}`;
  }
});

function echoCalibrationFromControls() {
  const calibration = {};
  for (const input of diagnosticsEchoInputs) {
    const number = Number(input.value);
    if (Number.isFinite(number)) calibration[input.dataset.echoCalibrationKey] = number;
  }
  return {
    delayMs: Math.max(0, Math.min(500, Number(calibration.delayMs) || 0)),
    suppressionStrength: Math.max(0, Math.min(1, Number(calibration.suppressionStrength) || 0)),
    leakageThreshold: Math.max(0.05, Math.min(1, Number(calibration.leakageThreshold) || 0.65)),
    doubleTalkSensitivity: Math.max(0, Math.min(1, Number(calibration.doubleTalkSensitivity) || 0)),
    echoTailMs: Number(latestEchoStatus?.calibration?.echoTailMs) || 350,
  };
}

function paintEchoStatus(status) {
  if (!status || typeof status !== "object") return;
  latestEchoStatus = status;
  const calibration = status.calibration || {};
  for (const input of diagnosticsEchoInputs) {
    const value = calibration[input.dataset.echoCalibrationKey];
    if (Number.isFinite(Number(value))) input.value = String(value);
  }
  diagnosticsEchoDevicePair.textContent = status.devicePair || "device pair pending";
  diagnosticsEchoOutputLatency.textContent =
    `Output latency: ${Number(status.outputLatencyMs || 0).toFixed(1)} ms`;
  const requested = status.requestedMode || settings.echoGuard;
  const effective = status.effectiveMode || (requested === "strict" ? "strict-fallback" : "native");
  const engine = status.available
    ? status.engine || "AEC3"
    : status.error || status.loaderReason || "Native browser AEC fallback";
  diagnosticsAudioStatus.textContent =
    `Requested/effective echo: ${requested}/${effective} · reference ${status.referenceWired === false ? "missing" : "wired"} · ${engine}.`;
  setDiagnosticWarning(
    "aec3",
    requested === "adaptive" && !status.available
      ? "Adaptive requested, but the verified AEC3 module is unavailable; Native browser AEC is active."
      : "",
  );
}

async function saveActiveEchoCalibration({ useMeasuredDelay = false } = {}) {
  const pair = latestEchoStatus?.devicePair;
  if (!pair) {
    diagnosticsAudioStatus.textContent = "Start a conversation before saving device-pair calibration.";
    return;
  }
  if (useMeasuredDelay) {
    const echo = [...pipelineMetrics].reverse().find((metric) => metric.stage === "echo_guard");
    const measured = Number(echo?.detail?.lag_ms);
    const outputLatency = Number(latestEchoStatus?.outputLatencyMs || 0);
    if (!Number.isFinite(measured)) {
      diagnosticsAudioStatus.textContent = "No stable AEC3 delay measurement is available yet.";
      return;
    }
    const delayInput = diagnosticsEchoInputs.find(
      (input) => input.dataset.echoCalibrationKey === "delayMs",
    );
    if (delayInput) delayInput.value = String(Math.max(0, Math.round(measured - outputLatency)));
  }
  const calibration = echoCalibrationFromControls();
  settings.echoCalibrations = { ...(settings.echoCalibrations || {}), [pair]: calibration };
  client?.setEchoCalibration(calibration);
  const saved = await saveSettings(settings);
  if (!saved.ok) {
    diagnosticsAudioStatus.textContent = `Calibration save failed: ${saved.error}`;
    return;
  }
  paintEchoStatus({ ...latestEchoStatus, calibration });
}

diagnosticsEchoSave?.addEventListener("click", () => {
  void saveActiveEchoCalibration();
});
diagnosticsEchoUseMeasured?.addEventListener("click", () => {
  void saveActiveEchoCalibration({ useMeasuredDelay: true });
});
/** @type {HTMLButtonElement} */
const toolsBtn = $("#tools-btn");
/** @type {HTMLDialogElement} */
const toolsModal = $("#tools-modal");
/** @type {HTMLButtonElement} */
const toolsClose = $("#tools-close");
/** @type {HTMLInputElement} */
const toolWebSwitch = $("#tool-web");
/** @type {HTMLInputElement} */
const toolCamSwitch = $("#tool-cam");
/** @type {HTMLElement} */
const toolWebRow = $("#tool-web-row");
/** @type {HTMLElement} */
const toolWebHint = $("#tool-web-hint");
/** @type {HTMLElement} */
const toolCamHint = $("#tool-cam-hint");
/** @type {HTMLInputElement} */
const searchKeyInput = $("#search-key");
/** @type {HTMLElement} */
const camPip = $("#cam-pip");
/** @type {HTMLVideoElement} */
const camVideo = $("#cam-video");

/** @type {HTMLInputElement} */
const inputLbUrl = $("#lb-url");
/** @type {HTMLElement} */
const connField = $("#conn-field");
/** @type {HTMLElement} */
const connHint = $("#conn-hint");
/** @type {HTMLSelectElement} */
const inputVoice = $("#voice");
const profileLibraryStatus = $("#profile-library-status");
const inputProfileName = $("#profile-name");
const inputProfileRefText = $("#profile-ref-text");
const inputProfileLanguage = $("#profile-language");
const inputProfileAudio = $("#profile-audio");
const profileCreateBtn = $("#profile-create");
const profileSaveBtn = $("#profile-save");
const profileDeleteBtn = $("#profile-delete");
const profileImportBtn = $("#profile-import");
const modelInventoryStatus = $("#model-inventory-status");
const fasterModelLoadBtn = $("#faster-model-load");
const fasterModelSwitchBtn = $("#faster-model-switch");
const fasterModelUnloadBtn = $("#faster-model-unload");
const validateAudioCppBtn = $("#validate-audio-cpp");
const audioCppStatus = $("#audio-cpp-status");
const ttsStreamingStatus = $("#tts-streaming-status");
/** @type {HTMLSelectElement} */
const inputTtsBackend = $("#tts-backend");
/** @type {HTMLSelectElement} */
const inputModelProvider = $("#model-provider");
/** @type {HTMLInputElement} */
const inputModelUrl = $("#model-url");
/** @type {HTMLInputElement} */
const inputModelName = $("#model-name");
/** @type {HTMLInputElement} */
const inputModelApiKey = $("#model-api-key");
/** @type {HTMLElement} */
const remoteModelFields = $("#remote-model-fields");
/** @type {HTMLButtonElement} */
const testModelConnectionBtn = $("#test-model-connection");
/** @type {HTMLElement} */
const modelConnectionStatus = $("#model-connection-status");
/** @type {HTMLTextAreaElement} */
const inputInstructions = $("#instructions");
/** @type {HTMLInputElement} */
const inputNoiseGate = $("#noise-gate");
/** @type {HTMLSelectElement} */
const inputEchoGuard = $("#echo-guard");
/** @type {HTMLInputElement} */
const inputFullBufferTts = $("#full-buffer-tts");
/** @type {HTMLInputElement} */
const inputLiveTranscript = $("#live-transcript");
/** @type {HTMLInputElement} */
const inputMaxResponseTokens = $("#max-response-tokens");
/** @type {HTMLElement} */
const gateValue = $("#gate-value");
/** @type {HTMLElement} */
const gateMeterFill = $("#gate-meter-fill");
/** @type {HTMLElement} */
const micGate = $("#mic-gate");
const mgaArc = /** @type {SVGSVGElement} */ (document.querySelector("#mic-gate-arc"));
const mgaTrack = /** @type {SVGPathElement} */ (document.querySelector("#mga-track"));
const mgaFill = /** @type {SVGPathElement} */ (document.querySelector("#mga-fill"));
const mgaHit = /** @type {SVGPathElement} */ (document.querySelector("#mga-hit"));
const mgaHandle = /** @type {SVGCircleElement} */ (document.querySelector("#mga-handle"));
/** @type {HTMLButtonElement} */
const restartBtn = $("#restart-conversation");
/** @type {HTMLElement} */
const restartHint = $("#restart-hint");
const settingsForm = /** @type {HTMLFormElement} */ (settingsModal.querySelector("form"));
const settingsSaveStatus = $("#settings-save-status");

/** @type {AppState} */
let currentState = "idle";
let settings = loadSettings();
/** @type {Array<{ id: string; voice: string; name: string; created_at?: string }>} */
let voiceProfiles = [];
let defaultVoice = DEFAULT_VOICE;
let fasterModelInventory = null;
let profileLibraryWritable = false;
let localPipeline = null;
/** @type {Record<string, any>} */
let ttsBackendStatuses = {};
let localIdentityRequest = 0;
let localIdentityRetryTimer = 0;
// Startup status is a probe, not a background service manager.  Retry a
// transient failure a small, bounded number of times; a focus, backend switch,
// or explicit refresh starts a fresh probe budget.
const MAX_LOCAL_IDENTITY_RETRIES = 2;
let localIdentityRetryAttempts = 0;
// A backend switch can finish before an older inventory response.  Only the
// most recently requested backend is allowed to change the voice picker.
let voiceInventoryRequest = 0;
let diagnosticsOpen = localStorage.getItem(STORAGE_KEYS.diagnostics) === "1";
/** @type {Array<any>} */
let pipelineMetrics = [];
const EXPECTED_UI_API_VERSION = 21;
const EXPECTED_BACKEND_API_VERSION = 7;
const DIAGNOSTIC_STAGES = ["mic", "echo_guard", "vad", "transcription", "gemma", "context", "tool", "tts", "playback"];
const DIAGNOSTIC_STAGE_LABELS = { echo_guard: "Echo Guard" };
const diagnosticWarnings = new Map();
let backendRuntime = null;
let backendMetricTimer = 0;
let backendRuntimeTimer = 0;

// ── Connection target ────────────────────────────────────────────────────────
// Two modes, decided by the deploy via /api/config:
//   • LOAD_BALANCER_URL set  -> original flow: POST the same-origin /api/session
//     proxy (the server forwards to the LB; the LB address is never sent here).
//   • unset (allowDirect)    -> the user sets a speech-to-speech server URL and
//     the browser connects to it directly (no load balancer, no /session).
let lbMode = false;
// Fail open: direct entry is allowed unless /api/config reports an LB URL. This
// way a missing/unreachable config (e.g. static hosting) leaves the field
// usable rather than locked.
let allowDirect = true;

// ── Tool state ──────────────────────────────────────────────────────────────
let toolsEnabled = loadTools();
// Whether the server holds a Serper key (learned from /api/config on load).
let serverSearchKey = false;
// A user-supplied key (fallback when the deploy has none). localStorage only.
let userSearchKey = localStorage.getItem(STORAGE_KEYS.searchKey) || "";
/** @type {MediaStream | null} */
let cameraStream = null;

/** Search is usable if the server has a key or the user supplied one. */
function searchAvailable() {
  return serverSearchKey || !!userSearchKey;
}

/** Tool definitions for the currently-enabled (and usable) tools. */
function activeToolDefs() {
  const defs = [];
  if (toolsEnabled.web_search && searchAvailable()) defs.push(TOOL_DEFS.web_search);
  if (toolsEnabled.camera_snapshot) defs.push(TOOL_DEFS.camera_snapshot);
  return defs;
}

/** Instructions plus the hidden tool-use hint when any tool is active. */
function effectiveInstructions() {
  const base = settings.instructions;
  return activeToolDefs().length ? base + TOOL_USE_HINT : base;
}

/** Push the active tool set to a live session so toggles apply mid-call. */
function pushToolsToSession() {
  if (!client || !LIVE_STATES.has(currentState)) return;
  client.setTools(activeToolDefs());
  // The hidden tool-use hint depends on whether any tool is active, so refresh
  // instructions alongside the tool set.
  client.updateSession({ instructions: effectiveInstructions() });
}

// ── Chat view ───────────────────────────────────────────────────────────────
// Owns the history panel, the ephemeral bubbles, and all transcript/tool
// streaming state. The client's events are forwarded to its on* methods.
const chat = new ChatView();

// ── Account / limiter ─────────────────────────────────────────────────────
// Login chip + daily-limit modal (inert unless the deploy is in LB mode). The
// server meters conversation time; the client just heartbeats a live session
// and tears down when the server reports the budget is spent.
const account = new Account();
let limiterOn = false;
let heartbeatTimer = 0;
let trackedSessionId = "";
let trackedTier = "";
// The waiting-queue ticket id while we're in line (else ""). Used to leave the
// queue on teardown / tab-close so we don't hold a phantom place.
let queuedTicketId = "";

/** @type {S2sWsRealtimeClient | null} */
let client = null;
const startAttempts = new StartAttemptController();
/** @type {MediaStream | null} */
let micStream = null;
let micMuted = false;

/** @param {AppState} next */
function setState(next) {
  currentState = next;
  const view = STATE_VIEWS[next];
  circleBtn.disabled = view.disabled;
  circleBtn.className = `circle ${STATE_CLASS[next]}`;
  if (next !== "error") setCaption(view.caption);

  const live = LIVE_STATES.has(next);
  orbWrap.classList.toggle("live", live);
  micBtn.setAttribute("aria-hidden", live ? "false" : "true");
  stopBtn.setAttribute("aria-hidden", live ? "false" : "true");
  micBtn.tabIndex = live ? 0 : -1;
  stopBtn.tabIndex = live ? 0 : -1;

  // Queue affordances: "Leave queue" whenever we're in line; "Join now" only once
  // it's our turn (a slot is held for us). Both live under #queue-actions.
  const yourTurn = next === "your-turn";
  const inLine = next === "queued" || yourTurn;
  queueActions.hidden = !inLine;
  joinQueueBtn.hidden = !yourTurn;
  joinQueueBtn.tabIndex = yourTurn ? 0 : -1;
  leaveQueueBtn.hidden = !inLine;
  leaveQueueBtn.tabIndex = inLine ? 0 : -1;
  if (!yourTurn) stopJoinCountdown();

  // Warm reassurance under the terse position, only while waiting in line.
  if (next === "queued") {
    circleSubcaption.textContent =
      "Sorry, we overhugged! 🤗 Every slot is busy, so we saved you a spot. Hang tight, you're moving up.";
    circleSubcaption.hidden = false;
  } else {
    circleSubcaption.hidden = true;
  }

  updateRestartAvailability();
}

function updateRestartAvailability() {
  // Restart works from any settled state — it tears down a live call (if any)
  // and reconnects with the current settings. Only block while mid-connect or
  // while waiting in the queue (restarting from there would just re-queue).
  restartBtn.disabled =
    currentState === "connecting" || currentState === "queued" || currentState === "your-turn";
  restartHint.hidden = false;
  restartHint.textContent = LIVE_STATES.has(currentState)
    ? "Reconnects now with the settings above."
    : "Starts a conversation with the settings above.";
}

/**
 * @param {string} text
 * @param {"" | "error" | "muted"} [kind]
 */
function setCaption(text, kind = "") {
  const trimmed = text.trim();
  circleCaption.textContent = trimmed;
  circleCaption.className = `circle-caption${kind ? ` ${kind}` : ""}${trimmed ? "" : " empty"}`;
}

function openSettings() {
  syncConnectionUi();
  renderTtsBackendOptions();
  inputTtsBackend.value = settings.ttsBackend;
  // Always bind the selector to a fresh inventory for the currently selected
  // backend. In particular, do not reuse the last provider's in-memory list
  // when Settings opens after server-managed preferences were restored.
  void (async () => {
    await Promise.all([fetchVoiceProfiles(), refreshTtsBackends()]);
    await refreshFasterModelInventory();
    renderSelectedTtsBackendStatus();
  })();
  inputModelProvider.value = settings.modelProvider;
  inputModelUrl.value = settings.modelUrl;
  inputModelName.value = settings.modelName;
  inputModelApiKey.value = settings.modelApiKey;
  syncModelProviderUi();
  inputInstructions.value = settings.instructions;
  inputEchoGuard.value = settings.echoGuard;
  void refreshCandidateTuningProfiles().catch((error) => console.warn("candidate tuning refresh failed", error));
  inputFullBufferTts.checked = settings.fullBufferTts;
  inputLiveTranscript.checked = settings.liveTranscript;
  inputMaxResponseTokens.value = String(settings.maxResponseTokens);
  syncGateUi();
  updateRestartAvailability();
  settingsModal.showModal();
}

function renderVoiceOptions() {
  inputVoice.replaceChildren();
  if (!voiceProfiles.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No live Base clone profiles found for this backend";
    option.disabled = true;
    option.selected = true;
    inputVoice.append(option);
    inputVoice.disabled = true;
    return;
  }

  inputVoice.disabled = false;
  const voices = new Set(voiceProfiles.map((profile) => profile.voice));
  if (!voices.has(settings.voice)) {
    settings.voice = voices.has(defaultVoice) ? defaultVoice : voiceProfiles[0].voice;
    settings.voiceByBackend = { ...(settings.voiceByBackend || {}), [settings.ttsBackend]: settings.voice };
    saveSettings(settings);
    profileLibraryStatus.textContent = `Selected ${voiceProfiles.find((item) => item.voice === settings.voice)?.name || settings.voice} because the saved clone is unavailable on this backend.`;
  }

  for (const profile of voiceProfiles) {
    const option = document.createElement("option");
    option.value = profile.voice;
    option.textContent = `${profile.name} (${profile.voice})`;
    option.selected = profile.voice === settings.voice;
    inputVoice.append(option);
  }
  syncSelectedProfileEditor();
}

function clearVoiceProfileOptions(backend, message = `Loading live Base clone profiles for ${backend}…`) {
  voiceProfiles = [];
  profileLibraryWritable = false;
  defaultVoice = "";
  inputVoice.replaceChildren();
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = message;
  placeholder.disabled = true;
  placeholder.selected = true;
  inputVoice.append(placeholder);
  inputVoice.disabled = true;
  syncSelectedProfileEditor();
}

function selectedProfile() {
  const voice = inputVoice.value || settings.voice;
  return voiceProfiles.find((profile) => profile.voice === voice) || null;
}

function syncSelectedProfileEditor() {
  const profile = selectedProfile();
  profileLibraryStatus.textContent = profileLibraryWritable
    ? `Changes persist only to the selected ${settings.ttsBackend} clone library.`
    : "The selected backend is inventory-only here; use its own Voice Studio for profile changes.";
  for (const button of [profileCreateBtn, profileSaveBtn, profileDeleteBtn, profileImportBtn]) {
    button.disabled = !profileLibraryWritable;
  }
  if (!profile) return;
  inputProfileName.value = profile.name || "";
  inputProfileRefText.value = profile.ref_text || "";
  inputProfileLanguage.value = profile.language || "Auto";
  profileDeleteBtn.disabled = !profileLibraryWritable || profile.id === DEFAULT_VOICE.replace("clone:", "");
}

async function qwen3Json(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
  return payload;
}

function applyVoiceProfilePayload(payload, backend = settings.ttsBackend) {
  if (backend !== settings.ttsBackend || (payload.backend && payload.backend !== backend)) return;
  defaultVoice = payload.defaultVoice || DEFAULT_VOICE;
  voiceProfiles = Array.isArray(payload.voices) ? payload.voices : [];
  profileLibraryWritable = !!payload.writable;
  const liveVoices = new Set(voiceProfiles.map((profile) => profile.voice));
  const saved = settings.voiceByBackend?.[backend] || settings.voice;
  const providerSelected = payload.selectedVoice;
  if (liveVoices.size) {
    settings.voice = liveVoices.has(saved)
      ? saved
      : (liveVoices.has(providerSelected) ? providerSelected : (liveVoices.has(defaultVoice) ? defaultVoice : voiceProfiles[0].voice));
    settings.voiceByBackend = { ...(settings.voiceByBackend || {}), [backend]: settings.voice };
  }
  renderVoiceOptions();
}

async function selectVoiceProfile() {
  const profile = selectedProfile();
  if (!profile) return;
  settings.voice = profile.voice;
  settings.voiceByBackend = { ...(settings.voiceByBackend || {}), [settings.ttsBackend]: profile.voice };
  const saved = await saveSettings(settings);
  if (!saved.ok) {
    profileLibraryStatus.textContent = `Could not save selected voice: ${saved.error}`;
  }
  if (profileLibraryWritable) {
    try {
      const payload = await qwen3Json(`api/tts/backends/${encodeURIComponent(settings.ttsBackend)}/profiles/${encodeURIComponent(profile.id)}/select`, { method: "POST", body: JSON.stringify({}) });
      applyVoiceProfilePayload(payload);
    } catch (err) {
      profileLibraryStatus.textContent = `Could not persist selection: ${err instanceof Error ? err.message : String(err)}`;
    }
  }
  // Validation is scoped to provider + resident model + clone.  Refresh only
  // after the selected clone has reached server-backed settings; otherwise the
  // status endpoint necessarily reports the previous clone and blocks Start.
  await refreshTtsBackends();
  await refreshLocalIdentity();
  renderSelectedTtsBackendStatus(profile);
}

function renderSelectedTtsBackendStatus(profile = selectedProfile()) {
  const status = ttsBackendStatuses[settings.ttsBackend];
  const name = status?.displayName || settings.ttsBackend;
  if (!status?.reachable) {
    audioCppStatus.textContent = `${name} is unavailable. Start it externally, then refresh or reopen Settings.`;
    return;
  }
  if (status.explicitValidation) {
    const validation = status.validation;
    audioCppStatus.textContent = validation?.speech && validation.voice === profile?.voice
      ? `${profile.name || profile.voice} (${profile.voice}) is validated for the resident ${name} model.`
      : `Needs validation for ${profile?.name || profile?.voice || "the selected clone"}. Use Validate selected TTS backend before starting a conversation.`;
    return;
  }
  audioCppStatus.textContent = status.ready && profile
    ? `${name} is ready with ${profile.name || profile.voice} (${profile.voice}).`
    : `${name} is reachable but its selected Base clone is not ready.`;
}

/** dB position (clamped to the slider axis) as a 0..1 fraction of the track.
 * @param {number} db */
function dbToFraction(db) {
  const clamped = Math.min(GATE_MAX_DB, Math.max(GATE_OFF_DB, db));
  return (clamped - GATE_OFF_DB) / (GATE_MAX_DB - GATE_OFF_DB);
}

/** @param {number} f @returns {number} dB at a 0..1 position on the gate axis. */
function fractionToDb(f) {
  const clamped = Math.min(1, Math.max(0, f));
  return Math.round(GATE_OFF_DB + clamped * (GATE_MAX_DB - GATE_OFF_DB));
}

// ── Radial gate arc (around the mic button, live during a call) ─────────────
// A 270° arc with the gap facing the orb (right). Fraction 0 (=Off) sits at the
// bottom-ish start; 1 (=max) at the top-ish end. The level fill and the
// threshold handle ride this same axis, mirroring the Settings widget.
const ARC_R = 40;
// A ~200° arc centred on the left (180°) so the wide gap faces the orb (right).
const ARC_SPAN_DEG = 200;
const ARC_START_DEG = 180 - ARC_SPAN_DEG / 2; // lower-left start; Off end

/** Point at fraction f (0..1) and radius r, in the 0..100 viewBox.
 * @param {number} f @param {number} [r] */
function arcPoint(f, r = ARC_R) {
  const deg = ARC_START_DEG + f * ARC_SPAN_DEG;
  const rad = (deg * Math.PI) / 180;
  return { x: 50 + r * Math.cos(rad), y: 50 + r * Math.sin(rad) };
}

/** SVG path `d` for the full 0..1 arc (clockwise). */
function fullArcD() {
  const a = arcPoint(0);
  const b = arcPoint(1);
  const largeArc = ARC_SPAN_DEG > 180 ? 1 : 0;
  return `M ${a.x} ${a.y} A ${ARC_R} ${ARC_R} 0 ${largeArc} 1 ${b.x} ${b.y}`;
}

/** One-time geometry: track, fill (dash-revealed) and the transparent hit band. */
function initGateArc() {
  const d = fullArcD();
  mgaTrack.setAttribute("d", d);
  mgaFill.setAttribute("d", d);
  mgaHit.setAttribute("d", d);
  // pathLength 100 lets us reveal the fill by fraction via dashoffset.
  mgaFill.setAttribute("pathLength", "100");
  mgaFill.style.strokeDasharray = "100 100";
  mgaFill.style.strokeDashoffset = "100"; // empty until levels arrive
  renderGateHandle();
}

/** Place the threshold bead on the arc at the stored threshold; flag off state. */
function renderGateHandle() {
  const off = settings.noiseGate <= GATE_OFF_DB;
  const p = arcPoint(dbToFraction(settings.noiseGate));
  mgaHandle.setAttribute("cx", String(p.x));
  mgaHandle.setAttribute("cy", String(p.y));
  micGate.classList.toggle("gate-off", off);
}

/** Paint a 0..1 live level onto the arc fill (and the Settings meter if open).
 * Brightens the tick when the level crosses the threshold — i.e. the gate is
 * actually open — but only when gating is enabled.
 * @param {number} rms */
function paintInputLevel(rms) {
  const db = rms > 0 ? 20 * Math.log10(rms) : GATE_OFF_DB;
  const f = dbToFraction(db);
  mgaFill.style.strokeDashoffset = String(100 * (1 - f));
  if (settingsModal.open) gateMeterFill.style.width = `${f * 100}%`;
  const enabled = settings.noiseGate > GATE_OFF_DB;
  micGate.classList.toggle("gate-open", enabled && f >= dbToFraction(settings.noiseGate));
}

/** The single place that commits a new gate threshold: updates both controls,
 * persists, and applies live to the running session.
 * @param {number} db */
function setGateThreshold(db) {
  settings.noiseGate = Math.min(GATE_MAX_DB, Math.max(GATE_OFF_DB, Math.round(db)));
  const off = settings.noiseGate <= GATE_OFF_DB;
  inputNoiseGate.value = String(settings.noiseGate);
  gateValue.textContent = off ? "Off" : `${settings.noiseGate} dB`;
  renderGateHandle();
  localStorage.setItem(STORAGE_KEYS.noiseGate, String(settings.noiseGate));
  if (client && LIVE_STATES.has(currentState)) {
    client.setNoiseGate(gateParams(settings.noiseGate));
  }
}

/** Reflect the stored gate threshold into the slider, label and arc handle. */
function syncGateUi() {
  inputNoiseGate.value = String(settings.noiseGate);
  const off = settings.noiseGate <= GATE_OFF_DB;
  gateValue.textContent = off ? "Off" : `${settings.noiseGate} dB`;
  renderGateHandle();
}

// Drag along the arc band to set the threshold (a tap on the glyph still mutes).
let gateDragging = false;
/** @param {PointerEvent} e */
function gatePointerToDb(e) {
  const rect = mgaArc.getBoundingClientRect();
  const cx = rect.left + rect.width / 2;
  const cy = rect.top + rect.height / 2;
  let deg = (Math.atan2(e.clientY - cy, e.clientX - cx) * 180) / Math.PI;
  if (deg < 0) deg += 360;
  // Map the on-arc angle to a fraction; angles in the right-side gap fall
  // outside [0,1] and fractionToDb clamps them to the nearest end (just-below
  // start -> Off, just-past end -> max).
  const f = (deg - ARC_START_DEG) / ARC_SPAN_DEG;
  return fractionToDb(f);
}
mgaHit.addEventListener("pointerdown", (e) => {
  gateDragging = true;
  mgaHit.setPointerCapture(e.pointerId);
  setGateThreshold(gatePointerToDb(e));
});
mgaHit.addEventListener("pointermove", (e) => {
  if (gateDragging) setGateThreshold(gatePointerToDb(e));
});
const endGateDrag = (/** @type {PointerEvent} */ e) => {
  if (!gateDragging) return;
  gateDragging = false;
  try { mgaHit.releasePointerCapture(e.pointerId); } catch {}
};
mgaHit.addEventListener("pointerup", endGateDrag);
mgaHit.addEventListener("pointercancel", endGateDrag);

settingsBtn.addEventListener("click", openSettings);

diagnosticsBtn.addEventListener("click", () => setDiagnosticsOpen(!diagnosticsOpen));
diagnosticsClose.addEventListener("click", () => setDiagnosticsOpen(false));
setDiagnosticsOpen(diagnosticsOpen);
initDiagnosticsGeometry();

// About panel: native <dialog>, Esc closes for free; also close on the X and
// on a click in the backdrop (a click whose target is the dialog itself).
aboutBtn.addEventListener("click", () => { if (settings.modelProvider === "local") void refreshLocalPipeline(); aboutModal.showModal(); });
// Mobile twin of the (i), living in the right-hand control cluster.
$("#about-btn-m").addEventListener("click", () => { if (settings.modelProvider === "local") void refreshLocalPipeline(); aboutModal.showModal(); });
aboutClose.addEventListener("click", () => aboutModal.close());
aboutModal.addEventListener("click", (e) => {
  if (e.target === aboutModal) aboutModal.close();
});

// ── Tools panel ───────────────────────────────────────────────────────────

/** Reflect the current tool state into the panel controls. */
function syncToolsUi() {
  const avail = searchAvailable();
  toolWebSwitch.checked = toolsEnabled.web_search && avail;
  toolWebSwitch.disabled = !avail;
  toolWebRow.classList.toggle("disabled", !avail);
  toolCamSwitch.checked = toolsEnabled.camera_snapshot;

  if (serverSearchKey) {
    // Key lives server-side: show it as configured, never expose it.
    searchKeyInput.value = "";
    searchKeyInput.placeholder = "••••••••  · provided by the server";
    searchKeyInput.disabled = true;
    toolWebHint.textContent = "Ready. The search key is held server-side and never sent to your browser.";
  } else {
    searchKeyInput.disabled = false;
    searchKeyInput.value = userSearchKey;
    searchKeyInput.placeholder = "Paste a Serper key to enable web search";
    toolWebHint.textContent = userSearchKey
      ? "Using your key — stored in this browser only."
      : "No server key configured. Add your own Serper key to enable web search.";
  }
}

toolsBtn.addEventListener("click", () => { syncToolsUi(); toolsModal.showModal(); });
toolsClose.addEventListener("click", () => toolsModal.close());
toolsModal.addEventListener("click", (e) => {
  if (e.target === toolsModal) toolsModal.close();
});

toolWebSwitch.addEventListener("change", () => {
  if (toolWebSwitch.checked && !searchAvailable()) {
    toolWebSwitch.checked = false; // guard: can't enable without a key
    return;
  }
  toolsEnabled.web_search = toolWebSwitch.checked;
  saveTools();
  pushToolsToSession();
});

toolCamSwitch.addEventListener("change", async () => {
  if (toolCamSwitch.checked) {
    try {
      // Flipping the switch always re-requests the camera, so a permission that
      // was only dismissed earlier is asked again here.
      await enableCamera();
    } catch (err) {
      toolCamSwitch.checked = false;
      const denied = err instanceof Error && (err.name === "NotAllowedError" || err.name === "SecurityError");
      toolCamHint.textContent = denied
        ? "Camera blocked. Allow camera access from the browser address bar, then switch this on again."
        : `Camera unavailable${err instanceof Error ? `: ${err.message}` : ""}`;
      return;
    }
    toolsEnabled.camera_snapshot = true;
    toolCamHint.textContent = "Camera on. The assistant can take a snapshot when it needs to see.";
  } else {
    disableCamera();
    toolsEnabled.camera_snapshot = false;
    toolCamHint.textContent = "Let the assistant see through your webcam.";
  }
  saveTools();
  pushToolsToSession();
});

searchKeyInput.addEventListener("input", () => {
  if (serverSearchKey) return;
  userSearchKey = searchKeyInput.value.trim();
  if (userSearchKey) localStorage.setItem(STORAGE_KEYS.searchKey, userSearchKey);
  else localStorage.removeItem(STORAGE_KEYS.searchKey);

  const avail = searchAvailable();
  toolWebSwitch.disabled = !avail;
  toolWebRow.classList.toggle("disabled", !avail);
  // Losing the key disables a previously-enabled tool.
  if (!avail && toolsEnabled.web_search) {
    toolsEnabled.web_search = false;
    toolWebSwitch.checked = false;
    saveTools();
    pushToolsToSession();
  }
  toolWebHint.textContent = userSearchKey
    ? "Using your key — stored in this browser only."
    : "No server key configured. Add your own Serper key to enable web search.";
});

// ── Camera ──────────────────────────────────────────────────────────────────

async function enableCamera() {
  if (cameraStream) return;
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("camera API is unavailable on this page");
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user" },
    audio: false,
  });
  cameraStream = stream;
  camVideo.srcObject = cameraStream;
  try { await camVideo.play(); } catch { /* autoplay quirks; muted video is fine */ }
  camPip.classList.add("visible");
  camPip.setAttribute("aria-hidden", "false");
  // Lets the footer reflow to the bottom-right (and hide on mobile) while the
  // webcam preview occupies the bottom of the stage.
  document.body.classList.add("cam-on");
}

function disableCamera() {
  if (cameraStream) {
    for (const t of cameraStream.getTracks()) t.stop();
    cameraStream = null;
  }
  camVideo.srcObject = null;
  camPip.classList.remove("visible");
  camPip.setAttribute("aria-hidden", "true");
  document.body.classList.remove("cam-on");
}

/** Start the webcam only after an explicit user action or after the browser
 *  reports that camera permission has been granted. */
async function autoStartCamera() {
  if (!toolsEnabled.camera_snapshot || cameraStream) return;
  try {
    await enableCamera();
  } catch (err) {
    console.warn("[main] camera auto-start declined/failed:", err);
    toolsEnabled.camera_snapshot = false;
    saveTools();
    syncToolsUi();
  }
}

/** Track the browser's camera permission so a later re-grant (e.g. the user
 *  unblocks it from the address bar after a denial) turns the camera back on
 *  without another toggle, and a revoke turns it off. Best-effort: the
 *  Permissions API doesn't support "camera" everywhere (e.g. Safari). */
async function watchCameraPermission() {
  try {
    const status = await navigator.permissions?.query?.({ name: /** @type {any} */ ("camera") });
    if (!status) return;
    status.addEventListener("change", () => {
      if (status.state === "granted") {
        if (!toolsEnabled.camera_snapshot) {
          toolsEnabled.camera_snapshot = true;
          saveTools();
          pushToolsToSession();
        }
        void autoStartCamera();
        syncToolsUi();
      } else if (status.state === "denied") {
        disableCamera();
        if (toolsEnabled.camera_snapshot) {
          toolsEnabled.camera_snapshot = false;
          saveTools();
          pushToolsToSession();
        }
        syncToolsUi();
      }
    });
  } catch {
    // Permissions API unavailable for "camera" — the toggle still re-asks.
  }
}

/**
 * Grab the current webcam frame as a downscaled JPEG data URL. The preview is
 * mirrored in CSS for a natural self-view, but we draw the raw (un-mirrored)
 * video here so the model sees the scene in its true orientation.
 * @returns {string | null}
 */
function captureSnapshot() {
  if (!cameraStream || !camVideo.videoWidth) return null;
  const vw = camVideo.videoWidth;
  const vh = camVideo.videoHeight;
  const scale = Math.min(1, SNAPSHOT_MAX_EDGE / Math.max(vw, vh));
  const w = Math.max(1, Math.round(vw * scale));
  const h = Math.max(1, Math.round(vh * scale));
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.drawImage(camVideo, 0, 0, w, h);
  return canvas.toDataURL("image/jpeg", SNAPSHOT_QUALITY);
}

/** Brief shutter flash on the preview so the user sees a snapshot was taken. */
function flashPreview() {
  camPip.classList.remove("flash");
  void camPip.offsetWidth; // reflow so the animation restarts
  camPip.classList.add("flash");
}

// ── Tool executor ─────────────────────────────────────────────────────────
// Runs the function the model called, returns the result, and asks for a
// response so the model speaks it. Errors come back as the tool output too, so
// the model can recover gracefully instead of the turn stalling.

/**
 * Run the function the model called, return its result to the backend, and ask
 * for a follow-up response. We also hand the result back to the caller so it
 * can be shown in the conversation once the tool has actually run.
 * @param {S2sWsRealtimeClient} sessionClient
 * @param {string} name @param {string} argsJson @param {string} callId
 * @returns {Promise<{ output: string, image?: string }>}
 */
async function runTool(sessionClient, name, argsJson, callId) {
  if (client !== sessionClient) return { output: "" };
  let args = /** @type {Record<string, unknown>} */ ({});
  try { args = JSON.parse(argsJson || "{}"); } catch { /* keep {} */ }

  if (DEBUG) console.debug(`[tool] run name=${name} callId=${JSON.stringify(callId)} args=${argsJson}`);
  if (!callId) console.warn("[tool] empty call_id — the backend didn't tag the call, can't return a function_call_output");

  /** @type {{ output: string, image?: string }} */
  let result = { output: "" };
  const toolStartedAt = performance.now();
  addPipelineMetric({ stage: "tool", status: "active", detail: { name } });
  try {
    if (name === "web_search") {
      const query = typeof args.query === "string" ? args.query : "";
      result.output = await execWebSearch(query);
    } else if (name === "camera_snapshot") {
      const dataUrl = captureSnapshot();
      if (dataUrl) {
        if (DEBUG) console.debug(`[tool] camera_snapshot captured frame (${dataUrl.length} chars), sending image + output`);
        result = { output: "Snapshot captured from the webcam and attached as an image.", image: dataUrl };
        // Return the tool output; the frame itself rides along with the
        // response.create below (sent right before it), so the model sees the
        // snapshot in the very response it's about to speak.
        flashPreview();
      } else {
        console.warn("[tool] camera_snapshot: no frame — camera off or not ready");
        result.output = "The camera is not available right now.";
      }
    } else {
      result.output = `Unknown tool: ${name}`;
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    result.output = `Tool failed: ${msg}`;
  }
  if (client !== sessionClient) return result;
  try {
    addPipelineMetric({ stage: "tool", status: "sending_output", detail: { name, callId } });
    const outputAck = sessionClient.sendToolOutput(callId, result.output);
    // Hosted ordering: output, optional image, then response.create. The
    // backend owns the response-ID barrier and starts exactly one follow-up.
    sessionClient.requestToolResponse(result.image ? { image: result.image } : undefined);
    await outputAck;
    addPipelineMetric({ stage: "tool", status: "output_acknowledged", detail: { name, callId } });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    result.output = `Tool output was not accepted: ${msg}`;
    addPipelineMetric({ stage: "tool", status: "failed", elapsed_ms: performance.now() - toolStartedAt, detail: { name } });
    return result;
  }
  if (DEBUG) console.debug(`[tool] requesting model response after ${name}`);
  addPipelineMetric({ stage: "tool", status: "done", elapsed_ms: performance.now() - toolStartedAt, detail: { name } });
  return result;
}

function renderTtsBackendOptions() {
  inputTtsBackend.replaceChildren();
  for (const id of ["faster", "groxaxo", "qwen3tts-audiocpp"]) {
    const status = ttsBackendStatuses[id];
    const option = document.createElement("option");
    option.value = id;
    const name = status?.displayName || (id === "faster" ? "FasterQwen3TTS" : id === "groxaxo" ? "Groxaxo candidate" : "Qwen3TTS audio.cpp");
    const model = status?.currentModel || "no Base model";
    const state = status?.ready ? "ready" : status?.reachable ? "needs validation" : "unavailable";
    option.textContent = `${name} - ${model} (${state})`;
    inputTtsBackend.append(option);
  }
  inputTtsBackend.value = settings.ttsBackend;
}

/** @param {string} query @returns {Promise<string>} */
async function execWebSearch(query) {
  if (!query) return "No query provided.";
  /** @type {Record<string, string>} */
  const body = { query };
  // Only send a user key when there's no server key (server prefers its own).
  if (!serverSearchKey && userSearchKey) body.key = userSearchKey;

  const res = await fetch("api/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = String(res.status);
    try { const j = await res.json(); if (j.detail) detail = j.detail; } catch {}
    throw new Error(`search error (${detail})`);
  }
  const json = await res.json();
  // Date-stamp the header so the model treats these as fresh realtime facts
  // rather than its (older) training knowledge.
  const today = new Date().toISOString().slice(0, 10);
  /** @type {string[]} */
  const lines = [`Google search result from ${today}:`];
  if (json.answer) lines.push(`Answer: ${json.answer}`);
  for (const r of json.results || []) {
    lines.push(`- ${r.title}: ${r.snippet} (${r.url})`);
  }
  return lines.length > 1 ? lines.join("\n") : `${lines[0]}\nNo results found.`;
}

/** Learn server config (search key + connection target), then refresh the UI. */
function setDiagnosticsOpen(open) {
  diagnosticsOpen = open;
  diagnosticsPanel.hidden = !open;
  diagnosticsBtn.classList.toggle("active", open);
  localStorage.setItem(STORAGE_KEYS.diagnostics, open ? "1" : "0");
  renderDiagnostics();
}

function initDiagnosticsGeometry() {
  if (!diagnosticsPanel) return;
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEYS.diagnosticsGeometry) || "null");
    if (saved && innerWidth > 600) {
      Object.assign(diagnosticsPanel.style, { left: `${saved.x}px`, top: `${saved.y}px`, right: "auto", bottom: "auto", width: `${saved.w}px`, height: `${saved.h}px` });
    }
  } catch {}
  const header = diagnosticsPanel.querySelector(".diagnostics-header");
  let drag = null;
  header?.addEventListener("pointerdown", (event) => {
    if (innerWidth <= 600 || event.target.closest("button")) return;
    const rect = diagnosticsPanel.getBoundingClientRect();
    drag = { dx: event.clientX - rect.left, dy: event.clientY - rect.top };
    header.setPointerCapture(event.pointerId);
  });
  header?.addEventListener("pointermove", (event) => {
    if (!drag) return;
    const x = Math.max(0, Math.min(innerWidth - diagnosticsPanel.offsetWidth, event.clientX - drag.dx));
    const y = Math.max(0, Math.min(innerHeight - diagnosticsPanel.offsetHeight, event.clientY - drag.dy));
    Object.assign(diagnosticsPanel.style, { left: `${x}px`, top: `${y}px`, right: "auto", bottom: "auto" });
  });
  const persist = () => {
    drag = null;
    if (innerWidth <= 600) return;
    const rect = diagnosticsPanel.getBoundingClientRect();
    localStorage.setItem(STORAGE_KEYS.diagnosticsGeometry, JSON.stringify({ x: rect.left, y: rect.top, w: rect.width, h: rect.height }));
  };
  header?.addEventListener("pointerup", persist);
  header?.addEventListener("pointercancel", persist);
  new ResizeObserver(persist).observe(diagnosticsPanel);
}

function addPipelineMetric(metric) {
  pipelineMetrics.push({ ...metric, receivedAt: performance.now() });
  if (pipelineMetrics.length > 80) pipelineMetrics = pipelineMetrics.slice(-80);
  renderDiagnostics();
}

function setDiagnosticWarning(key, message = "") {
  if (message) diagnosticWarnings.set(key, message);
  else diagnosticWarnings.delete(key);
  diagnosticsWarning.hidden = diagnosticWarnings.size === 0;
  diagnosticsWarning.textContent = Array.from(diagnosticWarnings.values()).join(" ");
}

function renderDiagnostics() {
  if (!diagnosticsList || !diagnosticsSummary || !diagnosticsGraph || !diagnosticsWaterfall) return;
  const turnStart = pipelineMetrics.map((m) => `${m.stage}.${m.status}`).lastIndexOf("mic.speaking");
  const visibleMetrics = turnStart >= 0 ? pipelineMetrics.slice(turnStart) : pipelineMetrics;
  const ttsStart = [...visibleMetrics].reverse().find((m) => m.stage === "tts" && m.status === "request_start");
  const ttsFirst = [...visibleMetrics].reverse().find((m) => m.stage === "tts" && m.status === "first_audio");
  const ttsDone = [...visibleMetrics].reverse().find((m) => m.stage === "tts" && m.status === "done");
  const ttsFailure = [...visibleMetrics].reverse().find((m) => m.stage === "tts" && ["failed", "runaway_aborted"].includes(m.status));
  const llmFirstPhrase = [...visibleMetrics].reverse().find((m) => m.stage === "gemma" && m.status === "first_stable_phrase");
  const firstPlayback = [...visibleMetrics].reverse().find((m) => m.stage === "playback" && m.status === "first_audio");
  const playbackQueue = [...visibleMetrics].reverse().find((m) => (
    m.stage === "playback" && Number.isFinite(Number(m.detail?.prime_target_ms))
  ));
  const responseLifecycle = [...visibleMetrics].reverse().find((m) => (
    m.stage === "response" && m.status === "lifecycle"
  ));
  const responseDone = [...visibleMetrics].reverse().find((m) => m.stage === "response" && m.status === "done");
  if (diagnosticsAudioMetrics && (ttsStart || ttsFirst || ttsDone || llmFirstPhrase || firstPlayback || playbackQueue || responseLifecycle || responseDone)) {
    const start = ttsStart?.detail || {}; const first = ttsFirst?.detail || {};
    const done = ttsFailure && (!ttsDone || ttsFailure.receivedAt > ttsDone.receivedAt) ? ttsFailure.detail || {} : ttsDone?.detail || {};
    const selectedTtsStatus = ttsBackendStatuses[settings.ttsBackend];
    const gpu = done.gpu || selectedTtsStatus?.health?.backend?.runtime?.gpu_now;
    const gpuText = gpu && typeof gpu === "object"
      ? `${gpu.freeMiB ?? "unknown"} MiB free / ${gpu.utilizationPercent ?? "unknown"}% utilization`
      : gpu ?? "unknown";
    const firstPhraseMs = llmFirstPhrase?.elapsed_ms ?? llmFirstPhrase?.detail?.first_stable_phrase_ms ?? "unknown";
    const firstPlaybackMs = firstPlayback?.elapsed_ms ?? firstPlayback?.detail?.first_playback_ms ?? "unknown";
    const endToEndMs = responseDone?.elapsed_ms ?? responseDone?.detail?.end_to_end_ms ?? "unknown";
    const referenceSeconds = (value) => Number.isFinite(Number(value)) ? `${Number(value).toFixed(3)} s` : "unknown";
    const referenceLimit = done.reference_limit_applied == null
      ? "limit status unknown"
      : done.reference_limit_applied ? "limit applied" : "limit not applied";
    const referencePairing = done.reference_pairing ?? "unknown pairing";
    const playback = playbackQueue?.detail || {};
    const requestedLanguage = done.requested_language ?? first.requested_language ?? start.requested_language ?? "unknown";
    const effectiveLanguage = done.effective_language ?? first.effective_language ?? start.effective_language ?? "unknown";
    const autoLanguageSupport = done.language_auto_supported ?? first.language_auto_supported ?? start.language_auto_supported;
    const autoLanguageText = autoLanguageSupport == null ? "unknown" : autoLanguageSupport ? "verified" : "not verified";
    const lifecycleDetail = responseLifecycle?.detail || {};
    const lifecycle = lifecycleDetail.lifecycle_state ?? done.lifecycle_state ?? start.lifecycle_state ?? "unknown";
    const responseEpoch = lifecycleDetail.response_epoch ?? done.response_epoch ?? start.response_epoch ?? "unknown";
    const inputEpoch = lifecycleDetail.input_epoch ?? done.input_epoch ?? start.input_epoch ?? "unknown";
    const supersessionReason = lifecycleDetail.supersession_reason ?? done.supersession_reason ?? "none";
    const frozenProfile = done.frozen_profile_id ?? done.profile_id ?? start.tts_profile_id ?? "unknown";
    diagnosticsAudioMetrics.textContent = `Lifecycle: ${lifecycle} · input epoch ${inputEpoch} · response epoch ${responseEpoch} · supersession ${supersessionReason}\nProfile: selected ${realtimeTuningState(settings.ttsBackend).profile_id || "unknown"} · frozen ${frozenProfile} rev ${done.profile_revision ?? "unknown"} · clone ${done.clone_fingerprint ?? "unknown"} · seed ${done.seed_policy ?? done.seed ?? "unknown"} · phrase ${done.phrase_index ?? "unknown"}\nModel: ${done.model ?? start.model ?? "unknown model"} · Mode: ${done.delivery_mode ?? done.mode ?? start.streaming_mode ?? "unknown"}\nLanguage: requested ${requestedLanguage} · effective ${effectiveLanguage} · Auto ${autoLanguageText}\nLLM first stable phrase: ${firstPhraseMs} ms\nTTS first PCM: ${first.first_pcm_ms ?? ttsFirst?.elapsed_ms ?? "unknown"} ms · First playback: ${firstPlaybackMs} ms\nPlayback: ${playback.continuity_mode ?? settings.playbackContinuity} · prime ${playback.prime_target_ms ?? 0} ms · queued ${Math.round(Number(playback.queued_ms || 0))} ms · underruns ${playback.underruns ?? 0} · re-primes ${playback.reprimes ?? 0} · stale ${playback.stale_chunks ?? 0}${playback.provider_unsustainable ? " · Provider cannot sustain realtime" : ""}\nSynthesis RTF: ${done.rtf ?? "unknown"} · End-to-end: ${endToEndMs} ms\nGeneration: ${done.generation_ms ?? "unknown"} ms · Audio: ${done.audio_duration_ms ?? "unknown"} ms · GPU: ${gpuText}\nReference: source ${referenceSeconds(done.reference_source_seconds)} · requested ${referenceSeconds(done.reference_requested_limit_seconds)} · used ${referenceSeconds(done.reference_used_seconds)} · ${referenceLimit} · ${referencePairing}`;
  }
  if (diagnosticsAudioMetrics && (ttsFirst || ttsDone || ttsFailure)) {
    const end = ttsFailure && (!ttsDone || ttsFailure.receivedAt > ttsDone.receivedAt) ? ttsFailure : ttsDone;
    const detail = end?.detail || {};
    const first = ttsFirst?.detail || {};
    diagnosticsAudioMetrics.textContent += `\nEngine stages: prompt ${detail.engine_prompt_ms ?? "unknown"} ms · prefill ${detail.engine_prefill_ms ?? "unknown"} ms · talker ${detail.engine_talker_ms ?? "unknown"} ms · decode ${detail.engine_decode_ms ?? "unknown"} ms\nCandidate-observed first PCM (includes admission): ${first.engine_first_pcm_ms ?? detail.engine_first_pcm_ms ?? "unknown"} ms · native proof hold: ${first.native_proof_hold_ms ?? detail.native_proof_hold_ms ?? "unknown"} ms\nCodec frames: ${detail.generated_codec_frames ?? "unknown"} / cap ${detail.generation_cap_frames ?? "unknown"} · engine EOS: ${detail.engine_eos == null ? "unknown" : detail.engine_eos ? "yes" : "no"} · outcome: ${end?.status ?? "pending"}`;
  }
  const echo = [...visibleMetrics].reverse().find((m) => m.stage === "echo_guard");
  if (echo && diagnosticsAudioStatus) {
    const detail = echo.detail || {};
    const requested = detail.requested_mode || latestEchoStatus?.requestedMode || settings.echoGuard;
    const effective = detail.effective_mode || latestEchoStatus?.effectiveMode || "native";
    const doubleTalk = detail.double_talk == null ? "unknown" : detail.double_talk ? "yes" : "no";
    diagnosticsAudioStatus.textContent = `Requested/effective echo: ${requested}/${effective} · AEC3 ${detail.module_available ? "ready" : "fallback"} · reference ${detail.playback_active ? "active" : "idle"} · delay ${detail.lag_ms ?? "unknown"} ms · confidence ${detail.prediction_confidence ?? detail.correlation ?? "unknown"} · double-talk ${doubleTalk}.`;
  }
  const latestByStage = new Map();
  for (const metric of visibleMetrics) latestByStage.set(metric.stage, metric);
  const measured = visibleMetrics.filter((m) => typeof m.elapsed_ms === "number" && m.elapsed_ms >= 0);
  const bottleneck = measured.reduce((best, item) => !best || item.elapsed_ms > best.elapsed_ms ? item : best, null);
  const latestContext = [...pipelineMetrics].reverse().find((metric) => metric.stage === "context");
  const context = latestContext || (backendRuntime?.context
    ? { status: "ready", detail: backendRuntime.context }
    : null);
  const contextMax = context?.detail?.max_tokens ?? localPipeline?.gemma?.contextWindow;
  const contextUsed = context?.detail?.history_tokens ?? 0;
  renderHistoryCompactionDiagnostics(context?.detail || null);
  const contextText = contextMax ? `Context ${contextUsed.toLocaleString()} / ${contextMax.toLocaleString()}` : "";
  const modelText = localPipeline?.gemma
    ? `${localPipeline.gemma.provider === "remote" ? "Remote" : "Local"} model ${localPipeline.gemma.model || "unknown"} @ ${localPipeline.gemma.baseUrl || "unknown"}`
    : "";
  const runtimeText = backendRuntime
    ? `Backend API ${backendRuntime.api_version} | PID ${backendRuntime.pid} | started ${backendRuntime.started_at_utc}`
    : "Backend identity pending";
  diagnosticsSummary.textContent = bottleneck
    ? `${runtimeText}${modelText ? ` | ${modelText}` : ""} | ${contextText}${contextText ? " | " : ""}Likely bottleneck: ${bottleneck.stage} ${Math.round(bottleneck.elapsed_ms)} ms`
    : `${runtimeText}${modelText ? ` | ${modelText}` : ""}${contextText ? ` | ${contextText}` : ""}`;
  diagnosticsGraph.replaceChildren();
  for (const stage of DIAGNOSTIC_STAGES) {
    const metric = latestByStage.get(stage);
    const node = document.createElement("div");
    node.className = `diag-node ${metric ? "has-data" : ""} ${metric?.status || "idle"}`;
    const name = document.createElement("strong");
    name.textContent = DIAGNOSTIC_STAGE_LABELS[stage] || stage.replace(/^./, (c) => c.toUpperCase());
    const state = document.createElement("span");
    state.textContent = stage === "context" && contextMax
      ? `${contextUsed.toLocaleString()} / ${contextMax.toLocaleString()}`
      : metric ? `${metric.status}${typeof metric.elapsed_ms === "number" ? ` ${Math.round(metric.elapsed_ms)} ms` : ""}` : "idle";
    node.append(name, state);
    diagnosticsGraph.append(node);
  }
  diagnosticsWaterfall.replaceChildren();
  for (const metric of measured.slice(-8)) {
    const row = document.createElement("div");
    row.className = "diag-waterfall-row";
    const label = document.createElement("span"); label.textContent = metric.stage;
    const bar = document.createElement("i"); bar.style.width = `${Math.max(3, Math.min(100, metric.elapsed_ms / Math.max(1, bottleneck?.elapsed_ms || 1) * 100))}%`;
    const value = document.createElement("b"); value.textContent = `${Math.round(metric.elapsed_ms)} ms`;
    row.append(label, bar, value); diagnosticsWaterfall.append(row);
  }
  diagnosticsList.replaceChildren();
  for (const metric of visibleMetrics.slice(-24).reverse()) {
    const row = document.createElement("div");
    row.className = "diag-row";
    const label = document.createElement("span");
    label.className = "diag-stage";
    label.textContent = `${metric.stage}.${metric.status}`;
    const value = document.createElement("span");
    value.className = "diag-value";
    const bits = [];
    if (typeof metric.elapsed_ms === "number") bits.push(`${Math.round(metric.elapsed_ms)} ms`);
    if (metric.detail && Object.keys(metric.detail).length) bits.push(JSON.stringify(metric.detail));
    value.textContent = bits.join("  ") || "now";
    row.append(label, value);
    diagnosticsList.append(row);
  }
  updateRealtimeAudioSummary();
}

async function refreshLocalPipeline() {
  return refreshLocalIdentity({ retry: true, resetRetry: true });
}

async function refreshTtsBackends() {
  try {
    const res = await fetch("api/tts/backends", { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const json = await res.json();
    ttsBackendStatuses = Object.fromEntries((json.backends || []).map((item) => [item.id, item]));
  } catch (err) {
    console.warn("[ui] failed to load TTS backend status:", err);
    ttsBackendStatuses = {};
  }
  renderTtsBackendOptions();
  updateRealtimeAudioSummary();
}

/**
 * Reconcile selected-provider identity with no cached state. A newer refresh
 * wins over a late focus/startup result so a just-selected backend never shows
 * the previous provider's model. This probes only; it never manages services.
 */
async function refreshLocalIdentity({ retry = false, resetRetry = false } = {}) {
  if (resetRetry) {
    if (localIdentityRetryTimer) window.clearTimeout(localIdentityRetryTimer);
    localIdentityRetryTimer = 0;
    localIdentityRetryAttempts = 0;
  }
  const request = ++localIdentityRequest;
  const selectedBackend = settings.ttsBackend;
  const pipelinePromise = fetch("api/local-pipeline", { cache: "no-store" }).then(async (res) => {
    if (!res.ok) throw new Error(`local pipeline HTTP ${res.status}`);
    return res.json();
  });
  const backendPromise = fetch("api/tts/backends", { cache: "no-store" }).then(async (res) => {
    if (!res.ok) throw new Error(`TTS backends HTTP ${res.status}`);
    return res.json();
  });

  // Identity is useful before the (potentially slower) clone/backend inventory
  // has completed.  Render the selected provider as soon as local-pipeline
  // settles; otherwise a delayed inventory call leaves the static Checking…
  // labels visible despite a healthy local service.
  const pipelineResult = await pipelinePromise.then(
    (value) => ({ status: "fulfilled", value }),
    (reason) => ({ status: "rejected", reason }),
  );
  if (request !== localIdentityRequest || selectedBackend !== settings.ttsBackend) return false;
  const pipelineOk = pipelineResult.status === "fulfilled";
  localPipeline = pipelineOk ? pipelineResult.value : {
    mode: "unavailable", gemma: { reachable: false },
    tts: { backend: selectedBackend, reachable: false, displayName: selectedBackend },
    vad: {}, tools: {}, unavailableReason: pipelineResult.reason?.message || "Local status unavailable",
  };
  // This fallback intentionally uses local-pipeline's selected-provider data
  // until the richer backend inventory arrives.
  renderLocalPipeline();

  const backendResult = await backendPromise.then(
    (value) => ({ status: "fulfilled", value }),
    (reason) => ({ status: "rejected", reason }),
  );
  if (request !== localIdentityRequest || selectedBackend !== settings.ttsBackend) return false;
  const backendsOk = backendResult.status === "fulfilled";
  ttsBackendStatuses = backendsOk
    ? Object.fromEntries((backendResult.value.backends || []).map((item) => [item.id, item]))
    : {};
  renderTtsBackendOptions();
  renderLocalPipeline();
  updateRealtimeAudioSummary();
  if (!pipelineOk || !backendsOk) {
    setDiagnosticWarning("local-identity", `Selected provider status unavailable: ${pipelineResult.reason?.message || backendResult.reason?.message || "probe failed"}`);
    if (retry && !localIdentityRetryTimer && localIdentityRetryAttempts < MAX_LOCAL_IDENTITY_RETRIES) {
      localIdentityRetryAttempts += 1;
      localIdentityRetryTimer = window.setTimeout(() => {
        localIdentityRetryTimer = 0;
        void refreshLocalIdentity({ retry: true });
      }, 1500);
    }
  } else {
    localIdentityRetryAttempts = 0;
    if (localIdentityRetryTimer) window.clearTimeout(localIdentityRetryTimer);
    localIdentityRetryTimer = 0;
    setDiagnosticWarning("local-identity");
  }
  return pipelineOk && backendsOk;
}

async function assertTtsBackendReady() {
  await refreshTtsBackends();
  // Candidate startup requires a live supervisor profile resolve.  Do not let
  // a stale cached profile reach the WebSocket and fail after the mic starts.
  await refreshCandidateTuningProfiles();
  await fetchVoiceProfiles();
  const status = ttsBackendStatuses[settings.ttsBackend];
  const selectedExists = voiceProfiles.some((profile) => profile.voice === settings.voice);
  if (status?.ready && selectedExists && (!status.explicitValidation || status.validation?.voice === settings.voice)) {
    // Freeze this browser-side resolve for the initial pipeline update.  Later
    // diagnostics changes deliberately send a new update for the *next*
    // response; they cannot mutate this initial session snapshot in transit.
    return requiredCandidateTtsTuning(settings.ttsBackend);
  }
  if (!selectedExists) throw new Error("The selected clone is not present in the selected TTS backend. Refresh or choose an available clone.");
  if (settings.ttsBackend === "groxaxo") {
    throw new Error("Groxaxo is unavailable or has no 0.6B-Base/1.7B-Base model loaded in Voice Studio.");
  }
  if (settings.ttsBackend === "qwen3tts-audiocpp" || settings.ttsBackend === "audio-cpp") {
    throw new Error(status?.error || "audio.cpp is not loaded, profile-compatible, or validated for native PCM or the buffered fallback.");
  }
  throw new Error("FasterQwen3TTS is unavailable or its 1.7B-Base clone model is not ready.");
}

async function assertModelEndpointReady() {
  const response = await fetch("api/model/test", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(modelEndpointConfig(settings)),
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || "The selected model endpoint is unavailable.");
}

function renderLocalPipeline() {
  if (!localPipeline) return;
  const set = (id, value) => { const el = document.getElementById(id); if (el) el.textContent = value; };
  const provider = (component, fallback) => {
    const online = component?.reachable;
    const label = component?.model || fallback;
    return `${label} (${online ? "online" : "unavailable"})`;
  };
  set("local-provider-gemma", provider(localPipeline.gemma, "Local Gemma via llama.cpp"));
  const selectedTts = ttsBackendStatuses[settings.ttsBackend];
  const selectedTtsName = settings.ttsBackend === "groxaxo"
    ? "Groxaxo candidate"
    : (settings.ttsBackend === "qwen3tts-audiocpp" || settings.ttsBackend === "audio-cpp")
      ? `Qwen3TTS audio.cpp (${selectedTts?.nativeStreaming ? "native PCM" : "buffered fallback"})`
      : "FasterQwen3TTS";
  const selectedTtsLabel = selectedTts
    ? `${selectedTtsName} ${selectedTts.currentModel || "Base model"} (${selectedTts.ready ? "online" : "unavailable"})`
    : provider(localPipeline.tts, "Local Qwen3-TTS 1.7B Base via FasterQwen3TTS");
  set("local-provider-tts", selectedTtsLabel);
  set("about-vad-model", `${localPipeline.vad?.name || "Silero VAD"} (${localPipeline.vad?.device || "CPU"})`);
  set("about-gemma-model", localPipeline.gemma?.model || "Gemma audio model");
  set("about-gemma-url", `${localPipeline.gemma?.baseUrl || ""}${localPipeline.gemma?.reachable ? " (online)" : " (not reached)"}`);
  set("about-tts-model", `${selectedTtsName} ${selectedTts?.currentModel || localPipeline.tts?.model || "Base model"}`);
  set("about-tts-url", `${selectedTts?.endpoint || localPipeline.tts?.baseUrl || ""}${selectedTts?.ready ? " (online)" : " (not ready)"}`);
  set("about-tools-status", `Serper ${localPipeline.tools?.serper ? "configured" : "needs key"}; camera available`);
}

async function fetchConfig() {
  try {
    const res = await fetch("api/config");
    if (res.ok) {
      const json = await res.json();
      if (json.apiVersion !== EXPECTED_UI_API_VERSION) {
        setDiagnosticWarning("frontend-version", `Frontend server restart required (API ${json.apiVersion ?? "missing"}, expected ${EXPECTED_UI_API_VERSION}).`);
      } else {
        setDiagnosticWarning("frontend-version");
      }
      serverSearchKey = !!json.search;
      lbMode = !!json.lb;
      // Lock to LB mode only when the deploy reports a load balancer.
      allowDirect = json.allowDirect ?? !lbMode;
      // The conversation-time limiter rides on the LB being present.
      limiterOn = lbMode;
    }
    // Non-OK response: leave the fail-open default (allowDirect = true).
  } catch {
    // Config endpoint unreachable (e.g. static hosting): keep direct entry.
  }
  if (DEBUG) console.debug(`[ui] config: allowDirect=${allowDirect} lbMode=${lbMode}`);
  // Login chip + remaining-budget (no-op / hidden when the limiter is off).
  void account.refresh();
  await fetchVoiceProfiles();
  await refreshTtsBackends();
  await refreshLocalIdentity();
  await refreshFasterModelInventory();
  ttsStreamingStatus.textContent = settings.fullBufferTts
    ? "Non-streaming assistant dispatch is selected. The backend's actual PCM capability remains separately reported."
    : "Streaming assistant dispatch is selected by default; phrase-sized text reaches the backend as it is generated.";
  syncToolsUi();
  syncConnectionUi();
}

async function fetchVoiceProfiles() {
  const backend = settings.ttsBackend;
  const request = ++voiceInventoryRequest;
  // Do not leave a previous provider's clones visible while the selected
  // provider is loading or unavailable.
  if (!backend) {
    clearVoiceProfileOptions("", "Select a TTS backend to load its live clone profiles");
    return;
  }
  clearVoiceProfileOptions(backend);
  try {
    const res = await fetch(`api/tts/backends/${encodeURIComponent(backend)}/voices`, { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const json = await res.json();
    if (request !== voiceInventoryRequest || backend !== settings.ttsBackend) return;
    applyVoiceProfilePayload(json, backend);
    return;
  } catch (err) {
    if (request !== voiceInventoryRequest || backend !== settings.ttsBackend) return;
    console.warn(`[ui] failed to load ${backend} clone voices:`, err);
    defaultVoice = DEFAULT_VOICE;
    voiceProfiles = [];
    profileLibraryWritable = false;
  }
  renderVoiceOptions();
}

async function refreshFasterModelInventory() {
  const status = ttsBackendStatuses[settings.ttsBackend];
  const model = status?.currentModel || status?.requiredModel || "no resident Base model";
  const state = status?.ready ? "ready" : status?.reachable ? "reachable; validation required" : "unavailable";
  modelInventoryStatus.textContent = `${status?.displayName || settings.ttsBackend}: ${model}; ${state}. HFRT does not change backend model residency.`;
  for (const button of [fasterModelLoadBtn, fasterModelSwitchBtn, fasterModelUnloadBtn]) button.disabled = true;
}

async function runFasterModelOperation(action) {
  try {
    await qwen3Json(`api/qwen3/models/${action}`, { method: "POST", body: JSON.stringify({}) });
  } catch (err) {
    modelInventoryStatus.textContent = err instanceof Error ? err.message : String(err);
  }
  await refreshFasterModelInventory();
}

async function validateAudioCppCandidate() {
  validateAudioCppBtn.disabled = true;
  audioCppStatus.textContent = "Checking the loaded Voice Studio model, selected clone, native PCM, and buffered fallback…";
  try {
    const profile = selectedProfile();
    const status = await qwen3Json(`api/tts/backends/${encodeURIComponent(settings.ttsBackend)}/validate`, {
      method: "POST",
      body: JSON.stringify({ voice: profile?.voice || settings.voice }),
    });
    const limitations = Array.isArray(status.limitations) ? status.limitations.join(" ") : "";
    const probe = status.speechProbe || {};
    const probeDetail = probe.bytes
      ? `; ${probe.bytes} bytes in ${probe.chunks || 1} chunk(s)${probe.firstChunkMs != null ? `, first in ${probe.firstChunkMs} ms` : ""}`
      : "";
    audioCppStatus.textContent = status.ready
      ? `${status.displayName || settings.ttsBackend} is verified for ${profile?.name || settings.voice}; mode ${status.mode || status.deliveryMode || "unknown"}${probeDetail}. ${limitations}`.trim()
      : `${status.error || "Selected backend is not ready."} ${limitations}`.trim();
  } catch (err) {
    audioCppStatus.textContent = `Backend validation failed: ${err instanceof Error ? err.message : String(err)}`;
  } finally {
    validateAudioCppBtn.disabled = false;
    await refreshTtsBackends();
  }
}

inputVoice.addEventListener("change", () => { void selectVoiceProfile(); });
inputTtsBackend.addEventListener("change", async () => {
  const previous = settings.ttsBackend;
  settings.voiceByBackend = { ...(settings.voiceByBackend || {}), [previous]: settings.voice };
  settings.ttsBackend = normalizeTtsProvider(inputTtsBackend.value);
  settings.voice = settings.voiceByBackend[settings.ttsBackend] || "";
  // The previous provider's profile resolution is never valid for the newly
  // selected backend.  Clear it synchronously before any inventory request.
  candidateTuningResolved = null;
  // Invalidate any older provider request and remove its identities before the
  // first await; stale clones must never remain selectable during a switch.
  voiceInventoryRequest += 1;
  clearVoiceProfileOptions(settings.ttsBackend || "", settings.ttsBackend
    ? `Loading live Base clone profiles for ${settings.ttsBackend}…`
    : "Select a TTS backend to load its live clone profiles");
  audioCppStatus.textContent = `Loading ${settings.ttsBackend} status…`;
  await Promise.all([
    refreshCandidateTuningProfiles().catch((error) => console.warn("candidate tuning refresh failed", error)),
    fetchVoiceProfiles(),
  ]);
  const saved = await saveSettings(settings);
  if (!saved.ok) profileLibraryStatus.textContent = `Could not save selected backend: ${saved.error}`;
  await refreshTtsBackends();
  await refreshLocalIdentity({ retry: true, resetRetry: true });
  await refreshFasterModelInventory();
  renderSelectedTtsBackendStatus();
  updateRealtimeAudioSummary();
});
profileCreateBtn.addEventListener("click", async () => {
  const source = selectedProfile();
  try {
    const payload = await qwen3Json(`api/tts/backends/${encodeURIComponent(settings.ttsBackend)}/profiles`, {
      method: "POST",
      body: JSON.stringify({
        name: inputProfileName.value.trim(), ref_text: inputProfileRefText.value,
        language: inputProfileLanguage.value.trim() || "Auto", source_profile_id: source?.id,
      }),
    });
    applyVoiceProfilePayload(payload);
    profileLibraryStatus.textContent = "Clone duplicated in the selected backend library.";
  } catch (err) { profileLibraryStatus.textContent = err instanceof Error ? err.message : String(err); }
});
profileSaveBtn.addEventListener("click", async () => {
  const profile = selectedProfile();
  if (!profile) return;
  try {
    const payload = await qwen3Json(`api/tts/backends/${encodeURIComponent(settings.ttsBackend)}/profiles/${encodeURIComponent(profile.id)}`, {
      method: "PATCH",
      body: JSON.stringify({ name: inputProfileName.value.trim(), ref_text: inputProfileRefText.value, language: inputProfileLanguage.value.trim() || "Auto" }),
    });
    applyVoiceProfilePayload(payload);
    profileLibraryStatus.textContent = "Clone profile saved.";
  } catch (err) { profileLibraryStatus.textContent = err instanceof Error ? err.message : String(err); }
});
profileDeleteBtn.addEventListener("click", async () => {
  const profile = selectedProfile();
  if (!profile || !window.confirm(`Delete clone profile ${profile.name}?`)) return;
  try {
    const payload = await qwen3Json(`api/tts/backends/${encodeURIComponent(settings.ttsBackend)}/profiles/${encodeURIComponent(profile.id)}`, { method: "DELETE" });
    if (settings.voice === profile.voice) settings.voice = defaultVoice;
    saveSettings(settings);
    applyVoiceProfilePayload(payload);
    profileLibraryStatus.textContent = "Clone profile deleted.";
  } catch (err) { profileLibraryStatus.textContent = err instanceof Error ? err.message : String(err); }
});
profileImportBtn.addEventListener("click", async () => {
  const file = inputProfileAudio.files?.[0];
  if (!file) { profileLibraryStatus.textContent = "Choose a WAV reference file to import."; return; }
  try {
    const dataUrl = await new Promise((resolve, reject) => { const reader = new FileReader(); reader.onload = () => resolve(reader.result); reader.onerror = () => reject(reader.error); reader.readAsDataURL(file); });
    const audio_base64 = String(dataUrl).split(",", 2)[1] || "";
    const payload = await qwen3Json(`api/tts/backends/${encodeURIComponent(settings.ttsBackend)}/profiles`, {
      method: "POST",
      body: JSON.stringify({ name: inputProfileName.value.trim(), ref_text: inputProfileRefText.value, language: inputProfileLanguage.value.trim() || "Auto", audio_base64, audio_filename: file.name }),
    });
    applyVoiceProfilePayload(payload);
    profileLibraryStatus.textContent = "Reference WAV imported into the selected backend library.";
  } catch (err) { profileLibraryStatus.textContent = err instanceof Error ? err.message : String(err); }
});
fasterModelLoadBtn.addEventListener("click", () => { void runFasterModelOperation("load"); });
fasterModelSwitchBtn.addEventListener("click", () => { void runFasterModelOperation("switch"); });
fasterModelUnloadBtn.addEventListener("click", () => { void runFasterModelOperation("unload"); });
validateAudioCppBtn.addEventListener("click", () => { void validateAudioCppCandidate(); });

/**
 * Resolve where to connect, per the deploy's mode:
 *   • LB mode  -> `{ sessionUrl }`, the client POSTs the same-origin /api/session
 *     proxy and the server forwards to the LB (its address stays server-side).
 *   • direct   -> `{ directUrl }`, connect straight to the s2s WebSocket.
 * Throws a user-facing error if direct mode is on but no URL was entered.
 * @returns {{ sessionUrl: string } | { directUrl: string }}
 */
function connectionTarget() {
  if (!allowDirect) {
    return { sessionUrl: "api/session" };
  }
  const directUrl = buildDirectWsUrl(settings.directUrl);
  if (!directUrl) {
    throw new Error("Enter a speech-to-speech server URL in Settings.");
  }
  return { directUrl };
}

/**
 * Normalise a user-typed server address into a realtime WebSocket URL.
 * Accepts bare hosts (`localhost:8080`), http(s) URLs, or ws(s) URLs, and adds
 * the `/v1/realtime` path when none is given. A full connect URL (with path
 * and/or query) is preserved as-is.
 * @param {string} raw @returns {string}
 */
function buildDirectWsUrl(raw) {
  let s = (raw || "").trim();
  if (!s) return "";
  if (!/^wss?:\/\//i.test(s)) {
    if (/^https?:\/\//i.test(s)) {
      s = s.replace(/^http/i, "ws"); // http→ws, https→wss
    } else {
      const isLocal = /^(localhost|127\.0\.0\.1|\[::1\])(:|\/|$)/i.test(s);
      s = (isLocal ? "ws://" : "wss://") + s;
    }
  }
  try {
    const u = new URL(s);
    if (u.pathname === "" || u.pathname === "/") u.pathname = "/v1/realtime";
    return u.toString();
  } catch {
    return s;
  }
}

/** Create + resume an AudioContext synchronously (must run inside the user
 *  gesture so iOS lets it start). Returns null if construction fails. */
function createResumedAudioContext() {
  try {
    const Ctx = window.AudioContext || /** @type {any} */ (window).webkitAudioContext;
    const ctx = new Ctx({ latencyHint: "interactive" });
    if (ctx.state === "suspended") void ctx.resume().catch(() => {});
    return /** @type {AudioContext} */ (ctx);
  } catch (err) {
    console.warn("[main] AudioContext init failed:", err);
    return null;
  }
}

/** Read the editable settings out of the form. The URL field is only honoured
 *  in direct mode (in LB mode it's locked and server-owned). */
function readSettingsFromForm() {
  const backend = inputTtsBackend.value || "faster";
  const voice = inputVoice.value || settings.voiceByBackend?.[backend] || defaultVoice;
  return {
    directUrl: allowDirect ? inputLbUrl.value.trim() : settings.directUrl,
    voice,
    voiceByBackend: { ...(settings.voiceByBackend || {}), [backend]: voice },
    instructions: inputInstructions.value.trim() || DEFAULT_INSTRUCTIONS,
    noiseGate: readGateThreshold(),
    echoGuard: ["native", "adaptive", "strict"].includes(inputEchoGuard.value) ? inputEchoGuard.value : "native",
    echoCalibrations: settings.echoCalibrations || {},
    fullBufferTts: inputFullBufferTts.checked,
    playbackContinuity: settings.playbackContinuity === "fast-start" ? "fast-start" : "adaptive",
    ttsDeliveryMode: settings.ttsDeliveryMode === "native_incremental_pcm"
      ? "native_incremental_pcm"
      : "buffered_phrase",
    liveTranscript: inputLiveTranscript.checked,
    maxResponseTokens: Math.min(1024, Math.max(64, Number(inputMaxResponseTokens.value) || 384)),
    ttsBackend: backend,
    ttsProfileByBackend: { ...(settings.ttsProfileByBackend || {}) },
    tts_tuning: backend === AUDIO_CPP_PROVIDER ? activeTtsTuning(backend) : undefined,
    modelProvider: inputModelProvider.value === "remote" ? "remote" : "local",
    modelUrl: inputModelUrl.value.trim(),
    modelName: inputModelName.value.trim(),
    modelApiKey: inputModelApiKey.value,
  };
}

function syncModelProviderUi() {
  const remote = inputModelProvider.value === "remote";
  remoteModelFields.hidden = !remote;
  modelConnectionStatus.textContent = remote
    ? "Test this endpoint before starting a conversation."
    : "Local llama.cpp is selected.";
}

function modelEndpointConfig(s = settings) {
  if (s.modelProvider !== "remote") return { provider: "local" };
  return {
    provider: "remote",
    base_url: s.modelUrl,
    model: s.modelName,
    api_key: s.modelApiKey,
  };
}

inputModelProvider.addEventListener("change", syncModelProviderUi);
testModelConnectionBtn.addEventListener("click", async () => {
  const candidate = readSettingsFromForm();
  testModelConnectionBtn.disabled = true;
  modelConnectionStatus.textContent = "Checking model endpoint...";
  try {
    const response = await fetch("api/model/test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(modelEndpointConfig(candidate)),
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.detail || `HTTP ${response.status}`);
    modelConnectionStatus.textContent = `${body.model || candidate.modelName || "Model"} online${body.context_window ? `; context ${body.context_window.toLocaleString()}` : ""}.`;
  } catch (err) {
    modelConnectionStatus.textContent = `Connection failed: ${err instanceof Error ? err.message : String(err)}`;
  } finally {
    testModelConnectionBtn.disabled = false;
  }
});

/** Gate threshold (dBFS) currently shown on the slider, clamped to range. */
function readGateThreshold() {
  const v = Math.round(Number(inputNoiseGate.value));
  if (!Number.isFinite(v)) return GATE_OFF_DB;
  return Math.min(GATE_MAX_DB, Math.max(GATE_OFF_DB, v));
}

/** Adapt the connection field to the mode learned from /api/config. */
function syncConnectionUi() {
  if (allowDirect) {
    // Direct mode: the user sets their own s2s server URL.
    connField.hidden = false;
    inputLbUrl.value = settings.directUrl;
    inputLbUrl.placeholder = "http://127.0.0.1:8765";
    connHint.classList.remove("error");
    connHint.textContent =
      "URL of your speech-to-speech server, e.g. http://127.0.0.1:8765 (the app adds /v1/realtime).";
  } else {
    // LB mode: the load balancer URL is deployment-owned — hide it entirely so
    // its address is never exposed in Settings.
    connField.hidden = true;
  }
}

/** True when the user must supply a server URL before connecting (direct mode
 *  with nothing set). */
function missingServerUrl() {
  return allowDirect && !buildDirectWsUrl(settings.directUrl);
}

/** Open Settings and point the user at the empty server-URL field. */
function promptServerUrl() {
  if (settingsModal.open) syncConnectionUi();
  else openSettings();
  connHint.textContent = "Set the speech-to-speech server URL to start.";
  connHint.classList.add("error");
  inputLbUrl.focus();
}

settingsForm.addEventListener("submit", async (event) => {
  const submitter = /** @type {HTMLButtonElement | null} */ ((/** @type {SubmitEvent} */ (event)).submitter);
  if (submitter?.value !== "save") return;
  event.preventDefault();

  settings = readSettingsFromForm();
  settingsSaveStatus.textContent = "Saving...";
  const saved = await saveSettings(settings);
  settingsSaveStatus.textContent = saved.ok ? "Saved to managed runtime storage." : `Browser saved; server persistence failed: ${saved.error}`;
  ttsStreamingStatus.textContent = settings.fullBufferTts
    ? "Non-streaming assistant dispatch is selected. The backend's actual PCM capability remains separately reported."
    : "Streaming assistant dispatch is selected by default; phrase-sized text reaches the backend as it is generated.";

  // Voice + instructions can apply to a live session without reconnecting; a
  // changed connection URL only takes effect on the next restart.
  if (client && LIVE_STATES.has(currentState)) {
    const tuning = activeTtsTuning(settings.ttsBackend);
    client.updateSession({ voice: settings.voice, instructions: effectiveInstructions() });
    client.updateLocalPipeline(
      {
        full_buffer_tts: settings.fullBufferTts,
        live_transcription: settings.liveTranscript,
        max_response_tokens: settings.maxResponseTokens,
        tts_backend: settings.ttsBackend,
        ...(tuning ? { tts_tuning: tuning } : {}),
      },
      activePlaybackConfig(settings.ttsBackend),
    );
    client.setEchoGuard(settings.echoGuard);
  }
  if (saved.ok) window.setTimeout(() => settingsModal.close(), 350);
});

// The noise gate applies live (worklet param), so tune it without a restart:
// update the label/marker, persist, and push straight to the running client.
inputNoiseGate.addEventListener("input", () => {
  setGateThreshold(readGateThreshold());
});

restartBtn.addEventListener("click", async () => {
  if (currentState === "connecting") return; // a connect is already underway
  settings = readSettingsFromForm();
  saveSettings(settings);
  if (missingServerUrl()) { promptServerUrl(); return; } // keep settings open
  settingsModal.close();
  // Grab the AudioContext NOW, inside the click gesture — teardown() awaits, and
  // creating it afterwards would fall outside the gesture (silent on iOS).
  const audioContext = createResumedAudioContext();
  try {
    if (client) await teardown();
    await doStart(audioContext);
  } catch (err) {
    await handleStartError(err);
  }
});

circleBtn.addEventListener("click", async () => {
  try {
    if (currentState === "idle" || currentState === "error") {
      if (missingServerUrl()) { promptServerUrl(); return; }
      await doStart();
    }
  } catch (err) {
    await handleStartError(err);
  }
});

/** A failed start is either the daily limit (show the modal, return to idle) or
 *  a real fault (surface it). doStart already closed any orphan AudioContext.
 *  @param {any} err */
async function handleStartError(err) {
  if (err && err.code === "limit") {
    await teardown();
    account.showLimit(err.tier);
    return;
  }
  // The user left the queue (close() aborted the wait): teardown already reset
  // the UI to idle, so there's nothing to report.
  if (err && err.code === "aborted") return;
  // The whole waiting line is full: a warm, reassuring modal rather than an error.
  if (err && err.code === "queue-full") {
    await teardown();
    account.showBusy();
    return;
  }
  // Our place lapsed (ticket reaped, or the join window ran out). Recoverable, not
  // a fault: land on the retry state with a kind, plain-language reason.
  if (err && (err.code === "queue-expired" || err.code === "join-expired")) {
    await teardown();
    setState("error");
    setCaption(
      err.code === "join-expired"
        ? "Your spot expired. Tap to rejoin."
        : "That took a while. Tap to rejoin.",
      "error",
    );
    return;
  }
  onFatalError(err);
}

micBtn.addEventListener("click", () => {
  if (!micStream || !client) return;
  micMuted = !micMuted;
  for (const track of micStream.getAudioTracks()) {
    track.enabled = !micMuted;
  }
  client.setMuted(micMuted);
  micBtn.classList.toggle("muted", micMuted);
  micBtn.setAttribute("aria-label", micMuted ? "Unmute" : "Mute");
  micBtn.title = micMuted ? "Unmute" : "Mute";
});

stopBtn.addEventListener("click", async () => {
  await teardown();
});

// "Leave queue": tear down the pending connect (aborts the poll wait) and drop
// our place in line. Same teardown path as stopping a live call.
leaveQueueBtn.addEventListener("click", async () => {
  await teardown();
});

// "Join now": accept the held slot. The click is a user gesture, so the client
// re-resumes the AudioContext here (iOS) before dialing.
joinQueueBtn.addEventListener("click", () => {
  stopJoinCountdown();
  if (client) client.join();
});

const MIC_CONSTRAINTS = {
  audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
};

/** Prompt for mic permission up front, then immediately release the tracks so no
 *  recording indicator lingers during a queue wait. Throws a friendly error if the
 *  user denies. */
async function primeMicPermission() {
  try {
    const s = await navigator.mediaDevices.getUserMedia(MIC_CONSTRAINTS);
    for (const track of s.getTracks()) track.stop();
  } catch (err) {
    throw new Error(
      `Microphone access denied${err instanceof Error ? `: ${err.message}` : ""}`,
    );
  }
}

/** Acquire the live capture stream once a slot is granted. Permission was primed
 *  in the tap gesture, so this is silent. Stored module-side for mute + teardown. */
async function acquireMicStream() {
  return navigator.mediaDevices.getUserMedia(MIC_CONSTRAINTS);
}

/** @param {number} position Update the queued caption ("You're #N in line"). */
function onQueuePosition(position) {
  const n = Number(position) || 0;
  setCaption(n > 0 ? `You're #${n} in line` : "Finding you a spot…", "muted");
}

// ── "Your turn" join countdown ──────────────────────────────────────────────
// While a slot is held for us, show how long is left to accept it. The client's
// join gate expires just before the load balancer reclaims the slot.
let joinCountdownTimer = 0;

/** @param {number} sec */
function startJoinCountdown(sec) {
  stopJoinCountdown();
  let left = Math.max(0, Math.floor(sec));
  const paint = () => {
    joinQueueBtn.textContent = left > 0 ? `Join now (${left}s)` : "Join now";
  };
  paint();
  joinCountdownTimer = window.setInterval(() => {
    left -= 1;
    if (left <= 0) {
      stopJoinCountdown();
      joinQueueBtn.textContent = "Join now";
      return;
    }
    paint();
  }, 1000);
}

function stopJoinCountdown() {
  if (joinCountdownTimer) {
    clearInterval(joinCountdownTimer);
    joinCountdownTimer = 0;
  }
}

function staleStartError() {
  const error = /** @type {Error & { code?: string }} */ (new Error("A newer conversation start replaced this attempt"));
  error.code = "aborted";
  return error;
}

/**
 * Abort an obsolete start without disturbing a replacement client. If the old
 * attempt already owns a client, its own close path releases mic/context/socket.
 * @param {number} attempt @param {AudioContext | null} audioContext
 * @param {S2sWsRealtimeClient | null} [candidate]
 */
async function ensureCurrentStartAttempt(attempt, audioContext, candidate = null) {
  if (startAttempts.isCurrent(attempt)) return;
  if (candidate) {
    try { await candidate.close(); } catch { /* best-effort stale cleanup */ }
  } else if (audioContext) {
    try { await audioContext.close(); } catch { /* best-effort stale cleanup */ }
  }
  throw staleStartError();
}

/** Await a start preflight and suppress any stale success or failure. */
async function awaitStartPreflight(attempt, preflight, audioContext) {
  let value;
  try {
    value = await preflight;
  } catch (error) {
    await ensureCurrentStartAttempt(attempt, audioContext);
    throw error;
  }
  await ensureCurrentStartAttempt(attempt, audioContext);
  return value;
}

/**
 * Start a conversation. Pass a pre-created AudioContext when the caller already
 * made one inside the tap/click gesture (required on iOS); otherwise one is
 * created here, which is still inside the gesture for a direct orb tap.
 * @param {AudioContext | null} [audioContext]
 */
async function doStart(audioContext = null) {
  // Claim synchronously, before any readiness/network await. This makes a fast
  // second tap invalidate the older preflight instead of allowing two mics or
  // WebSockets to race into the same page state.
  const startAttempt = startAttempts.claim();
  setState("connecting");
  setCaption("Asking for mic…", "muted");
  // Resolve the target before touching mic/audio so a misconfiguration (e.g.
  // direct mode with no URL) fails fast with a clear message.
  const target = connectionTarget();

  // Create + resume the AudioContext SYNCHRONOUSLY, still inside the gesture.
  // iOS Safari only starts an AudioContext from a user gesture; if we waited
  // until after the preflight awaits below, it would stay suspended and the
  // whole pipeline would be silent.
  if (!audioContext) audioContext = createResumedAudioContext();

  let initialTtsTuning = null;
  try {
    initialTtsTuning = await awaitStartPreflight(startAttempt, assertTtsBackendReady(), audioContext);
    await awaitStartPreflight(startAttempt, assertModelEndpointReady(), audioContext);
  } catch (err) {
    if (audioContext) void audioContext.close().catch(() => {});
    throw err;
  }

  chat.clear();
  chat.reset();
  pipelineMetrics = [];
  backendRuntime = null;
  clearTimeout(backendMetricTimer);
  clearTimeout(backendRuntimeTimer);
  setDiagnosticWarning("backend-version");
  setDiagnosticWarning("backend-metrics");
  // Prime the mic permission now (get the prompt out of the way up front), then
  // release it. The real capture stream is acquired only once a slot is granted
  // (see acquireMicStream), so the mic 'in use' indicator never lights while we
  // sit in the queue. Permission persists, so the later acquire is silent.
  try {
    await awaitStartPreflight(startAttempt, primeMicPermission(), audioContext);
  } catch (err) {
    if (audioContext) void audioContext.close().catch(() => {});
    throw err;
  }

  // The webcam is started on arrival (autoStartCamera), so nothing to do here;
  // a still-pending grant just means the snapshot tool isn't ready yet.

  const c = new S2sWsRealtimeClient({
    ...target,
    voice: settings.voice,
    instructions: effectiveInstructions(),
    acquireMic: async () => {
      const stream = await acquireMicStream();
      if (!startAttempts.isCurrent(startAttempt)) {
        for (const track of stream.getTracks()) track.stop();
        throw staleStartError();
      }
      // Do not publish an obsolete stream globally: a replacement start may
      // already own the live capture while this delayed permission request
      // resolves. The stale path above stops only its own stream.
      micStream = stream;
      return stream;
    },
    tools: activeToolDefs(),
    noiseGate: gateParams(settings.noiseGate),
    echoGuard: settings.echoGuard,
    echoCalibrations: settings.echoCalibrations,
    pipelineConfig: {
      full_buffer_tts: settings.fullBufferTts,
      live_transcription: settings.liveTranscript,
      max_response_tokens: settings.maxResponseTokens,
      history_compaction: normalizeHistoryCompaction(settings.historyCompaction),
      tts_backend: settings.ttsBackend,
      ...(normalizeTtsProvider(settings.ttsBackend) === AUDIO_CPP_PROVIDER
        ? { tts_tuning: initialTtsTuning }
        : {}),
      model_endpoint: modelEndpointConfig(settings),
    },
    playbackConfig: activePlaybackConfig(settings.ttsBackend),
    ...(audioContext ? { audioContext } : {}),
  });
  client = c;

  c.addEventListener("queue", (e) => {
    if (client !== c) return;
    const { position, queueId } = /** @type {CustomEvent<{ position: number; queueId: string }>} */ (e).detail;
    if (queueId) queuedTicketId = queueId;
    onQueuePosition(position);
  });

  c.addEventListener("ready-to-join", (e) => {
    if (client !== c) return;
    const { info, expiresSec } = /** @type {CustomEvent<{ info: import("./ws/s2s-ws-client.js").WsSessionInfo; expiresSec: number }>} */ (e).detail;
    // A slot is held for us. We're out of the queue now, so drop the ticket ref.
    // Track the granted session id already so that leaving (or letting the timer
    // lapse) refunds the budget the server reserved at claim, even before we dial.
    queuedTicketId = "";
    if (info?.sessionId) {
      trackedSessionId = info.sessionId;
      trackedTier = info.tier || "anon";
    }
    startJoinCountdown(expiresSec);
  });

  c.addEventListener("status", (e) => {
    if (client !== c) return;
    const detail = /** @type {CustomEvent<{ status: string }>} */ (e).detail;
    onClientStatus(detail.status);
  });
  c.addEventListener("transcript", (e) => {
    if (client !== c) return;
    const d = /** @type {CustomEvent<{ role: "user" | "assistant"; text: string; partial: boolean; itemId?: string; responseId?: string }>} */ (e).detail;
    chat.onTranscript(d, { showUserBubble: settings.liveTranscript });
  });

  c.addEventListener("response-finished", (e) => {
    if (client !== c) return;
    const detail = /** @type {CustomEvent<{ responseId: string; status: string; audible?: boolean; transcript?: string; responseEpoch?: number | null; committed?: boolean }>} */ (e).detail;
    chat.onResponseFinished(detail);
  });

  c.addEventListener("toolcall", (e) => {
    if (client !== c) return;
    const { name, arguments: args, callId } = /** @type {CustomEvent<{ name: string; arguments: string; callId: string }>} */ (e).detail;
    chat.onToolCall(name, args, callId);
    // Execute the tool, then push it to the conversation once the result is in,
    // so the toggle shows both the call input and its output together.
    void runTool(c, name, args, callId).then(({ output, image }) => {
      if (client !== c) return;
      chat.onToolResult(name, args, output, image, callId);
    });
  });
  c.addEventListener("error", (e) => {
    if (client !== c) return;
    const detail = /** @type {CustomEvent<{ error: unknown }>} */ (e).detail;
    onFatalError(detail.error);
  });
  c.addEventListener("server-error", (e) => {
    if (client !== c) return;
    // Non-fatal: the backend reported an error mid-session. Log it, keep the
    // socket and the conversation alive (the model can recover on its own).
    const detail = /** @type {CustomEvent<{ error: unknown }>} */ (e).detail;
    const msg = detail.error instanceof Error ? detail.error.message : String(detail.error);
    console.warn("[main] server error (non-fatal):", msg);
  });
  c.addEventListener("session", (e) => {
    if (client !== c) return;
    const info = /** @type {CustomEvent<{ info: import("./ws/s2s-ws-client.js").WsSessionInfo }>} */ (e).detail.info;
    console.log("[ws] session created:", info.sessionId);
    backendRuntimeTimer = window.setTimeout(() => {
      if (!backendRuntime) {
        setDiagnosticWarning("backend-version", "Backend runtime identity is missing. Restart the speech-to-speech backend.");
      }
    }, 1500);
    // A slot was granted — we're out of the queue; drop the ticket reference so
    // teardown doesn't try to leave a line we already left.
    queuedTicketId = "";
    // A metered tier (anon / free): heartbeat so the server can extend the
    // reservation and tell us when the daily budget runs out. PRO isn't limited.
    if (info.limited && info.sessionId) {
      trackedSessionId = info.sessionId;
      trackedTier = info.tier || "anon";
      startHeartbeat(info.heartbeatSec || 5);
    }
  });
  c.addEventListener("input-level", (e) => {
    if (client !== c) return;
    const { rms } = /** @type {CustomEvent<{ rms: number }>} */ (e).detail;
    paintInputLevel(rms);
  });
  c.addEventListener("echo-status", (e) => {
    if (client !== c) return;
    paintEchoStatus(/** @type {CustomEvent<any>} */ (e).detail);
  });
  c.addEventListener("pipeline-metric", (e) => {
    if (client !== c) return;
    const metric = /** @type {CustomEvent<any>} */ (e).detail;
    if (metric.source === "backend") {
      clearTimeout(backendMetricTimer);
      setDiagnosticWarning("backend-metrics");
    }
    addPipelineMetric(metric);
  });
  c.addEventListener("backend-runtime", (e) => {
    if (client !== c) return;
    backendRuntime = /** @type {CustomEvent<any>} */ (e).detail;
    clearTimeout(backendRuntimeTimer);
    if (backendRuntime.api_version !== EXPECTED_BACKEND_API_VERSION) {
      setDiagnosticWarning("backend-version", `Backend restart required (API ${backendRuntime.api_version ?? "missing"}, expected ${EXPECTED_BACKEND_API_VERSION}).`);
    } else {
      setDiagnosticWarning("backend-version");
    }
    addPipelineMetric({ stage: "context", status: "runtime", source: "backend", detail: backendRuntime.context || {} });
  });
  c.addEventListener("local-pipeline-updated", (e) => {
    if (client !== c) return;
    const config = /** @type {CustomEvent<any>} */ (e).detail;
    if (config?.history_compaction) {
      acknowledgedHistoryCompaction = normalizeHistoryCompaction(config.history_compaction);
      settings.historyCompaction = acknowledgedHistoryCompaction;
      syncHistoryCompactionControls();
      renderHistoryCompactionDiagnostics();
    }
    if (config?.model_endpoint) {
      const endpoint = config.model_endpoint;
      localPipeline = localPipeline || {};
      localPipeline.gemma = {
        reachable: true,
        model: endpoint.advertised_model || endpoint.model,
        baseUrl: endpoint.base_url,
        provider: endpoint.provider,
        contextWindow: endpoint.context_window,
      };
      renderLocalPipeline();
    }
  });
  c.addEventListener("turn-state", (e) => {
    if (client !== c) return;
    const status = /** @type {CustomEvent<any>} */ (e).detail.status;
    if (status !== "speech_stopped") return;
    chat.onUserTurnPending();
    clearTimeout(backendMetricTimer);
    backendMetricTimer = window.setTimeout(() => {
      setDiagnosticWarning("backend-metrics", "No backend metrics arrived for this turn. The backend may be stale or disconnected.");
    }, 2500);
  });

  try {
    await c.connect();
    await ensureCurrentStartAttempt(startAttempt, audioContext, c);
  } catch (err) {
    if (!startAttempts.isCurrent(startAttempt)) {
      await ensureCurrentStartAttempt(startAttempt, audioContext, c);
    }
    // The grant can be refused (402 → limit) or the dial can fail. In LB mode
    // the AudioContext hasn't been adopted by the client yet (the session POST
    // runs first), so close the one we created here to avoid leaking it.
    if (audioContext) void audioContext.close().catch(() => {});
    throw err;
  }
}

// ── Conversation-time heartbeat ─────────────────────────────────────────────

/** Ping the server every `sec` seconds so it can meter the live session; when
 *  it reports the daily budget is spent, cut the call and show the limit modal.
 *  @param {number} sec */
function startHeartbeat(sec) {
  stopHeartbeat();
  heartbeatTimer = window.setInterval(async () => {
    if (!trackedSessionId) return;
    try {
      const res = await fetch("api/session/heartbeat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sessionId: trackedSessionId }),
        keepalive: true,
      });
      const json = await res.json().catch(() => ({}));
      if (json.expired) await onLimitReached();
    } catch (err) {
      // A transient network blip shouldn't kill the call; the next tick retries.
      if (DEBUG) console.debug("[ui] heartbeat failed:", err);
    }
  }, Math.max(1, sec) * 1000);
}

function stopHeartbeat() {
  if (heartbeatTimer) {
    clearInterval(heartbeatTimer);
    heartbeatTimer = 0;
  }
}

/** The server cut the live session: tear down and explain why. */
async function onLimitReached() {
  const tier = trackedTier;
  stopHeartbeat();
  await teardown();
  account.showLimit(tier);
}

/** Tell the server a session ended so it reconciles + refunds the unused chunk.
 *  Uses sendBeacon so it still fires when the tab is closing. */
function endTrackedSession() {
  if (!trackedSessionId) return;
  const body = JSON.stringify({ sessionId: trackedSessionId });
  try {
    const blob = new Blob([body], { type: "application/json" });
    if (!navigator.sendBeacon("api/session/end", blob)) {
      void fetch("api/session/end", {
        method: "POST", headers: { "Content-Type": "application/json" }, body, keepalive: true,
      }).catch(() => {});
    }
  } catch {
    // Best-effort; the server sweep reaps the session anyway.
  }
  trackedSessionId = "";
  trackedTier = "";
}

/** Leave the waiting queue so the LB frees our place. sendBeacon so it still
 *  fires on tab close; the LB also reaps the ticket on TTL as a backstop. */
function endQueueTicket() {
  if (!queuedTicketId) return;
  const body = JSON.stringify({ queueId: queuedTicketId });
  try {
    const blob = new Blob([body], { type: "application/json" });
    if (!navigator.sendBeacon("api/queue/end", blob)) {
      void fetch("api/queue/end", {
        method: "POST", headers: { "Content-Type": "application/json" }, body, keepalive: true,
      }).catch(() => {});
    }
  } catch {
    // Best-effort; the LB reaps the ticket on TTL anyway.
  }
  queuedTicketId = "";
}

/** @param {string} status */
function onClientStatus(status) {
  switch (status) {
    case "creating-session":
    case "connecting":
      setState("connecting");
      break;
    case "queued":
      setState("queued");
      break;
    case "your-turn":
      setState("your-turn");
      break;
    case "connected":
      setState("listening");
      break;
    case "user-speaking":
      setState("user-speaking");
      break;
    case "processing":
      setState("processing");
      break;
    case "ai-speaking":
      setState("ai-speaking");
      break;
    case "closed":
      // teardown() will move us to idle
      break;
    case "error":
      setState("error");
      break;
  }
}

async function teardown() {
  // Invalidate a queued/connecting start before asynchronous resource cleanup;
  // its next preflight checkpoint closes itself rather than reviving this UI.
  startAttempts.invalidate();
  stopHeartbeat();
  stopJoinCountdown();
  endTrackedSession();
  endQueueTicket();
  chat.reset({ dismiss: true });
  const closingClient = client;
  client = null;
  if (micStream) {
    for (const track of micStream.getTracks()) track.stop();
    micStream = null;
  }
  if (closingClient) {
    try {
      await closingClient.close();
    } catch (err) {
      console.warn("[main] error closing client:", err);
    }
  }
  // The webcam is independent of the call lifecycle (it runs while the user is
  // on the page), so we leave it on here — only the camera toggle stops it.
  micMuted = false;
  micBtn.classList.remove("muted");
  setState("idle");
  // Refresh the chip's remaining-today after the budget moved.
  if (limiterOn) void account.refresh();
}

/** @param {unknown} err */
function onFatalError(err) {
  console.error("[main] fatal:", err);
  const message = err instanceof Error ? err.message : String(err);
  const showError = () => {
    setState("error");
    setCaption(truncateError(message), "error");
  };
  showError();
  // teardown() intentionally returns normal stops to Idle. A fatal startup
  // failure must still release every resource, but it must remain visible once
  // cleanup completes instead of silently landing back on the idle orb.
  void teardown().then(showError, showError);
}

async function initializeApp() {
  // Restore the managed backend/voice selection before the first inventory
  // request. Starting both operations concurrently allowed a Faster request
  // to win while the UI later displayed audio.cpp, leaving a foreign or empty
  // clone list attached to the selected provider.
  try {
    await restorePersistentSettings();
    syncHistoryCompactionControls();
    renderHistoryCompactionDiagnostics();
    await fetchConfig();
    await refreshLocalIdentity({ retry: true, resetRetry: true });
    await refreshCandidateTuningProfiles().catch((error) => console.warn("candidate tuning refresh failed", error));
    await fetchVoiceProfiles();
    // The initial render awaits the uncached selected-provider identity above.
  } catch (error) {
    console.warn("initial local identity refresh failed", error);
    if (!localPipeline) {
      localPipeline = {
        mode: "unavailable", gemma: { reachable: false },
        tts: { backend: settings.ttsBackend, reachable: false, displayName: settings.ttsBackend },
        vad: {}, tools: {}, unavailableReason: "Initial local status unavailable",
      };
      renderLocalPipeline();
    }
  } finally {
    // Do not finish the initial shell until the selected-provider identity
    // probe settled into either a live result or a truthful unavailable state.
    document.body.classList.remove("booting");
  }
}

setState("idle");
chat.renderEmptyState();
initGateArc();
void initializeApp();
// Restore an already-enabled camera after a reload. Browsers only prompt if the
// user has not yet made a permission choice.
void autoStartCamera();
void watchCameraPermission();

// Reconcile a live session if the tab is closed/hidden mid-call (no teardown).
window.addEventListener("pagehide", () => { endTrackedSession(); endQueueTicket(); });
window.addEventListener("focus", () => { void refreshLocalIdentity({ retry: true, resetRetry: true }); });
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) void refreshLocalIdentity({ retry: true, resetRetry: true });
});
