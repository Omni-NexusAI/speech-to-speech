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

import { S2sWsRealtimeClient } from "./ws/s2s-ws-client.js?v=12-adaptive-v3";
import { $, truncateError, DEBUG } from "./ui/dom.js";
import { ChatView } from "./ui/chat.js";
import { Account } from "./ui/account.js";

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
  diagnostics: "s2s.ws.diagnostics",
  diagnosticsGeometry: "s2s.ws.diagnosticsGeometry",
  fullBufferTts: "s2s.ws.fullBufferTts",
  liveTranscript: "s2s.ws.liveTranscript",
  maxResponseTokens: "s2s.ws.maxResponseTokens",
  ttsBackend: "s2s.ws.ttsBackend",
  modelProvider: "s2s.ws.modelProvider",
  modelUrl: "s2s.ws.modelUrl",
  modelName: "s2s.ws.modelName",
  modelApiKey: "s2s.ws.modelApiKey",
};

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
  const storedEchoGuard = localStorage.getItem(STORAGE_KEYS.echoGuard);
  const echoGuardVersion = localStorage.getItem(STORAGE_KEYS.echoGuardVersion);
  const echoGuard =
    echoGuardVersion === "2" && ["off", "adaptive", "strict"].includes(storedEchoGuard || "")
      ? storedEchoGuard
      : storedEchoGuard === "strict"
        ? "strict"
        : "adaptive";
  if (echoGuardVersion !== "2") {
    localStorage.setItem(STORAGE_KEYS.echoGuard, echoGuard);
    localStorage.setItem(STORAGE_KEYS.echoGuardVersion, "2");
  }
  return {
    directUrl: localStorage.getItem(STORAGE_KEYS.directUrl) || "http://127.0.0.1:8765",
    voice: localStorage.getItem(STORAGE_KEYS.voice) || DEFAULT_VOICE,
    instructions: localStorage.getItem(STORAGE_KEYS.instructions) || DEFAULT_INSTRUCTIONS,
    noiseGate: loadGateThreshold(),
    echoGuard,
    fullBufferTts: localStorage.getItem(STORAGE_KEYS.fullBufferTts) === "1",
    liveTranscript: localStorage.getItem(STORAGE_KEYS.liveTranscript) === "1",
    maxResponseTokens: Math.min(1024, Math.max(64, Number(localStorage.getItem(STORAGE_KEYS.maxResponseTokens)) || 384)),
    ttsBackend: localStorage.getItem(STORAGE_KEYS.ttsBackend) || "faster",
    modelProvider: localStorage.getItem(STORAGE_KEYS.modelProvider) || "local",
    modelUrl: localStorage.getItem(STORAGE_KEYS.modelUrl) || "",
    modelName: localStorage.getItem(STORAGE_KEYS.modelName) || "",
    modelApiKey: localStorage.getItem(STORAGE_KEYS.modelApiKey) || "",
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
  localStorage.setItem(STORAGE_KEYS.directUrl, s.directUrl);
  localStorage.setItem(STORAGE_KEYS.voice, s.voice);
  localStorage.setItem(STORAGE_KEYS.instructions, s.instructions);
  localStorage.setItem(STORAGE_KEYS.noiseGate, String(s.noiseGate));
  localStorage.setItem(STORAGE_KEYS.echoGuard, s.echoGuard);
  localStorage.setItem(STORAGE_KEYS.echoGuardVersion, "2");
  localStorage.setItem(STORAGE_KEYS.fullBufferTts, s.fullBufferTts ? "1" : "0");
  localStorage.setItem(STORAGE_KEYS.liveTranscript, s.liveTranscript ? "1" : "0");
  localStorage.setItem(STORAGE_KEYS.maxResponseTokens, String(s.maxResponseTokens));
  localStorage.setItem(STORAGE_KEYS.ttsBackend, s.ttsBackend);
  localStorage.setItem(STORAGE_KEYS.modelProvider, s.modelProvider);
  localStorage.setItem(STORAGE_KEYS.modelUrl, s.modelUrl);
  localStorage.setItem(STORAGE_KEYS.modelName, s.modelName);
  localStorage.setItem(STORAGE_KEYS.modelApiKey, s.modelApiKey);
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

/** @type {AppState} */
let currentState = "idle";
let settings = loadSettings();
/** @type {Array<{ id: string; voice: string; name: string; created_at?: string }>} */
let voiceProfiles = [];
let defaultVoice = DEFAULT_VOICE;
let localPipeline = null;
/** @type {Record<string, any>} */
let ttsBackendStatuses = {};
let diagnosticsOpen = localStorage.getItem(STORAGE_KEYS.diagnostics) === "1";
/** @type {Array<any>} */
let pipelineMetrics = [];
const EXPECTED_UI_API_VERSION = 12;
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
  renderVoiceOptions();
  inputVoice.value = settings.voice;
  renderTtsBackendOptions();
  inputTtsBackend.value = settings.ttsBackend;
  inputModelProvider.value = settings.modelProvider;
  inputModelUrl.value = settings.modelUrl;
  inputModelName.value = settings.modelName;
  inputModelApiKey.value = settings.modelApiKey;
  syncModelProviderUi();
  inputInstructions.value = settings.instructions;
  inputEchoGuard.value = settings.echoGuard;
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
    option.value = settings.voice || defaultVoice;
    option.textContent = "No Base clone profiles found";
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
    saveSettings(settings);
  }

  for (const profile of voiceProfiles) {
    const option = document.createElement("option");
    option.value = profile.voice;
    option.textContent = `${profile.name} (${profile.voice})`;
    option.selected = profile.voice === settings.voice;
    inputVoice.append(option);
  }
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
  for (const id of ["faster", "groxaxo"]) {
    const status = ttsBackendStatuses[id];
    const option = document.createElement("option");
    option.value = id;
    const name = id === "faster" ? "FasterQwen3TTS" : "Groxaxo candidate";
    const model = status?.currentModel || "no Base model";
    option.textContent = `${name} - ${model} (${status?.ready ? "ready" : "unavailable"})`;
    inputTtsBackend.append(option);
  }
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
}

async function refreshLocalPipeline() {
  try {
    const res = await fetch("api/local-pipeline");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    localPipeline = await res.json();
    renderLocalPipeline();
  } catch (err) {
    console.warn("[ui] failed to load local pipeline status:", err);
  }
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
}

async function assertTtsBackendReady() {
  await refreshTtsBackends();
  const status = ttsBackendStatuses[settings.ttsBackend];
  if (status?.ready) return;
  if (settings.ttsBackend === "groxaxo") {
    throw new Error("Groxaxo is unavailable or has no 0.6B-Base/1.7B-Base model loaded in Voice Studio.");
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
  const selectedTtsName = settings.ttsBackend === "groxaxo" ? "Groxaxo candidate" : "FasterQwen3TTS";
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
  syncToolsUi();
  syncConnectionUi();
}

async function fetchVoiceProfiles() {
  try {
    const res = await fetch("api/qwen3/voices");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const json = await res.json();
    defaultVoice = json.defaultVoice || DEFAULT_VOICE;
    voiceProfiles = Array.isArray(json.voices) ? json.voices : [];
  } catch (err) {
    console.warn("[ui] failed to load Qwen3 clone voices:", err);
    defaultVoice = DEFAULT_VOICE;
    voiceProfiles = [];
  }
  renderVoiceOptions();
}

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
  return {
    directUrl: allowDirect ? inputLbUrl.value.trim() : settings.directUrl,
    voice: inputVoice.value || defaultVoice,
    instructions: inputInstructions.value.trim() || DEFAULT_INSTRUCTIONS,
    noiseGate: readGateThreshold(),
    echoGuard: ["off", "adaptive", "strict"].includes(inputEchoGuard.value) ? inputEchoGuard.value : "adaptive",
    fullBufferTts: inputFullBufferTts.checked,
    liveTranscript: inputLiveTranscript.checked,
    maxResponseTokens: Math.min(1024, Math.max(64, Number(inputMaxResponseTokens.value) || 384)),
    ttsBackend: inputTtsBackend.value || "faster",
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

settingsForm.addEventListener("submit", (event) => {
  const submitter = /** @type {HTMLButtonElement | null} */ ((/** @type {SubmitEvent} */ (event)).submitter);
  if (submitter?.value !== "save") return;

  settings = readSettingsFromForm();
  saveSettings(settings);

  // Voice + instructions can apply to a live session without reconnecting; a
  // changed connection URL only takes effect on the next restart.
  if (client && LIVE_STATES.has(currentState)) {
    client.updateSession({ voice: settings.voice, instructions: effectiveInstructions() });
    client.updateLocalPipeline({
      full_buffer_tts: settings.fullBufferTts,
      live_transcription: settings.liveTranscript,
      max_response_tokens: settings.maxResponseTokens,
    });
    client.setEchoGuard(settings.echoGuard);
  }
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
  micStream = await navigator.mediaDevices.getUserMedia(MIC_CONSTRAINTS);
  return micStream;
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

/**
 * Start a conversation. Pass a pre-created AudioContext when the caller already
 * made one inside the tap/click gesture (required on iOS); otherwise one is
 * created here, which is still inside the gesture for a direct orb tap.
 * @param {AudioContext | null} [audioContext]
 */
async function doStart(audioContext = null) {
  // Resolve the target before touching mic/audio so a misconfiguration (e.g.
  // direct mode with no URL) fails fast with a clear message.
  const target = connectionTarget();
  await assertTtsBackendReady();
  await assertModelEndpointReady();

  chat.clear();
  chat.reset();
  pipelineMetrics = [];
  backendRuntime = null;
  clearTimeout(backendMetricTimer);
  clearTimeout(backendRuntimeTimer);
  setDiagnosticWarning("backend-version");
  setDiagnosticWarning("backend-metrics");
  setState("connecting");
  setCaption("Asking for mic…", "muted");

  // Create + resume the AudioContext SYNCHRONOUSLY, still inside the gesture.
  // iOS Safari only starts an AudioContext from a user gesture; if we waited
  // until after the getUserMedia / session-creation awaits below, it would stay
  // suspended and the whole pipeline would be silent.
  if (!audioContext) audioContext = createResumedAudioContext();

  // Prime the mic permission now (get the prompt out of the way up front), then
  // release it. The real capture stream is acquired only once a slot is granted
  // (see acquireMicStream), so the mic 'in use' indicator never lights while we
  // sit in the queue. Permission persists, so the later acquire is silent.
  try {
    await primeMicPermission();
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
    acquireMic: acquireMicStream,
    tools: activeToolDefs(),
    noiseGate: gateParams(settings.noiseGate),
    echoGuard: settings.echoGuard,
    pipelineConfig: {
      full_buffer_tts: settings.fullBufferTts,
      live_transcription: settings.liveTranscript,
      max_response_tokens: settings.maxResponseTokens,
      tts_backend: settings.ttsBackend,
      model_endpoint: modelEndpointConfig(settings),
    },
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
    const detail = /** @type {CustomEvent<{ responseId: string; status: string; audible?: boolean; transcript?: string }>} */ (e).detail;
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
  } catch (err) {
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
  setState("error");
  const message = err instanceof Error ? err.message : String(err);
  setCaption(truncateError(message), "error");
  void teardown().catch(() => {
    setState("error");
    setCaption(truncateError(message), "error");
  });
}

setState("idle");
chat.renderEmptyState();
initGateArc();
void fetchConfig();
if (settings.modelProvider === "local") void refreshLocalPipeline();
// Restore an already-enabled camera after a reload. Browsers only prompt if the
// user has not yet made a permission choice.
void autoStartCamera();
void watchCameraPermission();

// Reconcile a live session if the tab is closed/hidden mid-call (no teardown).
window.addEventListener("pagehide", () => { endTrackedSession(); endQueueTicket(); });

requestAnimationFrame(() => {
  document.body.classList.remove("booting");
});
