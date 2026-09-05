// @ts-check
/**
 * Minimal WebSocket client for the Hugging Face speech-to-speech load balancer.
 *
 * Two-step handshake (same /session route as the WebRTC client):
 *
 *   1. POST `<lb_url>/session` -> JSON `{ connect_url: wss://<compute>/v1/realtime?session_token=<JWT>, ... }`
 *   2. Open a WebSocket directly on `connect_url` (no rewrite, unlike the WebRTC client).
 *
 * Once the socket is open we follow the OpenAI Realtime GA WebSocket
 * protocol:
 *
 *   - Server pushes `session.created` immediately after upgrade.
 *   - We send `session.update` (GA schema: `session.audio.{input,output}`,
 *     `session.output_modalities`, ...).
 *   - We stream mic audio as PCM16 16 kHz mono base64 chunks via
 *     `input_audio_buffer.append`.
 *   - The server pushes `response.output_audio.delta` as PCM16 mono base64.
 *     The default/Faster/Groxaxo transport remains 16 kHz; the isolated
 *     audio.cpp candidate acknowledges its model-native 24 kHz clock before
 *     playback begins.
 *
 * Audio is handled internally via two AudioWorklet processors so the
 * client owns the full mic-in / speaker-out pipeline. The main app only
 * sees high-level lifecycle events (`status`, `transcript`, `error`,
 * `session`), the same shape as the WebRTC client.
 *
 * @typedef {"idle" | "creating-session" | "queued" | "your-turn" | "connecting" |
 *           "connected" | "user-speaking" | "processing" | "ai-speaking" |
 *           "closed" | "error"
 * } WsStatus
 *
 * @typedef {Object} WsSessionInfo
 * @property {string} sessionId
 * @property {string} connectUrl
 * @property {string} websocketUrl
 * @property {string} sessionToken
 * @property {number} pendingTimeoutS
 * @property {string} [tier] Login tier from the session proxy ("anon"|"free"|"pro").
 * @property {boolean} [limited] Whether this session is metered (heartbeat needed).
 * @property {number} [heartbeatSec] Suggested heartbeat cadence in seconds.
 * @property {number} [remainingSec] Daily budget left after this grant (display).
 *
 * @typedef {Object} WsClientOptions
 * @property {string} [sessionUrl] URL to POST for the session handshake (returns
 *   `{ connect_url, ... }`). Usually a same-origin proxy like `api/session` so the
 *   load-balancer address stays server-side. Provide this OR `directUrl`.
 * @property {string} [loadBalancerUrl] Load-balancer base URL. Legacy/direct
 *   alternative to `sessionUrl`: the client POSTs `<lb>/session` itself. Prefer
 *   `sessionUrl` so the LB address isn't exposed to the browser.
 * @property {string} [directUrl] Full WebSocket URL of an s2s realtime endpoint
 *   (e.g. `ws://localhost:8080/v1/realtime`). When set, the client skips the
 *   session POST and dials it directly — no load balancer in between.
 * @property {string} voice
 * @property {string} instructions
 * @property {MediaStream} [micStream] Live mic stream. Provide this OR `acquireMic`.
 * @property {() => Promise<MediaStream>} [acquireMic] Lazily obtain the mic stream,
 *   called only once a session is actually granted (after any queue wait). Lets the
 *   caller prime mic permission up front but not hold the mic 'in use' indicator on
 *   while waiting in line. Ignored if `micStream` is already set.
 * @property {AudioContext} [audioContext] Pre-created (and resumed) context.
 *   iOS Safari only lets an AudioContext start from within a user gesture, so
 *   the caller creates/resumes it synchronously on the orb tap and hands it
 *   here; otherwise it stays suspended (silent) after the mic/session awaits.
 * @property {ToolDef[]} [tools] Function tools declared to the backend in the
 *   initial `session.update`. The model decides when to call them; the caller
 *   executes and replies via `sendToolOutput` + `requestResponse`.
 * @property {NoiseGate} [noiseGate] Client-side noise gate applied to the mic
 *   before it's sent. Tunable live via `setNoiseGate`.
 * @property {EchoGuardMode} [echoGuard] Playback-reference echo suppression.
 * @property {Record<string, EchoCalibration>} [echoCalibrations] Saved AEC3
 *   calibration indexed by microphone/output-device pair.
 * @property {Record<string, any>} [pipelineConfig] Conversation-scoped model and TTS routing.
 * @property {PlaybackConfig} [playbackConfig] Browser-only snapshot of the
 *   selected provider's validated delivery mode and resolved tuning profile.
 *   It is applied only when the corresponding pipeline update is acknowledged.
 *
 * @typedef {Object} NoiseGate
 * @property {boolean} enabled
 * @property {number} thresholdDb Open threshold in dBFS (e.g. -45).
 *
 * @typedef {"native" | "adaptive" | "strict"} EchoGuardMode
 *
 * @typedef {Object} EchoCalibration
 * @property {number} delayMs
 * @property {number} suppressionStrength
 * @property {number} leakageThreshold
 * @property {number} doubleTalkSensitivity
 * @property {number} [echoTailMs]
 *
 * @typedef {Object} PlaybackConfig
 * @property {string} [provider]
 * @property {string} [profileId]
 * @property {number | null} [profileRevision]
 * @property {boolean} [nativeStreaming]
 * @property {number} [resolvedPrimeMs]
 * @property {"adaptive" | "fast-start"} [continuityMode]
 * @property {number} [configToken]
 *
 * @typedef {Object} ToolDef
 * @property {"function"} type
 * @property {string} name
 * @property {string} description
 * @property {object} parameters JSON Schema for the call arguments.
 *
 * @typedef {Object} TranscriptEvent
 * @property {"user" | "assistant"} role
 * @property {string} text
 * @property {boolean} partial
 */

import {
  base64FromArrayBuffer,
  base64ToBytes,
  extractResponseTranscript,
  trimTrailingSlash,
} from "./codec.js";
import { OrbVisualiser, VIS_FFT_SIZE } from "./orb-visualizer.js";
import {
  aec3ProcessorOptions,
  loadAec3Worklet,
} from "../worklets/aec3/aec3-loader.js";

/** Build an Error carrying a `code` (and optional extra fields) so callers can
 *  branch on the failure kind: "limit" | "queue-full" | "queue-expired" | "aborted".
 *  @param {string} message @param {string} code @param {object} [extra] */
function _codedError(message, code, extra) {
  const err = /** @type {Error & { code?: string }} */ (new Error(message));
  err.code = code;
  if (extra) Object.assign(err, extra);
  return err;
}

// The server acknowledges the outbound PCM clock as part of its local pipeline
// configuration.  It is not an OpenAI `audio.output.format` option: adding an
// unknown shape there would make strict clients reject the whole update.
// 16 kHz is deliberately the safe initial/default-provider fallback.
const DEFAULT_OUTPUT_SAMPLE_RATE = 16000;
const MIC_CHUNK_MS = 40;
export const PIPELINE_CONFIG_ACK_TIMEOUT_MS = 15_000;
export const MAX_PLAYBACK_PRIME_MS = 2_000;
export const PLAYBACK_CONTINUITY_ADAPTIVE = "adaptive";
export const PLAYBACK_CONTINUITY_FAST_START = "fast-start";
const MAX_PLAYBACK_RESPONSE_TOMBSTONES = 512;
const MAX_TOOL_CALL_TOMBSTONES = 512;
const TERMINAL_RESPONSE_LIFECYCLE_STATES = new Set([
  "completed", "cancelled", "canceled", "failed", "incomplete", "error",
]);
export const AUDIO_CPP_PLAYBACK_PRIME_MS = Object.freeze({
  "low-latency": 800,
  balanced: 1280,
  quality: 1760,
});

function _normalisePlaybackProvider(value) {
  const provider = String(value || "").trim().toLowerCase();
  return provider === "audio-cpp" ? "qwen3tts-audiocpp" : provider;
}

function _normaliseProfileId(value) {
  return String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[_\s]+/g, "-");
}

/** @param {unknown} value */
function _normaliseProfileRevision(value) {
  const revision = Number(value);
  return Number.isSafeInteger(revision) && revision > 0 ? revision : null;
}

/** @param {unknown} value */
function _isTerminalResponseLifecycleState(value) {
  return TERMINAL_RESPONSE_LIFECYCLE_STATES.has(String(value || "").toLowerCase());
}

/**
 * Resolve the browser queue target without changing the WebSocket schema.
 * Both native and buffered-phrase audio.cpp PCM use the fixed model clock and
 * benefit from a bounded startup reservoir. Built-ins use fixed acceptance
 * targets; a custom profile uses its locally resolved first-block duration.
 *
 * @param {Record<string, any>} config Acknowledged pipeline config.
 * @param {PlaybackConfig} [hint] Browser-only validated profile snapshot.
 */
export function resolvePlaybackPrimeMs(config = {}, hint = {}) {
  const provider = _normalisePlaybackProvider(config.tts_backend || hint.provider);
  if (provider !== "qwen3tts-audiocpp") return 0;

  const profileId = _normaliseProfileId(config.tts_tuning?.profile_id || hint.profileId);
  if (profileId === "low" || profileId === "lowlatency") {
    return AUDIO_CPP_PLAYBACK_PRIME_MS["low-latency"];
  }
  if (Object.hasOwn(AUDIO_CPP_PLAYBACK_PRIME_MS, profileId)) {
    return AUDIO_CPP_PLAYBACK_PRIME_MS[profileId];
  }

  const overrideFrames = Number(config.tts_tuning?.overrides?.first_block_frames);
  const resolved = Number.isFinite(Number(hint.resolvedPrimeMs))
    ? Number(hint.resolvedPrimeMs)
    : Number.isFinite(overrideFrames) ? overrideFrames * 80 : 0;
  return Math.max(0, Math.min(MAX_PLAYBACK_PRIME_MS, resolved));
}

function _responseEpoch(event = {}) {
  const candidate = event.response_epoch ?? event.response?.response_epoch;
  const epoch = Number(candidate);
  return Number.isSafeInteger(epoch) && epoch >= 0 ? epoch : null;
}

function _inputEpoch(event = {}) {
  const candidate = event.input_epoch ?? event.response?.input_epoch;
  const epoch = Number(candidate);
  return Number.isSafeInteger(epoch) && epoch >= 0 ? epoch : null;
}

/** Return a valid local PCM source clock or null for standard peers. */
function _responseOutputSampleRate(value) {
  const rate = Number(value);
  return Number.isInteger(rate) && rate > 0 ? rate : null;
}

function _continuityMode(value) {
  return value === PLAYBACK_CONTINUITY_FAST_START
    ? PLAYBACK_CONTINUITY_FAST_START
    : PLAYBACK_CONTINUITY_ADAPTIVE;
}

/**
 * The local server freezes this complete policy when it claims response
 * ownership.  A later config acknowledgement is deliberately next-response
 * only, so the lifecycle copy wins whenever it is available.  Older local
 * servers sent only `output_sample_rate`; retain that compatibility fallback.
 * @param {Record<string, any>} event
 * @param {number | null} lifecycleRate
 * @returns {{ sourceSampleRate: number, primeMs: number, continuityMode: string, nativeStreaming: boolean, maxPrimeMs: number, provider: string, profileId: string, profileRevision: number | null } | null}
 */
function _lifecyclePlaybackPolicy(event, lifecycleRate) {
  const raw = event?.playback_policy ?? event?.playbackPolicy;
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return null;
  const sourceSampleRate = _responseOutputSampleRate(
    raw.source_sample_rate ?? raw.sourceSampleRate,
  ) ?? lifecycleRate;
  if (sourceSampleRate === null) return null;
  const maxCandidate = Number(raw.max_prime_ms ?? raw.maxPrimeMs);
  const maxPrimeMs = Number.isFinite(maxCandidate)
    ? Math.max(0, Math.min(MAX_PLAYBACK_PRIME_MS, Math.round(maxCandidate)))
    : MAX_PLAYBACK_PRIME_MS;
  const primeCandidate = Number(raw.prime_target_ms ?? raw.primeMs);
  const primeMs = Number.isFinite(primeCandidate)
    ? Math.max(0, Math.min(maxPrimeMs, Math.round(primeCandidate)))
    : 0;
  return {
    sourceSampleRate,
    primeMs,
    continuityMode: _continuityMode(raw.continuity_mode ?? raw.continuityMode),
    nativeStreaming: raw.native_streaming === true || raw.nativeStreaming === true,
    maxPrimeMs,
    provider: _normalisePlaybackProvider(raw.provider),
    profileId: _normaliseProfileId(raw.profile_id ?? raw.profileId),
    profileRevision: _normaliseProfileRevision(raw.profile_revision ?? raw.profileRevision),
  };
}
const DEFAULT_ECHO_CALIBRATION = Object.freeze({
  delayMs: 0,
  suppressionStrength: 0.65,
  leakageThreshold: 0.65,
  doubleTalkSensitivity: 0.5,
  echoTailMs: 350,
});

/** @param {Partial<EchoCalibration> | null | undefined} value */
function normalizeEchoCalibration(value) {
  const clamp = (candidate, minimum, maximum, fallback) => {
    const number = Number(candidate);
    return Number.isFinite(number) ? Math.max(minimum, Math.min(maximum, number)) : fallback;
  };
  return {
    delayMs: clamp(value?.delayMs, 0, 500, DEFAULT_ECHO_CALIBRATION.delayMs),
    suppressionStrength: clamp(
      value?.suppressionStrength, 0, 1, DEFAULT_ECHO_CALIBRATION.suppressionStrength,
    ),
    leakageThreshold: clamp(
      value?.leakageThreshold, 0.05, 1, DEFAULT_ECHO_CALIBRATION.leakageThreshold,
    ),
    doubleTalkSensitivity: clamp(
      value?.doubleTalkSensitivity, 0, 1, DEFAULT_ECHO_CALIBRATION.doubleTalkSensitivity,
    ),
    echoTailMs: clamp(value?.echoTailMs, 0, 1000, DEFAULT_ECHO_CALIBRATION.echoTailMs),
  };
}

export class S2sWsRealtimeClient extends EventTarget {
  /** @param {WsClientOptions} options */
  constructor(options) {
    super();
    /** @type {WsClientOptions} */
    this.options = options;
    /** @type {ToolDef[]} Function tools declared to the backend. */
    this._tools = options.tools ?? [];
    /** @type {string} Direct realtime WS URL (set => skip the LB session POST). */
    this._directUrl = options.directUrl ?? "";
    /** @type {string} Where to POST for the session handshake. Prefer the
     *  explicit `sessionUrl`; fall back to `<loadBalancerUrl>/session` for callers
     *  that still pass the LB address directly. */
    this._sessionUrl = options.sessionUrl
      ? options.sessionUrl
      : options.loadBalancerUrl
        ? `${trimTrailingSlash(options.loadBalancerUrl)}/session`
        : "";
    /** @type {(() => Promise<MediaStream>) | null} Lazy mic acquisition (post-grant). */
    this._acquireMic = options.acquireMic ?? null;
    /** @type {boolean} Set by close() to abort a queue wait in progress. */
    this._closed = false;
    /** @type {string} The active queue ticket id while waiting (else ""). */
    this._queueId = "";
    /** @type {(() => void) | null} Wakes the queue poll sleep early on close(). */
    this._queueWake = null;
    /** @type {ReturnType<typeof setTimeout> | 0} */
    this._queueTimer = 0;
    // Join gate: after waiting in line the caller must explicitly `join()` before
    // we dial, so a slot isn't spent on someone who walked away. Resolved by
    // join(), rejected on timeout (the LB reclaims the slot) or close().
    /** @type {(() => void) | null} */
    this._joinResolve = null;
    /** @type {((err: Error) => void) | null} */
    this._joinReject = null;
    /** @type {ReturnType<typeof setTimeout> | 0} */
    this._joinTimer = 0;
    /** @type {NoiseGate} Mic noise gate; off by default. */
    this._noiseGate = options.noiseGate ?? { enabled: false, thresholdDb: -45 };
    /** @type {EchoGuardMode} */
    this._echoGuard = options.echoGuard ?? "native";
    /** @type {Record<string, EchoCalibration>} */
    this._echoCalibrations = options.echoCalibrations ?? {};
    this._echoDevicePair = "";
    /** @type {EchoCalibration | null} */
    this._echoCalibration = null;
    this._aec3Status = null;
    /** @type {WebSocket | null} */
    this._ws = null;
    /** @type {AudioContext | null} */
    this._ctx = null;
    /** @type {MediaStreamAudioSourceNode | null} */
    this._micSrc = null;
    /** @type {AudioWorkletNode | null} */
    this._captureNode = null;
    /** @type {AudioWorkletNode | null} */
    this._playbackNode = null;
    this._playbackGeneration = 0;
    this._playbackGenerationInvalidated = false;
    // The browser only changes this after the server commits a local-pipeline
    // update.  Mic/VAD capture remains independently fixed at 16 kHz.
    this._playbackSampleRate = DEFAULT_OUTPUT_SAMPLE_RATE;
    this._playbackPrimeMs = 0;
    this._adaptivePlaybackPrimeMs = 0;
    this._playbackContinuityMode = PLAYBACK_CONTINUITY_ADAPTIVE;
    this._playbackUnsustainable = false;
    this._healthyPlaybackResponses = 0;
    // Monotonic browser-side identity for one acknowledged playback policy.
    // Profile revision alone is insufficient because unsaved session overrides
    // may alter first-block priming without changing the persisted revision.
    this._playbackConfigToken = 0;
    this._latestInputEpoch = -1;
    this._latestResponseEpoch = -1;
    /** @type {{ inputEpoch: number | null, effectiveInterrupt: boolean, reason: string } | null} */
    this._pendingSpeechStartDecision = null;
    /** @type {PlaybackConfig} */
    this._acknowledgedPlaybackConfig = {
      provider: "",
      profileId: "",
      profileRevision: null,
      nativeStreaming: false,
      resolvedPrimeMs: 0,
    };
    /** @type {{ expectedProvider: string, expectedProfile: string, expectedRevision: number | null, hint: PlaybackConfig }[]} */
    this._pendingPlaybackConfigs = [];
    /** @type {Map<string, { generation: number, primeMs: number, reprimeMs: number, continuityMode: string, nativeStreaming: boolean, maxPrimeMs: number, provider: string, profileId: string, profileRevision: number | null, responseEpoch: number | null, sourceSampleRate: number, audioStarted: boolean, ended: boolean, playbackAcked: boolean, playbackDrained: boolean, playbackCleared: boolean, clearReason: string, drainAudible: boolean, playbackUnsustainable: boolean, hadUnderrun: boolean, terminal: boolean, terminalSettled: boolean, pendingTerminalDetail: Record<string, any> | null }>} */
    this._playbackByResponse = new Map();
    /**
     * Frozen lifecycle playback policy awaiting a response ID. Ownership is
     * claimed before `response.created`, so a live config acknowledgement in
     * that gap must not retime or re-prime an already-owned response.
      * @type {Map<number, { sourceSampleRate: number, primeMs: number, continuityMode: string, nativeStreaming: boolean, maxPrimeMs: number, provider: string, profileId: string }>}
     */
    this._playbackPolicyByResponseEpoch = new Map();
    /** @type {Set<string>} Bounded completed/cancelled response IDs. */
    this._stalePlaybackResponses = new Set();
    /** @type {Set<string>} Response terminal IDs already delivered to the UI.
     * A repeated response.done must not settle or roll back a later user turn. */
    this._terminalResponseIds = new Set();
    /** @type {Set<string>} Per-connection call IDs already dispatched to the UI.
     * Realtime peers may replay a completed function-call event after its output
     * acknowledgement; never run an external browser tool twice for that ID. */
    this._toolCallTombstones = new Set();
    /** @type {Set<string>} Response IDs that committed a tool transaction.
     * Their origin user history is durable even when the response produced no PCM. */
    this._toolCommittedResponses = new Set();
    /** @type {Set<string>} Epoch-only terminal lifecycle events already settled.
     * A response can fail before OpenAI allocates an ID; do not repeatedly roll
     * back the same provisional user row if transport cleanup repeats it. */
    this._terminalResponseEpochs = new Set();
    /** @type {Set<number>} Epoch tombstones reject stale PCM even before a
     * successor response is claimed. Response IDs are absent on one early
     * cancellation path, so an ID-only tombstone is insufficient. */
    this._cancelledResponseEpochs = new Set();
    /** @type {GainNode | null} */
    this._captureSink = null;
    /** @type {AnalyserNode | null} */
    this._micAnalyser = null;
    /** @type {AnalyserNode | null} */
    this._outAnalyser = null;
    /** @type {OrbVisualiser | null} */
    this._visualiser = null;
    /** @type {WsStatus} */
    this._status = "idle";
    this._aiSpeaking = false;
    this._speechStoppedAtMs = null;
    this._firstPlaybackReported = false;
    /** @type {Set<string>} Response IDs whose first PCM sample was confirmed by
     * the worklet's `started` event. Network audio receipt is intentionally not
     * evidence that the user heard a response. */
    this._heardResponses = new Set();
    /** @type {Map<string, string>} The CURRENT assistant transcript segment per
     * response, accumulated from streamed deltas (reset on each segment's done). */
    this._asstTranscriptByResp = new Map();
    /** @type {Map<string, string>} Completed assistant transcript segments per
     * response, space-joined. A single response can emit several
     * `*.transcript.done` events; we concatenate them until response.done. */
    this._asstFullByResp = new Map();
    this._muted = false;
    // ── Response lock ────────────────────────────────────────────────────
    // The backend allows only ONE response in flight: creating a second while
    // one is active fails with `conversation_already_has_active_response`. So
    // we serialize response.create. `_openResponses` counts responses the
    // server has confirmed (response.created) but not yet finished
    // (response.done) — it's cumulative, so every create maps to one done.
    // `_createInFlight` covers the window after we send a create but before its
    // response.created echo. Any requestResponse() made while locked is queued
    // and replayed, one at a time, as each response.done frees the slot.
    this._openResponses = 0;
    this._createInFlight = false;
    /** @type {{ image?: string }[]} Pending response.create payloads, one per
     * queued requestResponse(). A payload may carry an image to send just
     * before its create (so the frame travels with the create, not eagerly). */
    this._createQueue = [];
    /** @type {Map<string, { resolve: () => void, reject: (reason?: unknown) => void, timer: number }>} */
    this._toolOutputAcks = new Map();
    /** @type {Set<{ resolve: () => void, reject: (reason?: unknown) => void, timer: number }>} */
    this._responseIdleWaiters = new Set();
    this._sessionConfigured = false;
    /** @type {Promise<void> | null} */
    this._configAckPromise = null;
    /** @type {(() => void) | null} */
    this._configAckResolve = null;
    /** @type {((error: Error) => void) | null} */
    this._configAckReject = null;
    /** @type {ReturnType<typeof setTimeout> | 0} */
    this._configAckTimer = 0;
    this._debug = (() => { try { return localStorage.getItem("s2s.debug") === "1"; } catch { return false; } })();
  }

  get status() {
    return this._status;
  }

  /** @param {WsStatus} status */
  _setStatus(status) {
    if (this._closed && status !== "closed") return;
    if (this._status === status) return;
    this._status = status;
    this.dispatchEvent(new CustomEvent("status", { detail: { status } }));
  }

  _waitForInitialConfigAck() {
    if (this._sessionConfigured) return Promise.resolve();
    if (this._configAckPromise) return this._configAckPromise;
    this._configAckPromise = new Promise((resolve, reject) => {
      this._configAckResolve = resolve;
      this._configAckReject = reject;
      this._configAckTimer = setTimeout(() => {
        this._rejectInitialConfig(
          _codedError(
            "Timed out waiting for the speech pipeline configuration acknowledgement",
            "pipeline-config-timeout",
          ),
        );
      }, PIPELINE_CONFIG_ACK_TIMEOUT_MS);
    });
    return this._configAckPromise;
  }

  _resolveInitialConfig() {
    if (this._configAckTimer) clearTimeout(this._configAckTimer);
    this._configAckTimer = 0;
    const resolve = this._configAckResolve;
    this._configAckResolve = null;
    this._configAckReject = null;
    resolve?.();
  }

  /** @param {Record<string, any>} config @param {PlaybackConfig | null | undefined} hint */
  _queuePlaybackConfig(config, hint) {
    const requestedProvider = _normalisePlaybackProvider(
      config.tts_backend || hint?.provider || this._acknowledgedPlaybackConfig.provider,
    );
    const sameProvider = requestedProvider === this._acknowledgedPlaybackConfig.provider;
    const effectiveHint = {
      ...(sameProvider ? this._acknowledgedPlaybackConfig : {}),
      ...(hint || {}),
      provider: requestedProvider,
      profileId: String(
        config.tts_tuning?.profile_id
          || hint?.profileId
          || (sameProvider ? this._acknowledgedPlaybackConfig.profileId : ""),
      ),
      profileRevision: _normaliseProfileRevision(
        config.tts_tuning?.profile_revision
          ?? hint?.profileRevision
          ?? (sameProvider ? this._acknowledgedPlaybackConfig.profileRevision : null),
      ),
    };
    this._pendingPlaybackConfigs.push({
      expectedProvider: requestedProvider,
      expectedProfile: _normaliseProfileId(effectiveHint.profileId),
      expectedRevision: _normaliseProfileRevision(effectiveHint.profileRevision),
      hint: effectiveHint,
    });
    // A page should have at most one update awaiting acknowledgement. Keep a
    // hard bound anyway so a broken peer cannot grow browser memory forever.
    if (this._pendingPlaybackConfigs.length > 16) this._pendingPlaybackConfigs.shift();
  }

  /**
   * Build the bounded local extension that the server freezes at response
   * admission. This is deliberately part of pipeline.config.update, never an
   * OpenAI session audio-format field. The server validates and echoes it.
   * @param {Record<string, any>} config
   * @param {PlaybackConfig | null | undefined} hint
   */
  _requestedPlaybackPolicy(config, hint) {
    const provider = _normalisePlaybackProvider(
      config.tts_backend || hint?.provider || this._acknowledgedPlaybackConfig.provider,
    );
    const nativeStreaming = provider === "qwen3tts-audiocpp" && hint?.nativeStreaming === true;
    const sourceSampleRate = provider === "qwen3tts-audiocpp" ? 24_000 : DEFAULT_OUTPUT_SAMPLE_RATE;
    const continuityMode = _continuityMode(hint?.continuityMode);
    const primeMs = provider === "qwen3tts-audiocpp"
      ? resolvePlaybackPrimeMs(
        { ...config, tts_backend: provider, audio_output_sample_rate: sourceSampleRate },
        { ...hint, provider, nativeStreaming, continuityMode },
      )
      : 0;
    return {
      prime_target_ms: Math.max(0, Math.min(MAX_PLAYBACK_PRIME_MS, Math.round(primeMs))),
      continuity_mode: continuityMode,
      native_streaming: nativeStreaming,
      max_prime_ms: MAX_PLAYBACK_PRIME_MS,
    };
  }

  /** Apply browser playback policy only after the backend committed the config. */
  /** @param {Record<string, any>} config */
  _applyAcknowledgedPlaybackConfig(config) {
    const provider = _normalisePlaybackProvider(config.tts_backend);
    const profileId = _normaliseProfileId(config.tts_tuning?.profile_id);
    const profileRevision = _normaliseProfileRevision(config.tts_tuning?.profile_revision);
    let pendingIndex = this._pendingPlaybackConfigs.findIndex((pending) => (
      pending.expectedProvider === provider
      && (!pending.expectedProfile || pending.expectedProfile === profileId)
      && (pending.expectedRevision === null || pending.expectedRevision === profileRevision)
    ));
    if (pendingIndex < 0 && this._pendingPlaybackConfigs.length === 1) pendingIndex = 0;
    const pending = pendingIndex >= 0
      ? this._pendingPlaybackConfigs.splice(pendingIndex, 1)[0]
      : null;
    const canReuseAcknowledged = provider === this._acknowledgedPlaybackConfig.provider;
    const hint = {
      ...(canReuseAcknowledged ? this._acknowledgedPlaybackConfig : {}),
      ...(pending?.hint || {}),
      provider,
      profileId: profileId || pending?.hint?.profileId || "",
      profileRevision: profileRevision ?? pending?.hint?.profileRevision ?? null,
    };
    const configuredRate = Number(config.audio_output_sample_rate);
    const acknowledgedRate = Number.isInteger(configuredRate) && configuredRate > 0
      ? configuredRate
      : DEFAULT_OUTPUT_SAMPLE_RATE;
    const serverPolicy = _lifecyclePlaybackPolicy(
      { playback_policy: config.playback_policy ?? config.playbackPolicy },
      acknowledgedRate,
    );
    const primeMs = serverPolicy?.primeMs ?? resolvePlaybackPrimeMs(config, hint);
    this._playbackConfigToken += 1;
    this._acknowledgedPlaybackConfig = {
      ...hint,
      nativeStreaming: serverPolicy?.nativeStreaming ?? hint.nativeStreaming,
      configToken: this._playbackConfigToken,
    };
    this._playbackSampleRate = serverPolicy?.sourceSampleRate ?? acknowledgedRate;
    // Reconfigure in place so an already-created worklet always uses the
    // server-acknowledged source clock. This changes only interpolation into
    // the AudioContext; it never time-stretches, pitches, or rewrites PCM.
    this._playbackNode?.port.postMessage({
      kind: "config",
      inputRate: this._playbackSampleRate,
      generation: this._playbackGeneration,
    });
    this._playbackPrimeMs = primeMs;
    this._playbackContinuityMode = serverPolicy?.continuityMode ?? _continuityMode(hint.continuityMode);
    this._adaptivePlaybackPrimeMs = primeMs;
    this._playbackUnsustainable = false;
    this._healthyPlaybackResponses = 0;
    this.dispatchEvent(new CustomEvent("pipeline-metric", {
      detail: {
        stage: "playback",
        status: "configured",
        source: "browser",
        detail: {
          provider,
          profile_id: hint.profileId || null,
          profile_revision: hint.profileRevision ?? null,
          native_streaming: hint.nativeStreaming === true,
          source_sample_rate: this._playbackSampleRate,
          continuity_mode: this._playbackContinuityMode,
          prime_target_ms: primeMs,
          acknowledged: true,
        },
      },
    }));
  }

  /** @param {number | null} lifecycleRate */
  _capturePlaybackPolicy(lifecycleRate = null) {
    const sourceSampleRate = lifecycleRate ?? this._playbackSampleRate;
    const continuityMode = this._playbackContinuityMode;
    return {
      sourceSampleRate,
      primeMs: continuityMode === PLAYBACK_CONTINUITY_ADAPTIVE
        ? Math.max(this._playbackPrimeMs, this._adaptivePlaybackPrimeMs)
        : this._playbackPrimeMs,
      continuityMode,
      nativeStreaming: this._acknowledgedPlaybackConfig.nativeStreaming === true,
      maxPrimeMs: MAX_PLAYBACK_PRIME_MS,
      provider: _normalisePlaybackProvider(this._acknowledgedPlaybackConfig.provider),
      profileId: _normaliseProfileId(this._acknowledgedPlaybackConfig.profileId),
      profileRevision: _normaliseProfileRevision(this._acknowledgedPlaybackConfig.profileRevision),
      configPrimeMs: this._playbackPrimeMs,
      configToken: this._playbackConfigToken,
    };
  }

  /**
   * Capture policy once at response ownership, before a response ID exists.
   * A later provider/profile acknowledgement is next-response-only; it cannot
   * change the old response's source clock or startup reservoir.
   * @param {number | null} responseEpoch
   * @param {number | null} lifecycleRate
   * @param {string} responseId
   */
  _captureLifecyclePlaybackPolicy(responseEpoch, lifecycleRate, responseId = "", serverPolicy = null) {
    if (responseEpoch === null) return null;
    let policy = this._playbackPolicyByResponseEpoch.get(responseEpoch);
    if (!policy) {
      // The server's admission snapshot is the authoritative ordering boundary.
      // The browser-local default is only a compatibility fallback for older
      // servers that did not yet include playback_policy in pipeline.response.
      const fallbackPolicy = this._capturePlaybackPolicy(lifecycleRate);
      if (serverPolicy) {
        const provider = serverPolicy.provider || fallbackPolicy.provider;
        const profileId = serverPolicy.profileId || fallbackPolicy.profileId;
        const profileRevision = serverPolicy.profileRevision ?? fallbackPolicy.profileRevision;
        const sameAcknowledgedConfig = (
          _normalisePlaybackProvider(provider) === _normalisePlaybackProvider(fallbackPolicy.provider)
          && _normaliseProfileId(profileId) === _normaliseProfileId(fallbackPolicy.profileId)
          && _normaliseProfileRevision(profileRevision)
            === _normaliseProfileRevision(fallbackPolicy.profileRevision)
          && serverPolicy.continuityMode === fallbackPolicy.continuityMode
          && serverPolicy.nativeStreaming === fallbackPolicy.nativeStreaming
          && Number(serverPolicy.primeMs) === Number(fallbackPolicy.configPrimeMs)
        );
        policy = {
          ...fallbackPolicy,
          ...serverPolicy,
          // Older lifecycle payloads froze the transport fields but did not
          // name their provider/profile. Preserve the admission-time browser
          // identity rather than leaving late feedback unscoped.
          provider,
          profileId,
          profileRevision,
          // The server owns the immutable base policy. Browser learning may
          // raise only the response reservoir, and only for the exact same
          // acknowledged configuration. A provider/profile/session-override
          // switch must start from its own target.
          primeMs: sameAcknowledgedConfig
            && serverPolicy.continuityMode === PLAYBACK_CONTINUITY_ADAPTIVE
            ? Math.max(serverPolicy.primeMs, this._adaptivePlaybackPrimeMs)
            : serverPolicy.primeMs,
          configPrimeMs: serverPolicy.primeMs,
          configToken: sameAcknowledgedConfig ? fallbackPolicy.configToken : -1,
        };
      } else {
        policy = fallbackPolicy;
      }
      this._playbackPolicyByResponseEpoch.set(responseEpoch, policy);
    }
    const snapshot = responseId ? this._playbackByResponse.get(responseId) : null;
    // A delayed lifecycle event may repair an unstarted fallback snapshot, but
    // never mutate PCM already handed to the worklet.
    if (snapshot && !snapshot.audioStarted) {
      snapshot.sourceSampleRate = policy.sourceSampleRate;
      snapshot.primeMs = policy.primeMs;
      snapshot.reprimeMs = policy.continuityMode === PLAYBACK_CONTINUITY_ADAPTIVE && policy.primeMs > 0
        ? Math.min(policy.maxPrimeMs, Math.max(80, Math.min(policy.primeMs, 240)))
        : policy.primeMs;
      snapshot.continuityMode = policy.continuityMode;
      snapshot.nativeStreaming = policy.nativeStreaming;
      snapshot.maxPrimeMs = policy.maxPrimeMs;
      snapshot.provider = policy.provider;
      snapshot.profileId = policy.profileId;
      snapshot.profileRevision = policy.profileRevision;
      snapshot.configPrimeMs = policy.configPrimeMs;
      snapshot.configToken = policy.configToken;
    }
    return policy;
  }

  /** @param {string} responseId @param {number | null} [responseEpoch] */
  _playbackSnapshot(responseId, responseEpoch = null) {
    if (responseId) {
      const existing = this._playbackByResponse.get(responseId);
      if (existing) {
        if (existing.responseEpoch === null && responseEpoch !== null) existing.responseEpoch = responseEpoch;
        return existing;
      }
    }
    const frozenPolicy = responseEpoch === null
      ? null
      : this._playbackPolicyByResponseEpoch.get(responseEpoch) ?? null;
    if (responseEpoch !== null) this._playbackPolicyByResponseEpoch.delete(responseEpoch);
    const policy = frozenPolicy ?? this._capturePlaybackPolicy();
    const snapshot = {
      generation: this._playbackGeneration,
      primeMs: policy.primeMs,
      // Cold startup may require a larger reservoir. A genuine underrun
      // recovers from this bounded steady-state target instead of replaying
      // the first-response delay.
      reprimeMs: policy.continuityMode === PLAYBACK_CONTINUITY_ADAPTIVE && policy.primeMs > 0
        ? Math.min(policy.maxPrimeMs, Math.max(80, Math.min(policy.primeMs, 240)))
        : policy.primeMs,
      continuityMode: policy.continuityMode,
      nativeStreaming: policy.nativeStreaming,
      maxPrimeMs: policy.maxPrimeMs,
      provider: policy.provider,
      profileId: policy.profileId,
      profileRevision: policy.profileRevision,
      configPrimeMs: policy.configPrimeMs,
      configToken: policy.configToken,
      responseEpoch,
      // Clock and reservoir are immutable for this response. A live provider
      // switch may acknowledge different defaults while this response remains
      // queued, but must never change its pitch, duration, or priming policy.
      sourceSampleRate: policy.sourceSampleRate,
      // Once PCM has been queued, an out-of-order lifecycle event may not
      // change the source clock underneath audio already in the worklet FIFO.
      audioStarted: false,
      ended: false,
      playbackAcked: false,
      playbackDrained: false,
      playbackCleared: false,
      clearReason: "",
      drainAudible: false,
      playbackUnsustainable: false,
      hadUnderrun: false,
      // A completed protocol response can still be priming in the worklet.
      // Keep it addressable until `started`/`drained`; network completion is
      // deliberately not evidence of audible playback.
      terminal: false,
      // `response.done` can race the worklet's first rendered sample. Keep the
      // terminal transaction here until a concrete started/drained/cleared
      // result decides whether it was actually heard.
      terminalSettled: false,
      pendingTerminalDetail: null,
    };
    snapshot.playbackUnsustainable = this._playbackUnsustainable
      && this._playbackFeedbackMatches(snapshot);
    if (responseId) this._playbackByResponse.set(responseId, snapshot);
    return snapshot;
  }

  /** Bind the local lifecycle event emitted after response.created. */
  _bindPlaybackResponseEpoch(responseId, responseEpoch) {
    if (!responseId || !this._acceptResponseEpoch(responseEpoch)) return false;
    const snapshot = this._playbackSnapshot(responseId, responseEpoch);
    if (snapshot.responseEpoch !== null && responseEpoch !== null && snapshot.responseEpoch !== responseEpoch) return false;
    snapshot.responseEpoch = responseEpoch;
    return true;
  }

  /** @param {number | null} epoch */
  _isStaleResponseEpoch(epoch) {
    return epoch !== null && (epoch < this._latestResponseEpoch || this._cancelledResponseEpochs.has(epoch));
  }

  /**
   * A newer response may be promoted after the old response is protocol
   * complete while its already-admitted PCM is still draining from the browser
   * FIFO. Accept lifecycle messages only for that exact surviving snapshot;
   * network/model output continues to use the stricter monotonic epoch gate.
   * @param {string} responseId
   * @param {number | null} responseEpoch
   * @param {number} generation
   * @param {string} [kind]
   */
  _acceptQueuedPlaybackLifecycle(responseId, responseEpoch, generation, kind = "") {
    if (!responseId || this._stalePlaybackResponses.has(responseId)) return false;
    const snapshot = this._playbackByResponse.get(responseId);
    if (!snapshot || Number(snapshot.generation) !== generation) return false;
    if (snapshot.responseEpoch !== null && responseEpoch !== null
        && snapshot.responseEpoch !== responseEpoch) return false;
    // A worklet can render the first sample and post `started` immediately
    // before a cross-thread barge-in clear, while the WebSocket cancellation
    // reaches this task first. The exact surviving snapshot is proof of that
    // already-rendered sample until a clear/drain says otherwise. This never
    // admits more network PCM for the cancelled epoch.
    if (kind === "started") {
      return !snapshot.playbackCleared && !snapshot.playbackDrained;
    }
    if (kind === "drained") return true;
    return responseEpoch === null || !this._cancelledResponseEpochs.has(responseEpoch);
  }

  /** @param {number | null} epoch */
  _tombstoneResponseEpoch(epoch) {
    if (epoch === null) return;
    this._cancelledResponseEpochs.add(epoch);
    while (this._cancelledResponseEpochs.size > MAX_PLAYBACK_RESPONSE_TOMBSTONES) {
      const oldest = this._cancelledResponseEpochs.values().next().value;
      if (oldest === undefined) break;
      this._cancelledResponseEpochs.delete(oldest);
    }
  }

  /** @param {number | null} epoch */
  _acceptResponseEpoch(epoch) {
    if (epoch === null) return true; // Compatibility for non-local peers.
    if (this._isStaleResponseEpoch(epoch)) return false;
    this._latestResponseEpoch = Math.max(this._latestResponseEpoch, epoch);
    return true;
  }

  /**
   * Resolve a response-bound event through its immutable playback snapshot.
   * Compatible peers may omit response_epoch from late transcript frames, but
   * a cleared older generation must never be rebound to the current response.
   * @param {string} responseId
   * @param {number | null} eventEpoch
   */
  _acceptResponseBoundEvent(responseId, eventEpoch) {
    if (responseId && this._stalePlaybackResponses.has(responseId)) return false;
    const snapshot = responseId ? this._playbackByResponse.get(responseId) ?? null : null;
    if (snapshot && Number(snapshot.generation) < this._playbackGeneration) return false;
    if (snapshot && snapshot.responseEpoch !== null && eventEpoch !== null
        && snapshot.responseEpoch !== eventEpoch) return false;
    return this._acceptResponseEpoch(eventEpoch ?? snapshot?.responseEpoch ?? null);
  }

  /** @param {number | null} epoch */
  _acceptInputEpoch(epoch) {
    if (epoch === null) return true;
    if (epoch < this._latestInputEpoch) return false;
    this._latestInputEpoch = Math.max(this._latestInputEpoch, epoch);
    return true;
  }

  _playbackFeedbackMatches(snapshot) {
    if (!snapshot || snapshot.continuityMode !== PLAYBACK_CONTINUITY_ADAPTIVE) return false;
    return this._playbackContinuityMode === PLAYBACK_CONTINUITY_ADAPTIVE
      && _normalisePlaybackProvider(snapshot.provider) === "qwen3tts-audiocpp"
      && _normalisePlaybackProvider(snapshot.provider)
        === _normalisePlaybackProvider(this._acknowledgedPlaybackConfig.provider)
      && _normaliseProfileId(snapshot.profileId)
        === _normaliseProfileId(this._acknowledgedPlaybackConfig.profileId)
      && _normaliseProfileRevision(snapshot.profileRevision)
        === _normaliseProfileRevision(this._acknowledgedPlaybackConfig.profileRevision)
      && Number(snapshot.configPrimeMs) === Number(this._playbackPrimeMs)
      && Number(snapshot.configToken) === Number(this._playbackConfigToken)
      && (snapshot.nativeStreaming === true)
        === (this._acknowledgedPlaybackConfig.nativeStreaming === true);
  }

  _raiseAdaptiveReserve(workletDetail = null, snapshot = null) {
    if (!this._playbackFeedbackMatches(snapshot)) return;
    const previous = this._adaptivePlaybackPrimeMs;
    const observedGap = Number(workletDetail?.maxObservedChunkGapMs ?? workletDetail?.observedChunkGapMs);
    const measuredReserve = Number.isFinite(observedGap) && observedGap > 0
      ? Math.ceil(observedGap + 120)
      : 0;
    // Preserve the unclamped demand for the failure contract.  A first large
    // gap may require more than the two-second safety ceiling; waiting for a
    // second underrun before reporting that fact makes the diagnostics lie
    // about the provider's realtime viability.
    const requiredReserveMs = Math.max(
      this._playbackPrimeMs,
      previous + 160,
      measuredReserve,
    );
    this._adaptivePlaybackPrimeMs = Math.min(
      MAX_PLAYBACK_PRIME_MS,
      requiredReserveMs,
    );
    if (requiredReserveMs > MAX_PLAYBACK_PRIME_MS) {
      this._markPlaybackUnsustainable("reservoir_cap", snapshot);
    }
  }

  /** @param {string} reason */
  _markPlaybackUnsustainable(reason, snapshot = null) {
    if (snapshot) snapshot.playbackUnsustainable = true;
    if (snapshot && !this._playbackFeedbackMatches(snapshot)) return;
    if (this._playbackUnsustainable) return;
    this._playbackUnsustainable = true;
    const provider = snapshot?.provider || this._acknowledgedPlaybackConfig.provider;
    const nativeStreaming = snapshot?.nativeStreaming
      ?? (this._acknowledgedPlaybackConfig.nativeStreaming === true);
    this.dispatchEvent(new CustomEvent("pipeline-metric", {
      detail: {
        stage: "playback", status: "provider_unsustainable", source: "browser",
        detail: {
          reason, provider,
          profile_id: snapshot?.profileId || this._acknowledgedPlaybackConfig.profileId || null,
          continuity_mode: snapshot?.continuityMode || this._playbackContinuityMode,
          adaptive_prime_target_ms: this._adaptivePlaybackPrimeMs,
          max_prime_ms: MAX_PLAYBACK_PRIME_MS,
          message: nativeStreaming
            ? "Provider cannot sustain realtime; choose buffered phrase mode for reliable completion."
            : "Provider cannot sustain progressive phrase playback within the two-second reservoir; use the explicit full-buffer fallback for reliable completion.",
        },
      },
    }));
  }

  /** Retain a bounded tombstone so late PCM cannot recreate a current snapshot. */
  /** @param {string} responseId */
  _retirePlaybackResponse(responseId) {
    if (!responseId) return;
    const snapshot = this._playbackByResponse.get(responseId);
    // A terminal response that never reached the worklet's first sample is
    // unheard only once a drain/clear/retirement boundary is final.  Do this
    // before dropping the snapshot so a late started event cannot revive it.
    this._settleDeferredPlaybackTerminal(
      responseId,
      this._heardResponses.has(responseId),
    );
    this._playbackByResponse.delete(responseId);
    if (snapshot && snapshot.responseEpoch !== null) {
      this._playbackPolicyByResponseEpoch.delete(snapshot.responseEpoch);
      this._tombstoneResponseEpoch(snapshot.responseEpoch);
    }
    this._stalePlaybackResponses.delete(responseId);
    this._stalePlaybackResponses.add(responseId);
    while (this._stalePlaybackResponses.size > MAX_PLAYBACK_RESPONSE_TOMBSTONES) {
      const oldest = this._stalePlaybackResponses.values().next().value;
      if (!oldest) break;
      this._stalePlaybackResponses.delete(oldest);
    }
  }

  /**
   * Mark every response invalidated by one generation clear. A worklet clear
   * may beat response.done during barge-in. Keep the exact cleared outcome
   * until the protocol terminal settles the provisional ChatView transaction;
   * otherwise the later cancellation appears stale and its row never closes.
   */
  /** @param {number} clearedGeneration */
  /** @param {string} [reason] */
  _retireClearedPlaybackGenerations(clearedGeneration, reason = "clear") {
    if (!Number.isSafeInteger(clearedGeneration)) return;
    for (const [responseId, snapshot] of [...this._playbackByResponse.entries()]) {
      if (Number(snapshot?.generation) < clearedGeneration) {
        snapshot.playbackDrained = true;
        snapshot.playbackCleared = true;
        snapshot.clearReason = reason;
        snapshot.drainAudible = this._heardResponses.has(responseId);
        const settled = this._settleDeferredPlaybackTerminal(
          responseId,
          snapshot.drainAudible,
        );
        if (snapshot.terminal || settled) this._retirePlaybackResponse(responseId);
      }
    }
  }

  /** @param {Set<string>} set @param {string} value @param {number} limit */
  _rememberBounded(set, value, limit) {
    if (!value || set.has(value)) return false;
    set.add(value);
    while (set.size > limit) {
      const oldest = set.values().next().value;
      if (!oldest) break;
      set.delete(oldest);
    }
    return true;
  }

  /**
   * Dispatch a response terminal exactly once after local playback has a
   * definitive outcome.  `response.done` itself is protocol completion, not
   * evidence that primed PCM was heard.
   * @param {string} responseId
   * @param {boolean} audible
   * @returns {boolean}
   */
  _settleDeferredPlaybackTerminal(responseId, audible) {
    if (!responseId) return false;
    const snapshot = this._playbackByResponse.get(responseId);
    const detail = snapshot?.pendingTerminalDetail;
    if (!snapshot || !detail || snapshot.terminalSettled) return false;
    snapshot.terminalSettled = true;
    snapshot.pendingTerminalDetail = null;
    this.dispatchEvent(new CustomEvent("response-finished", {
      detail: { ...detail, audible: audible === true },
    }));
    return true;
  }

  /** @param {Record<string, any>} detail */
  _dispatchResponseFinished(detail) {
    this.dispatchEvent(new CustomEvent("response-finished", { detail }));
  }

  /** @param {string} responseId */
  _finishPlaybackResponse(responseId) {
    if (responseId && this._stalePlaybackResponses.has(responseId)) return;
    const snapshot = this._playbackSnapshot(responseId);
    if (snapshot.ended) return;
    snapshot.ended = true;
    this._playbackNode?.port.postMessage({
      kind: "end",
      generation: snapshot.generation,
      streamId: responseId,
      responseEpoch: snapshot.responseEpoch,
      sourceSampleRate: snapshot.sourceSampleRate,
    });
  }

  /** Clear a generation once; late response chunks retain their older tag. */
  /** @param {string} reason */
  _invalidatePlayback(reason) {
    if (this._playbackGenerationInvalidated) return;
    this._playbackGeneration += 1;
    this._playbackGenerationInvalidated = true;
    this._playbackNode?.port.postMessage({
      kind: "clear",
      generation: this._playbackGeneration,
      reason,
    });
    this.dispatchEvent(new CustomEvent("pipeline-metric", {
      detail: {
        stage: "playback",
        status: "cleared",
        source: "browser",
        detail: { generation: this._playbackGeneration, reason },
      },
    }));
  }

  /** @param {Error} error */
  _rejectInitialConfig(error) {
    if (this._configAckTimer) clearTimeout(this._configAckTimer);
    this._configAckTimer = 0;
    const reject = this._configAckReject;
    this._configAckResolve = null;
    this._configAckReject = null;
    reject?.(error);
  }

  /** Full assistant transcript so far for a response: the completed segments
   *  plus the in-progress one, all space-joined.
   *  @param {string} rid @returns {string} */
  _asstDisplay(rid) {
    const full = this._asstFullByResp.get(rid) || "";
    const seg = this._asstTranscriptByResp.get(rid) || "";
    if (!seg) return full;
    return full ? `${full} ${seg}` : seg;
  }

  _markPlaybackStarted() {
    if (this._status === "ai-speaking") return;
    if (this._status === "closed" || this._status === "error") return;
    this._setStatus("ai-speaking");
  }

  /**
   * Full handshake. Resolves once the WS is open AND the audio pipeline is
   * ready to send/receive samples.
   * @returns {Promise<void>}
   */
  async connect() {
    if (this._ws) throw new Error("Already connected");

    let connectUrl;
    if (this._directUrl) {
      // Direct mode: no load balancer, no /session POST — dial the realtime
      // endpoint straight away (e.g. a local s2s server).
      connectUrl = this._directUrl;
      this._setStatus("connecting");
    } else {
      if (!this._sessionUrl) {
        throw new Error("No session endpoint or direct URL configured");
      }
      this._setStatus("creating-session");
      const { grant, waited } = await this._createSessionOrQueue();
      if (this._closed) throw _codedError("connect aborted", "aborted");
      // If we waited in line, don't dial until the user explicitly joins — this
      // keeps a freed slot from being spent on someone who stepped away, and the
      // click is a fresh gesture (re-arms the AudioContext on iOS).
      if (waited) {
        await this._awaitJoin(grant);
        if (this._closed) throw _codedError("connect aborted", "aborted");
      }
      this.dispatchEvent(new CustomEvent("session", { detail: { info: grant } }));
      connectUrl = grant.connectUrl;
      this._setStatus("connecting");
    }

    // Acquire the mic now — only once a slot is actually ours. The caller primed
    // permission up front, so this is silent and the 'in use' indicator lights
    // only for a real, connecting session (never during a queue wait).
    if (!this.options.micStream && this._acquireMic) {
      this.options.micStream = await this._acquireMic();
    }

    // Spin up the AudioContext + worklets in parallel with the WS dial.
    const configReady = this._waitForInitialConfigAck();
    const audioReady = this._setupAudio();
    const wsReady = this._openWebSocket(connectUrl);
    try {
      await Promise.all([audioReady, wsReady, configReady]);
    } catch (error) {
      await this.close();
      throw error;
    }
  }

  /**
   * POST the session handshake; if the pool is busy, wait in the queue (polling
   * position) until a slot is claimed. Resolves to a grant plus whether we had to
   * wait (which decides if an explicit join is required before dialing).
   * @returns {Promise<{ grant: WsSessionInfo, waited: boolean }>}
   */
  async _createSessionOrQueue() {
    const first = await this._postSession();
    if (first.state === "queued") {
      this._setStatus("queued");
      const grant = await this._pollQueue(first);
      return { grant, waited: true };
    }
    return { grant: first.grant, waited: false };
  }

  /**
   * Hold at the front of the line until the user clicks join (resolves the gate)
   * or the grant lapses. Announces "your-turn" + a deadline the UI counts down.
   * @param {WsSessionInfo} grant
   * @returns {Promise<void>}
   */
  _awaitJoin(grant) {
    // The LB reclaims an unclaimed slot at its pending timeout; expire the gate a
    // touch earlier so we never dial a session the LB just reaped.
    const windowS = Math.max(3, (grant.pendingTimeoutS || 60) - 3);
    this._setStatus("your-turn");
    this.dispatchEvent(
      new CustomEvent("ready-to-join", { detail: { info: grant, expiresSec: windowS } }),
    );
    return new Promise((resolve, reject) => {
      this._joinResolve = resolve;
      this._joinReject = reject;
      this._joinTimer = setTimeout(() => {
        this._joinResolve = null;
        this._joinReject = null;
        reject(_codedError("Your spot expired", "join-expired"));
      }, windowS * 1000);
    });
  }

  /** Accept the held slot and let connect() proceed to dial. Called from the
   *  "Join now" click, so it's a user gesture: re-resume the AudioContext, which
   *  iOS may have suspended while we waited. */
  join() {
    if (this._joinTimer) {
      clearTimeout(this._joinTimer);
      this._joinTimer = 0;
    }
    try {
      void this.options.audioContext?.resume();
    } catch {
      // best-effort; _setupAudio resumes again
    }
    const resolve = this._joinResolve;
    this._joinResolve = null;
    this._joinReject = null;
    resolve?.();
  }

  /**
   * POST /session once. Returns either a granted session or a queue ticket.
   * @returns {Promise<{ state: "granted", grant: WsSessionInfo } | { state: "queued", queueId: string, position: number, pollIntervalS: number }>}
   */
  async _postSession() {
    const url = this._sessionUrl;
    console.log("[ws] POST", url);
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    if (response.status === 402) {
      // The session proxy refused: today's per-tier time budget is spent. Surface
      // it as a typed error so the UI shows the limit modal, not a crash.
      const body = await response.json().catch(() => ({}));
      throw _codedError("Daily conversation limit reached", "limit", { tier: body?.tier });
    }
    if (response.status === 503) {
      const body = await response.json().catch(() => ({}));
      if (body?.state === "at_capacity") {
        throw _codedError("The queue is full — try again shortly.", "queue-full");
      }
      throw new Error("/session failed (503)");
    }
    if (!response.ok) {
      const text = await response.text().catch(() => "");
      throw new Error(`/session failed (${response.status}): ${text}`);
    }
    const json = await response.json();
    if (json.state === "queued") {
      return {
        state: "queued",
        queueId: json.queue_id,
        position: json.position,
        pollIntervalS: json.poll_interval_s,
      };
    }
    return { state: "granted", grant: this._parseGrant(json) };
  }

  /**
   * Poll the waiting queue until this ticket claims a slot. Emits `queue` events
   * ({ position }) as the line advances. Throws on limit (402), expiry (404), or
   * close(). Transient network/5xx blips are ignored and retried next tick.
   * @param {{ queueId: string, position: number, pollIntervalS: number }} ticket
   * @returns {Promise<WsSessionInfo>}
   */
  async _pollQueue(ticket) {
    const intervalMs = Math.max(1, ticket.pollIntervalS || 2) * 1000;
    this._queueId = ticket.queueId;
    this._emitQueue(ticket.position);

    while (true) {
      await this._queueSleep(intervalMs);
      if (this._closed) throw _codedError("queue wait aborted", "aborted");

      let response;
      try {
        response = await fetch(`api/queue/${encodeURIComponent(this._queueId)}`, {
          headers: { "Content-Type": "application/json" },
        });
      } catch {
        continue; // network blip — keep our place, retry next tick
      }

      if (response.status === 402) {
        const body = await response.json().catch(() => ({}));
        throw _codedError("Daily conversation limit reached", "limit", { tier: body?.tier });
      }
      if (response.status === 404) {
        throw _codedError("Queue timed out", "queue-expired");
      }
      if (!response.ok) continue; // 502/503 — transient, retry

      const json = await response.json().catch(() => null);
      if (!json) continue;
      if (json.state === "queued") {
        this._emitQueue(json.position);
        continue;
      }
      // Reached the front and claimed a slot.
      this._queueId = "";
      return this._parseGrant(json);
    }
  }

  /** @param {number} position */
  _emitQueue(position) {
    this.dispatchEvent(
      new CustomEvent("queue", { detail: { position, queueId: this._queueId } }),
    );
  }

  /** A sleep that close() can cut short so a queued client tears down promptly.
   *  @param {number} ms */
  _queueSleep(ms) {
    return new Promise((resolve) => {
      this._queueWake = resolve;
      this._queueTimer = setTimeout(() => {
        this._queueWake = null;
        resolve();
      }, ms);
    });
  }

  /** @param {any} json @returns {WsSessionInfo} */
  _parseGrant(json) {
    return {
      sessionId: json.session_id,
      connectUrl: json.connect_url,
      websocketUrl: json.websocket_url,
      sessionToken: json.session_token,
      pendingTimeoutS: json.pending_timeout_s,
      tier: json.tier,
      limited: json.limited,
      heartbeatSec: json.heartbeatSec,
      remainingSec: json.remainingSec,
    };
  }

  async _setupAudio() {
    // Prefer a context the caller already created + resumed inside the tap
    // gesture (required on iOS). Fall back to creating one here for callers
    // that don't (desktop is lenient about the gesture timing).
    // Most desktops give us 48 kHz, mobiles can give 44.1/24/16 kHz; the
    // capture worklet handles any rate (linear interp fallback).
    const ctx = this.options.audioContext ?? new AudioContext({ latencyHint: "interactive" });
    this._ctx = ctx;

    // Resume if still suspended. This is best-effort here — on iOS the resume
    // that actually counts is the one the caller did synchronously on tap.
    if (ctx.state === "suspended") {
      try {
        await ctx.resume();
      } catch (err) {
        console.warn("[ws] AudioContext resume failed:", err);
      }
    }

    // The worklets live at the repo root, one level up from this module.
    const base = new URL("../worklets/", import.meta.url);
    await ctx.audioWorklet.addModule(new URL("audio-playback.js?v=16-epoch-continuity", base).href);
    const aec3 = await loadAec3Worklet(ctx);
    if (!aec3.available) {
      await ctx.audioWorklet.addModule(new URL("mic-capture.js?v=12-aec3-fallback", base).href);
    }

    const micTrack = this.options.micStream?.getAudioTracks?.()[0];
    const micSettings = micTrack?.getSettings?.() || {};
    const microphoneId = micSettings.deviceId || micTrack?.label || "default-microphone";
    const outputId = typeof ctx.sinkId === "string" && ctx.sinkId
      ? ctx.sinkId
      : "default-output";
    this._echoDevicePair = `${microphoneId}::${outputId}`;
    this._echoCalibration = normalizeEchoCalibration(
      this._echoCalibrations[this._echoDevicePair],
    );
    const outputLatencyMs = Number.isFinite(ctx.outputLatency)
      ? Math.max(0, ctx.outputLatency * 1000)
      : 0;

    const captureNode = new AudioWorkletNode(ctx, aec3.processorName, {
      numberOfInputs: 2,
      numberOfOutputs: 0,
      processorOptions: aec3ProcessorOptions(aec3, { chunkMs: MIC_CHUNK_MS }),
    });
    captureNode.port.onmessage = (e) => {
      if (this._closed) return;
      const data = e.data;
      if (data instanceof ArrayBuffer) {
        this._onMicChunk(data);
      } else if (data?.kind === "level") {
        // Raw pre-gate mic RMS for the Settings meter.
        this.dispatchEvent(new CustomEvent("input-level", { detail: { rms: data.rms } }));
      } else if (data?.kind === "echo_metric") {
        this.dispatchEvent(new CustomEvent("pipeline-metric", {
          detail: {
            stage: "echo_guard",
            status: data.suppressing ? "suppressing" : data.doubleTalk ? "double_talk" : "monitoring",
            source: "browser",
            detail: {
              mode: data.mode,
              requested_mode: data.requestedMode || this._echoGuard,
              effective_mode: data.mode,
              native_aec: !!data.nativeAec,
              module_available: !!data.moduleAvailable,
              reference_wired: !!data.referenceWired,
              correlation: Number(data.correlation || 0),
              residual: Number.isFinite(data.residual) ? Number(data.residual) : null,
              residual_energy: Number.isFinite(data.residualEnergy) ? Number(data.residualEnergy) : null,
              erle_db: Number.isFinite(data.erleDb) ? Number(data.erleDb) : null,
              lag_ms: Number.isFinite(data.lagMs) ? Number(data.lagMs) : null,
              output_latency_ms: outputLatencyMs,
              device_pair: this._echoDevicePair,
              model_ready: !!data.modelReady,
              prediction_confidence: Number.isFinite(data.predictionConfidence)
                ? Number(data.predictionConfidence)
                : null,
              candidate_ms: Number(data.candidateMs || 0),
              suppressed_ms: Number(data.suppressedMs || 0),
              double_talk: data.doubleTalk == null ? null : !!data.doubleTalk,
              double_talk_source: data.doubleTalkSource || null,
              playback_active: !!data.playbackActive,
            },
          },
        }));
      } else if (data?.kind === "aec3_status") {
        this._aec3Status = {
          ...data,
          devicePair: this._echoDevicePair,
          outputLatencyMs,
          calibration: this._echoCalibration,
          loaderAvailable: aec3.available,
          loaderReason: aec3.reason || "",
        };
        this.dispatchEvent(new CustomEvent("echo-status", { detail: this._aec3Status }));
      } else if (data?.kind === "aec3_error") {
        this.dispatchEvent(new CustomEvent("echo-status", {
          detail: {
            available: false,
            requestedMode: this._echoGuard,
            effectiveMode: this._echoGuard === "strict" ? "strict-fallback" : "native",
            devicePair: this._echoDevicePair,
            outputLatencyMs,
            calibration: this._echoCalibration,
            error: data.error || "AEC3 worklet failed",
          },
        }));
      }
    };
    // Push the initial gate config now that the worklet exists.
    captureNode.port.postMessage({ kind: "gate", ...this._noiseGate });
    const nativeAec = !!micTrack?.getSettings?.().echoCancellation;
    captureNode.port.postMessage({ kind: "echo_guard", mode: this._echoGuard, nativeAec });
    captureNode.port.postMessage({
      kind: "echo_calibration",
      ...this._echoCalibration,
      outputLatencyMs,
    });
    this._captureNode = captureNode;

    const micSrc = ctx.createMediaStreamSource(this.options.micStream);
    micSrc.connect(captureNode);
    this._micSrc = micSrc;

    // Mic analyser: tap the mic in parallel with the worklet so we get the
    // raw (un-resampled, un-clipped) signal for the visualiser.
    const micAnalyser = ctx.createAnalyser();
    micAnalyser.fftSize = VIS_FFT_SIZE;
    micAnalyser.smoothingTimeConstant = 0;
    micSrc.connect(micAnalyser);
    this._micAnalyser = micAnalyser;

    const playbackNode = new AudioWorkletNode(ctx, "audio-playback", {
      numberOfInputs: 0,
      numberOfOutputs: 1,
      outputChannelCount: [1],
    });
    playbackNode.port.postMessage({
      kind: "config",
      inputRate: this._playbackSampleRate,
      generation: this._playbackGeneration,
    });
    playbackNode.port.onmessage = (e) => this._onPlaybackMessage(e.data);

    // Output analyser sits between the playback worklet and the speakers.
    const outAnalyser = ctx.createAnalyser();
    outAnalyser.fftSize = VIS_FFT_SIZE;
    outAnalyser.smoothingTimeConstant = 0.3;
    playbackNode.connect(outAnalyser);
    playbackNode.connect(captureNode, 0, 1);
    outAnalyser.connect(ctx.destination);
    this._outAnalyser = outAnalyser;
    this._playbackNode = playbackNode;

    this._visualiser = new OrbVisualiser(micAnalyser, outAnalyser, () => this._aiSpeaking);
    this._visualiser.start();
  }

  /** @param {string} connectUrl */
  _openWebSocket(connectUrl) {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(connectUrl);
      ws.binaryType = "arraybuffer";
      this._ws = ws;

      const onceOpen = () => {
        ws.removeEventListener("open", onceOpen);
        ws.removeEventListener("error", onceErr);
        resolve();
      };
      const onceErr = (e) => {
        ws.removeEventListener("open", onceOpen);
        ws.removeEventListener("error", onceErr);
        reject(new Error(`WebSocket failed to open: ${e?.type ?? "error"}`));
      };
      ws.addEventListener("open", onceOpen);
      ws.addEventListener("error", onceErr);

      ws.addEventListener("message", (e) => this._onWsMessage(e.data));
      ws.addEventListener("close", (e) => this._onWsClose(e));
      ws.addEventListener("error", (e) => {
        console.error("[ws] socket error", e);
      });
    });
  }

  /** @param {Record<string, any>} data */
  _onPlaybackMessage(data) {
    if (this._closed) return;
    const generation = Number(data?.generation);
    const streamId = typeof data?.streamId === "string" ? data.streamId : "";
    const playbackSnapshot = streamId ? this._playbackByResponse.get(streamId) : null;
    const queuedLifecycleAccepted = Number.isSafeInteger(generation)
      && this._acceptQueuedPlaybackLifecycle(
        streamId,
        _responseEpoch({ response_epoch: data?.responseEpoch }),
        generation,
        data?.kind,
      );
    if (Number.isSafeInteger(generation)
        && generation !== this._playbackGeneration
        && data?.kind !== "stale_chunk_rejected"
        && !queuedLifecycleAccepted) return;
    const queueDetail = {
      queued_ms: Number(data?.queuedMs || 0),
      prime_target_ms: Number(data?.primeTargetMs || 0),
      generation: Number.isSafeInteger(generation) ? generation : this._playbackGeneration,
      state: data?.state || "unknown",
      underruns: Number(data?.underruns || 0),
      reprimes: Number(data?.reprimes || 0),
      reprime_target_ms: Number(data?.reprimeTargetMs || 0),
      observed_chunk_gap_ms: Number(data?.observedChunkGapMs || 0),
      max_observed_chunk_gap_ms: Number(data?.maxObservedChunkGapMs || 0),
      stale_chunks: Number(data?.staleChunks || 0),
      clears: Number(data?.clears || 0),
      // `Number(null)` is zero. Treat a worklet clear/drain without a single
      // response owner as unscoped instead of incorrectly rejecting it as an
      // obsolete epoch after any real response has played.
      response_epoch: _responseEpoch({ response_epoch: data?.responseEpoch }),
      continuity_mode: playbackSnapshot?.continuityMode || this._playbackContinuityMode,
      provider: playbackSnapshot?.provider || this._acknowledgedPlaybackConfig.provider || null,
      profile_id: playbackSnapshot?.profileId || this._acknowledgedPlaybackConfig.profileId || null,
      adaptive_prime_target_ms: (
        data?.reprimeTargetMs !== null
        && data?.reprimeTargetMs !== undefined
        && Number.isFinite(Number(data.reprimeTargetMs))
      )
        ? Number(data.reprimeTargetMs)
        : playbackSnapshot && Number.isFinite(Number(playbackSnapshot.reprimeMs))
          ? Number(playbackSnapshot.reprimeMs)
          : streamId
            ? null
            : this._adaptivePlaybackPrimeMs,
      // Do not attach the newly selected backend/profile's global health to a
      // late metric from an older response snapshot.
      provider_unsustainable: playbackSnapshot
        ? playbackSnapshot.playbackUnsustainable === true
        : null,
      queue_empty: data?.queueEmpty !== false,
    };
    if (this._isStaleResponseEpoch(queueDetail.response_epoch)
        && !queuedLifecycleAccepted
        && !(data?.kind === "cleared" && queueDetail.generation === this._playbackGeneration)) return;
    if (data?.kind === "started") {
      const snapshot = streamId ? this._playbackByResponse.get(streamId) : null;
      // A completion may precede first rendered PCM. It remains valid only if
      // the snapshot survives; cancelled/stale responses were retired and must
      // never be revived by a delayed worklet message.
      if (!streamId || this._stalePlaybackResponses.has(streamId) || !snapshot) return;
      this._heardResponses.add(streamId);
      if (!snapshot.playbackAcked) {
        snapshot.playbackAcked = true;
        this._send({
          type: "pipeline.playback.started",
          response_id: streamId,
          response_epoch: snapshot.responseEpoch,
        });
      }
      // response.done may have arrived while this PCM was still primed. The
      // first rendered sample is the only browser proof that its provisional
      // chat transaction should be retained.
      this._settleDeferredPlaybackTerminal(streamId, true);
      // A worklet message can cross a barge-in on the main-thread queue. Its
      // exact old response still needs an acknowledgement/history settlement,
      // but it must not overwrite the successor's user-speaking state or
      // first-playback metrics. Only the current playback generation owns
      // global presentation state.
      if (queueDetail.generation !== this._playbackGeneration) return;
      this._aiSpeaking = true;
      this._markPlaybackStarted();
      if (!this._firstPlaybackReported) {
        this._firstPlaybackReported = true;
        const elapsed = this._speechStoppedAtMs == null ? null : Math.max(0, performance.now() - this._speechStoppedAtMs);
        this.dispatchEvent(new CustomEvent("pipeline-metric", {
          detail: {
            stage: "playback",
            status: "first_audio",
            source: "browser",
            elapsed_ms: elapsed,
            detail: {
              ...queueDetail,
              first_playback_ms: elapsed,
              reprime: !!data.reprime,
              stream_id: streamId || null,
            },
          },
        }));
      }
      return;
    }
    if (data?.kind === "stats") {
      this.dispatchEvent(new CustomEvent("pipeline-metric", {
        detail: {
          stage: "playback",
          status: "queue",
          source: "browser",
          detail: { ...queueDetail, played: Number(data.played || 0) },
        },
      }));
      return;
    }
    if (data?.kind === "underrun") {
      if (playbackSnapshot) playbackSnapshot.hadUnderrun = true;
      if (this._playbackFeedbackMatches(playbackSnapshot)) {
        this._healthyPlaybackResponses = 0;
      }
      this._raiseAdaptiveReserve(data, playbackSnapshot);
      this.dispatchEvent(new CustomEvent("pipeline-metric", {
        detail: { stage: "playback", status: "underrun", source: "browser", detail: queueDetail },
      }));
      return;
    }
    if ((data?.kind === "primed" || data?.kind === "reprimed")
        && playbackSnapshot?.hadUnderrun) {
      // The cadence that caused an underrun is known only when a later chunk
      // arrives. Re-evaluate on the re-prime diagnostic so a first gap beyond
      // the two-second ceiling is reported without waiting for another loss.
      this._raiseAdaptiveReserve(data, playbackSnapshot);
      queueDetail.provider_unsustainable = playbackSnapshot.playbackUnsustainable === true;
    }
    if (data?.kind === "cleared") {
      // A shared FIFO can contain multiple completed-but-unheard response
      // streams. One generation clear invalidates all of them, even though the
      // worklet's diagnostic can name only one active stream.
      this._retireClearedPlaybackGenerations(
        queueDetail.generation,
        typeof data.reason === "string" && data.reason ? data.reason : "clear",
      );
      this._aiSpeaking = false;
      if (this._status === "ai-speaking") {
        this._setStatus(this._responseActive() ? "processing" : "connected");
      }
    }
    if (data?.kind === "drained") {
      if (!data?.cleared && !playbackSnapshot?.hadUnderrun
          && this._playbackFeedbackMatches(playbackSnapshot)) {
        this._healthyPlaybackResponses += 1;
        if (this._healthyPlaybackResponses >= 3 && this._adaptivePlaybackPrimeMs > this._playbackPrimeMs) {
          this._adaptivePlaybackPrimeMs = Math.max(this._playbackPrimeMs, this._adaptivePlaybackPrimeMs - 80);
          this._healthyPlaybackResponses = 0;
        }
      }
      if (data?.queueEmpty !== false) {
        this._aiSpeaking = false;
        // A barge-in sets user-speaking before the clear reaches the worklet. Do
        // not let the resulting drained event overwrite that newer microphone
        // state; only retire an active playback status here.
        if (this._status === "ai-speaking") {
          this._setStatus(this._responseActive() ? "processing" : "connected");
        }
      }
      // A terminal response stays retained through the final `started` event
      // above. Draining is the definitive local playback boundary; late worklet
      // messages after it must not create a second acknowledgement.
      if (streamId) {
        // Drained and cleared are the definitive negative outcome when no
        // `started` message was seen. A later worklet start from this retired
        // stream is rejected by the response tombstone.
        const snapshot = this._playbackByResponse.get(streamId);
        if (snapshot) {
          snapshot.playbackDrained = true;
          snapshot.drainAudible = this._heardResponses.has(streamId);
        }
        const settled = this._settleDeferredPlaybackTerminal(
          streamId,
          this._heardResponses.has(streamId),
        );
        // Audio can drain before response.done arrives. Keep that exact
        // snapshot until the protocol terminal settles the provisional chat;
        // otherwise tombstoning now would make the later terminal look stale.
        if (!snapshot || snapshot.terminal || settled) {
          this._retirePlaybackResponse(streamId);
        }
      }
    }
    if (["primed", "reprimed", "drained", "cleared", "stale_chunk_rejected"].includes(data?.kind)) {
      this.dispatchEvent(new CustomEvent("pipeline-metric", {
        detail: {
          stage: "playback",
          status: data.kind,
          source: "browser",
          detail: {
            ...queueDetail,
            forced: !!data.forced,
            reason: data.reason || null,
            cleared: !!data.cleared,
            stream_id: typeof data.streamId === "string" && data.streamId ? data.streamId : null,
            rejected_kind: data.rejectedKind || null,
            received_generation: Number.isSafeInteger(Number(data.receivedGeneration))
              ? Number(data.receivedGeneration)
              : null,
          },
        },
      }));
    }
  }

  /**
   * Mic worklet just sent us a ~40 ms PCM16 16 kHz mono chunk.
   * Base64-encode and forward via the WS.
   * @param {ArrayBuffer} pcm16Buffer
   */
  _onMicChunk(pcm16Buffer) {
    if (this._closed) return;
    if (!this._ws || this._ws.readyState !== WebSocket.OPEN) return;
    if (!this._sessionConfigured) return; // Server rejects audio before session.update.
    if (this._muted) return;
    const b64 = base64FromArrayBuffer(pcm16Buffer);
    this._send({ type: "input_audio_buffer.append", audio: b64 });
  }

  /**
   * @param {string | ArrayBuffer | Blob} raw
   */
  async _onWsMessage(raw) {
    if (this._closed) return;
    let text;
    if (typeof raw === "string") {
      text = raw;
    } else if (raw instanceof ArrayBuffer) {
      text = new TextDecoder("utf-8").decode(raw);
    } else if (raw instanceof Blob) {
      text = await raw.text();
    } else {
      return;
    }
    if (this._closed) return;

    let event;
    try {
      event = JSON.parse(text);
    } catch {
      return;
    }

    const type = event?.type;
    if (typeof type !== "string") return;
    // Opt-in event tracing for diagnosing turn/transcript issues. Enable with
    // `localStorage.setItem("s2s.debug", "1")` in the browser console.
    if (this._debug) {
      const extra = type.startsWith("conversation.item.input_audio_transcription")
        ? ` item=${event.item_id} ci=${event.content_index} ${event.delta ?? event.transcript ?? ""}`
        : type.startsWith("response.")
          ? ` resp=${event.response_id ?? event.response?.id ?? ""} status=${event.response?.status ?? ""} ${event.transcript ?? ""}`
          : "";
      console.debug(`[ws] ${type}${extra}`);
    }

    switch (type) {
      case "session.created":
        // Endpoint validation must finish before session.update enables mic audio.
        // Browser clients can prove that `started` means a sample reached the
        // local playback worklet. Nonbrowser peers retain first-delivered-PCM
        // settlement on the server for backward compatibility.
        this._send({ type: "pipeline.playback.capability", rendered_playback_ack: true });
        this.updateLocalPipeline(
          this.options.pipelineConfig || {},
          this.options.playbackConfig || null,
        );
        break;

      case "session.updated":
        // Acknowledged by server, nothing to do.
        break;

      case "pipeline.runtime":
        this.dispatchEvent(new CustomEvent("backend-runtime", { detail: event.runtime || {} }));
        break;

      case "pipeline.input_audio.speech_started": {
        const inputEpoch = _inputEpoch(event);
        if (!this._acceptInputEpoch(inputEpoch)) break;
        this._pendingSpeechStartDecision = {
          inputEpoch,
          effectiveInterrupt: event.effective_interrupt === true,
          reason: typeof event.reason === "string" ? event.reason : "speech_started",
        };
        this.dispatchEvent(new CustomEvent("pipeline-metric", {
          detail: {
            stage: "playback",
            status: event.effective_interrupt === true ? "barge_in" : "retained",
            source: "backend",
            detail: {
              input_epoch: inputEpoch,
              response_epoch: _responseEpoch(event),
              effective_interrupt: event.effective_interrupt === true,
              reason: event.reason || null,
            },
          },
        }));
        break;
      }

      case "pipeline.response": {
        // The local server emits this immediately after OpenAI-compatible
        // response.created. Only it carries the durable response/input epochs.
        const responseId = typeof event.response_id === "string" ? event.response_id : "";
        const responseEpoch = _responseEpoch(event);
        const frozenRate = _responseOutputSampleRate(event.output_sample_rate);
        this._captureLifecyclePlaybackPolicy(
          responseEpoch,
          frozenRate,
          responseId,
          _lifecyclePlaybackPolicy(event, frozenRate),
        );
        // Pending ownership is intentionally emitted before an OpenAI response
        // ID exists. Track its epoch for diagnostics and stale filtering first;
        // bind playback only once a response ID is actually available.
        if (responseId) {
          if (!this._bindPlaybackResponseEpoch(responseId, responseEpoch)) break;
        } else if (!this._acceptResponseEpoch(responseEpoch)) {
          break;
        }
        const inputEpoch = _inputEpoch(event);
        this._acceptInputEpoch(inputEpoch);
        const lifecycleState = event.lifecycle_state || event.state || "pending";
        this.dispatchEvent(new CustomEvent("pipeline-metric", {
          detail: {
            stage: "response", status: "lifecycle", source: "backend",
            detail: {
              response_id: responseId, response_epoch: responseEpoch,
              input_epoch: inputEpoch,
              lifecycle_state: lifecycleState,
              output_sample_rate: frozenRate,
              supersession_reason: event.reason || null,
            },
          },
        }));
        // A direct-audio generation can terminate before it ever reaches
        // response.created. There is then no stock response.done to free the
        // browser's create guard or settle the provisional user row. The local
        // lifecycle epoch is authoritative for that exceptional path only.
        if (!responseId && responseEpoch !== null && _isTerminalResponseLifecycleState(lifecycleState)) {
          this._playbackPolicyByResponseEpoch.delete(responseEpoch);
          this._tombstoneResponseEpoch(responseEpoch);
          const terminalKey = String(responseEpoch);
          if (!this._rememberBounded(
            this._terminalResponseEpochs,
            terminalKey,
            MAX_PLAYBACK_RESPONSE_TOMBSTONES,
          )) {
            break;
          }
          this._createInFlight = false;
          if (this._status === "processing" && !this._aiSpeaking) {
            this._setStatus("connected");
          }
          this.dispatchEvent(new CustomEvent("response-finished", {
            detail: {
              responseId: "",
              status: lifecycleState,
              audible: false,
              transcript: "",
              responseEpoch,
              committed: false,
            },
          }));
          if (!this._responseActive()) this._resolveResponseIdleWaiters();
          this._flushQueuedCreate();
        }
        break;
      }

      case "input_audio_buffer.speech_started":
        {
          const inputEpoch = _inputEpoch(event);
          if (!this._acceptInputEpoch(inputEpoch)) break;
          const decision = this._pendingSpeechStartDecision;
          const hasDecision = decision !== null
            && (decision.inputEpoch === null || inputEpoch === null || decision.inputEpoch === inputEpoch);
          this._pendingSpeechStartDecision = null;
          // New local servers send the authoritative decision immediately
          // before this stock event. With interruption disabled, capture may
          // continue while the old audible response keeps its generation.
          // Older compatible peers omit the extension and retain the previous
          // unconditional barge-in behavior.
          if (!hasDecision || decision.effectiveInterrupt) {
            this._invalidatePlayback(hasDecision ? decision.reason : "barge-in");
          }
        }
        this._setStatus("user-speaking");
        this.dispatchEvent(new CustomEvent("turn-state", { detail: { status: "speech_started" } }));
        this.dispatchEvent(new CustomEvent("pipeline-metric", {
          detail: { stage: "mic", status: "speaking", source: "browser", detail: {} },
        }));
        break;

      case "input_audio_buffer.speech_stopped":
        if (!this._acceptInputEpoch(_inputEpoch(event))) break;
        this._playbackGenerationInvalidated = false;
        if (this._status === "user-speaking") this._setStatus("processing");
        this._speechStoppedAtMs = performance.now();
        this._firstPlaybackReported = false;
        this.dispatchEvent(new CustomEvent("turn-state", { detail: { status: "speech_stopped" } }));
        this.dispatchEvent(new CustomEvent("pipeline-metric", {
          detail: { stage: "mic", status: "captured", source: "browser", detail: {} },
        }));
        break;

      case "response.created": {
        const responseEpoch = _responseEpoch(event);
        if (!this._acceptResponseEpoch(responseEpoch)) break;
        // A response now owns the slot — count it and clear our create guard
        // (this confirms either our create or a server-initiated one).
        this._openResponses++;
        this._createInFlight = false;
        this._playbackSnapshot(
          typeof event.response?.id === "string" ? event.response.id : "",
          responseEpoch,
        );
        if (this._status === "connected" || this._status === "user-speaking") {
          this._setStatus("processing");
        }
        break;
      }

      case "response.output_item.added":
        if (this._status === "connected" || this._status === "user-speaking") {
          this._setStatus("processing");
        }
        break;

      case "response.audio.delta":
      case "response.output_audio.delta": {
        const rid = typeof (event.response_id ?? event.response?.id) === "string"
          ? (event.response_id ?? event.response?.id)
          : "";
        const responseEpoch = _responseEpoch(event) ?? this._playbackByResponse.get(rid)?.responseEpoch ?? null;
        this._pushAudioDelta(event.delta, rid, responseEpoch);
        break;
      }

      case "response.audio.done":
      case "response.output_audio.done": {
        const rid = typeof (event.response_id ?? event.response?.id) === "string"
          ? (event.response_id ?? event.response?.id)
          : "";
        if (!this._isStaleResponseEpoch(_responseEpoch(event))) this._finishPlaybackResponse(rid);
        break;
      }

      case "response.content_part.added": {
        // A declared audio part is not audible until the worklet renders it.
        break;
      }

      case "response.done": {
        const status = event.response?.status ?? "completed";
        const responseId = typeof event.response?.id === "string" ? event.response.id : "";
        const responsePlayback = responseId
          ? this._playbackByResponse.get(responseId)
          : null;
        const responseEpoch = _responseEpoch(event) ?? responsePlayback?.responseEpoch ?? null;
        if (responseId && !this._rememberBounded(
          this._terminalResponseIds,
          responseId,
          MAX_PLAYBACK_RESPONSE_TOMBSTONES,
        )) {
          break;
        }
        // This response freed the slot (completion OR cancellation both arrive
        // as response.done). Decrement and, if a create was waiting, replay it.
        this._openResponses = Math.max(0, this._openResponses - 1);
        if (this._status === "processing" && !this._aiSpeaking) {
          this._setStatus("connected");
        }
        // A response closes here for BOTH normal completion and cancellation
        // (the s2s server signals a speculative-turn interrupt as
        // `response.done` with status "cancelled" — there is no separate
        // `response.cancelled` event). Surface the id + status so the UI can
        // drop a cancelled response's transcript and commit a completed one.
        const staleEpoch = this._isStaleResponseEpoch(responseEpoch);
        const responseRetired = responseId
          ? this._stalePlaybackResponses.has(responseId)
          : false;
        const committed = responseId
          ? this._toolCommittedResponses.has(responseId)
          : false;
        const clearedBeforeTerminal = responsePlayback?.playbackCleared === true;
        // A local generation advance is already a definitive invalidation even
        // if the asynchronous worklet `cleared` callback has not arrived yet.
        // Let the old terminal settle its provisional ChatView transaction;
        // otherwise retiring the snapshot here leaves no owner for the later
        // clear acknowledgement and the row remains stuck forever.
        const invalidatedBeforeTerminal = !!responsePlayback
          && Number(responsePlayback.generation) < this._playbackGeneration;
        if (staleEpoch && !clearedBeforeTerminal && !invalidatedBeforeTerminal) {
          this._retirePlaybackResponse(responseId);
          this._asstTranscriptByResp.delete(responseId);
          this._asstFullByResp.delete(responseId);
          if (!this._responseActive()) this._resolveResponseIdleWaiters();
          this._flushQueuedCreate();
          break;
        }
        if (status === "cancelled" || status === "canceled" || status === "failed") {
          this._tombstoneResponseEpoch(responseEpoch);
          // Barge-in already advanced the generation. Do not clear the new turn
          // a second time when the cancelled old response closes afterward.
          if (!responseRetired
              && ((status !== "failed" && !responsePlayback)
                || responsePlayback?.generation === this._playbackGeneration)) {
            this._invalidatePlayback(status === "failed" ? "response-failed" : "response-cancelled");
          }
        } else if (!responseRetired) {
          // Some compatible peers omit output_audio.done. Keep the end flush
          // idempotent and use response.done as the terminal fallback.
          this._finishPlaybackResponse(responseId);
        }
        // response.done(cancelled) may itself be the first operation that
        // advances the playback generation. Preserve that just-performed
        // invalidation as a valid cross-thread ordering boundary: a `started`
        // message already posted by the old worklet generation can still prove
        // the response became audible before the clear took effect.
        const invalidatedForTerminal = invalidatedBeforeTerminal || (
          !!responsePlayback
          && Number(responsePlayback.generation) < this._playbackGeneration
        );
        const endToEndMs = this._speechStoppedAtMs == null
          ? null
          : Math.max(0, performance.now() - this._speechStoppedAtMs);
        // A clear-before-terminal cancellation is admitted only to settle its
        // provisional UI transaction. It remains stale for current-response
        // diagnostics and cannot contribute a terminal metric to the new turn.
        if (!staleEpoch) {
          this.dispatchEvent(new CustomEvent("pipeline-metric", {
            detail: {
              stage: "response",
              status: "done",
              source: "browser",
              elapsed_ms: endToEndMs,
              detail: { end_to_end_ms: endToEndMs, response_status: status, response_epoch: responseEpoch },
            },
          }));
        }
        // Did the worklet actually render this response's first sample? This
        // remains false when response.done beats startup priming; network PCM
        // receipt alone must not make a speculative response look heard.
        const audible = responseId ? this._heardResponses.has(responseId) : false;
        // Keep the heard marker together with the retained snapshot if audio
        // has ended at the protocol layer but remains primed in the worklet.
        // Pull whatever transcript the response carries, falling back to the
        // segments we concatenated from the `*.transcript.done` events (plus any
        // in-progress delta). For an interrupted reply the response payload may
        // be empty, so this is the last chance to capture the text.
        const transcript =
          extractResponseTranscript(event.response) ||
          this._asstDisplay(responseId) ||
          "";
        // Response finished — clear both transcript accumulators for it.
        this._asstTranscriptByResp.delete(responseId);
        this._asstFullByResp.delete(responseId);
        const terminalDetail = { responseId, status, audible, transcript, responseEpoch, committed };
        const playbackAlreadyDrained = responsePlayback?.playbackDrained === true;
        const cancelled = status === "cancelled" || status === "canceled";
        const delayedRenderedProofPending = cancelled
          && !responseRetired
          && !!responsePlayback?.audioStarted
          && !playbackAlreadyDrained
          && !clearedBeforeTerminal
          && invalidatedForTerminal
          && audible === false;
        const mayStillStart = (
          !cancelled
          && !responseRetired
          && !!responsePlayback?.audioStarted
          && audible === false
        ) || delayedRenderedProofPending;
        if (mayStillStart) {
          // Leave the provisional chat transaction intact until the worklet
          // confirms first render or a later drain/clear proves it never did.
          responsePlayback.terminal = true;
          responsePlayback.pendingTerminalDetail = terminalDetail;
        } else {
          this._dispatchResponseFinished(terminalDetail);
        }
        if ((cancelled && !delayedRenderedProofPending) || responseRetired) {
          this._retirePlaybackResponse(responseId);
        } else if (playbackAlreadyDrained) {
          // The worklet's drain won the race with response.done. The terminal
          // has now been dispatched, so the retained snapshot can be retired.
          this._retirePlaybackResponse(responseId);
        } else if (!mayStillStart && !audible && !committed) {
          // The ChatView rolls this completed-but-unheard transaction back
          // immediately.  Tombstone it at the same boundary: compatible peers
          // sometimes send a late transcript frame without response_epoch, and
          // that frame must not recreate the discarded assistant row.
          this._retirePlaybackResponse(responseId);
        } else if (responsePlayback) {
          responsePlayback.terminal = true;
        }
        if (!this._responseActive()) this._resolveResponseIdleWaiters();
        // The slot is free now — replay a queued create (e.g. a tool follow-up
        // that arrived while this response was still running).
        this._flushQueuedCreate();
        break;
      }

      case "pipeline.metric": {
        const responseEpoch = _responseEpoch(event) ?? _responseEpoch(event.detail || {});
        if (this._isStaleResponseEpoch(responseEpoch)) break;
        if (event.stage === "tts" && ["failed", "runaway_aborted"].includes(event.status)
            && responseEpoch !== null) {
          const responseId = event.response_id ?? event.detail?.response_id;
          const failedSnapshot = responseId
            ? this._playbackByResponse.get(responseId)
            : [...this._playbackByResponse.values()].find(snapshot => snapshot.responseEpoch === responseEpoch);
          if (failedSnapshot?.responseEpoch === responseEpoch
              && failedSnapshot.generation === this._playbackGeneration) {
            // The owned failure metric precedes output_audio.done. Clear now,
            // before that ordinary end event could release a primed bad tail.
            this._tombstoneResponseEpoch(responseEpoch);
            this._invalidatePlayback("response-failed");
          }
        }
        const rtf = Number(event.detail?.rtf);
        if (event.stage === "tts" && event.status === "done" && Number.isFinite(rtf) && rtf > 1) {
          const responseId = typeof (event.response_id ?? event.detail?.response_id) === "string"
            ? (event.response_id ?? event.detail?.response_id)
            : "";
          const metricSnapshot = responseId
            ? this._playbackByResponse.get(responseId) ?? null
            : [...this._playbackByResponse.values()].find(
              (snapshot) => snapshot.responseEpoch === responseEpoch,
            ) ?? null;
          // A backend metric without exact response ownership cannot tune the
          // mutable browser reservoir safely.
          if (metricSnapshot) this._markPlaybackUnsustainable("backend_rtf", metricSnapshot);
        }
        this.dispatchEvent(new CustomEvent("pipeline-metric", { detail: { ...event, source: "backend" } }));
        break;
      }

      case "pipeline.config.updated": {
        this._applyAcknowledgedPlaybackConfig(event.config || {});
        this.dispatchEvent(new CustomEvent("local-pipeline-updated", { detail: event.config || {} }));
        if (!this._sessionConfigured) {
          this._sendSessionUpdate();
          this._sessionConfigured = true;
          this._resolveInitialConfig();
          if (this._status === "connecting") this._setStatus("connected");
        }
        break;
      }

      case "local.pipeline.updated": {
        this.dispatchEvent(new CustomEvent("local-pipeline-updated", { detail: event.config || {} }));
        break;
      }

      case "conversation.item.created": {
        const item = event.item || {};
        const pending = item.type === "function_call_output" ? this._toolOutputAcks.get(item.call_id) : null;
        if (pending) {
          clearTimeout(pending.timer);
          this._toolOutputAcks.delete(item.call_id);
          pending.resolve();
        }
        break;
      }

      case "response.function_call_arguments.done": {
        const name = typeof event.name === "string" ? event.name : "";
        const args = typeof event.arguments === "string" ? event.arguments : "{}";
        const callId = typeof event.call_id === "string" ? event.call_id : "";
        const responseId = typeof (event.response_id ?? event.response?.id) === "string"
          ? (event.response_id ?? event.response?.id)
          : "";
        const responsePlayback = responseId
          ? this._playbackByResponse.get(responseId)
          : null;
        const eventResponseEpoch = _responseEpoch(event);
        if (responseId && this._stalePlaybackResponses.has(responseId)) break;
        if (responsePlayback && responsePlayback.responseEpoch !== null
            && eventResponseEpoch !== null
            && responsePlayback.responseEpoch !== eventResponseEpoch) {
          break;
        }
        const responseEpoch = eventResponseEpoch ?? responsePlayback?.responseEpoch ?? null;
        if (!this._acceptResponseEpoch(responseEpoch)) break;
        if (responseId) this._rememberBounded(this._toolCommittedResponses, responseId, MAX_TOOL_CALL_TOMBSTONES);
        if (callId && !this._rememberBounded(this._toolCallTombstones, callId, MAX_TOOL_CALL_TOMBSTONES)) {
          if (this._debug) console.debug(`[ws] duplicate function call ignored (${callId})`);
          break;
        }
        if (name) {
          this.dispatchEvent(new CustomEvent("toolcall", {
            detail: { name, arguments: args, callId },
          }));
        } else {
          // A nameless call can't be executed, so no function_call_output is
          // ever sent and the model would wait forever for a result. The
          // backend shouldn't emit these; warn loudly rather than stall silently.
          console.warn(`[ws] function_call_arguments.done with no name (call_id=${callId}); cannot run tool — turn may stall`);
        }
        break;
      }

      case "conversation.item.input_audio_transcription.delta": {
        const delta = typeof event.delta === "string" ? event.delta : "";
        if (delta) {
          // `itemId` is REUSED across a speculative continuation, so the UI
          // groups both segments into one message. The delta carries the full
          // cumulative transcript so far (not an increment).
          this.dispatchEvent(
            new CustomEvent("transcript", {
              detail: {
                role: "user",
                text: delta,
                partial: true,
                itemId: typeof event.item_id === "string" ? event.item_id : "",
              },
            }),
          );
        }
        break;
      }

      case "conversation.item.input_audio_transcription.completed": {
        const transcript = typeof event.transcript === "string" ? event.transcript : "";
        if (transcript) {
          this.dispatchEvent(
            new CustomEvent("transcript", {
              detail: {
                role: "user",
                text: transcript,
                partial: false,
                itemId: typeof event.item_id === "string" ? event.item_id : "",
              },
            }),
          );
        }
        break;
      }

      case "response.audio_transcript.delta":
      case "response.output_audio_transcript.delta": {
        // Stream the assistant transcript live: accumulate the incremental
        // deltas and push the running text to the UI. Every transcribe event we
        // receive reaches the conversation, so an interrupted reply already has
        // its partial text even if the `.done` never fires.
        const rid = typeof event.response_id === "string" ? event.response_id : "";
        // Some compatible peers omit epochs on late transcript frames. A
        // response ID retired by cancellation/barge-in is still authoritative
        // in that legacy shape; never let it resurrect an unheard row.
        if (!this._acceptResponseBoundEvent(rid, _responseEpoch(event))) break;
        const delta = typeof event.delta === "string" ? event.delta : "";
        if (delta) {
          this._asstTranscriptByResp.set(rid, (this._asstTranscriptByResp.get(rid) || "") + delta);
          // Show completed segments + the segment streaming in right now.
          this.dispatchEvent(
            new CustomEvent("transcript", {
              detail: { role: "assistant", text: this._asstDisplay(rid), partial: true, responseId: rid },
            }),
          );
        }
        break;
      }

      case "response.audio_transcript.done":
      case "response.output_audio_transcript.done": {
        const rid = typeof event.response_id === "string" ? event.response_id : "";
        if (!this._acceptResponseBoundEvent(rid, _responseEpoch(event))) break;
        // This is ONE completed segment. A response can emit several; concatenate
        // them, space-separated, until response.done clears the accumulator.
        const segment =
          (typeof event.transcript === "string" && event.transcript) ||
          this._asstTranscriptByResp.get(rid) ||
          "";
        this._asstTranscriptByResp.delete(rid); // segment finished; next one starts fresh
        if (segment) {
          const prev = this._asstFullByResp.get(rid) || "";
          this._asstFullByResp.set(rid, prev ? `${prev} ${segment}` : segment);
        }
        const full = this._asstFullByResp.get(rid) || "";
        if (full) {
          this.dispatchEvent(
            new CustomEvent("transcript", {
              detail: { role: "assistant", text: full, partial: false, responseId: rid },
            }),
          );
        }
        break;
      }

      case "error": {
        const err = event.error;
        console.error("[ws] server error:", err);
        if (this._pendingPlaybackConfigs.length > 0
            && /(?:pipeline|tts_tuning|tts_backend|model_provider|max_response)/i.test(
              `${err?.type || ""} ${err?.code || ""}`,
            )) {
          this._pendingPlaybackConfigs.shift();
        }
        if (!this._sessionConfigured) {
          const failure = _codedError(
            err?.message ?? "The speech pipeline rejected its initial configuration",
            err?.code ?? err?.type ?? "pipeline-config-rejected",
          );
          this._rejectInitialConfig(failure);
          await this.close();
          break;
        }
        // The "another response is already active" race: our optimistic create
        // collided with a still-running response. Don't surface it — clear the
        // in-flight guard and re-queue, so the create replays on the next
        // response.done (never retried immediately, which would just collide
        // again).
        if (err?.type === "conversation_already_has_active_response" ||
            err?.code === "conversation_already_has_active_response") {
          this._createInFlight = false;
          this.dispatchEvent(new CustomEvent("server-error", { detail: { error: new Error(err?.message ?? "Response already active") } }));
          break;
        }
        // Every other server error is non-fatal: surface it for logging but
        // NEVER tear the socket down. Only transport failures (close / failed
        // open) are fatal, and those come through their own paths.
        this.dispatchEvent(
          new CustomEvent("server-error", { detail: { error: new Error(err?.message ?? "Server error") } }),
        );
        break;
      }
    }
  }

  /** @param {string} b64 @param {string} responseId @param {number | null} responseEpoch */
  _pushAudioDelta(b64, responseId = "", responseEpoch = null) {
    if (!this._playbackNode) return;
    if (!b64) return;
    if (this._isStaleResponseEpoch(responseEpoch)) return;
    if (responseId && this._stalePlaybackResponses.has(responseId)) {
      this.dispatchEvent(new CustomEvent("pipeline-metric", {
        detail: {
          stage: "playback",
          status: "stale_chunk_rejected",
          source: "browser",
          detail: { reason: "response_retired", current_generation: this._playbackGeneration },
        },
      }));
      return;
    }
    const snapshot = this._playbackSnapshot(responseId, responseEpoch);
    if (snapshot.responseEpoch !== null && responseEpoch !== null && snapshot.responseEpoch !== responseEpoch) return;
    if (snapshot.ended) {
      this.dispatchEvent(new CustomEvent("pipeline-metric", {
        detail: {
          stage: "playback",
          status: "stale_chunk_rejected",
          source: "browser",
          detail: {
            reason: "response_already_ended",
            generation: snapshot.generation,
            current_generation: this._playbackGeneration,
          },
        },
      }));
      return;
    }
    const bytes = base64ToBytes(b64);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const samples = new Float32Array(bytes.byteLength / 2);
    for (let i = 0; i < samples.length; i++) {
      const s = view.getInt16(i * 2, true);
      samples[i] = s < 0 ? s / 0x8000 : s / 0x7fff;
    }
    snapshot.audioStarted = true;
    this._playbackNode.port.postMessage({
      kind: "audio",
      samples,
      generation: snapshot.generation,
      primeMs: snapshot.primeMs,
      reprimeMs: snapshot.reprimeMs,
      maxPrimeMs: snapshot.maxPrimeMs,
      continuityMode: snapshot.continuityMode,
      responseEpoch: snapshot.responseEpoch,
      sourceSampleRate: snapshot.sourceSampleRate,
      streamId: responseId,
    }, [samples.buffer]);
  }

  /** @param {CloseEvent} ev */
  _onWsClose(ev) {
    console.log("[ws] socket closed:", ev.code, ev.reason);
    this._invalidatePlayback("disconnect");
    if (!this._sessionConfigured) {
      this._rejectInitialConfig(
        new Error(`WebSocket closed before pipeline configuration (${ev.code}) ${ev.reason || ""}`.trim()),
      );
    }
    if (this._status === "closed" || this._status === "error") return;
    if (ev.code === 1000) {
      this._setStatus("closed");
    } else {
      this.dispatchEvent(
        new CustomEvent("error", {
          detail: { error: new Error(`WebSocket closed (${ev.code}) ${ev.reason || ""}`.trim()) },
        }),
      );
      this._setStatus("error");
    }
  }

  _sendSessionUpdate() {
    // Minimal payload: only the bits the user is allowed to configure.
    // The s2s server already defaults to server_vad, whisper-1
    // transcription plus pipeline-native 16 kHz PCM input and output, so we don't
    // need (and must not send) `audio.input.format`, `audio.input.transcription`,
    // `audio.input.turn_detection` or `audio.output.format`: the pydantic
    // validator on the server rejects the whole event if any unknown or
    // future-shaped sub-field shows up.
    /** @type {Record<string, any>} */
    const session = {
      type: "realtime",
      instructions: this.options.instructions,
      audio: {
        output: { voice: this.options.voice },
      },
    };
    // Tools are declared here; the backend already accepts them in
    // session.update and emits response.function_call_arguments.done when the
    // model decides to call one. Only include the keys when we actually have
    // tools — the server's pydantic validator is strict about shapes.
    if (this._tools.length) {
      session.tools = this._tools;
      session.tool_choice = "auto";
    }
    this._send({ type: "session.update", session });
  }

  /** Update voice/instructions on a live session without tearing down. */
  /** @param {{ voice?: string; instructions?: string }} patch */
  updateSession(patch) {
    /** @type {Record<string, any>} */
    const session = { type: "realtime" };
    if (patch.instructions) session.instructions = patch.instructions;
    if (patch.voice) session.audio = { output: { voice: patch.voice } };
    if (Object.keys(session).length > 1) {
      this._send({ type: "session.update", session });
    }
  }

  /**
   * @param {Record<string, any>} config
   * @param {PlaybackConfig | null} [playbackConfig]
   */
  updateLocalPipeline(config, playbackConfig = null) {
    const requested = {
      ...config,
      playback_policy: this._requestedPlaybackPolicy(config, playbackConfig),
    };
    this._queuePlaybackConfig(requested, playbackConfig);
    this._send({ type: "pipeline.config.update", config: requested });
  }

  /**
   * Replace the declared tool set on a live session (e.g. the user flipped a
   * tool switch mid-conversation). Always sends `tools` — an empty array
   * clears them — so toggling the last tool off actually removes it.
   * @param {ToolDef[]} tools
   */
  setTools(tools) {
    this._tools = tools;
    this._send({
      type: "session.update",
      session: { type: "realtime", tools, tool_choice: tools.length ? "auto" : "none" },
    });
  }

  /**
   * Return a tool's result to the model. Pairs with the `toolcall` event's
   * `callId`. Caller follows this with `requestResponse()` so the model speaks.
   * @param {string} callId
   * @param {string} output Plain text / JSON string the model will read.
   */
  sendToolOutput(callId, output) {
    if (!callId) return Promise.reject(new Error("Missing function call id"));
    if (this._toolOutputAcks.has(callId)) return Promise.reject(new Error(`Tool output already pending (${callId})`));
    if (!this._ws || this._ws.readyState !== WebSocket.OPEN) return Promise.reject(new Error("WebSocket is not connected"));
    const ack = new Promise((resolve, reject) => {
      const timer = window.setTimeout(() => {
        this._toolOutputAcks.delete(callId);
        reject(new Error(`Timed out waiting for tool output acknowledgement (${callId})`));
      }, 15000);
      this._toolOutputAcks.set(callId, { resolve, reject, timer });
    });
    this._send({
      type: "conversation.item.create",
      item: { type: "function_call_output", call_id: callId, output },
    });
    return ack;
  }

  /** Wait until the response that emitted a tool call has fully closed. */
  waitForResponseIdle(timeoutMs = 20000) {
    if (!this._responseActive()) return Promise.resolve();
    return new Promise((resolve, reject) => {
      const waiter = {
        resolve,
        reject,
        timer: window.setTimeout(() => {
          this._responseIdleWaiters.delete(waiter);
          reject(new Error("Timed out waiting for the originating response to close"));
        }, timeoutMs),
      };
      this._responseIdleWaiters.add(waiter);
    });
  }

  _resolveResponseIdleWaiters() {
    for (const waiter of this._responseIdleWaiters) {
      clearTimeout(waiter.timer);
      waiter.resolve();
    }
    this._responseIdleWaiters.clear();
  }

  /**
   * Add an image to the conversation as user content, so the vision-language
   * model can see it (used by the camera tool). `dataUrl` is a
   * `data:image/jpeg;base64,...` string.
   * @param {string} dataUrl
   */
  sendUserImage(dataUrl) {
    this._send({
      type: "conversation.item.create",
      item: {
        type: "message",
        role: "user",
        content: [{ type: "input_image", image_url: dataUrl }],
      },
    });
  }

  /**
   * Ask the model to generate a response now (after feeding tool results).
   * Serialized: if a response is already in flight we queue this request and
   * replay it once the active response finishes, so we never trip the
   * backend's `conversation_already_has_active_response` guard.
   *
   * @param {{ image?: string }} [opts] Optional `image` (a data URL) sent as a
   *   user `input_image` immediately before this response.create — so the frame
   *   travels with the create (and is deferred together with it if queued),
   *   rather than being added to the conversation eagerly. Used by the camera
   *   tool so the model sees the snapshot in the response it's about to speak.
   */
  requestResponse(opts = {}) {
    if (this._responseActive()) {
      this._createQueue.push(opts);
      if (this._debug) console.debug(`[ws] response.create queued (a response is active); pending=${this._createQueue.length}`);
      return;
    }
    this._createResponseNow(opts);
  }

  /** Send a tool follow-up immediately. The backend binds it to the active
   *  call ID transaction and starts it after the originating response closes.
   *  @param {{ image?: string }} [opts] */
  requestToolResponse(opts = {}) {
    if (!this._ws || this._ws.readyState !== WebSocket.OPEN) return;
    if (opts.image) this.sendUserImage(opts.image);
    this._createInFlight = true;
    this._send({ type: "response.create" });
  }

  /** True while a response occupies the single backend slot. */
  _responseActive() {
    return this._openResponses > 0 || this._createInFlight;
  }

  /** Send a response.create immediately and arm the in-flight guard. Any image
   *  on the payload is added as user content right before the create.
   *  @param {{ image?: string }} [opts] */
  _createResponseNow(opts = {}) {
    if (!this._ws || this._ws.readyState !== WebSocket.OPEN) return;
    if (opts.image) this.sendUserImage(opts.image);
    this._createInFlight = true;
    this._send({ type: "response.create" });
  }

  /** Replay one queued response.create if the slot is now free. Called on every
   *  response.done, so queued creates drain one-per-completion. */
  _flushQueuedCreate() {
    if (this._createQueue.length > 0 && !this._responseActive()) {
      const opts = this._createQueue.shift();
      if (this._debug) console.debug(`[ws] replaying queued response.create; remaining=${this._createQueue.length}`);
      this._createResponseNow(opts);
    }
  }

  /** @param {boolean} muted */
  setMuted(muted) {
    this._muted = muted;
  }

  /**
   * Update the mic noise gate live (the user moved the Settings cursor).
   * @param {NoiseGate} gate
   */
  setNoiseGate(gate) {
    this._noiseGate = gate;
    this._captureNode?.port.postMessage({ kind: "gate", ...gate });
  }

  /** @param {EchoGuardMode} mode */
  setEchoGuard(mode) {
    this._echoGuard = ["native", "adaptive", "strict"].includes(mode) ? mode : "native";
    const micTrack = this.options.micStream?.getAudioTracks?.()[0];
    const nativeAec = !!micTrack?.getSettings?.().echoCancellation;
    this._captureNode?.port.postMessage({ kind: "echo_guard", mode: this._echoGuard, nativeAec });
  }

  /** Persisted by the UI under the current microphone/output-device pair. */
  setEchoCalibration(calibration) {
    this._echoCalibration = normalizeEchoCalibration(calibration);
    if (this._echoDevicePair) {
      this._echoCalibrations[this._echoDevicePair] = this._echoCalibration;
    }
    const outputLatencyMs = Number.isFinite(this._ctx?.outputLatency)
      ? Math.max(0, this._ctx.outputLatency * 1000)
      : 0;
    this._captureNode?.port.postMessage({
      kind: "echo_calibration",
      ...this._echoCalibration,
      outputLatencyMs,
    });
  }

  /** @param {Record<string, unknown>} event */
  _send(event) {
    if (!this._ws || this._ws.readyState !== WebSocket.OPEN) return;
    this._ws.send(JSON.stringify(event));
  }

  async close() {
    // Abort a queue wait in progress: flag it and wake the poll sleep so
    // `_pollQueue` throws "aborted" and connect() unwinds cleanly.
    if (!this._closed) this._invalidatePlayback("stop");
    this._closed = true;
    this._sessionConfigured = false;
    this._rejectInitialConfig(new Error("Connection closed before pipeline configuration completed"));
    this._muted = true;
    this._captureNode?.port.postMessage({ kind: "echo_reset" });
    this._captureNode?.port.postMessage({ kind: "enable", value: false });
    for (const track of this.options.micStream?.getTracks?.() ?? []) {
      track.stop();
    }
    this.options.micStream = undefined;
    for (const pending of this._toolOutputAcks.values()) {
      clearTimeout(pending.timer);
      pending.reject(new Error("WebSocket closed before tool output was acknowledged"));
    }
    this._toolOutputAcks.clear();
    for (const waiter of this._responseIdleWaiters) {
      clearTimeout(waiter.timer);
      waiter.reject(new Error("WebSocket closed before the response finished"));
    }
    this._responseIdleWaiters.clear();
    if (this._queueWake) {
      clearTimeout(this._queueTimer);
      const wake = this._queueWake;
      this._queueWake = null;
      wake();
    }
    if (this._joinTimer) {
      clearTimeout(this._joinTimer);
      this._joinTimer = 0;
    }
    if (this._joinReject) {
      const reject = this._joinReject;
      this._joinResolve = null;
      this._joinReject = null;
      reject(_codedError("join aborted", "aborted"));
    }
    this._visualiser?.stop();
    this._visualiser = null;
    try {
      if (this._ws && this._ws.readyState <= WebSocket.OPEN) {
        this._ws.close(1000, "client closed");
      }
    } catch {
      // ignored
    }
    this._ws = null;

    try {
      this._captureNode?.port.close?.();
    } catch {
      // ignored
    }
    try {
      this._micSrc?.disconnect();
    } catch {
      // ignored
    }
    try {
      this._captureNode?.disconnect();
    } catch {
      // ignored
    }
    try {
      this._micAnalyser?.disconnect();
    } catch {
      // ignored
    }
    try {
      this._outAnalyser?.disconnect();
    } catch {
      // ignored
    }
    try {
      this._playbackNode?.disconnect();
    } catch {
      // ignored
    }
    try {
      await this._ctx?.close();
    } catch {
      // ignored
    }
    this._ctx = null;
    this._captureNode = null;
    this._playbackNode = null;
    this._micSrc = null;
    this._micAnalyser = null;
    this._outAnalyser = null;
    this._pendingPlaybackConfigs.length = 0;
    this._playbackByResponse.clear();
    this._playbackPolicyByResponseEpoch.clear();
    this._stalePlaybackResponses.clear();
    this._terminalResponseIds.clear();
    this._heardResponses.clear();
    this._toolCallTombstones.clear();
    this._toolCommittedResponses.clear();
    this._terminalResponseEpochs.clear();
    this._cancelledResponseEpochs.clear();
    this._setStatus("closed");
  }
}
