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
 *   - The server pushes `response.output_audio.delta` (pipeline-native PCM16
 *     16 kHz mono base64) and transcript deltas.
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
 * @property {string | number} [profileRevision]
 * @property {string} [model]
 * @property {string} [clone]
 * @property {boolean} [nativeStreaming]
 * @property {number} [resolvedPrimeMs]
 * @property {number} [firstBlockFrames]
 * @property {number} [steadyBlockFrames]
 * @property {number} [outputRate]
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

// The s2s pipeline runs internally at 16 kHz mono PCM. The WebRTC transport
// resamples to 48 kHz for Opus, but the WebSocket transport emits the
// native pipeline rate. We don't (can't) override it via `audio.output.format`
// because the server's pydantic validator rejects the whole `session.update`
// as soon as a sub-field shape it doesn't know about appears.
const OUTPUT_SAMPLE_RATE = 16000;
const MIC_CHUNK_MS = 40;
export const PIPELINE_CONFIG_ACK_TIMEOUT_MS = 15_000;
export const MAX_PLAYBACK_PRIME_MS = 2_000;
const MAX_PLAYBACK_RESPONSE_TOMBSTONES = 512;
export const PLAYBACK_LEARNING_VERSION = 1;
export const PLAYBACK_LEARNING_TTL_MS = 7 * 24 * 60 * 60 * 1_000;
export const MAX_PLAYBACK_LEARNING_SIGNATURES = 16;
export const PLAYBACK_GAP_WINDOW = 8;
const PLAYBACK_LEARNING_STORAGE_KEY = "s2s.playback.safe-start.v1";
const AUDIO_CPP_CODEC_FRAME_MS = 80;
const PLAYBACK_WARM_FIRST_MARGIN_MS = 160;
const PLAYBACK_WARM_GAP_MARGIN_MS = 64;
export const AUDIO_CPP_PLAYBACK_PRIME_MS = Object.freeze({
  "low-latency": 800,
  balanced: 1280,
  quality: 1760,
});

function _normalisePlaybackProvider(value) {
  const provider = String(value || "").trim().toLowerCase();
  return provider === "audio-cpp" ? "qwen3tts-audiocpp" : provider;
}

/** @param {unknown} value @returns {value is Record<string, unknown>} */
function _isPlainObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

/**
 * Validate a value against the small JSON-Schema subset used by browser tools.
 * Return only a field path and error class: caller diagnostics must never echo
 * raw argument values.
 * @param {unknown} value
 * @param {Record<string, any>} schema
 * @param {string} path
 * @returns {{ path: string, errorClass: string } | null}
 */
function _toolSchemaError(value, schema, path = "$") {
  if (Array.isArray(schema.enum) && !schema.enum.some((candidate) => Object.is(candidate, value))) {
    return { path, errorClass: "not_in_enum" };
  }
  const type = typeof schema.type === "string" ? schema.type : "";
  if (type === "object") {
    if (!_isPlainObject(value)) return { path, errorClass: "expected_object" };
    const properties = _isPlainObject(schema.properties) ? schema.properties : {};
    const required = Array.isArray(schema.required) ? schema.required : [];
    for (const field of required) {
      if (typeof field === "string" && !Object.prototype.hasOwnProperty.call(value, field)) {
        return { path: `${path}.${field}`, errorClass: "required" };
      }
    }
    if (schema.additionalProperties === false) {
      for (const field of Object.keys(value)) {
        if (!Object.prototype.hasOwnProperty.call(properties, field)) {
          // The field name is untrusted model output. Keep diagnostics on the
          // declared-schema path instead of copying that name into cards,
          // metrics, tool output, or console traces.
          return { path: `${path}.*`, errorClass: "unexpected_property" };
        }
      }
    }
    for (const [field, childSchema] of Object.entries(properties)) {
      if (!Object.prototype.hasOwnProperty.call(value, field) || !_isPlainObject(childSchema)) continue;
      const childError = _toolSchemaError(value[field], childSchema, `${path}.${field}`);
      if (childError) return childError;
    }
    return null;
  }
  if (type === "array") {
    if (!Array.isArray(value)) return { path, errorClass: "expected_array" };
    if (_isPlainObject(schema.items)) {
      for (let index = 0; index < value.length; index += 1) {
        const childError = _toolSchemaError(value[index], schema.items, `${path}[${index}]`);
        if (childError) return childError;
      }
    }
    return null;
  }
  if (type === "string") {
    if (typeof value !== "string") return { path, errorClass: "expected_string" };
    if (Number.isFinite(schema.minLength) && value.length < Number(schema.minLength)) {
      return { path, errorClass: "min_length" };
    }
    if (Number.isFinite(schema.maxLength) && value.length > Number(schema.maxLength)) {
      return { path, errorClass: "max_length" };
    }
    if (typeof schema.pattern === "string" && !(new RegExp(schema.pattern)).test(value)) {
      return { path, errorClass: "pattern_mismatch" };
    }
    return null;
  }
  if (type === "number" && (typeof value !== "number" || !Number.isFinite(value))) {
    return { path, errorClass: "expected_number" };
  }
  if (type === "integer" && !Number.isInteger(value)) {
    return { path, errorClass: "expected_integer" };
  }
  if (type === "boolean" && typeof value !== "boolean") {
    return { path, errorClass: "expected_boolean" };
  }
  return null;
}

/**
 * Parse and validate tool arguments without coercion. The raw JSON string stays
 * on the public call event for exact server/history fidelity; only the local
 * browser executor consumes the parsed object returned here.
 * @param {unknown} argsJson
 * @param {Record<string, any>} schema
 * @returns {{ ok: true, args: Record<string, unknown> } | { ok: false, code: string, path: string, errorClass: string }}
 */
export function validateToolArguments(argsJson, schema) {
  if (typeof argsJson !== "string") {
    return { ok: false, code: "malformed_json", path: "$", errorClass: "expected_json_string" };
  }
  let parsed;
  try {
    parsed = JSON.parse(argsJson);
  } catch {
    return { ok: false, code: "malformed_json", path: "$", errorClass: "json_parse_error" };
  }
  const error = _toolSchemaError(parsed, schema, "$");
  if (error) return { ok: false, code: "schema_validation_failed", ...error };
  if (!_isPlainObject(parsed)) {
    return { ok: false, code: "schema_validation_failed", path: "$", errorClass: "expected_object" };
  }
  return { ok: true, args: parsed };
}

/**
 * Build the function output for invalid arguments. It identifies the tool,
 * field path, and validation class without reproducing user/model-provided
 * argument values.
 * @param {string} tool
 * @param {{ ok: false, code: string, path: string, errorClass: string }} failure
 */
export function invalidToolArgumentsOutput(tool, failure) {
  return JSON.stringify({
    type: "invalid_tool_arguments",
    tool,
    code: failure.code,
    path: failure.path,
    error_class: failure.errorClass,
    message: "The tool arguments were invalid. Submit a new call that matches the declared schema.",
  });
}

/**
 * Prepare the only tool-argument representation the browser may persist or
 * display. Valid calls expose the schema-validated object; malformed,
 * schema-invalid, and unknown calls expose content-free failure metadata.
 * The original argument string remains available only to the protocol event
 * and validator and must never be handed to a durable UI surface.
 * @param {unknown} tool
 * @param {unknown} argsJson
 * @param {Record<string, any> | null | undefined} schema
 * @returns {{
 *   tool: string,
 *   displayArguments: string,
 *   validation: { ok: true, args: Record<string, unknown> } |
 *     { ok: false, code: string, path: string, errorClass: string }
 * }}
 */
export function prepareToolArgumentsForBrowser(tool, argsJson, schema) {
  const hasSchema = _isPlainObject(schema);
  const safeTool = hasSchema && typeof tool === "string" && tool ? tool : "unknown_tool";
  const validation = hasSchema
    ? validateToolArguments(argsJson, schema)
    : { ok: false, code: "unknown_tool", path: "$", errorClass: "unknown_tool" };
  return {
    tool: safeTool,
    validation,
    displayArguments: validation.ok
      ? JSON.stringify(validation.args)
      : invalidToolArgumentsOutput(safeTool, validation),
  };
}

function _normaliseProfileId(value) {
  return String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[_\s]+/g, "-");
}

/**
 * Resolve the browser queue target without changing the WebSocket schema.
 * Only a positively validated native audio.cpp stream is primed. Built-ins use
 * fixed acceptance targets; a custom profile uses its locally resolved first
 * block duration and is bounded to two seconds.
 *
 * @param {Record<string, any>} config Acknowledged pipeline config.
 * @param {PlaybackConfig} [hint] Browser-only validated profile snapshot.
 */
export function resolvePlaybackPrimeMs(config = {}, hint = {}) {
  const provider = _normalisePlaybackProvider(config.tts_backend || hint.provider);
  if (provider !== "qwen3tts-audiocpp" || hint.nativeStreaming !== true) return 0;

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

function _positiveInteger(value) {
  const number = Number(value);
  return Number.isSafeInteger(number) && number > 0 ? number : 0;
}

function _identityString(value) {
  return typeof value === "string" ? value.trim() : String(value ?? "").trim();
}

function _runtimePlaybackIdentity(runtime) {
  if (!_isPlainObject(runtime)) return "";
  const apiVersion = _identityString(runtime.api_version);
  const startedAt = _identityString(runtime.started_at_utc);
  const pid = _positiveInteger(runtime.pid);
  if (!apiVersion || !startedAt || !pid) return "";
  const build = _identityString(
    runtime.git_commit || runtime.commit || runtime.build_id || runtime.build || runtime.image,
  );
  return JSON.stringify([apiVersion, startedAt, pid, build]);
}

/**
 * Build the exact immutable identity used for adaptive safe-start learning.
 * Fields absent from the server acknowledgement remain conservative: the
 * acknowledgement is the commit barrier, while model/clone/runtime/rate come
 * from the already validated browser snapshot.
 *
 * @param {Record<string, any>} config
 * @param {PlaybackConfig} hint
 * @param {string} runtimeIdentity
 * @param {boolean} ackMatched
 */
export function resolveAdaptivePlaybackSignature(
  config = {},
  hint = {},
  runtimeIdentity = "",
  ackMatched = true,
) {
  const provider = _normalisePlaybackProvider(config.tts_backend || hint.provider);
  const nativeStreaming = hint.nativeStreaming === true;
  const profileId = _normaliseProfileId(config.tts_tuning?.profile_id || hint.profileId);
  const hintedProfileId = _normaliseProfileId(hint.profileId);
  const profileMatches = !profileId || !hintedProfileId || profileId === hintedProfileId;
  const firstBlockFrames = _positiveInteger(
    hint.firstBlockFrames ?? config.tts_tuning?.overrides?.first_block_frames,
  );
  const steadyBlockFrames = _positiveInteger(
    hint.steadyBlockFrames ?? config.tts_tuning?.overrides?.steady_block_frames,
  );
  const acknowledgedFirst = _positiveInteger(config.tts_tuning?.overrides?.first_block_frames);
  const acknowledgedSteady = _positiveInteger(config.tts_tuning?.overrides?.steady_block_frames);
  const overrideMatches = (
    (!acknowledgedFirst || acknowledgedFirst === firstBlockFrames)
    && (!acknowledgedSteady || acknowledgedSteady === steadyBlockFrames)
  );
  const outputRate = _positiveInteger(hint.outputRate || OUTPUT_SAMPLE_RATE);
  const model = _identityString(hint.model);
  const clone = _identityString(hint.clone);
  const profileRevision = _identityString(hint.profileRevision);
  const firstMs = firstBlockFrames * AUDIO_CPP_CODEC_FRAME_MS;
  const eligible = provider === "qwen3tts-audiocpp" && nativeStreaming;
  const hasCompleteFramePair = firstBlockFrames > 0 && steadyBlockFrames > 0;
  const framedCeilingMs = hasCompleteFramePair
    ? (firstBlockFrames + steadyBlockFrames) * AUDIO_CPP_CODEC_FRAME_MS
    : 0;
  const valid = eligible
    && ackMatched
    && profileMatches
    && overrideMatches
    && !!model
    && !!clone
    && !!profileId
    && !!profileRevision
    && !!runtimeIdentity
    && hasCompleteFramePair
    && outputRate > 0;
  const namedConservativeCeiling = AUDIO_CPP_PLAYBACK_PRIME_MS[profileId];
  const conservativeCeilingMs = Number.isFinite(namedConservativeCeiling)
    ? namedConservativeCeiling
    : MAX_PLAYBACK_PRIME_MS;
  const ceilingMs = eligible
    ? Math.max(0, Math.min(
      MAX_PLAYBACK_PRIME_MS,
      valid ? framedCeilingMs : conservativeCeilingMs,
    ))
    : 0;
  const fields = Object.freeze({
    provider,
    model,
    clone,
    profileId,
    profileRevision,
    firstBlockFrames,
    steadyBlockFrames,
    nativeStreaming,
    outputRate,
    runtimeIdentity,
  });
  return Object.freeze({
    valid,
    eligible,
    key: valid ? JSON.stringify(Object.values(fields)) : "",
    fields,
    firstMs: Math.min(ceilingMs, firstMs),
    steadyMs: steadyBlockFrames * AUDIO_CPP_CODEC_FRAME_MS,
    ceilingMs,
  });
}

function _nearestRankP95(values) {
  if (!values.length) return 0;
  const ordered = [...values].sort((left, right) => left - right);
  return ordered[Math.max(0, Math.ceil(ordered.length * 0.95) - 1)];
}

function _safeLocalStorage() {
  try {
    return globalThis.localStorage || null;
  } catch {
    return null;
  }
}

/** Versioned, bounded, content-free adaptive safe-start learning. */
export class AdaptivePlaybackPolicyStore {
  /** @param {{ storage?: Storage | null, now?: () => number }} [options] */
  constructor(options = {}) {
    this._storage = options.storage === undefined ? _safeLocalStorage() : options.storage;
    this._now = typeof options.now === "function" ? options.now : () => Date.now();
    /** @type {Map<string, Record<string, any>>} */
    this._records = new Map();
    this._load();
  }

  _load() {
    let payload;
    try {
      payload = JSON.parse(this._storage?.getItem(PLAYBACK_LEARNING_STORAGE_KEY) || "null");
    } catch {
      return;
    }
    if (!_isPlainObject(payload)
        || payload.version !== PLAYBACK_LEARNING_VERSION
        || !Array.isArray(payload.entries)) return;
    const now = this._now();
    for (const entry of payload.entries.slice(0, MAX_PLAYBACK_LEARNING_SIGNATURES)) {
      if (!_isPlainObject(entry)
          || typeof entry.key !== "string"
          || !entry.key
          || entry.key.length > 4096
          || !Number.isFinite(entry.expiresAt)
          || entry.expiresAt <= now) continue;
      const gaps = Array.isArray(entry.gaps)
        ? entry.gaps
          .filter((gap) => Number.isFinite(gap) && gap > 0)
          .slice(-PLAYBACK_GAP_WINDOW)
        : [];
      const fullPrimeCleanCount = Math.max(
        0,
        Math.min(2, Number(entry.fullPrimeCleanCount) || 0),
      );
      const disabled = entry.disabled === true;
      this._records.set(entry.key, {
        key: entry.key,
        gaps,
        fullPrimeCleanCount,
        // Treat storage as untrusted input. Warm eligibility is derived from
        // the two clean-response evidence counter, never from a persisted flag.
        warmEligible: !disabled && fullPrimeCleanCount >= 2,
        disabled,
        recoveryCleanCount: Math.max(0, Math.min(3, Number(entry.recoveryCleanCount) || 0)),
        lastUsed: Number.isFinite(entry.lastUsed) ? entry.lastUsed : now,
        expiresAt: entry.expiresAt,
      });
    }
    this._prune(now);
  }

  _prune(now = this._now()) {
    for (const [key, record] of this._records) {
      if (!Number.isFinite(record.expiresAt) || record.expiresAt <= now) this._records.delete(key);
    }
    const ordered = [...this._records.values()].sort((left, right) => right.lastUsed - left.lastUsed);
    for (const record of ordered.slice(MAX_PLAYBACK_LEARNING_SIGNATURES)) {
      this._records.delete(record.key);
    }
  }

  _persist() {
    this._prune();
    try {
      this._storage?.setItem(PLAYBACK_LEARNING_STORAGE_KEY, JSON.stringify({
        version: PLAYBACK_LEARNING_VERSION,
        entries: [...this._records.values()].sort((left, right) => right.lastUsed - left.lastUsed),
      }));
    } catch {
      // Storage is an optimization only; private/blocked modes stay conservative.
    }
  }

  _record(signatureKey) {
    const now = this._now();
    let record = this._records.get(signatureKey);
    if (!record) {
      record = {
        key: signatureKey,
        gaps: [],
        fullPrimeCleanCount: 0,
        warmEligible: false,
        disabled: false,
        recoveryCleanCount: 0,
        lastUsed: now,
        expiresAt: now + PLAYBACK_LEARNING_TTL_MS,
      };
      this._records.set(signatureKey, record);
    }
    record.lastUsed = now;
    record.expiresAt = now + PLAYBACK_LEARNING_TTL_MS;
    return record;
  }

  /** @param {ReturnType<typeof resolveAdaptivePlaybackSignature>} signature */
  policy(signature) {
    if (!signature.eligible) {
      return Object.freeze({
        learning: false,
        signatureKey: "",
        mode: "immediate",
        targetMs: 0,
        ceilingMs: 0,
        firstMs: 0,
        gapP95Ms: 0,
        firstBlockSamples: 0,
        steadyBlockSamples: 0,
        jitterMarginMs: PLAYBACK_WARM_GAP_MARGIN_MS,
        fallbackReason: "non_native_or_buffered",
      });
    }
    if (!signature.valid) {
      return Object.freeze({
        learning: false,
        signatureKey: "",
        mode: "conservative",
        targetMs: signature.ceilingMs,
        ceilingMs: signature.ceilingMs,
        firstMs: signature.firstMs,
        gapP95Ms: 0,
        firstBlockSamples: 0,
        steadyBlockSamples: 0,
        jitterMarginMs: PLAYBACK_WARM_GAP_MARGIN_MS,
        fallbackReason: "signature_incomplete_or_unordered",
      });
    }
    const record = this._record(signature.key);
    const warm = record.warmEligible && !record.disabled;
    const gapP95Ms = _nearestRankP95(record.gaps);
    const warmTarget = Math.max(
      signature.firstMs + PLAYBACK_WARM_FIRST_MARGIN_MS,
      gapP95Ms + PLAYBACK_WARM_GAP_MARGIN_MS,
    );
    const policy = Object.freeze({
      learning: true,
      signatureKey: signature.key,
      mode: warm ? "warm" : (record.disabled ? "recovery" : "cold"),
      targetMs: warm ? Math.min(signature.ceilingMs, warmTarget) : signature.ceilingMs,
      ceilingMs: signature.ceilingMs,
      firstMs: signature.firstMs,
      gapP95Ms,
      firstBlockSamples: Math.ceil(
        (signature.fields.firstBlockFrames * AUDIO_CPP_CODEC_FRAME_MS
          * signature.fields.outputRate) / 1_000,
      ),
      steadyBlockSamples: Math.ceil(
        (signature.fields.steadyBlockFrames * AUDIO_CPP_CODEC_FRAME_MS
          * signature.fields.outputRate) / 1_000,
      ),
      jitterMarginMs: PLAYBACK_WARM_GAP_MARGIN_MS,
      fallbackReason: warm
        ? "learned_block_cadence"
        : (record.disabled ? "underrun_recovery" : "cold_evidence"),
    });
    this._persist();
    return policy;
  }

  /**
   * @param {ReturnType<AdaptivePlaybackPolicyStore["policy"]>} policy
   * @param {{ gaps?: number[], underrun?: boolean, cleanFullPrime?: boolean }} observation
   */
  record(policy, observation = {}) {
    if (!policy.learning || !policy.signatureKey) return;
    const record = this._record(policy.signatureKey);
    const gaps = Array.isArray(observation.gaps)
      ? observation.gaps.filter((gap) => Number.isFinite(gap) && gap > 0)
      : [];
    record.gaps = [...record.gaps, ...gaps].slice(-PLAYBACK_GAP_WINDOW);
    if (observation.underrun) {
      record.disabled = true;
      record.warmEligible = false;
      record.fullPrimeCleanCount = 0;
      record.recoveryCleanCount = 0;
      this._persist();
      return;
    }
    const cleanFullPrime = observation.cleanFullPrime === true;
    if (record.disabled) {
      record.recoveryCleanCount = cleanFullPrime ? record.recoveryCleanCount + 1 : 0;
      if (record.recoveryCleanCount >= 3) {
        record.disabled = false;
        record.warmEligible = true;
        record.fullPrimeCleanCount = 2;
        record.recoveryCleanCount = 0;
      }
    } else if (!record.warmEligible) {
      record.fullPrimeCleanCount = cleanFullPrime ? record.fullPrimeCleanCount + 1 : 0;
      if (record.fullPrimeCleanCount >= 2) record.warmEligible = true;
    }
    this._persist();
  }
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
    this._playbackPrimeMs = 0;
    this._playbackCeilingMs = 0;
    this._playbackRuntimeIdentity = "";
    this._playbackLearning = new AdaptivePlaybackPolicyStore();
    this._playbackClock = () => performance.now();
    this._playbackTurnSerial = 0;
    this._activePlaybackTurn = null;
    this._anonymousPlaybackResponseId = "";
    this._anonymousPlaybackSerial = 0;
    this._acknowledgedPlaybackSignature = resolveAdaptivePlaybackSignature({}, {}, "", false);
    this._lastAcknowledgedPipelineConfig = {};
    /** @type {PlaybackConfig} */
    this._acknowledgedPlaybackConfig = {
      provider: "",
      profileId: "",
      profileRevision: "",
      model: "",
      clone: "",
      nativeStreaming: false,
      resolvedPrimeMs: 0,
      firstBlockFrames: 0,
      steadyBlockFrames: 0,
      outputRate: OUTPUT_SAMPLE_RATE,
    };
    /** @type {{ expectedProvider: string, expectedProfile: string, hint: PlaybackConfig }[]} */
    this._pendingPlaybackConfigs = [];
    /** @type {Map<string, Record<string, any>>} */
    this._playbackByResponse = new Map();
    /** @type {Map<string, Record<string, any>>} Completed network responses awaiting worklet drain. */
    this._completedPlaybackResponses = new Map();
    /** @type {Set<string>} Response IDs observed through response.created and not yet terminal. */
    this._openPlaybackResponseIds = new Set();
    /** @type {Set<string>} Bounded completed/cancelled response IDs. */
    this._stalePlaybackResponses = new Set();
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
    };
    this._pendingPlaybackConfigs.push({
      expectedProvider: requestedProvider,
      expectedProfile: _normaliseProfileId(effectiveHint.profileId),
      hint: effectiveHint,
    });
    // A page should have at most one update awaiting acknowledgement. Keep a
    // hard bound anyway so a broken peer cannot grow browser memory forever.
    if (this._pendingPlaybackConfigs.length > 16) this._pendingPlaybackConfigs.shift();
  }

  /** Apply browser playback policy only after the backend committed the config. */
  /** @param {Record<string, any>} config */
  _applyAcknowledgedPlaybackConfig(config) {
    const provider = _normalisePlaybackProvider(config.tts_backend);
    const profileId = _normaliseProfileId(config.tts_tuning?.profile_id);
    const pendingIndex = this._pendingPlaybackConfigs.findIndex((pending) => (
      pending.expectedProvider === provider
      && (!pending.expectedProfile || pending.expectedProfile === profileId)
    ));
    const orderedAck = pendingIndex === 0;
    const pending = pendingIndex >= 0
      ? this._pendingPlaybackConfigs.splice(0, pendingIndex + 1).at(-1)
      : this._pendingPlaybackConfigs.shift() || null;
    const canReuseAcknowledged = provider === this._acknowledgedPlaybackConfig.provider;
    const hint = {
      ...(canReuseAcknowledged ? this._acknowledgedPlaybackConfig : {}),
      ...(pending?.hint || {}),
      provider,
      profileId: profileId || pending?.hint?.profileId || "",
    };
    const previousKey = this._acknowledgedPlaybackSignature.key;
    const signature = resolveAdaptivePlaybackSignature(
      config,
      hint,
      this._playbackRuntimeIdentity,
      orderedAck,
    );
    const policy = this._playbackLearning.policy(signature);
    this._acknowledgedPlaybackConfig = hint;
    this._acknowledgedPlaybackSignature = signature;
    this._lastAcknowledgedPipelineConfig = { ...config };
    this._playbackPrimeMs = policy.targetMs;
    this._playbackCeilingMs = policy.ceilingMs;
    this.dispatchEvent(new CustomEvent("pipeline-metric", {
      detail: {
        stage: "playback",
        status: "configured",
        source: "browser",
        detail: {
          provider,
          profile_id: hint.profileId || null,
          native_streaming: hint.nativeStreaming === true,
          prime_target_ms: policy.targetMs,
          prime_ceiling_ms: policy.ceilingMs,
          cold_ceiling_ms: policy.ceilingMs,
          effective_target_ms: policy.targetMs,
          safe_start_mode: policy.mode,
          latest_logical_block_gap_ms: null,
          logical_block_gap_p95_ms: policy.gapP95Ms,
          jitter_margin_ms: policy.jitterMarginMs,
          fallback_reason: policy.fallbackReason,
          signature_valid: signature.valid,
          signature_reset: !!previousKey && previousKey !== signature.key,
          acknowledged: true,
        },
      },
    }));
  }

  _freezePlaybackTurnPolicy() {
    this._playbackTurnSerial += 1;
    const policy = this._playbackLearning.policy(this._acknowledgedPlaybackSignature);
    this._activePlaybackTurn = Object.freeze({
      id: this._playbackTurnSerial,
      generation: this._playbackGeneration,
      policy,
    });
    this._anonymousPlaybackResponseId = "";
    return this._activePlaybackTurn;
  }

  _conservativePlaybackPolicy() {
    const signature = this._acknowledgedPlaybackSignature;
    if (!signature.eligible) {
      return Object.freeze({
        learning: false,
        signatureKey: "",
        mode: "immediate",
        targetMs: 0,
        ceilingMs: 0,
        firstMs: 0,
        gapP95Ms: 0,
        firstBlockSamples: 0,
        steadyBlockSamples: 0,
        jitterMarginMs: PLAYBACK_WARM_GAP_MARGIN_MS,
        fallbackReason: "non_native_or_buffered",
      });
    }
    return Object.freeze({
      learning: false,
      signatureKey: "",
      mode: "conservative",
      targetMs: signature.ceilingMs,
      ceilingMs: signature.ceilingMs,
      firstMs: signature.firstMs,
      gapP95Ms: 0,
      firstBlockSamples: 0,
      steadyBlockSamples: 0,
      jitterMarginMs: PLAYBACK_WARM_GAP_MARGIN_MS,
      fallbackReason: "response_identity_unordered",
    });
  }

  /** @param {string} responseId @param {boolean} [ordered] */
  _playbackSnapshot(responseId, ordered = true) {
    if (responseId) {
      const existing = this._playbackByResponse.get(responseId)
        || this._completedPlaybackResponses.get(responseId);
      if (existing) return existing;
    }
    const turn = this._activePlaybackTurn;
    const identityOrdered = ordered
      && !!turn
      && turn.generation === this._playbackGeneration;
    const policy = identityOrdered ? turn.policy : this._conservativePlaybackPolicy();
    const targetSamples = Math.ceil((policy.targetMs * OUTPUT_SAMPLE_RATE) / 1_000);
    const ceilingSamples = Math.ceil((policy.ceilingMs * OUTPUT_SAMPLE_RATE) / 1_000);
    const snapshot = {
      generation: this._playbackGeneration,
      turnId: turn?.id || 0,
      policy,
      primeMs: policy.targetMs,
      ceilingMs: policy.ceilingMs,
      targetSamples,
      ceilingSamples,
      ended: false,
      inputSamples: 0,
      chunkCount: 0,
      gaps: [],
      logicalBlockCount: 0,
      nextLogicalBoundarySamples: policy.firstBlockSamples,
      lastLogicalBlockAt: null,
      latestLogicalBlockGapMs: null,
      ordered: identityOrdered,
      networkDone: false,
      responseStatus: "",
      workletDrained: false,
      forcedShort: false,
      underrun: false,
      cancelled: false,
      learningRecorded: false,
    };
    if (responseId) this._playbackByResponse.set(responseId, snapshot);
    return snapshot;
  }

  /** @param {string} responseId */
  _resolvePlaybackResponseId(responseId) {
    if (typeof responseId === "string" && responseId) return responseId;
    if (!this._anonymousPlaybackResponseId) {
      this._anonymousPlaybackSerial += 1;
      this._anonymousPlaybackResponseId = (
        `anonymous-${this._playbackGeneration}-${this._playbackTurnSerial}-${this._anonymousPlaybackSerial}`
      );
    }
    return this._anonymousPlaybackResponseId;
  }

  /** @param {string} responseId */
  _markPlaybackResponseUnordered(responseId) {
    const resolvedId = this._resolvePlaybackResponseId(responseId);
    const snapshot = this._playbackSnapshot(resolvedId, false);
    snapshot.ordered = false;
    snapshot.policy = this._conservativePlaybackPolicy();
    snapshot.primeMs = snapshot.policy.targetMs;
    snapshot.ceilingMs = snapshot.policy.ceilingMs;
    snapshot.targetSamples = Math.ceil((snapshot.primeMs * OUTPUT_SAMPLE_RATE) / 1_000);
    snapshot.ceilingSamples = Math.ceil((snapshot.ceilingMs * OUTPUT_SAMPLE_RATE) / 1_000);
    snapshot.gaps = [];
    snapshot.logicalBlockCount = 0;
    snapshot.nextLogicalBoundarySamples = 0;
    snapshot.lastLogicalBlockAt = null;
    snapshot.latestLogicalBlockGapMs = null;
    return { responseId: resolvedId, snapshot };
  }

  /** @param {string} responseId */
  _finalizePlaybackLearning(responseId) {
    const snapshot = this._playbackByResponse.get(responseId)
      || this._completedPlaybackResponses.get(responseId);
    if (!snapshot || !snapshot.networkDone || !snapshot.workletDrained) return;
    if (!snapshot.learningRecorded) {
      snapshot.learningRecorded = true;
      const cleanFullPrime = (
        snapshot.ordered
        && snapshot.responseStatus === "completed"
        && !snapshot.cancelled
        && !snapshot.underrun
        && !snapshot.forcedShort
        && snapshot.policy.targetMs >= snapshot.policy.ceilingMs
        && snapshot.inputSamples >= snapshot.ceilingSamples
      );
      if (snapshot.ordered
          && snapshot.responseStatus === "completed"
          && !snapshot.cancelled
          && !snapshot.underrun
          && !snapshot.forcedShort) {
        this._playbackLearning.record(snapshot.policy, {
          gaps: snapshot.gaps,
          cleanFullPrime,
        });
      }
    }
    this._completedPlaybackResponses.delete(responseId);
  }

  /** Retain a bounded tombstone so late PCM cannot recreate a current snapshot. */
  /** @param {string} responseId */
  _retirePlaybackResponse(responseId) {
    if (!responseId) return;
    const snapshot = this._playbackByResponse.get(responseId);
    if (snapshot) {
      this._completedPlaybackResponses.delete(responseId);
      this._completedPlaybackResponses.set(responseId, snapshot);
      while (this._completedPlaybackResponses.size > MAX_PLAYBACK_RESPONSE_TOMBSTONES) {
        const oldest = this._completedPlaybackResponses.keys().next().value;
        if (!oldest) break;
        this._completedPlaybackResponses.delete(oldest);
      }
    }
    this._playbackByResponse.delete(responseId);
    this._stalePlaybackResponses.delete(responseId);
    this._stalePlaybackResponses.add(responseId);
    while (this._stalePlaybackResponses.size > MAX_PLAYBACK_RESPONSE_TOMBSTONES) {
      const oldest = this._stalePlaybackResponses.values().next().value;
      if (!oldest) break;
      this._stalePlaybackResponses.delete(oldest);
    }
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
      inputSamples: snapshot.inputSamples,
    });
  }

  /** Clear a generation once; late response chunks retain their older tag. */
  /** @param {string} reason */
  _invalidatePlayback(reason) {
    if (this._playbackGenerationInvalidated) return;
    this._playbackGeneration += 1;
    this._playbackGenerationInvalidated = true;
    this._activePlaybackTurn = null;
    this._anonymousPlaybackResponseId = "";
    for (const snapshot of [...this._playbackByResponse.values(), ...this._completedPlaybackResponses.values()]) {
      if (snapshot.generation < this._playbackGeneration) snapshot.cancelled = true;
    }
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
    await ctx.audioWorklet.addModule(new URL("audio-playback.js?v=16-adaptive-safe-start", base).href);
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
      inputRate: OUTPUT_SAMPLE_RATE,
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
    if (Number.isSafeInteger(generation)
        && generation !== this._playbackGeneration
        && data?.kind !== "stale_chunk_rejected") return;
    const queueDetail = {
      queued_ms: Number(data?.queuedMs || 0),
      queued_samples: Number(data?.queuedSamples || 0),
      prime_target_ms: Number(data?.primeTargetMs || 0),
      prime_ceiling_ms: Number(data?.primeCeilingMs || 0),
      target_samples: Number(data?.targetSamples || 0),
      ceiling_samples: Number(data?.ceilingSamples || 0),
      input_samples: Number(data?.inputSamples || 0),
      generation: Number.isSafeInteger(generation) ? generation : this._playbackGeneration,
      state: data?.state || "unknown",
      underruns: Number(data?.underruns || 0),
      reprimes: Number(data?.reprimes || 0),
      stale_chunks: Number(data?.staleChunks || 0),
      clears: Number(data?.clears || 0),
    };
    const streamId = typeof data?.streamId === "string" ? data.streamId : "";
    const snapshot = streamId
      ? this._playbackByResponse.get(streamId) || this._completedPlaybackResponses.get(streamId)
      : null;
    queueDetail.safe_start_mode = snapshot?.policy?.mode || null;
    queueDetail.turn_id = snapshot?.turnId || null;
    queueDetail.cold_ceiling_ms = snapshot?.policy?.ceilingMs ?? Number(data?.primeCeilingMs || 0);
    queueDetail.effective_target_ms = snapshot?.policy?.targetMs ?? Number(data?.primeTargetMs || 0);
    queueDetail.latest_logical_block_gap_ms = snapshot?.latestLogicalBlockGapMs ?? null;
    queueDetail.logical_block_gap_p95_ms = snapshot?.gaps?.length
      ? _nearestRankP95(snapshot.gaps)
      : (snapshot?.policy?.gapP95Ms ?? 0);
    queueDetail.jitter_margin_ms = snapshot?.policy?.jitterMarginMs ?? PLAYBACK_WARM_GAP_MARGIN_MS;
    queueDetail.fallback_reason = snapshot?.policy?.fallbackReason || "playback_snapshot_unavailable";
    if (data?.kind === "started") {
      if (streamId && !this._stalePlaybackResponses.has(streamId)) {
        this._heardResponses.add(streamId);
      }
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
      if (snapshot) {
        snapshot.underrun = true;
        if (!snapshot.learningRecorded) {
          this._playbackLearning.record(snapshot.policy, {
            gaps: snapshot.gaps,
            underrun: true,
          });
          snapshot.learningRecorded = true;
        }
      }
      this.dispatchEvent(new CustomEvent("pipeline-metric", {
        detail: { stage: "playback", status: "underrun", source: "browser", detail: queueDetail },
      }));
      return;
    }
    if ((data?.kind === "primed" || data?.kind === "reprimed") && snapshot && data.forced) {
      snapshot.forcedShort = true;
    }
    if ((data?.kind === "stream_drained" || data?.kind === "drained") && snapshot) {
      snapshot.workletDrained = true;
      if (data.cleared) snapshot.cancelled = true;
      this._finalizePlaybackLearning(streamId);
    }
    if (data?.kind === "drained") {
      this._aiSpeaking = false;
      // A barge-in sets user-speaking before the clear reaches the worklet. Do
      // not let the resulting drained event overwrite that newer microphone
      // state; only retire an active playback status here.
      if (this._status === "ai-speaking") {
        this._setStatus(this._responseActive() ? "processing" : "connected");
      }
    }
    if (["primed", "reprimed", "stream_drained", "drained", "cleared", "stale_chunk_rejected"].includes(data?.kind)) {
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
    // Opt-in content-free event tracing for diagnosing turn/transcript issues. Enable with
    // `localStorage.setItem("s2s.debug", "1")` in the browser console.
    if (this._debug) {
      const extra = type.startsWith("conversation.item.input_audio_transcription")
        ? ` item=${event.item_id} ci=${event.content_index} chars=${String(event.delta ?? event.transcript ?? "").length}`
        : type.startsWith("response.")
          ? ` resp=${event.response_id ?? event.response?.id ?? ""} status=${event.response?.status ?? ""} chars=${String(event.transcript ?? "").length}`
          : "";
      console.debug(`[ws] ${type}${extra}`);
    }

    switch (type) {
      case "session.created":
        // Endpoint validation must finish before session.update enables mic audio.
        this.updateLocalPipeline(
          this.options.pipelineConfig || {},
          this.options.playbackConfig || null,
        );
        break;

      case "session.updated":
        // Acknowledged by server, nothing to do.
        break;

      case "pipeline.runtime":
        {
          const runtimeIdentity = _runtimePlaybackIdentity(event.runtime);
          if (this._playbackRuntimeIdentity && runtimeIdentity !== this._playbackRuntimeIdentity) {
            this._acknowledgedPlaybackSignature = resolveAdaptivePlaybackSignature(
              this._lastAcknowledgedPipelineConfig,
              this._acknowledgedPlaybackConfig,
              runtimeIdentity,
              false,
            );
            const conservative = this._playbackLearning.policy(this._acknowledgedPlaybackSignature);
            this._playbackPrimeMs = conservative.targetMs;
            this._playbackCeilingMs = conservative.ceilingMs;
          }
          this._playbackRuntimeIdentity = runtimeIdentity;
        }
        this.dispatchEvent(new CustomEvent("backend-runtime", { detail: event.runtime || {} }));
        break;

      case "input_audio_buffer.speech_started":
        // Stop every playing or queued sample. The worklet's resulting drained
        // event owns `_aiSpeaking`; user-speaking takes status precedence while
        // that asynchronous clear acknowledgement is in flight.
        this._invalidatePlayback("barge-in");
        this._setStatus("user-speaking");
        this.dispatchEvent(new CustomEvent("turn-state", {
          detail: {
            status: "speech_started",
            itemId: typeof event.item_id === "string" ? event.item_id : "",
          },
        }));
        this.dispatchEvent(new CustomEvent("pipeline-metric", {
          detail: { stage: "mic", status: "speaking", source: "browser", detail: {} },
        }));
        break;

      case "input_audio_buffer.speech_stopped":
        this._playbackGenerationInvalidated = false;
        this._freezePlaybackTurnPolicy();
        if (this._status === "user-speaking") this._setStatus("processing");
        this._speechStoppedAtMs = performance.now();
        this._firstPlaybackReported = false;
        this.dispatchEvent(new CustomEvent("turn-state", {
          detail: {
            status: "speech_stopped",
            itemId: typeof event.item_id === "string" ? event.item_id : "",
          },
        }));
        this.dispatchEvent(new CustomEvent("pipeline-metric", {
          detail: { stage: "mic", status: "captured", source: "browser", detail: {} },
        }));
        break;

      case "response.created":
        // A response now owns the slot — count it and clear our create guard
        // (this confirms either our create or a server-initiated one).
        this._openResponses++;
        this._createInFlight = false;
        {
          const rawResponseId = typeof event.response?.id === "string" ? event.response.id : "";
          if (!rawResponseId) this._anonymousPlaybackResponseId = "";
          const responseId = this._resolvePlaybackResponseId(rawResponseId);
          const ordered = !!rawResponseId
            && !this._openPlaybackResponseIds.has(responseId)
            && !this._stalePlaybackResponses.has(responseId)
            && this._openPlaybackResponseIds.size === 0;
          if (ordered) {
            this._openPlaybackResponseIds.add(responseId);
            this._playbackSnapshot(responseId, true);
          } else {
            this._openPlaybackResponseIds.add(responseId);
            this._markPlaybackResponseUnordered(responseId);
          }
        }
        if (this._status === "connected" || this._status === "user-speaking") {
          this._setStatus("processing");
        }
        break;

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
        this._pushAudioDelta(event.delta, rid);
        break;
      }

      case "response.audio.done":
      case "response.output_audio.done": {
        const rawRid = typeof (event.response_id ?? event.response?.id) === "string"
          ? (event.response_id ?? event.response?.id)
          : "";
        const rid = this._resolvePlaybackResponseId(rawRid);
        if (!rawRid || !this._openPlaybackResponseIds.has(rid)) {
          this._markPlaybackResponseUnordered(rid);
        }
        this._finishPlaybackResponse(rid);
        break;
      }

      case "response.content_part.added": {
        // A declared audio part is not audible until the worklet renders it.
        break;
      }

      case "response.done": {
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
        const status = event.response?.status ?? "completed";
        const responseId = event.response?.id ?? "";
        const playbackResponseId = this._resolvePlaybackResponseId(responseId);
        if (!responseId || !this._openPlaybackResponseIds.has(playbackResponseId)) {
          this._markPlaybackResponseUnordered(playbackResponseId);
        }
        this._openPlaybackResponseIds.delete(playbackResponseId);
        const responsePlayback = playbackResponseId
          ? this._playbackByResponse.get(playbackResponseId)
          : null;
        const responseRetired = playbackResponseId
          ? this._stalePlaybackResponses.has(playbackResponseId)
          : false;
        if (status === "cancelled" || status === "canceled") {
          // Barge-in already advanced the generation. Do not clear the new turn
          // a second time when the cancelled old response closes afterward.
          if (!responseRetired
              && (!responsePlayback || responsePlayback.generation === this._playbackGeneration)) {
            this._invalidatePlayback("response-cancelled");
          }
        } else if (!responseRetired) {
          // Some compatible peers omit output_audio.done. Keep the end flush
          // idempotent and use response.done as the terminal fallback.
          this._finishPlaybackResponse(playbackResponseId);
        }
        if (responsePlayback) {
          responsePlayback.networkDone = true;
          responsePlayback.responseStatus = status;
          responsePlayback.cancelled = status === "cancelled" || status === "canceled";
          responsePlayback.forcedShort = responsePlayback.inputSamples < responsePlayback.targetSamples;
        }
        const endToEndMs = this._speechStoppedAtMs == null
          ? null
          : Math.max(0, performance.now() - this._speechStoppedAtMs);
        this.dispatchEvent(new CustomEvent("pipeline-metric", {
          detail: {
            stage: "response",
            status: "done",
            source: "browser",
            elapsed_ms: endToEndMs,
            detail: { end_to_end_ms: endToEndMs, response_status: status },
          },
        }));
        // Did the worklet actually render this response's first sample? This
        // remains false when response.done beats startup priming; network PCM
        // receipt alone must not make a speculative response look heard.
        const audible = responseId ? this._heardResponses.has(responseId) : false;
        this._heardResponses.delete(responseId);
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
        this.dispatchEvent(new CustomEvent("response-finished", {
          detail: { responseId, status, audible, transcript },
        }));
        this._retirePlaybackResponse(playbackResponseId);
        this._finalizePlaybackLearning(playbackResponseId);
        if (!this._responseActive()) this._resolveResponseIdleWaiters();
        // The slot is free now — replay a queued create (e.g. a tool follow-up
        // that arrived while this response was still running).
        this._flushQueuedCreate();
        break;
      }

      case "pipeline.metric": {
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
        // Preserve the public argument string exactly. A non-string value is a
        // malformed call, not an implicit empty object; the browser executor
        // will return invalid_tool_arguments through the normal result path.
        const args = typeof event.arguments === "string" ? event.arguments : null;
        const callId = typeof event.call_id === "string" ? event.call_id : "";
        if (name && callId.trim()) {
          this.dispatchEvent(new CustomEvent("toolcall", {
            detail: {
              name,
              arguments: args,
              callId,
              responseId: typeof event.response_id === "string" ? event.response_id : "",
              itemId: typeof event.item_id === "string" ? event.item_id : "",
            },
          }));
        } else {
          // Without both fields there is no safe call/result transaction. Do
          // not dispatch `toolcall`: that would execute a side effect which
          // cannot be paired with function_call_output. The UI renders this
          // fixed, content-free protocol failure instead.
          const code = name ? "missing_call_id" : "missing_tool_name";
          this.dispatchEvent(new CustomEvent("tool-protocol-error", { detail: { code } }));
          console.warn(`[ws] rejected invalid function_call_arguments.done (${code})`);
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

  /** @param {string} b64 @param {string} responseId */
  _pushAudioDelta(b64, responseId = "") {
    if (!this._playbackNode) return;
    if (!b64) return;
    const rawResponseId = responseId;
    const resolvedResponseId = this._resolvePlaybackResponseId(rawResponseId);
    if (resolvedResponseId && this._stalePlaybackResponses.has(resolvedResponseId)) {
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
    const knownOrderedResponse = !!rawResponseId && this._openPlaybackResponseIds.has(resolvedResponseId);
    const snapshot = knownOrderedResponse
      ? this._playbackSnapshot(resolvedResponseId, true)
      : this._markPlaybackResponseUnordered(resolvedResponseId).snapshot;
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
    if (snapshot.generation !== this._playbackGeneration) {
      this.dispatchEvent(new CustomEvent("pipeline-metric", {
        detail: {
          stage: "playback",
          status: "stale_chunk_rejected",
          source: "browser",
          detail: {
            reason: "generation_mismatch",
            generation: snapshot.generation,
            current_generation: this._playbackGeneration,
          },
        },
      }));
      return;
    }
    const bytes = base64ToBytes(b64);
    if (bytes.byteLength === 0 || bytes.byteLength % 2 !== 0) {
      snapshot.ordered = false;
      this.dispatchEvent(new CustomEvent("pipeline-metric", {
        detail: {
          stage: "playback",
          status: "stale_chunk_rejected",
          source: "browser",
          detail: { reason: "invalid_pcm_length", current_generation: this._playbackGeneration },
        },
      }));
      return;
    }
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const samples = new Float32Array(bytes.byteLength / 2);
    for (let i = 0; i < samples.length; i++) {
      const s = view.getInt16(i * 2, true);
      samples[i] = s < 0 ? s / 0x8000 : s / 0x7fff;
    }
    const inputSampleOffset = snapshot.inputSamples;
    snapshot.inputSamples += samples.length;
    snapshot.chunkCount += 1;
    const nextBoundary = snapshot.nextLogicalBoundarySamples;
    const steadyBlockSamples = snapshot.policy.steadyBlockSamples;
    if (snapshot.policy.learning
        && nextBoundary > 0
        && steadyBlockSamples > 0
        && snapshot.inputSamples >= nextBoundary) {
      const boundariesCrossed = 1 + Math.floor(
        (snapshot.inputSamples - nextBoundary) / steadyBlockSamples,
      );
      snapshot.logicalBlockCount += boundariesCrossed;
      snapshot.nextLogicalBoundarySamples += boundariesCrossed * steadyBlockSamples;
      // A coalesced transport chunk can contain several complete decoder
      // blocks. It contributes one arrival timestamp, never artificial zero or
      // packet-sized gaps between those boundaries.
      const now = this._playbackClock();
      if (Number.isFinite(snapshot.lastLogicalBlockAt)) {
        const gap = now - snapshot.lastLogicalBlockAt;
        if (Number.isFinite(gap) && gap > 0) {
          snapshot.gaps = [...snapshot.gaps, gap].slice(-PLAYBACK_GAP_WINDOW);
          snapshot.latestLogicalBlockGapMs = gap;
        }
      }
      snapshot.lastLogicalBlockAt = now;
    }
    this._playbackNode.port.postMessage({
      kind: "audio",
      samples,
      generation: snapshot.generation,
      primeMs: snapshot.primeMs,
      ceilingMs: snapshot.ceilingMs,
      targetSamples: snapshot.targetSamples,
      ceilingSamples: snapshot.ceilingSamples,
      inputSampleOffset,
      inputSampleCount: samples.length,
      streamId: resolvedResponseId,
    }, [samples.buffer]);
    // A clear is idempotent only until a genuinely new current-generation
    // stream is accepted. Non-mic replacements must re-arm Stop/barge-in so
    // their queued samples can be cleared independently of the prior response.
    this._playbackGenerationInvalidated = false;
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
    this._queuePlaybackConfig(config, playbackConfig);
    this._send({ type: "pipeline.config.update", config });
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
   * @param {{ image?: string, toolChoice?: "auto"|"none", tools?: object[] }} [opts] Optional
   *   response-scoped overrides. `image` is a data URL sent as a
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
   *  @param {{ image?: string, toolChoice?: "auto"|"none", tools?: object[] }} [opts] */
  requestToolResponse(opts = {}) {
    if (!this._ws || this._ws.readyState !== WebSocket.OPEN) return;
    if (opts.image) this.sendUserImage(opts.image);
    this._createInFlight = true;
    this._send({
      type: "response.create",
      ...((opts.toolChoice || opts.tools) ? {
        response: {
          ...(opts.tools ? { tools: opts.tools } : {}),
          ...(opts.toolChoice ? { tool_choice: opts.toolChoice } : {}),
        },
      } : {}),
    });
  }

  /** True while a response occupies the single backend slot. */
  _responseActive() {
    return this._openResponses > 0 || this._createInFlight;
  }

  /** Send a response.create immediately and arm the in-flight guard. Any image
   *  on the payload is added as user content right before the create.
   *  @param {{ image?: string, toolChoice?: "auto"|"none", tools?: object[] }} [opts] */
  _createResponseNow(opts = {}) {
    if (!this._ws || this._ws.readyState !== WebSocket.OPEN) return;
    if (opts.image) this.sendUserImage(opts.image);
    this._createInFlight = true;
    this._send({
      type: "response.create",
      ...((opts.toolChoice || opts.tools) ? {
        response: {
          ...(opts.tools ? { tools: opts.tools } : {}),
          ...(opts.toolChoice ? { tool_choice: opts.toolChoice } : {}),
        },
      } : {}),
    });
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
    this._activePlaybackTurn = null;
    this._anonymousPlaybackResponseId = "";
    this._playbackByResponse.clear();
    this._completedPlaybackResponses.clear();
    this._openPlaybackResponseIds.clear();
    this._stalePlaybackResponses.clear();
    this._heardResponses.clear();
    this._setStatus("closed");
  }
}
