// @ts-check

const ROUTE_KEY_DOMAIN = "hf-realtime-voice/audio-route/v1";
const MIN_CALIBRATION_SAMPLES = 20;
const MIN_CALIBRATION_SPAN_MS = 2000;
const MAX_CALIBRATION_SAMPLES = 256;
const MAX_CALIBRATION_JITTER_MS = 120;
const MAX_CALIBRATION_LAG_MS = 1000;
const MAX_SAVED_ROUTES = 16;
const ECHO_ROUTE_KEY_RE = /^route_[0-9a-f]{64}$/;

export const DEFAULT_ECHO_CALIBRATION = Object.freeze({
  delayMs: 0,
  suppressionStrength: 0.65,
  leakageThreshold: 0.65,
  doubleTalkSensitivity: 0.5,
  echoTailMs: 350,
});

function finiteNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function percentile(sorted, percentileValue) {
  if (!sorted.length) return null;
  const position = (sorted.length - 1) * percentileValue;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sorted[lower];
  const fraction = position - lower;
  return sorted[lower] + ((sorted[upper] - sorted[lower]) * fraction);
}

function boundedMetric(value) {
  return Math.round(value * 10) / 10;
}

function clamp(value, minimum, maximum, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(minimum, Math.min(maximum, number)) : fallback;
}

/** @param {unknown} value */
export function isOpaqueEchoRouteKey(value) {
  return typeof value === "string" && ECHO_ROUTE_KEY_RE.test(value);
}

/** @param {Record<string, unknown> | null | undefined} value */
export function normalizeEchoCalibration(value) {
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
    echoTailMs: clamp(value?.echoTailMs, 350, 1000, DEFAULT_ECHO_CALIBRATION.echoTailMs),
  };
}

/**
 * Keep only bounded calibrations keyed by this client's opaque route digest.
 * Legacy raw `microphone::output` keys are intentionally discarded.
 *
 * @param {unknown} value
 * @param {number} [limit]
 */
export function sanitizeEchoCalibrations(value, limit = MAX_SAVED_ROUTES) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};
  const entries = Object.entries(value)
    .filter(([routeKey, calibration]) => (
      isOpaqueEchoRouteKey(routeKey)
      && calibration
      && typeof calibration === "object"
      && !Array.isArray(calibration)
    ))
    .slice(-Math.max(1, Math.min(MAX_SAVED_ROUTES, Math.floor(limit) || MAX_SAVED_ROUTES)));
  return Object.fromEntries(entries.map(([routeKey, calibration]) => [
    routeKey,
    normalizeEchoCalibration(/** @type {Record<string, unknown>} */ (calibration)),
  ]));
}

/**
 * Insert one current calibration while retaining at most the newest 16 routes.
 * @param {unknown} current
 * @param {unknown} routeKey
 * @param {Record<string, unknown>} calibration
 */
export function upsertEchoCalibration(current, routeKey, calibration) {
  if (!isOpaqueEchoRouteKey(routeKey)) return sanitizeEchoCalibrations(current);
  const sanitized = sanitizeEchoCalibrations(current);
  delete sanitized[routeKey];
  sanitized[routeKey] = normalizeEchoCalibration(calibration);
  return sanitizeEchoCalibrations(sanitized);
}

/**
 * Load and migrate browser persistence without leaving legacy raw route keys
 * behind. If rewriting fails after removal, the safe outcome is no persisted
 * calibration rather than retaining identifying legacy material.
 * @param {{ getItem: Function, setItem: Function, removeItem: Function }} storage
 * @param {string} storageKey
 */
export function loadSanitizedEchoCalibrations(storage, storageKey) {
  let raw = "";
  try {
    raw = storage.getItem(storageKey) || "";
  } catch {
    return {};
  }
  let parsed = {};
  try {
    parsed = raw ? JSON.parse(raw) : {};
  } catch {
    try { storage.removeItem(storageKey); } catch { /* ignored */ }
    return {};
  }
  const sanitized = sanitizeEchoCalibrations(parsed);
  const encoded = JSON.stringify(sanitized);
  if (raw !== encoded) {
    try {
      storage.removeItem(storageKey);
      if (encoded !== "{}") storage.setItem(storageKey, encoded);
    } catch {
      try { storage.removeItem(storageKey); } catch { /* ignored */ }
    }
  }
  return sanitized;
}

/** @param {unknown} status */
export function echoRouteIdentityMatches(status, routeKey, routeEpoch) {
  return !!status
    && isOpaqueEchoRouteKey(routeKey)
    && Number.isSafeInteger(routeEpoch)
    && status.routeKey === routeKey
    && status.routeEpoch === routeEpoch;
}

/**
 * Build a domain-separated opaque key for the active microphone/output route.
 * Raw route identifiers are used only as transient digest input and are never
 * included in the result or an error.
 *
 * @param {unknown} microphoneRoute
 * @param {unknown} outputRoute
 * @param {SubtleCrypto} [subtle]
 * @returns {Promise<string>}
 */
export async function fingerprintEchoRoute(
  microphoneRoute,
  outputRoute,
  subtle = globalThis.crypto?.subtle,
) {
  if (!subtle || typeof subtle.digest !== "function") {
    throw new Error("Secure route fingerprinting is unavailable");
  }
  if (
    typeof microphoneRoute !== "string"
    || !microphoneRoute.length
    || typeof outputRoute !== "string"
    || !outputRoute.length
  ) {
    throw new Error("Resolved audio route is unavailable");
  }
  const microphone = microphoneRoute;
  const output = outputRoute;
  const payload = new TextEncoder().encode(
    `${ROUTE_KEY_DOMAIN}\u0000${microphone.length}:${microphone}\u0000${output.length}:${output}`,
  );
  const digest = new Uint8Array(await subtle.digest("SHA-256", payload));
  let hex = "";
  for (const value of digest) hex += value.toString(16).padStart(2, "0");
  return `route_${hex}`;
}

/**
 * Content-free robust delay calibration. It retains only bounded timing
 * samples that were measured while playback was active and double-talk was
 * absent. A negative eligible measurement rejects the complete cohort.
 */
export class EchoRouteCalibration {
  constructor() {
    /** @type {{ lagMs: number, timestampMs: number }[]} */
    this._samples = [];
    this._invalidReason = "";
  }

  reset() {
    this._samples = [];
    this._invalidReason = "";
  }

  /**
   * @param {{ lagMs?: unknown, timestampMs?: unknown, playbackActive?: unknown, doubleTalk?: unknown }} sample
   * @returns {{ accepted: boolean, reason: string, sampleCount: number }}
   */
  add(sample) {
    if (!sample || sample.playbackActive !== true || sample.doubleTalk !== false) {
      return { accepted: false, reason: "ineligible", sampleCount: this._samples.length };
    }
    const lagMs = finiteNumber(sample.lagMs);
    const timestampMs = finiteNumber(sample.timestampMs);
    if (lagMs === null || timestampMs === null) {
      return { accepted: false, reason: "non_finite", sampleCount: this._samples.length };
    }
    if (lagMs < 0) {
      this._invalidReason = "negative_lag";
      return { accepted: false, reason: this._invalidReason, sampleCount: this._samples.length };
    }
    if (lagMs > MAX_CALIBRATION_LAG_MS) {
      this._invalidReason = "lag_out_of_range";
      return { accepted: false, reason: this._invalidReason, sampleCount: this._samples.length };
    }
    this._samples.push({ lagMs, timestampMs });
    if (this._samples.length > MAX_CALIBRATION_SAMPLES) this._samples.shift();
    return { accepted: true, reason: "", sampleCount: this._samples.length };
  }

  /**
   * @param {{ outputLatencyMs?: unknown }} [options]
   * @returns {{
   *   accepted: boolean,
   *   reason: string,
   *   sampleCount: number,
   *   spanMs: number,
   *   medianMs: number | null,
   *   p95Ms: number | null,
   *   jitterMs: number | null,
   *   delayMs: number | null,
   *   outputLatencyMs: number,
   * }}
   */
  result(options = {}) {
    const outputLatencyValue = finiteNumber(options.outputLatencyMs);
    const outputLatencyMs = outputLatencyValue === null
      ? 0
      : Math.max(0, Math.min(500, outputLatencyValue));
    const sampleCount = this._samples.length;
    const timestamps = this._samples.map((sample) => sample.timestampMs);
    const spanMs = sampleCount
      ? Math.max(...timestamps) - Math.min(...timestamps)
      : 0;
    const base = {
      accepted: false,
      reason: this._invalidReason,
      sampleCount,
      spanMs: boundedMetric(Math.max(0, spanMs)),
      medianMs: null,
      p95Ms: null,
      jitterMs: null,
      delayMs: null,
      outputLatencyMs: boundedMetric(outputLatencyMs),
    };
    if (this._invalidReason) return base;
    if (sampleCount < MIN_CALIBRATION_SAMPLES) {
      return { ...base, reason: "insufficient_samples" };
    }
    if (spanMs < MIN_CALIBRATION_SPAN_MS) {
      return { ...base, reason: "insufficient_span" };
    }

    const sorted = this._samples.map((sample) => sample.lagMs).sort((a, b) => a - b);
    const medianMs = percentile(sorted, 0.5);
    const p05Ms = percentile(sorted, 0.05);
    const p95Ms = percentile(sorted, 0.95);
    const jitterMs = p95Ms - p05Ms;
    const metrics = {
      ...base,
      medianMs: boundedMetric(medianMs),
      p95Ms: boundedMetric(p95Ms),
      jitterMs: boundedMetric(jitterMs),
      delayMs: boundedMetric(Math.max(0, medianMs - outputLatencyMs)),
    };
    if (jitterMs > MAX_CALIBRATION_JITTER_MS) {
      return { ...metrics, reason: "unstable_jitter" };
    }
    return { ...metrics, accepted: true, reason: "" };
  }
}

export const echoCalibrationLimits = Object.freeze({
  minimumSamples: MIN_CALIBRATION_SAMPLES,
  minimumSpanMs: MIN_CALIBRATION_SPAN_MS,
  maximumJitterMs: MAX_CALIBRATION_JITTER_MS,
  maximumSavedRoutes: MAX_SAVED_ROUTES,
});
