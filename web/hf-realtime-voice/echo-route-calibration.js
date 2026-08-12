// @ts-check

const ROUTE_KEY_DOMAIN = "hf-realtime-voice/audio-route/v1";
const MIN_CALIBRATION_SAMPLES = 20;
const MIN_CALIBRATION_SPAN_MS = 2000;
const MAX_CALIBRATION_SAMPLES = 256;
const MAX_CALIBRATION_JITTER_MS = 120;
const MAX_CALIBRATION_LAG_MS = 1000;

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
  const microphone = typeof microphoneRoute === "string" && microphoneRoute
    ? microphoneRoute
    : "default-input";
  const output = typeof outputRoute === "string" && outputRoute
    ? outputRoute
    : "default-output";
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
});
