// @ts-check

export const STRICT_MAX_BUFFER_MS = 120;
export const STRICT_NEAR_END_CONFIRM_MS = 60;
export const STRICT_NEAR_END_HOLD_MS = 250;

function finite(value) {
  return Number.isFinite(Number(value)) ? Number(value) : null;
}

/**
 * Translate content-free AEC3 metrics into conservative Strict evidence.
 * A double-talk flag is not sufficient by itself because far-end echo can
 * occasionally trip the derived detector.
 *
 * @param {boolean} playbackActive
 * @param {Record<string, any> | null} metrics
 * @param {{ suppressionStrength: number, leakageThreshold: number, doubleTalkSensitivity: number }} calibration
 */
export function classifyStrictEcho(playbackActive, metrics, calibration) {
  if (!playbackActive || !metrics) {
    return { nearEndEvidence: false, highConfidenceEchoOnly: false };
  }
  const residual = finite(metrics.residualEchoLikelihood);
  const captureRms = finite(metrics.captureRms);
  const outputRms = finite(metrics.outputRms);
  const outputRatio = captureRms !== null && captureRms > 1e-6 && outputRms !== null
    ? outputRms / captureRms
    : null;
  const nearEndFloor = 0.65 - (0.5 * calibration.doubleTalkSensitivity);
  const nearEndEvidence = metrics.doubleTalk === true
    && outputRatio !== null
    && outputRatio >= nearEndFloor
    && residual !== null
    && residual < 0.8;
  const strength = calibration.suppressionStrength;
  const echoThreshold = calibration.leakageThreshold * (1.2 - (0.7 * strength));
  const highConfidenceEchoOnly = !nearEndEvidence
    && residual !== null
    && residual >= echoThreshold
    && outputRatio !== null
    && outputRatio < nearEndFloor;
  return { nearEndEvidence, highConfidenceEchoOnly };
}

/**
 * A small decision buffer for Strict mode. Frames are never synthesized or
 * replaced with silence: each retained frame is either emitted byte-for-byte
 * or omitted only after high-confidence echo-only evidence.
 */
export class StrictEchoGate {
  /** @param {number} [frameMs] */
  constructor(frameMs = 10) {
    this.frameMs = frameMs;
    this.maxFrames = Math.max(1, Math.floor(STRICT_MAX_BUFFER_MS / frameMs));
    this.confirmFrames = Math.max(1, Math.ceil(STRICT_NEAR_END_CONFIRM_MS / frameMs));
    this.holdFrames = Math.max(1, Math.ceil(STRICT_NEAR_END_HOLD_MS / frameMs));
    this.reset();
  }

  reset() {
    /** @type {{ frame: Float32Array, nearEnd: boolean, echoOnly: boolean }[]} */
    this.pending = [];
    this.nearEndRun = 0;
    this.holdRemaining = 0;
  }

  /**
   * @param {Float32Array} frame
   * @param {{ playbackActive: boolean, nearEndEvidence: boolean, highConfidenceEchoOnly: boolean }} state
   * @returns {{ emit: Float32Array[], suppressedFrames: number, suppressing: boolean, pendingMs: number, nearEndConfirmed: boolean }}
   */
  consume(frame, state) {
    const output = {
      emit: /** @type {Float32Array[]} */ ([]),
      suppressedFrames: 0,
      suppressing: false,
      pendingMs: 0,
      nearEndConfirmed: false,
    };
    if (!state.playbackActive) {
      this._resolvePending(output);
      output.emit.push(frame);
      this.nearEndRun = 0;
      this.holdRemaining = 0;
      return output;
    }

    if (this.holdRemaining > 0) {
      output.emit.push(frame);
      this.holdRemaining -= 1;
      if (state.nearEndEvidence) this.holdRemaining = this.holdFrames;
      return output;
    }

    this.pending.push({
      frame,
      nearEnd: state.nearEndEvidence,
      echoOnly: state.highConfidenceEchoOnly,
    });
    this.nearEndRun = state.nearEndEvidence ? this.nearEndRun + 1 : 0;

    if (this.nearEndRun >= this.confirmFrames) {
      const onsetIndex = Math.max(0, this.pending.length - this.nearEndRun);
      const beforeOnset = this.pending.splice(0, onsetIndex);
      for (const entry of beforeOnset) {
        if (entry.echoOnly) output.suppressedFrames += 1;
        else output.emit.push(entry.frame);
      }
      for (const entry of this.pending) output.emit.push(entry.frame);
      this.pending = [];
      this.nearEndRun = 0;
      this.holdRemaining = this.holdFrames;
      output.nearEndConfirmed = true;
      output.suppressing = output.suppressedFrames > 0;
      return output;
    }

    while (this.pending.length > this.maxFrames) {
      const oldest = this.pending.shift();
      if (!oldest) break;
      if (oldest.echoOnly) output.suppressedFrames += 1;
      else output.emit.push(oldest.frame);
    }
    output.suppressing = output.suppressedFrames > 0
      || (this.pending.length > 0 && this.pending.every((entry) => entry.echoOnly));
    output.pendingMs = this.pending.length * this.frameMs;
    return output;
  }

  /**
   * Resolve buffered state during reset/mode changes without dropping uncertain
   * frames. High-confidence echo-only frames remain omitted.
   * @returns {{ emit: Float32Array[], suppressedFrames: number }}
   */
  flush() {
    const output = {
      emit: /** @type {Float32Array[]} */ ([]),
      suppressedFrames: 0,
      suppressing: false,
      pendingMs: 0,
      nearEndConfirmed: false,
    };
    this._resolvePending(output);
    this.nearEndRun = 0;
    this.holdRemaining = 0;
    return { emit: output.emit, suppressedFrames: output.suppressedFrames };
  }

  _resolvePending(output) {
    for (const entry of this.pending) {
      if (entry.echoOnly) output.suppressedFrames += 1;
      else output.emit.push(entry.frame);
    }
    this.pending = [];
  }
}
