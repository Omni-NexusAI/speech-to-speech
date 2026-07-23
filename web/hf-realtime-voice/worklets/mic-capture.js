// @ts-check
/**
 * AudioWorkletProcessor that resamples the AudioContext rate (typically 48 kHz)
 * down to 16 kHz, packs the result as little-endian Int16 PCM, and posts it
 * back to the main thread in fixed-size chunks.
 *
 * The Hugging Face speech-to-speech WebSocket route expects the
 * `input_audio_buffer.append` payload at 16 kHz PCM16 mono.
 *
 * Design notes:
 *   - 48 -> 16 is an exact 3:1 ratio so we use a 3-tap boxcar average as a
 *     cheap low-pass before decimating. Good enough for voice STT; we lose
 *     a tiny bit of >8 kHz content which the pipeline discards anyway.
 *   - Output frames are emitted at the cadence dictated by `chunkMs`
 *     (default 40 ms = 640 samples = 1280 bytes). The OpenAI Realtime
 *     server batches incoming audio so the cadence is flexible; 20-100 ms
 *     is the sweet spot.
 *   - Float -> Int16 saturates to [-1, 1] before scaling.
 *   - Optional noise gate: per-chunk RMS decides open/closed against a
 *     threshold; the gain ramps (fast attack, hold, slow release) so word
 *     onsets aren't clipped and quiet tails don't click. The gate only
 *     affects the audio we SEND; the main-thread visualiser taps the raw
 *     mic separately. We post the chunk RMS up every frame so the Settings
 *     mic meter can show the live level against the threshold.
 */

const TARGET_RATE = 16000;
const DEFAULT_CHUNK_MS = 40;
// Gate envelope timing (fixed; only the threshold is user-tunable).
const GATE_ATTACK_MS = 5; // open almost instantly so word onsets survive
const GATE_HOLD_MS = 250; // stay open this long after the level drops back under
const GATE_RELEASE_MS = 80; // then fade closed over this long (no click)
const ECHO_HISTORY_MS = 400;
const ECHO_MAX_LAG_MS = 250;
const ECHO_TAIL_MS = 250;
// Adaptive mode waits briefly to distinguish real speech from room playback,
// but preserves the onset in a short local buffer.
const DOUBLE_TALK_MS = 160;
const ECHO_FILTER_TAPS = 512;
const ECHO_NLMS_STEP = 0.12;
const ECHO_NLMS_EPSILON = 1e-7;
const ECHO_CORRELATION_MIN = 0.42;
const ECHO_RESIDUAL_MAX = 0.72;
const REFERENCE_ACTIVE_RMS = 0.001;
const HUMAN_ACTIVE_RMS = 0.004;

class MicCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const chunkMs = options?.processorOptions?.chunkMs ?? DEFAULT_CHUNK_MS;
    this._inputRate = sampleRate;
    this._ratio = this._inputRate / TARGET_RATE;
    this._chunkSamples16k = Math.round((TARGET_RATE * chunkMs) / 1000);
    this._scratch = new Float32Array(0);
    this._referenceScratch = new Float32Array(0);
    this._decimated = new Float32Array(this._chunkSamples16k);
    this._referenceDecimated = new Float32Array(this._chunkSamples16k);
    this._enabled = true;

    this._echoMode = "adaptive";
    this._nativeAec = false;
    this._referenceHistory = new Float32Array(0);
    this._echoFilter = new Float32Array(ECHO_FILTER_TAPS);
    this._echoDelaySamples = 0;
    this._echoDelayConfidence = 0;
    this._echoTailRemaining = 0;
    this._doubleTalkSamples = 0;
    this._pendingHumanChunks = [];
    this._suppressedMs = 0;
    this._echoMetricCounter = 0;

    // Noise gate state. Disabled by default (pure passthrough).
    this._gateEnabled = false;
    this._thresholdLin = 0; // linear amplitude; signal RMS must exceed this to open
    this._gateGain = 1; // smoothed gain currently applied
    this._holdRemaining = 0; // samples left before the gate may start closing
    this._attackCoef = Math.exp(-1 / ((GATE_ATTACK_MS / 1000) * TARGET_RATE));
    this._releaseCoef = Math.exp(-1 / ((GATE_RELEASE_MS / 1000) * TARGET_RATE));
    this._holdSamples = Math.round((GATE_HOLD_MS / 1000) * TARGET_RATE);

    this.port.onmessage = (e) => {
      const data = e.data;
      if (data?.kind === "enable") this._enabled = !!data.value;
      else if (data?.kind === "gate") {
        this._gateEnabled = !!data.enabled;
        // dB -> linear amplitude. When off, threshold 0 keeps the gate open.
        this._thresholdLin = data.enabled ? Math.pow(10, data.thresholdDb / 20) : 0;
      } else if (data?.kind === "echo_guard") {
        const nextMode = ["off", "adaptive", "strict"].includes(data.mode) ? data.mode : "adaptive";
        if (nextMode !== this._echoMode) this._resetEchoState();
        this._echoMode = nextMode;
        this._nativeAec = !!data.nativeAec;
      } else if (data?.kind === "echo_reset") {
        this._resetEchoState();
      }
    };
  }

  _resetEchoState() {
    this._referenceHistory = new Float32Array(0);
    this._echoFilter = new Float32Array(ECHO_FILTER_TAPS);
    this._echoDelaySamples = 0;
    this._echoDelayConfidence = 0;
    this._echoTailRemaining = 0;
    this._doubleTalkSamples = 0;
    this._pendingHumanChunks = [];
    this._suppressedMs = 0;
  }

  /**
   * Append `incoming` to the internal scratch buffer, then emit as many
   * full output chunks as we have material for.
   * @param {Float32Array} incoming
   * @param {Float32Array | null} reference
   */
  _ingest(incoming, reference) {
    if (incoming.length === 0) return;
    const next = new Float32Array(this._scratch.length + incoming.length);
    next.set(this._scratch, 0);
    next.set(incoming, this._scratch.length);
    this._scratch = next;
    const referenceNext = new Float32Array(this._referenceScratch.length + incoming.length);
    referenceNext.set(this._referenceScratch, 0);
    if (reference) referenceNext.set(reference.subarray(0, incoming.length), this._referenceScratch.length);
    this._referenceScratch = referenceNext;
    this._maybeEmit();
  }

  _appendReference(chunk) {
    const maxSamples = Math.round((ECHO_HISTORY_MS / 1000) * TARGET_RATE);
    const next = new Float32Array(Math.min(maxSamples, this._referenceHistory.length + chunk.length));
    const keep = Math.max(0, next.length - chunk.length);
    if (keep > 0) next.set(this._referenceHistory.subarray(this._referenceHistory.length - keep), 0);
    next.set(chunk.subarray(Math.max(0, chunk.length - next.length)), keep);
    this._referenceHistory = next;
  }

  _echoMatch(mic, micEnergy) {
    const history = this._referenceHistory;
    const n = mic.length;
    const maxLag = Math.min(Math.round((ECHO_MAX_LAG_MS / 1000) * TARGET_RATE), history.length - n);
    if (micEnergy <= 1e-9 || maxLag < 0) {
      return { correlation: 0, residual: 1, lagMs: 0, lagSamples: 0, start: -1 };
    }
    let bestCorrelation = 0;
    let bestStart = -1;
    let bestRefEnergy = 0;
    for (let lag = 0; lag <= maxLag; lag += 16) {
      const start = history.length - n - lag;
      let dot = 0;
      let refEnergy = 0;
      for (let i = 0; i < n; i++) {
        const ref = history[start + i];
        dot += mic[i] * ref;
        refEnergy += ref * ref;
      }
      if (refEnergy <= 1e-9) continue;
      const correlation = Math.abs(dot) / Math.sqrt(micEnergy * refEnergy);
      if (correlation > bestCorrelation) {
        bestCorrelation = correlation;
        bestStart = start;
        bestRefEnergy = refEnergy;
      }
    }
    if (bestStart < 0) {
      return { correlation: 0, residual: 1, lagMs: 0, lagSamples: 0, start: -1 };
    }
    let dot = 0;
    for (let i = 0; i < n; i++) dot += mic[i] * history[bestStart + i];
    const gain = dot / bestRefEnergy;
    let residualEnergy = 0;
    for (let i = 0; i < n; i++) {
      const residual = mic[i] - gain * history[bestStart + i];
      residualEnergy += residual * residual;
    }
    const lagSamples = history.length - n - bestStart;
    return {
      correlation: bestCorrelation,
      residual: Math.sqrt(residualEnergy / micEnergy),
      lagMs: (lagSamples / TARGET_RATE) * 1000,
      lagSamples,
      start: bestStart,
    };
  }

  _cancelEcho(mic, adapt) {
    const history = this._referenceHistory;
    const n = mic.length;
    const residual = new Float32Array(n);
    const start = history.length - n - this._echoDelaySamples;
    if (start < 0) {
      residual.set(mic);
      return { residual, residualRatio: 1, residualRms: Math.sqrt(mic.reduce((sum, value) => sum + value * value, 0) / n), erleDb: 0 };
    }

    let micEnergy = 0;
    let residualEnergy = 0;
    const weights = this._echoFilter;
    for (let i = 0; i < n; i++) {
      let predicted = 0;
      let norm = ECHO_NLMS_EPSILON;
      const referenceIndex = start + i;
      const taps = Math.min(ECHO_FILTER_TAPS, referenceIndex + 1);
      for (let tap = 0; tap < taps; tap++) {
        const reference = history[referenceIndex - tap];
        predicted += weights[tap] * reference;
        norm += reference * reference;
      }
      const error = mic[i] - predicted;
      residual[i] = error;
      micEnergy += mic[i] * mic[i];
      residualEnergy += error * error;
      if (adapt) {
        const scale = (ECHO_NLMS_STEP * error) / norm;
        for (let tap = 0; tap < taps; tap++) {
          weights[tap] += scale * history[referenceIndex - tap];
        }
      }
    }
    const residualRatio = Math.sqrt(residualEnergy / Math.max(micEnergy, ECHO_NLMS_EPSILON));
    return {
      residual,
      residualRatio,
      residualRms: Math.sqrt(residualEnergy / n),
      erleDb: 10 * Math.log10(Math.max(micEnergy, ECHO_NLMS_EPSILON) / Math.max(residualEnergy, ECHO_NLMS_EPSILON)),
    };
  }

  _maybeEmit() {
    const r = this._ratio;
    const n = this._chunkSamples16k;
    const needIn = Math.ceil(n * r);
    const dec = this._decimated;
    const refDec = this._referenceDecimated;
    while (this._scratch.length >= needIn) {
      // 1. Decimate to 16 kHz floats and accumulate energy for the gate/meter.
      let sumSq = 0;
      if (Math.abs(r - 3) < 1e-6) {
        // 48 kHz -> 16 kHz fast path with boxcar lowpass.
        for (let i = 0; i < n; i++) {
          const idx = i * 3;
          const s = (this._scratch[idx] + this._scratch[idx + 1] + this._scratch[idx + 2]) / 3;
          dec[i] = s;
          refDec[i] = (this._referenceScratch[idx] + this._referenceScratch[idx + 1] + this._referenceScratch[idx + 2]) / 3;
          sumSq += s * s;
        }
      } else {
        // Generic path: linear interpolation. Slower but works at any rate
        // (e.g. some Windows boxes report sampleRate=44100).
        for (let i = 0; i < n; i++) {
          const srcPos = i * r;
          const idx = Math.floor(srcPos);
          const frac = srcPos - idx;
          const a = this._scratch[idx];
          const b = this._scratch[idx + 1] ?? a;
          const s = a + (b - a) * frac;
          dec[i] = s;
          const refA = this._referenceScratch[idx] || 0;
          const refB = this._referenceScratch[idx + 1] ?? refA;
          refDec[i] = refA + (refB - refA) * frac;
          sumSq += s * s;
        }
      }
      const rms = Math.sqrt(sumSq / n);
      let refSumSq = 0;
      for (let i = 0; i < n; i++) refSumSq += refDec[i] * refDec[i];
      const referenceRms = Math.sqrt(refSumSq / n);
      this._appendReference(refDec);
      if (referenceRms >= REFERENCE_ACTIVE_RMS) {
        this._echoTailRemaining = Math.round((ECHO_TAIL_MS / 1000) * TARGET_RATE);
      } else {
        this._echoTailRemaining = Math.max(0, this._echoTailRemaining - n);
      }
      const playbackActive = referenceRms >= REFERENCE_ACTIVE_RMS || this._echoTailRemaining > 0;
      const match = this._echoMatch(dec, sumSq);
      if (playbackActive && match.correlation >= 0.2) {
        if (
          this._echoDelayConfidence === 0
          || Math.abs(match.lagSamples - this._echoDelaySamples) <= Math.round(0.02 * TARGET_RATE)
        ) {
          this._echoDelaySamples = match.lagSamples;
        } else {
          this._echoDelaySamples = Math.round(this._echoDelaySamples * 0.75 + match.lagSamples * 0.25);
        }
        this._echoDelayConfidence = match.correlation;
      }
      const possibleDoubleTalk =
        playbackActive && rms >= HUMAN_ACTIVE_RMS && match.correlation < ECHO_CORRELATION_MIN;
      const cancelled = this._cancelEcho(dec, playbackActive && !possibleDoubleTalk);
      const echoDominant =
        playbackActive
        && match.correlation >= ECHO_CORRELATION_MIN
        && (match.residual <= ECHO_RESIDUAL_MAX || cancelled.residualRatio <= ECHO_RESIDUAL_MAX);
      const humanCandidate =
        playbackActive
        && cancelled.residualRms >= HUMAN_ACTIVE_RMS
        && !echoDominant;

      if (humanCandidate) this._doubleTalkSamples += n;
      else this._doubleTalkSamples = 0;
      const doubleTalk = this._doubleTalkSamples >= Math.round((DOUBLE_TALK_MS / 1000) * TARGET_RATE);
      let framesToSend = [dec];
      let suppressEcho = this._echoMode === "strict" ? playbackActive : false;
      if (this._echoMode === "strict" && playbackActive) {
        framesToSend = [new Float32Array(n)];
      } else if (this._echoMode === "adaptive" && playbackActive) {
        if (echoDominant) {
          this._pendingHumanChunks = [];
          this._doubleTalkSamples = 0;
          framesToSend = [new Float32Array(n)];
          suppressEcho = true;
        } else if (humanCandidate && !doubleTalk) {
          this._pendingHumanChunks.push(cancelled.residual.slice());
          if (this._pendingHumanChunks.length > 4) this._pendingHumanChunks.shift();
          framesToSend = [];
          suppressEcho = true;
        } else if (doubleTalk) {
          framesToSend = [...this._pendingHumanChunks, cancelled.residual];
          this._pendingHumanChunks = [];
          suppressEcho = false;
        } else {
          framesToSend = [new Float32Array(n)];
          suppressEcho = true;
        }
      } else if (this._pendingHumanChunks.length) {
        framesToSend = [...this._pendingHumanChunks, dec];
        this._pendingHumanChunks = [];
      }
      if (suppressEcho) this._suppressedMs += (n / TARGET_RATE) * 1000;

      // 2. Decide the gate target for this chunk, then ramp sample-by-sample.
      let target = 1;
      if (this._gateEnabled) {
        if (rms >= this._thresholdLin) {
          this._holdRemaining = this._holdSamples; // re-arm the hold
        } else if (this._holdRemaining > 0) {
          this._holdRemaining -= n; // coasting through the hold window
        } else {
          target = 0;
        }
      }

      // 3. Pack any confirmation-buffered human onset before this frame.
      let gain = this._gateGain;
      const packedFrames = [];
      for (const frame of framesToSend) {
        const out = new Int16Array(n);
        for (let i = 0; i < n; i++) {
          const coef = target > gain ? this._attackCoef : this._releaseCoef;
          gain = target + (gain - target) * coef;
          const s = frame[i] * gain * (suppressEcho ? 0 : 1);
          const clamped = s < -1 ? -1 : s > 1 ? 1 : s;
          out[i] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
        }
        packedFrames.push(out);
      }
      this._gateGain = gain;

      // Shift the scratch buffer to keep only the trailing unused samples.
      const consumed = Math.floor(n * r);
      this._scratch = this._scratch.slice(consumed);
      this._referenceScratch = this._referenceScratch.slice(consumed);

      // Live input level for the Settings meter (raw RMS, pre-gate).
      this.port.postMessage({ kind: "level", rms });
      this._echoMetricCounter += 1;
      if (this._echoMetricCounter >= 5) {
        this._echoMetricCounter = 0;
        this.port.postMessage({
          kind: "echo_metric",
          mode: this._echoMode,
          nativeAec: this._nativeAec,
          correlation: match.correlation,
          residual: cancelled.residualRatio,
          residualEnergy: cancelled.residualRms * cancelled.residualRms,
          erleDb: cancelled.erleDb,
          lagMs: (this._echoDelaySamples / TARGET_RATE) * 1000,
          suppressedMs: this._suppressedMs,
          suppressing: suppressEcho,
          doubleTalk,
          playbackActive,
        });
      }

      if (this._enabled) {
        for (const out of packedFrames) {
          this.port.postMessage(out.buffer, [out.buffer]);
        }
      }
      // When disabled (mic muted) we silently consume input so the worklet
      // stays alive and the buffer never grows unbounded.
    }
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0 || !input[0]) return true;
    const mono = input[0];
    const reference = inputs[1]?.[0] || null;
    if (mono.length > 0) this._ingest(mono, reference);
    return true;
  }
}

registerProcessor("mic-capture", MicCaptureProcessor);
