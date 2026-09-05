// @ts-check

/**
 * Native-browser-AEC fallback capture worklet.
 *
 * This processor resamples microphone capture to 16 kHz PCM16. It deliberately
 * does not implement a custom echo predictor: requested Adaptive resolves to
 * Native here. The separate aec3-capture processor owns real reference-aware
 * AEC3 when its authenticated WASM module is available.
 */

const TARGET_RATE = 16000;
const DEFAULT_CHUNK_MS = 40;
const GATE_ATTACK_MS = 5;
const GATE_HOLD_MS = 250;
const GATE_RELEASE_MS = 80;
const REFERENCE_ACTIVE_RMS = 0.001;
const ECHO_TAIL_MS = 350;

function rms(samples) {
  if (!samples.length) return 0;
  let energy = 0;
  for (let index = 0; index < samples.length; index += 1) {
    energy += samples[index] * samples[index];
  }
  return Math.sqrt(energy / samples.length);
}

class MicCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const chunkMs = Number(options?.processorOptions?.chunkMs) || DEFAULT_CHUNK_MS;
    this._inputRate = sampleRate;
    this._ratio = this._inputRate / TARGET_RATE;
    this._chunkSamples = Math.round((TARGET_RATE * chunkMs) / 1000);
    this._scratch = new Float32Array(0);
    this._referenceScratch = new Float32Array(0);
    this._enabled = true;
    this._requestedMode = "native";
    this._echoMode = "native";
    this._nativeAec = false;
    this._echoTailRemaining = 0;
    this._suppressedMs = 0;
    this._metricCounter = 0;

    this._gateEnabled = false;
    this._thresholdLin = 0;
    this._gateGain = 1;
    this._holdRemaining = 0;
    this._attackCoef = Math.exp(-1 / ((GATE_ATTACK_MS / 1000) * TARGET_RATE));
    this._releaseCoef = Math.exp(-1 / ((GATE_RELEASE_MS / 1000) * TARGET_RATE));
    this._holdSamples = Math.round((GATE_HOLD_MS / 1000) * TARGET_RATE);

    this.port.onmessage = (event) => {
      const data = event.data;
      if (data?.kind === "enable") {
        this._enabled = !!data.value;
      } else if (data?.kind === "gate") {
        this._gateEnabled = !!data.enabled;
        this._thresholdLin = this._gateEnabled
          ? Math.pow(10, Number(data.thresholdDb) / 20)
          : 0;
      } else if (data?.kind === "echo_guard") {
        this._requestedMode = ["native", "adaptive", "strict"].includes(data.mode)
          ? data.mode
          : "native";
        const requested = this._requestedMode;
        this._echoMode = requested === "strict" ? "strict" : "native";
        this._nativeAec = !!data.nativeAec;
        this._postStatus();
      } else if (data?.kind === "echo_reset") {
        this._echoTailRemaining = 0;
        this._suppressedMs = 0;
        this._postStatus();
      }
    };

    this._postStatus();
  }

  _postStatus() {
    this.port.postMessage({
      kind: "aec3_status",
      available: false,
      requestedMode: this._requestedMode,
      effectiveMode: this._echoMode,
      nativeAec: this._nativeAec,
      referenceWired: true,
      error: this._requestedMode === "adaptive"
        ? "AEC3 module unavailable; Adaptive resolved to Native"
        : "",
    });
  }

  _ingest(incoming, reference) {
    if (!incoming.length) return;
    const next = new Float32Array(this._scratch.length + incoming.length);
    next.set(this._scratch);
    next.set(incoming, this._scratch.length);
    this._scratch = next;

    const referenceNext = new Float32Array(this._referenceScratch.length + incoming.length);
    referenceNext.set(this._referenceScratch);
    if (reference) {
      referenceNext.set(reference.subarray(0, incoming.length), this._referenceScratch.length);
    }
    this._referenceScratch = referenceNext;
    this._maybeEmit();
  }

  _resample(source, output) {
    if (Math.abs(this._ratio - 3) < 1e-6) {
      for (let index = 0; index < output.length; index += 1) {
        const offset = index * 3;
        output[index] = (source[offset] + source[offset + 1] + source[offset + 2]) / 3;
      }
      return;
    }
    for (let index = 0; index < output.length; index += 1) {
      const sourcePosition = index * this._ratio;
      const sourceIndex = Math.floor(sourcePosition);
      const fraction = sourcePosition - sourceIndex;
      const a = source[sourceIndex] || 0;
      const b = source[Math.min(source.length - 1, sourceIndex + 1)] || a;
      output[index] = a + (b - a) * fraction;
    }
  }

  _maybeEmit() {
    const needed = Math.ceil(this._chunkSamples * this._ratio);
    while (this._scratch.length >= needed) {
      const microphone = new Float32Array(this._chunkSamples);
      const reference = new Float32Array(this._chunkSamples);
      this._resample(this._scratch.subarray(0, needed), microphone);
      this._resample(this._referenceScratch.subarray(0, needed), reference);

      const microphoneRms = rms(microphone);
      const referenceRms = rms(reference);
      if (referenceRms >= REFERENCE_ACTIVE_RMS) {
        this._echoTailRemaining = Math.round((ECHO_TAIL_MS / 1000) * TARGET_RATE);
      } else {
        this._echoTailRemaining = Math.max(0, this._echoTailRemaining - this._chunkSamples);
      }
      const playbackActive =
        referenceRms >= REFERENCE_ACTIVE_RMS || this._echoTailRemaining > 0;
      const suppressing = this._echoMode === "strict" && playbackActive;
      if (suppressing) {
        // Strict fails closed by omitting the frame; it never inserts zero PCM.
        this._suppressedMs += (this._chunkSamples / TARGET_RATE) * 1000;
      } else if (this._enabled) {
        this._emitPcm(microphone, microphoneRms);
      }

      this.port.postMessage({ kind: "level", rms: microphoneRms });
      this._metricCounter += 1;
      if (this._metricCounter >= 5) {
        this._metricCounter = 0;
        this.port.postMessage({
          kind: "echo_metric",
          mode: this._echoMode,
          requestedMode: this._requestedMode,
          nativeAec: this._nativeAec,
          moduleAvailable: false,
          referenceWired: true,
          referenceRms,
          captureRms: microphoneRms,
          correlation: 0,
          residual: null,
          residualEnergy: null,
          erleDb: null,
          lagMs: null,
          modelReady: false,
          predictionConfidence: null,
          candidateMs: 0,
          suppressedMs: this._suppressedMs,
          suppressing,
          doubleTalk: null,
          playbackActive,
        });
      }

      const consumed = Math.floor(this._chunkSamples * this._ratio);
      this._scratch = this._scratch.slice(consumed);
      this._referenceScratch = this._referenceScratch.slice(consumed);
    }
  }

  _emitPcm(microphone, microphoneRms) {
    let target = 1;
    if (this._gateEnabled) {
      if (microphoneRms >= this._thresholdLin) {
        this._holdRemaining = this._holdSamples;
      } else if (this._holdRemaining > 0) {
        this._holdRemaining -= microphone.length;
      } else {
        target = 0;
      }
    }

    const output = new Int16Array(microphone.length);
    let gain = this._gateGain;
    for (let index = 0; index < microphone.length; index += 1) {
      const coefficient = target > gain ? this._attackCoef : this._releaseCoef;
      gain = target + (gain - target) * coefficient;
      const sample = Math.max(-1, Math.min(1, microphone[index] * gain));
      output[index] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
    }
    this._gateGain = gain;
    this.port.postMessage(output.buffer, [output.buffer]);
  }

  process(inputs) {
    const microphone = inputs[0]?.[0];
    if (!microphone || !microphone.length) return true;
    const reference = inputs[1]?.[0] || null;
    this._ingest(microphone, reference);
    return true;
  }
}

registerProcessor("mic-capture", MicCaptureProcessor);
