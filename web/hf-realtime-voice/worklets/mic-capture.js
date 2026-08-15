// @ts-check

import {
  mixedChannelSample,
  StatefulPolyphaseResampler,
} from "./capture-resampler.js?v=1-stateful-polyphase";

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
const DEFAULT_ECHO_TAIL_MS = 350;

function clamp(value, minimum, maximum, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(minimum, Math.min(maximum, number)) : fallback;
}

function rms(samples, sampleCount = samples.length) {
  if (!sampleCount) return 0;
  let energy = 0;
  for (let index = 0; index < sampleCount; index += 1) {
    energy += samples[index] * samples[index];
  }
  return Math.sqrt(energy / sampleCount);
}

class MicCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const chunkMs = Number(options?.processorOptions?.chunkMs) || DEFAULT_CHUNK_MS;
    this._inputRate = sampleRate;
    this._chunkSamples = Math.round((TARGET_RATE * chunkMs) / 1000);
    this._microphoneResampler = new StatefulPolyphaseResampler(this._inputRate, TARGET_RATE);
    this._referenceResampler = new StatefulPolyphaseResampler(this._inputRate, TARGET_RATE);
    this._microphoneChunk = new Float32Array(this._chunkSamples * 4);
    this._referenceChunk = new Float32Array(this._chunkSamples * 4);
    this._chunkWrite = 0;
    this._referenceInputChunkSamples = Math.max(
      1,
      Math.round((this._inputRate * chunkMs) / 1000),
    );
    this._referenceInputEnergy = 0;
    this._referenceInputCount = 0;
    this._referenceLevels = [];
    this._flushingCapture = false;
    this._flushedSampleCount = 0;
    this._enabled = true;
    this._requestedMode = "native";
    this._echoMode = "native";
    this._nativeAec = false;
    this._echoTailRemaining = 0;
    this._echoTailMs = DEFAULT_ECHO_TAIL_MS;
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
      } else if (data?.kind === "echo_calibration") {
        this._echoTailMs = clamp(data.echoTailMs, 350, 1000, this._echoTailMs);
        this._postStatus();
      } else if (data?.kind === "echo_reset") {
        // Resolve the old route once, then clear its FIR/reference accounting so
        // no level or filter history can be attributed to the replacement route.
        this._flushCapture();
        this._echoTailRemaining = 0;
        this._suppressedMs = 0;
        this._postStatus();
      } else if (data?.kind === "capture_flush") {
        const sampleCount = this._flushCapture();
        this.port.postMessage({ kind: "capture_flushed", sampleCount });
      } else if (data?.kind === "capture_abort") {
        this._abortCapture();
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
      echoTailMs: this._echoTailMs,
      error: this._requestedMode === "adaptive"
        ? "AEC3 module unavailable; Adaptive resolved to Native"
        : "",
    });
  }

  _ingest(incoming, reference) {
    if (!incoming.length) return;
    this._ingestChannels([incoming], reference?.length ? [reference] : [], incoming.length);
  }

  _ingestChannels(microphoneChannels, referenceChannels, frameCount) {
    for (let index = 0; index < frameCount; index += 1) {
      const microphone = mixedChannelSample(microphoneChannels, index);
      const reference = mixedChannelSample(referenceChannels, index);
      this._referenceInputEnergy += reference * reference;
      this._referenceInputCount += 1;
      this._microphoneResampler.writeSample(microphone);
      this._referenceResampler.writeSample(reference);
      while (this._microphoneResampler.canDrain() && this._referenceResampler.canDrain()) {
        this._appendResampled(
          this._microphoneResampler.drainSample(),
          this._referenceResampler.drainSample(),
        );
      }
      if (this._referenceInputCount === this._referenceInputChunkSamples) {
        this._referenceLevels.push(
          Math.sqrt(this._referenceInputEnergy / this._referenceInputCount),
        );
        this._referenceInputEnergy = 0;
        this._referenceInputCount = 0;
      }
    }
  }

  _appendResampled(microphone, reference) {
    this._microphoneChunk[this._chunkWrite] = microphone;
    this._referenceChunk[this._chunkWrite] = reference;
    this._chunkWrite += 1;
    if (this._chunkWrite === this._chunkSamples) this._emitPendingChunk();
  }

  _emitPendingChunk() {
    if (!this._chunkWrite) return 0;
    const sampleCount = this._chunkWrite;
    const microphoneRms = rms(this._microphoneChunk, sampleCount);
    // Echo-tail timing follows the original input block, not the FIR's
    // bounded ring-down, so the anti-alias repair cannot lengthen suppression.
    const referenceRms = this._referenceLevels.length
      ? this._referenceLevels.shift()
      : rms(this._referenceChunk, sampleCount);
    if (referenceRms >= REFERENCE_ACTIVE_RMS) {
      this._echoTailRemaining = Math.round((this._echoTailMs / 1000) * TARGET_RATE);
    } else {
      this._echoTailRemaining = Math.max(0, this._echoTailRemaining - sampleCount);
    }
    const playbackActive =
      referenceRms >= REFERENCE_ACTIVE_RMS || this._echoTailRemaining > 0;
    const suppressing = this._echoMode === "strict" && playbackActive;
    if (suppressing) {
      // Strict fails closed by omitting the frame; it never inserts zero PCM.
      this._suppressedMs += (sampleCount / TARGET_RATE) * 1000;
    } else if (this._enabled) {
      this._emitPcm(this._microphoneChunk, microphoneRms, sampleCount);
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

    this._chunkWrite = 0;
    return sampleCount;
  }

  _flushCapture() {
    this._flushingCapture = true;
    this._flushedSampleCount = 0;
    const finalInputCount = this._microphoneResampler.inputCount;
    if (this._referenceInputCount > 0) {
      this._referenceLevels.push(
        Math.sqrt(this._referenceInputEnergy / this._referenceInputCount),
      );
    }
    while (this._microphoneResampler.sourceIndex < finalInputCount) {
      this._appendResampled(
        this._microphoneResampler.drainFlushedSample(finalInputCount),
        this._referenceResampler.drainFlushedSample(finalInputCount),
      );
    }
    this._microphoneResampler.reset();
    this._referenceResampler.reset();
    this._referenceInputEnergy = 0;
    this._referenceInputCount = 0;
    this._emitPendingChunk();
    this._referenceLevels.length = 0;
    this._flushingCapture = false;
    return this._flushedSampleCount;
  }

  _abortCapture() {
    this._microphoneResampler.reset();
    this._referenceResampler.reset();
    this._chunkWrite = 0;
    this._referenceInputEnergy = 0;
    this._referenceInputCount = 0;
    this._referenceLevels.length = 0;
    this._echoTailRemaining = 0;
    this._suppressedMs = 0;
    this._metricCounter = 0;
    this._holdRemaining = 0;
    this._gateGain = 1;
    this._flushingCapture = false;
    this._flushedSampleCount = 0;
  }

  _emitPcm(microphone, microphoneRms, sampleCount = microphone.length) {
    let target = 1;
    if (this._gateEnabled) {
      if (microphoneRms >= this._thresholdLin) {
        this._holdRemaining = this._holdSamples;
      } else if (this._holdRemaining > 0) {
        this._holdRemaining -= sampleCount;
      } else {
        target = 0;
      }
    }

    const output = new Int16Array(sampleCount);
    let gain = this._gateGain;
    for (let index = 0; index < sampleCount; index += 1) {
      const coefficient = target > gain ? this._attackCoef : this._releaseCoef;
      gain = target + (gain - target) * coefficient;
      const sample = Math.max(-1, Math.min(1, microphone[index] * gain));
      output[index] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
    }
    this._gateGain = gain;
    this.port.postMessage(output.buffer, [output.buffer]);
    if (this._flushingCapture) this._flushedSampleCount += sampleCount;
  }

  process(inputs) {
    const microphoneChannels = inputs[0] || [];
    let frameCount = 0;
    for (let index = 0; index < microphoneChannels.length; index += 1) {
      frameCount = Math.max(frameCount, microphoneChannels[index]?.length || 0);
    }
    if (!frameCount) return true;
    this._ingestChannels(microphoneChannels, inputs[1] || [], frameCount);
    return true;
  }
}

registerProcessor("mic-capture", MicCaptureProcessor);
