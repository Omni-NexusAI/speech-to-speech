// @ts-check

import { AEC3_FRAME_MS, AEC3_OUTPUT_RATE, Aec3WasmSession } from "./aec3-abi.js?v=3-opaque-echo-route";
import { StrictEchoGate, classifyStrictEcho } from "./strict-echo-gate.js?v=2-opaque-echo-route";
import {
  mixedChannelSample,
  StatefulPolyphaseResampler,
} from "../capture-resampler.js?v=1-stateful-polyphase";

const DEFAULT_CHUNK_MS = 40;
const GATE_ATTACK_MS = 5;
const GATE_HOLD_MS = 250;
const GATE_RELEASE_MS = 80;
const REFERENCE_ACTIVE_RMS = 0.001;
const DEFAULT_ECHO_TAIL_MS = 350;

class FloatRing {
  constructor(capacity) {
    this.buffer = new Float32Array(capacity);
    this.read = 0;
    this.write = 0;
    this.length = 0;
  }

  push(samples, count = samples?.length || 0) {
    for (let index = 0; index < count; index += 1) {
      if (this.length === this.buffer.length) {
        this.read = (this.read + 1) % this.buffer.length;
        this.length -= 1;
      }
      this.buffer[this.write] = samples && index < samples.length ? samples[index] : 0;
      this.write = (this.write + 1) % this.buffer.length;
      this.length += 1;
    }
  }

  pushOne(sample) {
    if (this.length === this.buffer.length) {
      this.read = (this.read + 1) % this.buffer.length;
      this.length -= 1;
    }
    this.buffer[this.write] = Number.isFinite(sample) ? sample : 0;
    this.write = (this.write + 1) % this.buffer.length;
    this.length += 1;
  }

  readInto(target) {
    if (target.length > this.length) return false;
    for (let index = 0; index < target.length; index += 1) {
      target[index] = this.buffer[this.read];
      this.read = (this.read + 1) % this.buffer.length;
      this.length -= 1;
    }
    return true;
  }

  readPartialInto(target) {
    const count = Math.min(target.length, this.length);
    target.fill(0);
    for (let index = 0; index < count; index += 1) {
      target[index] = this.buffer[this.read];
      this.read = (this.read + 1) % this.buffer.length;
      this.length -= 1;
    }
    return count;
  }

  clear() {
    this.read = 0;
    this.write = 0;
    this.length = 0;
  }
}

function rms(samples) {
  if (!samples.length) return 0;
  let energy = 0;
  for (let index = 0; index < samples.length; index += 1) energy += samples[index] * samples[index];
  return Math.sqrt(energy / samples.length);
}

function clamp(value, minimum, maximum, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(minimum, Math.min(maximum, number)) : fallback;
}

class Aec3CaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const processorOptions = options?.processorOptions || {};
    const chunkMs = clamp(processorOptions.chunkMs, 10, 100, DEFAULT_CHUNK_MS);
    this._inputRate = sampleRate;
    this._frameSamples = Math.floor(this._inputRate / 100);
    this._chunkSamples = Math.round((AEC3_OUTPUT_RATE * chunkMs) / 1000);
    this._micRing = new FloatRing(this._frameSamples * 8);
    this._referenceRing = new FloatRing(this._frameSamples * 8);
    this._micFrame = new Float32Array(this._frameSamples);
    this._referenceFrame = new Float32Array(this._frameSamples);
    this._outputResampler = new StatefulPolyphaseResampler(this._inputRate, AEC3_OUTPUT_RATE);
    this._chunk = new Float32Array(this._chunkSamples);
    this._chunkWrite = 0;
    this._framesPerChunk = Math.max(1, Math.round(chunkMs / AEC3_FRAME_MS));
    this._framesSinceChunk = 0;
    this._flushingCapture = false;
    this._flushedSampleCount = 0;
    this._enabled = true;
    this._requestedMode = "native";
    this._effectiveMode = "native";
    this._nativeAec = false;
    this._moduleReady = false;
    this._moduleError = "AEC3 module was not supplied";
    this._session = null;
    this._manifest = processorOptions.aec3Manifest || {};
    this._processedSamples = 0;
    this._metricFrames = 0;
    this._suppressedMs = 0;
    this._referenceTailSamples = 0;
    this._lastMetrics = null;
    this._lastPlaybackActive = false;
    this._lastSuppressing = false;
    this._lastCandidateMs = 0;
    this._strictGate = new StrictEchoGate(AEC3_FRAME_MS);

    this._calibration = {
      delayMs: 0,
      outputLatencyMs: 0,
      suppressionStrength: 0.65,
      leakageThreshold: 0.65,
      doubleTalkSensitivity: 0.5,
      echoTailMs: DEFAULT_ECHO_TAIL_MS,
    };

    this._gateEnabled = false;
    this._thresholdLin = 0;
    this._gateGain = 1;
    this._holdRemaining = 0;
    this._attackCoef = Math.exp(-1 / ((GATE_ATTACK_MS / 1000) * AEC3_OUTPUT_RATE));
    this._releaseCoef = Math.exp(-1 / ((GATE_RELEASE_MS / 1000) * AEC3_OUTPUT_RATE));
    this._holdSamples = Math.round((GATE_HOLD_MS / 1000) * AEC3_OUTPUT_RATE);

    try {
      if (!(processorOptions.aec3Module instanceof WebAssembly.Module)) {
        throw new Error(this._moduleError);
      }
      const instance = new WebAssembly.Instance(processorOptions.aec3Module, {});
      this._session = new Aec3WasmSession(instance.exports, this._inputRate);
      this._moduleReady = true;
      this._moduleError = "";
    } catch (error) {
      this._moduleReady = false;
      this._moduleError = error instanceof Error ? error.message : String(error);
    }

    this.port.onmessage = (event) => this._onMessage(event.data);
    this._postStatus();
  }

  _onMessage(data) {
    if (!data || typeof data !== "object") return;
    if (data.kind === "enable") {
      this._enabled = !!data.value;
    } else if (data.kind === "gate") {
      this._gateEnabled = !!data.enabled;
      this._thresholdLin = this._gateEnabled ? Math.pow(10, Number(data.thresholdDb) / 20) : 0;
    } else if (data.kind === "echo_guard") {
      const requested = data.mode === "off" ? "native" : data.mode;
      this._requestedMode = ["native", "adaptive", "strict"].includes(requested)
        ? requested
        : "native";
      this._nativeAec = !!data.nativeAec;
      this._resolveMode();
      this._postStatus();
    } else if (data.kind === "echo_calibration") {
      this._calibration = {
        delayMs: clamp(data.delayMs, 0, 500, this._calibration.delayMs),
        outputLatencyMs: clamp(data.outputLatencyMs, 0, 500, this._calibration.outputLatencyMs),
        suppressionStrength: clamp(data.suppressionStrength, 0, 1, this._calibration.suppressionStrength),
        leakageThreshold: clamp(data.leakageThreshold, 0.05, 1, this._calibration.leakageThreshold),
        doubleTalkSensitivity: clamp(data.doubleTalkSensitivity, 0, 1, this._calibration.doubleTalkSensitivity),
        echoTailMs: clamp(data.echoTailMs, 350, 1000, this._calibration.echoTailMs),
      };
      this._postStatus();
    } else if (data.kind === "echo_reset") {
      this._flushCapture();
      this._referenceTailSamples = 0;
      this._processedSamples = 0;
      this._strictGate.reset();
      try {
        this._session?.reset();
      } catch (error) {
        this._disableModule(error);
      }
      this._postStatus();
    } else if (data.kind === "capture_flush") {
      const sampleCount = this._flushCapture();
      this.port.postMessage({ kind: "capture_flushed", sampleCount });
    } else if (data.kind === "capture_abort") {
      this._abortCapture();
    }
  }

  _resolveMode() {
    const previousMode = this._effectiveMode;
    let nextMode;
    if (this._requestedMode === "native") nextMode = "native";
    else if (this._requestedMode === "adaptive") {
      nextMode = this._moduleReady ? "adaptive" : "native";
    } else {
      nextMode = this._moduleReady ? "strict" : "strict-fallback";
    }
    if (previousMode !== nextMode) {
      if (previousMode === "strict") {
        this._flushStrictPending();
        this._flushResamplerTail();
      }
      this._effectiveMode = nextMode;
      this._strictGate.reset();
      this._lastCandidateMs = 0;
    }
  }

  _disableModule(error) {
    try { this._session?.destroy(); } catch (_) {}
    this._session = null;
    this._moduleReady = false;
    this._moduleError = error instanceof Error ? error.message : String(error);
    this._resolveMode();
    this.port.postMessage({ kind: "aec3_error", error: this._moduleError });
  }

  _postStatus() {
    this.port.postMessage({
      kind: "aec3_status",
      available: this._moduleReady,
      requestedMode: this._requestedMode,
      effectiveMode: this._effectiveMode,
      nativeAec: this._nativeAec,
      referenceWired: true,
      abiVersion: this._manifest.abiVersion || null,
      engine: this._manifest.engine || null,
      sourceRevision: this._manifest.sourceRevision || null,
      frameMs: AEC3_FRAME_MS,
      frameSamples: this._frameSamples,
      delayMs: this._calibration.delayMs,
      outputLatencyMs: this._calibration.outputLatencyMs,
      echoTailMs: this._calibration.echoTailMs,
      error: this._moduleError,
    });
  }

  _playbackState(reference) {
    const referenceRms = rms(reference);
    if (referenceRms >= REFERENCE_ACTIVE_RMS) {
      this._referenceTailSamples = Math.round(
        (this._calibration.echoTailMs / 1000) * this._inputRate,
      );
    } else {
      this._referenceTailSamples = Math.max(0, this._referenceTailSamples - reference.length);
    }
    return {
      referenceRms,
      playbackActive: referenceRms >= REFERENCE_ACTIVE_RMS || this._referenceTailSamples > 0,
    };
  }

  _appendOutput(frame, levelRms, maximumCenterExclusive = Number.POSITIVE_INFINITY) {
    for (let index = 0; index < frame.length; index += 1) {
      this._outputResampler.writeSample(frame[index]);
      while (this._outputResampler.canDrain(maximumCenterExclusive)) {
        this._appendResampledSample(this._outputResampler.drainSample(), levelRms);
      }
    }
  }

  _appendResampledSample(sample, levelRms) {
    this._chunk[this._chunkWrite] = sample;
    this._chunkWrite += 1;
    if (this._chunkWrite === this._chunk.length) {
      // One resampled transport chunk can span four different 10 ms AEC
      // frames. Gate it from the complete chunk, not only its final frame.
      this._emitChunk(rms(this._chunk));
      this._chunkWrite = 0;
    }
  }

  _flushResamplerTail(maximumCenterExclusive = this._outputResampler.inputCount) {
    while (this._outputResampler.sourceIndex < maximumCenterExclusive) {
      this._appendResampledSample(
        this._outputResampler.drainFlushedSample(maximumCenterExclusive),
        0,
      );
    }
    this._outputResampler.reset();
  }

  _flushStrictPending(maximumCenterExclusive = Number.POSITIVE_INFINITY) {
    const decision = this._strictGate.flush();
    for (const retainedFrame of decision.emit) {
      this._appendOutput(retainedFrame, rms(retainedFrame), maximumCenterExclusive);
    }
    this._suppressedMs += decision.suppressedFrames * AEC3_FRAME_MS;
    this._lastCandidateMs = 0;
  }

  _flushOutputChunk() {
    if (this._chunkWrite <= 0) return;
    const sampleCount = this._chunkWrite;
    this._emitChunk(rms(this._chunk.subarray(0, sampleCount)), sampleCount);
    this._chunkWrite = 0;
  }

  _emitChunk(levelRms, sampleCount = this._chunk.length) {
    let target = 1;
    if (this._gateEnabled) {
      if (levelRms >= this._thresholdLin) this._holdRemaining = this._holdSamples;
      else if (this._holdRemaining > 0) this._holdRemaining -= sampleCount;
      else target = 0;
    }
    const output = new Int16Array(sampleCount);
    let gain = this._gateGain;
    for (let index = 0; index < sampleCount; index += 1) {
      const coefficient = target > gain ? this._attackCoef : this._releaseCoef;
      gain = target + (gain - target) * coefficient;
      const sample = Math.max(-1, Math.min(1, this._chunk[index] * gain));
      output[index] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
    }
    this._gateGain = gain;
    if (this._enabled) {
      this.port.postMessage(output.buffer, [output.buffer]);
      if (this._flushingCapture) this._flushedSampleCount += sampleCount;
    }
  }

  _postMetric(captureRms, referenceRms, playbackActive, suppressing, metrics) {
    this._metricFrames += 1;
    if (this._metricFrames < 10) return;
    this._metricFrames = 0;
    const residual = Number.isFinite(metrics?.residualEchoLikelihood)
      ? metrics.residualEchoLikelihood
      : null;
    this.port.postMessage({
      kind: "echo_metric",
      mode: this._effectiveMode,
      requestedMode: this._requestedMode,
      nativeAec: this._nativeAec,
      moduleAvailable: this._moduleReady,
      referenceWired: true,
      referenceRms,
      captureRms,
      residual,
      residualEnergy: Number.isFinite(metrics?.outputRms) ? metrics.outputRms ** 2 : null,
      erleDb: metrics?.erleDb ?? null,
      lagMs: metrics?.delayMs ?? null,
      configuredDelayMs: this._calibration.delayMs + this._calibration.outputLatencyMs,
      modelReady: this._moduleReady,
      predictionConfidence: residual === null ? null : Math.max(0, Math.min(1, 1 - residual)),
      candidateMs: this._lastCandidateMs,
      suppressedMs: this._suppressedMs,
      suppressing,
      doubleTalk: metrics?.doubleTalk ?? null,
      doubleTalkSource: metrics?.doubleTalkSource || null,
      playbackActive,
    });
  }

  _processFrame(maximumCenterExclusive = Number.POSITIVE_INFINITY) {
    const capture = this._micFrame;
    const reference = this._referenceFrame;
    const captureRms = rms(capture);
    const { referenceRms, playbackActive } = this._playbackState(reference);
    const timestampMs = (this._processedSamples / this._inputRate) * 1000;
    this._processedSamples += this._frameSamples;
    let output = capture;
    let metrics = null;

    if (this._effectiveMode === "adaptive" || this._effectiveMode === "strict") {
      try {
        const result = this._session.process(reference, capture, {
          delayMs: this._calibration.delayMs + this._calibration.outputLatencyMs,
          timestampMs,
          strict: this._effectiveMode === "strict",
        });
        output = result.output;
        metrics = result.metrics;
      } catch (error) {
        this._disableModule(error);
        output = capture;
      }
    }

    let suppressing = false;
    if (this._effectiveMode === "strict") {
      const evidence = classifyStrictEcho(playbackActive, metrics, this._calibration);
      const decision = this._strictGate.consume(output, {
        playbackActive,
        ...evidence,
      });
      for (const retainedFrame of decision.emit) {
        this._appendOutput(retainedFrame, rms(retainedFrame), maximumCenterExclusive);
      }
      this._suppressedMs += decision.suppressedFrames * AEC3_FRAME_MS;
      suppressing = decision.suppressing;
      this._lastCandidateMs = decision.pendingMs;
    } else if (this._effectiveMode === "strict-fallback" && playbackActive) {
      // A missing or failed AEC3 module has no trustworthy near-end evidence.
      // Strict therefore remains fail closed for referenced playback and tail.
      this._suppressedMs += AEC3_FRAME_MS;
      suppressing = true;
      this._lastCandidateMs = 0;
    } else {
      this._appendOutput(output, rms(output), maximumCenterExclusive);
      this._lastCandidateMs = 0;
    }
    this._lastMetrics = metrics;
    this._lastPlaybackActive = playbackActive;
    this._lastSuppressing = suppressing;
    this.port.postMessage({ kind: "level", rms: captureRms });
    this._postMetric(captureRms, referenceRms, playbackActive, suppressing, metrics);
    this._framesSinceChunk += 1;
    if (this._framesSinceChunk >= this._framesPerChunk) {
      this._framesSinceChunk = 0;
      this._flushOutputChunk();
    }
  }

  _flushCapture() {
    this._flushingCapture = true;
    this._flushedSampleCount = 0;
    this._flushStrictPending();
    if (this._micRing.length > 0) {
      const validSamples = this._micRing.length;
      this._micRing.readPartialInto(this._micFrame);
      this._referenceRing.readPartialInto(this._referenceFrame);
      const before = this._outputResampler.inputCount;
      const maximumCenterExclusive = before + validSamples;
      this._processFrame(maximumCenterExclusive);
      // Strict can retain the final partial frame while it waits for near-end
      // confirmation. Resolve that frame before draining the endpoint, but keep
      // the validity bound so zero padding cannot extend the captured duration.
      this._flushStrictPending(maximumCenterExclusive);
      this._flushResamplerTail(
        Math.min(maximumCenterExclusive, this._outputResampler.inputCount),
      );
    } else {
      this._flushResamplerTail();
    }
    this._flushOutputChunk();
    this._micRing.clear();
    this._referenceRing.clear();
    this._framesSinceChunk = 0;
    this._flushingCapture = false;
    return this._flushedSampleCount;
  }

  _abortCapture() {
    this._micRing.clear();
    this._referenceRing.clear();
    this._outputResampler.reset();
    this._chunkWrite = 0;
    this._framesSinceChunk = 0;
    this._strictGate.reset();
    this._referenceTailSamples = 0;
    this._processedSamples = 0;
    this._lastCandidateMs = 0;
    this._lastMetrics = null;
    this._lastPlaybackActive = false;
    this._lastSuppressing = false;
    this._flushingCapture = false;
    this._flushedSampleCount = 0;
  }

  process(inputs) {
    const microphoneChannels = inputs[0] || [];
    const referenceChannels = inputs[1] || [];
    let frameCount = 0;
    for (let channel = 0; channel < microphoneChannels.length; channel += 1) {
      frameCount = Math.max(frameCount, microphoneChannels[channel]?.length || 0);
    }
    if (!frameCount) return true;
    for (let index = 0; index < frameCount; index += 1) {
      this._micRing.pushOne(mixedChannelSample(microphoneChannels, index));
      this._referenceRing.pushOne(mixedChannelSample(referenceChannels, index));
      if (
        this._micRing.length >= this._frameSamples
        && this._referenceRing.length >= this._frameSamples
      ) {
        this._micRing.readInto(this._micFrame);
        this._referenceRing.readInto(this._referenceFrame);
        this._processFrame();
      }
    }
    return true;
  }
}

registerProcessor("aec3-capture", Aec3CaptureProcessor);
