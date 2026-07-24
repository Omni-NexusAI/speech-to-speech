import assert from "node:assert/strict";

globalThis.sampleRate = 48000;
globalThis.AudioWorkletProcessor = class {
  constructor() {
    this.messages = [];
    this.port = {
      onmessage: null,
      postMessage: (value) => this.messages.push(value),
    };
  }
};

let Processor = null;
globalThis.registerProcessor = (_name, implementation) => {
  Processor = implementation;
};
await import(new URL("../web/hf-realtime-voice/worklets/mic-capture.js", import.meta.url));
assert.ok(Processor, "mic-capture worklet registered");

const INPUT_SAMPLES = 1920;

function tone(frequency, phase = 0) {
  const values = new Float32Array(INPUT_SAMPLES);
  for (let i = 0; i < values.length; i++) {
    values[i] = 0.2 * Math.sin((2 * Math.PI * frequency * i) / sampleRate + phase);
  }
  return values;
}

function configure(mode) {
  const processor = new Processor({ processorOptions: { chunkMs: 40 } });
  processor.port.onmessage({ data: { kind: "echo_guard", mode, nativeAec: true } });
  return processor;
}

function outputBuffers(processor) {
  return processor.messages.filter((value) => value instanceof ArrayBuffer).map((value) => new Int16Array(value));
}

function hasSignal(buffer) {
  return !!buffer && buffer.some((sample) => Math.abs(sample) > 8);
}

function deterministicNoise(length) {
  const values = new Float32Array(length);
  let state = 0x12345678;
  for (let i = 0; i < length; i++) {
    state = (1664525 * state + 1013904223) >>> 0;
    values[i] = (((state / 0xffffffff) * 2) - 1) * 0.15;
  }
  return values;
}

function roomEcho(reference, delaySamples) {
  const values = new Float32Array(reference.length);
  for (let i = 0; i < values.length; i++) {
    const direct = i - delaySamples;
    const reflectionA = direct - 240;
    const reflectionB = direct - 600;
    values[i] =
      (direct >= 0 ? reference[direct] * 0.55 : 0)
      + (reflectionA >= 0 ? reference[reflectionA] * 0.23 : 0)
      + (reflectionB >= 0 ? reference[reflectionB] * 0.11 : 0);
  }
  return values;
}

const echo = tone(440);

const adaptive = configure("adaptive");
adaptive._ingest(echo, echo);
assert.equal(hasSignal(outputBuffers(adaptive).at(-1)), false, "adaptive mode suppresses correlated playback");

const off = configure("off");
off._ingest(echo, echo);
assert.equal(hasSignal(outputBuffers(off).at(-1)), true, "off mode preserves microphone audio");

const strict = configure("strict");
strict._ingest(echo, echo);
assert.equal(hasSignal(outputBuffers(strict).at(-1)), false, "strict mode suspends upload during playback");

const doubleTalk = configure("adaptive");
const playback = tone(440);
const human = tone(997, 0.37);
for (let i = 0; i < 5; i++) doubleTalk._ingest(human, playback);
assert.equal(hasSignal(outputBuffers(doubleTalk).at(-1)), true, "adaptive mode releases sustained uncorrelated speech");

const metric = doubleTalk.messages.find((value) => value?.kind === "echo_metric");
assert.ok(metric, "echo diagnostics are emitted");
assert.equal(metric.nativeAec, true);
assert.equal(metric.doubleTalk, true);

const delayed = configure("adaptive");
const longReference = deterministicNoise(INPUT_SAMPLES * 9);
const delayedEcho = roomEcho(longReference, Math.round(sampleRate * 0.08));
for (let offset = 0; offset < longReference.length; offset += INPUT_SAMPLES) {
  delayed._ingest(
    delayedEcho.subarray(offset, offset + INPUT_SAMPLES),
    longReference.subarray(offset, offset + INPUT_SAMPLES),
  );
}
const delayedMetric = delayed.messages.filter((value) => value?.kind === "echo_metric").at(-1);
assert.ok(delayedMetric, "delayed echo diagnostics are emitted");
assert.ok(delayedMetric.lagMs >= 40 && delayedMetric.lagMs <= 120, "adaptive mode tracks room delay");
assert.equal(hasSignal(outputBuffers(delayed).at(-1)), false, "adaptive mode suppresses delayed reverberant echo");

delayed.port.onmessage({ data: { kind: "echo_reset" } });
assert.equal(delayed._referenceHistory.length, 0, "echo reset clears playback history");

console.log("echo guard tests passed");
