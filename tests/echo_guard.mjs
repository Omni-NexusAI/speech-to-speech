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
  return buffer.some((sample) => Math.abs(sample) > 8);
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

console.log("echo guard tests passed");
