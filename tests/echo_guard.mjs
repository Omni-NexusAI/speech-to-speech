import assert from "node:assert/strict";
import fs from "node:fs";

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
const moduleUrl = new URL(
  "../web/hf-realtime-voice/worklets/mic-capture.js",
  import.meta.url,
);
await import(moduleUrl);
assert.ok(Processor, "mic-capture fallback registered");

const INPUT_SAMPLES = 1920;

function tone(frequency, amplitude = 0.2) {
  const values = new Float32Array(INPUT_SAMPLES);
  for (let index = 0; index < values.length; index += 1) {
    values[index] = amplitude * Math.sin((2 * Math.PI * frequency * index) / sampleRate);
  }
  return values;
}

function configure(mode) {
  const processor = new Processor({ processorOptions: { chunkMs: 40 } });
  processor.port.onmessage({ data: { kind: "echo_guard", mode, nativeAec: true } });
  return processor;
}

function audioBuffers(processor) {
  return processor.messages.filter((value) => value instanceof ArrayBuffer);
}

function completeFirstCaptureChunk(processor, microphone, playback) {
  processor._ingest(microphone, playback);
  // The 121-tap FIR needs 58 more 48 kHz frames before the last center of the
  // first exact 640-sample output chunk has its 60-sample lookahead.
  processor._ingest(new Float32Array(58).fill(0.2), new Float32Array(58));
}

const microphone = tone(997);
const playback = tone(440);

const native = configure("native");
completeFirstCaptureChunk(native, microphone, playback);
assert.equal(audioBuffers(native).length, 1, "Native emits browser-captured microphone PCM");
assert.ok(
  new Int16Array(audioBuffers(native)[0]).some((sample) => Math.abs(sample) > 8),
  "Native PCM preserves microphone signal",
);

const adaptiveFallback = configure("adaptive");
completeFirstCaptureChunk(adaptiveFallback, microphone, playback);
assert.equal(
  audioBuffers(adaptiveFallback).length,
  1,
  "Adaptive falls back to Native instead of a custom residual processor",
);
const status = adaptiveFallback.messages.filter((value) => value?.kind === "aec3_status").at(-1);
assert.equal(status.available, false);
assert.equal(status.requestedMode, "adaptive");
assert.equal(status.effectiveMode, "native");

const strict = configure("strict");
completeFirstCaptureChunk(strict, microphone, playback);
assert.equal(audioBuffers(strict).length, 0, "Strict omits capture during referenced playback");
for (let frame = 0; frame < 9; frame += 1) {
  strict._ingest(microphone, new Float32Array(INPUT_SAMPLES));
}
assert.equal(audioBuffers(strict).length, 1, "Strict resumes after the configured echo tail");
assert.ok(
  new Int16Array(audioBuffers(strict)[0]).some((sample) => Math.abs(sample) > 8),
  "Strict never substitutes a zero-filled residual frame",
);

const source = fs.readFileSync(moduleUrl, "utf8");
for (const forbidden of ["ECHO_NLMS_STEP", "_echoPrediction", "_echoFilter"]) {
  assert.equal(source.includes(forbidden), false, `legacy predictor symbol remains: ${forbidden}`);
}

console.log("native fallback echo guard tests passed");
