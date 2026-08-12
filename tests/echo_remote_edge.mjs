import assert from "node:assert/strict";
import { webcrypto } from "node:crypto";

import {
  EchoRouteCalibration,
  fingerprintEchoRoute,
} from "../web/hf-realtime-voice/echo-route-calibration.js";
import {
  StrictEchoGate,
  classifyStrictEcho,
} from "../web/hf-realtime-voice/worklets/aec3/strict-echo-gate.js";

const calibration = {
  suppressionStrength: 0.65,
  leakageThreshold: 0.65,
  doubleTalkSensitivity: 0.5,
};

const routeA = await fingerprintEchoRoute("microphone-device-a", "remote-output-a", webcrypto.subtle);
const routeARepeat = await fingerprintEchoRoute("microphone-device-a", "remote-output-a", webcrypto.subtle);
const routeB = await fingerprintEchoRoute("microphone-device-a", "remote-output-b", webcrypto.subtle);
assert.match(routeA, /^route_[0-9a-f]{64}$/);
assert.equal(routeA, routeARepeat, "the same resolved route has a stable opaque key");
assert.notEqual(routeA, routeB, "an output-route change rekeys calibration");
assert.equal(routeA.includes("microphone"), false, "fingerprints never expose raw route IDs");
await assert.rejects(
  fingerprintEchoRoute("mic", "out", /** @type {any} */ (null)),
  /unavailable/,
);

for (const centerMs of [80, 150, 300]) {
  const collector = new EchoRouteCalibration();
  const offsets = [-30, -20, -10, 0, 10, 20, 30];
  for (let index = 0; index < 21; index += 1) {
    collector.add({
      lagMs: centerMs + offsets[index % offsets.length],
      timestampMs: index * 100,
      playbackActive: true,
      doubleTalk: false,
    });
  }
  // Ineligible observations do not contaminate the quiet playback cohort.
  collector.add({ lagMs: 999, timestampMs: 2100, playbackActive: false, doubleTalk: false });
  collector.add({ lagMs: 999, timestampMs: 2200, playbackActive: true, doubleTalk: true });
  collector.add({ lagMs: 999, timestampMs: 2300, playbackActive: true, doubleTalk: null });
  collector.add({ lagMs: 999, timestampMs: 2400, playbackActive: true });
  const result = collector.result({ outputLatencyMs: 20 });
  assert.equal(result.accepted, true, `${centerMs} ms remote-route calibration is stable`);
  assert.equal(result.sampleCount, 21);
  assert.equal(result.spanMs, 2000);
  assert.ok(Math.abs(result.medianMs - centerMs) <= 10);
  assert.ok(result.jitterMs <= 60);
  assert.equal(result.delayMs, Math.max(0, result.medianMs - 20));
}

const tooShort = new EchoRouteCalibration();
for (let index = 0; index < 20; index += 1) {
  tooShort.add({ lagMs: 150, timestampMs: index * 50, playbackActive: true, doubleTalk: false });
}
assert.equal(tooShort.result().reason, "insufficient_span");

const negative = new EchoRouteCalibration();
for (let index = 0; index < 21; index += 1) {
  negative.add({
    lagMs: index === 10 ? -1 : 150,
    timestampMs: index * 100,
    playbackActive: true,
    doubleTalk: false,
  });
}
assert.equal(negative.result().reason, "negative_lag");

const unstable = new EchoRouteCalibration();
for (let index = 0; index < 21; index += 1) {
  unstable.add({
    lagMs: index % 2 ? 30 : 330,
    timestampMs: index * 100,
    playbackActive: true,
    doubleTalk: false,
  });
}
assert.equal(unstable.result().reason, "unstable_jitter");

function frame(value) {
  return new Float32Array(480).fill(value);
}

const pureEchoMetrics = {
  doubleTalk: false,
  residualEchoLikelihood: 0.95,
  captureRms: 0.2,
  outputRms: 0.01,
};
const falseDoubleTalkEchoMetrics = { ...pureEchoMetrics, doubleTalk: true };
const nearEndMetrics = {
  doubleTalk: true,
  residualEchoLikelihood: 0.2,
  captureRms: 0.3,
  outputRms: 0.24,
};

assert.deepEqual(
  classifyStrictEcho(true, falseDoubleTalkEchoMetrics, calibration),
  { nearEndEvidence: false, highConfidenceEchoOnly: true },
  "a derived double-talk flag alone cannot release far-end echo",
);
assert.deepEqual(
  classifyStrictEcho(true, nearEndMetrics, calibration),
  { nearEndEvidence: true, highConfidenceEchoOnly: false },
  "independent post-AEC output supplies near-end evidence",
);

const echoGate = new StrictEchoGate(10);
let emitted = 0;
let suppressed = 0;
for (let index = 0; index < 30; index += 1) {
  const evidence = classifyStrictEcho(true, pureEchoMetrics, calibration);
  const decision = echoGate.consume(frame(index), { playbackActive: true, ...evidence });
  emitted += decision.emit.length;
  suppressed += decision.suppressedFrames;
  assert.ok(decision.pendingMs <= 120, "Strict decision buffer never exceeds 120 ms");
}
assert.equal(emitted, 0, "high-confidence echo-only frames never enter semantic PCM");
assert.equal(suppressed, 18, "old echo-only frames are retired after the bounded hold");
const afterEcho = echoGate.consume(frame(31), {
  playbackActive: false,
  nearEndEvidence: false,
  highConfidenceEchoOnly: false,
});
assert.equal(afterEcho.emit.length, 1, "capture resumes when playback and its tail end");
assert.equal(afterEcho.suppressedFrames, 12, "the remaining confirmed echo queue is discarded");

const falseDoubleTalkGate = new StrictEchoGate(10);
for (let index = 0; index < 20; index += 1) {
  const evidence = classifyStrictEcho(true, falseDoubleTalkEchoMetrics, calibration);
  const decision = falseDoubleTalkGate.consume(frame(index), { playbackActive: true, ...evidence });
  assert.equal(decision.emit.length, 0, "false double-talk echo stays suppressed");
}

const bargeInGate = new StrictEchoGate(10);
for (let index = 0; index < 12; index += 1) {
  const evidence = classifyStrictEcho(true, pureEchoMetrics, calibration);
  bargeInGate.consume(frame(-1), { playbackActive: true, ...evidence });
}
let onset = [];
for (let index = 0; index < 6; index += 1) {
  const evidence = classifyStrictEcho(true, nearEndMetrics, calibration);
  const decision = bargeInGate.consume(frame(index + 1), { playbackActive: true, ...evidence });
  onset = onset.concat(decision.emit);
  if (index < 5) assert.equal(decision.nearEndConfirmed, false);
  else assert.equal(decision.nearEndConfirmed, true, "60 ms confirms near-end speech");
}
assert.equal(onset.length, 6, "the complete 60 ms near-end onset is retro-released");
assert.deepEqual(onset.map((values) => values[0]), [1, 2, 3, 4, 5, 6]);
for (let index = 0; index < 25; index += 1) {
  const decision = bargeInGate.consume(frame(10 + index), {
    playbackActive: true,
    nearEndEvidence: false,
    highConfidenceEchoOnly: true,
  });
  assert.equal(decision.emit.length, 1, "the confirmed near-end decision holds for 250 ms");
}

// Exercise the no-module worklet independently: Strict remains fail closed,
// honors the configurable tail, and Native/Adaptive behavior is unchanged.
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
let FallbackProcessor = null;
globalThis.registerProcessor = (_name, implementation) => {
  FallbackProcessor = implementation;
};
await import(`../web/hf-realtime-voice/worklets/mic-capture.js?remote-edge=${Date.now()}`);
assert.ok(FallbackProcessor, "fallback worklet registered");

const input = new Float32Array(1920).fill(0.2);
const reference = new Float32Array(1920).fill(0.2);
const silence = new Float32Array(1920);
const buffers = (processor) => processor.messages.filter((value) => value instanceof ArrayBuffer);

const strictFallback = new FallbackProcessor({ processorOptions: { chunkMs: 40 } });
strictFallback.port.onmessage({ data: { kind: "echo_guard", mode: "strict", nativeAec: true } });
strictFallback.port.onmessage({ data: { kind: "echo_calibration", echoTailMs: 1000 } });
strictFallback._ingest(input, reference);
for (let index = 0; index < 24; index += 1) strictFallback._ingest(input, silence);
assert.equal(buffers(strictFallback).length, 0, "fallback Strict suppresses through a 1000 ms tail");
strictFallback._ingest(input, silence);
assert.equal(buffers(strictFallback).length, 1, "fallback Strict resumes immediately after the tail");

const clampedFallback = new FallbackProcessor({ processorOptions: { chunkMs: 40 } });
clampedFallback.port.onmessage({ data: { kind: "echo_guard", mode: "strict", nativeAec: true } });
clampedFallback.port.onmessage({ data: { kind: "echo_calibration", echoTailMs: 0 } });
const clampedStatus = clampedFallback.messages.filter((value) => value?.kind === "aec3_status").at(-1);
assert.equal(clampedStatus.echoTailMs, 350, "echo tail clamps to the 350 ms safety floor");

for (const mode of ["native", "adaptive"]) {
  const processor = new FallbackProcessor({ processorOptions: { chunkMs: 40 } });
  processor.port.onmessage({ data: { kind: "echo_guard", mode, nativeAec: true } });
  processor._ingest(input, reference);
  assert.equal(buffers(processor).length, 1, `${mode} fallback preserves microphone PCM`);
}

console.log("remote-route echo edge tests passed");
