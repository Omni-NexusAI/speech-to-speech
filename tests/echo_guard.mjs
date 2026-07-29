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
const OUTPUT_SAMPLES = 640;

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

function deterministicNoise(length, seed = 0x12345678) {
  const values = new Float32Array(length);
  let state = seed;
  for (let i = 0; i < length; i++) {
    state = (1664525 * state + 1013904223) >>> 0;
    values[i] = (((state / 0xffffffff) * 2) - 1) * 0.15;
  }
  return values;
}

function roomEcho(reference, delaySamples, gain = 0.55) {
  const values = new Float32Array(reference.length);
  for (let i = 0; i < values.length; i++) {
    const direct = i - delaySamples;
    const reflectionA = direct - 240;
    const reflectionB = direct - 600;
    values[i] =
      (direct >= 0 ? reference[direct] * gain : 0)
      + (reflectionA >= 0 ? reference[reflectionA] * 0.23 : 0)
      + (reflectionB >= 0 ? reference[reflectionB] * 0.11 : 0);
  }
  return values;
}

function directEcho(reference, delaySamples, gain) {
  const values = new Float32Array(reference.length);
  for (let i = delaySamples; i < values.length; i++) {
    values[i] = reference[i - delaySamples] * gain;
  }
  return values;
}

function add(a, b) {
  const values = new Float32Array(a.length);
  for (let i = 0; i < values.length; i++) values[i] = a[i] + b[i];
  return values;
}

function scale(values, gain) {
  const result = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) result[i] = values[i] * gain;
  return result;
}

function clip(values, limit = 0.12) {
  const clipped = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) {
    clipped[i] = Math.max(-limit, Math.min(limit, values[i]));
  }
  return clipped;
}

function decimate48k(values) {
  const result = new Float32Array(OUTPUT_SAMPLES);
  for (let i = 0; i < result.length; i++) {
    const offset = i * 3;
    result[i] = (values[offset] + values[offset + 1] + values[offset + 2]) / 3;
  }
  return result;
}

function pcmCorrelation(actual, expected) {
  let dot = 0;
  let actualEnergy = 0;
  let expectedEnergy = 0;
  for (let i = 0; i < actual.length; i++) {
    const a = actual[i] / 32768;
    const b = expected[i];
    dot += a * b;
    actualEnergy += a * a;
    expectedEnergy += b * b;
  }
  return dot / Math.sqrt(actualEnergy * expectedEnergy);
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

const isolatedHeadset = configure("adaptive");
const isolatedReference = deterministicNoise(INPUT_SAMPLES * 16, 0x2468ace0);
const isolatedFrames = [];
for (let frame = 0; frame < 12; frame++) {
  const offset = frame * INPUT_SAMPLES;
  // This simulates a headset/native-AEC capture path: the assistant's exact
  // playback reference is active, but the microphone receives only the user.
  const human = scale(tone(733, frame * 0.13), 0.012);
  isolatedFrames.push(decimate48k(human));
  isolatedHeadset._ingest(human, isolatedReference.subarray(offset, offset + INPUT_SAMPLES));
  if (frame < 11) {
    assert.equal(outputBuffers(isolatedHeadset).length, 0, "isolated speech stays local before 450 ms confirmation");
  }
}
assert.equal(isolatedHeadset._echoModelReady, false, "an isolated headset does not need an echo model");
assert.equal(isolatedHeadset._acousticState, "uncoupled", "sustained independent speech resolves uncoupled capture");
const isolatedReleased = outputBuffers(isolatedHeadset);
assert.equal(isolatedReleased.length, 12, "isolated headset speech releases after 450 ms");
assert.ok(
  pcmCorrelation(isolatedReleased[0], isolatedFrames[0]) > 0.995,
  "isolated headset release is untouched original mic PCM",
);

const aecClean = configure("adaptive");
const aecReference = deterministicNoise(INPUT_SAMPLES * 14, 0x13579bdf);
for (let frame = 0; frame < 8; frame++) {
  const offset = frame * INPUT_SAMPLES;
  aecClean._ingest(new Float32Array(INPUT_SAMPLES), aecReference.subarray(offset, offset + INPUT_SAMPLES));
}
assert.equal(aecClean._acousticState, "uncoupled", "clean native AEC capture is recognized without predictor warmup");
assert.equal(outputBuffers(aecClean).length, 0, "AEC-clean silence produces no uploaded PCM");

const delayed = configure("adaptive");
const longReference = deterministicNoise(INPUT_SAMPLES * 40);
const delayedEcho = roomEcho(longReference, Math.round(sampleRate * 0.5));
for (let offset = 0; offset < longReference.length; offset += INPUT_SAMPLES) {
  delayed._ingest(
    delayedEcho.subarray(offset, offset + INPUT_SAMPLES),
    longReference.subarray(offset, offset + INPUT_SAMPLES),
  );
}
const delayedMetric = delayed.messages.filter((value) => value?.kind === "echo_metric").at(-1);
assert.ok(delayedMetric, "delayed echo diagnostics are emitted");
assert.equal(delayedMetric.modelReady, true, "adaptive model becomes ready after echo-only warmup");
assert.ok(delayedMetric.lagMs >= 430 && delayedMetric.lagMs <= 500, "adaptive mode tracks Bluetooth-scale delay");
assert.equal(hasSignal(outputBuffers(delayed).at(-1)), false, "adaptive mode suppresses delayed reverberant echo");

const distorted = configure("adaptive");
const trainingReference = deterministicNoise(INPUT_SAMPLES * 35);
const trainingEcho = roomEcho(trainingReference, Math.round(sampleRate * 0.12));
for (let offset = 0; offset < trainingReference.length; offset += INPUT_SAMPLES) {
  distorted._ingest(
    trainingEcho.subarray(offset, offset + INPUT_SAMPLES),
    trainingReference.subarray(offset, offset + INPUT_SAMPLES),
  );
}
const distortedReference = deterministicNoise(INPUT_SAMPLES * 8);
const distortedEcho = clip(roomEcho(distortedReference, Math.round(sampleRate * 0.12), 0.8));
for (let offset = 0; offset < distortedReference.length; offset += INPUT_SAMPLES) {
  distorted._ingest(
    distortedEcho.subarray(offset, offset + INPUT_SAMPLES),
    distortedReference.subarray(offset, offset + INPUT_SAMPLES),
  );
}
assert.equal(outputBuffers(distorted).length, 0, "adaptive mode suppresses clipped echo without uploading residuals");

const persistent = configure("adaptive");
const firstResponseFrames = 24;
const gapFrames = 8;
const secondResponseFrames = 12;
const twoResponseReference = new Float32Array(
  INPUT_SAMPLES * (firstResponseFrames + gapFrames + secondResponseFrames),
);
twoResponseReference.set(deterministicNoise(INPUT_SAMPLES * firstResponseFrames), 0);
twoResponseReference.set(
  deterministicNoise(INPUT_SAMPLES * secondResponseFrames, 0x87654321),
  INPUT_SAMPLES * (firstResponseFrames + gapFrames),
);
const twoResponseEcho = roomEcho(twoResponseReference, Math.round(sampleRate * 0.12), 0.42);
const backgroundNoise = scale(deterministicNoise(twoResponseReference.length, 0xdeadbeef), 0.02);
const noisyTwoResponseEcho = add(twoResponseEcho, backgroundNoise);
for (let frame = 0; frame < firstResponseFrames; frame++) {
  const offset = frame * INPUT_SAMPLES;
  persistent._ingest(
    noisyTwoResponseEcho.subarray(offset, offset + INPUT_SAMPLES),
    twoResponseReference.subarray(offset, offset + INPUT_SAMPLES),
  );
}
assert.equal(persistent._echoModelReady, true, "noisy attenuated echo trains the classifier");
assert.equal(outputBuffers(persistent).length, 0, "noisy attenuated echo uploads no PCM");
const learnedWeightEnergy = persistent._echoFilter.reduce((sum, value) => sum + value * value, 0);
assert.ok(learnedWeightEnergy > 0, "adaptive predictor learns an echo path");

for (let frame = firstResponseFrames; frame < firstResponseFrames + gapFrames; frame++) {
  const offset = frame * INPUT_SAMPLES;
  persistent._ingest(
    noisyTwoResponseEcho.subarray(offset, offset + INPUT_SAMPLES),
    twoResponseReference.subarray(offset, offset + INPUT_SAMPLES),
  );
}
assert.equal(outputBuffers(persistent).length, 0, "playback gaps and the 350 ms tail remain suppressed");
assert.equal(persistent._echoModelReady, true, "playback gaps do not discard the learned path");

for (
  let frame = firstResponseFrames + gapFrames;
  frame < firstResponseFrames + gapFrames + secondResponseFrames;
  frame++
) {
  const offset = frame * INPUT_SAMPLES;
  persistent._ingest(
    noisyTwoResponseEcho.subarray(offset, offset + INPUT_SAMPLES),
    twoResponseReference.subarray(offset, offset + INPUT_SAMPLES),
  );
}
assert.equal(outputBuffers(persistent).length, 0, "a later assistant response remains echo-only");
assert.equal(persistent._echoModelReady, true, "learned state persists between assistant responses");
const persistentWeightEnergy = persistent._echoFilter.reduce((sum, value) => sum + value * value, 0);
persistent.port.onmessage({ data: { kind: "echo_guard", mode: "strict", nativeAec: true } });
persistent.port.onmessage({ data: { kind: "echo_guard", mode: "adaptive", nativeAec: true } });
assert.equal(persistent._echoModelReady, true, "explicit mode changes retain the learned predictor");
assert.equal(
  persistent._echoFilter.reduce((sum, value) => sum + value * value, 0),
  persistentWeightEnergy,
  "explicit mode changes do not rewrite predictor weights",
);

const doubleTalk = configure("adaptive");
const reference = deterministicNoise(INPUT_SAMPLES * 40);
const acousticEcho = roomEcho(reference, Math.round(sampleRate * 0.08), 0.35);
for (let offset = 0; offset < INPUT_SAMPLES * 20; offset += INPUT_SAMPLES) {
  doubleTalk._ingest(
    acousticEcho.subarray(offset, offset + INPUT_SAMPLES),
    reference.subarray(offset, offset + INPUT_SAMPLES),
  );
}
assert.equal(outputBuffers(doubleTalk).length, 0, "echo-only warmup uploads no PCM");

const ambiguousOffset = INPUT_SAMPLES * 20;
doubleTalk._ingest(
  add(
    acousticEcho.subarray(ambiguousOffset, ambiguousOffset + INPUT_SAMPLES),
    scale(tone(997, 0.17), 0.02),
  ),
  reference.subarray(ambiguousOffset, ambiguousOffset + INPUT_SAMPLES),
);
assert.equal(outputBuffers(doubleTalk).length, 0, "one uncertain frame fails closed");

const humanFrames = [];
for (let frame = 0; frame < 12; frame++) {
  const offset = INPUT_SAMPLES * (21 + frame);
  const human = tone(997, 0.37 + frame * 0.11);
  const mic = add(acousticEcho.subarray(offset, offset + INPUT_SAMPLES), human);
  humanFrames.push(decimate48k(mic));
  doubleTalk._ingest(mic, reference.subarray(offset, offset + INPUT_SAMPLES));
  if (frame < 11) {
    assert.equal(outputBuffers(doubleTalk).length, 0, "candidate audio remains local before 450 ms confirmation");
  }
}
const released = outputBuffers(doubleTalk);
assert.equal(released.length, 12, "confirmed double-talk releases the complete buffered onset");
assert.ok(
  pcmCorrelation(released[0], humanFrames[0]) > 0.995,
  "released barge-in is original mic PCM rather than predictor residual",
);

for (let frame = 33; frame < 38; frame++) {
  const offset = INPUT_SAMPLES * frame;
  const human = tone(997, 0.37 + frame * 0.11);
  doubleTalk._ingest(
    add(acousticEcho.subarray(offset, offset + INPUT_SAMPLES), human),
    reference.subarray(offset, offset + INPUT_SAMPLES),
  );
}
const metric = doubleTalk.messages.filter((value) => value?.kind === "echo_metric").at(-1);
assert.ok(metric, "echo diagnostics are emitted");
assert.equal(metric.nativeAec, true);
assert.equal(metric.modelReady, true);
assert.equal(metric.doubleTalk, true);
assert.ok(Number.isFinite(metric.erleDb));
assert.ok(metric.candidateMs >= 450);

const naturalBargeIn = configure("adaptive");
const naturalReference = deterministicNoise(INPUT_SAMPLES * 48, 0x13572468);
const naturalEcho = directEcho(naturalReference, Math.round(sampleRate * 0.08), 0.08);
for (let frame = 0; frame < 20; frame++) {
  const offset = frame * INPUT_SAMPLES;
  naturalBargeIn._ingest(
    naturalEcho.subarray(offset, offset + INPUT_SAMPLES),
    naturalReference.subarray(offset, offset + INPUT_SAMPLES),
  );
}
assert.equal(naturalBargeIn._echoModelReady, true, "quiet hardware-style echo trains the predictor");
assert.equal(outputBuffers(naturalBargeIn).length, 0, "quiet echo-only playback remains suppressed");

let humanEvidenceFrames = 0;
const naturalMicFrames = [];
for (let frame = 20; frame < 34; frame++) {
  const offset = frame * INPUT_SAMPLES;
  const isBriefSpeechGap = frame === 23 || frame === 28;
  const human = isBriefSpeechGap
    ? new Float32Array(INPUT_SAMPLES)
    : scale(tone(997, 0.21 + frame * 0.07), 0.08);
  const mic = add(naturalEcho.subarray(offset, offset + INPUT_SAMPLES), human);
  naturalMicFrames.push(decimate48k(mic));
  naturalBargeIn._ingest(mic, naturalReference.subarray(offset, offset + INPUT_SAMPLES));
  if (!isBriefSpeechGap) humanEvidenceFrames++;
  if (humanEvidenceFrames < 12) {
    assert.equal(
      outputBuffers(naturalBargeIn).length,
      0,
      "natural barge-in stays local until 450 ms of human evidence",
    );
  }
}
const naturalReleased = outputBuffers(naturalBargeIn);
assert.equal(naturalReleased.length, 14, "brief speech gaps do not prevent confirmed barge-in");
assert.ok(
  pcmCorrelation(naturalReleased[0], naturalMicFrames[0]) > 0.995,
  "natural barge-in still releases untouched original mic PCM",
);
for (let frame = 34; frame < 40; frame++) {
  const offset = frame * INPUT_SAMPLES;
  naturalBargeIn._ingest(
    naturalEcho.subarray(offset, offset + INPUT_SAMPLES),
    naturalReference.subarray(offset, offset + INPUT_SAMPLES),
  );
}
assert.equal(
  outputBuffers(naturalBargeIn).length,
  naturalReleased.length,
  "echo after a genuine interruption is suppressed instead of extending the user turn",
);

delayed.port.onmessage({ data: { kind: "echo_reset" } });
assert.equal(delayed._referenceHistory.length, 0, "echo reset clears playback history");
assert.equal(delayed._echoModelReady, false, "echo reset clears adaptive readiness");
assert.equal(delayed._echoFilter.some((value) => value !== 0), false, "echo reset clears predictor weights");

console.log("echo guard tests passed");
