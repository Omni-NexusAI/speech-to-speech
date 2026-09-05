// Response-scoped source-clock regression. A live backend acknowledgement may
// change the default 16 kHz <-> 24 kHz rate while old PCM remains queued; that
// old response must keep its original pitch/duration and the next one must use
// the new clock.
import assert from "node:assert/strict";

globalThis.sampleRate = 48_000;
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
globalThis.registerProcessor = (name, implementation) => {
  if (name === "audio-playback") Processor = implementation;
};

await import(new URL("../web/hf-realtime-voice/worklets/audio-playback.js", import.meta.url));
assert.ok(Processor, "audio playback worklet registered");

const processor = new Processor();
const send = (data) => processor.port.onmessage({ data });

// First response: 160 samples at 16 kHz = exactly 10 ms = 480 context frames.
send({ kind: "config", inputRate: 16_000, generation: 0 });
send({
  kind: "audio", generation: 0, streamId: "old-response", responseEpoch: 1,
  primeMs: 0, sourceSampleRate: 16_000, samples: new Float32Array(160).fill(0.25),
});
send({ kind: "end", generation: 0, streamId: "old-response", responseEpoch: 1, sourceSampleRate: 16_000 });

// The backend now switches to 24 kHz. It only changes the default; old queued
// PCM still carries the 16 kHz response snapshot.
send({ kind: "config", inputRate: 24_000, generation: 0 });
send({
  kind: "audio", generation: 0, streamId: "new-response", responseEpoch: 2,
  primeMs: 0, sourceSampleRate: 24_000, samples: new Float32Array(240).fill(0.75),
});
send({ kind: "end", generation: 0, streamId: "new-response", responseEpoch: 2, sourceSampleRate: 24_000 });

const rendered = [];
for (let frame = 0; frame < 16; frame += 1) {
  const output = new Float32Array(128);
  processor.process([], [[output]]);
  rendered.push(...output);
}

// The cross-response interpolation spans at most the final two old-clock
// frames. A global 24 kHz config bug would reach the new PCM around frame 320.
const firstNew = rendered.findIndex((sample) => sample >= 0.70);
assert.ok(firstNew >= 480 && firstNew <= 481,
  `old 16 kHz response was retimed by live config (new PCM began at ${firstNew})`);

// 160/16k plus 240/24k is 20 ms total => 960 context frames. This checks both
// old-response duration and that the next response adopted the acknowledged
// 24 kHz rate rather than inheriting the old one.
let lastAudible = -1;
for (let index = 0; index < rendered.length; index += 1) {
  if (Math.abs(rendered[index]) > 1e-6) lastAudible = index;
}
assert.ok(lastAudible >= 959 && lastAudible <= 960,
  `response-scoped source clocks produced ${lastAudible + 1} audible frames, expected 960`);
assert.equal(processor._inputRate, 24_000, "new response must become the active source clock after FIFO crossover");
const responseDrains = processor.messages.filter((message) => message.kind === "drained");
assert.deepEqual(responseDrains.map((message) => message.streamId), ["old-response", "new-response"],
  "each response in a continuous FIFO must retire at its own playback boundary");
assert.equal(responseDrains[0].queueEmpty, false,
  "the old response boundary must not claim the shared FIFO is empty");
assert.equal(responseDrains[1].queueEmpty, true,
  "the final response must own the queue-empty drain");

// Cold priming and an underrun recovery are intentionally distinct. The first
// response waits for its configured reservoir; recovery begins from the
// steady target, expands only from observed cadence, and remains capped at two
// seconds without changing the PCM samples or their source clock.
const recovery = new Processor();
const recoveryMessages = recovery.messages;
const recoverySend = (data) => recovery.port.onmessage({ data });
const recoveryRender = (frames) => {
  const output = new Float32Array(frames);
  recovery.process([], [[output]]);
  return output;
};
recoverySend({ kind: "config", inputRate: 48_000, generation: 0 });
recoverySend({
  kind: "audio", generation: 0, streamId: "recovery", responseEpoch: 3,
  primeMs: 1_000, reprimeMs: 80, maxPrimeMs: 2_000, continuityMode: "adaptive", sourceSampleRate: 48_000,
  samples: new Float32Array(48_000).fill(0.5),
});
recoveryRender(48_000);
recoveryRender(480); // Empty for 10ms: enough to produce one genuine underrun.
assert.equal(recoveryMessages.filter((message) => message.kind === "underrun").length, 1);
assert.equal(recovery._state, "priming");
recoverySend({
  kind: "audio", generation: 0, streamId: "recovery", responseEpoch: 3,
  primeMs: 1_000, reprimeMs: 80, maxPrimeMs: 2_000, continuityMode: "adaptive", sourceSampleRate: 48_000,
  samples: new Float32Array(48_000).fill(0.5),
});
assert.ok(recovery._reprimeTargetMs > 80 && recovery._reprimeTargetMs <= 2_000,
  `cadence-derived recovery target escaped its bounded reservoir: ${recovery._reprimeTargetMs}`);
assert.equal(recovery._state, "priming", "one second is below the learned re-prime target");
recoverySend({
  kind: "audio", generation: 0, streamId: "recovery", responseEpoch: 3,
  primeMs: 1_000, reprimeMs: 80, maxPrimeMs: 2_000, continuityMode: "adaptive", sourceSampleRate: 48_000,
  samples: new Float32Array(48_000).fill(0.5),
});
const recoveryOutput = recoveryRender(8);
assert.equal(recoveryMessages.filter((message) => message.kind === "started").at(-1).reprime, true);
assert.equal(recovery._inputRate, 48_000, "adaptive recovery must not alter the response PCM clock");
assert.ok(recoveryOutput.every((sample) => sample === 0.5), "adaptive recovery must not stretch, duplicate, or alter PCM");

// Cadence and all recovery bounds belong to the response whose PCM supplied
// them. A long gap in response A must not leak into queued response B, and one
// later short B gap must not erase B's own previously learned long gap.
const scopedCadence = new Processor();
const cadenceSend = (data) => scopedCadence.port.onmessage({ data });
globalThis.currentFrame = 0;
cadenceSend({
  kind: "audio", generation: 0, streamId: "cadence-a", responseEpoch: 10,
  primeMs: 0, reprimeMs: 80, maxPrimeMs: 2_000, sourceSampleRate: 48_000,
  samples: new Float32Array(32).fill(0.1),
});
globalThis.currentFrame = 86_400; // 1.8 seconds at the AudioContext clock.
cadenceSend({
  kind: "audio", generation: 0, streamId: "cadence-a", responseEpoch: 10,
  primeMs: 0, reprimeMs: 80, maxPrimeMs: 2_000, sourceSampleRate: 48_000,
  samples: new Float32Array(32).fill(0.1),
});
assert.equal(scopedCadence._reprimeTargetMs, 1_880,
  "response A did not learn its long-gap recovery reservoir");

cadenceSend({
  kind: "audio", generation: 0, streamId: "cadence-b", responseEpoch: 11,
  primeMs: 0, reprimeMs: 80, maxPrimeMs: 1_000, sourceSampleRate: 48_000,
  samples: new Float32Array(32).fill(0.2),
});
globalThis.currentFrame = 115_200; // B sees a 600ms gap.
cadenceSend({
  kind: "audio", generation: 0, streamId: "cadence-b", responseEpoch: 11,
  primeMs: 0, reprimeMs: 80, maxPrimeMs: 1_000, sourceSampleRate: 48_000,
  samples: new Float32Array(32).fill(0.2),
});
const queuedB = scopedCadence._queue.find((buffer) => buffer.streamId === "cadence-b");
scopedCadence._activateQueuedStream(queuedB);
assert.equal(scopedCadence._reprimeTargetMs, 680,
  "queued response B inherited response A's cadence or recovery ceiling");
globalThis.currentFrame = 120_000; // A later 100ms B gap must not lower 680ms.
cadenceSend({
  kind: "audio", generation: 0, streamId: "cadence-b", responseEpoch: 11,
  primeMs: 0, reprimeMs: 80, maxPrimeMs: 1_000, sourceSampleRate: 48_000,
  samples: new Float32Array(32).fill(0.2),
});
assert.equal(scopedCadence._observedChunkGapMs, 100);
assert.equal(scopedCadence._maxObservedChunkGapMs, 600);
assert.equal(scopedCadence._reprimeTargetMs, 680,
  "a short follow-up chunk lowered the learned same-response recovery reserve");
delete globalThis.currentFrame;

// Fast Start remains a fixed low-reservoir comparison mode even when a long
// backend gap is observed; only Adaptive is permitted to learn cadence.
const fastStart = new Processor();
const fastSend = (data) => fastStart.port.onmessage({ data });
globalThis.currentFrame = 0;
fastSend({
  kind: "audio", generation: 0, streamId: "fast", responseEpoch: 12,
  primeMs: 0, reprimeMs: 0, maxPrimeMs: 2_000, continuityMode: "fast-start",
  sourceSampleRate: 48_000, samples: new Float32Array(32).fill(0.3),
});
globalThis.currentFrame = 96_000;
fastSend({
  kind: "audio", generation: 0, streamId: "fast", responseEpoch: 12,
  primeMs: 0, reprimeMs: 0, maxPrimeMs: 2_000, continuityMode: "fast-start",
  sourceSampleRate: 48_000, samples: new Float32Array(32).fill(0.3),
});
assert.equal(fastStart._maxObservedChunkGapMs, 2_000);
assert.equal(fastStart._reprimeTargetMs, 0,
  "Fast Start must not learn or raise its fixed re-prime reservoir");
delete globalThis.currentFrame;

console.log("response-scoped playback sample-rate tests passed");
