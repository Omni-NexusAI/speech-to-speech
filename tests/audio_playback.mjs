import assert from "node:assert/strict";

globalThis.sampleRate = 1000;
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

await import(new URL(
  "../web/hf-realtime-voice/worklets/audio-playback.js",
  import.meta.url,
));
assert.ok(Processor, "audio-playback worklet registered");

function createProcessor(generation = 0) {
  const processor = new Processor();
  processor.port.onmessage({
    data: { kind: "config", inputRate: 1000, generation },
  });
  return processor;
}

function message(processor, data) {
  processor.port.onmessage({ data });
}

function audio(processor, values, {
  generation = 0,
  primeMs = 0,
  streamId = "",
} = {}) {
  message(processor, {
    kind: "audio",
    samples: Float32Array.from(values),
    generation,
    primeMs,
    streamId,
  });
}

function render(processor, frames) {
  const output = new Float32Array(frames);
  processor.process([], [[output]]);
  return output;
}

function kinds(processor, kind) {
  return processor.messages.filter((entry) => entry?.kind === kind);
}

// Startup priming waits for the full target and never consumes or attenuates
// the head while waiting.
{
  const processor = createProcessor();
  const first = Array.from({ length: 400 }, (_, index) => (index + 1) / 1000);
  const second = Array.from({ length: 400 }, (_, index) => (index + 401) / 1000);
  audio(processor, first, { primeMs: 800, streamId: "response-startup" });
  assert.deepEqual([...render(processor, 32)], Array(32).fill(0));
  audio(processor, second, { primeMs: 800, streamId: "response-startup" });
  assert.deepEqual([...render(processor, 800)], [...Float32Array.from([...first, ...second])]);
  assert.equal(kinds(processor, "started").length, 1);
  assert.equal(kinds(processor, "started")[0].streamId, "response-startup");
  assert.equal(kinds(processor, "primed").at(-1).primeTargetMs, 800);
}

// Punctuation-sized chunks share one FIFO. A completed response can drain and
// a same-generation tool-result continuation reuses the same worklet/queue.
{
  const processor = createProcessor();
  const phraseA = [0.1, 0.2, 0.3];
  const phraseB = [0.4, 0.5, 0.6];
  audio(processor, phraseA, { streamId: "response-preamble" });
  audio(processor, phraseB, { streamId: "response-preamble" });
  message(processor, { kind: "end", generation: 0, streamId: "response-preamble" });
  assert.deepEqual(
    [...render(processor, 2)],
    [...Float32Array.from(phraseA.slice(0, 2))],
  );
  // The tool-result response starts while the preamble's PCM is still queued.
  // It must append to that queue instead of being rejected by the prior end.
  audio(processor, [0.7, 0.8], { streamId: "response-tool-result" });
  message(processor, { kind: "end", generation: 0, streamId: "response-tool-result" });
  assert.deepEqual(
    [...render(processor, 6)],
    [...Float32Array.from([...phraseA.slice(2), ...phraseB, 0.7, 0.8])],
  );
  render(processor, 1);
  assert.equal(kinds(processor, "drained").length, 1);
  assert.equal(kinds(processor, "underrun").length, 0);
  assert.deepEqual(
    kinds(processor, "started").map((entry) => entry.streamId),
    ["response-preamble", "response-tool-result"],
    "continuous tool continuation reports each response only when its first sample renders",
  );

  audio(processor, [0.9], { generation: 0, primeMs: 0, streamId: "response-later" });
  message(processor, { kind: "end", generation: 0, streamId: "response-later" });
  assert.deepEqual([...render(processor, 1)], [...Float32Array.from([0.9])]);
}

// When an acknowledged profile changes between responses already sharing the
// FIFO, the new target becomes active only after playback crosses that response
// boundary. A later underrun therefore uses the new response's own snapshot.
{
  const processor = createProcessor();
  audio(processor, [0.1, 0.2, 0.3, 0.4], {
    primeMs: 4,
    streamId: "response-low-latency",
  });
  message(processor, {
    kind: "end",
    generation: 0,
    streamId: "response-low-latency",
  });
  render(processor, 2);
  audio(processor, [0.5, 0.6], {
    primeMs: 8,
    streamId: "response-new-profile",
  });
  render(processor, 5);
  assert.equal(kinds(processor, "underrun").at(-1).primeTargetMs, 8);
  audio(processor, [0.7, 0.8, 0.9, 0.7], {
    primeMs: 8,
    streamId: "response-new-profile",
  });
  assert.deepEqual([...render(processor, 2)], [0, 0]);
  audio(processor, [0.6, 0.5, 0.4, 0.3], {
    primeMs: 8,
    streamId: "response-new-profile",
  });
  assert.equal(kinds(processor, "reprimed").at(-1).primeTargetMs, 8);
}

// Irregular delivery remains silent until the accumulated duration reaches the
// target, and then emits every queued sample in original order.
{
  const processor = createProcessor();
  const delivered = [];
  for (const values of [[0.1, 0.2, 0.3], [0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 0.2, 0.3]]) {
    delivered.push(...values);
    audio(processor, values, { primeMs: 12 });
    assert.deepEqual([...render(processor, 2)], [0, 0]);
  }
  audio(processor, [0.4], { primeMs: 12 });
  delivered.push(0.4);
  assert.deepEqual([...render(processor, 12)], [...Float32Array.from(delivered)]);
}

// A genuine underrun re-primes to the full target instead of restarting from
// the next one-sample block.
{
  const processor = createProcessor();
  audio(processor, [0.1, 0.2, 0.3, 0.4], { primeMs: 4 });
  assert.deepEqual(
    [...render(processor, 5)],
    [...Float32Array.from([0.1, 0.2, 0.3, 0.4]), 0],
  );
  assert.equal(kinds(processor, "underrun").length, 1);
  audio(processor, [0.5], { primeMs: 4 });
  assert.deepEqual([...render(processor, 2)], [0, 0]);
  audio(processor, [0.6, 0.7, 0.8], { primeMs: 4 });
  assert.deepEqual(
    [...render(processor, 4)],
    [...Float32Array.from([0.5, 0.6, 0.7, 0.8])],
  );
  assert.equal(kinds(processor, "reprimed").length, 1);
  assert.equal(kinds(processor, "reprimed")[0].primeTargetMs, 4);
}

// output_audio.done is idempotent and forces a short final response through
// immediately even when it is below the normal startup target.
{
  const processor = createProcessor();
  audio(processor, [0.25, 0.5, 0.75], { primeMs: 800 });
  assert.deepEqual([...render(processor, 2)], [0, 0]);
  message(processor, { kind: "end", generation: 0 });
  message(processor, { kind: "end", generation: 0 });
  assert.deepEqual([...render(processor, 3)], [...Float32Array.from([0.25, 0.5, 0.75])]);
  render(processor, 1);
  assert.equal(kinds(processor, "drained").length, 1);
  assert.equal(kinds(processor, "primed").filter((entry) => entry.forced).length, 1);
}

// Cancellation clears once, advances the generation, and rejects a stale tail.
{
  const processor = createProcessor();
  audio(processor, Array(100).fill(0.4), {
    primeMs: 0,
    streamId: "response-cancelled",
  });
  render(processor, 8);
  assert.equal(kinds(processor, "started").at(-1).streamId, "response-cancelled");
  message(processor, { kind: "clear", generation: 1, reason: "barge-in" });
  message(processor, { kind: "clear", generation: 1, reason: "barge-in" });
  assert.deepEqual([...render(processor, 8)], Array(8).fill(0));
  assert.equal(kinds(processor, "cleared").length, 1);
  assert.equal(kinds(processor, "drained").length, 1);
  assert.equal(kinds(processor, "drained")[0].cleared, true);
  assert.equal(kinds(processor, "drained")[0].streamId, "response-cancelled");

  audio(processor, [0.9], { generation: 0, primeMs: 0 });
  assert.equal(kinds(processor, "stale_chunk_rejected").length, 1);
  audio(processor, [0.6], { generation: 1, primeMs: 0 });
  message(processor, { kind: "end", generation: 1 });
  assert.deepEqual([...render(processor, 1)], [...Float32Array.from([0.6])]);
}

// Reconnect creates a fresh processor with no queued state from the old node.
{
  const oldProcessor = createProcessor();
  audio(oldProcessor, Array(20).fill(0.8), { primeMs: 800 });
  const replacement = createProcessor();
  assert.deepEqual([...render(replacement, 4)], [0, 0, 0, 0]);
  audio(replacement, [0.2], { primeMs: 0 });
  assert.deepEqual([...render(replacement, 1)], [...Float32Array.from([0.2])]);
}

console.log("continuous generation-safe audio playback tests passed");
