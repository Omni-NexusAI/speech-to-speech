import assert from "node:assert/strict";
import crypto from "node:crypto";
import fs from "node:fs";
import { performance } from "node:perf_hooks";

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
globalThis.registerProcessor = (name, implementation) => {
  if (name === "aec3-capture") Processor = implementation;
};

const assetRoot = new URL(
  "../web/hf-realtime-voice/worklets/aec3/",
  import.meta.url,
);
const wasmUrl = new URL("aec3.wasm", assetRoot);
const manifestUrl = new URL("aec3.manifest.json", assetRoot);
const processorUrl = new URL("aec3-capture.js", assetRoot);
const loaderUrl = new URL("aec3-loader.js", assetRoot);
const bytes = fs.readFileSync(wasmUrl);
const manifest = JSON.parse(fs.readFileSync(manifestUrl, "utf8"));
assert.equal(
  crypto.createHash("sha256").update(bytes).digest("hex"),
  manifest.sha256,
  "manifest authenticates the installed AEC3 WASM",
);
const wasmModule = await WebAssembly.compile(bytes);
assert.deepEqual(WebAssembly.Module.imports(wasmModule), [], "AEC3 WASM is import-free");
assert.ok(structuredClone(wasmModule) instanceof WebAssembly.Module);

await import(processorUrl);
assert.ok(Processor, "aec3-capture worklet registered");

const processor = new Processor({
  processorOptions: {
    chunkMs: 40,
    aec3Module: wasmModule,
    aec3Manifest: manifest,
  },
});
processor.port.onmessage({
  data: {
    kind: "echo_guard",
    mode: "adaptive",
    nativeAec: true,
  },
});
processor.port.onmessage({
  data: {
    kind: "echo_calibration",
    delayMs: 50,
    outputLatencyMs: 0,
    suppressionStrength: 0.65,
    leakageThreshold: 0.65,
    doubleTalkSensitivity: 0.5,
  },
});

const history = [];
let state = 0x12345678;
let elapsedMs = 0;
let echoInputEnergy = 0;
let echoInputSamples = 0;
let echoOutputEnergy = 0;
let echoOutputSamples = 0;
for (let frame = 0; frame < 500; frame += 1) {
  const far = new Float32Array(480);
  for (let index = 0; index < far.length; index += 1) {
    state = (1664525 * state + 1013904223) >>> 0;
    far[index] = (((state / 0xffffffff) * 2) - 1) * 0.18;
  }
  history.push(far);
  const echo = history[Math.max(0, frame - 5)];
  const capture = new Float32Array(480);
  for (let index = 0; index < capture.length; index += 1) {
    capture[index] = echo[index] * 0.45;
    if (frame >= 350) {
      echoInputEnergy += capture[index] * capture[index];
      echoInputSamples += 1;
    }
  }
  const before = processor.messages.length;
  const started = performance.now();
  processor.process([[capture], [far]]);
  elapsedMs += performance.now() - started;
  if (frame >= 350) {
    for (const message of processor.messages.slice(before)) {
      if (!(message instanceof ArrayBuffer)) continue;
      const pcm = new Int16Array(message);
      for (const sample of pcm) {
        echoOutputEnergy += (sample / 32768) ** 2;
        echoOutputSamples += 1;
      }
    }
  }
}

const metric = processor.messages.filter((value) => value?.kind === "echo_metric").at(-1);
assert.ok(metric, "AEC3 metrics emitted");
assert.equal(metric.moduleAvailable, true);
assert.equal(metric.mode, "adaptive");
assert.equal(metric.referenceWired, true);
assert.ok(metric.erleDb > 8, `expected useful ERLE, received ${metric.erleDb}`);
assert.ok(metric.lagMs >= 40 && metric.lagMs <= 60, `unexpected delay ${metric.lagMs}`);
assert.ok(elapsedMs / 500 < 10, "AEC3 stays inside each 10 ms frame budget");
const echoReductionDb = 10 * Math.log10(
  (echoInputEnergy / echoInputSamples)
    / Math.max(echoOutputEnergy / echoOutputSamples, 1e-20),
);
assert.ok(
  echoReductionDb > 8,
  `worklet expected >8 dB echo reduction, received ${echoReductionDb.toFixed(2)} dB`,
);

let nearEndOutputEnergy = 0;
let nearEndInputEnergy = 0;
for (let frame = 0; frame < 20; frame += 1) {
  const far = new Float32Array(480);
  const echo = history[history.length - 5 + (frame % 5)];
  const capture = new Float32Array(480);
  for (let index = 0; index < capture.length; index += 1) {
    const near = 0.3 * Math.sin((2 * Math.PI * 997 * index) / sampleRate + frame * 0.1);
    far[index] = history[history.length - 1][index];
    capture[index] = echo[index] * 0.45 + near;
    nearEndInputEnergy += near * near;
  }
  const before = processor.messages.length;
  processor.process([[capture], [far]]);
  for (const message of processor.messages.slice(before)) {
    if (!(message instanceof ArrayBuffer)) continue;
    const pcm = new Int16Array(message);
    for (const sample of pcm) nearEndOutputEnergy += (sample / 32768) ** 2;
  }
}
assert.ok(
  nearEndOutputEnergy > (nearEndInputEnergy / 3) * 0.1,
  `AEC3 preserves independent near-end speech for barge-in (output=${nearEndOutputEnergy}, input=${nearEndInputEnergy})`,
);
const doubleTalkMetric = processor.messages
  .filter((value) => value?.kind === "echo_metric")
  .at(-1);
assert.equal(doubleTalkMetric.doubleTalk, true);
assert.equal(doubleTalkMetric.doubleTalkSource, "aec3-output-evidence");

const makeProcessor = () => new Processor({
  processorOptions: {
    chunkMs: 40,
    aec3Module: wasmModule,
    aec3Manifest: manifest,
  },
});
const pendingNearEndFrame = () => new Float32Array(480).fill(0.2);
const queueUncertainNearEnd = (target) => {
  const decision = target._strictGate.consume(pendingNearEndFrame(), {
    playbackActive: true,
    nearEndEvidence: true,
    highConfidenceEchoOnly: false,
  });
  assert.equal(decision.emit.length, 0);
  assert.equal(target._strictGate.pending.length, 1);
};
const buffersAfter = (target, before) => target.messages
  .slice(before)
  .filter((value) => value instanceof ArrayBuffer);
const assertResetFlushesOnset = (target, label) => {
  const before = target.messages.length;
  target.port.onmessage({ data: { kind: "echo_reset" } });
  const emitted = buffersAfter(target, before);
  assert.equal(emitted.length, 1, `${label} remains available to the output path`);
  const pcm = new Int16Array(emitted[0]);
  assert.equal(pcm.length, 160, `${label} preserves the exact 10 ms onset`);
  assert.ok([...pcm].every((sample) => sample > 0), `${label} emits retained PCM rather than silence`);
};

const modeTransitionProcessor = makeProcessor();
modeTransitionProcessor.port.onmessage({
  data: { kind: "echo_guard", mode: "strict", nativeAec: true },
});
queueUncertainNearEnd(modeTransitionProcessor);
modeTransitionProcessor.port.onmessage({
  data: { kind: "echo_guard", mode: "native", nativeAec: true },
});
assert.equal(modeTransitionProcessor._strictGate.pending.length, 0);
assert.equal(
  modeTransitionProcessor._chunkWrite,
  160,
  "leaving Strict routes pending uncertain speech into the normal output FIFO",
);
assertResetFlushesOnset(modeTransitionProcessor, "mode-transition onset");

const moduleFailureProcessor = makeProcessor();
moduleFailureProcessor.port.onmessage({
  data: { kind: "echo_guard", mode: "strict", nativeAec: true },
});
queueUncertainNearEnd(moduleFailureProcessor);
moduleFailureProcessor._disableModule(new Error("synthetic module failure"));
assert.equal(moduleFailureProcessor._strictGate.pending.length, 0);
assert.equal(
  moduleFailureProcessor._chunkWrite,
  160,
  "a Strict module failure preserves pending uncertain speech before fallback",
);
assertResetFlushesOnset(moduleFailureProcessor, "module-failure onset");

const resetProcessor = makeProcessor();
resetProcessor.port.onmessage({
  data: { kind: "echo_guard", mode: "strict", nativeAec: true },
});
queueUncertainNearEnd(resetProcessor);
const beforeReset = resetProcessor.messages.length;
resetProcessor.port.onmessage({ data: { kind: "echo_reset" } });
const resetBuffers = buffersAfter(resetProcessor, beforeReset);
assert.equal(resetBuffers.length, 1, "echo reset flushes pending uncertain speech");
assert.equal(new Int16Array(resetBuffers[0]).length, 160, "reset preserves the exact 10 ms onset");
assert.ok(
  [...new Int16Array(resetBuffers[0])].every((sample) => sample > 0),
  "reset emits retained onset PCM rather than silence",
);
assert.equal(resetProcessor._chunkWrite, 0);
assert.equal(resetProcessor._strictGate.pending.length, 0);

const { loadAec3Worklet } = await import(loaderUrl);
const addedModules = [];
const fetchImpl = async (url) => {
  const href = String(url);
  if (href.endsWith("aec3.manifest.json")) {
    return new Response(JSON.stringify(manifest), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  }
  if (href.endsWith("aec3.wasm")) return new Response(bytes, { status: 200 });
  return new Response("missing", { status: 404 });
};
const loaded = await loadAec3Worklet(
  { audioWorklet: { addModule: async (url) => addedModules.push(url) } },
  { manifestUrl, fetchImpl },
);
assert.equal(loaded.available, true);
assert.equal(loaded.processorName, "aec3-capture");
assert.equal(addedModules.length, 1);

const badManifest = { ...manifest, sha256: "0".repeat(64) };
const rejected = await loadAec3Worklet(
  { audioWorklet: { addModule: async () => assert.fail("bad WASM must not register") } },
  {
    manifestUrl,
    fetchImpl: async (url) => String(url).endsWith(".json")
      ? new Response(JSON.stringify(badManifest), { status: 200 })
      : new Response(bytes, { status: 200 }),
  },
);
assert.equal(rejected.available, false);
assert.match(rejected.reason, /SHA-256/);

console.log("real AEC3 worklet tests passed");
