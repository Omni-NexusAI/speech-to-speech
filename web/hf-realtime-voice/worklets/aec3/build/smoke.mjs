import assert from "node:assert/strict";
import fs from "node:fs";
import { performance } from "node:perf_hooks";

const wasmPath = process.argv[2];
assert.ok(wasmPath, "usage: node smoke.mjs <aec3.wasm>");
const bytes = fs.readFileSync(wasmPath);
const module = await WebAssembly.compile(bytes);
assert.deepEqual(WebAssembly.Module.imports(module), [], "AEC3 WASM must be import-free");
assert.ok(structuredClone(module) instanceof WebAssembly.Module, "module must be structured-cloneable");
const { exports: api } = await WebAssembly.instantiate(module, {});
const handle = api.aec3_create(48000, 1);
assert.ok(handle > 0, "AEC3 session created");
assert.equal(api.aec3_abi_version(), 1);
assert.equal(api.aec3_frame_samples(handle), 480);

const view = (pointer) => new Float32Array(api.memory.buffer, pointer, 480);
const render = view(api.aec3_render_ptr(handle));
const capture = view(api.aec3_capture_ptr(handle));
const output = view(api.aec3_output_ptr(handle));
const history = [];
let state = 0x12345678;
let inputEnergy = 0;
let outputEnergy = 0;
const started = performance.now();

for (let frame = 0; frame < 500; frame += 1) {
  const far = new Float32Array(480);
  for (let index = 0; index < far.length; index += 1) {
    state = (1664525 * state + 1013904223) >>> 0;
    far[index] = (((state / 0xffffffff) * 2) - 1) * 0.18;
  }
  history.push(far);
  render.set(far);
  assert.equal(api.aec3_process_render(handle, frame * 10), 0);
  const echo = history[Math.max(0, frame - 5)];
  for (let index = 0; index < capture.length; index += 1) capture[index] = echo[index] * 0.45;
  assert.equal(api.aec3_process_capture(handle, 50, frame * 10), 0);
  if (frame >= 350) {
    for (let index = 0; index < capture.length; index += 1) {
      inputEnergy += capture[index] * capture[index];
      outputEnergy += output[index] * output[index];
    }
  }
}

const elapsed = performance.now() - started;
const reductionDb = 10 * Math.log10(inputEnergy / Math.max(outputEnergy, 1e-20));
assert.ok(reductionDb > 8, `expected >8 dB echo reduction, received ${reductionDb.toFixed(2)} dB`);
assert.ok(elapsed / 500 < 10, `10 ms processing deadline missed: ${(elapsed / 500).toFixed(3)} ms`);
console.log(JSON.stringify({
  wasmBytes: bytes.length,
  meanFrameMs: elapsed / 500,
  echoReductionDb: reductionDb,
  erleDb: api.aec3_get_erle_db(handle),
  delayMs: api.aec3_get_delay_ms(handle),
}, null, 2));
api.aec3_destroy(handle);
