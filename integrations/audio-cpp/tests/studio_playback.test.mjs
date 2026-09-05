import assert from "node:assert/strict";
import fs from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import path from "node:path";

globalThis.sampleRate = 1000;
let RegisteredProcessor = null;

class FakeAudioWorkletProcessor {
  constructor() {
    this.port = {
      messages: [],
      onmessage: null,
      postMessage: (message) => this.port.messages.push(message),
    };
  }
}

globalThis.AudioWorkletProcessor = FakeAudioWorkletProcessor;
globalThis.registerProcessor = (name, constructor) => {
  assert.equal(name, "studio-playback");
  RegisteredProcessor = constructor;
};

const workletPath = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../../../web/hf-realtime-voice/worklets/studio-playback.js",
);
await import(pathToFileURL(workletPath).href + `?test=${Date.now()}`);
assert.ok(RegisteredProcessor, "worklet must register its processor");

function processor() {
  return new RegisteredProcessor();
}

function send(instance, data) {
  instance.port.onmessage({ data });
}

function render(instance, frames = 128) {
  const output = new Float32Array(frames);
  instance.process([], [[output]]);
  return output;
}

function messages(instance, kind) {
  return instance.port.messages.filter((message) => message.kind === kind);
}

{
  const instance = processor();
  send(instance, { kind: "begin", turnId: 1, inputRate: 1000, startupMs: 20 });
  send(instance, { kind: "audio", turnId: 1, samples: new Float32Array(10).fill(0.2) });
  render(instance, 4);
  assert.equal(messages(instance, "started").length, 0, "first block alone must not start native playback");
  send(instance, { kind: "audio", turnId: 1, samples: new Float32Array(10).fill(0.3) });
  render(instance, 1);
  assert.equal(messages(instance, "started").length, 1, "first plus steady duration starts on first rendered frame");
  send(instance, { kind: "end", turnId: 1 });
  render(instance, 64);
  assert.equal(messages(instance, "drained").length, 1, "completed turn drains once");
}

{
  const instance = processor();
  send(instance, { kind: "begin", turnId: 2, inputRate: 1000, startupMs: 1000 });
  send(instance, { kind: "audio", turnId: 2, samples: new Float32Array(5).fill(0.2) });
  send(instance, { kind: "end", turnId: 2 });
  render(instance, 32);
  assert.equal(messages(instance, "started").length, 1, "a completed short utterance starts without padding to the startup target");
  assert.equal(messages(instance, "drained").length, 1);
}

{
  const instance = processor();
  send(instance, { kind: "begin", turnId: 3, inputRate: 1000, startupMs: 0 });
  send(instance, { kind: "audio", turnId: 3, samples: new Float32Array(4).fill(0.4) });
  render(instance, 16);
  assert.equal(messages(instance, "underrun").length, 1, "queue starvation is observable before turn end");
  render(instance, 64);
  send(instance, { kind: "audio", turnId: 3, samples: new Float32Array(4).fill(0.5) });
  render(instance, 64);
  assert.equal(messages(instance, "started").at(-1).resumed, true, "queue can resume after an underrun");

  send(instance, { kind: "clear", turnId: 4 });
  const queuedBeforeStale = messages(instance, "queued").length;
  send(instance, { kind: "audio", turnId: 3, samples: new Float32Array(40).fill(0.8) });
  render(instance, 16);
  assert.equal(messages(instance, "clear").length, 1);
  assert.equal(messages(instance, "queued").length, queuedBeforeStale, "cleared turn rejects stale PCM");
  assert.ok(render(instance, 4).every((sample) => sample === 0), "clear leaves no stale tail in the output");
}

{
  const instance = processor();
  send(instance, { kind: "begin", turnId: 4, inputRate: 1000, startupMs: 100, reprimeMs: 20 });
  send(instance, { kind: "audio", turnId: 4, samples: new Float32Array(100).fill(0.2) });
  render(instance, 140); // Drain the cold block and the fixed fade-out.
  assert.equal(messages(instance, "underrun").length, 1, "a true starvation enters recovery priming");
  send(instance, { kind: "audio", turnId: 4, samples: new Float32Array(19).fill(0.3) });
  render(instance, 1);
  assert.equal(messages(instance, "started").length, 1, "recovery must not wait for the original cold 100ms reservoir");
  send(instance, { kind: "audio", turnId: 4, samples: new Float32Array(1).fill(0.3) });
  render(instance, 1);
  assert.equal(messages(instance, "started").at(-1).resumed, true, "recovery starts at its explicit steady-state re-prime target");
}

console.log("studio playback worklet tests passed");

const uiPath = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../gradio_voice_studio.py");
const activeWidget = fs.readFileSync(uiPath, "utf8").split("def _build_streaming_widget_html(").at(-1);
const embedded = activeWidget.match(/<script type="text\/plain"[^>]*>([\s\S]*?)<\/script>/);
assert.ok(embedded, "active Voice Studio widget must contain its boot script");
assert.doesNotThrow(
  () => new Function(embedded[1].replace("__CONFIG__", "{}")),
  "active Voice Studio boot script must parse as JavaScript",
);
console.log("studio embedded widget syntax passed");
