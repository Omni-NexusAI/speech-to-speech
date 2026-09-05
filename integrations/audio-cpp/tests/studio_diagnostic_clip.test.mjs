import assert from 'node:assert/strict';
import fs from 'node:fs';
const source = fs.readFileSync(new URL('../gradio_voice_studio.py', import.meta.url), 'utf8');
const start = source.indexOf('  async function sendDiagnosticClip()');
const end = source.indexOf('\n  function ', start);
const code = source.slice(start, end);
assert.ok(start > 0 && end > start);
const state = { turnId: 1, isLive: false, isProcessing: false, isSpeaking: false };
const status = [];
const sent = [];
const refs = { 'diagnostic-clip': { files: [] }, 'llm-input-rate': { value: '16000' }, 'capture-diagnostics': {} };
let resolveDecode;
let decoding = Promise.resolve({ length: 32, duration: .002, numberOfChannels: 1, sampleRate: 16000, getChannelData: () => new Float32Array(32) });
const run = new Function('state', 'refs', 'setStatus', 'setProcessing', 'getAudioContext', 'pcmToWav', 'resamplePcm', 'sendToLLM', `${code}\nreturn sendDiagnosticClip;`)(
  state, refs, text => status.push(text), active => { state.isProcessing = active; },
  () => ({ resume: async () => {}, decodeAudioData: () => decoding }),
  (pcm, rate) => ({ pcm, rate }), pcm => pcm, clip => { sent.push(clip); state.isProcessing = false; },
);
await run();
assert.equal(sent.length, 0);
assert.match(status.at(-1), /Choose a complete local WAV/);
refs['diagnostic-clip'].files = [{ name: 'explicit-test.wav', size: 64, arrayBuffer: async () => new ArrayBuffer(64) }];
await run();
assert.equal(sent.length, 1);
assert.equal(sent[0].rate, 16000);
assert.match(refs['capture-diagnostics'].textContent, /Microphone and VAD bypassed/);
state.isLive = true;
await run();
assert.equal(sent.length, 1, 'diagnostic must not capture or interrupt a live microphone');
state.isLive = false;
decoding = new Promise(resolve => { resolveDecode = resolve; });
const pending = run();
await new Promise(resolve => setImmediate(resolve));
state.turnId++;
state.isProcessing = false;
resolveDecode({ length: 32, duration: .002 });
await pending;
assert.equal(sent.length, 1, 'cancelled file decode must not submit a late turn');
decoding = Promise.resolve({ length: 32, duration: 31 });
await run();
assert.equal(sent.length, 1);
assert.equal(state.isProcessing, false);
assert.match(status.at(-1), /at most 30 seconds/);
assert.ok(source.includes("refs['send-diagnostic-clip'].addEventListener('click', sendDiagnosticClip)"));
console.log('Studio explicit diagnostic clip admission, cancellation and recovery passed');
