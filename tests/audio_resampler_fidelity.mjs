import assert from "node:assert/strict";

import {
  downmixToMono,
  StatefulPolyphaseResampler,
} from "../web/hf-realtime-voice/worklets/capture-resampler.js";

const OUTPUT_RATE = 16000;

function tone(rate, frequency, seconds, amplitude = 0.8) {
  const samples = new Float32Array(Math.round(rate * seconds));
  for (let index = 0; index < samples.length; index += 1) {
    samples[index] = amplitude * Math.sin((2 * Math.PI * frequency * index) / rate);
  }
  return samples;
}

function rms(samples, trim = 0) {
  const start = Math.min(samples.length, Math.max(0, trim));
  const end = Math.max(start, samples.length - trim);
  let energy = 0;
  for (let index = start; index < end; index += 1) energy += samples[index] ** 2;
  return end > start ? Math.sqrt(energy / (end - start)) : 0;
}

function legacyBrowserResample(samples, inputRate) {
  const ratio = inputRate / OUTPUT_RATE;
  const chunkSamples = 640;
  const needed = Math.ceil(chunkSamples * ratio);
  const chunks = Math.floor(samples.length / needed);
  const output = new Float32Array(chunks * chunkSamples);
  for (let chunk = 0; chunk < chunks; chunk += 1) {
    const inputOffset = chunk * needed;
    const outputOffset = chunk * chunkSamples;
    for (let index = 0; index < chunkSamples; index += 1) {
      if (Math.abs(ratio - 3) < 1e-6) {
        const source = inputOffset + (index * 3);
        output[outputOffset + index] = (
          samples[source] + samples[source + 1] + samples[source + 2]
        ) / 3;
      } else {
        const position = inputOffset + (index * ratio);
        const source = Math.floor(position);
        const fraction = position - source;
        const a = samples[source] || 0;
        const b = samples[Math.min(inputOffset + needed - 1, source + 1)] || a;
        output[outputOffset + index] = a + ((b - a) * fraction);
      }
    }
  }
  return output;
}

function resampleInChunks(samples, inputRate, chunkSizes) {
  const resampler = new StatefulPolyphaseResampler(inputRate, OUTPUT_RATE);
  const chunks = [];
  let offset = 0;
  let chunkIndex = 0;
  while (offset < samples.length) {
    const size = Math.min(chunkSizes[chunkIndex % chunkSizes.length], samples.length - offset);
    chunks.push(resampler.push(samples.subarray(offset, offset + size)));
    offset += size;
    chunkIndex += 1;
  }
  chunks.push(resampler.flush());
  const length = chunks.reduce((total, chunk) => total + chunk.length, 0);
  const output = new Float32Array(length);
  let write = 0;
  for (const chunk of chunks) {
    output.set(chunk, write);
    write += chunk.length;
  }
  return output;
}

function dbRatio(numerator, denominator) {
  return 20 * Math.log10(Math.max(Number.MIN_VALUE, numerator) / Math.max(Number.MIN_VALUE, denominator));
}

const speechBand = tone(48000, 1000, 1);
const upperSpeechBand = tone(48000, 6000, 1);
const passbandEdge = tone(48000, 6400, 1);
const stopbandEdge = tone(48000, 8100, 1);
const aliasBand = tone(48000, 12000, 1);
const legacySpeech = legacyBrowserResample(speechBand, 48000);
const legacyAlias = legacyBrowserResample(aliasBand, 48000);
const polyphaseSpeech = resampleInChunks(speechBand, 48000, [128]);
const polyphaseUpperSpeech = resampleInChunks(upperSpeechBand, 48000, [128]);
const polyphasePassbandEdge = resampleInChunks(passbandEdge, 48000, [128]);
const polyphaseStopbandEdge = resampleInChunks(stopbandEdge, 48000, [128]);
const polyphaseAlias = resampleInChunks(aliasBand, 48000, [128]);
const trim = 256;
const speechGainDb = dbRatio(rms(polyphaseSpeech, trim), 0.8 / Math.sqrt(2));
const upperSpeechGainDb = dbRatio(rms(polyphaseUpperSpeech, trim), 0.8 / Math.sqrt(2));
const passbandEdgeGainDb = dbRatio(rms(polyphasePassbandEdge, trim), 0.8 / Math.sqrt(2));
const stopbandAttenuationDb = dbRatio(0.8 / Math.sqrt(2), rms(polyphaseStopbandEdge, trim));
const legacyAliasRms = rms(legacyAlias, trim);
const polyphaseAliasRms = rms(polyphaseAlias, trim);
const aliasImprovementDb = dbRatio(legacyAliasRms, polyphaseAliasRms);

assert.ok(Math.abs(speechGainDb) <= 0.05, `1 kHz passband gain drifted ${speechGainDb.toFixed(3)} dB`);
assert.ok(
  Math.abs(upperSpeechGainDb) <= 0.05,
  `6 kHz upper speech-band gain drifted ${upperSpeechGainDb.toFixed(3)} dB`,
);
assert.ok(
  Math.abs(passbandEdgeGainDb) <= 0.05,
  `6.4 kHz passband edge drifted ${passbandEdgeGainDb.toFixed(3)} dB`,
);
assert.ok(
  stopbandAttenuationDb >= 60,
  `8.1 kHz stopband attenuation too small: ${stopbandAttenuationDb.toFixed(2)} dB`,
);
assert.ok(legacyAliasRms > 0.1, "legacy 48 kHz box decimator must expose the measured alias regression");
assert.ok(polyphaseAliasRms < 0.001, `12 kHz alias RMS too high: ${polyphaseAliasRms}`);
assert.ok(aliasImprovementDb > 40, `anti-alias improvement too small: ${aliasImprovementDb.toFixed(2)} dB`);

const aliasBand44100 = tone(44100, 12000, 1);
const legacyAlias44100 = rms(legacyBrowserResample(aliasBand44100, 44100), trim);
const polyphaseAlias44100 = rms(resampleInChunks(aliasBand44100, 44100, [128]), trim);
const passbandEdge44100 = rms(resampleInChunks(tone(44100, 6400, 1), 44100, [128]), trim);
const stopbandEdge44100 = rms(resampleInChunks(tone(44100, 8100, 1), 44100, [128]), trim);
const passbandEdge44100Db = dbRatio(passbandEdge44100, 0.8 / Math.sqrt(2));
const stopbandAttenuation44100Db = dbRatio(0.8 / Math.sqrt(2), stopbandEdge44100);
assert.ok(legacyAlias44100 > 0.2, "legacy 44.1 kHz interpolation must expose the measured alias regression");
assert.ok(polyphaseAlias44100 < 0.001, `44.1 kHz alias RMS too high: ${polyphaseAlias44100}`);
assert.ok(Math.abs(passbandEdge44100Db) <= 0.05, "44.1 kHz passband edge exceeded 0.05 dB");
assert.ok(stopbandAttenuation44100Db >= 60, "44.1 kHz stopband attenuation fell below 60 dB");

for (const inputRate of [44100, 48000]) {
  const fixture = tone(inputRate, 1733, 0.25, 0.6);
  const oneBlock = resampleInChunks(fixture, inputRate, [fixture.length]);
  const fragmented = resampleInChunks(fixture, inputRate, [1, 7, 128, 13, 257, 64]);
  assert.deepEqual(fragmented, oneBlock, `${inputRate} Hz output must be invariant to input chunk boundaries`);
  assert.equal(
    oneBlock.length,
    Math.ceil((fixture.length * OUTPUT_RATE) / inputRate),
    `${inputRate} Hz clean flush must preserve exact duration`,
  );
}

const left = Float32Array.from([0.8, 0.4, -0.2]);
const right = Float32Array.from([0.2, -0.4, 0.2]);
assert.deepEqual(
  [...downmixToMono([left, right])].map((value) => Number(value.toFixed(6))),
  [0.5, 0, 0],
  "stereo capture must be averaged rather than dropping the second channel",
);
assert.deepEqual(
  [...downmixToMono([Float32Array.from([1, -1]), Float32Array.from([-1, 1])])],
  [0, 0],
  "anti-phase stereo must cancel in the mono mix",
);
assert.deepEqual(
  [...downmixToMono([], 3)],
  [0, 0, 0],
  "missing reference channels must produce a duration-matched zero reference",
);
assert.deepEqual(
  [...downmixToMono([Float32Array.from([Number.NaN, Number.POSITIVE_INFINITY])])],
  [0, 0],
  "non-finite capture values must become finite silence",
);

const impulse = new Float32Array(480);
impulse[impulse.length - 1] = 1;
const endpoint = new StatefulPolyphaseResampler(48000, OUTPUT_RATE);
const beforeFlush = endpoint.push(impulse);
const tail = endpoint.flush();
assert.ok(tail.length > 0, "endpoint flush must emit the bounded FIR tail");
assert.ok(rms(tail) > 0, "endpoint flush must retain the final input impulse");
assert.equal(endpoint.flush().length, 0, "endpoint flush must be idempotent after reset");
assert.equal(
  beforeFlush.length + tail.length,
  Math.ceil((impulse.length * OUTPUT_RATE) / 48000),
  "clean flush must preserve exact duration without appending extra tail frames",
);
const leadingImpulse = new Float32Array(480);
leadingImpulse[0] = 1;
const leadingOutput = resampleInChunks(leadingImpulse, 48000, [7, 128, 1]);
assert.equal(leadingOutput.length, 160, "leading impulse keeps exact flushed duration");
assert.ok(rms(leadingOutput) > 0, "leading impulse must survive the anti-alias path");

for (const inputRate of [44100, 48000]) {
  const contract = new StatefulPolyphaseResampler(inputRate, OUTPUT_RATE);
  assert.equal(contract.tapCount, 121, `${inputRate} Hz filter must retain the bounded 121-tap contract`);
  assert.ok(contract.bankBytes <= 77440, `${inputRate} Hz coefficient bank exceeded 77,440 bytes`);
  assert.ok(contract.lookaheadMs <= 1.37, `${inputRate} Hz lookahead exceeded 1.37 ms`);
  let maximumDcError = 0;
  for (const phase of contract._bank) {
    let total = 0;
    for (const coefficient of phase) total += coefficient;
    maximumDcError = Math.max(maximumDcError, Math.abs(1 - total));
  }
  assert.ok(maximumDcError <= 2e-6, `${inputRate} Hz phase DC error exceeded 2e-6`);
}

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
let fallbackProcessorClass = null;
globalThis.registerProcessor = (_name, implementation) => {
  fallbackProcessorClass = implementation;
};
await import(`../web/hf-realtime-voice/worklets/mic-capture.js?fidelity=${Date.now()}`);
assert.ok(fallbackProcessorClass, "native fallback worklet registered");

function pcmBuffers(processor) {
  return processor.messages.filter((value) => value instanceof ArrayBuffer);
}

const positiveClip = new fallbackProcessorClass({ processorOptions: { chunkMs: 40 } });
positiveClip._ingest(new Float32Array(1978).fill(1.5), new Float32Array(1978));
const positivePcm = new Int16Array(pcmBuffers(positiveClip)[0]);
assert.ok(
  [...positivePcm].every((sample) => sample >= -32768 && sample <= 32767)
    && [...positivePcm].some((sample) => sample === 32767),
  "positive overflow must saturate without PCM16 wrap",
);
const negativeClip = new fallbackProcessorClass({ processorOptions: { chunkMs: 40 } });
negativeClip._ingest(new Float32Array(1978).fill(-1.5), new Float32Array(1978));
const negativePcm = new Int16Array(pcmBuffers(negativeClip)[0]);
assert.ok(
  [...negativePcm].every((sample) => sample >= -32768 && sample <= 32767)
    && [...negativePcm].some((sample) => sample === -32768),
  "negative overflow must saturate without PCM16 wrap",
);

const stereoProcessor = new fallbackProcessorClass({ processorOptions: { chunkMs: 40 } });
stereoProcessor.process([
  [new Float32Array(1978).fill(0.8), new Float32Array(1978).fill(0.2)],
  [new Float32Array(1978)],
]);
const stereoPcm = new Int16Array(pcmBuffers(stereoProcessor)[0]);
assert.ok(
  [...stereoPcm.subarray(100)].every((sample) => sample >= 16382 && sample <= 16384),
  "production capture must average both microphone channels before resampling",
);

const alignedReference = new fallbackProcessorClass({ processorOptions: { chunkMs: 40 } });
alignedReference._ingest(
  new Float32Array(1920).fill(0.25),
  new Float32Array(1920).fill(0.1),
);
assert.equal(
  pcmBuffers(alignedReference).length,
  0,
  "FIR lookahead must not publish a short 620-sample capture chunk",
);
assert.equal(alignedReference._referenceLevels.length, 1);
alignedReference._ingest(new Float32Array(58).fill(0.25), new Float32Array(58));
assert.equal(new Int16Array(pcmBuffers(alignedReference)[0]).length, 640);
assert.equal(
  alignedReference._referenceLevels.length,
  0,
  "each exact output chunk consumes its corresponding native reference RMS",
);

const routeReset = new fallbackProcessorClass({ processorOptions: { chunkMs: 40 } });
routeReset.port.onmessage({ data: { kind: "echo_guard", mode: "strict" } });
routeReset._ingest(new Float32Array(1000).fill(0.25), new Float32Array(1000).fill(0.1));
routeReset.port.onmessage({ data: { kind: "echo_reset" } });
assert.equal(routeReset._microphoneResampler.inputCount, 0);
assert.equal(routeReset._referenceResampler.inputCount, 0);
assert.equal(routeReset._chunkWrite, 0);
assert.equal(routeReset._referenceInputEnergy, 0);
assert.equal(routeReset._referenceInputCount, 0);
assert.equal(routeReset._referenceLevels.length, 0);
routeReset.messages.length = 0;
routeReset._ingest(new Float32Array(1978).fill(0.25), new Float32Array(1978));
const routeResetBuffers = pcmBuffers(routeReset);
assert.equal(routeResetBuffers.length, 1, "new-route speech must not inherit old-route echo state");
assert.equal(new Int16Array(routeResetBuffers[0]).length, 640);

function renderFallbackPcm(samples, partitions) {
  const processor = new fallbackProcessorClass({ processorOptions: { chunkMs: 40 } });
  let offset = 0;
  let partition = 0;
  while (offset < samples.length) {
    const size = Math.min(partitions[partition % partitions.length], samples.length - offset);
    processor._ingest(samples.subarray(offset, offset + size), new Float32Array(size));
    offset += size;
    partition += 1;
  }
  processor.port.onmessage({ data: { kind: "capture_flush" } });
  const chunks = pcmBuffers(processor).map((buffer) => new Int16Array(buffer));
  const output = new Int16Array(chunks.reduce((total, chunk) => total + chunk.length, 0));
  let write = 0;
  for (const chunk of chunks) {
    output.set(chunk, write);
    write += chunk.length;
  }
  return output;
}

const pcmFixture = tone(48000, 997, 0.25, 0.99);
const pcmOneShot = renderFallbackPcm(pcmFixture, [pcmFixture.length]);
const pcmFragmented = renderFallbackPcm(pcmFixture, [1, 7, 128, 13, 257, 64]);
assert.deepEqual(pcmFragmented, pcmOneShot, "PCM16 output must be invariant to render partitions");
assert.equal(pcmOneShot.length, Math.ceil(pcmFixture.length / 3));
assert.ok(
  [...pcmOneShot].every((sample) => sample > -32768 && sample < 32767),
  "a 0.99 speech-band tone must not clip",
);

let aecProcessorClass = null;
globalThis.registerProcessor = (_name, implementation) => {
  aecProcessorClass = implementation;
};
await import(`../web/hf-realtime-voice/worklets/aec3/aec3-capture.js?fidelity=${Date.now()}`);
assert.ok(aecProcessorClass, "AEC3 capture worklet registered");
const aecStereo = new aecProcessorClass({ processorOptions: { chunkMs: 40 } });
aecStereo.process([
  [new Float32Array(1920).fill(0.8), new Float32Array(1920).fill(0.2)],
  [new Float32Array(1920)],
]);
const aecStereoInitialBuffers = pcmBuffers(aecStereo);
assert.equal(aecStereoInitialBuffers.length, 1);
assert.equal(
  new Int16Array(aecStereoInitialBuffers[0]).length,
  620,
  "the 40 ms AEC boundary publishes only FIR-ready samples without padding",
);
aecStereo.port.onmessage({ data: { kind: "capture_flush" } });
const aecStereoBuffers = pcmBuffers(aecStereo).map((buffer) => new Int16Array(buffer));
assert.deepEqual(
  aecStereoBuffers.map((buffer) => buffer.length),
  [620, 20],
  "explicit clean flush completes the exact 640-sample AEC duration",
);
const aecStereoPcm = new Int16Array(640);
aecStereoPcm.set(aecStereoBuffers[0]);
aecStereoPcm.set(aecStereoBuffers[1], aecStereoBuffers[0].length);
assert.equal(aecStereoPcm.length, 640);
assert.ok(
  [...aecStereoPcm.subarray(100, 540)].every((sample) => sample >= 16382 && sample <= 16384),
  "post-AEC native capture must share the multichannel mono/resampler path",
);

const aecRouteReset = new aecProcessorClass({ processorOptions: { chunkMs: 40 } });
aecRouteReset.process([
  [new Float32Array(1000).fill(0.25)],
  [new Float32Array(1000)],
]);
aecRouteReset.port.onmessage({ data: { kind: "echo_reset" } });
assert.equal(
  new Int16Array(pcmBuffers(aecRouteReset)[0]).length,
  Math.ceil(1000 / 3),
  "AEC route reset resolves the old route's sub-frame tail exactly once",
);
assert.equal(aecRouteReset._framesSinceChunk, 0);
aecRouteReset.messages.length = 0;
aecRouteReset.process([
  [new Float32Array(960).fill(0.25)],
  [new Float32Array(960)],
]);
assert.equal(
  pcmBuffers(aecRouteReset).length,
  0,
  "new AEC route must start a fresh four-frame transport boundary",
);
aecRouteReset.process([
  [new Float32Array(960).fill(0.25)],
  [new Float32Array(960)],
]);
assert.equal(new Int16Array(pcmBuffers(aecRouteReset)[0]).length, 620);

const aecChunkGate = new aecProcessorClass({ processorOptions: { chunkMs: 40 } });
aecChunkGate.port.onmessage({ data: { kind: "gate", enabled: true, thresholdDb: -30 } });
for (let frame = 0; frame < 4; frame += 1) {
  aecChunkGate.process([[new Float32Array(480)], [new Float32Array(480)]]);
}
for (let frame = 0; frame < 3; frame += 1) {
  aecChunkGate.process([[new Float32Array(480).fill(0.25)], [new Float32Array(480)]]);
}
aecChunkGate.process([[new Float32Array(480)], [new Float32Array(480)]]);
assert.ok(
  aecChunkGate._gateGain > 0.95,
  "AEC gate must classify a full resampled chunk from all 40 ms, not its quiet final frame",
);

const aecPartial = new aecProcessorClass({ processorOptions: { chunkMs: 40 } });
aecPartial.process([
  [new Float32Array(333).fill(0.25)],
  [new Float32Array(333)],
]);
assert.equal(pcmBuffers(aecPartial).length, 0, "sub-frame AEC input waits for clean endpoint flush");
aecPartial.port.onmessage({ data: { kind: "capture_flush" } });
const aecPartialBuffers = pcmBuffers(aecPartial);
assert.equal(aecPartialBuffers.length, 1, "clean endpoint flush emits a partial AEC frame once");
assert.equal(
  new Int16Array(aecPartialBuffers[0]).length,
  Math.ceil((333 * OUTPUT_RATE) / 48000),
  "AEC endpoint flush preserves exact valid input duration",
);
const aecFlushAcks = aecPartial.messages.filter((value) => value?.kind === "capture_flushed");
assert.equal(aecFlushAcks.at(-1)?.sampleCount, 111);
aecPartial.port.onmessage({ data: { kind: "capture_flush" } });
assert.equal(pcmBuffers(aecPartial).length, 1, "second AEC flush is idempotent");
assert.equal(
  aecPartial.messages.filter((value) => value?.kind === "capture_flushed").at(-1)?.sampleCount,
  0,
);

const strictPartial = new aecProcessorClass({ processorOptions: { chunkMs: 40 } });
strictPartial._moduleReady = true;
strictPartial._effectiveMode = "strict";
strictPartial._session = {
  process: (_reference, capture) => ({
    output: capture,
    metrics: {
      residualEchoLikelihood: 0.2,
      captureRms: 0.25,
      outputRms: 0.1,
      doubleTalk: false,
    },
  }),
};
strictPartial.process([
  [new Float32Array(333).fill(0.25)],
  [new Float32Array(333).fill(0.1)],
]);
strictPartial.port.onmessage({ data: { kind: "capture_flush" } });
const strictPartialBuffers = pcmBuffers(strictPartial);
assert.equal(
  strictPartialBuffers.length,
  1,
  "Strict clean endpoint must resolve an uncertain final frame instead of dropping it",
);
assert.equal(
  new Int16Array(strictPartialBuffers[0]).length,
  111,
  "Strict endpoint resolution must exclude the partial AEC frame's zero padding",
);
assert.ok(
  new Int16Array(strictPartialBuffers[0]).some((sample) => sample !== 0),
  "Strict endpoint must retain uncertain near-end PCM",
);
assert.equal(strictPartial._strictGate.pending.length, 0);
assert.equal(
  strictPartial.messages.filter((value) => value?.kind === "capture_flushed").at(-1)?.sampleCount,
  111,
);
strictPartial.port.onmessage({ data: { kind: "capture_flush" } });
assert.equal(pcmBuffers(strictPartial).length, 1, "second Strict flush must not delay more PCM");
assert.equal(strictPartial._strictGate.pending.length, 0);
assert.equal(
  strictPartial.messages.filter((value) => value?.kind === "capture_flushed").at(-1)?.sampleCount,
  0,
);

const strictEchoOnlyPartial = new aecProcessorClass({ processorOptions: { chunkMs: 40 } });
strictEchoOnlyPartial._moduleReady = true;
strictEchoOnlyPartial._effectiveMode = "strict";
strictEchoOnlyPartial._session = {
  process: (_reference, capture) => ({
    output: capture,
    metrics: {
      residualEchoLikelihood: 0.9,
      captureRms: 0.25,
      outputRms: 0.01,
      doubleTalk: false,
    },
  }),
};
strictEchoOnlyPartial.process([
  [new Float32Array(333).fill(0.25)],
  [new Float32Array(333).fill(0.1)],
]);
strictEchoOnlyPartial.port.onmessage({ data: { kind: "capture_flush" } });
assert.equal(
  pcmBuffers(strictEchoOnlyPartial).length,
  0,
  "Strict clean endpoint must still omit a high-confidence echo-only final frame",
);
assert.equal(
  strictEchoOnlyPartial.messages.filter((value) => value?.kind === "capture_flushed").at(-1)?.sampleCount,
  0,
);

const fallbackPartial = new fallbackProcessorClass({ processorOptions: { chunkMs: 40 } });
fallbackPartial._ingest(new Float32Array(333).fill(0.25), new Float32Array(333).fill(0.1));
fallbackPartial.port.onmessage({ data: { kind: "capture_flush" } });
assert.equal(new Int16Array(pcmBuffers(fallbackPartial)[0]).length, 111);
assert.equal(fallbackPartial._referenceLevels.length, 0, "clean flush consumes its reference level");
assert.equal(fallbackPartial._referenceInputCount, 0, "clean flush resets reference accounting");

const fallbackFullFlush = new fallbackProcessorClass({ processorOptions: { chunkMs: 40 } });
fallbackFullFlush._ingest(new Float32Array(1920).fill(0.25), new Float32Array(1920));
assert.equal(pcmBuffers(fallbackFullFlush).length, 0);
fallbackFullFlush.port.onmessage({ data: { kind: "capture_flush" } });
assert.equal(new Int16Array(pcmBuffers(fallbackFullFlush)[0]).length, 640);
assert.equal(
  fallbackFullFlush.messages.filter((value) => value?.kind === "capture_flushed").at(-1)?.sampleCount,
  640,
  "fallback flush acknowledgement must count a full chunk completed inside FIR drain",
);

const abortedFallback = new fallbackProcessorClass({ processorOptions: { chunkMs: 40 } });
abortedFallback._ingest(new Float32Array(333).fill(0.25), new Float32Array(333));
abortedFallback.port.onmessage({ data: { kind: "capture_abort" } });
assert.equal(pcmBuffers(abortedFallback).length, 0, "abort discards fallback tail without emitting PCM");
assert.equal(abortedFallback._referenceLevels.length, 0, "abort discards queued reference levels");
const abortedAec = new aecProcessorClass({ processorOptions: { chunkMs: 40 } });
abortedAec.process([[new Float32Array(333).fill(0.25)], [new Float32Array(333)]]);
abortedAec.port.onmessage({ data: { kind: "capture_abort" } });
assert.equal(pcmBuffers(abortedAec).length, 0, "abort discards partial AEC input without emitting PCM");

console.log(JSON.stringify({
  version: 1,
  legacy_alias_rms: legacyAliasRms,
  polyphase_alias_rms: polyphaseAliasRms,
  alias_improvement_db: aliasImprovementDb,
  speech_gain_db: speechGainDb,
  upper_speech_gain_db: upperSpeechGainDb,
  passband_edge_gain_db: passbandEdgeGainDb,
  stopband_edge_attenuation_db: stopbandAttenuationDb,
  legacy_alias_44100_rms: legacyAlias44100,
  polyphase_alias_44100_rms: polyphaseAlias44100,
  passband_edge_44100_db: passbandEdge44100Db,
  stopband_edge_44100_attenuation_db: stopbandAttenuation44100Db,
  chunk_invariant_rates: [44100, 48000],
  mono_channels: 2,
  endpoint_tail_samples: tail.length,
  pcm16_clip: [-32768, 32767],
}));
