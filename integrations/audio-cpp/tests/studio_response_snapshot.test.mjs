import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const uiPath = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../gradio_voice_studio.py");
const source = fs.readFileSync(uiPath, "utf8");
const script = source.match(/<script type="text\/plain"[^>]*>([\s\S]*?)<\/script>/)?.[1];
assert.ok(script, "Voice Studio must retain an embedded browser boot script");
const start = script.indexOf("  function randomUint32()");
const end = script.indexOf("  function drainImmediateAndCappedPhrases(");
assert.ok(start >= 0 && end > start, "response snapshot helpers must be present in the active widget");
const helpers = script.slice(start, end);

const cloneHashes = {
  jarvis: "a".repeat(64),
  next: "b".repeat(64),
};
const fetches = [];
const state = {
  turnId: 7,
  responseSnapshot: null,
  responseSnapshotPromise: null,
  responseSnapshotTurn: null,
  isProcessing: true,
  isSpeaking: false,
  activeLlmRequest: null,
  activeTtsRequest: null,
  phrasePumping: false,
  pendingNextResponseConfig: null,
};
const config = {
  proxyBaseUrl: "/api/audio-cpp",
  ttsVoice: "clone:jarvis",
  tuningProfileId: "low-latency",
  language: "English",
  nativeStreaming: true,
  nativeStreamingAvailable: true,
  streamingMode: "native-incremental-pcm",
  tuning: { overrides: {} },
};
function profile(profileId) {
  const isNext = profileId === "next";
  return {
    id: isNext ? "quality" : "low-latency",
    revision: isNext ? 9 : 4,
    model: "Qwen3-TTS-1.7B-Base",
    clone_mode: true,
    max_reference_seconds: null,
    first_block_frames: 8,
    steady_block_frames: 16,
    left_context_frames: 72,
    text_lookahead: 48,
    phrase_flush_ms: 120,
    temperature: 1,
    top_k: 50,
    top_p: 0.9,
    repetition_penalty: 1.05,
    seed: isNext ? 424242 : null,
  };
}
async function fakeFetch(url, options = {}) {
  fetches.push({ url, options });
  if (url.endsWith("/control/status")) {
    return { ok: true, json: async () => ({
      state: "loaded", activeModel: "Qwen3-TTS-1.7B-Base", engineEpoch: 17,
      supervisorInstanceId: "0123456789abcdef0123456789abcdef",
    }) };
  }
  const cloneMatch = url.match(/\/v1\/voices\/profiles\/([^/]+)$/);
  if (cloneMatch) {
    const profileId = decodeURIComponent(cloneMatch[1]);
    return { ok: true, json: async () => ({
      profile_id: profileId, content_hash: cloneHashes[profileId], content_revision: profileId === "next" ? 3 : 2,
      ref_audio: `data:audio/wav;base64,${profileId}`, ref_text: `reference ${profileId}`,
    }) };
  }
  if (url.endsWith("/tuning/resolve")) {
    const cloneId = String(config.ttsVoice).replace(/^clone:/, "");
    return { ok: true, json: async () => ({ profile: profile(cloneId), temporaryOverrides: {} }) };
  }
  throw new Error(`unexpected snapshot route: ${url}`);
}

const statuses = [];
const refs = { "voice-name": { textContent: "jarvis" }, "tts-mode": { textContent: "Native PCM (experimental)" } };
const factory = new Function("state", "config", "fetch", "window", "DOMException", "refs", "setStatus", `${helpers}\nreturn { freezeResponseSnapshot, ttsPayloadForPhrase, phrasePolicyForResponse, queueNextResponseConfig, applyNextResponseConfig };`);
const { freezeResponseSnapshot, ttsPayloadForPhrase, phrasePolicyForResponse, queueNextResponseConfig, applyNextResponseConfig } = factory(
  state,
  config,
  fakeFetch,
  { crypto: { getRandomValues: (values) => { values[0] = 0x89abcdef; return values; } }, addEventListener: () => {} },
  DOMException,
  refs,
  (message) => statuses.push(message),
);

const firstSnapshot = await freezeResponseSnapshot(7);
const phraseOne = ttsPayloadForPhrase("First stable sentence.", firstSnapshot);
// A control update while the answer is active must not replace the widget,
// worklet, active request, or immutable snapshot. It queues a next-turn
// configuration instead.
assert.equal(queueNextResponseConfig({
  profileId: "next", tuningProfileId: "quality", playbackMode: "Buffered phrase PCM",
  sessionTuning: { overrides: { text_lookahead: 48 } },
}), false);
assert.equal(config.ttsVoice, "clone:jarvis");
assert.equal(config.tuningProfileId, "low-latency");
assert.equal(firstSnapshot.cloneId, "jarvis");
assert.ok(statuses.at(-1).includes("queued for the next response"));
const phraseTwo = ttsPayloadForPhrase("Second stable sentence.", firstSnapshot);

assert.notEqual(phraseOne.input, phraseTwo.input);
for (const key of ["model", "voice", "language", "stream", "expected_engine_epoch", "expected_supervisor_instance_id"]) {
  assert.equal(phraseOne[key], phraseTwo[key], `all phrases must reuse frozen ${key}`);
}
assert.equal(phraseOne.clone_snapshot.content_hash, phraseTwo.clone_snapshot.content_hash);
assert.equal(phraseOne.clone_snapshot.content_hash, cloneHashes.jarvis);
assert.equal(phraseOne.tuning.profile_revision, phraseTwo.tuning.profile_revision);
assert.equal(phraseOne.tuning.profile_revision, 4);
assert.equal(phraseOne.tuning.effective.seed, phraseTwo.tuning.effective.seed);
assert.equal(phraseOne.tuning.effective.seed, 0x89abcdef);
assert.equal(firstSnapshot.seed, 0x89abcdef);
assert.deepEqual(firstSnapshot.phrasePolicy, {
  textLookahead: 48,
  phraseFlushMs: 120,
  phraseHardCap: 192,
});
// A profile/widget edit while this answer is still speaking must affect only
// the next answer, not the phrase cuts or idle deadline already frozen here.
config.textLookahead = 512;
config.phraseFlushMs = 3000;
config.phraseHardCap = 512;
assert.deepEqual(phrasePolicyForResponse(), firstSnapshot.phrasePolicy);
assert.equal(fetches.length, 3, "one answer freezes the snapshot once before every phrase request");
assert.deepEqual(fetches.map(({ url }) => url), [
  "/api/audio-cpp/control/status",
  "/api/audio-cpp/v1/voices/profiles/jarvis",
  "/api/audio-cpp/tuning/resolve",
]);
for (const { options } of fetches) assert.equal(options.cache, "no-store");
assert.equal(fetches[2].options.method, "POST");
assert.deepEqual(JSON.parse(fetches[2].options.body), {
  provider: "qwen3tts-audiocpp", scope: "voice-studio", profile_id: "low-latency", overrides: {},
});

// A later answer freezes the new live controls rather than mutating the old
// snapshot. Its explicit profile seed takes precedence over generated seed.
state.turnId = 8;
state.responseSnapshot = null;
state.responseSnapshotPromise = null;
state.responseSnapshotTurn = null;
state.isProcessing = false;
assert.equal(applyNextResponseConfig(), true);
assert.equal(config.ttsVoice, "clone:next");
assert.equal(config.tuningProfileId, "quality");
assert.equal(config.nativeStreaming, false);
assert.equal(refs["voice-name"].textContent, "next");
const nextSnapshot = await freezeResponseSnapshot(8);
const nextPhrase = ttsPayloadForPhrase("Next answer.", nextSnapshot);
assert.equal(nextPhrase.voice, "clone:next");
assert.equal(nextPhrase.language, "English");
assert.equal(nextPhrase.stream, false);
assert.equal(nextPhrase.clone_snapshot.content_hash, cloneHashes.next);
assert.equal(nextPhrase.tuning.profile_revision, 9);
assert.equal(nextPhrase.tuning.effective.seed, 424242);

console.log("studio response snapshot payload tests passed");

const failureStart = script.indexOf("  function terminatePhrasesAfterProviderFailure(");
const failureEnd = script.indexOf("  function synthesizeSpeech(", failureStart);
assert.ok(failureStart >= 0 && failureEnd > failureStart, "phrase-provider failure terminal helper must be present");
const failureHelper = script.slice(failureStart, failureEnd);
const failureState = {
  turnId: 21,
  ttsFailureTerminal: false,
  phraseQueue: ["later phrase"],
  phraseText: "unfinished text",
  finalPhrasePending: true,
  llmFinished: false,
  phraseIdleTimer: null,
  activeLlmRequest: { aborted: false, abort() { this.aborted = true; } },
};
const failureStatuses = [];
const finalized = [];
const failureFactory = new Function(
  "state", "setStatus", "finishPlaybackTurn", "setSpeaking", "setProcessing", "clearTimeout",
  `${failureHelper}\nreturn terminatePhrasesAfterProviderFailure;`,
);
const terminatePhrasesAfterProviderFailure = failureFactory(
  failureState,
  (message) => failureStatuses.push(message),
  (turnId, snapshot) => { finalized.push({ turnId, snapshot }); return Promise.resolve(); },
  () => {},
  () => {},
  () => {},
);
const failureSnapshot = { nativeStreaming: false };
assert.equal(
  terminatePhrasesAfterProviderFailure(21, new Error("upstream lost"), failureSnapshot, false),
  true,
  "the first provider failure terminates the current response",
);
assert.equal(failureState.ttsFailureTerminal, true);
assert.deepEqual(failureState.phraseQueue, [], "queued later phrases are discarded");
assert.equal(failureState.phraseText, "", "uncommitted text cannot become a new phrase");
assert.equal(failureState.finalPhrasePending, false);
assert.equal(failureState.llmFinished, true);
assert.equal(failureState.activeLlmRequest, null, "the producing LLM is cancelled after terminal provider failure");
assert.equal(finalized.length, 1, "accepted PCM is still given one terminal worklet end");
assert.ok(failureStatuses.at(-1).includes("Remaining phrases were suppressed"));
assert.equal(
  terminatePhrasesAfterProviderFailure(21, new Error("duplicate"), failureSnapshot, false),
  false,
  "a terminal response cannot retry or dispatch another phrase after a provider failure",
);
console.log("studio provider failure terminal tests passed");

const outcomeStart = script.indexOf("  function lookupNativeAudioOutcome(");
const outcomeEnd = script.indexOf("  function terminatePhrasesAfterProviderFailure(", outcomeStart);
assert.ok(outcomeStart >= 0 && outcomeEnd > outcomeStart, "native outcome lookup must be present");
const outcomeHelper = script.slice(outcomeStart, outcomeEnd);
const outcomeFetches = [];
const lookupNativeAudioOutcome = new Function("config", "fetch", `${outcomeHelper}\nreturn lookupNativeAudioOutcome;`)(
  config,
  async (url, options) => {
    outcomeFetches.push({ url, options });
    return { ok: true, json: async () => ({ requestId: "0123456789abcdef0123456789abcdef", state: "cancelled", reason: "request superseded" }) };
  },
);
assert.equal(
  await lookupNativeAudioOutcome("0123456789abcdef0123456789abcdef"),
  "Candidate outcome cancelled: request superseded",
);
assert.deepEqual(outcomeFetches, [{
  url: "/api/audio-cpp/audio/outcomes/0123456789abcdef0123456789abcdef",
  options: { cache: "no-store" },
}]);
assert.equal(await lookupNativeAudioOutcome("not-an-outcome"), "");
assert.match(await lookupNativeAudioOutcome("not-an-outcome", true), /could not be verified/);
assert.equal(outcomeFetches.length, 1, "invalid IDs must not become a relay request");
const progressiveDispatch = script.slice(
  script.indexOf("  function pumpProgressiveSpeech("),
  script.indexOf("  function lookupNativeAudioOutcome("),
);
assert.ok(progressiveDispatch.includes("lookupNativeAudioOutcome(outcomeRequestId)"));
assert.ok(progressiveDispatch.includes("responseSnapshot, true, true"));
assert.ok(!progressiveDispatch.includes("buffered_phrase"), "late native failure must not switch delivery mode");
console.log("studio native outcome failure tests passed");

// Exercise the actual phrase pump: an asynchronous outcome lookup must never
// release the next phrase after a transport error or flush the failed tail.
for (const nativeStreaming of [true, false]) {
  let settleLookup;
  let calls = 0;
  let cleared = 0;
  let finalizedCount = 0;
  const pendingLookup = new Promise(resolve => { settleLookup = resolve; });
  const pumpState = {
    turnId: 33, responseSnapshotTurn: 33, responseSnapshot: { nativeStreaming },
    phraseQueue: ['first sentence', 'must never dispatch'], phraseText: 'partial',
    phrasePumping: false, phraseIndex: 0, pcmChunks: [new Uint8Array([0, 0])],
    playbackReadyPromise: Promise.resolve(), ttsStartedAt: 1, ttsFirstPcmMs: null,
    activeLlmRequest: { abort() {} }, llmFinished: false,
  };
  const pumpFactory = new Function(
    'state', 'config', 'refs', 'fetch', 'ttsPayloadForPhrase', 'lookupNativeAudioOutcome',
    'cancelScheduledPlayback', 'setStatus', 'setSpeaking', 'setProcessing', 'finishPlaybackTurn',
    `${failureHelper}\n${progressiveDispatch}\nreturn pumpProgressiveSpeech;`,
  );
  const pump = pumpFactory(
    pumpState, config, { tts: {} },
    async () => { calls++; throw new Error('transport closed'); },
    text => ({ input: text }), () => pendingLookup,
    () => { cleared++; }, () => {}, () => {}, () => {},
    () => { finalizedCount++; return Promise.resolve(); },
  );
  pump();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(pumpState.ttsFailureTerminal, true);
  assert.equal(cleared, 1, 'the exact response queue is cleared before awaiting diagnostics');
  assert.equal(calls, 1);
  assert.equal(finalizedCount, 0, 'failed PCM must never be ended/drained as successful');
  pump();
  assert.equal(calls, 1, 'later LLM phrases cannot restart this failed response');
  settleLookup('limited');
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(pumpState.phrasePumping, false);
  assert.equal(pumpState.phraseIndex, 1);
  assert.equal(calls, 1);
  assert.deepEqual(pumpState.phraseQueue, []);
}
console.log('studio real phrase-pump asynchronous failure isolation passed');
