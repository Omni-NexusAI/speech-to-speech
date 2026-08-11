import assert from "node:assert/strict";

if (typeof globalThis.CustomEvent === "undefined") {
  globalThis.CustomEvent = class CustomEvent extends Event {
    constructor(type, init = {}) {
      super(type);
      this.detail = init.detail;
    }
  };
}
globalThis.WebSocket = { OPEN: 1 };

const {
  AdaptivePlaybackPolicyStore,
  AUDIO_CPP_PLAYBACK_PRIME_MS,
  MAX_PLAYBACK_LEARNING_SIGNATURES,
  MAX_PLAYBACK_PRIME_MS,
  PIPELINE_CONFIG_ACK_TIMEOUT_MS,
  PLAYBACK_GAP_WINDOW,
  PLAYBACK_LEARNING_TTL_MS,
  S2sWsRealtimeClient,
  resolveAdaptivePlaybackSignature,
  resolvePlaybackPrimeMs,
} = await import(
  "../web/hf-realtime-voice/ws/s2s-ws-client.js"
);

assert.equal(PIPELINE_CONFIG_ACK_TIMEOUT_MS, 15_000);
assert.equal(AUDIO_CPP_PLAYBACK_PRIME_MS["low-latency"], 800);
assert.equal(AUDIO_CPP_PLAYBACK_PRIME_MS.balanced, 1280);
assert.equal(AUDIO_CPP_PLAYBACK_PRIME_MS.quality, 1760);
assert.equal(MAX_PLAYBACK_PRIME_MS, 2000);
for (const [profileId, expected] of [["low-latency", 800], ["balanced", 1280], ["quality", 1760]]) {
  assert.equal(resolvePlaybackPrimeMs(
    { tts_backend: "qwen3tts-audiocpp", tts_tuning: { profile_id: profileId } },
    { nativeStreaming: true, profileId },
  ), expected);
}
assert.equal(resolvePlaybackPrimeMs(
  { tts_backend: "qwen3tts-audiocpp", tts_tuning: { profile_id: "custom-studio" } },
  { nativeStreaming: true, profileId: "custom-studio", resolvedPrimeMs: 5000 },
), 2000, "custom resolved priming is capped");
assert.equal(resolvePlaybackPrimeMs(
  { tts_backend: "qwen3tts-audiocpp", tts_tuning: { profile_id: "quality" } },
  { nativeStreaming: false, profileId: "quality" },
), 0, "buffered audio.cpp retains immediate playback");
assert.equal(resolvePlaybackPrimeMs(
  { tts_backend: "faster" },
  { nativeStreaming: true, profileId: "quality" },
), 0, "Faster never inherits candidate priming");

class MemoryStorage {
  constructor() {
    this.values = new Map();
  }

  getItem(key) {
    return this.values.get(key) ?? null;
  }

  setItem(key, value) {
    this.values.set(key, String(value));
  }
}

const adaptiveConfig = {
  tts_backend: "qwen3tts-audiocpp",
  tts_tuning: {
    profile_id: "balanced",
    overrides: { first_block_frames: 4, steady_block_frames: 12 },
  },
};
const adaptiveHint = {
  provider: "qwen3tts-audiocpp",
  profileId: "balanced",
  profileRevision: 7,
  model: "qwen3-tts-1.7b-base-bf16",
  clone: "clone:test",
  nativeStreaming: true,
  firstBlockFrames: 4,
  steadyBlockFrames: 12,
  outputRate: 16000,
};
const adaptiveSignature = resolveAdaptivePlaybackSignature(
  adaptiveConfig,
  adaptiveHint,
  "runtime-v1",
  true,
);
assert.equal(adaptiveSignature.valid, true);
assert.equal(adaptiveSignature.firstMs, 320);
assert.equal(adaptiveSignature.ceilingMs, 1280);
assert.equal(resolveAdaptivePlaybackSignature(
  adaptiveConfig,
  { ...adaptiveHint, profileRevision: "" },
  "runtime-v1",
  true,
).valid, false, "a missing immutable signature component stays conservative");
assert.equal(resolveAdaptivePlaybackSignature(
  adaptiveConfig,
  adaptiveHint,
  "runtime-v1",
  false,
).valid, false, "a mismatched or reordered acknowledgement cannot enable learning");

const invalidBuiltinSignature = resolveAdaptivePlaybackSignature(
  {
    tts_backend: "qwen3tts-audiocpp",
    tts_tuning: { profile_id: "balanced", overrides: {} },
  },
  { ...adaptiveHint, steadyBlockFrames: 0, resolvedPrimeMs: 320 },
  "runtime-v1",
  true,
);
assert.equal(invalidBuiltinSignature.valid, false);
assert.equal(
  invalidBuiltinSignature.ceilingMs,
  1280,
  "an invalid built-in signature uses its named full two-block ceiling",
);
const invalidCustomSignature = resolveAdaptivePlaybackSignature(
  {
    tts_backend: "qwen3tts-audiocpp",
    tts_tuning: { profile_id: "custom-studio", overrides: { first_block_frames: 4 } },
  },
  {
    ...adaptiveHint,
    profileId: "custom-studio",
    steadyBlockFrames: 0,
    resolvedPrimeMs: 320,
  },
  "runtime-v1",
  true,
);
assert.equal(invalidCustomSignature.valid, false);
assert.equal(
  invalidCustomSignature.ceilingMs,
  2000,
  "an invalid custom signature never mistakes first-block duration for its safe ceiling",
);
assert.equal(
  new AdaptivePlaybackPolicyStore({ storage: new MemoryStorage() }).policy(invalidCustomSignature).targetMs,
  2000,
);

// Persisted flags are untrusted. The loader derives eligibility only from two
// clean responses and an enabled record, so stale or forged booleans cannot
// shorten startup.
{
  const storage = new MemoryStorage();
  const seedStore = new AdaptivePlaybackPolicyStore({ storage, now: () => 1000 });
  seedStore.policy(adaptiveSignature);
  const storageKey = [...storage.values.keys()][0];
  const payload = JSON.parse(storage.getItem(storageKey));
  payload.entries[0].warmEligible = true;
  payload.entries[0].fullPrimeCleanCount = 0;
  storage.setItem(storageKey, JSON.stringify(payload));
  assert.equal(
    new AdaptivePlaybackPolicyStore({ storage, now: () => 1000 }).policy(adaptiveSignature).mode,
    "cold",
  );

  payload.entries[0].warmEligible = false;
  payload.entries[0].fullPrimeCleanCount = 2;
  payload.entries[0].disabled = false;
  storage.setItem(storageKey, JSON.stringify(payload));
  assert.equal(
    new AdaptivePlaybackPolicyStore({ storage, now: () => 1000 }).policy(adaptiveSignature).mode,
    "warm",
  );

  payload.entries[0].warmEligible = true;
  payload.entries[0].disabled = true;
  storage.setItem(storageKey, JSON.stringify(payload));
  assert.equal(
    new AdaptivePlaybackPolicyStore({ storage, now: () => 1000 }).policy(adaptiveSignature).mode,
    "recovery",
  );
}

// Two clean full-prime responses establish evidence. Warm startup uses the
// first-block and observed-gap margins, then a single underrun forces the full
// ceiling until three clean recovery responses complete.
{
  const storage = new MemoryStorage();
  let now = 1000;
  const store = new AdaptivePlaybackPolicyStore({ storage, now: () => now });
  let policy = store.policy(adaptiveSignature);
  assert.equal(policy.mode, "cold");
  assert.equal(policy.targetMs, 1280);
  store.record(policy, { gaps: [180, 200], cleanFullPrime: true });
  policy = store.policy(adaptiveSignature);
  assert.equal(policy.mode, "cold", "one full-prime response is not warm evidence");
  store.record(policy, { gaps: [220, 240], cleanFullPrime: true });
  policy = store.policy(adaptiveSignature);
  assert.equal(policy.mode, "warm");
  assert.equal(policy.targetMs, 480, "warm target is max(first+160, p95+64)");

  store.record(policy, { gaps: [600], underrun: true });
  policy = store.policy(adaptiveSignature);
  assert.equal(policy.mode, "recovery");
  assert.equal(policy.targetMs, 1280);
  for (let index = 0; index < 2; index += 1) {
    store.record(policy, { cleanFullPrime: true });
    policy = store.policy(adaptiveSignature);
    assert.equal(policy.mode, "recovery");
  }
  store.record(policy, { cleanFullPrime: true });
  assert.equal(store.policy(adaptiveSignature).mode, "warm");

  const persisted = JSON.parse([...storage.values.values()][0]);
  assert.ok(persisted.entries[0].gaps.length <= PLAYBACK_GAP_WINDOW);
  assert.equal(JSON.stringify(persisted).includes("transcript"), false);
  now += PLAYBACK_LEARNING_TTL_MS + 1;
  const expiredStore = new AdaptivePlaybackPolicyStore({ storage, now: () => now });
  assert.equal(expiredStore.policy(adaptiveSignature).mode, "cold", "expired evidence is ignored");
}

// Persistence is an LRU-bounded optimization, never an unbounded per-clone log.
{
  const storage = new MemoryStorage();
  let now = 10;
  const store = new AdaptivePlaybackPolicyStore({ storage, now: () => now++ });
  for (let index = 0; index < MAX_PLAYBACK_LEARNING_SIGNATURES + 4; index += 1) {
    const signature = resolveAdaptivePlaybackSignature(
      adaptiveConfig,
      { ...adaptiveHint, model: `model-${index}` },
      "runtime-v1",
      true,
    );
    store.policy(signature);
  }
  const persisted = JSON.parse([...storage.values.values()][0]);
  assert.equal(persisted.entries.length, MAX_PLAYBACK_LEARNING_SIGNATURES);
}

function createClient() {
  const sent = [];
  const client = new S2sWsRealtimeClient({
    directUrl: "ws://127.0.0.1:8765/v1/realtime",
    voice: "clone:test",
    instructions: "test",
    pipelineConfig: {},
  });
  client._status = "connecting";
  client._ws = {
    readyState: WebSocket.OPEN,
    send: (value) => sent.push(JSON.parse(value)),
    close: () => {},
  };
  return { client, sent };
}

function pcm16Base64(sampleCount) {
  return Buffer.alloc(sampleCount * 2).toString("base64");
}

// The server acknowledgement is the atomic signature commit. A policy frozen
// at the user-turn boundary remains immutable across a profile update and a
// tool-result continuation; only the next accepted turn sees the new profile.
{
  const { client } = createClient();
  client._sessionConfigured = true;
  client._playbackLearning = new AdaptivePlaybackPolicyStore({ storage: new MemoryStorage() });
  await client._onWsMessage(JSON.stringify({
    type: "pipeline.runtime",
    runtime: { api_version: "3", started_at_utc: "2026-08-11T00:00:00Z", pid: 1234 },
  }));
  client.updateLocalPipeline(adaptiveConfig, adaptiveHint);
  await client._onWsMessage(JSON.stringify({
    type: "pipeline.config.updated",
    config: adaptiveConfig,
  }));
  assert.equal(client._acknowledgedPlaybackSignature.valid, true);
  assert.equal(client._playbackPrimeMs, 1280);

  client._freezePlaybackTurnPolicy();
  const origin = client._playbackSnapshot("response-origin");
  const qualityConfig = {
    tts_backend: "qwen3tts-audiocpp",
    tts_tuning: {
      profile_id: "quality",
      overrides: { first_block_frames: 6, steady_block_frames: 16 },
    },
  };
  const qualityHint = {
    ...adaptiveHint,
    profileId: "quality",
    profileRevision: 8,
    firstBlockFrames: 6,
    steadyBlockFrames: 16,
  };
  client.updateLocalPipeline(qualityConfig, qualityHint);
  await client._onWsMessage(JSON.stringify({
    type: "pipeline.config.updated",
    config: qualityConfig,
  }));
  const continuation = client._playbackSnapshot("response-tool-continuation");
  assert.equal(origin.turnId, continuation.turnId);
  assert.equal(continuation.primeMs, 1280, "tool continuation keeps the turn-frozen policy");
  client._freezePlaybackTurnPolicy();
  assert.equal(client._playbackSnapshot("response-next-turn").primeMs, 1760);
  await client._onWsMessage(JSON.stringify({
    type: "pipeline.runtime",
    runtime: { api_version: "3", started_at_utc: "2026-08-11T00:01:00Z", pid: 4321 },
  }));
  assert.equal(client._acknowledgedPlaybackSignature.valid, false, "runtime replacement resets evidence");
  client._freezePlaybackTurnPolicy();
  assert.equal(client._activePlaybackTurn.policy.mode, "conservative");
  await client.close();
}

// Exact PCM accounting is sent to the worklet, and missing/reordered response
// identities remain audible only under the conservative full ceiling.
{
  const { client } = createClient();
  client._sessionConfigured = true;
  const playbackMessages = [];
  client._playbackNode = {
    port: { postMessage: (value) => playbackMessages.push(value) },
    disconnect: () => {},
  };
  client._playbackRuntimeIdentity = "runtime-v1";
  client._acknowledgedPlaybackSignature = adaptiveSignature;
  client._playbackLearning = new AdaptivePlaybackPolicyStore({ storage: new MemoryStorage() });
  let policy = client._playbackLearning.policy(adaptiveSignature);
  client._playbackLearning.record(policy, { cleanFullPrime: true });
  policy = client._playbackLearning.policy(adaptiveSignature);
  client._playbackLearning.record(policy, { cleanFullPrime: true });
  client._freezePlaybackTurnPolicy();
  assert.equal(client._activePlaybackTurn.policy.mode, "warm");

  await client._onWsMessage(JSON.stringify({
    type: "response.created",
    response: { id: "response-accounted" },
  }));
  for (let index = 0; index < 2; index += 1) {
    await client._onWsMessage(JSON.stringify({
      type: "response.output_audio.delta",
      response_id: "response-accounted",
      delta: "AAA=",
    }));
  }
  await client._onWsMessage(JSON.stringify({
    type: "response.output_audio.done",
    response_id: "response-accounted",
  }));
  const audioMessages = playbackMessages.filter((message) => message.kind === "audio");
  assert.deepEqual(audioMessages.map((message) => message.inputSampleOffset), [0, 1]);
  assert.deepEqual(audioMessages.map((message) => message.inputSampleCount), [1, 1]);
  assert.equal(audioMessages[0].targetSamples, 7680, "warm target is encoded as exact 16 kHz samples");
  assert.equal(audioMessages[0].ceilingSamples, 20480);
  assert.equal(playbackMessages.find((message) => message.kind === "end").inputSamples, 2);

  client._pushAudioDelta("AAA=", "response-before-created");
  await client._onWsMessage(JSON.stringify({
    type: "response.created",
    response: { id: "response-before-created" },
  }));
  assert.equal(
    client._playbackSnapshot("response-before-created").primeMs,
    1280,
    "audio before response.created fails conservative and cannot be upgraded",
  );
  await client.close();
}

// Adaptive timing observes complete decoder blocks, not arbitrary WebSocket
// packet cadence. Packetized and coalesced delivery therefore produce the
// same logical gaps and the same learned warm target.
{
  const observeCadence = async (delivery) => {
    const { client } = createClient();
    client._sessionConfigured = true;
    client._playbackNode = { port: { postMessage: () => {} }, disconnect: () => {} };
    client._acknowledgedPlaybackSignature = adaptiveSignature;
    client._playbackLearning = new AdaptivePlaybackPolicyStore({ storage: new MemoryStorage() });
    client._freezePlaybackTurnPolicy();
    let now = 0;
    client._playbackClock = () => now;
    await client._onWsMessage(JSON.stringify({
      type: "response.created",
      response: { id: "response-cadence" },
    }));
    for (const [atMs, sampleCount] of delivery) {
      now = atMs;
      client._pushAudioDelta(pcm16Base64(sampleCount), "response-cadence");
    }
    const snapshot = client._playbackSnapshot("response-cadence");
    const observation = {
      gaps: [...snapshot.gaps],
      logicalBlockCount: snapshot.logicalBlockCount,
      latestLogicalBlockGapMs: snapshot.latestLogicalBlockGapMs,
    };
    await client.close();
    return observation;
  };

  const packetized = await observeCadence([
    [10, 2000],
    [40, 2000],
    [100, 1120],
    [200, 5000],
    [300, 5000],
    [500, 5360],
    [1100, 30720],
  ]);
  const blockAligned = await observeCadence([
    [100, 5120],
    [500, 15360],
    [1100, 30720],
  ]);
  assert.deepEqual(packetized.gaps, [400, 600]);
  assert.deepEqual(blockAligned.gaps, packetized.gaps);
  assert.equal(packetized.logicalBlockCount, 4);
  assert.equal(blockAligned.logicalBlockCount, 4);
  assert.equal(packetized.latestLogicalBlockGapMs, 600);

  const store = new AdaptivePlaybackPolicyStore({ storage: new MemoryStorage() });
  let policy = store.policy(adaptiveSignature);
  store.record(policy, { gaps: packetized.gaps, cleanFullPrime: true });
  policy = store.policy(adaptiveSignature);
  store.record(policy, { gaps: blockAligned.gaps, cleanFullPrime: true });
  policy = store.policy(adaptiveSignature);
  assert.equal(policy.mode, "warm");
  assert.equal(policy.gapP95Ms, 600);
  assert.equal(policy.targetMs, 664, "warm target uses logical block p95 plus the fixed jitter margin");
}

// Peers that omit response IDs receive distinct conservative identities rather
// than inheriting warm evidence or colliding with a prior tombstone.
{
  const { client } = createClient();
  client._sessionConfigured = true;
  client._playbackNode = { port: { postMessage: () => {} }, disconnect: () => {} };
  client._acknowledgedPlaybackSignature = adaptiveSignature;
  client._freezePlaybackTurnPolicy();
  await client._onWsMessage(JSON.stringify({ type: "response.created", response: {} }));
  const firstId = [...client._openPlaybackResponseIds][0];
  assert.equal(client._playbackSnapshot(firstId).primeMs, 1280);
  await client._onWsMessage(JSON.stringify({
    type: "response.done",
    response: { status: "completed", output: [] },
  }));
  await client._onWsMessage(JSON.stringify({ type: "response.created", response: {} }));
  const secondId = [...client._openPlaybackResponseIds][0];
  assert.notEqual(firstId, secondId);
  assert.equal(client._playbackSnapshot(secondId).primeMs, 1280);
  await client.close();
}

// A clear remains idempotent for stale tails, then accepted PCM from a new
// non-microphone response re-arms the next Stop. This covers replacement/tool
// responses that do not pass through input_audio_buffer.speech_stopped.
{
  const { client } = createClient();
  client._sessionConfigured = true;
  const playbackMessages = [];
  client._playbackNode = {
    port: { postMessage: (value) => playbackMessages.push(value) },
    disconnect: () => {},
  };
  client._freezePlaybackTurnPolicy();
  await client._onWsMessage(JSON.stringify({
    type: "response.created",
    response: { id: "response-old-generation" },
  }));
  client._pushAudioDelta("AAA=", "response-old-generation");
  client._invalidatePlayback("barge-in");
  assert.equal(client._playbackGenerationInvalidated, true);
  assert.equal(playbackMessages.filter((message) => message.kind === "clear").length, 1);

  client._pushAudioDelta("AAA=", "response-old-generation");
  assert.equal(
    client._playbackGenerationInvalidated,
    true,
    "an old-generation tail rejected before the worklet cannot re-arm clearing",
  );
  await client._onWsMessage(JSON.stringify({
    type: "response.done",
    response: { id: "response-old-generation", status: "cancelled", output: [] },
  }));
  assert.equal(playbackMessages.filter((message) => message.kind === "clear").length, 1);

  await client._onWsMessage(JSON.stringify({
    type: "response.created",
    response: { id: "response-non-mic-replacement" },
  }));
  client._pushAudioDelta("AAA=", "response-non-mic-replacement");
  assert.equal(
    client._playbackGenerationInvalidated,
    false,
    "accepted current-generation PCM re-arms Stop without a microphone boundary",
  );
  await client._onWsMessage(JSON.stringify({
    type: "response.done",
    response: { id: "response-old-generation", status: "cancelled", output: [] },
  }));
  assert.equal(playbackMessages.filter((message) => message.kind === "clear").length, 1);

  client._invalidatePlayback("stop");
  assert.equal(client._playbackGenerationInvalidated, true);
  assert.equal(playbackMessages.filter((message) => message.kind === "clear").length, 2);
  await client.close();
  assert.equal(
    playbackMessages.filter((message) => message.kind === "clear").length,
    2,
    "close remains idempotent after Stop",
  );
}

// Safe-start diagnostics expose only bounded timing/counter state. They never
// serialize the immutable signature or its model/clone identity.
{
  const { client } = createClient();
  client._sessionConfigured = true;
  const metrics = [];
  client.addEventListener("pipeline-metric", (event) => metrics.push(event.detail));
  client._playbackNode = { port: { postMessage: () => {} }, disconnect: () => {} };
  client._playbackRuntimeIdentity = "runtime-v1";
  client.updateLocalPipeline(adaptiveConfig, adaptiveHint);
  await client._onWsMessage(JSON.stringify({
    type: "pipeline.config.updated",
    config: adaptiveConfig,
  }));
  const configured = metrics.find((metric) => (
    metric.stage === "playback" && metric.status === "configured"
  ));
  assert.ok(configured);
  assert.equal(configured.detail.cold_ceiling_ms, 1280);
  assert.equal(configured.detail.effective_target_ms, 1280);
  assert.equal(configured.detail.safe_start_mode, "cold");
  assert.equal(configured.detail.logical_block_gap_p95_ms, 0);
  assert.equal(configured.detail.jitter_margin_ms, 64);
  assert.equal(configured.detail.fallback_reason, "cold_evidence");

  client._freezePlaybackTurnPolicy();
  await client._onWsMessage(JSON.stringify({
    type: "response.created",
    response: { id: "response-diagnostics" },
  }));
  let now = 100;
  client._playbackClock = () => now;
  client._pushAudioDelta(pcm16Base64(5120), "response-diagnostics");
  now = 500;
  client._pushAudioDelta(pcm16Base64(15360), "response-diagnostics");
  client._onPlaybackMessage({
    kind: "stats",
    generation: client._playbackGeneration,
    streamId: "response-diagnostics",
    queuedMs: 960,
    queuedSamples: 15360,
    primeTargetMs: 1280,
    primeCeilingMs: 1280,
    underruns: 2,
    reprimes: 1,
  });
  const queue = metrics.findLast((metric) => (
    metric.stage === "playback" && metric.status === "queue"
  ));
  assert.ok(queue);
  assert.equal(queue.detail.cold_ceiling_ms, 1280);
  assert.equal(queue.detail.effective_target_ms, 1280);
  assert.equal(queue.detail.safe_start_mode, "cold");
  assert.equal(queue.detail.latest_logical_block_gap_ms, 400);
  assert.equal(queue.detail.logical_block_gap_p95_ms, 400);
  assert.equal(queue.detail.jitter_margin_ms, 64);
  assert.equal(queue.detail.queued_ms, 960);
  assert.equal(queue.detail.underruns, 2);
  assert.equal(queue.detail.reprimes, 1);
  assert.equal(queue.detail.fallback_reason, "cold_evidence");
  const serializedMetrics = JSON.stringify(metrics);
  assert.equal(serializedMetrics.includes(adaptiveHint.model), false);
  assert.equal(serializedMetrics.includes(adaptiveHint.clone), false);
  assert.equal(serializedMetrics.includes(adaptiveSignature.key), false);
  await client.close();
}

{
  const { client, sent } = createClient();
  const ready = client._waitForInitialConfigAck();
  await client._onWsMessage(JSON.stringify({
    type: "pipeline.config.updated",
    config: { tts_backend: "qwen3tts-audiocpp" },
  }));
  await ready;
  assert.equal(client._sessionConfigured, true);
  assert.equal(client.status, "connected");
  assert.deepEqual(sent.map((event) => event.type), ["session.update"]);
  await client.close();
}

{
  const { client } = createClient();
  client._sessionConfigured = true;
  client.updateLocalPipeline(
    {
      tts_backend: "qwen3tts-audiocpp",
      tts_tuning: { profile_id: "low-latency", overrides: {} },
    },
    { provider: "qwen3tts-audiocpp", profileId: "low-latency", nativeStreaming: true },
  );
  assert.equal(client._playbackPrimeMs, 0, "requested config does not apply before acknowledgement");
  await client._onWsMessage(JSON.stringify({
    type: "pipeline.config.updated",
    config: {
      tts_backend: "qwen3tts-audiocpp",
      tts_tuning: { profile_id: "low-latency", overrides: {} },
    },
  }));
  assert.equal(client._playbackPrimeMs, 800);
  const queuedResponse = client._playbackSnapshot("response-before-update");

  client.updateLocalPipeline(
    {
      tts_backend: "qwen3tts-audiocpp",
      tts_tuning: { profile_id: "quality", overrides: {} },
    },
    { provider: "qwen3tts-audiocpp", profileId: "quality", nativeStreaming: true },
  );
  assert.equal(client._playbackPrimeMs, 800, "new target remains pending before ack");
  await client._onWsMessage(JSON.stringify({
    type: "pipeline.config.updated",
    config: {
      tts_backend: "qwen3tts-audiocpp",
      tts_tuning: { profile_id: "quality", overrides: {} },
    },
  }));
  assert.equal(client._playbackPrimeMs, 1760);
  assert.equal(queuedResponse.primeMs, 800, "queued response keeps its acknowledged snapshot");
  assert.equal(client._playbackSnapshot("response-after-update").primeMs, 1760);
  await client.close();
}

{
  const { client } = createClient();
  const ready = client._waitForInitialConfigAck();
  await client._onWsMessage(JSON.stringify({
    type: "error",
    error: {
      type: "invalid_tts_tuning",
      code: "invalid_tts_tuning",
      message: "tts_tuning contains unsupported fields",
    },
  }));
  await assert.rejects(ready, /unsupported fields/);
  assert.equal(client._closed, true);
  assert.equal(client.status, "closed");
}

{
  const { client } = createClient();
  const ready = client._waitForInitialConfigAck();
  await client.close();
  await assert.rejects(ready, /before pipeline configuration completed/);
}

{
  const nativeSetTimeout = globalThis.setTimeout;
  let requestedDelay = null;
  globalThis.setTimeout = (callback, delay) => {
    requestedDelay = delay;
    return nativeSetTimeout(callback, 0);
  };
  const { client } = createClient();
  const ready = client._waitForInitialConfigAck();
  await assert.rejects(ready, (error) => error?.code === "pipeline-config-timeout");
  assert.equal(requestedDelay, 15_000);
  globalThis.setTimeout = nativeSetTimeout;
  await client.close();
}

{
  const { client } = createClient();
  client._sessionConfigured = true;
  const playbackMessages = [];
  client._playbackNode = {
    port: { postMessage: (value) => playbackMessages.push(value) },
    disconnect: () => {},
  };
  client._playbackSnapshot("response-old");
  client._invalidatePlayback("barge-in");
  client._playbackGenerationInvalidated = false; // the replacement user turn stopped
  await client._onWsMessage(JSON.stringify({
    type: "response.done",
    response: { id: "response-old", status: "cancelled", output: [] },
  }));
  assert.equal(
    playbackMessages.filter((message) => message.kind === "clear").length,
    1,
    "delayed cancellation does not clear a replacement generation twice",
  );
  await client.close();
}

{
  const { client } = createClient();
  const playbackMessages = [];
  client._playbackNode = {
    port: { postMessage: (value) => playbackMessages.push(value) },
    disconnect: () => {},
  };
  for (let index = 0; index < 600; index += 1) {
    const responseId = `response-retired-${index}`;
    const snapshot = client._playbackSnapshot(responseId);
    snapshot.ended = true;
    client._retirePlaybackResponse(responseId);
  }
  assert.equal(client._playbackByResponse.size, 0, "completed response snapshots do not grow forever");
  assert.equal(client._stalePlaybackResponses.size, 512, "late-response tombstones stay bounded");
  const before = playbackMessages.length;
  client._finishPlaybackResponse("response-retired-599");
  assert.equal(playbackMessages.length, before, "a retired response cannot reopen playback");
  await client.close();
}

// Network protocol progress and response locks are independent from audible
// playback. Only worklet started/drained messages may enter/leave ai-speaking.
{
  const { client } = createClient();
  client._sessionConfigured = true;
  client._status = "connected";
  const playbackMessages = [];
  const finished = [];
  client._playbackNode = {
    port: { postMessage: (value) => playbackMessages.push(value) },
    disconnect: () => {},
  };
  client.addEventListener("response-finished", (event) => finished.push(event.detail));

  await client._onWsMessage(JSON.stringify({
    type: "response.created",
    response: { id: "response-lifecycle" },
  }));
  await client._onWsMessage(JSON.stringify({
    type: "response.content_part.added",
    response_id: "response-lifecycle",
    part: { type: "output_audio" },
  }));
  await client._onWsMessage(JSON.stringify({
    type: "response.output_audio_transcript.delta",
    response_id: "response-lifecycle",
    delta: "not an audible signal",
  }));
  await client._onWsMessage(JSON.stringify({
    type: "response.output_audio.delta",
    response_id: "response-lifecycle",
    delta: "AAA=",
  }));
  assert.equal(client.status, "processing", "network audio receipt does not claim speaking");
  assert.equal(client._aiSpeaking, false);
  assert.equal(client._heardResponses.has("response-lifecycle"), false);

  client._onPlaybackMessage({
    kind: "started",
    generation: client._playbackGeneration,
    streamId: "response-lifecycle",
    queuedMs: 40,
    primeTargetMs: 800,
  });
  assert.equal(client.status, "ai-speaking");
  assert.equal(client._aiSpeaking, true);
  assert.equal(client._heardResponses.has("response-lifecycle"), true);

  await client._onWsMessage(JSON.stringify({
    type: "response.done",
    response: { id: "response-lifecycle", status: "completed", output: [] },
  }));
  assert.equal(client._openResponses, 0, "response slot releases before queued playback drains");
  assert.equal(client.status, "ai-speaking", "response.done cannot end audible state");
  assert.equal(client._aiSpeaking, true);
  assert.equal(finished.at(-1).audible, true, "audible means the worklet rendered a sample");

  client._onPlaybackMessage({
    kind: "drained",
    generation: client._playbackGeneration,
    streamId: "response-lifecycle",
    queuedMs: 0,
  });
  assert.equal(client.status, "connected");
  assert.equal(client._aiSpeaking, false);

  await client._onWsMessage(JSON.stringify({
    type: "response.created",
    response: { id: "response-completed-before-start" },
  }));
  await client._onWsMessage(JSON.stringify({
    type: "response.output_audio.delta",
    response_id: "response-completed-before-start",
    delta: "AAA=",
  }));
  await client._onWsMessage(JSON.stringify({
    type: "response.done",
    response: { id: "response-completed-before-start", status: "completed", output: [] },
  }));
  assert.equal(finished.at(-1).audible, false, "queued PCM is not reported as already heard");
  assert.equal(client._openResponses, 0);
  assert.equal(client.status, "connected");

  client._onPlaybackMessage({
    kind: "started",
    generation: client._playbackGeneration,
    streamId: "response-completed-before-start",
  });
  assert.equal(client.status, "ai-speaking", "completed network response may still begin queued playback");
  assert.equal(
    client._heardResponses.has("response-completed-before-start"),
    false,
    "late start of a retired response cannot leak heard-response bookkeeping",
  );
  client._onPlaybackMessage({
    kind: "drained",
    generation: client._playbackGeneration,
    streamId: "response-completed-before-start",
  });
  assert.equal(client.status, "connected");
  assert.equal(client._aiSpeaking, false);
  assert.ok(playbackMessages.some((message) => message.kind === "end"));
  await client.close();
}

console.log("realtime pipeline config acknowledgement tests passed");
