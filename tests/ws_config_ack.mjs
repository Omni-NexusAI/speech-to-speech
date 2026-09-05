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

const { S2sWsRealtimeClient, PIPELINE_CONFIG_ACK_TIMEOUT_MS } = await import(
  "../web/hf-realtime-voice/ws/s2s-ws-client.js"
);

assert.equal(PIPELINE_CONFIG_ACK_TIMEOUT_MS, 15_000);

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

// Browser playback admission is a local pipeline extension. It must accompany
// the update so the server can freeze it before a response ID/lifecycle event.
{
  const { client, sent } = createClient();
  client.updateLocalPipeline(
    { tts_backend: "qwen3tts-audiocpp", tts_tuning: { profile_id: "native-480" } },
    {
      provider: "qwen3tts-audiocpp",
      profileId: "native-480",
      nativeStreaming: true,
      resolvedPrimeMs: 480,
      continuityMode: "adaptive",
    },
  );
  const update = sent.at(-1);
  assert.equal(update?.type, "pipeline.config.update");
  assert.deepEqual(update?.config?.playback_policy, {
    prime_target_ms: 480,
    continuity_mode: "adaptive",
    native_streaming: true,
    max_prime_ms: 2_000,
  });
  await client.close();
}

// Adaptive continuity also protects the stable buffered-phrase fallback. The
// transport remains truthfully non-native while using the profile reservoir.
{
  const { client, sent } = createClient();
  client.updateLocalPipeline(
    { tts_backend: "qwen3tts-audiocpp", tts_tuning: { profile_id: "balanced" } },
    {
      provider: "qwen3tts-audiocpp",
      profileId: "balanced",
      nativeStreaming: false,
      continuityMode: "adaptive",
    },
  );
  const update = sent.at(-1);
  assert.deepEqual(update?.config?.playback_policy, {
    prime_target_ms: 1_280,
    continuity_mode: "adaptive",
    native_streaming: false,
    max_prime_ms: 2_000,
  });
  await client._onWsMessage(JSON.stringify({
    type: "pipeline.config.updated",
    config: {
      ...update.config,
      audio_output_sample_rate: 24_000,
    },
  }));
  assert.equal(client._playbackPrimeMs, 1_280);
  client._raiseAdaptiveReserve();
  assert.equal(client._adaptivePlaybackPrimeMs, 1_280,
    "unscoped feedback must not mutate the active adaptive policy");
  client._acknowledgedPlaybackConfig.profileRevision = 4;
  client._raiseAdaptiveReserve(null, {
    continuityMode: "adaptive",
    provider: "qwen3tts-audiocpp",
    profileId: "balanced",
    profileRevision: 4,
    nativeStreaming: false,
    configPrimeMs: 1_280,
    configToken: client._playbackConfigToken,
    playbackUnsustainable: false,
  });
  assert.equal(client._adaptivePlaybackPrimeMs, 1_440);
  await client.close();
}

{
  const { client, sent } = createClient();
  await client._onWsMessage(JSON.stringify({ type: "session.created" }));
  assert.equal(sent[0]?.type, "pipeline.playback.capability");
  assert.equal(sent[0]?.rendered_playback_ack, true);
  assert.equal(sent[1]?.type, "pipeline.config.update");
  await client.close();
}

{
  const { client } = createClient();
  const workletMessages = [];
  client._playbackNode = { port: { postMessage: (message) => workletMessages.push(message) } };
  await client._onWsMessage(JSON.stringify({
    type: "pipeline.config.updated",
    config: {
      tts_backend: "qwen3tts-audiocpp",
      // This is server-owned acknowledgement metadata, not a user-provided
      // OpenAI output-format field.
      audio_output_sample_rate: 24000,
    },
  }));
  assert.equal(client._playbackSampleRate, 24000);
  assert.deepEqual(workletMessages, [{
    kind: "config",
    inputRate: 24000,
    generation: 0,
  }]);
  await client.close();
}

// A live provider/rate acknowledgement changes only the default for later
// responses. PCM already bound to an older response keeps its sample clock in
// both chunk and terminal worklet messages.
{
  const { client } = createClient();
  const workletMessages = [];
  client._playbackNode = { port: { postMessage: (message) => workletMessages.push(message) } };
  await client._onWsMessage(JSON.stringify({
    type: "pipeline.config.updated",
    config: { tts_backend: "faster", audio_output_sample_rate: 16_000 },
  }));
  await client._onWsMessage(JSON.stringify({
    type: "response.created", response: { id: "old-rate" }, response_epoch: 1,
  }));
  await client._onWsMessage(JSON.stringify({
    type: "response.audio.delta", response_id: "old-rate", response_epoch: 1, delta: "AAAAAA==",
  }));
  await client._onWsMessage(JSON.stringify({
    type: "response.audio.done", response_id: "old-rate", response_epoch: 1,
  }));

  await client._onWsMessage(JSON.stringify({
    type: "pipeline.config.updated",
    config: { tts_backend: "qwen3tts-audiocpp", audio_output_sample_rate: 24_000 },
  }));
  await client._onWsMessage(JSON.stringify({
    type: "response.created", response: { id: "new-rate" }, response_epoch: 2,
  }));
  await client._onWsMessage(JSON.stringify({
    type: "response.audio.delta", response_id: "new-rate", response_epoch: 2, delta: "AAAAAA==",
  }));
  await client._onWsMessage(JSON.stringify({
    type: "response.audio.done", response_id: "new-rate", response_epoch: 2,
  }));

  const audioRates = workletMessages
    .filter((message) => message.kind === "audio")
    .map((message) => [message.streamId, message.sourceSampleRate]);
  const endRates = workletMessages
    .filter((message) => message.kind === "end")
    .map((message) => [message.streamId, message.sourceSampleRate]);
  assert.deepEqual(audioRates, [["old-rate", 16_000], ["new-rate", 24_000]]);
  assert.deepEqual(endRates, [["old-rate", 16_000], ["new-rate", 24_000]]);
  await client.close();
}

// The server owns admission policy. Simulate a delayed candidate lifecycle
// arriving only after a newer Faster acknowledgement: the complete policy on
// pipeline.response must still bind the older response.
{
  const { client } = createClient();
  const workletMessages = [];
  client._playbackNode = { port: { postMessage: (message) => workletMessages.push(message) } };
  client._queuePlaybackConfig(
    { tts_backend: "qwen3tts-audiocpp", tts_tuning: { profile_id: "native-480" } },
    {
      provider: "qwen3tts-audiocpp",
      profileId: "native-480",
      nativeStreaming: true,
      resolvedPrimeMs: 480,
      continuityMode: "adaptive",
    },
  );
  await client._onWsMessage(JSON.stringify({
    type: "pipeline.config.updated",
    config: {
      tts_backend: "qwen3tts-audiocpp",
      tts_tuning: { profile_id: "native-480" },
      audio_output_sample_rate: 24_000,
    },
  }));
  assert.equal(client._playbackPrimeMs, 480);
  await client._onWsMessage(JSON.stringify({
    type: "pipeline.config.updated",
    config: { tts_backend: "faster", audio_output_sample_rate: 16_000 },
  }));
  await client._onWsMessage(JSON.stringify({
    type: "pipeline.response",
    input_epoch: 31,
    response_epoch: 31,
    state: "pending",
    output_sample_rate: 24_000,
    playback_policy: {
      source_sample_rate: 24_000,
      prime_target_ms: 480,
      continuity_mode: "adaptive",
      native_streaming: true,
      max_prime_ms: 2_000,
    },
  }));
  await client._onWsMessage(JSON.stringify({
    type: "response.created", response: { id: "candidate-owned" }, response_epoch: 31,
  }));
  await client._onWsMessage(JSON.stringify({
    type: "response.audio.delta", response_id: "candidate-owned", response_epoch: 31, delta: "AAAAAA==",
  }));
  await client._onWsMessage(JSON.stringify({
    type: "response.audio.done", response_id: "candidate-owned", response_epoch: 31,
  }));

  await client._onWsMessage(JSON.stringify({
    type: "pipeline.response",
    input_epoch: 32,
    response_epoch: 32,
    state: "pending",
    output_sample_rate: 16_000,
    playback_policy: {
      source_sample_rate: 16_000,
      prime_target_ms: 0,
      continuity_mode: "fast-start",
      native_streaming: false,
      max_prime_ms: 2_000,
    },
  }));
  await client._onWsMessage(JSON.stringify({
    type: "response.created", response: { id: "faster-next" }, response_epoch: 32,
  }));
  await client._onWsMessage(JSON.stringify({
    type: "response.audio.delta", response_id: "faster-next", response_epoch: 32, delta: "AAAAAA==",
  }));
  await client._onWsMessage(JSON.stringify({
    type: "response.audio.done", response_id: "faster-next", response_epoch: 32,
  }));

  const audioPolicies = workletMessages
    .filter((message) => message.kind === "audio")
    .map((message) => [message.streamId, message.sourceSampleRate, message.primeMs]);
  const endRates = workletMessages
    .filter((message) => message.kind === "end")
    .map((message) => [message.streamId, message.sourceSampleRate]);
  assert.deepEqual(audioPolicies, [["candidate-owned", 24_000, 480], ["faster-next", 16_000, 0]]);
  assert.deepEqual(endRates, [["candidate-owned", 24_000], ["faster-next", 16_000]]);
  await client.close();
}

// Reverse ordering: a delayed Faster lifecycle must retain its zero-reservoir
// policy even after candidate Adaptive 480 ms has been acknowledged.
{
  const { client } = createClient();
  const workletMessages = [];
  client._playbackNode = { port: { postMessage: (message) => workletMessages.push(message) } };
  await client._onWsMessage(JSON.stringify({
    type: "pipeline.config.updated",
    config: { tts_backend: "faster", audio_output_sample_rate: 16_000 },
  }));
  client._queuePlaybackConfig(
    { tts_backend: "qwen3tts-audiocpp", tts_tuning: { profile_id: "native-480" } },
    {
      provider: "qwen3tts-audiocpp",
      profileId: "native-480",
      nativeStreaming: true,
      resolvedPrimeMs: 480,
      continuityMode: "adaptive",
    },
  );
  await client._onWsMessage(JSON.stringify({
    type: "pipeline.config.updated",
    config: {
      tts_backend: "qwen3tts-audiocpp",
      tts_tuning: { profile_id: "native-480" },
      audio_output_sample_rate: 24_000,
    },
  }));
  await client._onWsMessage(JSON.stringify({
    type: "pipeline.response", input_epoch: 41, response_epoch: 41,
    state: "pending", output_sample_rate: 16_000,
    playback_policy: {
      source_sample_rate: 16_000,
      prime_target_ms: 0,
      continuity_mode: "fast-start",
      native_streaming: false,
      max_prime_ms: 2_000,
    },
  }));
  await client._onWsMessage(JSON.stringify({
    type: "response.created", response: { id: "faster-owned" }, response_epoch: 41,
  }));
  await client._onWsMessage(JSON.stringify({
    type: "response.audio.delta", response_id: "faster-owned", response_epoch: 41, delta: "AAAAAA==",
  }));
  await client._onWsMessage(JSON.stringify({
    type: "response.audio.done", response_id: "faster-owned", response_epoch: 41,
  }));
  await client._onWsMessage(JSON.stringify({
    type: "pipeline.response", input_epoch: 42, response_epoch: 42,
    state: "pending", output_sample_rate: 24_000,
    playback_policy: {
      source_sample_rate: 24_000,
      prime_target_ms: 480,
      continuity_mode: "adaptive",
      native_streaming: true,
      max_prime_ms: 2_000,
    },
  }));
  await client._onWsMessage(JSON.stringify({
    type: "response.created", response: { id: "candidate-next" }, response_epoch: 42,
  }));
  await client._onWsMessage(JSON.stringify({
    type: "response.audio.delta", response_id: "candidate-next", response_epoch: 42, delta: "AAAAAA==",
  }));
  await client._onWsMessage(JSON.stringify({
    type: "response.audio.done", response_id: "candidate-next", response_epoch: 42,
  }));
  const audioPolicies = workletMessages
    .filter((message) => message.kind === "audio")
    .map((message) => [message.streamId, message.sourceSampleRate, message.primeMs]);
  assert.deepEqual(audioPolicies, [["faster-owned", 16_000, 0], ["candidate-next", 24_000, 480]]);
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

console.log("realtime pipeline config acknowledgement tests passed");
