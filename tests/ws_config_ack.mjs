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
  AUDIO_CPP_PLAYBACK_PRIME_MS,
  MAX_PLAYBACK_PRIME_MS,
  PIPELINE_CONFIG_ACK_TIMEOUT_MS,
  S2sWsRealtimeClient,
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
