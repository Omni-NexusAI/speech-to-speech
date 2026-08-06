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
