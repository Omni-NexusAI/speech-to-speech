// Browser-client behavior regressions for tool exactly-once handling and the
// epoch-only terminal lifecycle emitted when no OpenAI response ID exists.
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
globalThis.window ??= globalThis;

const { S2sWsRealtimeClient } = await import("../web/hf-realtime-voice/ws/s2s-ws-client.js");

function connectedClient() {
  const sent = [];
  const client = new S2sWsRealtimeClient({});
  client._ws = {
    readyState: WebSocket.OPEN,
    send(value) { sent.push(JSON.parse(value)); },
    close() {},
  };
  return { client, sent };
}

// A relay may repeat its terminal function-call event after the output has
// already been acknowledged. The external tool stays single-shot, while the
// original output acknowledgement remains functional.
{
  const { client, sent } = connectedClient();
  const calls = [];
  const finished = [];
  client.addEventListener("toolcall", (event) => calls.push(event.detail));
  client.addEventListener("response-finished", (event) => finished.push(event.detail));

  const functionDone = {
    type: "response.function_call_arguments.done",
    response_id: "tool-origin",
    response_epoch: 20,
    call_id: "call-unique",
    name: "camera_snapshot",
    arguments: "{}",
  };
  await client._onWsMessage(JSON.stringify(functionDone));
  assert.equal(calls.length, 1, "first function call must reach the tool runner");

  const outputAck = client.sendToolOutput("call-unique", "{\"ok\":true}");
  assert.deepEqual(sent.map((event) => event.type), ["conversation.item.create"]);
  await client._onWsMessage(JSON.stringify({
    type: "conversation.item.created",
    item: { type: "function_call_output", call_id: "call-unique" },
  }));
  await outputAck;

  await client._onWsMessage(JSON.stringify(functionDone));
  assert.equal(calls.length, 1, "duplicate terminal call must not execute an external tool twice");

  await client._onWsMessage(JSON.stringify({
    type: "response.created", response: { id: "tool-origin" }, response_epoch: 20,
  }));
  await client._onWsMessage(JSON.stringify({
    type: "response.done", response: { id: "tool-origin", status: "completed" }, response_epoch: 20,
  }));
  await client._onWsMessage(JSON.stringify({
    type: "response.done", response: { id: "tool-origin", status: "completed" }, response_epoch: 20,
  }));
  assert.equal(finished.length, 1);
  assert.equal(finished[0].committed, true, "tool-only origin must be retained despite absent PCM");
}

// A relay can emit an old function-call terminal after a newer response epoch
// owns the connection. Reject it before its call ID is tombstoned or dispatched;
// otherwise a stale external side effect could still run once.
{
  const { client } = connectedClient();
  const calls = [];
  client.addEventListener("toolcall", (event) => calls.push(event.detail));
  await client._onWsMessage(JSON.stringify({
    type: "response.created", response: { id: "stale-response" }, response_epoch: 30,
  }));
  await client._onWsMessage(JSON.stringify({
    type: "response.created", response: { id: "current-response" }, response_epoch: 31,
  }));
  await client._onWsMessage(JSON.stringify({
    type: "response.function_call_arguments.done", response_id: "stale-response",
    response_epoch: 30, call_id: "late-call", name: "camera_snapshot", arguments: "{}",
  }));
  await client._onWsMessage(JSON.stringify({
    type: "response.function_call_arguments.done", response_id: "stale-response",
    call_id: "late-call-no-epoch", name: "camera_snapshot", arguments: "{}",
  }));
  assert.equal(calls.length, 0, "stale function-call terminals must not execute browser tools");
  assert.equal(client._toolCallTombstones.has("late-call"), false, "stale call must not consume a valid future ID");
}

// A generation can be cancelled before response.created. Its epoch-only local
// lifecycle must clear the create guard, return to connected state, and reach
// the UI only once so the provisional user row can be rolled back.
{
  const { client } = connectedClient();
  const finished = [];
  client._status = "processing";
  client._createInFlight = true;
  client.addEventListener("response-finished", (event) => finished.push(event.detail));

  await client._onWsMessage(JSON.stringify({
    type: "pipeline.response", input_epoch: 41, response_epoch: 41, state: "pending",
  }));
  await client._onWsMessage(JSON.stringify({
    type: "pipeline.response", input_epoch: 41, response_epoch: 41,
    state: "cancelled", reason: "no_audio",
  }));
  await client._onWsMessage(JSON.stringify({
    type: "pipeline.response", input_epoch: 41, response_epoch: 41,
    state: "cancelled", reason: "duplicate_cleanup",
  }));

  assert.equal(client._createInFlight, false, "epoch-only terminal must release the response-create guard");
  assert.equal(client.status, "connected", "epoch-only terminal must leave processing state");
  assert.deepEqual(finished, [{
    responseId: "",
    status: "cancelled",
    audible: false,
    transcript: "",
    responseEpoch: 41,
    committed: false,
  }]);
}

console.log("tool tombstone and epoch-only terminal lifecycle tests passed");
