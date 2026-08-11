import assert from "node:assert/strict";

if (typeof globalThis.CustomEvent === "undefined") {
  globalThis.CustomEvent = class CustomEvent extends Event {
    constructor(type, init = {}) {
      super(type);
      this.detail = init.detail;
    }
  };
}
globalThis.window = globalThis;
globalThis.WebSocket = { OPEN: 1 };

const {
  S2sWsRealtimeClient,
  invalidToolArgumentsOutput,
  prepareToolArgumentsForBrowser,
  validateToolArguments,
} = await import("../web/hf-realtime-voice/ws/s2s-ws-client.js");
const { ChatView } = await import("../web/hf-realtime-voice/ui/chat.js");

const searchSchema = {
  type: "object",
  properties: { query: { type: "string", minLength: 1, pattern: "\\S" } },
  required: ["query"],
  additionalProperties: false,
};
const cameraSchema = {
  type: "object",
  properties: {},
  required: [],
  additionalProperties: false,
};

assert.deepEqual(
  validateToolArguments('{"query":"current weather"}', searchSchema),
  { ok: true, args: { query: "current weather" } },
);
for (const [raw, expectedCode, expectedPath, expectedClass] of [
  ['{"query":', "malformed_json", "$", "json_parse_error"],
  [null, "malformed_json", "$", "expected_json_string"],
  ['[]', "schema_validation_failed", "$", "expected_object"],
  ['{}', "schema_validation_failed", "$.query", "required"],
  ['{"query":7}', "schema_validation_failed", "$.query", "expected_string"],
  ['{"query":"   "}', "schema_validation_failed", "$.query", "pattern_mismatch"],
  ['{"query":"weather","hidden":"value"}', "schema_validation_failed", "$.*", "unexpected_property"],
]) {
  const failure = validateToolArguments(raw, searchSchema);
  assert.equal(failure.ok, false);
  assert.equal(failure.code, expectedCode);
  assert.equal(failure.path, expectedPath);
  assert.equal(failure.errorClass, expectedClass);
}
assert.equal(validateToolArguments('{}', cameraSchema).ok, true);
assert.deepEqual(
  validateToolArguments('{"reuse_previous":true}', cameraSchema),
  { ok: false, code: "schema_validation_failed", path: "$.*", errorClass: "unexpected_property" },
);

// Durable/card presentation never contains malformed values, unexpected field
// names, or values from unknown tool schemas.
{
  const secretField = "private_token_name";
  const secretValue = "never-persist-this-value";
  const invalid = prepareToolArgumentsForBrowser(
    "web_search",
    JSON.stringify({ query: "weather", [secretField]: secretValue }),
    searchSchema,
  );
  assert.equal(invalid.tool, "web_search");
  assert.equal(invalid.validation.ok, false);
  assert.equal(invalid.validation.path, "$.*");
  assert.equal(invalid.displayArguments.includes(secretField), false);
  assert.equal(invalid.displayArguments.includes(secretValue), false);

  const malformed = prepareToolArgumentsForBrowser(
    "web_search",
    `{"query":"${secretValue}`,
    searchSchema,
  );
  assert.equal(malformed.displayArguments.includes(secretValue), false);

  const unknownName = "secret_tool_name";
  const unknown = prepareToolArgumentsForBrowser(
    unknownName,
    JSON.stringify({ [secretField]: secretValue }),
    null,
  );
  assert.equal(unknown.tool, "unknown_tool");
  assert.equal(unknown.displayArguments.includes(unknownName), false);
  assert.equal(unknown.displayArguments.includes(secretField), false);
  assert.equal(unknown.displayArguments.includes(secretValue), false);
}

// The durable ChatView boundary receives only the prepared representation.
{
  const secretField = "durable_secret_key";
  const secretValue = "durable-secret-value";
  const prepared = prepareToolArgumentsForBrowser(
    "web_search",
    JSON.stringify({ query: "weather", [secretField]: secretValue }),
    searchSchema,
  );
  const recorded = [];
  const view = Object.create(ChatView.prototype);
  view._toolHistByCall = new Map();
  view._spawnBubble = () => ({});
  view._bumpDismiss = () => {};
  view._markUnread = () => {};
  view._appendHistTool = (name, args, output) => {
    recorded.push({ name, args, output });
    return { querySelector: () => null };
  };
  view.onToolCall(prepared.tool, prepared.displayArguments, "call-private", undefined);
  const serialized = JSON.stringify(recorded);
  assert.equal(serialized.includes(secretField), false);
  assert.equal(serialized.includes(secretValue), false);
}

// Camera cards use the capture generation as part of their durable identity,
// so repeated or missing backend call IDs can never collapse actual captures.
{
  const view = Object.create(ChatView.prototype);
  const first = view._toolHistoryKey("call-repeat", { captureGeneration: 1 });
  const second = view._toolHistoryKey("call-repeat", { captureGeneration: 2 });
  const missing = view._toolHistoryKey("", { captureGeneration: 3 });
  assert.notEqual(first, second);
  assert.match(missing, /camera:3:missing-call-id/);

  const meta = { hidden: true, textContent: "" };
  const card = { dataset: {}, querySelector: (selector) => selector === ".hist-tool-meta" ? meta : null };
  view._updateToolLifecycle(card, {
    captureGeneration: 2,
    requestedAtMs: 1_700_000_000_000,
    callId: "call-repeat",
    captureStatus: "unavailable",
    outputStatus: "acknowledged",
  });
  assert.equal(card.dataset.captureGeneration, "2");
  assert.equal(card.dataset.callId, "call-repeat");
  assert.equal(meta.hidden, false);
  assert.match(meta.textContent, /Capture #2/);
  assert.match(meta.textContent, /unavailable/);
  assert.match(meta.textContent, /output acknowledged/);
}

{
  const rawSecret = "do-not-copy-this-argument";
  const failure = validateToolArguments(`{"query":${rawSecret}`, searchSchema);
  assert.equal(failure.ok, false);
  const output = invalidToolArgumentsOutput("web_search", failure);
  const parsed = JSON.parse(output);
  assert.equal(parsed.type, "invalid_tool_arguments");
  assert.equal(parsed.tool, "web_search");
  assert.equal(parsed.code, "malformed_json");
  assert.equal(parsed.path, "$");
  assert.equal(parsed.error_class, "json_parse_error");
  assert.equal(output.includes(rawSecret), false, "invalid output never echoes raw argument values");
}

function createClient() {
  const sent = [];
  const client = new S2sWsRealtimeClient({
    directUrl: "ws://127.0.0.1:8765/v1/realtime",
    voice: "clone:test",
    instructions: "test",
    pipelineConfig: {},
  });
  client._ws = {
    readyState: WebSocket.OPEN,
    send: (value) => sent.push(JSON.parse(value)),
    close: () => {},
  };
  return { client, sent };
}

// Public call events preserve malformed raw strings exactly. Non-string
// protocol arguments stay invalid instead of being rewritten to an empty object.
{
  const { client } = createClient();
  const calls = [];
  client.addEventListener("toolcall", (event) => calls.push(event.detail));
  await client._onWsMessage(JSON.stringify({
    type: "response.function_call_arguments.done",
    name: "web_search",
    arguments: '{"query":',
    call_id: "call-malformed",
  }));
  await client._onWsMessage(JSON.stringify({
    type: "response.function_call_arguments.done",
    name: "camera_snapshot",
    arguments: { unexpected: true },
    call_id: "call-non-string",
  }));
  assert.equal(calls[0].arguments, '{"query":');
  assert.equal(calls[1].arguments, null);
  await client.close();
}

// A missing call ID is rejected before the executable toolcall event. No tool
// side effect and no response.create can occur from this protocol failure.
{
  const { client, sent } = createClient();
  const calls = [];
  const protocolErrors = [];
  client.addEventListener("toolcall", (event) => calls.push(event.detail));
  client.addEventListener("tool-protocol-error", (event) => protocolErrors.push(event.detail));
  await client._onWsMessage(JSON.stringify({
    type: "response.function_call_arguments.done",
    name: "camera_snapshot",
    arguments: "{}",
    call_id: "",
  }));
  assert.equal(calls.length, 0);
  assert.deepEqual(protocolErrors, [{ code: "missing_call_id" }]);
  assert.equal(sent.length, 0);

  const recorded = [];
  const view = Object.create(ChatView.prototype);
  view._spawnBubble = () => ({});
  view._bumpDismiss = () => {};
  view._markUnread = () => {};
  view._appendHistTool = (name, args, output) => recorded.push({ name, args, output });
  view.onToolProtocolFailure(protocolErrors[0].code);
  assert.equal(JSON.stringify(recorded).includes("camera_snapshot"), false);
  assert.match(recorded[0].output, /missing_call_id/);
  await client.close();
}

// Preserve hosted transaction ordering: function output, optional camera image,
// then one response.create. Acknowledgement may arrive afterward.
{
  const { client, sent } = createClient();
  const ack = client.sendToolOutput("call-camera-2", "captured");
  client.requestToolResponse({ image: "data:image/jpeg;base64,AAAA" });
  assert.deepEqual(sent.map((event) => event.type), [
    "conversation.item.create",
    "conversation.item.create",
    "response.create",
  ]);
  assert.equal(sent[0].item.type, "function_call_output");
  assert.equal(sent[0].item.call_id, "call-camera-2");
  assert.equal(sent[1].item.content[0].type, "input_image");
  assert.equal(sent.filter((event) => event.type === "response.create").length, 1);
  await client._onWsMessage(JSON.stringify({
    type: "conversation.item.created",
    item: { type: "function_call_output", call_id: "call-camera-2" },
  }));
  await ack;
  await client.close();
}

console.log("tool integrity tests passed");
