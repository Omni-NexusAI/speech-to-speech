// Behaviour regression for the durable chat history ownership boundary.
// Runs without a browser: ChatView's terminal bookkeeping is intentionally
// exercised with minimal DOM-shaped rows.
if (typeof globalThis.CustomEvent === "undefined") {
  globalThis.CustomEvent = class CustomEvent extends Event {
    constructor(type, init = {}) { super(type); this.detail = init.detail; }
  };
}
const { ChatView } = await import("../web/hf-realtime-voice/ui/chat.js");
const { S2sWsRealtimeClient } = await import("../web/hf-realtime-voice/ws/s2s-ws-client.js");

function row(text = "") {
  return {
    text,
    removed: false,
    interrupted: false,
    remove() { this.removed = true; },
    querySelector() { return null; },
  };
}

function view() {
  const chat = Object.create(ChatView.prototype);
  chat._userHistByItem = new Map();
  chat._userTurns = [];
  chat._userTurnByItem = new Map();
  chat._userTurnByResponse = new Map();
  chat._asstByResp = new Map();
  chat._pendingUserHist = null;
  chat._activeUserBubble = null;
  chat._activeUserItemId = "";
  chat._chatHistory = { querySelector: () => ({}) };
  chat._dismissBubble = (bubble) => { bubble.dismissed = true; };
  chat._markHistInterrupted = (hist) => { hist.interrupted = true; };
  chat._updateHistMsg = (hist, text) => { hist.text = text; };
  chat._appendHistMsg = (_role, text) => {
    const hist = row(text);
    chat.appended.push(hist);
    return hist;
  };
  chat.appended = [];
  return chat;
}

// A cancelled local response that never rendered must roll back both sides of
// its provisional transaction, even if an assistant transcript arrived first.
{
  const chat = view();
  const user = row("hello");
  const assistant = row("unheard reply");
  const bubble = {};
  chat._trackUserTurn(user, "u1");
  chat._asstByResp.set("r1", { hist: assistant, bubble });
  chat.onResponseFinished({ responseId: "r1", status: "cancelled", audible: false, responseEpoch: 11 });
  if (!user.removed || !assistant.removed || !bubble.dismissed) throw new Error("unheard local cancellation stayed in history");
  if (chat._userTurns.length || chat._asstByResp.size || chat._userHistByItem.size) throw new Error("unheard transaction bookkeeping leaked");
}

// `response.done` can reach the protocol before primed PCM reaches the
// AudioWorklet. Wire the real browser client into ChatView: completion must
// not roll back the provisional rows until started proves audibility, and the
// later drained event must not settle a second time.
{
  const chat = view();
  const user = row("primed user turn");
  const assistant = row("primed assistant turn");
  chat._trackUserTurn(user, "u-primed");
  chat._asstByResp.set("r-primed", { hist: assistant, bubble: {} });
  const client = new S2sWsRealtimeClient({});
  client._playbackNode = { port: { postMessage: () => {} } };
  const terminals = [];
  client.addEventListener("response-finished", (event) => {
    terminals.push(event.detail);
    chat.onResponseFinished(event.detail);
  });
  await client._onWsMessage(JSON.stringify({
    type: "response.created", response: { id: "r-primed" }, response_epoch: 20,
  }));
  await client._onWsMessage(JSON.stringify({
    type: "pipeline.response", response_id: "r-primed", input_epoch: 20, response_epoch: 20,
  }));
  await client._onWsMessage(JSON.stringify({
    type: "response.audio.delta", response_id: "r-primed", response_epoch: 20, delta: "AAAAAA==",
  }));
  await client._onWsMessage(JSON.stringify({
    type: "response.done", response: { id: "r-primed", status: "completed" }, response_epoch: 20,
  }));
  if (terminals.length !== 0 || user.removed || assistant.removed) {
    throw new Error("response.done settled a primed-but-unheard ChatView transaction");
  }
  client._onPlaybackMessage({ kind: "started", generation: 0, streamId: "r-primed", responseEpoch: 20 });
  if (terminals.length !== 1 || terminals[0].audible !== true || user.removed || assistant.removed) {
    throw new Error("worklet start did not retain the pending ChatView transaction exactly once");
  }
  client._onPlaybackMessage({ kind: "drained", generation: 0, streamId: "r-primed", responseEpoch: 20 });
  if (terminals.length !== 1) throw new Error("drained playback dispatched a duplicate terminal");
  await client.close();
}

// Conversely, a primed response that drains or clears without a rendered
// sample is finally unheard. This is the only false-audible path for a normal
// completed protocol response.
{
  const chat = view();
  const user = row("unheard primed user");
  const assistant = row("unheard primed assistant");
  chat._trackUserTurn(user, "u-drained");
  chat._asstByResp.set("r-drained", { hist: assistant, bubble: {} });
  const client = new S2sWsRealtimeClient({});
  client._playbackNode = { port: { postMessage: () => {} } };
  const terminals = [];
  client.addEventListener("response-finished", (event) => {
    terminals.push(event.detail);
    chat.onResponseFinished(event.detail);
  });
  await client._onWsMessage(JSON.stringify({
    type: "response.created", response: { id: "r-drained" }, response_epoch: 21,
  }));
  await client._onWsMessage(JSON.stringify({
    type: "pipeline.response", response_id: "r-drained", input_epoch: 21, response_epoch: 21,
  }));
  await client._onWsMessage(JSON.stringify({
    type: "response.audio.delta", response_id: "r-drained", response_epoch: 21, delta: "AAAAAA==",
  }));
  await client._onWsMessage(JSON.stringify({
    type: "response.done", response: { id: "r-drained", status: "completed" }, response_epoch: 21,
  }));
  client._onPlaybackMessage({ kind: "drained", generation: 0, streamId: "r-drained", responseEpoch: 21 });
  if (terminals.length !== 1 || terminals[0].audible !== false || !user.removed || !assistant.removed) {
    throw new Error("definitive drain did not settle the unheard ChatView transaction");
  }
  await client.close();
}

// Once a sample rendered, interruption is durable and appears exactly once.
{
  const chat = view();
  const user = row("hello again");
  const assistant = row("heard reply");
  chat._trackUserTurn(user, "u2");
  chat._asstByResp.set("r2", { hist: assistant, bubble: {} });
  chat.onResponseFinished({ responseId: "r2", status: "cancelled", audible: true, responseEpoch: 12, transcript: "heard reply" });
  if (user.removed || assistant.removed || !assistant.interrupted) throw new Error("audible interruption was not retained once");
  if (chat.appended.length !== 0) throw new Error("audible interruption duplicated its assistant row");
}

// The terminal payload can be the only transcript source after a barge-in.
// It still creates one durable interrupted row after an audible response.
{
  const chat = view();
  chat._trackUserTurn(row("late transcript"), "u-late");
  chat.onResponseFinished({ responseId: "r-late", status: "canceled", audible: true, responseEpoch: 13, transcript: "last audible words" });
  if (chat.appended.length !== 1 || !chat.appended[0].interrupted) throw new Error("audible terminal transcript was not retained exactly once");
}

// Tool-only direct-audio responses can have no PCM, but their server-side
// transaction is already committed. Keep the originating user row so the later
// spoken tool follow-up has visible chronology.
{
  const chat = view();
  const user = row("take a snapshot");
  chat._trackUserTurn(user, "u-tool");
  chat.onResponseFinished({
    responseId: "r-tool", status: "completed", audible: false,
    responseEpoch: 14, committed: true,
  });
  if (user.removed || chat._userTurns.length !== 0) {
    throw new Error("committed tool-only origin was rolled back or left provisional");
  }
}

// A local terminal lifecycle can arrive before response.created and therefore
// has no OpenAI response ID. The response epoch still owns and removes its
// provisional user row exactly once.
{
  const chat = view();
  const user = row("unheard no-id turn");
  chat._trackUserTurn(user, "u-no-id");
  chat.onResponseFinished({ responseId: "", status: "cancelled", audible: false, responseEpoch: 15 });
  if (!user.removed || chat._userTurns.length !== 0 || chat._userHistByItem.size !== 0) {
    throw new Error("epoch-only terminal did not roll back its provisional user row");
  }
}

// Standard OpenAI-compatible text completions have no local response epoch;
// they must retain the legacy transcript even without a playback acknowledgement.
{
  const chat = view();
  const user = row("text prompt");
  const assistant = row("text answer");
  chat._trackUserTurn(user, "u3");
  chat._asstByResp.set("r3", { hist: assistant, bubble: {} });
  // The websocket client includes this property with a null value for an
  // ordinary OpenAI event, so it must not be coerced to local epoch zero.
  chat.onResponseFinished({ responseId: "r3", status: "completed", audible: false, responseEpoch: null });
  if (user.removed || assistant.removed) throw new Error("standard OpenAI text completion was incorrectly rolled back");
}

console.log("chat unheard response history passed");
