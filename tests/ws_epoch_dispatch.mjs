// Lightweight Node regression for local response-epoch routing. It exercises
// the client dispatch table rather than relying on syntax-only checks.
globalThis.CustomEvent = class CustomEvent extends Event {
  constructor(type, init = {}) { super(type); this.detail = init.detail; }
};

const { S2sWsRealtimeClient } = await import("../web/hf-realtime-voice/ws/s2s-ws-client.js");
const client = new S2sWsRealtimeClient({});
const posted = [];
const sent = [];
client._playbackNode = { port: { postMessage: (message) => posted.push(message) } };
client._send = (message) => sent.push(message);

await client._onWsMessage(JSON.stringify({ type: "response.created", response: { id: "r-old" } }));
await client._onWsMessage(JSON.stringify({
  type: "pipeline.response", response_id: "r-old", input_epoch: 4, response_epoch: 4, lifecycle_state: "pending",
}));
await client._onWsMessage(JSON.stringify({
  type: "response.audio.delta", response_id: "r-old", delta: "AAAAAA==", // two PCM16 samples
}));
if (posted.length !== 1 || posted[0].responseEpoch !== 4 || posted[0].samples.length !== 2) {
  throw new Error("epochless PCM did not inherit the bound response epoch exactly once");
}

client._onPlaybackMessage({ kind: "started", generation: 0, streamId: "r-old", responseEpoch: 4 });
client._onPlaybackMessage({ kind: "started", generation: 0, streamId: "r-old", responseEpoch: 4, reprime: true });
if (sent.filter((message) => message.type === "pipeline.playback.started").length !== 1) {
  throw new Error("playback-start acknowledgement was not idempotent");
}

// A cancelled response ID remains a tombstone even when a late compatible
// transcript frame omits response_epoch. It must not recreate an assistant
// transcript/history row, while unrelated legacy IDs remain permitted.
const transcriptClient = new S2sWsRealtimeClient({});
const transcripts = [];
transcriptClient.addEventListener("transcript", (event) => transcripts.push(event.detail));
await transcriptClient._onWsMessage(JSON.stringify({
  type: "response.created", response: { id: "r-cancelled" }, response_epoch: 12,
}));
await transcriptClient._onWsMessage(JSON.stringify({
  type: "response.done", response: { id: "r-cancelled", status: "cancelled" }, response_epoch: 12,
}));
await transcriptClient._onWsMessage(JSON.stringify({
  type: "response.audio_transcript.done", response_id: "r-cancelled", transcript: "must not render",
}));
if (transcripts.length !== 0) throw new Error("retired epoch-less transcript resurrected a cancelled response");
await transcriptClient._onWsMessage(JSON.stringify({
  type: "response.audio_transcript.done", response_id: "legacy-unbound", transcript: "still valid",
}));
if (transcripts.length !== 1 || transcripts[0].text !== "still valid") {
  throw new Error("retired-ID guard blocked an unrelated legacy transcript");
}

// A response that completes before any audio reaches the worklet is rolled
// back by the ChatView.  The client must tombstone it too, otherwise a late
// epoch-less transcript can recreate the discarded assistant row.
const unheardClient = new S2sWsRealtimeClient({});
const unheardTranscripts = [];
const unheardFinished = [];
unheardClient.addEventListener("transcript", (event) => unheardTranscripts.push(event.detail));
unheardClient.addEventListener("response-finished", (event) => unheardFinished.push(event.detail));
await unheardClient._onWsMessage(JSON.stringify({
  type: "response.created", response: { id: "r-unheard" }, response_epoch: 13,
}));
await unheardClient._onWsMessage(JSON.stringify({
  type: "response.done", response: { id: "r-unheard", status: "completed" }, response_epoch: 13,
}));
if (unheardFinished.length !== 1 || unheardFinished[0].audible !== false
    || !unheardClient._stalePlaybackResponses.has("r-unheard")) {
  throw new Error("completed unheard response was not retired at its rollback boundary");
}
await unheardClient._onWsMessage(JSON.stringify({
  type: "response.output_audio_transcript.done", response_id: "r-unheard", transcript: "must stay discarded",
}));
if (unheardTranscripts.length !== 0) {
  throw new Error("late epoch-less transcript recreated a completed unheard response");
}

await client._onWsMessage(JSON.stringify({ type: "response.created", response: { id: "r-new" }, response_epoch: 5 }));
await client._onWsMessage(JSON.stringify({ type: "response.audio.delta", response_id: "r-old", response_epoch: 4, delta: "AAAAAA==" }));
if (posted.length !== 1) throw new Error("obsolete PCM was not rejected after epoch advanced");

// Ownership is claimed before the OpenAI response has an ID. Diagnostics must
// see that pending epoch, while playback binding waits for the later ID.
const pendingClient = new S2sWsRealtimeClient({});
const pendingMetrics = [];
pendingClient.addEventListener("pipeline-metric", (event) => pendingMetrics.push(event.detail));
await pendingClient._onWsMessage(JSON.stringify({
  type: "pipeline.response", input_epoch: 6, response_epoch: 6, state: "pending", reason: "speech_stopped",
}));
if (pendingClient._latestResponseEpoch !== 6 || pendingClient._playbackByResponse.size !== 0) {
  throw new Error("epoch-only pending lifecycle event incorrectly created playback state");
}
if (pendingMetrics.at(-1)?.detail?.supersession_reason !== "speech_stopped") {
  throw new Error("pending lifecycle diagnostics dropped supersession reason");
}
await pendingClient._onWsMessage(JSON.stringify({ type: "response.created", response: { id: "r-pending" }, response_epoch: 6 }));
if (!pendingClient._playbackByResponse.has("r-pending")) {
  throw new Error("response ID did not bind the prior pending lifecycle epoch");
}

// Protocol completion can precede worklet startup while PCM is still priming.
// It must remain eligible for exactly one local audible-playback acknowledgement.
const primingClient = new S2sWsRealtimeClient({});
const primingSent = [];
primingClient._playbackNode = { port: { postMessage: () => {} } };
primingClient._send = (message) => primingSent.push(message);
await primingClient._onWsMessage(JSON.stringify({
  type: "response.created", response: { id: "r-primed" }, response_epoch: 9,
}));
await primingClient._onWsMessage(JSON.stringify({
  type: "pipeline.response", response_id: "r-primed", input_epoch: 9, response_epoch: 9,
}));
await primingClient._onWsMessage(JSON.stringify({
  type: "response.audio.delta", response_id: "r-primed", response_epoch: 9, delta: "AAAAAA==",
}));
await primingClient._onWsMessage(JSON.stringify({ type: "response.audio.done", response_id: "r-primed", response_epoch: 9 }));
await primingClient._onWsMessage(JSON.stringify({
  type: "response.done", response: { id: "r-primed", status: "completed" }, response_epoch: 9,
}));
if (!primingClient._playbackByResponse.has("r-primed")) {
  throw new Error("completed priming response was retired before first rendered sample");
}
primingClient._onPlaybackMessage({ kind: "started", generation: 0, streamId: "r-primed", responseEpoch: 9 });
if (primingSent.filter((message) => message.type === "pipeline.playback.started").length !== 1) {
  throw new Error("completed priming response did not acknowledge first rendered sample exactly once");
}
primingClient._onPlaybackMessage({ kind: "drained", generation: 0, streamId: "r-primed", responseEpoch: 9 });
primingClient._onPlaybackMessage({ kind: "started", generation: 0, streamId: "r-primed", responseEpoch: 9 });
if (primingSent.filter((message) => message.type === "pipeline.playback.started").length !== 1) {
  throw new Error("retired response acknowledged a stale late worklet start");
}

// The inverse ordering is legal too: output_audio.done can make the worklet
// drain before response.done reaches the browser. Preserve the snapshot until
// that protocol terminal settles the provisional chat exactly once.
const earlyDrainClient = new S2sWsRealtimeClient({});
const earlyDrainFinished = [];
earlyDrainClient._playbackNode = { port: { postMessage: () => {} } };
earlyDrainClient._send = () => {};
earlyDrainClient.addEventListener("response-finished", (event) => earlyDrainFinished.push(event.detail));
await earlyDrainClient._onWsMessage(JSON.stringify({
  type: "response.created", response: { id: "r-early-drain" }, response_epoch: 10,
}));
await earlyDrainClient._onWsMessage(JSON.stringify({
  type: "pipeline.response", response_id: "r-early-drain", input_epoch: 10, response_epoch: 10,
}));
await earlyDrainClient._onWsMessage(JSON.stringify({
  type: "response.audio.delta", response_id: "r-early-drain", response_epoch: 10, delta: "AAAAAA==",
}));
earlyDrainClient._onPlaybackMessage({
  kind: "started", generation: 0, streamId: "r-early-drain", responseEpoch: 10,
});
await earlyDrainClient._onWsMessage(JSON.stringify({
  type: "response.audio.done", response_id: "r-early-drain", response_epoch: 10,
}));
earlyDrainClient._onPlaybackMessage({
  kind: "drained", generation: 0, streamId: "r-early-drain", responseEpoch: 10, queueEmpty: true,
});
if (!earlyDrainClient._playbackByResponse.has("r-early-drain") || earlyDrainFinished.length !== 0) {
  throw new Error("drain-before-terminal did not retain its playback snapshot");
}
await earlyDrainClient._onWsMessage(JSON.stringify({
  type: "response.done", response: { id: "r-early-drain", status: "completed" }, response_epoch: 10,
}));
if (earlyDrainFinished.length !== 1 || earlyDrainFinished[0].audible !== true
    || earlyDrainClient._playbackByResponse.has("r-early-drain")) {
  throw new Error("drain-before-terminal did not settle and retire exactly once");
}

// Promotion of a newer protocol response must not invalidate lifecycle events
// for older PCM already admitted to the browser FIFO. In particular, a newer
// tool-only/failed response may never strand the prior drain and leave the UI
// permanently speaking.
const promotedClient = new S2sWsRealtimeClient({});
const promotedSent = [];
const promotedFinished = [];
promotedClient._playbackNode = { port: { postMessage: () => {} } };
promotedClient._send = (message) => promotedSent.push(message);
promotedClient.addEventListener("response-finished", (event) => promotedFinished.push(event.detail));
await promotedClient._onWsMessage(JSON.stringify({
  type: "response.created", response: { id: "r-a" }, response_epoch: 40,
}));
await promotedClient._onWsMessage(JSON.stringify({
  type: "pipeline.response", response_id: "r-a", input_epoch: 40, response_epoch: 40,
}));
await promotedClient._onWsMessage(JSON.stringify({
  type: "response.audio.delta", response_id: "r-a", response_epoch: 40, delta: "AAAAAA==",
}));
promotedClient._onPlaybackMessage({
  kind: "started", generation: 0, streamId: "r-a", responseEpoch: 40,
});
await promotedClient._onWsMessage(JSON.stringify({
  type: "response.done", response: { id: "r-a", status: "completed" }, response_epoch: 40,
}));
await promotedClient._onWsMessage(JSON.stringify({
  type: "pipeline.response", response_id: "r-b", input_epoch: 41, response_epoch: 41,
}));
promotedClient._onPlaybackMessage({
  kind: "drained", generation: 0, streamId: "r-a", responseEpoch: 40, queueEmpty: true,
});
if (promotedClient._playbackByResponse.has("r-a") || promotedClient._aiSpeaking) {
  throw new Error("newer response promotion stranded the older browser playback drain");
}
if (promotedFinished.filter((detail) => detail.responseId === "r-a").length !== 1
    || promotedSent.filter((message) => message.type === "pipeline.playback.started"
      && message.response_id === "r-a").length !== 1) {
  throw new Error("older admitted playback did not settle exactly once after newer promotion");
}

// A definitive worklet clear is intentionally unscoped (`responseEpoch:null`)
// and can invalidate more than one completed-but-unheard stream in the shared
// generation. Number(null) must not turn it into stale epoch zero, and every
// affected terminal must settle exactly once as unheard.
const clearClient = new S2sWsRealtimeClient({});
const clearPosted = [];
const clearFinished = [];
clearClient._playbackNode = { port: { postMessage: (message) => clearPosted.push(message) } };
clearClient.addEventListener("response-finished", (event) => clearFinished.push(event.detail));
for (const [responseId, responseEpoch] of [["r-clear-one", 20], ["r-clear-two", 21]]) {
  await clearClient._onWsMessage(JSON.stringify({
    type: "response.created", response: { id: responseId }, response_epoch: responseEpoch,
  }));
  await clearClient._onWsMessage(JSON.stringify({
    type: "pipeline.response", response_id: responseId,
    input_epoch: responseEpoch, response_epoch: responseEpoch,
  }));
  await clearClient._onWsMessage(JSON.stringify({
    type: "response.audio.delta", response_id: responseId,
    response_epoch: responseEpoch, delta: "AAAAAA==",
  }));
  await clearClient._onWsMessage(JSON.stringify({
    type: "response.done", response: { id: responseId, status: "completed" },
    response_epoch: responseEpoch,
  }));
}
clearClient._invalidatePlayback("test-clear");
clearClient._onPlaybackMessage({
  kind: "cleared", generation: 1, responseEpoch: null, reason: "test-clear",
});
clearClient._onPlaybackMessage({
  kind: "drained", generation: 1, responseEpoch: null,
  cleared: true, reason: "test-clear", streamId: "r-clear-one",
});
if (clearFinished.length !== 2 || clearFinished.some((detail) => detail.audible !== false)) {
  throw new Error("generation clear did not settle every unheard response exactly once");
}
if (clearClient._playbackByResponse.size !== 0 || clearClient._aiSpeaking) {
  throw new Error("generation clear retained stale playback or speaking state");
}

// The local speech-start decision must preserve an audible response when
// interrupt_response is disabled. Stock speech_started alone remains the
// compatibility barge-in path for older peers.
const retentionClient = new S2sWsRealtimeClient({});
const retentionPosted = [];
retentionClient._playbackNode = { port: { postMessage: (message) => retentionPosted.push(message) } };
await retentionClient._onWsMessage(JSON.stringify({
  type: "response.created", response: { id: "r-retained" }, response_epoch: 30,
}));
await retentionClient._onWsMessage(JSON.stringify({
  type: "pipeline.response", response_id: "r-retained", input_epoch: 30, response_epoch: 30,
}));
await retentionClient._onWsMessage(JSON.stringify({
  type: "response.audio.delta", response_id: "r-retained", response_epoch: 30, delta: "AAAAAA==",
}));
await retentionClient._onWsMessage(JSON.stringify({
  type: "pipeline.input_audio.speech_started", input_epoch: 31, response_epoch: 30,
  effective_interrupt: false, reason: "interrupt_disabled",
}));
await retentionClient._onWsMessage(JSON.stringify({
  type: "input_audio_buffer.speech_started", input_epoch: 31, item_id: "item-retained",
}));
await retentionClient._onWsMessage(JSON.stringify({
  type: "response.audio.delta", response_id: "r-retained", response_epoch: 30, delta: "AAAAAA==",
}));
if (retentionClient._playbackGeneration !== 0
    || retentionPosted.some((message) => message.kind === "clear")
    || retentionPosted.filter((message) => message.kind === "audio").length !== 2) {
  throw new Error("interrupt-disabled speech start cleared or rejected retained response PCM");
}
await retentionClient._onWsMessage(JSON.stringify({
  type: "pipeline.input_audio.speech_started", input_epoch: 32, response_epoch: 30,
  effective_interrupt: true, reason: "barge_in",
}));
await retentionClient._onWsMessage(JSON.stringify({
  type: "input_audio_buffer.speech_started", input_epoch: 32, item_id: "item-barge",
}));
if (retentionClient._playbackGeneration !== 1
    || !retentionPosted.some((message) => message.kind === "clear" && message.generation === 1)) {
  throw new Error("authoritative barge-in decision did not invalidate playback exactly once");
}

// Adaptive feedback is restricted to the bounded browser-side reservoir. It
// must neither retime the source clock nor hide a provider that continues to
// underrun at the two-second maximum; Fast Start is unaffected by this path.
const adaptiveClient = new S2sWsRealtimeClient({});
const adaptiveMetrics = [];
adaptiveClient.addEventListener("pipeline-metric", (event) => adaptiveMetrics.push(event.detail));
adaptiveClient._playbackContinuityMode = "adaptive";
adaptiveClient._playbackPrimeMs = 160;
adaptiveClient._adaptivePlaybackPrimeMs = 160;
adaptiveClient._playbackSampleRate = 24_000;
adaptiveClient._acknowledgedPlaybackConfig = {
  provider: "qwen3tts-audiocpp", profileId: "low-latency", profileRevision: null,
  nativeStreaming: true, resolvedPrimeMs: 160,
};
const adaptiveSnapshot = {
  continuityMode: "adaptive", provider: "qwen3tts-audiocpp", profileId: "low-latency",
  profileRevision: null, nativeStreaming: true, playbackUnsustainable: false,
  configPrimeMs: 160, configToken: adaptiveClient._playbackConfigToken,
};
adaptiveClient._raiseAdaptiveReserve({ maxObservedChunkGapMs: 2_500 }, adaptiveSnapshot);
if (adaptiveClient._adaptivePlaybackPrimeMs !== 2_000 || adaptiveClient._playbackSampleRate !== 24_000) {
  throw new Error("Adaptive feedback changed more than its capped PCM reservoir");
}
if (!adaptiveClient._playbackUnsustainable
    || !adaptiveMetrics.some((metric) => metric.status === "provider_unsustainable"
      && metric.detail?.message?.includes("Provider cannot sustain realtime"))) {
  throw new Error("a first 2.5-second gap was not surfaced as unsustainable at the two-second reservoir cap");
}

// Feedback arriving from an older frozen profile or delivery mode cannot tune
// the newly acknowledged mode's reservoir or borrow its unsustainable label.
const scopedFeedbackClient = new S2sWsRealtimeClient({});
const scopedFeedbackMetrics = [];
scopedFeedbackClient.addEventListener("pipeline-metric", (event) => scopedFeedbackMetrics.push(event.detail));
scopedFeedbackClient._playbackContinuityMode = "adaptive";
scopedFeedbackClient._playbackPrimeMs = 240;
scopedFeedbackClient._adaptivePlaybackPrimeMs = 240;
scopedFeedbackClient._healthyPlaybackResponses = 3;
scopedFeedbackClient._acknowledgedPlaybackConfig = {
  provider: "qwen3tts-audiocpp", profileId: "balanced", profileRevision: 7,
  nativeStreaming: false, resolvedPrimeMs: 240,
};
scopedFeedbackClient._playbackByResponse.set("r-old-native", {
  generation: 0, primeMs: 160, reprimeMs: 80, continuityMode: "adaptive",
  nativeStreaming: true, maxPrimeMs: 2_000, provider: "qwen3tts-audiocpp",
  profileId: "balanced", profileRevision: 7, responseEpoch: 90, sourceSampleRate: 24_000,
  configPrimeMs: 240, configToken: scopedFeedbackClient._playbackConfigToken,
  audioStarted: true, ended: false, playbackAcked: true, playbackDrained: false,
  playbackCleared: false, clearReason: "", drainAudible: true,
  playbackUnsustainable: false, terminal: false, terminalSettled: false,
  pendingTerminalDetail: null,
});
scopedFeedbackClient._onPlaybackMessage({
  kind: "underrun", generation: 0, streamId: "r-old-native", responseEpoch: 90,
  primeTargetMs: 0, maxObservedChunkGapMs: 1_400,
});
if (scopedFeedbackClient._adaptivePlaybackPrimeMs !== 240
    || scopedFeedbackClient._playbackUnsustainable
    || scopedFeedbackClient._healthyPlaybackResponses !== 3) {
  throw new Error("late native-mode feedback mutated the selected buffered-mode reservoir");
}
const oldMetric = scopedFeedbackMetrics.at(-1)?.detail;
if (oldMetric?.provider !== "qwen3tts-audiocpp" || oldMetric?.profile_id !== "balanced"
    || oldMetric?.provider_unsustainable !== false || oldMetric?.adaptive_prime_target_ms !== 80) {
  throw new Error("late playback metric mixed frozen response identity with mutable mode health");
}

// A worklet clear can beat the backend's cancellation terminal. Preserve the
// definitive clear result until response.done arrives, even if a newer epoch is
// promoted in between, so the provisional ChatView row settles exactly once.
const clearBeforeDoneClient = new S2sWsRealtimeClient({});
const clearBeforeDoneFinished = [];
const clearBeforeDoneMetrics = [];
clearBeforeDoneClient._playbackNode = { port: { postMessage: () => {} } };
clearBeforeDoneClient._send = () => {};
clearBeforeDoneClient.addEventListener("response-finished", (event) => clearBeforeDoneFinished.push(event.detail));
clearBeforeDoneClient.addEventListener("pipeline-metric", (event) => clearBeforeDoneMetrics.push(event.detail));
await clearBeforeDoneClient._onWsMessage(JSON.stringify({
  type: "response.created", response: { id: "r-clear-first" }, response_epoch: 100,
}));
await clearBeforeDoneClient._onWsMessage(JSON.stringify({
  type: "pipeline.response", response_id: "r-clear-first", input_epoch: 100, response_epoch: 100,
}));
await clearBeforeDoneClient._onWsMessage(JSON.stringify({
  type: "response.audio.delta", response_id: "r-clear-first", response_epoch: 100, delta: "AAAAAA==",
}));
clearBeforeDoneClient._onPlaybackMessage({
  kind: "started", generation: 0, streamId: "r-clear-first", responseEpoch: 100,
});
clearBeforeDoneClient._invalidatePlayback("barge-in");
clearBeforeDoneClient._onPlaybackMessage({
  kind: "cleared", generation: 1, responseEpoch: null, reason: "barge-in",
});
if (!clearBeforeDoneClient._playbackByResponse.get("r-clear-first")?.playbackCleared) {
  throw new Error("clear-before-terminal discarded its definitive response outcome");
}
await clearBeforeDoneClient._onWsMessage(JSON.stringify({
  type: "pipeline.response", response_id: "r-successor", input_epoch: 101, response_epoch: 101,
}));
await clearBeforeDoneClient._onWsMessage(JSON.stringify({
  type: "response.done", response: { id: "r-clear-first", status: "cancelled" }, response_epoch: 100,
}));
if (clearBeforeDoneFinished.length !== 1
    || clearBeforeDoneFinished[0].responseId !== "r-clear-first"
    || clearBeforeDoneFinished[0].audible !== true
    || clearBeforeDoneClient._playbackByResponse.has("r-clear-first")) {
  throw new Error("clear-before-terminal did not settle and retire the interrupted response exactly once");
}
if (clearBeforeDoneMetrics.some((metric) => metric.stage === "response"
    && metric.status === "done" && metric.detail?.response_epoch === 100)) {
  throw new Error("stale clear-before-terminal response contaminated current response metrics");
}

// The worklet's first sample can render before a barge-in even when its
// cross-thread `started` message reaches this task after the cancellation.
// Retain the exact old snapshot until clear/drain proves the outcome; never
// revive its network PCM or acknowledge it twice.
const delayedStartClient = new S2sWsRealtimeClient({});
const delayedStartSent = [];
const delayedStartFinished = [];
const delayedStartPosted = [];
const delayedStartMetrics = [];
delayedStartClient._playbackNode = { port: { postMessage: (message) => delayedStartPosted.push(message) } };
delayedStartClient._send = (message) => delayedStartSent.push(message);
delayedStartClient.addEventListener("response-finished", (event) => delayedStartFinished.push(event.detail));
delayedStartClient.addEventListener("pipeline-metric", (event) => delayedStartMetrics.push(event.detail));
await delayedStartClient._onWsMessage(JSON.stringify({
  type: "response.created", response: { id: "r-delayed-start" }, response_epoch: 110,
}));
await delayedStartClient._onWsMessage(JSON.stringify({
  type: "pipeline.response", response_id: "r-delayed-start", input_epoch: 110, response_epoch: 110,
}));
await delayedStartClient._onWsMessage(JSON.stringify({
  type: "response.audio.delta", response_id: "r-delayed-start", response_epoch: 110, delta: "AAAAAA==",
}));
delayedStartClient._invalidatePlayback("barge-in");
delayedStartClient._setStatus("user-speaking");
await delayedStartClient._onWsMessage(JSON.stringify({
  type: "pipeline.response", response_id: "r-after-delayed", input_epoch: 111, response_epoch: 111,
}));
await delayedStartClient._onWsMessage(JSON.stringify({
  type: "response.done", response: { id: "r-delayed-start", status: "cancelled" }, response_epoch: 110,
}));
if (!delayedStartClient._playbackByResponse.has("r-delayed-start") || delayedStartFinished.length !== 0) {
  throw new Error("cancel terminal retired potential first-render proof before worklet ordering settled");
}
delayedStartClient._onPlaybackMessage({
  kind: "started", generation: 0, streamId: "r-delayed-start", responseEpoch: 110,
});
if (delayedStartFinished.length !== 1 || delayedStartFinished[0].audible !== true
    || delayedStartSent.filter((message) => message.type === "pipeline.playback.started"
      && message.response_id === "r-delayed-start").length !== 1) {
  throw new Error("delayed genuine first-render proof was not committed exactly once");
}
if (delayedStartClient._status !== "user-speaking" || delayedStartClient._aiSpeaking
    || delayedStartClient._firstPlaybackReported
    || delayedStartMetrics.some((metric) => metric.stage === "playback" && metric.status === "first_audio")) {
  throw new Error("historical first-render proof contaminated the successor presentation or metrics");
}
await delayedStartClient._onWsMessage(JSON.stringify({
  type: "response.audio.delta", response_id: "r-delayed-start", response_epoch: 110, delta: "AAAAAA==",
}));
if (delayedStartPosted.filter((message) => message.kind === "audio").length !== 1) {
  throw new Error("delayed render proof reopened obsolete network PCM admission");
}
delayedStartClient._onPlaybackMessage({
  kind: "cleared", generation: 1, responseEpoch: null, reason: "barge-in",
});
if (delayedStartClient._playbackByResponse.has("r-delayed-start")) {
  throw new Error("cleared delayed-start snapshot was not retired");
}
if (delayedStartClient._status !== "user-speaking") {
  throw new Error("old-generation clear overwrote the active user-speaking state");
}

// The cancellation terminal can itself be the first playback-generation
// invalidation. A first-render notification already posted by the old worklet
// must still settle the interrupted response as audible exactly once.
const terminalFirstClient = new S2sWsRealtimeClient({});
const terminalFirstSent = [];
const terminalFirstFinished = [];
terminalFirstClient._playbackNode = { port: { postMessage: () => {} } };
terminalFirstClient._send = (message) => terminalFirstSent.push(message);
terminalFirstClient.addEventListener("response-finished", (event) => terminalFirstFinished.push(event.detail));
await terminalFirstClient._onWsMessage(JSON.stringify({
  type: "response.created", response: { id: "r-terminal-first" }, response_epoch: 120,
}));
await terminalFirstClient._onWsMessage(JSON.stringify({
  type: "pipeline.response", response_id: "r-terminal-first", input_epoch: 120, response_epoch: 120,
}));
await terminalFirstClient._onWsMessage(JSON.stringify({
  type: "response.audio.delta", response_id: "r-terminal-first", response_epoch: 120, delta: "AAAAAA==",
}));
await terminalFirstClient._onWsMessage(JSON.stringify({
  type: "response.done", response: { id: "r-terminal-first", status: "cancelled" }, response_epoch: 120,
}));
if (!terminalFirstClient._playbackByResponse.has("r-terminal-first")
    || terminalFirstClient._playbackGeneration !== 1
    || terminalFirstFinished.length !== 0) {
  throw new Error("terminal-first invalidation discarded pending first-render proof");
}
terminalFirstClient._onPlaybackMessage({
  kind: "started", generation: 0, streamId: "r-terminal-first", responseEpoch: 120,
});
if (terminalFirstFinished.length !== 1 || terminalFirstFinished[0].audible !== true
    || terminalFirstSent.filter((message) => message.type === "pipeline.playback.started"
      && message.response_id === "r-terminal-first").length !== 1) {
  throw new Error("terminal-first delayed render was not acknowledged and settled exactly once");
}
terminalFirstClient._onPlaybackMessage({
  kind: "cleared", generation: 1, responseEpoch: null, reason: "response-cancelled",
});
if (terminalFirstClient._playbackByResponse.has("r-terminal-first")) {
  throw new Error("terminal-first delayed-render snapshot was not retired after clear");
}
console.log("ws epoch dispatch passed");
