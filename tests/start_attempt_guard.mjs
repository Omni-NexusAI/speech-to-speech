// Start attempts are browser UI work, so exercise their monotonic token without
// importing main.js (which requires a DOM). The static assertions below ensure
// main.js keeps using the guard at each asynchronous startup boundary.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const { StartAttemptController } = await import("../web/hf-realtime-voice/ui/start-attempt.js");

const attempts = new StartAttemptController();
let releasePreflight;
const delayedPreflight = new Promise((resolve) => { releasePreflight = resolve; });
let staleClientClosed = 0;
let oldAttemptDialed = 0;

const oldAttempt = attempts.claim();
const oldStart = (async () => {
  await delayedPreflight;
  if (!attempts.isCurrent(oldAttempt)) {
    staleClientClosed += 1;
    const error = new Error("superseded start");
    error.code = "aborted";
    throw error;
  }
  oldAttemptDialed += 1;
})();

// Simulate a fast second click while the first backend readiness probe is slow.
const newAttempt = attempts.claim();
releasePreflight();
await assert.rejects(oldStart, (error) => error?.code === "aborted");
assert.equal(staleClientClosed, 1, "a delayed obsolete start must clean itself up");
assert.equal(oldAttemptDialed, 0, "a delayed obsolete start must not create a second client/mic/socket");
assert.equal(attempts.isCurrent(newAttempt), true, "the newest start attempt remains eligible");

const main = await readFile(new URL("../web/hf-realtime-voice/main.js", import.meta.url), "utf8");
const startBody = main.slice(
  main.indexOf("async function doStart"),
  main.indexOf("// ── Conversation-time heartbeat"),
);
const firstAwait = startBody.indexOf("await ");
assert.ok(startBody.indexOf("const startAttempt = startAttempts.claim();") >= 0);
assert.ok(startBody.indexOf('setState("connecting");') >= 0);
assert.ok(startBody.indexOf("const startAttempt = startAttempts.claim();") < firstAwait,
  "doStart must claim its token before its first await");
assert.ok(startBody.indexOf('setState("connecting");') < firstAwait,
  "doStart must render connecting before its first await");
assert.equal((startBody.match(/await awaitStartPreflight\(/g) || []).length, 3,
  "all three async preflights must check for a superseding start");
assert.match(startBody, /await ensureCurrentStartAttempt\(startAttempt, audioContext, c\)/,
  "the post-connect checkpoint must close a stale candidate client");

console.log("start attempt guard tests passed");
