import assert from "node:assert/strict";

import { runtimeIdentityMatches } from "../web/hf-realtime-voice/runtime-identity.js";

const identity = {
  source_revision: "a".repeat(40),
  source_dirty: false,
  source_fingerprint: "b".repeat(64),
  ui_asset_generation: "main=34-opaque-echo-route;ws=21-opaque-echo-route;chat=5-opaque-echo-route;playback=16-adaptive-safe-start",
};
assert.equal(runtimeIdentityMatches(identity, { ...identity }), true);
for (const [field, value] of [
  ["source_revision", "c".repeat(40)],
  ["source_dirty", true],
  ["source_fingerprint", "d".repeat(64)],
  ["ui_asset_generation", identity.ui_asset_generation.replace("main=34", "main=35")],
]) {
  assert.equal(runtimeIdentityMatches(identity, { ...identity, [field]: value }), false, field);
}
assert.equal(runtimeIdentityMatches(identity, { ...identity, source_revision: "A".repeat(40) }), false);
assert.equal(runtimeIdentityMatches(identity, { ...identity, source_dirty: "false" }), false);
assert.equal(runtimeIdentityMatches(identity, { ...identity, source_fingerprint: "unknown" }), false);
assert.equal(runtimeIdentityMatches(identity, { ...identity, ui_asset_generation: "unknown" }), false);
assert.equal(runtimeIdentityMatches(identity, null), false);
console.log("runtime identity tests passed");
