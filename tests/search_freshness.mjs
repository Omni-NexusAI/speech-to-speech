import assert from "node:assert/strict";

import {
  SearchTurnPolicy,
  WEB_SEARCH_ARGUMENT_SCHEMA,
  canonicalSearchArguments,
  isDistinctNarrowerSearch,
  searchPolicyOutput,
} from "../web/hf-realtime-voice/tools/web-search.js";

assert.deepEqual(WEB_SEARCH_ARGUMENT_SCHEMA.required, ["query"]);
assert.equal(WEB_SEARCH_ARGUMENT_SCHEMA.additionalProperties, false);
assert.equal(WEB_SEARCH_ARGUMENT_SCHEMA.properties.query.minLength, 1);
assert.equal(WEB_SEARCH_ARGUMENT_SCHEMA.properties.query.maxLength, 500);
assert.equal(WEB_SEARCH_ARGUMENT_SCHEMA.properties.query.pattern, "\\S");
assert.deepEqual(WEB_SEARCH_ARGUMENT_SCHEMA.properties.mode.enum, ["auto", "web", "news"]);
assert.deepEqual(
  WEB_SEARCH_ARGUMENT_SCHEMA.properties.freshness.enum,
  ["none", "day", "week", "month", "year"],
);

assert.deepEqual(canonicalSearchArguments({ query: "  climate policy  " }), {
  query: "climate policy",
  mode: "auto",
  freshness: "none",
});
assert.deepEqual(canonicalSearchArguments({
  query: "latest climate policy",
  mode: "news",
  freshness: "week",
}), {
  query: "latest climate policy",
  mode: "news",
  freshness: "week",
});

assert.equal(isDistinctNarrowerSearch(
  { query: "climate policy", mode: "auto", freshness: "none" },
  { query: "climate policy", mode: "auto", freshness: "week" },
), true, "auto plus a freshness filter narrows web to recent news");
assert.equal(isDistinctNarrowerSearch(
  { query: "climate policy", mode: "auto", freshness: "week" },
  { query: "climate policy Canada", mode: "web", freshness: "week" },
), false, "an added term cannot compensate for broadening effective news to web");
assert.equal(isDistinctNarrowerSearch(
  { query: "climate policy", mode: "news", freshness: "week" },
  { query: "CLIMATE, policy!", mode: "news", freshness: "week" },
), false, "normalized duplicates are not refinements");
assert.equal(isDistinctNarrowerSearch(
  { query: "climate policy Canada", mode: "news", freshness: "week" },
  { query: "climate policy", mode: "news", freshness: "day" },
), false, "dropping an original term is broader even with tighter freshness");

const policy = new SearchTurnPolicy();
assert.deepEqual(policy.accept({ query: "climate policy" }), {
  accepted: true,
  args: { query: "climate policy", mode: "auto", freshness: "none" },
  terminal: false,
  refinement: false,
});
assert.deepEqual(policy.accept({
  query: "climate policy Canada",
  mode: "news",
  freshness: "week",
}), {
  accepted: true,
  args: { query: "climate policy Canada", mode: "news", freshness: "week" },
  terminal: true,
  refinement: true,
});
assert.deepEqual(policy.accept({ query: "climate policy Canada 2026" }), {
  accepted: false,
  args: { query: "climate policy Canada 2026", mode: "auto", freshness: "none" },
  terminal: true,
  reason: "limit_reached",
});

const duplicate = new SearchTurnPolicy();
duplicate.accept({ query: "Weather in Boston", mode: "web", freshness: "week" });
const duplicateDecision = duplicate.accept({
  query: "WEATHER, in Boston!",
  mode: "web",
  freshness: "week",
});
assert.equal(duplicateDecision.accepted, false);
assert.equal(duplicateDecision.reason, "not_narrower");
assert.deepEqual(JSON.parse(searchPolicyOutput(duplicateDecision.reason)), {
  type: "web_search_rejected",
  schema_version: 1,
  reason: "not_narrower",
  terminal: true,
});
assert.equal(searchPolicyOutput(duplicateDecision.reason).includes("Boston"), false);

duplicate.reset();
assert.equal(duplicate.accept({ query: "new turn" }).accepted, true);

console.log("search freshness policy tests passed");
