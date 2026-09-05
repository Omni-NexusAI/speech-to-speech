import assert from "node:assert/strict";

const { formatHistoryCompactionDiagnostics } = await import(
  "../web/hf-realtime-voice/ui/history-compaction-diagnostics.js"
);

const rendered = formatHistoryCompactionDiagnostics({
  configured: { enabled: true, trigger_ratio: 0.7, target_ratio: 0.5, recent_turns: 6 },
  acknowledged: { enabled: true, trigger_ratio: 0.8, target_ratio: 0.6, recent_turns: 4 },
  contextDetail: {
    history_tokens: 2_048,
    token_source: "serialized_char_estimate",
    history_compaction: {
      policy: { enabled: true, trigger_ratio: 0.7, target_ratio: 0.5, recent_turns: 6 },
      status: {
        status: "failed",
        last_failure: "target_not_met",
        budget: { estimated_request_tokens: 4_096, trigger_tokens: 5_734, target_tokens: 4_096, context_window: 8_192 },
      },
    },
  },
});

assert.equal(rendered.summary, "Enabled · trigger 70% · target 50% · retain 6 turns · backend policy");
assert.match(rendered.status, /Backend compaction failed · last failure target_not_met/);
assert.match(rendered.status, /Retained history 2,048 tokens \(serialized_char_estimate\)/);
assert.match(rendered.status, /Budget: request 4,096 · trigger 5,734 · target 4,096 · window 8,192/);
assert.doesNotMatch(rendered.status, /No compaction budget or result/);

const scheduled = formatHistoryCompactionDiagnostics({
  configured: { enabled: true, trigger_ratio: 0.7, target_ratio: 0.5, recent_turns: 6 },
  contextDetail: { history_compaction: { policy: { enabled: false, trigger_ratio: 0.8, target_ratio: 0.6, recent_turns: 4 }, status: { status: "scheduled", last_failure: null } } },
});
assert.equal(scheduled.summary, "Disabled · trigger 80% · target 60% · retain 4 turns · backend policy");
assert.equal(scheduled.status, "Backend compaction scheduled.");

const unknown = formatHistoryCompactionDiagnostics({
  configured: { enabled: true, trigger_ratio: 0.7, target_ratio: 0.5, recent_turns: 6 },
  contextDetail: { history_tokens: null, history_compaction: { status: {
    status: "unavailable_context", budget: { estimated_request_tokens: null, trigger_tokens: false,
      target_tokens: -1, context_window: Infinity }
  } } }
});
assert.equal(unknown.status, "Backend compaction unavailable_context.");

console.log("history compaction diagnostics tests passed");
