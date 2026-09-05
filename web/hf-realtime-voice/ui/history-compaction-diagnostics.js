/** Render only acknowledged backend history-compaction diagnostics. */

function finite(value) {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1e9 ? value : null;
}

function tokens(value) {
  const number = finite(value);
  return number == null ? null : Math.round(number).toLocaleString();
}

/**
 * Format a context metric without estimating any budget or outcome locally.
 * The backend owns the policy, status, failures, and budget numbers.
 */
export function formatHistoryCompactionDiagnostics({ configured, acknowledged, contextDetail } = {}) {
  const reported = contextDetail?.history_compaction;
  const policy = reported?.policy && typeof reported.policy === "object"
    ? reported.policy
    : acknowledged || configured;
  const active = policy && typeof policy === "object" ? policy : {};
  const summary = `${active.enabled !== false ? "Enabled" : "Disabled"}` +
    ` · trigger ${Math.round(Number(active.trigger_ratio) * 100)}%` +
    ` · target ${Math.round(Number(active.target_ratio) * 100)}%` +
    ` · retain ${Math.round(Number(active.recent_turns))} turns` +
    (reported?.policy ? " · backend policy" : acknowledged ? " · acknowledged" : " · pending acknowledgement");

  const status = reported?.status && typeof reported.status === "object" ? reported.status : null;
  const history = tokens(contextDetail?.history_tokens);
  const source = typeof contextDetail?.token_source === "string" ? ` (${contextDetail.token_source})` : "";
  if (status) {
    const outcome = typeof status.status === "string" ? status.status : "idle";
    const failure = typeof status.last_failure === "string" && status.last_failure ? ` · last failure ${status.last_failure}` : "";
    const budget = status.budget && typeof status.budget === "object" ? status.budget : null;
    const budgetParts = budget ? [
      ["request", tokens(budget.estimated_request_tokens)],
      ["trigger", tokens(budget.trigger_tokens)],
      ["target", tokens(budget.target_tokens)],
      ["window", tokens(budget.context_window)],
    ].filter(([, value]) => value != null).map(([label, value]) => `${label} ${value}`).join(" · ") : "";
    const historyText = history == null ? "" : ` Retained history ${history} tokens${source}.`;
    const budgetText = budgetParts ? ` Budget: ${budgetParts}.` : "";
    return { summary, status: `Backend compaction ${outcome}${failure}.${historyText}${budgetText}` };
  }
  if (history != null) {
    return { summary, status: `Backend reports retained history ${history} tokens${source}.` };
  }
  return {
    summary,
    status: acknowledged ? "Configuration acknowledged. Awaiting a backend context measurement." : "No backend usage measurement has arrived.",
  };
}
