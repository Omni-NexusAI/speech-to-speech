// @ts-check

/** Canonical arguments advertised to the Realtime model for browser search. */
export const WEB_SEARCH_ARGUMENT_SCHEMA = Object.freeze({
  type: "object",
  properties: {
    query: {
      type: "string",
      minLength: 1,
      maxLength: 500,
      pattern: "\\S",
      description: "The search query.",
    },
    mode: {
      type: "string",
      enum: ["auto", "web", "news"],
      description: "Use news for news coverage, web for general pages, or auto to infer from freshness.",
    },
    freshness: {
      type: "string",
      enum: ["none", "day", "week", "month", "year"],
      description: "Optional maximum result age. Use none when the request does not require recency.",
    },
  },
  required: ["query"],
  additionalProperties: false,
});

const FRESHNESS_RANK = Object.freeze({ none: 0, year: 1, month: 2, week: 3, day: 4 });

/**
 * @typedef {{ query: string, mode: "auto"|"web"|"news", freshness: "none"|"day"|"week"|"month"|"year" }} SearchArguments
 */

/** @param {Record<string, unknown>} args @returns {SearchArguments} */
export function canonicalSearchArguments(args) {
  return {
    query: typeof args.query === "string" ? args.query.trim() : "",
    mode: args.mode === "web" || args.mode === "news" ? args.mode : "auto",
    freshness: args.freshness === "day" || args.freshness === "week" ||
      args.freshness === "month" || args.freshness === "year" ? args.freshness : "none",
  };
}

/** @param {string} query @returns {string[]} */
function queryTerms(query) {
  return query
    .normalize("NFKC")
    .toLocaleLowerCase()
    .split(/[^\p{L}\p{N}]+/u)
    .filter(Boolean);
}

/** @param {SearchArguments} search @returns {"web"|"news"} */
function effectiveMode(search) {
  if (search.mode === "auto") return search.freshness === "none" ? "web" : "news";
  return search.mode;
}

/**
 * A refinement must retain the original terms and add a constraint, tighten
 * freshness, or narrow a general search to news. This is deliberately
 * deterministic: the browser never guesses semantic similarity from content.
 * @param {SearchArguments} original
 * @param {SearchArguments} candidate
 */
export function isDistinctNarrowerSearch(original, candidate) {
  const originalTerms = queryTerms(original.query);
  const candidateTerms = queryTerms(candidate.query);
  if (!originalTerms.length || !candidateTerms.length) return false;
  const originalSet = new Set(originalTerms);
  const candidateSet = new Set(candidateTerms);
  const retainsOriginal = [...originalSet].every((term) => candidateSet.has(term));
  if (!retainsOriginal) return false;

  const originalMode = effectiveMode(original);
  const candidateMode = effectiveMode(candidate);
  const preservesModeNarrowness = originalMode !== "news" || candidateMode === "news";
  const preservesFreshness = FRESHNESS_RANK[candidate.freshness] >= FRESHNESS_RANK[original.freshness];
  if (!preservesModeNarrowness || !preservesFreshness) return false;

  const addsQueryConstraint = candidateSet.size > originalSet.size;
  const tightensFreshness = FRESHNESS_RANK[candidate.freshness] > FRESHNESS_RANK[original.freshness];
  const narrowsToNews = originalMode === "web" && candidateMode === "news";
  return addsQueryConstraint || tightensFreshness || narrowsToNews;
}

/**
 * Browser-side, accepted-turn search budget. It permits one initial search and
 * at most one deterministic narrower refinement. A rejected follow-up is
 * terminal as well, preventing a model-driven retry loop.
 */
export class SearchTurnPolicy {
  constructor() {
    /** @type {SearchArguments | null} */
    this.initial = null;
    this.calls = 0;
  }

  reset() {
    this.initial = null;
    this.calls = 0;
  }

  /**
   * @param {Record<string, unknown>} rawArgs
   * @returns {{ accepted: true, args: SearchArguments, terminal: boolean, refinement: boolean } |
   *   { accepted: false, args: SearchArguments, terminal: true, reason: "limit_reached"|"not_narrower" }}
   */
  accept(rawArgs) {
    const args = canonicalSearchArguments(rawArgs);
    if (this.calls === 0) {
      this.initial = args;
      this.calls = 1;
      return { accepted: true, args, terminal: false, refinement: false };
    }
    if (this.calls >= 2 || !this.initial) {
      return { accepted: false, args, terminal: true, reason: "limit_reached" };
    }

    this.calls = 2;
    if (!isDistinctNarrowerSearch(this.initial, args)) {
      return { accepted: false, args, terminal: true, reason: "not_narrower" };
    }
    return { accepted: true, args, terminal: true, refinement: true };
  }
}

/** @param {"limit_reached"|"not_narrower"} reason @returns {string} */
export function searchPolicyOutput(reason) {
  return JSON.stringify({
    type: "web_search_rejected",
    schema_version: 1,
    reason,
    terminal: true,
  });
}
