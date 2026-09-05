// @ts-check
/**
 * ChatView — owns the whole conversation surface: the slide-in history panel,
 * the ephemeral on-orb bubbles, and all the transcript/tool/streaming
 * bookkeeping. main.js wires the realtime client's events straight to the
 * `on*` methods here and otherwise doesn't touch chat state.
 *
 * Two parallel surfaces share one shape (see `_buildMessageEl`):
 *   - ephemeral bubbles  (`.bubble` / `.bubble-*`)  fade on a timer
 *   - persistent history (`.hist-msg` / `.hist-*`)  the durable panel log
 *
 * Keying:
 *   - user transcripts by the server's `item_id` — a speculative continuation
 *     REUSES it, so both segments land in one row/bubble; deltas are CUMULATIVE
 *     (each carries the full sentence so far), so we replace text wholesale.
 *   - assistant transcripts by `response_id`, so a speculative reply can be
 *     rolled back before rendered playback or marked interrupted once heard.
 */

import { $, escHtml, DEBUG } from "./dom.js";

const WRENCH_PATH = `<path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/>`;
const CHAT_BUBBLE_SVG = `<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>`;
const EMPTY_STATE_HTML = `<div id="chat-empty" class="chat-empty">${CHAT_BUBBLE_SVG}<span class="chat-empty-title">No messages yet</span><span class="chat-empty-hint">Tap the orb and start talking</span></div>`;

export class ChatView {
  constructor() {
    /** @type {HTMLButtonElement} */
    this._chatBtn = $("#chat-btn");
    /** @type {HTMLSpanElement} */
    this._chatBadge = $("#chat-badge");
    /** @type {HTMLDivElement} */
    this._chatPanel = $("#chat-panel");
    /** @type {HTMLDivElement} */
    this._chatPanelBackdrop = $("#chat-panel-backdrop");
    /** @type {HTMLButtonElement} */
    this._chatPanelClose = $("#chat-panel-close");
    /** @type {HTMLDivElement} */
    this._chatHistory = $("#chat-history");
    /** @type {HTMLDivElement} */
    this._bubbleStack = $("#bubble-stack");

    this._panelOpen = false;
    this._scrollQueued = false;

    // ── User transcript state (keyed by item_id) ───────────────────────────
    /** @type {Map<string, HTMLElement>} */
    this._userHistByItem = new Map();
    /** @type {HTMLElement | null} */
    this._activeUserBubble = null;
    this._activeUserItemId = "";
    /** @type {HTMLElement | null} */
    this._pendingUserHist = null;
    /** @type {{ hist: HTMLElement, itemId: string, bubble: HTMLElement | null, responseId: string, responseEpoch: number | null }[]} */
    this._userTurns = [];
    /** @type {Map<string, { hist: HTMLElement, itemId: string, bubble: HTMLElement | null, responseId: string, responseEpoch: number | null }>} */
    this._userTurnByItem = new Map();
    /** @type {Map<string, { hist: HTMLElement, itemId: string, bubble: HTMLElement | null, responseId: string, responseEpoch: number | null }>} */
    this._userTurnByResponse = new Map();
    // Monotonic counter for synthesizing unique keys when the server omits an
    // item_id / response_id, so id-less messages never collapse onto each other.
    this._anonSeq = 0;

    // ── Assistant transcript state (keyed by response_id) ──────────────────
    /** @type {Map<string, { bubble: HTMLElement, hist: HTMLElement }>} */
    this._asstByResp = new Map();
    /** @type {Map<string, HTMLElement>} */
    this._toolHistByCall = new Map();

    // ── Ephemeral bubble auto-dismiss ──────────────────────────────────────
    // Per-element expiry (epoch ms). A bubble fades once its expiry passes —
    // but only in stack order (see _reapBubbles). Refreshing the expiry keeps a
    // bubble alive while it updates (e.g. the live user utterance).
    /** @type {WeakMap<HTMLElement, number>} */
    this._bubbleExpiry = new WeakMap();
    // Single pending reaper handle: one timer for the whole stack (not one per
    // bubble) so dismissal is strictly oldest-first regardless of per-bubble delays.
    this._reaperHandle = 0;

    this._chatBtn.addEventListener("click", () => (this._panelOpen ? this._closePanel() : this._openPanel()));
    this._chatPanelClose.addEventListener("click", () => this._closePanel());
    this._chatPanelBackdrop.addEventListener("click", () => this._closePanel());
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && this._panelOpen) this._closePanel();
    });
  }

  // ── Panel ───────────────────────────────────────────────────────────────

  _openPanel() {
    this._panelOpen = true;
    this._chatPanel.classList.add("open");
    this._chatBadge.classList.remove("visible");
    this._scrollToBottom();
  }

  _closePanel() {
    this._panelOpen = false;
    this._chatPanel.classList.remove("open");
  }

  // Coalesce scroll-to-bottom: a burst of cumulative transcript deltas would
  // otherwise queue one rAF per delta, all writing the same scrollTop.
  _scrollToBottom() {
    if (!this._panelOpen || this._scrollQueued) return;
    this._scrollQueued = true;
    requestAnimationFrame(() => {
      this._scrollQueued = false;
      this._chatHistory.scrollTop = this._chatHistory.scrollHeight;
    });
  }

  _markUnread() {
    if (this._panelOpen) {
      this._scrollToBottom();
      return;
    }
    this._chatBadge.classList.add("visible");
  }

  // ── Shared rendering ──────────────────────────────────────────────────────

  /**
   * Build a role-labelled message element. Ephemeral bubbles and persistent
   * history rows share the same shape and differ only in their class prefix
   * (`bubble`/`bubble-*` vs `hist-msg`/`hist-*`).
   * @param {{ container: string, prefix: string, role: "user"|"assistant", text: string, partial?: boolean }} o
   * @returns {HTMLElement}
   */
  _buildMessageEl({ container, prefix, role, text, partial = false }) {
    const el = document.createElement("div");
    el.className = `${container} ${role}`;
    const label = role === "user" ? "You" : "Assistant";
    el.innerHTML = `<div class="${prefix}-role">${label}</div><div class="${prefix}-body${partial ? " partial" : ""}">${escHtml(text)}</div>`;
    return el;
  }

  // ── Ephemeral bubbles ───────────────────────────────────────────────────

  /** @param {"user"|"assistant"|"tool"} role @param {string} text @returns {HTMLElement} */
  _spawnBubble(role, text) {
    let el;
    if (role === "tool") {
      el = document.createElement("div");
      el.className = "bubble tool";
      el.innerHTML = `<svg class="bubble-tool-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${WRENCH_PATH}</svg><span class="bubble-tool-text">${escHtml(text)}</span>`;
    } else {
      el = this._buildMessageEl({ container: "bubble", prefix: "bubble", role, text });
    }
    this._bubbleStack.appendChild(el);
    // Cap the stack at 3, but never evict the bubble the caller is still
    // actively updating (the live user bubble) — drop the next-oldest instead.
    const visible = /** @type {HTMLElement[]} */ ([...this._bubbleStack.querySelectorAll(".bubble:not(.out)")]);
    if (visible.length > 3) {
      this._dismissBubble(visible.find((b) => b !== this._activeUserBubble) ?? visible[0]);
    }
    requestAnimationFrame(() => el.classList.add("in"));
    return el;
  }

  /** @param {HTMLElement} el @param {string} text */
  _updateBubbleText(el, text) {
    const t = el.querySelector(".bubble-body");
    if (t) t.textContent = text;
  }

  /** @param {HTMLElement} el */
  _dismissBubble(el) {
    if (!el || el.classList.contains("out")) return; // idempotent
    this._bubbleExpiry.delete(el);
    el.classList.remove("in");
    el.classList.add("out");
    const remove = () => el.remove();
    el.addEventListener("transitionend", remove, { once: true });
    // Fallback: transitionend never fires if the bubble's visual state didn't
    // change (dismissed pre-paint) or the tab is backgrounded. The transition
    // is 0.3s, so force removal a little after.
    setTimeout(remove, 400);
  }

  /**
   * Fade bubbles whose expiry has passed — strictly oldest-first. We walk the
   * stack top (oldest) to bottom (newest) and stop at the first bubble still
   * alive: nothing newer may leave while an older bubble is still on screen. A
   * bubble that keeps updating pushes its own expiry forward, so it (and
   * everything behind it) stays put until it finally goes quiet.
   */
  _reapBubbles() {
    this._reaperHandle = 0;
    const now = Date.now();
    const visible = /** @type {HTMLElement[]} */ ([...this._bubbleStack.querySelectorAll(".bubble:not(.out)")]);
    let nextWake = Infinity;
    for (const el of visible) {
      const exp = this._bubbleExpiry.get(el) ?? now; // no expiry recorded → treat as due
      if (exp <= now) {
        this._dismissBubble(el);
      } else {
        // Oldest survivor isn't due yet; stop so nothing newer leaves before it.
        nextWake = exp;
        break;
      }
    }
    if (nextWake !== Infinity) {
      this._reaperHandle = setTimeout(() => this._reapBubbles(), Math.max(50, nextWake - Date.now()));
    }
  }

  /**
   * (Re)arm a bubble's auto-dismiss by pushing its expiry out by `delay`.
   * Calling it again resets the countdown — so a bubble that keeps updating
   * stays on screen and only fades once it goes quiet. Removal is ordered by the
   * shared reaper, so the oldest bubble always disappears first.
   * @param {HTMLElement} el @param {number} [delay]
   */
  _bumpDismiss(el, delay = 4000) {
    this._bubbleExpiry.set(el, Date.now() + delay);
    if (!this._reaperHandle) this._reaperHandle = setTimeout(() => this._reapBubbles(), delay);
  }

  // ── History ───────────────────────────────────────────────────────────────

  /** Render the empty-state placeholder into the history panel. */
  renderEmptyState() {
    this._chatHistory.innerHTML = EMPTY_STATE_HTML;
  }

  /** Reset the panel to the empty state and clear the unread badge. */
  clear() {
    this.renderEmptyState();
    this._chatBadge.classList.remove("visible");
  }

  /** @param {"user"|"assistant"} role @param {string} text @param {boolean} partial @returns {HTMLElement} */
  _appendHistMsg(role, text, partial) {
    const empty = this._chatHistory.querySelector(".chat-empty");
    if (empty) empty.remove();
    const el = this._buildMessageEl({ container: "hist-msg", prefix: "hist", role, text, partial });
    this._chatHistory.appendChild(el);
    this._scrollToBottom();
    return el;
  }

  /** @param {HTMLElement | null} el @param {string} text @param {boolean} partial */
  _updateHistMsg(el, text, partial) {
    if (!el) return;
    const body = /** @type {HTMLElement | null} */ (el.querySelector(".hist-body"));
    if (!body) return;
    body.textContent = text;
    body.classList.toggle("partial", partial);
    this._scrollToBottom();
  }

  /**
   * Track a user row until its response is known to have reached the listener.
   * The realtime protocol serializes model ownership, so the oldest unbound
   * user row belongs to the next response even when a barge-in has already
   * reserved the following row before the old response terminal arrives.
   * @param {HTMLElement} hist @param {string} [itemId]
   */
  _trackUserTurn(hist, itemId = "") {
    let turn = itemId ? this._userTurnByItem.get(itemId) : undefined;
    if (!turn) turn = this._userTurns.find((candidate) => candidate.hist === hist);
    if (!turn) {
      turn = { hist, itemId, bubble: null, responseId: "", responseEpoch: null };
      this._userTurns.push(turn);
    } else if (itemId && !turn.itemId) {
      turn.itemId = itemId;
    }
    if (itemId) this._userTurnByItem.set(itemId, turn);
    return turn;
  }

  /** @param {string} responseId @param {number | null | undefined} responseEpoch */
  _bindResponseUserTurn(responseId, responseEpoch) {
    if (!responseId) return null;
    let turn = this._userTurnByResponse.get(responseId) ?? null;
    if (!turn) {
      turn = this._userTurns.find((candidate) => !candidate.responseId) ?? null;
      if (turn) {
        turn.responseId = responseId;
        this._userTurnByResponse.set(responseId, turn);
      }
    }
    if (turn && Number.isSafeInteger(Number(responseEpoch))) turn.responseEpoch = Number(responseEpoch);
    return turn;
  }

  /** @param {{ hist: HTMLElement, itemId: string, bubble: HTMLElement | null, responseId: string, responseEpoch: number | null } | null} turn */
  _settleUserTurn(turn) {
    if (!turn) return;
    this._userTurns = this._userTurns.filter((candidate) => candidate !== turn);
    if (turn.itemId && this._userTurnByItem.get(turn.itemId) === turn) this._userTurnByItem.delete(turn.itemId);
    if (turn.responseId && this._userTurnByResponse.get(turn.responseId) === turn) this._userTurnByResponse.delete(turn.responseId);
    if (this._pendingUserHist === turn.hist) this._pendingUserHist = null;
  }

  /** Remove a local, superseded turn before either side became audible. */
  _discardUnheardUserTurn(turn) {
    if (!turn) return;
    if (turn.bubble) this._dismissBubble(turn.bubble);
    if (this._activeUserBubble === turn.bubble) {
      this._activeUserBubble = null;
      this._activeUserItemId = "";
    }
    turn.hist.remove();
    if (turn.itemId && this._userHistByItem.get(turn.itemId) === turn.hist) this._userHistByItem.delete(turn.itemId);
    this._settleUserTurn(turn);
  }

  /** Remove the provisional assistant rendering for an unheard local response. */
  _discardUnheardResponse(responseId, entry) {
    if (entry) {
      this._dismissBubble(entry.bubble);
      entry.hist.remove();
    }
    this._asstByResp.delete(responseId);
  }

  _renderEmptyHistoryIfNeeded() {
    if (!this._chatHistory.querySelector(".hist-msg")) this.renderEmptyState();
  }

  /**
   * Append a tool-call row to the conversation. We only add it once the tool
   * has run, so the expandable toggle carries BOTH the call input and its result.
   * @param {string} name @param {string} argsJson @param {string} output
   */
  _appendHistTool(name, argsJson, output) {
    const empty = this._chatHistory.querySelector(".chat-empty");
    if (empty) empty.remove();
    let pretty = argsJson;
    try { pretty = JSON.stringify(JSON.parse(argsJson), null, 2); } catch {}
    const el = document.createElement("div");
    el.className = "hist-msg tool";
    el.innerHTML = `
      <div class="hist-role">Tool call</div>
      <button class="hist-tool-header" aria-expanded="false">
        <svg class="hist-tool-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${WRENCH_PATH}</svg>
        <span class="hist-tool-name">${escHtml(name)}</span>
        <svg class="hist-tool-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>
      </button>
      <div class="hist-tool-body">
        <div class="hist-tool-label">Input</div>
        <div class="hist-tool-block">${escHtml(pretty)}</div>
        <div class="hist-tool-label">Output</div>
        <div class="hist-tool-block hist-tool-output">${escHtml(output || "(no output)")}</div>
      </div>
    `;
    const header = /** @type {HTMLButtonElement} */ (el.querySelector(".hist-tool-header"));
    const body = /** @type {HTMLDivElement} */ (el.querySelector(".hist-tool-body"));
    header.addEventListener("click", () => {
      const expanded = header.getAttribute("aria-expanded") === "true";
      header.setAttribute("aria-expanded", String(!expanded));
      body.classList.toggle("open", !expanded);
    });
    this._chatHistory.appendChild(el);
    this._scrollToBottom();
    return el;
  }

  /** Tag an assistant history row as interrupted (user barged in mid-reply).
   *  @param {HTMLElement | null} hist */
  _markHistInterrupted(hist) {
    if (!hist || hist.querySelector(".hist-note")) return;
    hist.classList.add("interrupted");
    const note = document.createElement("div");
    note.className = "hist-note";
    note.textContent = "Interrupted";
    hist.appendChild(note);
  }

  /** Render a captured webcam frame in the transcript (the camera tool result).
   *  @param {string} dataUrl */
  _appendHistImage(dataUrl) {
    const empty = this._chatHistory.querySelector(".chat-empty");
    if (empty) empty.remove();
    const el = document.createElement("div");
    el.className = "hist-msg tool";
    el.innerHTML = `<div class="hist-role">Snapshot</div><img class="hist-image" alt="Webcam snapshot sent to the model" />`;
    const img = /** @type {HTMLImageElement} */ (el.querySelector("img"));
    img.src = dataUrl;
    this._chatHistory.appendChild(el);
    this._scrollToBottom();
  }

  /**
   * Reset all streaming bookkeeping for session start / teardown. Pass
   * `dismiss` to also fade any bubbles still on screen.
   * @param {{ dismiss?: boolean }} [opts]
   */
  reset(opts) {
    if (opts?.dismiss) {
      if (this._activeUserBubble) this._dismissBubble(this._activeUserBubble);
      for (const { bubble } of this._asstByResp.values()) this._dismissBubble(bubble);
    }
    this._userHistByItem.clear();
    this._activeUserBubble = null;
    this._activeUserItemId = "";
    this._pendingUserHist = null;
    this._userTurns = [];
    this._userTurnByItem.clear();
    this._userTurnByResponse.clear();
    this._asstByResp.clear();
    this._toolHistByCall.clear();
  }

  // ── Client event handlers ─────────────────────────────────────────────────

  /**
   * A streamed transcript delta (user or assistant).
   * @param {{ role: "user" | "assistant"; text: string; partial: boolean; itemId?: string; responseId?: string }} d
   */
  onTranscript(d, options = {}) {
    if (DEBUG) console.debug(`[ui] transcript role=${d.role} partial=${d.partial} item=${d.itemId} resp=${d.responseId} text=${JSON.stringify(d.text)}`);

    if (d.role === "user") {
      // Group by item_id: a speculative continuation reuses the same id, so it
      // updates the same row/bubble. A missing id falls back to the active item
      // (same utterance) or a fresh unique key, never a shared sentinel that
      // would collapse distinct turns into one row.
      const id = d.itemId || this._activeUserItemId || `_u${++this._anonSeq}`;
      const text = d.text;

      let hist = this._userHistByItem.get(id);
      if (!hist) {
        hist = this._pendingUserHist || this._appendHistMsg("user", text, d.partial);
        this._pendingUserHist = null;
        this._userHistByItem.set(id, hist);
      } else {
        this._updateHistMsg(hist, text, d.partial);
      }
      const userTurn = this._trackUserTurn(hist, id);

      // One ephemeral bubble per active item. Purely timer-based: the timer is
      // refreshed on every delta, so it stays while the user keeps talking and
      // fades a few seconds after they stop — no dependency on a response ever
      // arriving, so it can never get stuck.
      if (options.showUserBubble && (d.partial || (this._activeUserItemId === id && this._activeUserBubble))) {
        if (this._activeUserItemId !== id || !this._activeUserBubble) {
          this._activeUserBubble = this._spawnBubble("user", text);
          this._activeUserItemId = id;
        } else {
          this._updateBubbleText(this._activeUserBubble, text);
        }
        this._bumpDismiss(this._activeUserBubble, 6000);
        if (this._activeUserItemId === id) userTurn.bubble = this._activeUserBubble;
      }
      this._markUnread();
    } else if (d.role === "assistant") {
      // Assistant transcript arrives once, as the full text, keyed by
      // response_id so a cancelled speculative response can be removed later. A
      // missing id gets a unique key so two id-less replies never collide.
      const rid = d.responseId || `_a${++this._anonSeq}`;
      this._bindResponseUserTurn(rid, null);
      const entry = this._asstByResp.get(rid);
      if (!entry) {
        const bubble = this._spawnBubble("assistant", d.text);
        this._asstByResp.set(rid, { bubble, hist: this._appendHistMsg("assistant", d.text, false) });
        this._bumpDismiss(bubble);
      } else {
        this._updateBubbleText(entry.bubble, d.text);
        this._updateHistMsg(entry.hist, d.text, false);
        this._bumpDismiss(entry.bubble);
      }
      this._markUnread();
    }
  }

  /** Reserve chronology at speech stop before a direct-audio result arrives. */
  onUserTurnPending() {
    if (this._pendingUserHist) return;
    this._pendingUserHist = this._appendHistMsg("user", "Transcribing audio…", true);
    this._trackUserTurn(this._pendingUserHist);
    this._markUnread();
  }

  /**
   * A response closed (completed or cancelled).
   * @param {{ responseId: string; status: string; audible?: boolean; transcript?: string; responseEpoch?: number | null; committed?: boolean }} detail
   */
  onResponseFinished(detail) {
    const { responseId, status, audible, transcript, responseEpoch, committed } = detail;
    if (DEBUG) console.debug(`[ui] response-finished resp=${responseId} status=${status} audible=${audible} known=${this._asstByResp.has(responseId)}`);
    // `audible` is meaningful only for the local epoch-owned path. Standard
    // OpenAI-compatible text events omit an epoch and retain their historical
    // completion behavior even though they have no browser playback signal.
    const localEpoch = responseEpoch != null && Number.isSafeInteger(Number(responseEpoch));
    // A terminal local lifecycle may legitimately have no OpenAI response ID
    // (for example a cancelled generation before response.created). Its epoch
    // still owns the oldest provisional user row and must settle it.
    const terminalKey = responseId || (localEpoch ? `_response_epoch_${responseEpoch}` : "");
    if (!terminalKey) return;
    const entry = responseId ? this._asstByResp.get(responseId) : undefined;
    const userTurn = this._bindResponseUserTurn(terminalKey, responseEpoch);
    // A tool-only response has already committed the originating user/tool
    // transaction even though it deliberately produced no PCM. Keep its user
    // row as the visible origin for the later audible tool follow-up.
    if (localEpoch && audible === false && committed !== true) {
      if (responseId) this._discardUnheardResponse(responseId, entry);
      this._discardUnheardUserTurn(userTurn);
      this._renderEmptyHistoryIfNeeded();
      return;
    }

    // Nothing was rendered for an ID-less terminal. Its committed user origin
    // remains in history, but it must leave provisional bookkeeping so a later
    // response cannot claim it again.
    if (!responseId) {
      this._settleUserTurn(userTurn);
      return;
    }

    if (status === "cancelled" || status === "canceled") {
      // Audible cancellation is durable: mark its one assistant row interrupted.
      // If `*.transcript.done` never fired, build it from response.done instead.
      let hist = entry?.hist ?? null;
      if (!hist && transcript) {
        hist = this._appendHistMsg("assistant", transcript, false);
      } else if (hist && transcript) {
        this._updateHistMsg(hist, transcript, false);
      }
      if (hist) this._markHistInterrupted(hist);
      this._asstByResp.delete(responseId);
      this._settleUserTurn(userTurn);
      return;
    }

    // Any other terminal close (completed / failed / incomplete / …): just
    // release the map entry. The bubble already auto-dismisses on its timer and
    // the history row persists as the conversation log. Crucially we do NOT
    // touch user state here — that lifecycle is fully independent.
    this._asstByResp.delete(responseId);
    this._settleUserTurn(userTurn);
  }

  /** The model called a tool — reserve its durable history position immediately.
   *  @param {string} name @param {string} argsJson @param {string} callId */
  onToolCall(name, argsJson, callId) {
    this._bumpDismiss(this._spawnBubble("tool", name));
    if (callId && !this._toolHistByCall.has(callId)) {
      this._toolHistByCall.set(callId, this._appendHistTool(name, argsJson, "Running…"));
    }
    this._markUnread();
  }

  /** The tool finished — update its durable card (and any captured image).
   *  @param {string} name @param {string} argsJson @param {string} output @param {string} [image] @param {string} [callId] */
  onToolResult(name, argsJson, output, image, callId = "") {
    const existing = callId ? this._toolHistByCall.get(callId) : null;
    const outputEl = existing?.querySelector(".hist-tool-output");
    if (outputEl) {
      outputEl.textContent = output || "(no output)";
      this._toolHistByCall.delete(callId);
    } else {
      this._appendHistTool(name, argsJson, output);
    }
    if (image) this._appendHistImage(image); // show the captured frame below the call
    this._markUnread();
  }
}
