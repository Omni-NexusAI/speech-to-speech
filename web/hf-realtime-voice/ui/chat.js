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
 *   - assistant transcripts by `response_id`, so a cancelled speculative reply
 *     can be marked interrupted without erasing what was already shown.
 */

import { $, escHtml, DEBUG } from "./dom.js";

const WRENCH_PATH = `<path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/>`;
const CHAT_BUBBLE_SVG = `<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>`;
const EMPTY_STATE_HTML = `<div id="chat-empty" class="chat-empty">${CHAT_BUBBLE_SVG}<span class="chat-empty-title">No messages yet</span><span class="chat-empty-hint">Tap the orb and start talking</span></div>`;

const MAX_CORRELATION_ID_LENGTH = 128;
const CORRELATION_ID_RE = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/;

/**
 * Keep protocol correlation visible without turning a server-controlled field
 * into an unbounded or content-bearing diagnostics channel.
 * @param {unknown} value
 */
export function boundedCorrelationId(value) {
  if (typeof value !== "string" || value.length > MAX_CORRELATION_ID_LENGTH) return "";
  return CORRELATION_ID_RE.test(value) ? value : "";
}

/**
 * @typedef {Object} CameraLifecycle
 * @property {number} captureGeneration
 * @property {number} requestedAtMs
 * @property {string} cardId
 * @property {string} callId
 * @property {string} responseId
 * @property {string} itemId
 * @property {string} acceptedTurnId
 * @property {string} captureStatus
 * @property {string} outputStatus
 */

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
   * Append a durable tool-call row immediately; its output and optional camera
   * frame are filled into the same card when execution finishes.
   * @param {string} name @param {unknown} argsJson @param {string} output
   * @param {CameraLifecycle | undefined} [lifecycle]
   */
  _appendHistTool(name, argsJson, output, lifecycle) {
    const empty = this._chatHistory.querySelector(".chat-empty");
    if (empty) empty.remove();
    let pretty = typeof argsJson === "string" ? argsJson : "(invalid non-string arguments)";
    try { pretty = JSON.stringify(JSON.parse(pretty), null, 2); } catch {}
    const el = document.createElement("div");
    el.className = "hist-msg tool";
    el.innerHTML = `
      <div class="hist-role">Tool call</div>
      <button class="hist-tool-header" aria-expanded="false">
        <svg class="hist-tool-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${WRENCH_PATH}</svg>
        <span class="hist-tool-name">${escHtml(name)}</span>
        <svg class="hist-tool-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>
      </button>
      <div class="hist-tool-meta" hidden></div>
      <div class="hist-tool-body">
        <div class="hist-tool-label">Input</div>
        <div class="hist-tool-block">${escHtml(pretty)}</div>
        <div class="hist-tool-label">Output</div>
        <div class="hist-tool-block hist-tool-output">${escHtml(output || "(no output)")}</div>
      </div>
    `;
    const header = /** @type {HTMLButtonElement} */ (el.querySelector(".hist-tool-header"));
    const body = /** @type {HTMLDivElement} */ (el.querySelector(".hist-tool-body"));
    this._updateToolLifecycle(el, lifecycle);
    header.addEventListener("click", () => {
      const expanded = header.getAttribute("aria-expanded") === "true";
      header.setAttribute("aria-expanded", String(!expanded));
      body.classList.toggle("open", !expanded);
    });
    this._chatHistory.appendChild(el);
    this._scrollToBottom();
    return el;
  }

  /**
   * A camera generation is part of the history identity so every actual call
   * receives its own durable card, even if the backend omits or repeats a call ID.
   * @param {string} callId
   * @param {Partial<CameraLifecycle> | undefined} lifecycle
   */
  _toolHistoryKey(callId, lifecycle) {
    if (Number.isFinite(lifecycle?.captureGeneration)) {
      const boundedCallId = boundedCorrelationId(lifecycle?.callId ?? callId);
      return `camera:${lifecycle.captureGeneration}:${boundedCallId || "missing-call-id"}`;
    }
    return callId;
  }

  /**
   * Render content-free camera lifecycle information. Never include argument,
   * transcript, image, or tool-result content in this metadata line.
   * @param {HTMLElement} el
   * @param {CameraLifecycle | undefined} lifecycle
   */
  _updateToolLifecycle(el, lifecycle) {
    if (!lifecycle) return;
    const cardId = boundedCorrelationId(lifecycle.cardId);
    const callId = boundedCorrelationId(lifecycle.callId);
    const responseId = boundedCorrelationId(lifecycle.responseId);
    const itemId = boundedCorrelationId(lifecycle.itemId);
    const acceptedTurnId = boundedCorrelationId(lifecycle.acceptedTurnId);
    el.dataset.captureGeneration = String(lifecycle.captureGeneration);
    el.dataset.cameraCardId = cardId;
    el.dataset.callId = callId;
    el.dataset.responseId = responseId;
    el.dataset.itemId = itemId;
    el.dataset.acceptedTurnId = acceptedTurnId;
    el.dataset.captureStatus = lifecycle.captureStatus;
    el.dataset.outputStatus = lifecycle.outputStatus;
    const meta = /** @type {HTMLElement | null} */ (el.querySelector(".hist-tool-meta"));
    if (!meta) return;
    const requested = Number.isFinite(lifecycle.requestedAtMs)
      ? new Date(lifecycle.requestedAtMs).toLocaleTimeString()
      : "time unavailable";
    const call = callId || "missing call ID";
    const response = responseId || "missing response ID";
    const item = itemId || "missing item ID";
    const turn = acceptedTurnId || "missing accepted turn ID";
    meta.hidden = false;
    meta.textContent = `Capture #${lifecycle.captureGeneration} | ${requested} | ${call} | response ${response} | item ${item} | turn ${turn} | ${lifecycle.captureStatus} | output ${lifecycle.outputStatus}`;
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

  /** Render a captured webcam frame inside its exact tool card.
   *  @param {string} dataUrl @param {HTMLElement} toolCard */
  _appendHistImage(dataUrl, toolCard) {
    const body = toolCard.querySelector(".hist-tool-body") || toolCard;
    const frame = document.createElement("div");
    frame.className = "hist-tool-snapshot";
    frame.innerHTML = `<div class="hist-tool-label">Captured frame</div><img class="hist-image" alt="Webcam snapshot sent to the model" />`;
    const img = /** @type {HTMLImageElement} */ (frame.querySelector("img"));
    img.src = dataUrl;
    body.appendChild(frame);
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
    this._asstByResp.clear();
    this._toolHistByCall.clear();
  }

  // ── Client event handlers ─────────────────────────────────────────────────

  /**
   * A streamed transcript delta (user or assistant).
   * @param {{ role: "user" | "assistant"; text: string; partial: boolean; itemId?: string; responseId?: string }} d
   */
  onTranscript(d, options = {}) {
    if (DEBUG) console.debug(`[ui] transcript role=${d.role} partial=${d.partial} item=${d.itemId} resp=${d.responseId} chars=${String(d.text || "").length}`);

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
      }
      this._markUnread();
    } else if (d.role === "assistant") {
      // Assistant transcript arrives once, as the full text, keyed by
      // response_id so a cancelled speculative response can be removed later. A
      // missing id gets a unique key so two id-less replies never collide.
      const rid = d.responseId || `_a${++this._anonSeq}`;
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
    this._markUnread();
  }

  /**
   * A response closed (completed or cancelled).
   * @param {{ responseId: string; status: string; audible?: boolean; transcript?: string }} detail
   */
  onResponseFinished(detail) {
    const { responseId, status, audible, transcript } = detail;
    if (DEBUG) console.debug(`[ui] response-finished resp=${responseId} status=${status} audible=${audible} known=${this._asstByResp.has(responseId)}`);
    // Without an id we can't target a specific response; the bubble will
    // auto-dismiss on its own timer regardless.
    if (!responseId) return;
    const entry = this._asstByResp.get(responseId);

    if (status === "cancelled") {
      // Keep every transcript that was received — mark it interrupted rather
      // than erasing it. If the `*.transcript.done` never fired, build the row
      // from the text carried in response.done.
      let hist = entry?.hist ?? null;
      if (!hist && transcript) {
        hist = this._appendHistMsg("assistant", transcript, false);
      } else if (hist && transcript) {
        this._updateHistMsg(hist, transcript, false);
      }
      if (hist) this._markHistInterrupted(hist);
      this._asstByResp.delete(responseId);
      return;
    }

    // Any other terminal close (completed / failed / incomplete / …): just
    // release the map entry. The bubble already auto-dismisses on its timer and
    // the history row persists as the conversation log. Crucially we do NOT
    // touch user state here — that lifecycle is fully independent.
    this._asstByResp.delete(responseId);
  }

  /** The model called a tool — reserve its durable history position immediately.
   *  @param {string} name @param {unknown} argsJson @param {string} callId
   *  @param {CameraLifecycle | undefined} [lifecycle] */
  onToolCall(name, argsJson, callId, lifecycle) {
    this._bumpDismiss(this._spawnBubble("tool", name));
    const historyKey = this._toolHistoryKey(callId, lifecycle);
    if (lifecycle && historyKey && !this._toolHistByCall.has(historyKey)) {
      this._toolHistByCall.set(historyKey, this._appendHistTool(name, argsJson, "Running...", lifecycle));
    }
    if (!lifecycle && callId && !this._toolHistByCall.has(callId)) {
      this._toolHistByCall.set(callId, this._appendHistTool(name, argsJson, "Running…"));
    }
    this._markUnread();
  }

  /** Render a non-executable, content-free tool protocol failure. Missing call
   * IDs cannot be paired with function output and must never reach the tool
   * executor or trigger response.create.
   * @param {string} code */
  onToolProtocolFailure(code) {
    const safeCode = code === "missing_call_id" ? "missing_call_id" : "missing_tool_name";
    const output = JSON.stringify({
      type: "invalid_tool_call",
      code: safeCode,
      message: "The tool call was rejected because its protocol identity was incomplete.",
    });
    this._bumpDismiss(this._spawnBubble("tool", "tool_protocol"));
    this._appendHistTool("tool_protocol", "{}", output);
    this._markUnread();
  }

  /** The tool finished — update its durable card (and any captured image).
   *  @param {string} name @param {unknown} argsJson @param {string} output @param {string} [image] @param {string} [callId]
   *  @param {CameraLifecycle | undefined} [lifecycle] */
  onToolResult(name, argsJson, output, image, callId = "", lifecycle) {
    const historyKey = this._toolHistoryKey(callId, lifecycle);
    let existing = historyKey ? this._toolHistByCall.get(historyKey) : null;
    const outputEl = existing?.querySelector(".hist-tool-output");
    if (outputEl) {
      outputEl.textContent = output || "(no output)";
      this._updateToolLifecycle(existing, lifecycle);
      this._toolHistByCall.delete(historyKey);
    } else {
      existing = this._appendHistTool(name, argsJson, output, lifecycle);
    }
    if (image && existing) this._appendHistImage(image, existing);
    this._markUnread();
  }
}
