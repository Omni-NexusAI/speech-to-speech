// @ts-check

/**
 * Owns the monotonically increasing UI-start lease. A new start or teardown
 * invalidates every older async preflight before it can acquire a microphone or
 * attach a WebSocket client.
 */
export class StartAttemptController {
  constructor() {
    this._current = 0;
  }

  claim() {
    this._current += 1;
    return this._current;
  }

  invalidate() {
    this._current += 1;
  }

  /** @param {number} token */
  isCurrent(token) {
    return token === this._current;
  }
}
