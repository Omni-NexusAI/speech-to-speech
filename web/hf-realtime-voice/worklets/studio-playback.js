// @ts-check
/**
 * Turn-scoped PCM playback queue for the copied Gradio Voice Studio.
 *
 * Main thread -> worklet:
 *   { kind: "begin", turnId, inputRate, startupMs, reprimeMs }
 *   { kind: "audio", turnId, samples: Float32Array }
 *   { kind: "end", turnId }
 *   { kind: "clear", turnId }
 *
 * Worklet -> main thread:
 *   { kind: "queued", turnId, queuedMs, blocks }
 *   { kind: "started", turnId, resumed }
 *   { kind: "underrun", turnId }
 *   { kind: "drained", turnId }
 *   { kind: "clear", turnId }
 *
 * Native playback waits until the configured first-plus-steady duration is
 * queued. A real underrun uses the smaller steady-state re-prime target. A
 * completed short utterance starts immediately after `end`. The
 * queue remains alive across every phrase in one assistant turn; `clear`
 * invalidates all queued audio so cancelled turns cannot leak a stale tail.
 */

const STATS_INTERVAL_FRAMES = 12000;
const FADE_FRAMES = 32;

class StudioPlaybackProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._turnId = 0;
    this._inputRate = 24000;
    this._stepRatio = this._inputRate / sampleRate;
    this._startupMs = 0;
    this._reprimeMs = 0;
    this._queue = [];
    this._readIdx = 0;
    this._fracPos = 0;
    this._playing = false;
    this._startedOnce = false;
    this._ended = false;
    this._drained = false;
    this._underrunReported = false;
    this._framesSinceStats = 0;
    this._totalPlayed = 0;
    this._fadeIn = 0;
    this._fadeOut = 0;
    this._lastSample = 0;
    this._startPending = null;

    this.port.onmessage = (event) => {
      const data = event.data;
      if (!data || typeof data !== "object") return;
      if (data.kind === "begin") {
        this._reset(Number(data.turnId) || 0);
        if (Number(data.inputRate) > 0) this._inputRate = Number(data.inputRate);
        this._stepRatio = this._inputRate / sampleRate;
        this._startupMs = Math.max(0, Number(data.startupMs) || 0);
        this._reprimeMs = Math.max(0, Math.min(2000, Number(data.reprimeMs) || Math.min(this._startupMs, 240)));
        return;
      }
      if (data.kind === "clear") {
        this._reset(Number(data.turnId) || this._turnId);
        this.port.postMessage({ kind: "clear", turnId: this._turnId });
        return;
      }
      if (Number(data.turnId) !== this._turnId) return;
      if (data.kind === "audio") {
        if (!(data.samples instanceof Float32Array) || data.samples.length === 0 || this._ended) return;
        this._queue.push(data.samples);
        this._drained = false;
        const queuedMs = this._queuedMs();
        this.port.postMessage({
          kind: "queued",
          turnId: this._turnId,
          queuedMs,
          blocks: this._queue.length,
        });
        this._armPlayback(false);
        return;
      }
      if (data.kind === "end") {
        this._ended = true;
        this._armPlayback(true);
        if (!this._playing && this._queue.length === 0) this._emitDrained();
      }
    };
  }

  _reset(turnId) {
    this._turnId = turnId;
    this._queue.length = 0;
    this._readIdx = 0;
    this._fracPos = 0;
    this._playing = false;
    this._startedOnce = false;
    this._ended = false;
    this._drained = false;
    this._underrunReported = false;
    this._fadeIn = 0;
    this._fadeOut = 0;
    this._lastSample = 0;
    this._startPending = null;
  }

  _queuedSamples() {
    let total = -this._readIdx;
    for (const block of this._queue) total += block.length;
    return Math.max(0, total);
  }

  _queuedMs() {
    return (this._queuedSamples() / this._inputRate) * 1000;
  }

  _armPlayback(completed) {
    if (this._playing || this._queue.length === 0) return;
    var targetMs = this._startedOnce ? this._reprimeMs : this._startupMs;
    if (!completed && this._queuedMs() + 0.001 < targetMs) return;
    const resumed = this._startedOnce;
    this._playing = true;
    this._startedOnce = true;
    this._underrunReported = false;
    this._fadeIn = FADE_FRAMES;
    this._fadeOut = 0;
    this._startPending = { resumed };
  }

  _emitDrained() {
    if (this._drained) return;
    this._drained = true;
    this._playing = false;
    this._lastSample = 0;
    this.port.postMessage({ kind: "drained", turnId: this._turnId });
  }

  _readInterpolated() {
    if (this._queue.length === 0) return null;
    const head = this._queue[0];
    const idx = this._readIdx;
    const frac = this._fracPos;
    const a = head[idx];
    let b = a;
    if (idx + 1 < head.length) b = head[idx + 1];
    else if (this._queue.length > 1) b = this._queue[1][0];
    return a + (b - a) * frac;
  }

  _advance() {
    this._fracPos += this._stepRatio;
    while (this._fracPos >= 1) {
      this._fracPos -= 1;
      this._readIdx += 1;
    }
    while (this._queue.length > 0 && this._readIdx >= this._queue[0].length) {
      this._readIdx -= this._queue[0].length;
      this._queue.shift();
    }
  }

  process(_inputs, outputs) {
    const channels = outputs[0];
    if (!channels || channels.length === 0) return true;
    const out = channels[0];
    const stereo = channels.length > 1 ? channels[1] : null;

    for (let index = 0; index < out.length; index += 1) {
      let sample = 0;
      if (this._playing) {
        const value = this._readInterpolated();
        if (value === null) {
          if (this._ended) {
            this._emitDrained();
          } else {
            if (!this._underrunReported) {
              this._underrunReported = true;
              this.port.postMessage({ kind: "underrun", turnId: this._turnId });
              this._fadeOut = FADE_FRAMES;
            }
            if (this._fadeOut > 0) {
              sample = this._lastSample * (this._fadeOut / FADE_FRAMES);
              this._fadeOut -= 1;
            }
            if (this._fadeOut === 0) {
              this._playing = false;
              this._lastSample = 0;
            }
          }
        } else {
          sample = value;
          if (this._startPending) {
            this.port.postMessage({ kind: "started", turnId: this._turnId, resumed: this._startPending.resumed });
            this._startPending = null;
          }
          this._lastSample = value;
          this._advance();
        }

        if (this._fadeIn > 0) {
          sample *= 1 - this._fadeIn / FADE_FRAMES;
          this._fadeIn -= 1;
        }
        this._totalPlayed += 1;
      }
      out[index] = sample;
      if (stereo) stereo[index] = sample;
    }

    this._framesSinceStats += out.length;
    if (this._framesSinceStats >= STATS_INTERVAL_FRAMES) {
      this._framesSinceStats = 0;
      this.port.postMessage({
        kind: "queued",
        turnId: this._turnId,
        queuedMs: this._queuedMs(),
        blocks: this._queue.length,
        played: this._totalPlayed,
      });
    }
    return true;
  }
}

registerProcessor("studio-playback", StudioPlaybackProcessor);
