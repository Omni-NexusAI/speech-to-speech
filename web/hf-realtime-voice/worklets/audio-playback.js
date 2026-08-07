// @ts-check
/**
 * Continuous mono PCM playback queue.
 *
 * The main thread assigns every response chunk a playback generation and an
 * acknowledged priming target. A clear advances the generation; audio from an
 * older generation is rejected in the worklet, so a cancelled response cannot
 * leak a stale tail after barge-in, Stop, replacement, or disconnect.
 *
 * Lifecycle / messaging:
 *
 *   main -> worklet:
 *     { kind: "config", inputRate, generation }          startup
 *     { kind: "audio", samples, generation, primeMs,
 *       streamId }                                      every PCM chunk
 *     { kind: "end", generation, streamId }             idempotent stream flush
 *     { kind: "clear", generation, reason }             cancellation boundary
 *
 *   worklet -> main:
 *     { kind: "primed" | "reprimed", ... }
 *     { kind: "started", ... }
 *     { kind: "stats", queuedMs, ... }
 *     { kind: "underrun" | "drained" | "cleared", ... }
 *     { kind: "stale_chunk_rejected", ... }
 *
 * Priming never consumes queued samples. A genuine mid-stream underrun returns
 * to the same priming target instead of restarting from the next tiny block.
 * `end` bypasses the target for a short final response and drains it exactly
 * once. FasterQwen3TTS, Groxaxo, and buffered fallback use a zero target and
 * therefore retain their immediate playback behavior.
 */

const STATS_INTERVAL_FRAMES = 12000;
const MAX_PRIME_MS = 2000;

function _generation(value, fallback) {
  const number = Number(value);
  return Number.isSafeInteger(number) && number >= 0 ? number : fallback;
}

function _primeMs(value) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(0, Math.min(MAX_PRIME_MS, number)) : 0;
}

class AudioPlaybackProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    // HF Realtime emits the pipeline's native 16 kHz PCM. The client still
    // sends an explicit config message at startup, but the safe default must
    // match the actual WebSocket transport if that message is delayed.
    this._inputRate = 16000;
    this._stepRatio = this._inputRate / sampleRate;
    /** @type {{ samples: Float32Array, primeMs: number, streamId: string }[]} */
    this._queue = [];
    this._readIdx = 0;
    this._fracPos = 0;
    /** @type {"idle" | "priming" | "playing"} */
    this._state = "idle";
    this._generation = 0;
    this._lastClearedGeneration = -1;
    this._primeTargetMs = 0;
    this._ended = false;
    this._activeStreamId = "";
    this._endedStreamId = "";
    this._drainReported = true;
    this._waitingAfterUnderrun = false;
    this._startPending = false;
    this._startWasReprime = false;
    this._startingStreamId = "";
    this._framesSinceStats = 0;
    this._totalPlayed = 0;
    this._underruns = 0;
    this._reprimes = 0;
    this._staleChunks = 0;
    this._clears = 0;

    this.port.onmessage = (event) => {
      const data = event.data;
      if (!data || typeof data !== "object") return;
      switch (data.kind) {
        case "config":
          if (typeof data.inputRate === "number" && data.inputRate > 0) {
            this._inputRate = data.inputRate;
            this._stepRatio = this._inputRate / sampleRate;
          }
          this._generation = _generation(data.generation, this._generation);
          break;
        case "audio":
          this._enqueue(data);
          break;
        case "end":
          this._end(data);
          break;
        case "clear":
          this._clear(data);
          break;
      }
    };
  }

  _queuedSamples() {
    let total = -this._readIdx;
    for (const buffer of this._queue) total += buffer.samples.length;
    return Math.max(0, total);
  }

  _queuedMs() {
    return (this._queuedSamples() / this._inputRate) * 1000;
  }

  _diagnostic(kind, detail = {}) {
    this.port.postMessage({
      kind,
      generation: this._generation,
      queuedMs: this._queuedMs(),
      primeTargetMs: this._primeTargetMs,
      state: this._state,
      underruns: this._underruns,
      reprimes: this._reprimes,
      staleChunks: this._staleChunks,
      clears: this._clears,
      ...detail,
    });
  }

  _reject(kind, receivedGeneration, reason) {
    this._staleChunks += 1;
    this._diagnostic("stale_chunk_rejected", {
      rejectedKind: kind,
      receivedGeneration,
      reason,
    });
  }

  _enqueue(data) {
    const receivedGeneration = _generation(data.generation, -1);
    if (receivedGeneration !== this._generation) {
      this._reject("audio", receivedGeneration, "generation_mismatch");
      return;
    }
    if (!(data.samples instanceof Float32Array) || data.samples.length === 0) return;
    const streamId = String(data.streamId || "");
    const chunkPrimeMs = _primeMs(data.primeMs);
    if (streamId && streamId === this._endedStreamId) {
      this._reject("audio", receivedGeneration, "audio_after_stream_end");
      return;
    }

    // A response and its tool-result continuation share a generation but have
    // distinct stream IDs. If the continuation arrives before the prior PCM has
    // drained, reopen the same FIFO and append it without clearing or re-priming.
    if (this._state === "idle") {
      this._primeTargetMs = chunkPrimeMs;
      this._ended = false;
      this._drainReported = false;
      this._waitingAfterUnderrun = false;
      this._state = "priming";
      this._activeStreamId = streamId;
    } else if (streamId && streamId !== this._activeStreamId) {
      this._activeStreamId = streamId;
      this._ended = false;
      if (this._state === "priming" && this._queuedSamples() === 0) {
        this._primeTargetMs = chunkPrimeMs;
      }
    } else if (this._ended) {
      this._reject("audio", receivedGeneration, "audio_after_stream_end");
      return;
    }

    this._queue.push({ samples: data.samples, primeMs: chunkPrimeMs, streamId });
    this._maybeStart(false);
  }

  _end(data) {
    const receivedGeneration = _generation(data.generation, -1);
    if (receivedGeneration !== this._generation) {
      this._reject("end", receivedGeneration, "generation_mismatch");
      return;
    }
    const streamId = String(data.streamId || "");
    if (streamId && streamId === this._endedStreamId) return;
    if (streamId && this._activeStreamId && streamId !== this._activeStreamId) {
      this._reject("end", receivedGeneration, "inactive_stream");
      return;
    }
    if (this._ended) return;
    this._ended = true;
    this._endedStreamId = streamId;
    if (this._queuedSamples() === 0) {
      this._markDrained();
      return;
    }
    // Synthesis is complete: a response shorter than the normal prime target
    // must still speak immediately, without consuming or trimming its head.
    this._maybeStart(true);
  }

  _clear(data) {
    const receivedGeneration = _generation(data.generation, -1);
    if (receivedGeneration < this._generation) {
      this._reject("clear", receivedGeneration, "generation_mismatch");
      return;
    }
    if (receivedGeneration === this._lastClearedGeneration) return;
    if (receivedGeneration < 0) return;

    const reason = String(data.reason || "clear");
    const drainedStreamId = this._queue[0]?.streamId || this._activeStreamId;
    const hadUndrainedPlayback = !this._drainReported
      || this._state !== "idle"
      || this._queuedSamples() > 0;
    this._generation = receivedGeneration;
    this._lastClearedGeneration = receivedGeneration;
    this._queue.length = 0;
    this._readIdx = 0;
    this._fracPos = 0;
    this._state = "idle";
    this._ended = false;
    this._activeStreamId = "";
    this._endedStreamId = "";
    this._drainReported = true;
    this._waitingAfterUnderrun = false;
    this._startPending = false;
    this._startWasReprime = false;
    this._startingStreamId = "";
    this._clears += 1;
    this._diagnostic("cleared", { reason });
    if (hadUndrainedPlayback) {
      // A generation clear is also the terminal playback event for the audible
      // stream it interrupted. The main thread owns speaking state from the
      // worklet's started/drained pair, independent of network response state.
      this._diagnostic("drained", {
        reason,
        cleared: true,
        streamId: drainedStreamId,
      });
    }
  }

  _maybeStart(force) {
    if (this._state !== "priming" || this._queuedSamples() === 0) return;
    const queuedMs = this._queuedMs();
    if (!force && this._primeTargetMs > 0 && queuedMs + 1e-6 < this._primeTargetMs) return;

    const reprime = this._waitingAfterUnderrun;
    if (reprime) this._reprimes += 1;
    this._state = "playing";
    this._waitingAfterUnderrun = false;
    this._startPending = true;
    this._startWasReprime = reprime;
    this._startingStreamId = this._queue[0]?.streamId || this._activeStreamId;
    this._diagnostic(reprime ? "reprimed" : "primed", { forced: !!force });
  }

  /** Linear-interpolated read at the current fractional position. */
  _readInterpolated() {
    if (this._queue.length === 0) return null;
    const head = this._queue[0].samples;
    const index = this._readIdx;
    const fraction = this._fracPos;
    const first = head[index];
    let second;
    if (index + 1 < head.length) {
      second = head[index + 1];
    } else if (this._queue.length > 1) {
      second = this._queue[1].samples[0];
    } else {
      second = first;
    }
    return first + (second - first) * fraction;
  }

  /** Advance by the resampling ratio and pop only fully consumed buffers. */
  _advance() {
    this._fracPos += this._stepRatio;
    while (this._fracPos >= 1) {
      this._fracPos -= 1;
      this._readIdx += 1;
    }
    while (this._queue.length > 0 && this._readIdx >= this._queue[0].samples.length) {
      const completedStream = this._queue[0].streamId;
      this._readIdx -= this._queue[0].samples.length;
      this._queue.shift();
      if (this._queue.length > 0 && this._queue[0].streamId !== completedStream) {
        // The new response already carries the config snapshot that was
        // acknowledged when it was created. It becomes the re-prime policy only
        // after playback crosses its FIFO boundary; older queued PCM remains on
        // the target it started with.
        this._primeTargetMs = this._queue[0].primeMs;
        // The FIFO remains continuous across tool/result responses, but report
        // the exact point at which the next response's first sample is rendered.
        // This is bookkeeping only: it does not pause or re-prime playback.
        this._startPending = true;
        this._startWasReprime = false;
        this._startingStreamId = this._queue[0].streamId;
      }
    }
  }

  _handleEmptyQueue() {
    if (this._ended) {
      this._markDrained();
      return;
    }
    if (this._state !== "playing") return;
    this._state = "priming";
    this._waitingAfterUnderrun = true;
    this._startPending = false;
    this._startingStreamId = "";
    this._underruns += 1;
    this._diagnostic("underrun");
  }

  _markDrained() {
    const streamId = this._activeStreamId || this._endedStreamId;
    this._state = "idle";
    this._readIdx = 0;
    this._fracPos = 0;
    this._waitingAfterUnderrun = false;
    this._startPending = false;
    this._startingStreamId = "";
    if (this._drainReported) return;
    this._drainReported = true;
    this._diagnostic("drained", { streamId });
  }

  process(_, outputs) {
    const channels = outputs[0];
    if (!channels || channels.length === 0) return true;
    const output = channels[0];
    const stereo = channels.length > 1 ? channels[1] : null;

    this._maybeStart(false);
    for (let index = 0; index < output.length; index += 1) {
      let sample = 0;
      if (this._state === "playing") {
        const value = this._readInterpolated();
        if (value === null) {
          this._handleEmptyQueue();
        } else {
          sample = value;
          if (this._startPending) {
            this._startPending = false;
            this._diagnostic("started", {
              reprime: this._startWasReprime,
              streamId: this._startingStreamId || this._queue[0]?.streamId || this._activeStreamId,
            });
            this._startingStreamId = "";
          }
          this._advance();
          this._totalPlayed += 1;
        }
      }
      output[index] = sample;
      if (stereo) stereo[index] = sample;
    }

    this._framesSinceStats += output.length;
    if (this._framesSinceStats >= STATS_INTERVAL_FRAMES) {
      this._framesSinceStats %= STATS_INTERVAL_FRAMES;
      this._diagnostic("stats", { played: this._totalPlayed });
    }
    return true;
  }
}

registerProcessor("audio-playback", AudioPlaybackProcessor);
