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
 *     { kind: "config", inputRate, generation }          next-response default
 *     { kind: "audio", samples, generation, primeMs, reprimeMs, maxPrimeMs,
 *       streamId, responseEpoch, sourceSampleRate }        every PCM chunk
 *     { kind: "end", generation, streamId, responseEpoch,
 *       sourceSampleRate }                                idempotent stream flush
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
 * to the bounded steady-state re-prime target, never the larger cold-start
 * reservoir. The worklet raises that target from observed chunk cadence.
 * `end` bypasses the target for a short final response and drains it exactly
 * once. FasterQwen3TTS, Groxaxo, and buffered fallback use a zero target and
 * therefore retain their immediate playback behavior.
 */

const STATS_INTERVAL_FRAMES = 12000;
const MAX_PRIME_MS = 2000;
const MAX_STREAM_TOMBSTONES = 256;

function _generation(value, fallback) {
  const number = Number(value);
  return Number.isSafeInteger(number) && number >= 0 ? number : fallback;
}

function _primeMs(value) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(0, Math.min(MAX_PRIME_MS, number)) : 0;
}

function _sourceSampleRate(value, fallback = 16000) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? number : fallback;
}

function _responseEpoch(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isSafeInteger(number) && number >= 0 ? number : null;
}

class AudioPlaybackProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    // The default/Faster/Groxaxo path is 16 kHz. The client replaces it with
    // the server-acknowledged candidate rate (24 kHz for audio.cpp) before
    // queued PCM is rendered; this only sets the source clock for interpolation.
    this._defaultInputRate = 16000;
    this._inputRate = this._defaultInputRate;
    this._stepRatio = this._inputRate / sampleRate;
    /** @type {{ samples: Float32Array, primeMs: number, reprimeMs: number, maxPrimeMs: number, continuityMode: "adaptive" | "fast-start", streamId: string, responseEpoch: number | null, sourceSampleRate: number }[]} */
    this._queue = [];
    this._readIdx = 0;
    this._fracPos = 0;
    /** @type {"idle" | "priming" | "playing"} */
    this._state = "idle";
    this._generation = 0;
    this._lastClearedGeneration = -1;
    this._primeTargetMs = 0;
    this._reprimeTargetMs = 0;
    this._maxPrimeMs = MAX_PRIME_MS;
    this._activeStreamId = "";
    this._activeResponseEpoch = null;
    /** @type {Map<string, number | null>} */
    this._endedStreams = new Map();
    /** @type {Set<string>} */
    this._drainedStreams = new Set();
    this._drainReported = true;
    this._waitingAfterUnderrun = false;
    this._startPending = false;
    this._startWasReprime = false;
    this._startingStreamId = "";
    this._framesSinceStats = 0;
    this._totalPlayed = 0;
    this._lastEnqueueFrame = null;
    /** @type {Map<string, number>} */
    this._lastEnqueueFrameByStream = new Map();
    /** @type {Map<string, number>} */
    this._observedChunkGapByStream = new Map();
    /** @type {Map<string, number>} */
    this._maxObservedChunkGapByStream = new Map();
    // Node contract tests do not expose AudioWorklet's global currentFrame.
    // Keep an equivalent local clock for that environment and as a harmless
    // fallback in unusual worklet hosts.
    this._clockFrames = 0;
    this._observedChunkGapMs = 0;
    this._maxObservedChunkGapMs = 0;
    this._underruns = 0;
    this._reprimes = 0;
    this._staleChunks = 0;
    this._clears = 0;

    this.port.onmessage = (event) => {
      const data = event.data;
      if (!data || typeof data !== "object") return;
      switch (data.kind) {
        case "config":
          this._defaultInputRate = _sourceSampleRate(data.inputRate, this._defaultInputRate);
          // A config acknowledgement chooses the default only for a response
          // created afterwards. Existing queued PCM retains its own source
          // clock, so a live 16 kHz <-> 24 kHz provider switch cannot retime it.
          if (this._queue.length === 0) this._setInputRate(this._defaultInputRate);
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

  _setInputRate(sourceSampleRate) {
    this._inputRate = _sourceSampleRate(sourceSampleRate, this._defaultInputRate);
    this._stepRatio = this._inputRate / sampleRate;
  }

  _queuedMs() {
    let seconds = 0;
    for (let index = 0; index < this._queue.length; index += 1) {
      const buffer = this._queue[index];
      const unread = buffer.samples.length - (index === 0 ? this._readIdx : 0);
      seconds += Math.max(0, unread) / buffer.sourceSampleRate;
    }
    return seconds * 1000;
  }

  _diagnostic(kind, detail = {}) {
    this.port.postMessage({
      kind,
      generation: this._generation,
      queuedMs: this._queuedMs(),
      primeTargetMs: this._primeTargetMs,
      reprimeTargetMs: this._reprimeTargetMs,
      maxPrimeMs: this._maxPrimeMs,
      observedChunkGapMs: this._observedChunkGapMs,
      maxObservedChunkGapMs: this._maxObservedChunkGapMs,
      state: this._state,
      underruns: this._underruns,
      reprimes: this._reprimes,
      staleChunks: this._staleChunks,
      clears: this._clears,
      responseEpoch: this._activeResponseEpoch,
      streamId: this._activeStreamId,
      sourceSampleRate: this._inputRate,
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
    const responseEpoch = _responseEpoch(data.responseEpoch);
    const chunkPrimeMs = _primeMs(data.primeMs);
    const maxPrimeMs = _primeMs(data.maxPrimeMs || MAX_PRIME_MS) || MAX_PRIME_MS;
    const requestedReprimeMs = _primeMs(data.reprimeMs);
    const continuityMode = data.continuityMode === "fast-start" ? "fast-start" : "adaptive";
    const sourceSampleRate = _sourceSampleRate(data.sourceSampleRate, this._defaultInputRate);
    if (streamId && (this._endedStreams.has(streamId) || this._drainedStreams.has(streamId))) {
      this._reject("audio", receivedGeneration, "audio_after_stream_end");
      return;
    }

    const enqueueFrame = typeof currentFrame === "number" ? currentFrame : this._clockFrames;
    const previousEnqueueFrame = this._lastEnqueueFrameByStream.get(streamId);
    if (previousEnqueueFrame !== undefined) {
      const gapMs = Math.max(0, ((enqueueFrame - previousEnqueueFrame) / sampleRate) * 1000);
      this._observedChunkGapByStream.set(streamId, gapMs);
      this._maxObservedChunkGapByStream.set(
        streamId,
        Math.max(this._maxObservedChunkGapByStream.get(streamId) || 0, gapMs),
      );
    }
    this._lastEnqueueFrameByStream.set(streamId, enqueueFrame);

    const adaptiveReprime = continuityMode === "adaptive"
      && (chunkPrimeMs > 0 || requestedReprimeMs > 0);
    const queued = {
      samples: data.samples,
      primeMs: chunkPrimeMs,
      reprimeMs: adaptiveReprime
        ? Math.min(maxPrimeMs, Math.max(80, requestedReprimeMs || Math.min(chunkPrimeMs, 240)))
        : requestedReprimeMs,
      maxPrimeMs,
      continuityMode,
      streamId,
      responseEpoch,
      sourceSampleRate,
    };
    const wasIdle = this._state === "idle";
    this._queue.push(queued);
    if (wasIdle) {
      this._drainReported = false;
      this._waitingAfterUnderrun = false;
      this._state = "priming";
      this._activateQueuedStream(queued);
    } else if (streamId === this._activeStreamId) {
      this._applyActiveStreamCadence(queued);
    }
    this._maybeStart(false);
  }

  _activateQueuedStream(buffer) {
    this._activeStreamId = buffer.streamId;
    this._activeResponseEpoch = buffer.responseEpoch;
    this._primeTargetMs = buffer.primeMs;
    this._maxPrimeMs = buffer.maxPrimeMs;
    this._reprimeTargetMs = buffer.reprimeMs;
    this._observedChunkGapMs = this._observedChunkGapByStream.get(buffer.streamId) || 0;
    this._maxObservedChunkGapMs = this._maxObservedChunkGapByStream.get(buffer.streamId) || 0;
    this._lastEnqueueFrame = this._lastEnqueueFrameByStream.get(buffer.streamId) ?? null;
    this._applyActiveStreamCadence(buffer);
    this._setInputRate(buffer.sourceSampleRate);
  }

  _applyActiveStreamCadence(buffer) {
    const gapMs = this._observedChunkGapByStream.get(buffer.streamId) || 0;
    const maxGapMs = this._maxObservedChunkGapByStream.get(buffer.streamId) || 0;
    this._observedChunkGapMs = gapMs;
    this._maxObservedChunkGapMs = maxGapMs;
    this._lastEnqueueFrame = this._lastEnqueueFrameByStream.get(buffer.streamId) ?? null;
    // Adaptive may learn a response-local recovery reservoir. Fast Start is
    // deliberately a fixed minimum-reservoir comparison mode: allowing it to
    // learn here would make its actual behaviour indistinguishable from
    // Adaptive after the first backend gap.
    this._reprimeTargetMs = buffer.continuityMode === "adaptive"
      ? Math.min(
        buffer.maxPrimeMs,
        // Learned recovery is monotonic for this response. A short chunk after
        // one long scheduler/backend gap must not lower the reservoir and cause
        // the same response to oscillate through repeated underruns.
        Math.max(buffer.reprimeMs, maxGapMs > 0 ? maxGapMs + 80 : 0),
      )
      : buffer.reprimeMs;
  }

  _end(data) {
    const receivedGeneration = _generation(data.generation, -1);
    if (receivedGeneration !== this._generation) {
      this._reject("end", receivedGeneration, "generation_mismatch");
      return;
    }
    const streamId = String(data.streamId || "");
    const responseEpoch = _responseEpoch(data.responseEpoch);
    if (streamId && (this._endedStreams.has(streamId) || this._drainedStreams.has(streamId))) return;
    if (streamId) this._endedStreams.set(streamId, responseEpoch);
    const hasQueuedStream = streamId
      ? this._queue.some((buffer) => buffer.streamId === streamId)
      : this._queue.length > 0;
    if (!hasQueuedStream) {
      // A stream may finish after its last buffer was rendered while another
      // response is already queued. Retire that response independently rather
      // than assigning the eventual shared-FIFO drain to the newer response.
      this._reportStreamDrained(streamId, responseEpoch, this._queue.length === 0);
      if (this._queue.length === 0 && (!this._activeStreamId || this._activeStreamId === streamId)) {
        this._markQueueIdle();
      }
      return;
    }
    // Synthesis is complete: a response shorter than the normal prime target
    // must still speak immediately, without consuming or trimming its head.
    if (!streamId || this._queue[0]?.streamId === streamId) this._maybeStart(true);
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
    this._activeStreamId = "";
    this._activeResponseEpoch = null;
    this._endedStreams.clear();
    this._drainedStreams.clear();
    this._drainReported = true;
    this._waitingAfterUnderrun = false;
    this._lastEnqueueFrame = null;
    this._lastEnqueueFrameByStream.clear();
    this._observedChunkGapByStream.clear();
    this._maxObservedChunkGapByStream.clear();
    this._observedChunkGapMs = 0;
    this._maxObservedChunkGapMs = 0;
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
    const reprime = this._waitingAfterUnderrun;
    const targetMs = reprime ? this._reprimeTargetMs : this._primeTargetMs;
    const queuedMs = this._queuedMs();
    if (!force && targetMs > 0 && queuedMs + 1e-6 < targetMs) return;
    if (reprime) this._reprimes += 1;
    this._state = "playing";
    this._waitingAfterUnderrun = false;
    this._startPending = true;
    this._startWasReprime = reprime;
    this._startingStreamId = this._queue[0]?.streamId || this._activeStreamId;
    this._diagnostic(reprime ? "reprimed" : "primed", { forced: !!force, targetMs });
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
    const completedRate = this._queue[0]?.sourceSampleRate ?? this._inputRate;
    this._fracPos += completedRate / sampleRate;
    while (this._fracPos >= 1) {
      this._fracPos -= 1;
      this._readIdx += 1;
    }
    while (this._queue.length > 0 && this._readIdx >= this._queue[0].samples.length) {
      const completedStream = this._queue[0].streamId;
      const completedEpoch = this._queue[0].responseEpoch;
      const completedBufferRate = this._queue[0].sourceSampleRate;
      this._readIdx -= this._queue[0].samples.length;
      this._queue.shift();
      if (this._queue.length > 0 && this._queue[0].sourceSampleRate !== completedBufferRate) {
        // `fracPos` is measured in source samples. Preserve the tiny physical
        // residual across a rate boundary rather than applying the old rate to
        // the new response's PCM.
        this._fracPos = (this._fracPos * this._queue[0].sourceSampleRate) / completedBufferRate;
        this._setInputRate(this._queue[0].sourceSampleRate);
        while (this._fracPos >= 1 && this._readIdx < this._queue[0].samples.length) {
          this._fracPos -= 1;
          this._readIdx += 1;
        }
      }
      if (this._queue.length > 0 && this._queue[0].streamId !== completedStream) {
        this._reportStreamDrained(completedStream, completedEpoch, false);
        // The new response already carries the config snapshot that was
        // acknowledged when it was created. It becomes the re-prime policy only
        // after playback crosses its FIFO boundary; older queued PCM remains on
        // the target it started with.
        this._activateQueuedStream(this._queue[0]);
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
    if (this._activeStreamId && this._endedStreams.has(this._activeStreamId)) {
      this._reportStreamDrained(
        this._activeStreamId,
        this._endedStreams.get(this._activeStreamId) ?? this._activeResponseEpoch,
        true,
      );
      this._markQueueIdle();
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

  _markQueueIdle() {
    this._state = "idle";
    this._readIdx = 0;
    this._fracPos = 0;
    this._waitingAfterUnderrun = false;
    this._startPending = false;
    this._startingStreamId = "";
    this._drainReported = true;
    this._activeStreamId = "";
    this._activeResponseEpoch = null;
  }

  _reportStreamDrained(streamId, responseEpoch, queueEmpty) {
    if (!streamId || this._drainedStreams.has(streamId)) return;
    this._drainedStreams.add(streamId);
    while (this._drainedStreams.size > MAX_STREAM_TOMBSTONES) {
      const oldest = this._drainedStreams.values().next().value;
      if (!oldest) break;
      this._drainedStreams.delete(oldest);
    }
    this._endedStreams.delete(streamId);
    this._lastEnqueueFrameByStream.delete(streamId);
    this._observedChunkGapByStream.delete(streamId);
    this._maxObservedChunkGapByStream.delete(streamId);
    this._diagnostic("drained", {
      streamId,
      responseEpoch,
      queueEmpty: queueEmpty === true,
    });
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
              responseEpoch: this._activeResponseEpoch,
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
    this._clockFrames += output.length;
    if (this._framesSinceStats >= STATS_INTERVAL_FRAMES) {
      this._framesSinceStats %= STATS_INTERVAL_FRAMES;
      this._diagnostic("stats", { played: this._totalPlayed });
    }
    return true;
  }
}

registerProcessor("audio-playback", AudioPlaybackProcessor);
