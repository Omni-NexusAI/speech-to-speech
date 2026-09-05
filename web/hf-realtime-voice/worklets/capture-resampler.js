// @ts-check

const DEFAULT_OUTPUT_RATE = 16000;
const HALF_LENGTH = 60;
const TAP_COUNT = (HALF_LENGTH * 2) + 1;
const CUTOFF_HZ = 7200;
const KAISER_BETA = 6.25;
const MAX_BANK_PHASES = 1024;
const RING_CAPACITY = 512;

/** @type {Map<string, Float32Array[]>} */
const FILTER_CACHE = new Map();

function gcd(left, right) {
  let a = Math.abs(Math.round(left));
  let b = Math.abs(Math.round(right));
  while (b) [a, b] = [b, a % b];
  return a || 1;
}
function sinc(value) {
  if (Math.abs(value) < 1e-12) return 1;
  const angle = Math.PI * value;
  return Math.sin(angle) / angle;
}

function besselI0(value) {
  const scaled = (value * value) / 4;
  let total = 1;
  let term = 1;
  for (let order = 1; order < 32; order += 1) {
    term *= scaled / (order * order);
    total += term;
    if (term <= total * 1e-15) break;
  }
  return total;
}

function filterBank(inputRate, phaseCount) {
  const key = `${inputRate}:${phaseCount}`;
  const cached = FILTER_CACHE.get(key);
  if (cached) return cached;
  const cutoff = CUTOFF_HZ / inputRate;
  const denominator = besselI0(KAISER_BETA);
  const phases = [];
  for (let phaseIndex = 0; phaseIndex < phaseCount; phaseIndex += 1) {
    const fraction = phaseIndex / phaseCount;
    const coefficients = new Float32Array(TAP_COUNT);
    let total = 0;
    for (let tapIndex = 0; tapIndex < TAP_COUNT; tapIndex += 1) {
      const tap = tapIndex - HALF_LENGTH;
      const normalized = tap / HALF_LENGTH;
      const window = besselI0(
        KAISER_BETA * Math.sqrt(Math.max(0, 1 - (normalized * normalized))),
      ) / denominator;
      const distance = tap - fraction;
      const coefficient = 2 * cutoff * sinc(2 * cutoff * distance) * window;
      coefficients[tapIndex] = coefficient;
      total += coefficient;
    }
    for (let tapIndex = 0; tapIndex < TAP_COUNT; tapIndex += 1) {
      coefficients[tapIndex] /= total;
    }
    let floatTotal = 0;
    for (const coefficient of coefficients) floatTotal += coefficient;
    coefficients[HALF_LENGTH] += 1 - floatTotal;
    phases.push(coefficients);
  }
  FILTER_CACHE.set(key, phases);
  return phases;
}

/**
 * Return one finite arithmetic channel mean without allocating a mono block.
 * @param {Float32Array[] | undefined | null} channels
 * @param {number} index
 */
export function mixedChannelSample(channels, index) {
  let total = 0;
  let count = 0;
  const sourceChannels = channels || [];
  for (let channelIndex = 0; channelIndex < sourceChannels.length; channelIndex += 1) {
    const channel = sourceChannels[channelIndex];
    if (!channel || index >= channel.length) continue;
    const value = Number(channel[index]);
    total += Number.isFinite(value) ? value : 0;
    count += 1;
  }
  return count ? total / count : 0;
}

/**
 * Test/utility wrapper around the allocation-free per-sample mixer.
 * @param {Float32Array[] | undefined | null} channels
 * @param {number} [fallbackLength]
 */
export function downmixToMono(channels, fallbackLength = 0) {
  const sourceChannels = channels || [];
  let length = Math.max(0, Math.floor(Number(fallbackLength) || 0));
  for (let index = 0; index < sourceChannels.length; index += 1) {
    length = Math.max(length, sourceChannels[index]?.length || 0);
  }
  const mono = new Float32Array(length);
  for (let index = 0; index < length; index += 1) {
    mono[index] = mixedChannelSample(sourceChannels, index);
  }
  return mono;
}

/**
 * Stateful exact-rational, phase-indexed Kaiser-sinc capture resampler.
 * Production uses `writeSample`/`drainSample` directly, which performs no block
 * allocation. `push` and array-returning `flush` exist for deterministic tests.
 */
export class StatefulPolyphaseResampler {
  /** @param {number} inputRate @param {number} [outputRate] */
  constructor(inputRate, outputRate = DEFAULT_OUTPUT_RATE) {
    if (!Number.isInteger(inputRate) || inputRate <= CUTOFF_HZ * 2) {
      throw new Error("inputRate must be an integer above the 14.4 kHz filter transition");
    }
    if (!Number.isInteger(outputRate) || outputRate <= 0) {
      throw new Error("outputRate must be a positive integer");
    }
    this.inputRate = inputRate;
    this.outputRate = outputRate;
    const divisor = gcd(inputRate, outputRate);
    this.interpolation = outputRate / divisor;
    this.decimation = inputRate / divisor;
    this.phaseCount = Math.min(this.interpolation, MAX_BANK_PHASES);
    this._bank = filterBank(inputRate, this.phaseCount);
    this.halfLength = HALF_LENGTH;
    this.tapCount = TAP_COUNT;
    this.bankBytes = this._bank.length * TAP_COUNT * Float32Array.BYTES_PER_ELEMENT;
    this.lookaheadMs = (HALF_LENGTH / inputRate) * 1000;
    this.reset();
  }

  reset() {
    this._ring = new Float32Array(RING_CAPACITY);
    this._inputCount = 0;
    this._sourceIndex = 0;
    this._phase = 0;
    this._outputCount = 0;
  }

  get inputCount() {
    return this._inputCount;
  }

  get outputCount() {
    return this._outputCount;
  }

  get sourceIndex() {
    return this._sourceIndex;
  }

  /** @param {number} value */
  writeSample(value) {
    const finite = Number(value);
    this._ring[this._inputCount % RING_CAPACITY] = Number.isFinite(finite) ? finite : 0;
    this._inputCount += 1;
  }

  /** @param {number} absoluteIndex */
  _sample(absoluteIndex) {
    if (absoluteIndex < 0 || absoluteIndex >= this._inputCount) return 0;
    if (this._inputCount - absoluteIndex > RING_CAPACITY) {
      throw new Error("resampler ring history was overwritten before drain");
    }
    return this._ring[absoluteIndex % RING_CAPACITY];
  }

  /** @param {number} [maximumCenterExclusive] */
  canDrain(maximumCenterExclusive = Number.POSITIVE_INFINITY) {
    return this._sourceIndex < maximumCenterExclusive
      && this._sourceIndex + HALF_LENGTH < this._inputCount;
  }

  _phaseIndex() {
    if (this.phaseCount === this.interpolation) return this._phase;
    return Math.min(
      this.phaseCount - 1,
      Math.round((this._phase / this.interpolation) * this.phaseCount),
    );
  }

  _advance() {
    this._phase += this.decimation;
    this._sourceIndex += Math.floor(this._phase / this.interpolation);
    this._phase %= this.interpolation;
    this._outputCount += 1;
  }

  drainSample() {
    if (!this.canDrain()) throw new Error("resampler output requires more input lookahead");
    const coefficients = this._bank[this._phaseIndex()];
    let value = 0;
    for (let tapIndex = 0; tapIndex < TAP_COUNT; tapIndex += 1) {
      value += coefficients[tapIndex]
        * this._sample(this._sourceIndex + tapIndex - HALF_LENGTH);
    }
    this._advance();
    return value;
  }

  /** Drain one zero-extended endpoint sample while its center remains valid. */
  drainFlushedSample(maximumCenterExclusive = this._inputCount) {
    if (this._sourceIndex >= maximumCenterExclusive) {
      throw new Error("resampler endpoint is already drained");
    }
    const coefficients = this._bank[this._phaseIndex()];
    let value = 0;
    for (let tapIndex = 0; tapIndex < TAP_COUNT; tapIndex += 1) {
      value += coefficients[tapIndex]
        * this._sample(this._sourceIndex + tapIndex - HALF_LENGTH);
    }
    this._advance();
    return value;
  }

  /**
   * Allocation-free channel ingestion for worklets.
   * @param {Float32Array[] | undefined | null} channels
   * @param {number} fallbackLength
   * @param {(sample: number) => void} emit
   */
  processChannels(channels, fallbackLength, emit) {
    const sourceChannels = channels || [];
    let length = Math.max(0, Math.floor(Number(fallbackLength) || 0));
    for (let index = 0; index < sourceChannels.length; index += 1) {
      length = Math.max(length, sourceChannels[index]?.length || 0);
    }
    let emitted = 0;
    for (let index = 0; index < length; index += 1) {
      this.writeSample(mixedChannelSample(sourceChannels, index));
      while (this.canDrain()) {
        emit(this.drainSample());
        emitted += 1;
      }
    }
    return emitted;
  }

  /** @param {Float32Array | number[]} samples */
  push(samples) {
    const values = [];
    const input = samples instanceof Float32Array ? samples : Float32Array.from(samples || []);
    this.processChannels([input], input.length, (sample) => values.push(sample));
    return Float32Array.from(values);
  }

  /**
   * Zero-extend once and drain exactly the remaining duration-preserving output.
   * @param {(sample: number) => void} [emit]
   * @param {number} [maximumCenterExclusive]
   */
  flush(emit, maximumCenterExclusive = this._inputCount) {
    const values = emit ? null : [];
    let emitted = 0;
    while (this._sourceIndex < maximumCenterExclusive) {
      const sample = this.drainFlushedSample(maximumCenterExclusive);
      if (emit) emit(sample);
      else values.push(sample);
      emitted += 1;
    }
    this.reset();
    return emit ? emitted : Float32Array.from(values);
  }
}
