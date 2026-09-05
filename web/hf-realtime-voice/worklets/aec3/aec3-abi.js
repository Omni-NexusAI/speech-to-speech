// @ts-check

export const AEC3_ABI_VERSION = 1;
export const AEC3_FRAME_MS = 10;
export const AEC3_OUTPUT_RATE = 16000;

const REQUIRED_EXPORTS = [
  "memory",
  "aec3_abi_version",
  "aec3_create",
  "aec3_destroy",
  "aec3_reset",
  "aec3_frame_samples",
  "aec3_render_ptr",
  "aec3_capture_ptr",
  "aec3_output_ptr",
  "aec3_set_suppression",
  "aec3_process_render",
  "aec3_process_capture",
  "aec3_get_delay_ms",
  "aec3_get_erle_db",
  "aec3_get_residual_echo_likelihood",
  "aec3_get_double_talk",
  "aec3_get_render_rms",
  "aec3_get_capture_rms",
  "aec3_get_output_rms",
];

/**
 * Validate the narrow, import-free Sonora AEC3 ABI used in the worklet.
 * @param {WebAssembly.Exports | Record<string, any>} exports
 * @returns {{ ok: boolean, missing: string[], reason: string }}
 */
export function validateAec3Exports(exports) {
  const missing = REQUIRED_EXPORTS.filter((name) => !(name in exports));
  if (missing.length) {
    return { ok: false, missing, reason: `AEC3 ABI is missing: ${missing.join(", ")}` };
  }
  if (!(exports.memory instanceof WebAssembly.Memory)) {
    return { ok: false, missing: ["memory"], reason: "AEC3 memory export is invalid" };
  }
  const version = Number(exports.aec3_abi_version());
  if (version !== AEC3_ABI_VERSION) {
    return {
      ok: false,
      missing: [],
      reason: `AEC3 ABI ${version} does not match expected ${AEC3_ABI_VERSION}`,
    };
  }
  return { ok: true, missing: [], reason: "" };
}

/** Thin synchronous wrapper around the import-free WebAssembly ABI. */
export class Aec3WasmSession {
  /**
   * @param {WebAssembly.Exports | Record<string, any>} exports
   * @param {number} sampleRateHz
   */
  constructor(exports, sampleRateHz) {
    const validation = validateAec3Exports(exports);
    if (!validation.ok) throw new Error(validation.reason);
    if (!Number.isInteger(sampleRateHz) || sampleRateHz < 8000 || sampleRateHz > 96000) {
      throw new Error(`Unsupported AEC3 sample rate: ${sampleRateHz}`);
    }
    this.exports = exports;
    this.sampleRateHz = sampleRateHz;
    this.handle = Number(exports.aec3_create(sampleRateHz, 1));
    if (!this.handle) throw new Error("AEC3 session creation failed");
    this.frameSamples = Number(exports.aec3_frame_samples(this.handle));
    const expected = Math.floor(sampleRateHz / 100);
    if (this.frameSamples !== expected || this.frameSamples <= 0) {
      this.destroy();
      throw new Error(`AEC3 returned ${this.frameSamples} frame samples; expected ${expected}`);
    }
    this.renderPtr = Number(exports.aec3_render_ptr(this.handle));
    this.capturePtr = Number(exports.aec3_capture_ptr(this.handle));
    this.outputPtr = Number(exports.aec3_output_ptr(this.handle));
    if (!this.renderPtr || !this.capturePtr || !this.outputPtr) {
      this.destroy();
      throw new Error("AEC3 returned an invalid audio buffer pointer");
    }
  }

  /** @param {number} pointer */
  _view(pointer) {
    return new Float32Array(this.exports.memory.buffer, pointer, this.frameSamples);
  }

  /**
   * Process one aligned render/capture pair. Render is always submitted first.
   * @param {Float32Array} render
   * @param {Float32Array} capture
   * @param {{ delayMs?: number, timestampMs?: number, strict?: boolean }} [options]
   */
  process(render, capture, options = {}) {
    if (!this.handle) throw new Error("AEC3 session is closed");
    if (render.length !== this.frameSamples || capture.length !== this.frameSamples) {
      throw new Error(`AEC3 requires aligned ${this.frameSamples}-sample frames`);
    }
    const delayMs = Math.max(0, Math.min(500, Math.round(Number(options.delayMs) || 0)));
    const timestampMs = Number.isFinite(options.timestampMs) ? Number(options.timestampMs) : 0;
    const strict = options.strict ? 1 : 0;
    if (Number(this.exports.aec3_set_suppression(this.handle, strict)) !== 0) {
      throw new Error("AEC3 suppression-mode update failed");
    }
    this._view(this.renderPtr).set(render);
    this._view(this.capturePtr).set(capture);
    if (Number(this.exports.aec3_process_render(this.handle, timestampMs)) !== 0) {
      throw new Error("AEC3 render processing failed");
    }
    if (Number(this.exports.aec3_process_capture(this.handle, delayMs, timestampMs)) !== 0) {
      throw new Error("AEC3 capture processing failed");
    }
    const output = this._view(this.outputPtr).slice();
    for (let index = 0; index < output.length; index += 1) {
      if (!Number.isFinite(output[index])) throw new Error("AEC3 produced non-finite PCM");
    }
    return { output, metrics: this.metrics() };
  }

  metrics() {
    const finiteOrNull = (value) => Number.isFinite(Number(value)) ? Number(value) : null;
    const doubleTalkRaw = Number(this.exports.aec3_get_double_talk(this.handle));
    return {
      delayMs: finiteOrNull(this.exports.aec3_get_delay_ms(this.handle)),
      erleDb: finiteOrNull(this.exports.aec3_get_erle_db(this.handle)),
      residualEchoLikelihood: finiteOrNull(
        this.exports.aec3_get_residual_echo_likelihood(this.handle),
      ),
      // Sonora keeps AEC3's internal near-end detector private. This exported
      // value is post-AEC near-end evidence for diagnostics only; it never gates
      // Adaptive audio or claims to be the private internal detector state.
      doubleTalk: doubleTalkRaw < 0 ? null : doubleTalkRaw !== 0,
      doubleTalkSource: "aec3-output-evidence",
      renderRms: finiteOrNull(this.exports.aec3_get_render_rms(this.handle)),
      captureRms: finiteOrNull(this.exports.aec3_get_capture_rms(this.handle)),
      outputRms: finiteOrNull(this.exports.aec3_get_output_rms(this.handle)),
    };
  }

  reset() {
    if (!this.handle || Number(this.exports.aec3_reset(this.handle)) !== 0) {
      throw new Error("AEC3 reset failed");
    }
  }

  destroy() {
    if (!this.handle) return;
    this.exports.aec3_destroy(this.handle);
    this.handle = 0;
  }
}
