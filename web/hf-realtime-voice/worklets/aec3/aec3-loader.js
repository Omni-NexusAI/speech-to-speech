// @ts-check

import { AEC3_ABI_VERSION } from "./aec3-abi.js?v=3-opaque-echo-route";

const DEFAULT_MANIFEST_URL = new URL("./aec3.manifest.json", import.meta.url);
const PROCESSOR_URL = new URL("./aec3-capture.js?v=8-stateful-polyphase", import.meta.url);

function hex(bytes) {
  return [...new Uint8Array(bytes)].map((value) => value.toString(16).padStart(2, "0")).join("");
}

/**
 * Fetch, authenticate, compile, and register the AEC3 AudioWorklet module.
 * Failure is data, not a fake capability: callers must keep Adaptive on Native.
 * @param {AudioContext} audioContext
 * @param {{ manifestUrl?: URL | string, fetchImpl?: typeof fetch }} [options]
 */
export async function loadAec3Worklet(audioContext, options = {}) {
  const fetchImpl = options.fetchImpl || fetch;
  const manifestUrl = new URL(String(options.manifestUrl || DEFAULT_MANIFEST_URL), import.meta.url);
  try {
    const manifestResponse = await fetchImpl(manifestUrl, { cache: "no-store" });
    if (!manifestResponse.ok) throw new Error(`manifest HTTP ${manifestResponse.status}`);
    const manifest = await manifestResponse.json();
    if (!manifest?.available) throw new Error(manifest?.blocker || "AEC3 artifact is disabled");
    if (Number(manifest.abiVersion) !== AEC3_ABI_VERSION) {
      throw new Error(`manifest ABI ${manifest.abiVersion} is unsupported`);
    }
    if (!/^[0-9a-f]{64}$/i.test(String(manifest.sha256 || ""))) {
      throw new Error("manifest SHA-256 is missing or invalid");
    }
    const wasmUrl = new URL(String(manifest.wasm || "aec3.wasm"), manifestUrl);
    const wasmResponse = await fetchImpl(wasmUrl, { cache: "no-store" });
    if (!wasmResponse.ok) throw new Error(`WASM HTTP ${wasmResponse.status}`);
    const wasmBytes = await wasmResponse.arrayBuffer();
    const digest = hex(await crypto.subtle.digest("SHA-256", wasmBytes));
    if (digest.toLowerCase() !== String(manifest.sha256).toLowerCase()) {
      throw new Error("AEC3 WASM SHA-256 does not match the pinned manifest");
    }
    const module = await WebAssembly.compile(wasmBytes);
    const imports = WebAssembly.Module.imports(module);
    if (imports.length) {
      throw new Error(`AEC3 WASM unexpectedly imports ${imports.map((item) => `${item.module}.${item.name}`).join(", ")}`);
    }
    await audioContext.audioWorklet.addModule(PROCESSOR_URL.href);
    return {
      available: true,
      module,
      manifest,
      processorName: "aec3-capture",
      processorUrl: PROCESSOR_URL.href,
      reason: "",
    };
  } catch (error) {
    return {
      available: false,
      module: null,
      manifest: null,
      processorName: "mic-capture",
      processorUrl: "",
      reason: error instanceof Error ? error.message : String(error),
    };
  }
}

/**
 * Build processor options for `new AudioWorkletNode(ctx, result.processorName, ...)`.
 * A compiled WebAssembly.Module is structured-cloneable in supported browsers.
 * @param {Awaited<ReturnType<typeof loadAec3Worklet>>} result
 * @param {Record<string, unknown>} [options]
 */
export function aec3ProcessorOptions(result, options = {}) {
  return {
    ...options,
    ...(result.available ? {
      aec3Module: result.module,
      aec3Manifest: {
        abiVersion: result.manifest.abiVersion,
        engine: result.manifest.engine,
        sourceRevision: result.manifest.sourceRevision,
      },
    } : {}),
  };
}
