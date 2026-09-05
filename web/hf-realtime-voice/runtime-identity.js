// @ts-check

const REVISION_RE = /^[0-9a-f]{7,64}$/i;
const FINGERPRINT_RE = /^[0-9a-f]{64}$/i;
const ASSET_RE = /^main=[A-Za-z0-9._-]+;ws=[A-Za-z0-9._-]+;chat=[A-Za-z0-9._-]+;playback=[A-Za-z0-9._-]+$/;

/** Compare the four immutable, content-free runtime identity fields exactly. */
export function runtimeIdentityMatches(frontend, backend) {
  if (!frontend || !backend || typeof frontend !== "object" || typeof backend !== "object") {
    return false;
  }
  if (!REVISION_RE.test(frontend.source_revision) || !REVISION_RE.test(backend.source_revision)) return false;
  if (typeof frontend.source_dirty !== "boolean" || typeof backend.source_dirty !== "boolean") return false;
  if (!FINGERPRINT_RE.test(frontend.source_fingerprint) || !FINGERPRINT_RE.test(backend.source_fingerprint)) return false;
  if (!ASSET_RE.test(frontend.ui_asset_generation) || !ASSET_RE.test(backend.ui_asset_generation)) return false;
  return frontend.source_revision === backend.source_revision
    && frontend.source_dirty === backend.source_dirty
    && frontend.source_fingerprint === backend.source_fingerprint
    && frontend.ui_asset_generation === backend.ui_asset_generation;
}
