from __future__ import annotations

import os
import re
from typing import Any

_REVISION_RE = re.compile(r"^[0-9a-f]{7,64}$", re.IGNORECASE)
_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
_ASSET_TOKEN = r"[A-Za-z0-9._-]+"
_ASSET_GENERATION_RE = re.compile(
    rf"^main={_ASSET_TOKEN};ws={_ASSET_TOKEN};chat={_ASSET_TOKEN};playback={_ASSET_TOKEN}$"
)
_MAX_ASSET_GENERATION_LENGTH = 256


def _bounded_environment_value(name: str, *, maximum: int) -> str | None:
    value = os.environ.get(name, "").strip()
    if not value or len(value) > maximum or any(ord(character) < 32 for character in value):
        return None
    return value


def runtime_source_identity() -> dict[str, Any]:
    """Return the content-free source identity captured by the managed launcher."""
    revision = _bounded_environment_value("S2S_RUNTIME_REVISION", maximum=64)
    fingerprint = _bounded_environment_value("S2S_RUNTIME_SOURCE_FINGERPRINT", maximum=64)
    asset_generation = _bounded_environment_value(
        "S2S_UI_ASSET_GENERATION",
        maximum=_MAX_ASSET_GENERATION_LENGTH,
    )
    dirty_value = os.environ.get("S2S_RUNTIME_DIRTY", "").strip().lower()
    dirty = {"0": False, "1": True}.get(dirty_value)
    return {
        "source_revision": revision.lower() if revision and _REVISION_RE.fullmatch(revision) else "unknown",
        "source_dirty": dirty,
        "source_fingerprint": (
            fingerprint.lower() if fingerprint and _FINGERPRINT_RE.fullmatch(fingerprint) else "unknown"
        ),
        "ui_asset_generation": (
            asset_generation
            if asset_generation and _ASSET_GENERATION_RE.fullmatch(asset_generation)
            else "unknown"
        ),
    }
