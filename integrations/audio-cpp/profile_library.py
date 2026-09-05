"""Canonical on-disk Base-clone inventory shared by the candidate surfaces."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

PROFILE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")
PROFILE_PROVIDER = "qwen3tts-audiocpp"
_HIDDEN_NAMES = {"__pycache__", ".cache", ".trash"}


def clone_content_hash(
    reference_audio: bytes,
    reference_text: str,
    excerpts: list[tuple[bytes, str]] | None = None,
) -> str:
    """Return the immutable conditioning identity for a Base clone.

    A clone ID identifies a mutable library slot.  The audio.cpp request path
    needs a distinct identity for the actual full-ICL material so an edit made
    between phrases cannot silently change an answer already in progress.
    Length-delimited fields avoid ambiguities such as ``[a, bc]`` versus
    ``[ab, c]`` while keeping the value independent of profile presentation
    metadata (name, description, timestamps).
    """

    digest = sha256()

    def update(value: bytes) -> None:
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)

    update(reference_audio)
    update(str(reference_text).encode("utf-8"))
    for excerpt_audio, excerpt_text in excerpts or []:
        update(excerpt_audio)
        update(str(excerpt_text).encode("utf-8"))
    return digest.hexdigest()


def _profile_content_material(profile_root: Path, raw: dict[str, Any], reference: Path) -> tuple[str, list[tuple[bytes, str]]] | None:
    """Read the exact conditioning pair(s) accepted by the supervisor."""

    reference_text = str(raw.get("ref_text") or "")
    excerpts: list[tuple[bytes, str]] = []
    raw_excerpts = raw.get("reference_excerpts")
    if isinstance(raw_excerpts, list):
        for item in raw_excerpts:
            if not isinstance(item, dict):
                continue
            filename = str(item.get("ref_audio_filename") or "").strip()
            text = str(item.get("ref_text") or "").strip()
            if not filename or Path(filename).name != filename or not text:
                continue
            path = profile_root / filename
            if not path.is_file():
                continue
            try:
                excerpts.append((path.read_bytes(), text))
            except OSError:
                return None
    return reference_text, excerpts


def valid_profile_id(profile_id: str) -> bool:
    """Return whether an on-disk directory name is a safe public profile id."""
    return bool(PROFILE_ID_PATTERN.fullmatch(profile_id)) and not profile_id.startswith(".")


def _created_at(reference: Path) -> str:
    """Derive a stable legacy timestamp from the immutable reference recording."""
    return datetime.fromtimestamp(reference.stat().st_mtime, timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_profile(
    library_dir: Path,
    profile_id: str,
    *,
    normalize: bool = True,
) -> dict[str, Any] | None:
    """Return one usable Base clone, atomically normalizing safe legacy metadata.

    The directory name is the identity authority. A profile is live only when
    its direct metadata file and referenced audio are both present. Hidden,
    cache-like, path-traversing, malformed, and non-Base entries never enter an
    API or Gradio inventory.
    """
    if not valid_profile_id(profile_id) or profile_id in _HIDDEN_NAMES:
        return None
    profile_root = library_dir / "profiles" / profile_id
    metadata_path = profile_root / "meta.json"
    if not profile_root.is_dir() or not metadata_path.is_file():
        return None
    try:
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(raw, dict):
        return None

    task_type = str(raw.get("task_type") or "Base").strip()
    if task_type.lower() != "base":
        return None
    reference_name = str(raw.get("ref_audio_filename") or "ref_audio.wav").strip()
    if not reference_name or Path(reference_name).name != reference_name:
        return None
    reference = profile_root / reference_name
    if not reference.is_file():
        return None
    try:
        if not reference.resolve().is_relative_to(profile_root.resolve()):
            return None
    except OSError:
        return None

    material = _profile_content_material(profile_root, raw, reference)
    if material is None:
        return None
    reference_text, excerpts = material
    try:
        content_hash = clone_content_hash(reference.read_bytes(), reference_text, excerpts)
    except OSError:
        return None
    raw_revision = raw.get("content_revision", 1)
    content_revision = raw_revision if isinstance(raw_revision, int) and not isinstance(raw_revision, bool) and raw_revision > 0 else 1
    # A file-level edit outside the normal supervisor mutation route must not
    # retain the previous conditioning revision. Legacy profiles without a
    # stored hash establish revision 1 on first canonicalization.
    stored_hash = raw.get("content_hash")
    if isinstance(stored_hash, str) and re.fullmatch(r"[0-9a-f]{64}", stored_hash) and stored_hash != content_hash:
        content_revision += 1

    normalized = dict(raw)
    normalized.update(
        profile_id=profile_id,
        name=str(raw.get("name") or profile_id),
        task_type="Base",
        created_at=str(raw.get("created_at") or _created_at(reference)),
        language=str(raw.get("language") or "Auto"),
        voice=f"clone:{profile_id}",
        instructions=str(raw.get("instructions") or ""),
        ref_text=reference_text,
        x_vector_only_mode=False,
        ref_audio_filename=reference_name,
        origin=str(raw.get("origin") or "audio.cpp Base clone"),
        provider=PROFILE_PROVIDER,
        content_revision=content_revision,
        content_hash=content_hash,
    )
    if normalize and normalized != raw:
        temporary = profile_root / ".meta.json.normalize.tmp"
        try:
            temporary.write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            temporary.replace(metadata_path)
        except OSError:
            temporary.unlink(missing_ok=True)
    return normalized


def live_profiles(library_dir: Path, *, normalize: bool = True) -> list[dict[str, Any]]:
    """Scan the current direct profile directories without inventory caching."""
    root = library_dir / "profiles"
    if not root.is_dir():
        return []
    profiles: list[dict[str, Any]] = []
    for child in sorted(root.iterdir(), key=lambda item: item.name.lower()):
        if not child.is_dir() or child.name.startswith(".") or child.name in _HIDDEN_NAMES:
            continue
        profile = canonical_profile(library_dir, child.name, normalize=normalize)
        if profile is not None:
            profiles.append(profile)
    profiles.sort(key=lambda item: (str(item.get("created_at") or ""), str(item["profile_id"])), reverse=True)
    return profiles
