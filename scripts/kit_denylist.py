#!/usr/bin/env python3
"""Kit-file denylist for scored submission zips (LESSONS §5 + check_compliance).

A *kit* is the code/weights archive uploaded to the platform (≤2 GB).
A *submission* is the scored FeatureCollection zip (≤100 MB, result.geojson).
Compliance must target the submission — never the kit (LESSONS §5).

Members that mark a zip as a kit (reject when validating a scored submission):
"""
from __future__ import annotations

# Basename / path-fragment denylist (case-insensitive match on zip member paths).
KIT_MEMBER_SUFFIXES = (
    ".pt",
    ".pth",
    ".onnx",
    ".ckpt",
    ".safetensors",
    ".bin",
    ".whl",
    ".so",
)

KIT_MEMBER_NAMES = {
    "geo_pixgbm.py",
    "geo_finds_gbm.py",
    "pix_legacy.py",
    "geo_backbone.py",
    "metadata.json",  # kit root marker when paired with models/
    "main.py",        # inference/main.py in kits
}

KIT_PATH_PREFIXES = (
    "models/",
    "expds/",
    "inference/",
    "configs/",
    "w4pix/",
    "analysis/",
    "pc/tools/",
)

KIT_NAME_HINTS = ("v3on", "kit", "koz3_", "submission_bot", "weights/")

# DOC-1 / EN denylist names (also banned inside scored archives).
DOC1_BANNED_NAMES = {
    "geo_pixgbm.py",
    "geo_finds_gbm.py",
    "pix_legacy.py",
}


def member_is_kit_file(name: str) -> str | None:
    """Return reason if zip member *name* looks like a kit file, else None."""
    n = name.replace("\\", "/").lstrip("./")
    low = n.lower()
    base = n.rsplit("/", 1)[-1].lower()
    if base in {x.lower() for x in KIT_MEMBER_NAMES} and (
        low.startswith("models/") or low.startswith("expds/") or low.startswith("inference/")
        or base.endswith(".py") or base == "metadata.json"
    ):
        # metadata.json / main.py alone in a geojson zip are rare; still flag under kit trees
        if base in ("metadata.json", "main.py") and not (
            low.startswith("inference/") or n == "metadata.json"
        ):
            pass
        elif base in ("metadata.json",) and n == "metadata.json":
            return f"kit root marker: {n}"
        elif base == "main.py" and "inference/" in low:
            return f"kit inference entry: {n}"
        elif base.endswith(".py") and (low.startswith("expds/") or base in DOC1_BANNED_NAMES):
            return f"kit/code member: {n}"
    if any(low.endswith(suf) for suf in KIT_MEMBER_SUFFIXES):
        return f"weight/binary member: {n}"
    if any(low.startswith(p) or f"/{p}" in f"/{low}" for p in KIT_PATH_PREFIXES):
        # Allow nothing under models/ etc. inside a scored submission
        return f"kit path prefix: {n}"
    if base in DOC1_BANNED_NAMES:
        return f"DOC-1 banned: {n}"
    return None


def kit_members(names: list[str]) -> list[str]:
    """List human-readable violations for kit-like members."""
    out = []
    for n in names:
        reason = member_is_kit_file(n)
        if reason:
            out.append(reason)
    return out
