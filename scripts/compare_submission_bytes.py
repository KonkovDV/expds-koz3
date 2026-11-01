#!/usr/bin/env python3
"""Byte-for-byte compare of two scored submissions (geojson or zip).

--v2-ref / REF may be:
  * result.geojson (or any .geojson FeatureCollection)
  * a submission zip containing a scored geojson (extracted for compare)

sha256 of the *given* REF file (as provided, before extract) is always reported.

Exit 0 if identical; 1 on mismatch; 2 on IO/args.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _looks_like_geojson(raw: bytes) -> bool:
    s = raw.lstrip()[:64]
    if not s.startswith(b"{") and not s.startswith(b"["):
        return False
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    if isinstance(obj, dict) and obj.get("type") == "FeatureCollection":
        return True
    return False


def extract_scored_bytes(path: Path) -> tuple[bytes, str]:
    """Return (payload_bytes, kind) where kind is 'geojson' or 'zip:<member>'.

    For zip: prefer result.geojson, else first *.geojson FeatureCollection.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"missing: {path}")
    raw = path.read_bytes()
    suffix = path.suffix.lower()
    if suffix in (".geojson", ".json") or _looks_like_geojson(raw):
        if not _looks_like_geojson(raw):
            raise ValueError(f"not a FeatureCollection geojson: {path}")
        return raw, "geojson"
    if suffix == ".zip" or zipfile.is_zipfile(io.BytesIO(raw)):
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            prefer = [n for n in names if n.replace("\\", "/").rsplit("/", 1)[-1].lower() == "result.geojson"]
            candidates = prefer or [n for n in names if n.lower().endswith(".geojson")]
            for n in candidates:
                data = zf.read(n)
                if _looks_like_geojson(data):
                    return data, f"zip:{n}"
            raise ValueError(f"no scored FeatureCollection geojson inside zip: {path}")
    raise ValueError(f"unsupported --v2-ref type (want .geojson or .zip): {path}")


def first_diff_offset(a: bytes, b: bytes) -> int | None:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return None


def hex_window(data: bytes, off: int, radius: int = 16) -> str:
    lo = max(0, off - radius)
    hi = min(len(data), off + radius)
    return data[lo:hi].hex()


def compare_payloads(ref_payload: bytes, cand_payload: bytes) -> list[str]:
    errs: list[str] = []
    if ref_payload == cand_payload:
        return []
    errs.append(f"size: REF_payload={len(ref_payload)} CAND_payload={len(cand_payload)}")
    errs.append(f"sha256 REF_payload={sha256_bytes(ref_payload)}")
    errs.append(f"sha256 CAND_payload={sha256_bytes(cand_payload)}")
    off = first_diff_offset(ref_payload, cand_payload)
    if off is not None:
        errs.append(f"first_diff_offset={off}")
        errs.append(f"REF[{off}±16]={hex_window(ref_payload, off)}")
        errs.append(f"CAND[{off}±16]={hex_window(cand_payload, off)}")
    return errs


def compare_paths(ref: Path, cand: Path) -> tuple[list[str], dict]:
    """Compare scored payloads. Returns (errs, meta with file/payload shas)."""
    ref, cand = Path(ref), Path(cand)
    meta: dict = {
        "ref_path": str(ref),
        "cand_path": str(cand),
        "ref_file_sha256": None,
        "cand_file_sha256": None,
    }
    try:
        if not ref.is_file():
            raise FileNotFoundError(f"missing: {ref}")
        if not cand.is_file():
            raise FileNotFoundError(f"missing: {cand}")
        meta["ref_file_sha256"] = sha256_file(ref)
        meta["cand_file_sha256"] = sha256_file(cand)
        ref_payload, ref_kind = extract_scored_bytes(ref)
        cand_payload, cand_kind = extract_scored_bytes(cand)
    except (OSError, ValueError) as exc:
        return [str(exc)], meta
    meta["ref_kind"] = ref_kind
    meta["cand_kind"] = cand_kind
    meta["ref_payload_sha256"] = sha256_bytes(ref_payload)
    meta["cand_payload_sha256"] = sha256_bytes(cand_payload)
    return compare_payloads(ref_payload, cand_payload), meta


def compare(ref: Path, cand: Path) -> list[str]:
    """Convenience: errs only (used by tests / callers)."""
    errs, _meta = compare_paths(Path(ref), Path(cand))
    return errs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ref", type=Path)
    ap.add_argument("candidate", type=Path)
    args = ap.parse_args(argv)
    try:
        errs, meta = compare_paths(args.ref, args.candidate)
    except OSError as exc:
        print(f"FAIL IO: {exc}", file=sys.stderr)
        return 2
    print(f"ref_file_sha256={meta.get('ref_file_sha256')}")
    if "ref_kind" in meta:
        print(f"ref_kind={meta['ref_kind']} payload_sha256={meta.get('ref_payload_sha256')}")
    if not errs:
        print("PASS byte-identical scored payload")
        return 0
    print("FAIL OFF≠v2 (or REF≠CAND) byte compare:", file=sys.stderr)
    for e in errs:
        print(f"  {e}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
