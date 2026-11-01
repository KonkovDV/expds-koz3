#!/usr/bin/env python3
"""Byte-for-byte compare of two files (scored submission or any artifact).

Exit 0 if identical; 1 on mismatch (prints a clear diff summary); 2 on IO error.

    python scripts/compare_submission_bytes.py REF CANDIDATE
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def compare(ref: Path, cand: Path, *, max_load: int = 64 * 1024 * 1024) -> list[str]:
    """Return list of mismatch lines (empty ⇒ identical)."""
    errs: list[str] = []
    if not ref.is_file():
        return [f"REF missing: {ref}"]
    if not cand.is_file():
        return [f"CANDIDATE missing: {cand}"]
    sa, sb = ref.stat().st_size, cand.stat().st_size
    ha, hb = sha256(ref), sha256(cand)
    if ha == hb and sa == sb:
        return []
    errs.append(f"size: REF={sa} CAND={sb}")
    errs.append(f"sha256 REF={ha}")
    errs.append(f"sha256 CAND={hb}")
    # Load for offset hint when both fit in memory budget
    if sa <= max_load and sb <= max_load:
        ba, bb = ref.read_bytes(), cand.read_bytes()
        off = first_diff_offset(ba, bb)
        if off is not None:
            errs.append(f"first_diff_offset={off}")
            errs.append(f"REF[{off}±16]={hex_window(ba, off)}")
            errs.append(f"CAND[{off}±16]={hex_window(bb, off)}")
    else:
        errs.append("files too large for inline hex window; sha256 mismatch is authoritative")
    return errs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ref", type=Path)
    ap.add_argument("candidate", type=Path)
    args = ap.parse_args(argv)
    try:
        errs = compare(args.ref, args.candidate)
    except OSError as exc:
        print(f"FAIL IO: {exc}", file=sys.stderr)
        return 2
    if not errs:
        print(f"PASS byte-identical sha256={sha256(args.ref)}")
        return 0
    print("FAIL OFF≠v2 (or REF≠CAND) byte compare:", file=sys.stderr)
    for e in errs:
        print(f"  {e}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
