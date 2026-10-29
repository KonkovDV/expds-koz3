#!/usr/bin/env python3
"""Hard FAIL if Or D-FINE produced zero detections (bot5 v4 e2e gate).

PROTOCOL: an Or-only `[dfine_aeor]` mock that yields n_raw==0 / post-thr count==0
must not be scored as 'no change' / PASS — that is how ImportError-on-PC
(looked like A≡B with Δ=0) slipped past A2. Exit 0 only when Or dets > 0.

Usage:
  python scripts/assert_dfine_aeor_or_nonzero.py \\
      --geojson OUT/result.geojson [--main-log OUT/main.log] [--dump-json stages.json]

Checks (any available source must show >0; missing sources are skipped):
  1) geojson features with threshold_group in {dfine_or, or_dfine}
  2) main.log lines `dfine_aeor n_det=<N>` (prefer final `n_det=N total`)
  3) dump JSON keys n_raw_boxes / n_features / n_would_pass_thr / n_det

Also FAIL immediately if log contains DFINE_AEOR_UNAVAILABLE / ImportError load fail.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

OR_GROUPS = frozenset({"dfine_or", "or_dfine"})


def _load_geojson_or_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    n = 0
    for feat in data.get("features") or []:
        props = feat.get("properties") or {}
        g = props.get("threshold_group") or props.get("group")
        if g in OR_GROUPS:
            n += 1
    return n


def _parse_main_log(path: Path) -> tuple[int | None, list[str]]:
    """Return (n_det_total or None, fail_markers)."""
    if not path.is_file():
        return None, []
    text = path.read_text(encoding="utf-8", errors="replace")
    markers = []
    if "DFINE_AEOR_UNAVAILABLE" in text:
        markers.append("DFINE_AEOR_UNAVAILABLE")
    if "SP_DFINE_UNAVAILABLE" in text and "ImportError" in text:
        # shared import_dfine path — same root cause for AeOr
        markers.append("SP_DFINE_UNAVAILABLE+ImportError")
    if re.search(r"load_failed:ImportError|cannot import name '_regex'|No module named 'transformers'", text):
        markers.append("ImportError-in-log")
    # Prefer summary line
    totals = [int(m) for m in re.findall(r"dfine_aeor n_det=(\d+)\s+total", text)]
    if totals:
        return totals[-1], markers
    per = [int(m) for m in re.findall(r"dfine_aeor n_det=(\d+)\s+region=", text)]
    if per:
        return sum(per), markers
    return None, markers


def _dump_raw(path: Path) -> int | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("n_raw_boxes", "n_features", "n_would_pass_thr", "n_det", "n_sp_dfine_in_state"):
        if key in data and isinstance(data[key], (int, float)):
            return int(data[key])
    totals = data.get("totals") or {}
    for key in ("n_raw_boxes", "n_features", "n_would_pass_thr", "n_det"):
        if key in totals and isinstance(totals[key], (int, float)):
            return int(totals[key])
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geojson", type=Path, required=True)
    ap.add_argument("--main-log", type=Path, default=None)
    ap.add_argument("--dump-json", type=Path, default=None)
    args = ap.parse_args()

    reasons: list[str] = []
    counts: dict[str, int | None] = {}

    n_geo = _load_geojson_or_count(args.geojson)
    counts["geojson_dfine_or"] = n_geo
    if n_geo is None:
        reasons.append(f"geojson missing: {args.geojson}")
    elif n_geo == 0:
        reasons.append("geojson Or/dfine_or count == 0")

    n_log, markers = (None, [])
    if args.main_log is not None:
        n_log, markers = _parse_main_log(args.main_log)
        counts["main_log_n_det"] = n_log
        for m in markers:
            reasons.append(f"log marker: {m}")
        if n_log is not None and n_log == 0:
            reasons.append("main.log dfine_aeor n_det == 0")

    n_dump = None
    if args.dump_json is not None:
        n_dump = _dump_raw(args.dump_json)
        counts["dump_n_raw"] = n_dump
        if n_dump is not None and n_dump == 0:
            reasons.append("dump n_raw/n_det == 0")

    print(json.dumps({"counts": counts, "markers": markers, "reasons": reasons}, ensure_ascii=False, indent=2))

    # Hard FAIL: any observed zero, or no positive evidence at all
    observed = [v for v in (n_geo, n_log, n_dump) if v is not None]
    if markers:
        print("FAIL: D-FINE import/unavailable markers present (n_raw=0 is not 'no change')", file=sys.stderr)
        return 2
    if not observed:
        print("FAIL: no Or detection evidence found (geojson/log/dump)", file=sys.stderr)
        return 2
    if any(v == 0 for v in observed):
        print("FAIL: Or detections == 0 (n_raw or post-threshold). Not a PASS.", file=sys.stderr)
        return 2
    if not any(v > 0 for v in observed):
        print("FAIL: Or detections not > 0", file=sys.stderr)
        return 2
    print("OK: Or dfine_aeor detections > 0", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
