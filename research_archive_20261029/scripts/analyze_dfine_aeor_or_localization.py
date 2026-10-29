#!/usr/bin/env python3
"""bot5 (c′) — Or D-FINE localization vs GT (reuse bot2 A3 checklist).

When (c′) Docker e2e lands, run immediately:

  python scripts/analyze_dfine_aeor_or_localization.py \\
      --pred pc/results/bot5_dfine_aeor_or_<sha>/out_on/result.geojson \\
      --gt   <path/to/gt.geojson> \\
      [--dump pc/results/.../dump_on/stages.json] \\
      [--off  pc/results/.../out_off/result.geojson] \\
      --out  pc/results/bot5_dfine_aeor_or_<sha>/analysis/

Reports (EPSG:3857 nearest-any-GT, same as bot2 A3):
  - n_or (threshold_group in {dfine_or, or_dfine})
  - score p50/p90/max
  - dist_any_m: median, within 2 / 50 / 200 m
  - soft TP count at match radius 2 m (POINT_R_TOLERANCE)
  - optional: Δ vs OFF feature counts; dump n_raw / n_would_pass_thr

FYI bot2 A3 (SpOr D-FINE): 13 pass_thr all FP, median ~388 m, thr-sweep ~0 TP
→ expect similar domain risk for Or weights; do NOT treat Δ=0 as success if
assert_nonzero only checked count>0.

No PC run from this script alone — analysis only.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

OR_GROUPS = frozenset({"dfine_or", "or_dfine", "or"})
MATCH_RADII_M = (2.0, 50.0, 200.0)


def _centroid_xy(feat: dict) -> tuple[float, float] | None:
    geom = feat.get("geometry") or {}
    t = geom.get("type")
    c = geom.get("coordinates")
    if not c:
        return None
    if t == "Point":
        return float(c[0]), float(c[1])
    if t == "Polygon":
        ring = c[0] if c else None
        if not ring:
            return None
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        return sum(xs) / len(xs), sum(ys) / len(ys)
    if t == "MultiPolygon":
        # first poly outer ring
        ring = c[0][0] if c and c[0] else None
        if not ring:
            return None
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        return sum(xs) / len(xs), sum(ys) / len(ys)
    return None


def _load_points(path: Path, *, or_only: bool = False, gt: bool = False) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for feat in data.get("features") or []:
        props = feat.get("properties") or {}
        if or_only:
            g = props.get("threshold_group") or props.get("group") or props.get("markup_type")
            # Or-only dets: threshold_group dfine_or/or_dfine, or markup or with dfine source
            src = str(props.get("source") or props.get("model") or "")
            tg = str(g or "")
            is_or = tg in OR_GROUPS or ("dfine" in tg.lower() and "or" in tg.lower())
            is_or = is_or or (tg == "or" and "dfine" in src.lower())
            if not is_or and tg not in OR_GROUPS:
                # also accept markup_type=or when threshold_group hints dfine
                if not (str(props.get("markup_type") or "") == "or" and "dfine" in (tg + src).lower()):
                    if tg not in OR_GROUPS:
                        continue
        xy = _centroid_xy(feat)
        if xy is None:
            continue
        conf = props.get("confidence", props.get("score", props.get("prob")))
        try:
            conf_f = float(conf) if conf is not None else None
        except (TypeError, ValueError):
            conf_f = None
        out.append({
            "x": xy[0], "y": xy[1],
            "conf": conf_f,
            "class": props.get("class_name") or props.get("class"),
            "group": props.get("threshold_group") or props.get("group"),
            "markup": props.get("markup_type"),
        })
    return out


def _nearest_dists(preds: list[dict], gts: list[dict]) -> list[float]:
    if not gts:
        return [float("inf")] * len(preds)
    dists = []
    for p in preds:
        best = float("inf")
        for g in gts:
            d = math.hypot(p["x"] - g["x"], p["y"] - g["y"])
            if d < best:
                best = d
        dists.append(best)
    return dists


def _pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * q
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def _dump_counts(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    keys = ("n_raw_boxes", "n_features", "n_would_pass_thr", "n_det", "n_ellipses")
    out = {}
    for k in keys:
        if k in data:
            out[k] = data[k]
    totals = data.get("totals") or {}
    for k in keys:
        if k in totals and k not in out:
            out[k] = totals[k]
    return out


def analyze(pred: Path, gt: Path, dump: Path | None, off: Path | None) -> dict:
    or_preds = _load_points(pred, or_only=True)
    # If filter too strict and got 0, fall back: all features whose group contains 'or'
    if not or_preds:
        all_p = _load_points(pred, or_only=False)
        or_preds = [
            p for p in all_p
            if str(p.get("group") or "").lower() in OR_GROUPS
            or "or" in str(p.get("group") or "").lower()
            or str(p.get("markup") or "") == "or"
        ]
    gts = _load_points(gt, gt=True)
    dists = _nearest_dists(or_preds, gts)
    confs = [p["conf"] for p in or_preds if p["conf"] is not None]
    within = {r: sum(1 for d in dists if d <= r) for r in MATCH_RADII_M}
    summary = {
        "n_or": len(or_preds),
        "n_gt": len(gts),
        "score": {
            "p50": _pct(confs, 0.5),
            "p90": _pct(confs, 0.9),
            "max": max(confs) if confs else None,
            "n_scored": len(confs),
        },
        "dist_any_m": {
            "median": statistics.median(dists) if dists else None,
            "p90": _pct(dists, 0.9) if dists else None,
            "min": min(dists) if dists else None,
            "within_2": within[2.0],
            "within_50": within[50.0],
            "within_200": within[200.0],
            "sorted_sample": sorted(dists)[:20] if dists else [],
        },
        "soft_tp_at_2m": within[2.0],  # optimistic unique-agnostic count
        "dump": _dump_counts(dump),
        "bot2_a3_fyi": {
            "note": "SpOr D-FINE A3: 13 pass_thr all FP, median dist~388m, thr-sweep~0 TP; domain/model not CRS",
            "report": "reports/bot2_dfine_a3_localization.md",
        },
    }
    if off and off.is_file():
        off_n = len(_load_points(off, or_only=False))
        on_n = len(_load_points(pred, or_only=False))
        summary["counts"] = {"n_off_all": off_n, "n_on_all": on_n, "delta_n": on_n - off_n}
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, type=Path, help="ON result.geojson")
    ap.add_argument("--gt", required=True, type=Path, help="GT geojson (same CRS as pred, ideally EPSG:3857)")
    ap.add_argument("--dump", type=Path, default=None, help="optional stages/dump JSON")
    ap.add_argument("--off", type=Path, default=None, help="optional OFF result.geojson")
    ap.add_argument("--out", type=Path, required=True, help="output directory for summary.json")
    args = ap.parse_args()
    if not args.pred.is_file():
        print(f"FAIL missing pred {args.pred}", file=sys.stderr)
        return 2
    if not args.gt.is_file():
        print(f"FAIL missing gt {args.gt}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    summary = analyze(args.pred, args.gt, args.dump, args.off)
    out_path = args.out / "summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # human one-liner
    d = summary["dist_any_m"]
    print(
        f"n_or={summary['n_or']} soft_tp@2m={summary['soft_tp_at_2m']} "
        f"median_dist_m={d['median']} within_2/50/200={d['within_2']}/{d['within_50']}/{d['within_200']} "
        f"score_max={summary['score']['max']} → {out_path}"
    )
    checklist = args.out / "CHECKLIST.md"
    checklist.write_text(
        "\n".join([
            "# (c′) Or localization checklist (bot5)",
            "",
            "1. `assert_dfine_aeor_or_nonzero.py` passed (n_raw/Or dets > 0) — else HARD FAIL.",
            "2. Run this script on `out_on/result.geojson` + GT (+ dump_on if present).",
            "3. Record: n_or, soft_tp@2m, median dist_any_m, within_2/50/200.",
            "4. Compare to bot2 A3: median~388 m / 0 TP@2m → if similar, localization domain issue (not CRS); do not accept on count>0 alone.",
            "5. PROTOCOL §5 still required: ΔALL>0 and no layer drop >0.01 vs flags-OFF Docker.",
            "6. Append EXPERIMENTS.csv + update reports/bot5_v4.md.",
            "",
            f"Generated summary: `{out_path.name}`",
            "",
        ]),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
