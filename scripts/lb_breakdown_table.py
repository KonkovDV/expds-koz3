#!/usr/bin/env python3
"""Side-by-side: LB scoring.log vs offline mock vs honest-10.

    python scripts/lb_breakdown_table.py \\
      --lb-log path/to/scoring_v2_eb268d6_0.41238.log \\
      --mock-json scoring_mock.json \\
      --honest-json scoring_honest10.json \\
      --out side_by_side.md

Или из pred/gt (считает F1Calculator):

    python scripts/lb_breakdown_table.py --lb-log ... \\
      --mock-pred mock/result.geojson --mock-gt gt.geojson \\
      --honest-pred honest/result.geojson --honest-gt gt.geojson \\
      --out side_by_side.md
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _bd():
    path = Path(__file__).resolve().parent / "lb_breakdown.py"
    spec = importlib.util.spec_from_file_location("lb_breakdown", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _load_source(bd, *, json_path=None, pred=None, gt=None, metric=None, name="x"):
    if json_path:
        raw = json.loads(Path(json_path).read_text(encoding="utf-8"))
        return raw if "soft_recall" in raw else bd.extract_breakdown(raw)
    if pred and gt:
        from expds.metric import score
        res = score(
            json.loads(Path(pred).read_text(encoding="utf-8")),
            json.loads(Path(gt).read_text(encoding="utf-8")),
            metric,
        )
        return bd.extract_breakdown(res)
    raise SystemExit(f"{name}: нужен --*-json или --*-pred + --*-gt")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lb-log", required=True)
    ap.add_argument("--mock-json", default=None)
    ap.add_argument("--honest-json", default=None)
    ap.add_argument("--mock-pred", default=None)
    ap.add_argument("--mock-gt", default=None)
    ap.add_argument("--honest-pred", default=None)
    ap.add_argument("--honest-gt", default=None)
    ap.add_argument("--metric", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    bd = _bd()
    sources = {
        "LB": bd.parse_lb_log(Path(args.lb_log).read_text(encoding="utf-8", errors="replace")),
    }
    if args.mock_json or (args.mock_pred and args.mock_gt):
        sources["mock"] = _load_source(
            bd, json_path=args.mock_json, pred=args.mock_pred, gt=args.mock_gt,
            metric=args.metric, name="mock")
    if args.honest_json or (args.honest_pred and args.honest_gt):
        sources["honest-10"] = _load_source(
            bd, json_path=args.honest_json, pred=args.honest_pred, gt=args.honest_gt,
            metric=args.metric, name="honest-10")
    if len(sources) < 2:
        raise SystemExit("нужен хотя бы mock или honest-10 источник")

    table = bd.side_by_side(sources)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(table + "\n", encoding="utf-8")
    print(table)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
