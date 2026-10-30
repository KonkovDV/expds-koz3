#!/usr/bin/env python3
"""Сравнение global vs layer matching: TP/FP/FN по классу×слою и список cross-layer пар.

Диагностика Δ honest global↔layer. Не меняет сдачу (только читает pred/gt).

Примеры:

    python scripts/diff_match_scope.py --pred stack.geojson --gt gt.geojson --out notes/diff_scope/

    # стек как у scoreboard (нужны --work/--train):
    python scripts/diff_match_scope.py --work F:/expds3/work --train F:/2/train-expds-3 \\
        --honest 10 --out notes/bot7_diff_scope/
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _counts(curves, tau: float = 0.0):
    out = {}
    for c, cv in curves.items():
        k = cv.conf >= tau
        tp = int(cv.hit[k].sum())
        out[c] = {"tp": tp, "fp": int(k.sum()) - tp, "fn": int(cv.n_gt - tp), "n_gt": int(cv.n_gt)}
    return out


def _layer_counts_from_pairs(pairs, curves):
    """TP по (class, pred_layer, gt_layer); FN/FP грубо не из пар — см. summary_by_class."""
    by = Counter()
    for p in pairs:
        by[(p["class"], p["pred_layer"], p["gt_layer"])] += 1
    return by


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", default=None, help="FeatureCollection предсказаний (или стек scoreboard)")
    ap.add_argument("--gt", default=None, help="FeatureCollection эталона")
    ap.add_argument("--work", default=None, help="как scoreboard: собрать стек из work/")
    ap.add_argument("--train", default=None, help="корень трейна для li_coarse (с --work)")
    ap.add_argument("--geo", default=None)
    ap.add_argument("--metric", default=str(ROOT / "materials" / "compute_metrics.py"))
    ap.add_argument("--out", required=True, help="каталог для CSV/JSON")
    ap.add_argument("--tau", type=float, default=0.0, help="порог conf для счётчиков (0 = все)")
    ap.add_argument("--honest", type=int, default=0,
                    help="как scoreboard: K фолдов семей для честного ALL (0=выкл); нужен --work")
    ap.add_argument("--honest-by", choices=("family", "region"), default="family")
    ap.add_argument("--honest-unseen", choices=("drop", "keep"), default="drop")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from expds import metric
    from expds.metric import build_curves_with_pairs, layer_key, score_at

    if args.work:
        if not args.train:
            raise SystemExit("--work требует --train")
        # ленивый импорт scoreboard
        sys.path.insert(0, str(ROOT / "scripts"))
        import scoreboard as sb
        work = Path(args.work)
        geo = Path(args.geo) if args.geo else work / "geo_cur.geojson"
        gt_path = Path(args.gt) if args.gt else work / "gt_train_v2.geojson"
        feats, gt, new, optical = sb.build_stack(work, Path(args.train), geo, gt_path)
        pred = {"type": "FeatureCollection", "features": feats}
        (out / "stack_dump.geojson").write_text(json.dumps(pred, ensure_ascii=False), encoding="utf-8")
    else:
        if not args.pred or not args.gt:
            raise SystemExit("нужны --pred/--gt или --work/--train")
        if args.honest:
            raise SystemExit("--honest требует --work/--train (нужны new/optical регионы)")
        pred = json.loads(Path(args.pred).read_text(encoding="utf-8"))
        gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))
        new, optical = set(), set()

    metric._UTM_GEOM_CACHE.clear()
    metric._SCOPE_WARNED.clear()
    g_curves, g_pairs = build_curves_with_pairs(pred, gt, args.metric, scope="global")
    metric._UTM_GEOM_CACHE.clear()
    metric._SCOPE_WARNED.clear()
    l_curves, l_pairs = build_curves_with_pairs(pred, gt, args.metric, scope="layer")

    g_score, _ = score_at(g_curves, {})
    l_score, _ = score_at(l_curves, {})
    g_cnt = _counts(g_curves, args.tau)
    l_cnt = _counts(l_curves, args.tau)

    cross = [p for p in g_pairs if p["cross_layer"]]
    same = [p for p in g_pairs if not p["cross_layer"]]

    honest_global = honest_layer = None
    if args.honest:
        from expds.metric import family_folds, honest_counts, score_counts
        from expds.naming import site_family

        def _honest(curves):
            unit_of = site_family if args.honest_by == "family" else (lambda r: r)
            regions = ({r for c in curves.values() for r in c.pred_region}
                       | {r for c in curves.values() for r in c.gt_regions})
            folds = family_folds(regions, unit_of, None if args.honest < 0 else args.honest)
            counts, info = honest_counts(curves, folds, unseen=args.honest_unseen)
            return {
                "all": score_counts(counts)[0],
                "new34": score_counts(counts, new)[0],
                "optical": score_counts(counts, optical)[0],
                "folds": info["folds"],
                "unseen_dropped": info["unseen_dropped"],
                "unseen_kept": info["unseen_kept"],
            }

        honest_global = _honest(g_curves)
        honest_layer = _honest(l_curves)
        print(
            f"honest-{args.honest} global={honest_global['all']:.6f} "
            f"layer={honest_layer['all']:.6f} "
            f"Δ={honest_layer['all'] - honest_global['all']:+.6f} "
            f"(34уч {honest_global['new34']:.6f}→{honest_layer['new34']:.6f}; "
            f"ae/opt {honest_global['optical']:.6f}→{honest_layer['optical']:.6f})",
            flush=True,
        )

    # summary JSON
    summary = {
        "tau": args.tau,
        "score_global": g_score,
        "score_layer": l_score,
        "delta": l_score - g_score,
        "honest_k": args.honest or None,
        "honest_global": honest_global,
        "honest_layer": honest_layer,
        "honest_delta": (None if honest_global is None
                         else honest_layer["all"] - honest_global["all"]),
        "n_tp_global": len(g_pairs),
        "n_tp_layer": len(l_pairs),
        "n_cross_layer_tp_global": len(cross),
        "n_same_layer_tp_global": len(same),
        "per_class_global": g_cnt,
        "per_class_layer": l_cnt,
        "cross_by_pred_gt_layer": [
            {"pred_layer": a, "gt_layer": b, "n": n}
            for (a, b), n in sorted(Counter((p["pred_layer"], p["gt_layer"]) for p in cross).items(),
                                    key=lambda x: -x[1])
        ],
        "cross_by_class": [
            {"class": c, "n": n}
            for c, n in sorted(Counter(p["class"] for p in cross).items(), key=lambda x: -x[1])
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # cross matches CSV
    with (out / "cross_matches.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["class", "pred_layer", "gt_layer", "conf", "quality",
                                          "pred_region", "gt_region"])
        w.writeheader()
        for p in sorted(cross, key=lambda x: (-x["quality"], x["class"])):
            w.writerow({k: p[k] for k in w.fieldnames})

    # per class × layer matrix from global pairs (TP attribution)
    with (out / "tp_by_class_layers.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "pred_layer", "gt_layer", "n_tp_global", "cross"])
        for (cls, pl, gl), n in sorted(Counter((p["class"], p["pred_layer"], p["gt_layer"])
                                               for p in g_pairs).items()):
            w.writerow([cls, pl, gl, n, int(pl != gl)])

    # class delta TP/FP/FN
    with (out / "delta_by_class.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "tp_g", "fp_g", "fn_g", "tp_l", "fp_l", "fn_l",
                    "d_tp", "d_fp", "d_fn"])
        for cls in sorted(set(g_cnt) | set(l_cnt)):
            a, b = g_cnt.get(cls, {"tp": 0, "fp": 0, "fn": 0}), l_cnt.get(cls, {"tp": 0, "fp": 0, "fn": 0})
            w.writerow([cls, a["tp"], a["fp"], a["fn"], b["tp"], b["fp"], b["fn"],
                        b["tp"] - a["tp"], b["fp"] - a["fp"], b["fn"] - a["fn"]])

    print(f"score global={g_score:.6f} layer={l_score:.6f} Δ={l_score - g_score:+.6f}", flush=True)
    print(f"TP global={len(g_pairs)} (cross={len(cross)} same={len(same)}) layer={len(l_pairs)}", flush=True)
    print(f"wrote {out}/summary.json cross_matches.csv tp_by_class_layers.csv delta_by_class.csv", flush=True)
    if cross:
        print("top cross pred→gt:", summary["cross_by_pred_gt_layer"][:8], flush=True)
        print("top cross class:", summary["cross_by_class"][:8], flush=True)


if __name__ == "__main__":
    main()
