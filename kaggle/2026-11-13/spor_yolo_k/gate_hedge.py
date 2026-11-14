#!/usr/bin/env python3
"""Хедж класса кургана спутника: второй класс, если рядом уже принят первый.

Их вложенный счёт: 0,3423 → 0,3620 при τ 0,4 и только при расхождении членов.
Безусловная вторая эмиссия у них уронила слияние. Здесь τ выбирается по фолдам
семей на нашем стеке, не переносится их 0,4: шкала уверенности другая.

Объект ниже пола своей группы и не ближе 8 м к уже принятому того же класса
добавляется, если в 8 м есть принятый курган другого класса. Полы eb268d6 не
двигаются. PASS: Δ ≥ +0,003 и P(Δ>0) ≥ 0,95.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
import tomllib
from collections import defaultdict
from pathlib import Path

import ijson
import numpy as np
from shapely.geometry import Point, mapping
from shapely.strtree import STRtree

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from expds.metric import threshold_group  # noqa: E402
from expds.post import _pair_ground_m  # noqa: E402

KUR = ("kurgany_povrezhdennye", "kurgany_tselye")
OTHER = {"kurgany_povrezhdennye": "kurgany_tselye", "kurgany_tselye": "kurgany_povrezhdennye"}
TAUS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60)
GROUP = "sp_hedge"


def load_gate():
    spec = importlib.util.spec_from_file_location("v3g", "/workspace/koz3/orch/bot-W4/work/v3_gate_lib.py")
    argv = sys.argv
    sys.argv = argv[:1]
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.argv = argv
    return mod


def floors():
    raw = subprocess.check_output(["git", "-C", str(REPO), "show", "eb268d6:configs/default.toml"])
    cfg = tomllib.loads(raw.decode())
    thr, own = cfg.get("thresholds") or {}, cfg.get("thresholds_by_markup") or {}
    dft = float(cfg.get("default_threshold", 0.15))

    def floor(g, c):
        return float((own.get(g) or {}).get(c, thr.get(c, dft)))

    return floor


def disk(x: float, y: float):
    r = 0.5 / max(np.cos(np.arctan(np.sinh(y / 6378137.0))), 1e-6)
    return Point(x, y).buffer(r, quad_segs=8)


def near(tree, pts, x, y, limit: float) -> bool:
    if tree is None:
        return False
    hit = tree.query(Point(x, y).buffer(40))
    return any(_pair_ground_m(Point(x, y), Point(*pts[int(j)])) < limit for j in np.atleast_1d(hit))


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("gate_hedge.json")
    t0 = time.perf_counter()
    G = load_gate()
    D, M, log, site_family = G.D, G.M, G.log, G.site_family
    floor = floors()
    acc: dict[str, dict[str, list]] = defaultdict(lambda: {c: [] for c in KUR})
    low: dict[str, dict[str, list]] = defaultdict(lambda: {c: [] for c in KUR})
    with Path("/workspace/koz3/w5data/stack_arch.geojson").open("rb") as fh:
        for feat in ijson.items(fh, "features.item"):
            q = feat.get("properties") or {}
            cls = q.get("class_name")
            if q.get("markup_type") != "sp_or" or cls not in OTHER:
                continue
            g = feat.get("geometry") or {}
            ring = (g.get("coordinates") or [None])[0] if g.get("type") == "Polygon" else None
            if not ring:
                continue
            x = float(np.mean([p[0] for p in ring]))
            y = float(np.mean([p[1] for p in ring]))
            conf = float(q.get("confidence") or 0)
            reg = q.get("region_name")
            fl = floor(threshold_group(q), cls)
            (acc if conf >= fl else low)[reg][cls].append((x, y, conf))
    log(f"участков спутника {len(acc)}")

    def candidates(tau: float) -> list[dict]:
        feats = []
        for reg, by in low.items():
            trees = {}
            for cls, pts in acc[reg].items():
                if pts:
                    trees[cls] = (STRtree([Point(x, y) for x, y, _ in pts]), [(x, y) for x, y, _ in pts])
            for cls, pts in by.items():
                other = trees.get(OTHER[cls])
                same = trees.get(cls)
                if other is None:
                    continue
                for x, y, conf in pts:
                    if conf < tau or near(same[0] if same else None, same[1] if same else [], x, y, 8):
                        continue
                    if not near(other[0], other[1], x, y, 8):
                        continue
                    geom = disk(x, y)
                    feats.append({"type": "Feature", "geometry": mapping(geom),
                                  "properties": {"class_name": cls, "confidence": round(conf * 0.99, 5),
                                                 "region_name": reg, "markup_type": "sp_or",
                                                 "threshold_group": GROUP, "crs": "EPSG:3857"}})
        return feats

    probe = {tau: len(candidates(tau)) for tau in TAUS}
    log(f"кандидаты по τ: {probe}")

    stk = D.Stack(Path("/workspace/koz3/w5data/stack_arch.geojson"),
                  Path("/workspace/koz3/w5data/gt_train_v2.geojson"),
                  Path("/workspace/koz3/w1/d3/split_arch"), Path("/workspace/koz3/w1/d3/cache"))
    tab = D.TableEngine(Path("/workspace/koz3/w5data/work/table.pkl"), stk)

    def filt(tab):
        import copy
        t2 = copy.copy(tab)
        t2.base = {}
        for c, b in tab.base.items():
            keep = np.array([float(cf) >= floor(g, c) for g, cf in zip(b["grp"], b["conf"])], bool)
            idx = np.flatnonzero(keep)
            new = {int(o): n for n, o in enumerate(idx)}
            t2.base[c] = dict(conf=b["conf"][idx], reg=b["reg"][idx], grp=b["grp"][idx],
                              pairs={new[i]: v for i, v in b["pairs"].items() if i in new})
        return t2

    base_tab = filt(tab)
    base_curves, _ = base_tab.curves()
    regs = {r for c in base_curves.values() for r in c.pred_region} | {r for c in base_curves.values() for r in c.gt_regions}
    folds = M.family_folds(regs, site_family, 10)
    groups = {g for c in base_curves.values() for g in c.pred_group if g}
    fixed = frozenset(groups | {GROUP})
    base_counts = D.honest(base_curves, folds, fixed=fixed)[0]
    del base_curves

    per_tau = {}
    built = {}
    for tau in TAUS:
        rows = candidates(tau)
        built[tau] = len(rows)
        curves, _ = base_tab.curves(rows)
        per_tau[tau] = D.honest(curves, folds, fixed=fixed)[0]
        log(f"τ {tau}: {len(rows)}")

    def f1_of(counts, cls, regions):
        tp = fp = ng = 0.0
        for r, (a, b, g) in counts[cls].items():
            if r in regions:
                tp += a
                fp += b
                ng += g
        return M._f1(tp, fp, ng - tp)

    every = set().union(*folds)
    nested = {c: {} for c in base_counts}
    chosen = []
    for held in folds:
        train = every - held
        best_tau, best_f = TAUS[0], -1.0
        for tau, counts in per_tau.items():
            f = sum(f1_of(counts, c, train) for c in KUR)
            if f > best_f:
                best_f, best_tau = f, tau
        chosen.append(best_tau)
        use = per_tau[best_tau]
        for cls, counts0 in base_counts.items():
            for r in held:
                src = use.get(cls, {})
                nested[cls][r] = list(src[r] if r in src else counts0.get(r, [0, 0, 0]))

    boot = M.paired_bootstrap_counts(nested, base_counts, site_family, 2000, 0)
    sa = M.score_counts(base_counts)[0]
    sb = M.score_counts(nested)[0]
    res = {"what": "хедж второго класса кургана спутника при расхождении, τ по фолдам семей",
           "probe": {str(k): v for k, v in probe.items()}, "built": {str(k): v for k, v in built.items()},
           "fold_taus": chosen, "base": sa, "new": sb, "delta": sb - sa,
           "P": boot.get("p_gt0"), "p5": boot.get("p5"), "p95": boot.get("p95"),
           "PASS": (sb - sa) >= 0.003 and (boot.get("p_gt0") or 0) >= 0.95,
           "sec": round(time.perf_counter() - t0)}
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"Δ {res['delta']:+.6f} P {res['P']} PASS {res['PASS']} за {res['sec']} с")


if __name__ == "__main__":
    main()
