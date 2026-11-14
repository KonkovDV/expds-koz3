#!/usr/bin/env python3
"""Согласие двух карт спутника: объект ниже пола выпускается, если вторая карта видит тот же класс в 16 м.

Это [sat_agree_accept] в expds/sat_recall_levers.py, по умолчанию выключенный.
Числа кода, не подобранные здесь: scale 0,85, радиус 16 м, группы sp_focal и sp_focal2.
Пол sp_focal — 0,364475 / 0,372525, пол sp_focal2 — 0,501645 / 0,645505 (архив eb268d6).
PASS: Δ ≥ +0,003 и P(Δ>0) ≥ 0,95.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import ijson
import numpy as np
from shapely.geometry import Point, mapping
from shapely.strtree import STRtree

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from expds.post import _pair_ground_m  # noqa: E402

KUR = ("kurgany_povrezhdennye", "kurgany_tselye")
FLOOR = {
    "sp_focal": {"kurgany_povrezhdennye": 0.364475, "kurgany_tselye": 0.372525},
    "sp_focal2": {"kurgany_povrezhdennye": 0.501645, "kurgany_tselye": 0.645505},
}
SCALE = 0.85
RADIUS = 16.0
GROUP = "sp_agree"


def load_gate():
    spec = importlib.util.spec_from_file_location("v3g", "/workspace/koz3/orch/bot-W4/work/v3_gate_lib.py")
    argv = sys.argv
    sys.argv = argv[:1]
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.argv = argv
    return mod


def grp(q: dict) -> str:
    g = q.get("threshold_group") or q.get("markup_type") or ""
    return "sp_focal" if g == "sp_or" else g


def disk(x: float, y: float):
    r = 0.5 / max(float(np.cos(np.arctan(np.sinh(y / 6378137.0)))), 1e-6)
    return Point(x, y).buffer(r, quad_segs=8)


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("gate_agree.json")
    t0 = time.perf_counter()
    G = load_gate()
    D, M, log, site_family = G.D, G.M, G.log, G.site_family
    by_reg: dict = defaultdict(lambda: defaultdict(list))
    with Path("/workspace/koz3/w5data/stack_arch.geojson").open("rb") as fh:
        for feat in ijson.items(fh, "features.item"):
            q = feat.get("properties") or {}
            cls = q.get("class_name")
            g = grp(q)
            if q.get("markup_type") != "sp_or" or cls not in FLOOR.get(g, {}):
                continue
            ring = ((feat.get("geometry") or {}).get("coordinates") or [None])[0]
            if not ring:
                continue
            x = float(np.mean([p[0] for p in ring]))
            y = float(np.mean([p[1] for p in ring]))
            by_reg[q.get("region_name")][(g, cls)].append((x, y, float(q.get("confidence") or 0)))

    extras = []
    for reg, buckets in by_reg.items():
        trees = {}
        for key, pts in buckets.items():
            if pts:
                geoms = [Point(x, y) for x, y, _ in pts]
                trees[key] = (STRtree(geoms), geoms)
        for (g, cls), pts in buckets.items():
            other = "sp_focal2" if g == "sp_focal" else "sp_focal"
            packed = trees.get((other, cls))
            if packed is None:
                continue
            tree, geoms = packed
            thr = FLOOR[g][cls]
            low = thr * SCALE
            for x, y, conf in pts:
                if conf < low or conf >= thr:
                    continue
                hit = tree.query(Point(x, y).buffer(RADIUS / 0.5))
                if not any(_pair_ground_m(Point(x, y), geoms[int(j)]) < RADIUS for j in np.atleast_1d(hit)):
                    continue
                extras.append({"type": "Feature", "geometry": mapping(disk(x, y)),
                               "properties": {"class_name": cls, "confidence": round(conf, 5),
                                              "region_name": reg, "markup_type": "sp_or",
                                              "threshold_group": GROUP, "crs": "EPSG:3857"}})
    log(f"согласованных ниже пола: {len(extras)}")

    def floor_of(g, c):
        if g in FLOOR and c in FLOOR[g]:
            return FLOOR[g][c]
        if g == "sp_focal" and c in FLOOR["sp_focal"]:
            return FLOOR["sp_focal"][c]
        return 0.15

    stk = D.Stack(Path("/workspace/koz3/w5data/stack_arch.geojson"),
                  Path("/workspace/koz3/w5data/gt_train_v2.geojson"),
                  Path("/workspace/koz3/w1/d3/split_arch"), Path("/workspace/koz3/w1/d3/cache"))
    tab = D.TableEngine(Path("/workspace/koz3/w5data/work/table.pkl"), stk)
    import copy
    base = copy.copy(tab)
    base.base = {}
    for c, b in tab.base.items():
        keep = np.array([float(cf) >= floor_of(g, c) for g, cf in zip(b["grp"], b["conf"])], bool)
        idx = np.flatnonzero(keep)
        new = {int(o): n for n, o in enumerate(idx)}
        base.base[c] = dict(conf=b["conf"][idx], reg=b["reg"][idx], grp=b["grp"][idx],
                            pairs={new[i]: v for i, v in b["pairs"].items() if i in new})
    curves, _ = base.curves()
    regs = {r for c in curves.values() for r in c.pred_region} | {r for c in curves.values() for r in c.gt_regions}
    folds = M.family_folds(regs, site_family, 10)
    groups = {g for c in curves.values() for g in c.pred_group if g} | {GROUP}
    fixed = frozenset(groups)
    base_counts = D.honest(curves, folds, fixed=fixed)[0]
    del curves
    add_curves, _ = base.curves(extras)
    add_counts = D.honest(add_curves, folds, fixed=fixed)[0]
    boot = M.paired_bootstrap_counts(add_counts, base_counts, site_family, 2000, 0)
    sa, pa = M.score_counts(base_counts)
    sb, pb = M.score_counts(add_counts)
    res = {"what": "sat_agree_accept scale 0.85 radius 16 м, полы архива",
           "n": len(extras), "base": sa, "new": sb, "delta": sb - sa,
           "P": boot.get("p_gt0"), "p5": boot.get("p5"), "p95": boot.get("p95"),
           "classes": {c: {"d_f1": pb[c]["f1"] - pa[c]["f1"], "tp": [pa[c]["tp"], pb[c]["tp"]],
                           "fp": [pa[c]["fp"], pb[c]["fp"]]} for c in KUR if c in pa},
           "PASS": (sb - sa) >= 0.003 and (boot.get("p_gt0") or 0) >= 0.95,
           "sec": round(time.perf_counter() - t0)}
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"Δ {res['delta']:+.6f} P {res['P']} n {res['n']} PASS {res['PASS']} за {res['sec']} с")


if __name__ == "__main__":
    main()
