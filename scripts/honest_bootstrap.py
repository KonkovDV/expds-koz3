#!/usr/bin/env python3
"""Шум честного табло: бутстреп по семьям, парные разности, MDE, чувствительность
к раскладке фолдов и симуляция публичной площадки (48 участков).

Вход — каталоги посемейных кривых (scripts/family_curves.py): база A и варианты.
Метрика — только официальный compute_metrics.py (через expds.metric).

    python scripts/honest_bootstrap.py --a work/curves --b nogor=work/curves_nogor \
        --b noarch=work/curves_noarch --out noise_measured.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from expds import metric as M  # noqa: E402
from expds.constants import CLASS_WEIGHTS  # noqa: E402
from expds.naming import site_family  # noqa: E402

HEAVY = ("kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha")


def load_curves(folder: Path) -> dict[str, M.ClassCurve]:
    parts: dict[str, list] = {}
    for p in sorted(folder.glob("*.pkl")):
        for cls, c in pickle.loads(p.read_bytes()).items():
            parts.setdefault(cls, []).append(c)
    out = {}
    for cls, cs in parts.items():
        conf = np.concatenate([c.conf for c in cs])
        hit = np.concatenate([c.hit for c in cs])
        reg = [r for c in cs for r in c.pred_region]
        grp = [g for c in cs for g in c.pred_group]
        order = np.argsort(-conf, kind="stable")
        gt = Counter()
        for c in cs:
            gt.update(c.gt_regions)
        out[cls] = M.ClassCurve(conf[order], hit[order], [reg[i] for i in order], int(sum(gt.values())), gt,
                                [grp[i] for i in order])
    return out


def regions_of(curves) -> set[str]:
    return {r for c in curves.values() for r in c.pred_region} | {r for c in curves.values() for r in c.gt_regions}


def insample_counts(curves):
    """Пороги на всех участках (как «все» в табло, быстрая оценка без verify_groups)."""
    every = regions_of(curves)
    res = M.sweep_groups(curves)
    taus: dict = {}
    for cls, v in res.items():
        for g, gv in v["groups"].items():
            taus.setdefault(g, {})[cls] = gv["threshold"]
    counts = {cls: {} for cls in curves}
    for cls, c in curves.items():
        per = counts[cls]
        for r, n in c.gt_regions.items():
            per.setdefault(r, [0, 0, 0])[2] += int(n)
        tau = np.array([taus.get(g, {}).get(cls, np.inf) for g in c.pred_group], dtype=float)
        kept = c.conf >= tau
        for r, h in zip(np.asarray(c.pred_region, dtype=object)[kept], c.hit[kept]):
            per.setdefault(r, [0, 0, 0])[0 if h else 1] += 1
    assert every
    return counts


def random_family_folds(regions, k: int, seed: int) -> list[set[str]]:
    fams: dict[str, set[str]] = {}
    for r in regions:
        fams.setdefault(site_family(r), set()).add(r)
    names = sorted(fams)
    rng = np.random.default_rng(seed)
    lab = rng.permutation(len(names)) % k
    return [set().union(*(fams[n] for n, l in zip(names, lab) if l == i)) for i in range(k)]


def stats(v) -> dict:
    v = np.asarray(v, dtype=float)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "p2_5": float(np.percentile(v, 2.5)),
            "p5": float(np.percentile(v, 5)), "p50": float(np.median(v)), "p95": float(np.percentile(v, 95)),
            "p97_5": float(np.percentile(v, 97.5)), "p_gt0": float((v > 0).mean()), "n": int(v.size)}


def fam_index(counts_list):
    regs = sorted({r for cs in counts_list for by in cs.values() for r in by})
    fams: dict[str, list[str]] = {}
    for r in regs:
        fams.setdefault(site_family(r), []).append(r)
    return fams


def boot(counts_list, n: int, seed: int, fams) -> list[np.ndarray]:
    """Бутстреп по семьям: балл каждого набора счётчиков на одних и тех же выборках."""
    names = sorted(fams)
    rng = np.random.default_rng(seed)
    res = [[] for _ in counts_list]
    for _ in range(n):
        pick = np.bincount(rng.integers(0, len(names), len(names)), minlength=len(names))
        w = {r: float(k) for f, k in zip(names, pick) if k for r in fams[f]}
        for i, cs in enumerate(counts_list):
            res[i].append(M.score_counts(cs, None, w)[0])
    return [np.asarray(x) for x in res]


def weighted_class_delta(a, b) -> dict:
    _, pa = M.score_counts(a)
    _, pb = M.score_counts(b)
    den = sum(CLASS_WEIGHTS.get(c, 0.0) for c in pa if pa[c]["n_gt"] > 0)
    return {c: {"dF1": pa[c]["f1"] - pb[c]["f1"],
                "weighted": CLASS_WEIGHTS.get(c, 0.0) * (pa[c]["f1"] - pb[c]["f1"]) / den}
            for c in pa if pa[c]["n_gt"] > 0}


def lb_sim(counts_list, n: int, size: int, seed: int, fams) -> dict:
    """Публичная площадка как подвыборка: (1) 48 случайных участков, (2) целые случайные семьи до ≥48 участков."""
    rng = np.random.default_rng(seed)
    regs = sorted(r for f in fams.values() for r in f)
    names = sorted(fams)
    out = {}
    for mode in ("regions", "families"):
        vals = [[] for _ in counts_list]
        for _ in range(n):
            if mode == "regions":
                sub = set(rng.choice(regs, size, replace=False).tolist())
            else:
                sub: set[str] = set()
                for f in rng.permutation(names):
                    if len(sub) >= size:
                        break
                    sub |= set(fams[f])
            for i, cs in enumerate(counts_list):
                vals[i].append(M.score_counts(cs, sub)[0])
        out[mode] = [np.asarray(v) for v in vals]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", action="append", default=[], help="имя=каталог кривых варианта")
    ap.add_argument("--new", default=str(ROOT / "notes" / "regions_las_new.txt"))
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--lb", type=int, default=2000)
    ap.add_argument("--lb-size", type=int, default=48)
    ap.add_argument("--seeds", type=int, default=5, help="случайных раскладок семей на 10 фолдов")
    ap.add_argument("--unseen", choices=("drop", "keep"), default="drop",
                    help="группа без порога на обучающих фолдах: drop — выбросить, keep — оставить целиком "
                         "(как сдача: etgr/terfl object_poly есть только в Гнёздово, в сдаче их пороги 0,096635/0,0)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    new = {ln.strip() for ln in Path(a.new).read_text(encoding="utf-8").splitlines() if ln.strip()}
    variants = {"A": load_curves(Path(a.a))}
    for spec in a.b:
        name, path = spec.split("=", 1)
        variants[name] = load_curves(Path(path))
    names = list(variants)
    regions = set().union(*(regions_of(c) for c in variants.values()))
    rep: dict = {"regions": len(regions), "families": len({site_family(r) for r in regions}), "variants": {}}
    canon, lofo = {}, {}
    folds10 = M.family_folds(regions, site_family, 10)
    for nm, cv in variants.items():
        ins = insample_counts(cv)
        c10, info10 = M.honest_counts(cv, folds10, unseen=a.unseen)
        canon[nm] = c10
        v = {"preds": int(sum(c.conf.size for c in cv.values())),
             "insample_all": M.score_counts(ins)[0], "insample_new34": M.score_counts(ins, new)[0],
             "honest10_all": M.score_counts(c10)[0], "honest10_new34": M.score_counts(c10, new)[0],
             "honest10_unseen_dropped": info10["unseen_dropped"], "honest10_unseen_kept": info10["unseen_kept"]}
        for k in (5, 20, -1):
            ck, ik = M.honest_counts(cv, M.family_folds(regions, site_family, None if k < 0 else k), unseen=a.unseen)
            if k < 0:
                lofo[nm] = ck
            v[f"honest{'LOFO' if k < 0 else k}_all"] = M.score_counts(ck)[0]
        v["honest10_random_seeds"] = []
        for s in range(a.seeds):
            cs, _ = M.honest_counts(cv, random_family_folds(regions, 10, s), unseen=a.unseen)
            v["honest10_random_seeds"].append(M.score_counts(cs)[0])
        v["per_class_honest10"] = {c: x for c, x in M.score_counts(c10)[1].items() if x["n_gt"] > 0}
        rep["variants"][nm] = v
        print(nm, {k: x for k, x in v.items() if not isinstance(x, (dict, list))}, f"{time.time() - t0:.0f}s",
              flush=True)
    fams = fam_index(list(canon.values()))
    lv = boot([canon[n] for n in names], a.boot, 0, fams)
    rep["level_bootstrap_honest10"] = {n: stats(x) for n, x in zip(names, lv)}
    rep["paired"] = {}
    sims = lb_sim([canon[n] for n in names], a.lb, a.lb_size, 1, fams)
    for i, nm in enumerate(names[1:], 1):
        d = lv[0] - lv[i]
        point = rep["variants"]["A"]["honest10_all"] - rep["variants"][nm]["honest10_all"]
        pr = {"point_honest10": point, "bootstrap_families": stats(d),
              "point_lofo": M.score_counts(lofo["A"])[0] - M.score_counts(lofo[nm])[0],
              "seed_deltas": [x - y for x, y in zip(rep["variants"]["A"]["honest10_random_seeds"],
                                                    rep["variants"][nm]["honest10_random_seeds"])],
              "per_class_weighted": weighted_class_delta(canon["A"], canon[nm]),
              "builtin_paired_bootstrap_counts": M.paired_bootstrap_counts(canon["A"], canon[nm], site_family, 1000)}
        se = float(np.std(d, ddof=1))
        pr["se"] = se
        pr["mde_one_sided_a0.10_power0.80"] = (1.2816 + 0.8416) * se
        pr["mde_two_sided_a0.05_power0.80"] = (1.96 + 0.8416) * se
        for mode, vals in sims.items():
            dd = vals[0] - vals[i]
            pr[f"lb_sim_{mode}"] = {"delta": stats(dd), "p_sign_matches_point": float(((dd > 0) == (point > 0)).mean())}
        rep["paired"][f"A-{nm}"] = pr
    rep["lb_sim_level_A"] = {m: stats(v[0]) for m, v in sims.items()}
    rep["seconds"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "variants"}, ensure_ascii=False, indent=1)[:6000])


if __name__ == "__main__":
    main()
