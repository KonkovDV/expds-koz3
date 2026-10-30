#!/usr/bin/env python3
"""Задача 16 (bot-2): recall по размерам GT на стеке A (honest-10) + диагностика потерь крупных объектов.

Повторяет expds.metric.match_collected (официальное жадное сопоставление по классам, сценарий А)
по семьям (как scripts/family_curves.py, с кластерами), но запоминает, какой эталон взял каждый
прогноз. Пороги — честные honest-10 (sweep_groups на остальных фолдах, unseen=drop) по кривым
bot-10/work/curves. Эталон «найден», если его прогноз проходит честный порог своего фолда.
Для пропущенных эталонов площадью > --diag-min м² — причина (см. classify()).
Только чтение входов; пишет CSV/JSON в --out.
"""
from __future__ import annotations
import argparse, csv, gc, json, pickle, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

B10 = Path("/workspace/koz3_shared/bot-10")
sys.path.insert(0, str(B10 / "repo")); sys.path.insert(0, str(B10 / "repo/scripts"))
from expds import metric as M  # noqa
from expds.naming import site_family  # noqa
from honest_bootstrap import load_curves, regions_of  # noqa
M._SHM_CACHE_DIR = Path("/proc/expds_no_cache")
BINS = [100.0, 500.0, 1000.0, 5000.0, 10000.0]
BIN_NAMES = ["<=100", "100-500", "500-1k", "1k-5k", "5k-10k", ">10k"]
DTYPE = {"la": "лидар", "li": "лидар", "ae": "аэрофото", "german_ae": "аэрофото", "sp_or": "спутник",
         "spor": "спутник", "or": "ортофото", "maggr": "геофизика", "etgr": "геофизика", "geogr": "геофизика",
         "terfl": "геофизика"}


def sbin(a):
    for i, u in enumerate(BINS):
        if a <= u:
            return BIN_NAMES[i]
    return BIN_NAMES[-1]


def collect(mod, calc, feats, is_gt, batch=20000):
    out, buf = {}, []
    def flush():
        if buf:
            for cls, lst in M._collect(mod, calc, {"type": "FeatureCollection", "features": list(buf)}, is_gt).items():
                out.setdefault(cls, []).extend(lst)
            buf.clear(); M._UTM_GEOM_CACHE.clear()
    for f in feats:
        buf.append(f)
        if len(buf) >= batch:
            flush()
    flush()
    return out


def honest_taus(curves, folds):
    """{регион: {(группа, класс): tau}} — как M.honest_counts (pick=max)."""
    every = set().union(*folds)
    by_reg = {}
    for held in folds:
        res = M.sweep_groups({c: M.curve_where(cv, every - held) for c, cv in curves.items()})
        taus = {(g, cls): gv["threshold"] for cls, v in res.items() for g, gv in v["groups"].items()}
        for r in held:
            by_reg[r] = taus
    return by_reg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fam", default=str(B10 / "work11/fam0"))
    ap.add_argument("--curves", default=str(B10 / "work/curves"))
    ap.add_argument("--gt", default="/workspace/rerank/gt_train_v2.geojson")
    ap.add_argument("--clusters", default=str(B10 / "work/family_clusters.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--diag-min", type=float, default=1000.0)
    ap.add_argument("--only", default="", help="семьи через запятую (отладка)")
    ap.add_argument("--code", default="", help="ветка bot-2/bigobj для --variant (var_curves.make_transform)")
    ap.add_argument("--variant", default="", help="JSON варианта F1/F2 (как var_curves.py); пусто — стек A")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    mod = M.load_official(None); calc = mod.F1Calculator()
    curves = load_curves(Path(a.curves))
    regions = regions_of(curves)
    folds = M.family_folds(regions, site_family, 10)
    taus = honest_taus(curves, folds)
    del curves; gc.collect()
    gt_by_fam = defaultdict(list)
    for i, f in enumerate(json.loads(Path(a.gt).read_text(encoding="utf-8"))["features"]):
        p = dict(f.get("properties") or {})
        p["threshold_group"] = f"{p.get('markup_type','')}|{i}"   # тип данных + индекс эталона
        gt_by_fam[site_family(str(p.get("region_name", "default_source")))].append(
            {"type": "Feature", "geometry": f["geometry"], "properties": p})
    fams = sorted(set(gt_by_fam) | {p.stem for p in Path(a.fam).glob("*.jsonl")})
    units = {f: [f] for f in fams}
    for members in json.loads(Path(a.clusters).read_text(encoding="utf-8"))["clusters"].values():
        for m in members:
            units.pop(m, None)
        units["+".join(sorted(members))] = sorted(members)
    only = {x for x in a.only.split(",") if x}
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import var_curves as V
    tf = V.make_transform(Path(a.code), json.loads(a.variant)) if a.variant else None
    rows, tpfp = [], defaultdict(lambda: [0, 0, 0])   # (cls, region) -> tp, fp, ngt
    from shapely.strtree import STRtree
    for unit, members in sorted(units.items()):
        if only and unit not in only:
            continue
        t0 = time.time()
        preds = collect(mod, calc, V.iter_unit(Path(a.fam), members, tf), False)
        gts = collect(mod, calc, iter([f for m in members for f in gt_by_fam.get(m, [])]), True)
        trees = {}
        for cls in calc.target_classes:
            p, g = preds.get(cls, []), gts.get(cls, [])
            order = sorted(range(len(p)), key=lambda i: p[i][1], reverse=True)
            taken = {}   # gt j -> rank
            if p and g:
                gg = [x[0] for x in g]
                tree = STRtree([x.buffer(calc.r_tolerance) if x.geom_type == "Point" else x for x in gg])
                pg_ord = [p[i][0] for i in order]
                q_pred, q_gt = tree.query(pg_ord)
                if len(q_pred):
                    splits = np.flatnonzero(np.diff(q_pred)) + 1
                    for rank, cand in zip(q_pred[np.r_[0, splits]], np.split(q_gt, splits)):
                        r = int(rank); pg = pg_ord[r]
                        if pg.geom_type != "Polygon":
                            continue
                        best, bj = float("-inf"), -1
                        for j in sorted(int(k) for k in cand):
                            if j in taken:
                                continue
                            ok, s = calc._match_quality(pg, gg[j])
                            if ok and s > best + 1e-9:
                                best, bj = s, j
                        if bj >= 0:
                            taken[bj] = r
            # TP/FP при честных порогах (для сверки с honest-10 A)
            hit_rank = set(taken.values())
            for r, i in enumerate(order):
                _, conf, reg, grp = p[i]
                t = taus.get(reg, {}).get((grp, cls))
                if t is not None and conf >= t:
                    tpfp[(cls, reg)][0 if r in hit_rank else 1] += 1
            for x in g:
                tpfp[(cls, x[2])][2] += 1
            if p:
                trees[cls] = (STRtree([x[0] for x in p]), p)
            for j, (geom, _, reg, grp) in enumerate(g):
                markup, gidx = grp.split("|")
                area = geom.area if geom.geom_type != "Point" else 0.0
                row = {"unit": unit, "region": reg, "cls": cls, "markup": markup, "dtype": DTYPE.get(markup, "прочее"),
                       "gt_idx": int(gidx), "area": round(area, 1), "bin": sbin(area) if area > 0 else "point",
                       "matched_any": 0, "kept": 0, "m_conf": "", "m_grp": "", "m_tau": "", "m_area": ""}
                if j in taken:
                    _, conf, preg, pgrp = p[order[taken[j]]]
                    t = taus.get(preg, {}).get((pgrp, cls))
                    row.update(matched_any=1, m_conf=round(conf, 5), m_grp=pgrp,
                               m_tau="unseen" if t is None else t, m_area=round(p[order[taken[j]]][0].area, 1),
                               kept=int(t is not None and conf >= t))
                row["_geom"] = geom
                rows.append(row)
        # диагностика пропущенных крупных
        for row in rows:
            if row.get("_geom") is None or row["unit"] != unit:
                continue
            geom = row.pop("_geom")
            if row["kept"] or row["area"] <= a.diag_min:
                continue
            d = diag(calc, geom, row, trees, taus)
            row.update(d)
        for row in rows:
            row.pop("_geom", None)
        del preds, gts, trees; M._UTM_GEOM_CACHE.clear(); gc.collect()
        print(f"{unit}\t{time.time()-t0:.1f}s\tgt {sum(1 for r in rows if r['unit']==unit)}", flush=True)
    # сверка с honest-10
    counts = defaultdict(dict)
    for (cls, reg), v in tpfp.items():
        counts[cls][reg] = v
    sc = M.score_counts(dict(counts))[0]
    keys = ["unit", "region", "cls", "markup", "dtype", "gt_idx", "area", "bin", "matched_any", "kept", "m_conf", "m_grp",
            "m_tau", "m_area", "reason", "n_same", "best_cov", "best_cov_rep", "ok_n", "ok_maxconf", "ok_grp", "ok_tau",
            "ok_area", "near_maxconf", "near_area", "other_ok_cls", "other_ok_conf"]
    with open(out / "gt_rows.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    (out / "check.json").write_text(json.dumps({"honest10_score_recomputed": sc, "n_gt": len(rows)}, ensure_ascii=False))
    print("honest10 recomputed", sc)


def diag(calc, geom, row, trees, taus):
    """Причина пропуска эталона (класс row['cls']):
    no_cand — ни один прогноз своего класса не пересекает эталон;
    shape — пересекают, но ни один не даёт cov>=0,5 и представительную точку внутри;
    below_tau — подходящий прогноз есть, но все ниже честного порога (или группа unseen);
    consumed — подходящий прогноз выше порога есть, но жадно ушёл другому эталону."""
    cls, reg = row["cls"], row["region"]
    d = {"n_same": 0, "best_cov": 0.0, "best_cov_rep": 0}
    if cls in trees:
        tree, p = trees[cls]
        idx = tree.query(geom, predicate="intersects")
        d["n_same"] = int(len(idx))
        ok = []
        for i in idx:
            pg = p[int(i)][0]
            if pg.geom_type != "Polygon":
                continue
            o, cov = calc._match_quality(pg, geom)
            if cov > d["best_cov"]:
                d["best_cov"], d["best_cov_rep"] = round(cov, 3), int(o)
            if o:
                ok.append(int(i))
        if idx.size:
            j = max((int(i) for i in idx), key=lambda i: p[i][1])
            d["near_maxconf"], d["near_area"] = round(p[j][1], 5), round(p[j][0].area, 1)
        d["ok_n"] = len(ok)
        if ok:
            j = max(ok, key=lambda i: p[i][1])
            t = taus.get(p[j][2], {}).get((p[j][3], cls))
            d.update(ok_maxconf=round(p[j][1], 5), ok_grp=p[j][3], ok_tau="unseen" if t is None else t,
                     ok_area=round(p[j][0].area, 1))
            above = [i for i in ok if (lambda t: t is not None and p[i][1] >= t)(taus.get(p[i][2], {}).get((p[i][3], cls)))]
            d["reason"] = "consumed" if above else "below_tau"
        else:
            d["reason"] = "shape" if idx.size else "no_cand"
    else:
        d["reason"] = "no_cand"
    best = None
    for oc, (tree, p) in trees.items():
        if oc == cls:
            continue
        for i in tree.query(geom, predicate="intersects"):
            pg = p[int(i)][0]
            if pg.geom_type == "Polygon" and calc._match_quality(pg, geom)[0]:
                if best is None or p[int(i)][1] > best[1]:
                    best = (oc, p[int(i)][1])
    if best:
        d["other_ok_cls"], d["other_ok_conf"] = best[0], round(best[1], 5)
    return d


if __name__ == "__main__":
    main()
