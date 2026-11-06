#!/usr/bin/env python3
"""Балл и честный балл готового стека (дампа scoreboard.py --dump-stack) при малой памяти —.

Большой стек (stack_arch.geojson: 1,39 млн объектов, 1,1 ГБ) целиком в 2 ГБ ОЗУ не помещается.
Здесь он потоково режется по семьям участков (expds.naming.site_family), а сопоставление идёт
по «кластерам» семей, чьи рамки в UTM (как в официальной метрике, + 2·r_tolerance) пересекаются.
Объекты разных кластеров не могут составить пару, а жадное сопоставление официальной метрики
внутри класса раскладывается на независимые части по связным компонентам перекрытий — поэтому
результат тот же, что у build_curves на всём файле (global, официальная область).

Что считается (как scoreboard.py evaluate + honest):
  база      — sweep_groups по объединённым кривым → пороги групп → apply_group_thresholds →
              build_curves на оставленных (по кластерам) → «все / 34 новых / без лидара»;
  честно    — honest_counts(family_folds(k)) → score_counts.
Проверки (по флагам):
  --check-reference  то же прежним кодом (EXPDS_FAST_METRIC=0): кривые каждого кластера побайтно,
                     пороги, баллы и честные счётчики — точное равенство;
  --official         официальный F1Calculator.calculate_f1 на оставленных объектах каждого кластера:
                     TP/FP/FN каждого класса == счётчики build_curves, итог по сумме счётчиков ==.
                     (официальный перебор пар без индекса — медленно на больших кластерах.)

python scripts/stack_scoreboard.py --stack STACK.geojson --gt GT.geojson --split-dir DIR \
    [--honest 10] [--check-reference] [--official] [--out OUT.json]
Переменные: EXPDS_GEOM_CACHE (каталог файлового кэша), EXPDS_METRIC_THREADS, EXPDS_FAST_METRIC.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pickle
import resource
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from expds import metric as M  # noqa: E402
from expds.naming import site_family  # noqa: E402


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def split(stack: Path, out: Path) -> dict:
    """Потоково: <out>/<семья>.tsv, строка = индекс в файле \\t исходный текст объекта."""
    done = out / "_counts.json"
    if done.exists():
        info = json.loads(done.read_text(encoding="utf-8"))
        st = stack.stat()
        if info.get("source") == str(stack) and info.get("size") == st.st_size and info.get("mtime") == st.st_mtime:
            return info
    out.mkdir(parents=True, exist_ok=True)
    for p in out.glob("*.tsv"):
        p.unlink()
    dec = json.JSONDecoder()
    t0 = time.time()
    with open(stack, "r", encoding="utf-8") as fh:
        buf = fh.read(1 << 20)
        pos = buf.index("[", buf.index('"features"')) + 1
        files: dict = {}
        counts: Counter = Counter()
        n = 0

        def more():
            nonlocal buf, pos
            chunk = fh.read(8 << 20)
            buf = buf[pos:] + chunk
            pos = 0
            return bool(chunk)
        while True:
            while pos < len(buf) and buf[pos] in " \n\r\t,":
                pos += 1
            if len(buf) - pos < (1 << 20) and more():
                continue
            if pos >= len(buf) or buf[pos] == "]":
                break
            obj, end = dec.raw_decode(buf, pos)
            text = buf[pos:end]
            pos = end
            reg = (obj.get("properties") or {}).get("region_name")
            fam = site_family(str(reg)) if reg else "_none"
            f = files.get(fam)
            if f is None:
                f = files[fam] = open(out / f"{fam}.tsv", "w", encoding="utf-8", newline="\n")
            f.write(f"{n}\t{text}\n")
            counts[fam] += 1
            n += 1
    for f in files.values():
        f.close()
    st = stack.stat()
    info = {"source": str(stack), "size": st.st_size, "mtime": st.st_mtime, "total": n,
            "families": dict(counts), "seconds": round(time.time() - t0, 1)}
    done.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    return info


def read_family(d: Path, fam: str) -> list[tuple[int, dict]]:
    p = d / f"{fam}.tsv"
    if not p.exists():
        return []
    out = []
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            i, text = line.split("\t", 1)
            out.append((int(i), json.loads(text)))
    return out


def fc(feats):
    return {"type": "FeatureCollection", "features": feats}


def utm_bounds(mod, calc, feats, is_gt):
    import shapely
    by = M._collect(mod, calc, fc(feats), is_gt)
    gs = [x[0] for lst in by.values() for x in lst]
    if not gs:
        return None
    b = shapely.total_bounds(np.asarray(gs, dtype=object))
    return [float(v) for v in b]


def clusters_of(bounds: dict, margin: float) -> list[list[str]]:
    names = sorted(bounds)
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            A, B = bounds[a], bounds[b]
            if A[0] - margin <= B[2] and B[0] - margin <= A[2] and A[1] - margin <= B[3] and B[1] - margin <= A[3]:
                parent[find(a)] = find(b)
    groups: dict = {}
    for n in names:
        groups.setdefault(find(n), []).append(n)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])


def blob(curves):
    return pickle.dumps([(k, v.conf.tobytes(), v.hit.tobytes(), v.pred_region, v.n_gt, sorted(v.gt_regions.items()),
                          v.pred_group) for k, v in curves.items()])


def merge(parts: list[dict]) -> dict:
    """Кривые кластеров → одна кривая на класс (уверенности по убыванию, стабильно)."""
    out = {}
    classes = list(parts[0]) if parts else []
    for cls in classes:
        cs = [p[cls] for p in parts]
        conf = np.concatenate([c.conf for c in cs]) if cs else np.zeros(0)
        hit = np.concatenate([c.hit for c in cs]) if cs else np.zeros(0, bool)
        reg = [r for c in cs for r in c.pred_region]
        grp = [g for c in cs for g in c.pred_group]
        order = np.argsort(-conf, kind="stable")
        gt_regions: Counter = Counter()
        for c in cs:
            gt_regions.update(c.gt_regions)
        out[cls] = M.ClassCurve(conf=conf[order], hit=hit[order], pred_region=[reg[i] for i in order],
                                n_gt=sum(c.n_gt for c in cs), gt_regions=gt_regions,
                                pred_group=[grp[i] for i in order] if grp else [])
    return out


def sweep_to_taus(res):
    taus: dict = {}
    for cls, v in res.items():
        if v["f1"] <= 0 and not any(g["kept"] for g in v["groups"].values()):
            continue
        for g, gv in v["groups"].items():
            taus.setdefault(g, {})[cls] = gv["threshold"]
    return taus


def counts_score(tot: dict, calc, mod) -> float:
    num = den = 0.0
    for c in calc.target_classes:
        tp, fp, fn = tot.get(c, (0, 0, 0))
        p = tp / (tp + 0.5 * fp + calc.epsilon)
        r = tp / (tp + fn + calc.epsilon)
        f1 = 2 * p * r / (p + r + calc.epsilon)
        if tp + fn > 0:
            num += f1 * mod.CLASS_WEIGHTS.get(c, 0.0)
            den += mod.CLASS_WEIGHTS.get(c, 0.0)
    return num / den if den > 0 else 0.0


class Fast:
    """Включает быстрый (1) или прежний (0) путь на время блока."""
    def __init__(self, on: bool):
        self.on = on

    def __enter__(self):
        self.old = os.environ.get("EXPDS_FAST_METRIC")
        os.environ["EXPDS_FAST_METRIC"] = "1" if self.on else "0"
        M._UTM_GEOM_CACHE.clear()

    def __exit__(self, *a):
        if self.old is None:
            os.environ.pop("EXPDS_FAST_METRIC", None)
        else:
            os.environ["EXPDS_FAST_METRIC"] = self.old
        M._UTM_GEOM_CACHE.clear()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--split-dir", type=Path, required=True)
    ap.add_argument("--metric", default=str(ROOT / "materials" / "compute_metrics.py"))
    ap.add_argument("--honest", type=int, default=10)
    ap.add_argument("--check-reference", action="store_true", help="оба пути в одном процессе (память ×2)")
    ap.add_argument("--mode", choices=("fast", "reference"), default="fast",
                    help="без --check-reference: какой путь считать (reference = EXPDS_FAST_METRIC=0); "
                         "два запуска сравниваются по curve_sha/base/honest в --out")
    ap.add_argument("--official", action="store_true")
    ap.add_argument("--official-max-kept", type=int, default=0, help="официальный расчёт только кластерам с ≤ N оставленных (0 — всем)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    T: dict = {}
    t0 = time.perf_counter()
    mod = M.load_official(a.metric)
    calc = mod.F1Calculator()
    info = split(a.stack, a.split_dir)
    T["split"] = time.perf_counter() - t0
    gt_all = json.loads(a.gt.read_bytes())["features"]
    gt_by: dict = {}
    for f in gt_all:
        reg = (f.get("properties") or {}).get("region_name")
        gt_by.setdefault(site_family(str(reg)) if reg else "_none", []).append(f)
    fams = sorted(set(info["families"]) | set(gt_by))
    # 1. рамки семей в UTM (заодно файловый кэш геометрий тёплый)
    t = time.perf_counter()
    bounds = {}
    li_regs: set = set()
    for fam in fams:
        pf = [f for _, f in read_family(a.split_dir, fam)]
        for f in pf:
            pr = f.get("properties") or {}
            if pr.get("markup_type") == "li":
                li_regs.add(pr.get("region_name"))
        bs = [b for b in (utm_bounds(mod, calc, pf, False), utm_bounds(mod, calc, gt_by.get(fam, []), True)) if b]
        if bs:
            bounds[fam] = [min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)]
        M._UTM_GEOM_CACHE.clear()
    cl = clusters_of(bounds, 2 * calc.r_tolerance + 1.0)
    cl += [[f] for f in fams if f not in bounds]
    T["bounds"] = time.perf_counter() - t

    def cluster_fc(group):
        rows = sorted((r for fam in group for r in read_family(a.split_dir, fam)), key=lambda r: r[0])
        gts = [f for f in gt_all if site_family(str((f.get("properties") or {}).get("region_name") or "")) in set(group)] \
            if any(g == "_none" for g in group) else [f for fam in group for f in gt_by.get(fam, [])]
        if len(group) > 1:   # порядок эталона — как в файле
            pos = {id(f): i for i, f in enumerate(gt_all)}
            gts.sort(key=lambda f: pos[id(f)])
        return [f for _, f in rows], gts

    modes = [True, False] if a.check_reference else [a.mode == "fast"]
    main_m = modes[0]
    res: dict = {"clusters": cl, "n_features": info["total"], "mode": "both" if a.check_reference else a.mode,
                 "check": {}, "curve_sha": []}
    parts = {m: [] for m in modes}
    tcur = {m: 0.0 for m in modes}
    curve_eq = []
    for group in cl:
        pf, gf = cluster_fc(group)
        blobs = {}
        for m in modes:
            with Fast(m):
                t = time.perf_counter()
                c = M.build_curves(fc(pf), fc(gf), a.metric, scope="global")
                tcur[m] += time.perf_counter() - t
            parts[m].append(c)
            blobs[m] = blob(c)
        res["curve_sha"].append(hashlib.sha256(blobs[main_m]).hexdigest())
        if a.check_reference:
            curve_eq.append(blobs[True] == blobs[False])
        print(f"кластер {'+'.join(group)}: объектов {len(pf)}, эталон {len(gf)}, ОЗУ пик {rss_mb():.0f} МБ", flush=True)
        del pf, gf, blobs
        gc.collect()
    for m in modes:
        T["curves_" + ("fast" if m else "reference")] = tcur[m]
    if a.check_reference:
        res["check"]["curves_equal_all_clusters"] = all(curve_eq)
    new = {ln.strip() for ln in (ROOT / "notes" / "regions_las_new.txt").read_text(encoding="utf-8").splitlines() if ln.strip()}
    optical = {(f.get("properties") or {}).get("region_name") for f in gt_all} - li_regs - {None}
    out_m = {}
    for m in modes:
        with Fast(m):
            t = time.perf_counter()
            curves = merge(parts[m])
            sw = M.sweep_groups(curves)
            taus = sweep_to_taus(sw)
            regions = {r for c in curves.values() for r in c.pred_region} | {r for c in curves.values() for r in c.gt_regions}
            folds = M.family_folds(regions, site_family, None if a.honest < 0 else a.honest)
            counts, hinfo = M.honest_counts(curves, folds)
            honest = [M.score_counts(counts)[0], M.score_counts(counts, new)[0], M.score_counts(counts, optical)[0]]
            T[f"sweep_honest_{'fast' if m else 'reference'}"] = time.perf_counter() - t
        out_m[m] = {"taus": taus, "honest": honest, "counts": counts, "folds": hinfo.get("folds")}
    # 2. база: пороги → оставленные → build_curves по кластерам (и официально)
    taus = out_m[main_m]["taus"]
    tot = {m: {s: {} for s in ("all", "new", "optical")} for m in modes}
    off_tot = {s: {} for s in ("all", "new", "optical")}
    off_eq = []
    t_off = 0.0
    tk = {m: 0.0 for m in modes}

    def add(d, per):
        for cls, v in per.items():
            x = d.setdefault(cls, [0, 0, 0])
            x[0] += v["tp"]; x[1] += v["fp"]; x[2] += v["fn"]
    for group in cl:
        pf, gf = cluster_fc(group)
        kept = M.apply_group_thresholds(fc(pf), taus, a.metric)["features"]
        for sname, regs in (("all", None), ("new", new), ("optical", optical)):
            kk = kept if regs is None else [f for f in kept if (f.get("properties") or {}).get("region_name") in regs]
            gg = gf if regs is None else [f for f in gf if (f.get("properties") or {}).get("region_name") in regs]
            pers = {}
            for m in modes:
                with Fast(m):
                    t = time.perf_counter()
                    _, per = M.score_at(M.build_curves(fc(kk), fc(gg), a.metric, scope="global"), {})
                    tk[m] += time.perf_counter() - t
                add(tot[m][sname], per)
                pers[m] = per
            if a.official and sname == "all" and (not a.official_max_kept or len(kk) <= a.official_max_kept):
                t = time.perf_counter()
                r = calc.calculate_f1(fc(kk), fc(gg))["per_class"]
                t_off += time.perf_counter() - t
                same = all((r[c]["tp"], r[c]["fp"], r[c]["fn"]) == (pers[main_m][c]["tp"], pers[main_m][c]["fp"], pers[main_m][c]["fn"])
                           for c in calc.target_classes)
                off_eq.append({"cluster": "+".join(group), "kept": len(kk), "equal": same})
                add(off_tot["all"], {c: r[c] for c in calc.target_classes})
                print(f"  официально {'+'.join(group)}: оставлено {len(kk)}, счётчики равны: {same}", flush=True)
        M._UTM_GEOM_CACHE.clear()
        gc.collect()
    for m in modes:
        T["kept_" + ("fast" if m else "reference")] = tk[m]
    base = {m: [counts_score(tot[m][s], calc, mod) for s in ("all", "new", "optical")] for m in modes}
    res.update({"base": base[main_m], "honest": out_m[main_m]["honest"], "folds": out_m[main_m]["folds"], "taus": taus,
                "counts_all": tot[main_m]["all"],
                "honest_counts_repr_sha": hashlib.sha256(repr(sorted(out_m[main_m]["counts"].items(), key=lambda kv: repr(kv[0]))).encode()).hexdigest()})
    if a.check_reference:
        res["check"].update({
            "taus_equal": out_m[True]["taus"] == out_m[False]["taus"],
            "honest_equal": out_m[True]["honest"] == out_m[False]["honest"]
                            and out_m[True]["counts"] == out_m[False]["counts"],
            "base_equal": base[True] == base[False] and tot[True] == tot[False],
            "reference_base": base[False], "reference_honest": out_m[False]["honest"]})
    if a.official:
        T["official"] = t_off
        res["check"]["official_clusters"] = off_eq
        res["check"]["official_all_equal"] = all(x["equal"] for x in off_eq)
        if len(off_eq) == len(cl):
            res["check"]["official_all_score"] = counts_score(off_tot["all"], calc, mod)
            res["check"]["official_score_equal"] = res["check"]["official_all_score"] == base[main_m][0]
    T["total"] = time.perf_counter() - t0
    res["times"] = {k: round(v, 2) for k, v in T.items()}
    res["peak_rss_mb"] = round(rss_mb())
    print(json.dumps({"base": res["base"], "honest": res["honest"], "check": {k: v for k, v in res["check"].items()
                                                                              if k != "official_clusters"},
                      "times": res["times"], "peak_rss_mb": res["peak_rss_mb"]}, ensure_ascii=False))
    if a.out:
        a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
