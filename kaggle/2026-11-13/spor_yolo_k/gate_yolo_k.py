#!/usr/bin/env python3
"""Гейт YOLO26s спутника, вариант K: добавить поверх ветки, не заменить её.

Правило записано в PREREG до просмотра этого OOF. Порог новой группы выбирается
по фолдам семей (на обучении фолда, проверка на отложенных). Старые группы
остаются на полах архива eb268d6. Дубль ближе 8 м к уже принятому объекту
спутника того же класса снимается. В сдачу класс идёт, только если добавка
из a попаданий и b ложных имеет a/(a+b) > F/(4-F), где F — F1 класса на базе.

    python gate_yolo_k.py OOF_DIR META.json INDEX.csv [OUT.json]
"""
from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import numpy as np
from affine import Affine
from pyproj import Transformer
from shapely.geometry import Point, mapping
from shapely.strtree import STRtree

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from expds.metric import threshold_group  # noqa: E402
from expds.post import _pair_ground_m  # noqa: E402

CLASSES = ["kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha", "arkhitektury", "fortifikatsii"]
RADIUS_M = {"kurgany_povrezhdennye": 6.0, "kurgany_tselye": 6.0, "arkhitektury": 6.0,
            "fortifikatsii": 10.0, "gorodishcha": 40.0}
CIRCLE_M = {"kurgany_povrezhdennye": 0.5, "kurgany_tselye": 0.5}
TAUS = (0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75)
GROUP = "sp_yolo"


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


def build_feats(oof_dir: Path, meta_path: Path, index_path: Path) -> list[dict]:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    idx = {r["tile"]: r for r in csv.DictReader(index_path.open(encoding="utf-8"))}
    pts: dict = {}
    for fold in range(5):
        path = oof_dir / f"oof_f{fold}.jsonl"
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            rec = json.loads(line)
            row = idx.get(rec["tile"])
            if row is None or int(row["fold"]) != fold:
                continue
            key = f"{row['site']}__{Path(row['image']).stem}"
            m = meta[key]
            tr = Affine(*m["transform"])
            x0, y0 = float(row["x0"]), float(row["y0"])
            for (bx0, by0, bx1, by1), conf, k in zip(rec["xyxy"], rec["conf"], rec["cls"]):
                X, Y = tr * (x0 + (bx0 + bx1) / 2, y0 + (by0 + by1) / 2)
                pts.setdefault((row["site"], int(k)), []).append(
                    (float(conf), float(X), float(Y), m["crs"], fold))
    tfs: dict = {}
    feats = []
    for (site, k), lst in sorted(pts.items()):
        name = CLASSES[k]
        lst.sort(key=lambda t: -t[0])
        keep = []
        rad = RADIUS_M[name]
        for conf, X, Y, crs, fold in lst:
            if any((X - a) ** 2 + (Y - b) ** 2 < rad * rad for _, a, b, _, _ in keep):
                continue
            keep.append((conf, X, Y, crs, fold))
        for conf, X, Y, crs, fold in keep:
            if crs not in tfs:
                tfs[crs] = Transformer.from_crs(crs, "EPSG:3857", always_xy=True).transform
            gx, gy = tfs[crs](X, Y)
            r_m = CIRCLE_M.get(name, 0.05)
            geom = Point(gx, gy).buffer(r_m / max(np.cos(np.arctan(np.sinh(gy / 6378137.0))), 1e-6), quad_segs=8)
            feats.append({"type": "Feature", "geometry": mapping(geom),
                          "properties": {"class_name": name, "confidence": round(conf, 5), "region_name": site,
                                         "markup_type": "sp_or", "threshold_group": GROUP, "fold": fold,
                                         "crs": "EPSG:3857"},
                          "_xy": (gx, gy)})
    return feats


def drop_near(feats: list[dict], occupied: dict[str, list[tuple[float, float]]]) -> list[dict]:
    out = []
    trees = {}
    for cls, pts in occupied.items():
        if pts:
            trees[cls] = (STRtree([Point(x, y) for x, y in pts]), pts)
    for f in feats:
        cls = f["properties"]["class_name"]
        gx, gy = f["_xy"]
        tree = trees.get(cls)
        if tree is not None:
            near = tree[0].query(Point(gx, gy).buffer(30))
            if any(_pair_ground_m(Point(gx, gy), Point(*tree[1][int(j)])) < 8.0 for j in np.atleast_1d(near)):
                continue
        out.append(f)
    return out


def as_features(feats: list[dict]) -> list[dict]:
    return [{k: v for k, v in f.items() if k != "_xy"} for f in feats]


def main() -> None:
    oof_dir, meta_path, index_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out = Path(sys.argv[4]) if len(sys.argv) > 4 else Path(__file__).with_name("gate_yolo_k.json")
    t0 = time.perf_counter()
    G = load_gate()
    D, M, log, site_family = G.D, G.M, G.log, G.site_family
    floor = floors()
    feats = build_feats(oof_dir, meta_path, index_path)
    log(f"после слияния окон: {len(feats)}")
    import ijson
    occupied: dict[str, list] = {c: [] for c in CLASSES}
    stack_p = Path("/workspace/koz3/w5data/stack_arch.geojson")
    with stack_p.open("rb") as fh:
        for feat in ijson.items(fh, "features.item"):
            q = feat.get("properties") or {}
            cls = q.get("class_name")
            if cls not in occupied or q.get("markup_type") != "sp_or":
                continue
            if float(q.get("confidence") or 0) < floor(threshold_group(q), cls):
                continue
            g = feat.get("geometry") or {}
            coords = g.get("coordinates")
            if not coords:
                continue
            # центроид кольца не нужен: у круга берём среднее первой оболочки
            ring = coords[0] if g.get("type") == "Polygon" else None
            if not ring:
                continue
            xs = [p[0] for p in ring]
            ys = [p[1] for p in ring]
            occupied[cls].append((float(np.mean(xs)), float(np.mean(ys))))
    before = len(feats)
    feats = drop_near(feats, occupied)
    log(f"после снятия дублей 8 м: {len(feats)} (было {before})")
    by_cls = {c: [f for f in feats if f["properties"]["class_name"] == c] for c in CLASSES}

    stk = D.Stack(stack_p, Path("/workspace/koz3/w5data/gt_train_v2.geojson"),
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
    base_counts = D.honest(base_curves, folds, fixed=frozenset(groups | {GROUP}))[0]
    del base_curves

    per_tau = {c: {} for c in CLASSES}
    for cls, rows in by_cls.items():
        if not rows:
            continue
        for tau in TAUS:
            kept = [f for f in rows if f["properties"]["confidence"] >= tau]
            curves, _ = base_tab.curves(as_features(kept))
            per_tau[cls][tau] = D.honest(curves, folds, fixed=frozenset(groups | {GROUP}))[0]
            log(f"{cls} τ {tau}: {len(kept)}")

    def f1_of(counts, cls, regions):
        tp = fp = ng = 0.0
        for r, (a, b, g) in counts[cls].items():
            if r in regions:
                tp += a
                fp += b
                ng += g
        return M._f1(tp, fp, ng - tp), tp, fp, ng

    every = set().union(*folds) if folds else set()
    nested = {c: {} for c in base_counts}
    chosen = []
    for fi, held in enumerate(folds):
        train = every - held
        taus = {}
        for cls in CLASSES:
            if not per_tau[cls]:
                continue
            best_tau, best_f = TAUS[0], -1.0
            for tau, counts in per_tau[cls].items():
                f, *_ = f1_of(counts, cls, train)
                if f > best_f:
                    best_f, best_tau = f, tau
            taus[cls] = best_tau
        chosen.append(taus)
        for cls, counts0 in base_counts.items():
            src = per_tau[cls].get(taus.get(cls)) if cls in taus else None
            use = src if src is not None else base_counts
            for r in held:
                if r in use.get(cls, {}):
                    nested[cls][r] = list(use[cls][r])
                elif r in counts0:
                    nested[cls][r] = list(counts0[r])

    boot = M.paired_bootstrap_counts(nested, base_counts, site_family, 2000, 0)
    sa, pa = M.score_counts(base_counts)
    sb, pb = M.score_counts(nested)
    classes = {}
    for cls in CLASSES:
        fa, tpa, fpa, nga = f1_of(base_counts, cls, every)
        fb, tpb, fpb, ngb = f1_of(nested, cls, every)
        a, b = tpb - tpa, fpb - fpa
        denom = a + b
        h = (a / denom) if denom > 0 else None
        need = fa / (4 - fa) if fa < 4 else 1.0
        classes[cls] = {"f_base": fa, "f_new": fb, "d_f1": fb - fa, "tp": [tpa, tpb], "fp": [fpa, fpb],
                        "n_added_hit_rate": h, "break_even": need,
                        "beats_break_even": bool(h is not None and h > need and a > 0),
                        "n_raw": len(by_cls[cls])}
    res = {"what": "YOLO26s SpOr, вариант K, порог по фолдам семей, дубль 8 м снят",
           "n_after_dedup": len(feats), "delta": sb - sa, "P": boot.get("p_gt0"),
           "p5": boot.get("p5"), "p95": boot.get("p95"), "base": sa, "new": sb,
           "fold_taus": chosen, "classes": classes,
           "PASS_delta": (sb - sa) >= 0.003 and (boot.get("p_gt0") or 0) >= 0.95,
           "sec": round(time.perf_counter() - t0)}
    res["PASS"] = bool(res["PASS_delta"] and any(v["beats_break_even"] for v in classes.values()))
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"Δ {res['delta']:+.6f} P {res['P']} PASS {res['PASS']} за {res['sec']} с")


if __name__ == "__main__":
    main()
