#!/usr/bin/env python3
"""Второй этап: переоценка кандидатов стека по контексту, вне обучения по семьям.

Пороги групп видят только уверенность самого объекта. Контекст, который знает
археолог, в ней не учтён:
- курганы стоят могильниками: у настоящего есть соседи своего класса в 10–100 м;
- кандидат, который подтверждает другой слой или другая модель, вероятнее;
- у кургана есть пара другой метки на том же месте (сдача выдаёт обе);
- ранг объекта внутри своего участка и группы, а не абсолютная уверенность;
- форма маски: фортификация вытянута, городище и курган компактны;
- рядом с фортификацией стоят городище и постройки, одиночный вал в поле чаще ложный.

Признаки считаются только из самого стека, растры не читаются. Оценщик — LightGBM
на класс, рост уверенности объекта не может понизить оценку (монотонное
ограничение). Каждая семья участков оценивается моделью, обученной без неё
(`expds.metric.family_folds`). Метка — «прошёл бы правило метрики хоть с одним
эталоном своего класса» (`F1Calculator._match_quality`), без сопоставления
один к одному. Классы, где попаданий меньше `--min-pos`, остаются как были.

Выход — тот же стек с новой `confidence` и старой в `confidence_base`. Пороги групп
подбирает табло заново:

    python scripts/scoreboard.py <флаги раздела 0> --dump-stack F:/expds3/work/stack_base.geojson
    python scripts/rerank_oof.py --stack F:/expds3/work/stack_base.geojson --gt F:/expds3/work/gt_train_v2.geojson \
        --out F:/expds3/work/stack_rerank.geojson
    python scripts/scoreboard.py <те же флаги> --rerank F:/expds3/work/stack_rerank.geojson --honest 10 --honest-by region

Остаточная утечка второго порядка: порог фолда подбирают участки, чьи оценки дала
модель, видевшая этот фолд. Порог — одно число на группу и класс, поэтому она мала;
решает честное табло и площадка, не эта оговорка.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from expds.metric import family_folds, load_official

log = logging.getLogger("rerank_oof")

KURGANS = ("kurgany_povrezhdennye", "kurgany_tselye")
#: Радиус подтверждения другим источником, м: курган — своя насыпь, городище — площадка.
SUPPORT_M = {"kurgany_povrezhdennye": 8.0, "kurgany_tselye": 8.0, "gorodishcha": 60.0, "fortifikatsii": 20.0,
             "arkhitektury": 10.0, "object_poly": 5.0, "finds_points": 2.0}
#: Классы, которые стоят рядом на одном памятнике: вал и ров у городища, постройки внутри.
RELATED = {"fortifikatsii": ("gorodishcha", "arkhitektury"), "gorodishcha": ("fortifikatsii", "arkhitektury"),
           "arkhitektury": ("gorodishcha", "fortifikatsii"), "object_poly": ("finds_points",),
           "finds_points": ("object_poly",)}
FEATURES = ["conf", "logit", "group", "markup", "log_area", "compact", "axis", "rank_pct", "log_n_rgc",
            "n10", "n30", "n100", "nbr_max30", "twin", "x_max", "x_groups", "x_any", "xf_max",
            "rel_max200", "n200_all", "reg_groups", "reg_has_li", "reg_log_n", "reg_top"]
CATEGORICAL = ("group", "markup")


def superclass(cls: str) -> str:
    return "kurgany" if cls in KURGANS else cls


def family_of_group(group: str) -> str:
    g = group.lower()
    if g.startswith(("li", "gor_phys", "la")):
        return "relief"
    if g.startswith(("sp", "ae", "or")):
        return "optical"
    if g.startswith(("maggr", "etgr", "geo", "terfl")):
        return "geophysics"
    return g


def _shape(g) -> tuple[float, float]:
    """Компактность 4πA/P² и отношение сторон минимального прямоугольника."""
    if g.geom_type == "Point" or g.area <= 0:
        return 1.0, 1.0
    compact = 4 * math.pi * g.area / max(g.length ** 2, 1e-9)
    try:
        rect = g.minimum_rotated_rectangle
        xs, ys = rect.exterior.coords.xy
        sides = sorted(math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i]) for i in range(2))
        axis = sides[1] / max(sides[0], 1e-6)
    except Exception:
        axis = 1.0
    return float(compact), float(axis)


def collect(stack: dict, gt: dict, metric_path: str | None) -> tuple[list[dict], np.ndarray]:
    """Кандидаты в метрах UTM (как в метрике) и метка «прошёл бы правило с эталоном своего класса»."""
    from shapely.strtree import STRtree

    mod = load_official(metric_path)
    calc = mod.F1Calculator()
    rows: list[dict] = []
    for idx, feature in enumerate(stack.get("features", [])):
        props = feature.get("properties", {}) or {}
        cls = mod.normalize_class_name(props.get("class_name", "unknown"))
        if cls is None or cls not in calc.target_classes:
            continue
        geom = mod._extract_geometry(feature)
        if geom is None:
            continue
        geom = mod._to_utm(geom, calc.input_crs_mode)
        if geom.geom_type == "Point":
            geom = mod._point_to_circle(geom, calc.r_tolerance, calc.circle_points)
        rows.append({"idx": idx, "cls": cls, "geom": geom, "conf": float(mod._get_confidence(feature)),
                     "region": str(props.get("region_name", "default_source")),
                     "group": str(props.get("threshold_group") or props.get("markup_type") or ""),
                     "markup": str(props.get("markup_type") or "")})
    gts: dict[str, list] = defaultdict(list)
    for feature in gt.get("features", []):
        props = feature.get("properties", {}) or {}
        cls = mod.normalize_class_name(props.get("class_name", "unknown"))
        g = mod._extract_geometry(feature)
        if cls in calc.target_classes and g is not None:
            gts[cls].append(mod._to_utm(g, calc.input_crs_mode))
    trees = {c: STRtree([x.buffer(calc.r_tolerance) if x.geom_type == "Point" else x for x in v])
             for c, v in gts.items() if v}
    y = np.zeros(len(rows), np.int8)
    for i, r in enumerate(rows):
        tree = trees.get(r["cls"])
        if tree is None:
            continue
        y[i] = any(calc._match_quality(r["geom"], gts[r["cls"]][int(j)])[0] for j in tree.query(r["geom"]))
    return rows, y


def features(rows: list[dict]) -> np.ndarray:
    """Признаки контекста из самого стека: соседи, другие источники, ранг, участок, форма."""
    from scipy.spatial import cKDTree

    n = len(rows)
    X = np.full((n, len(FEATURES)), np.nan, np.float64)
    col = {f: i for i, f in enumerate(FEATURES)}
    groups = sorted({r["group"] for r in rows})
    markups = sorted({r["markup"] for r in rows})
    gcode = {g: i for i, g in enumerate(groups)}
    mcode = {m: i for i, m in enumerate(markups)}
    xy = np.array([[r["geom"].centroid.x, r["geom"].centroid.y] for r in rows]) if n else np.zeros((0, 2))
    conf = np.array([r["conf"] for r in rows])
    by_region: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_region[r["region"]].append(i)
        c = min(max(r["conf"], 1e-6), 1 - 1e-6)
        X[i, col["conf"]] = r["conf"]
        X[i, col["logit"]] = math.log(c / (1 - c))
        X[i, col["group"]] = gcode[r["group"]]
        X[i, col["markup"]] = mcode[r["markup"]]
        X[i, col["log_area"]] = math.log1p(r["geom"].area)
        X[i, col["compact"]], X[i, col["axis"]] = _shape(r["geom"])
    for idx in by_region.values():
        idx = np.array(idx)
        tree = cKDTree(xy[idx])
        sup = np.array([superclass(rows[i]["cls"]) for i in idx])
        grp = np.array([rows[i]["group"] for i in idx])
        fam = np.array([family_of_group(g) for g in grp])
        cls = np.array([rows[i]["cls"] for i in idx])
        rgc: dict[tuple, list[int]] = defaultdict(list)
        for k, i in enumerate(idx):
            rgc[(grp[k], cls[k])].append(k)
        for members in rgc.values():
            m = np.array(members)
            order = np.argsort(-conf[idx[m]], kind="stable")
            pct = np.empty(len(m))
            pct[order] = 1.0 - np.arange(len(m)) / max(len(m), 1)
            top = float(np.mean(np.sort(conf[idx[m]])[::-1][:10]))
            for k, p in zip(m, pct):
                X[idx[k], col["rank_pct"]] = p
                X[idx[k], col["log_n_rgc"]] = math.log1p(len(m))
                X[idx[k], col["reg_top"]] = top
        n_groups = len(set(grp))
        has_li = float(any(f == "relief" for f in fam))
        near100 = tree.query_ball_point(xy[idx], 100.0)
        near200 = tree.query_ball_point(xy[idx], 200.0)
        for k, i in enumerate(idx):
            rel = RELATED.get(cls[k], ())
            rel_near = [j for j in near200[k] if cls[j] in rel]
            X[i, col["rel_max200"]] = float(conf[idx[rel_near]].max()) if rel_near else 0.0
            X[i, col["n200_all"]] = float(len(near200[k]) - 1)
            X[i, col["reg_groups"]] = n_groups
            X[i, col["reg_has_li"]] = has_li
            X[i, col["reg_log_n"]] = math.log1p(len(idx))
            p = xy[i]
            same = [j for j in near100[k] if j != k and grp[j] == grp[k] and sup[j] == sup[k]]
            d = np.hypot(*(xy[idx[same]] - p).T) if same else np.zeros(0)
            X[i, col["n10"]] = float((d <= 10).sum())
            X[i, col["n30"]] = float((d <= 30).sum())
            X[i, col["n100"]] = float(len(same))
            near30 = [j for j, dj in zip(same, d) if dj <= 30]
            X[i, col["nbr_max30"]] = float(conf[idx[near30]].max()) if near30 else 0.0
            if cls[k] in KURGANS:
                twins = [j for j in near100[k] if j != k and grp[j] == grp[k] and cls[j] in KURGANS
                         and cls[j] != cls[k] and math.hypot(*(xy[idx[j]] - p)) <= 1.5]
                X[i, col["twin"]] = float(conf[idx[twins]].max()) if twins else -1.0
            radius = SUPPORT_M.get(cls[k], 10.0)
            near = [j for j in tree.query_ball_point(p, radius) if j != k]
            other = [j for j in near if grp[j] != grp[k]]
            x_same = [j for j in other if sup[j] == sup[k]]
            X[i, col["x_max"]] = float(conf[idx[x_same]].max()) if x_same else 0.0
            X[i, col["x_groups"]] = float(len({grp[j] for j in x_same}))
            X[i, col["x_any"]] = float(conf[idx[other]].max()) if other else 0.0
            xf = [j for j in x_same if fam[j] != fam[k]]
            X[i, col["xf_max"]] = float(conf[idx[xf]].max()) if xf else 0.0
    return X


def oof_scores(rows: list[dict], X: np.ndarray, y: np.ndarray, folds: int = 5, min_pos: int = 30,
               rounds: int = 300, seed: int = 0) -> tuple[np.ndarray, dict]:
    """Оценки вне обучения по семьям участков; класс с малым числом попаданий не трогается."""
    import lightgbm as lgb

    from expds.naming import site_family

    out = np.array([r["conf"] for r in rows], np.float64)
    report: dict = {}
    regions = sorted({r["region"] for r in rows})
    split = family_folds(regions, site_family, folds)
    fold_of = {reg: k for k, regs in enumerate(split) for reg in regs}
    fold = np.array([fold_of[r["region"]] for r in rows])
    cls = np.array([r["cls"] for r in rows])
    mono = [1 if f in ("conf", "logit") else 0 for f in FEATURES]
    params = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 40,
              "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
              "monotone_constraints": mono, "verbose": -1, "seed": seed, "num_threads": 0}
    cat = [FEATURES.index(c) for c in CATEGORICAL]
    for c in sorted(set(cls)):
        m = cls == c
        pos = int(y[m].sum())
        if pos < min_pos or len(set(fold[m])) < 2:
            report[c] = {"rows": int(m.sum()), "pos": pos, "reranked": False}
            continue
        for k in sorted(set(fold[m])):
            tr = m & (fold != k)
            te = m & (fold == k)
            if not te.any():
                continue
            if y[tr].sum() == 0 or y[tr].sum() == tr.sum():
                continue
            ds = lgb.Dataset(X[tr], label=y[tr], categorical_feature=cat, free_raw_data=False)
            booster = lgb.train(params, ds, num_boost_round=rounds)
            out[te] = booster.predict(X[te])
        report[c] = {"rows": int(m.sum()), "pos": pos, "reranked": True}
    return out, report


def auc(score: np.ndarray, y: np.ndarray) -> float:
    pos, neg = score[y == 1], score[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]), kind="stable"), kind="stable") + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def rerank(stack: dict, gt: dict, metric_path: str | None = None, folds: int = 5, min_pos: int = 30) -> tuple[dict, dict]:
    rows, y = collect(stack, gt, metric_path)
    X = features(rows)
    new, report = oof_scores(rows, X, y, folds, min_pos)
    feats = [dict(f, properties=dict(f.get("properties") or {})) for f in stack.get("features", [])]
    cls = np.array([r["cls"] for r in rows])
    base = np.array([r["conf"] for r in rows])
    for c, rep in report.items():
        m = cls == c
        rep["auc_base"] = round(auc(base[m], y[m]), 4)
        rep["auc_rerank"] = round(auc(new[m], y[m]), 4)
    done = {c for c, rep in report.items() if rep["reranked"]}
    for f in feats:
        f["properties"]["confidence_base"] = f["properties"].get("confidence")
    for r, s in zip(rows, new):
        if r["cls"] in done:
            feats[r["idx"]]["properties"]["confidence"] = round(float(s), 6)
    return {"type": "FeatureCollection", "features": feats}, report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stack", required=True, help="стек кандидатов до порогов (scoreboard.py --dump-stack)")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--metric", default=str(ROOT / "materials" / "compute_metrics.py"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=5, help="фолды по семьям участков")
    ap.add_argument("--min-pos", type=int, default=30, help="меньше попаданий в классе — класс не переоценивается")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    stack = json.loads(Path(args.stack).read_text(encoding="utf-8"))
    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    out, report = rerank(stack, gt, args.metric, args.folds, args.min_pos)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
