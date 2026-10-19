"""Оценка официальной метрикой и точный подбор порогов по классам.

Официальный compute_metrics.py в архив не кладётся: путь к вашей копии с
платформы передаётся явно (--metric) или через EXPDS_METRIC. Свою метрику
не пишем: геометрические проверки берутся из официального кода.

Почему подбор точный и быстрый. Метрика сопоставляет предсказания жадно, по
убыванию confidence, и каждый класс считает отдельно. Отбросить предсказания
ниже порога τ — значит отрезать «хвост» отсортированного списка; решения для
более уверенных предсказаний от хвоста не зависят. Поэтому достаточно ОДНОГО
прохода сопоставления на всех кандидатах: запоминаем, какие попали (hit), и для
любого τ получаем TP = hit с conf ≥ τ, FP = (всего с conf ≥ τ) − TP, FN = GT − TP.
Перебираются все различимые пороги, а не сетка; совпадение с полным официальным
расчётом проверяет verify() (и тест test_metric_exact.py).

    python -m expds.metric score --pred out.geojson --gt work/gt_val.geojson --metric C:/koz3/compute_metrics.py
    python -m expds.metric sweep --pred cand.geojson --gt work/gt_train.geojson --metric ... \
        --markup li --regions holdout.txt --bootstrap 300 --out configs/thresholds.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from .constants import CLASS_WEIGHTS, CLASSES

EPS = 1e-6


@lru_cache(maxsize=2)
def load_official(path: str | None = None):
    """Импортирует официальный compute_metrics.py как модуль (рядом может лежать его config.py)."""
    path = path or os.environ.get("EXPDS_METRIC")
    if not path or not Path(path).exists():
        raise FileNotFoundError("укажите путь к официальному compute_metrics.py (--metric или EXPDS_METRIC)")
    parent = str(Path(path).resolve().parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location("koz3_compute_metrics", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def score(pred: dict, gt: dict, metric_path: str | None = None) -> dict:
    """Полный результат официальной метрики (final_score, per_class, auxiliary)."""
    mod = load_official(metric_path)
    return mod.F1Calculator().calculate_f1(pred, gt)


# ─────────────────────────── один проход сопоставления ───────────────────────

@dataclass
class ClassCurve:
    """Итог одного прохода для класса: уверенности по убыванию и попадания."""
    conf: np.ndarray
    hit: np.ndarray
    pred_region: list[str]
    n_gt: int
    gt_regions: Counter = field(default_factory=Counter)


def _collect(mod, calc, fc: dict, is_gt: bool) -> dict[str, list[tuple]]:
    """Повторяет F1Calculator._collect_by_class, сохраняя регион объекта."""
    out: dict[str, list[tuple]] = {}
    for feature in fc.get("features", []):
        props = feature.get("properties", {}) or {}
        cls = mod.normalize_class_name(props.get("class_name", "unknown"))
        if cls is None or cls not in calc.target_classes:
            continue
        geom = mod._extract_geometry(feature)
        if geom is None:
            continue
        geom = mod._to_utm(geom, calc.input_crs_mode)
        if geom.geom_type == "Point" and not is_gt:
            geom = mod._point_to_circle(geom, calc.r_tolerance, calc.circle_points)
        out.setdefault(cls, []).append((geom, mod._get_confidence(feature),
                                        str(props.get("region_name", "default_source"))))
    return out


def build_curves(pred: dict, gt: dict, metric_path: str | None = None) -> dict[str, ClassCurve]:
    """Официальное жадное сопоставление по каждому классу, один раз на всех кандидатах.

    Кандидаты на пару ищутся пространственным индексом; порядок перебора эталонов
    (по возрастанию индекса) и правило выбора лучшего — как в официальном коде.
    """
    from shapely.strtree import STRtree

    mod = load_official(metric_path)
    calc = mod.F1Calculator()
    preds, gts = _collect(mod, calc, pred, False), _collect(mod, calc, gt, True)
    curves: dict[str, ClassCurve] = {}
    for cls in calc.target_classes:
        p, g = preds.get(cls, []), gts.get(cls, [])
        order = sorted(range(len(p)), key=lambda i: p[i][1], reverse=True)   # стабильно, как sorted()
        hits = np.zeros(len(p), dtype=bool)
        if p and g:
            gg = [x[0] for x in g]
            tree = STRtree([x.buffer(calc.r_tolerance) if x.geom_type == "Point" else x for x in gg])
            matched: set[int] = set()
            for rank, i in enumerate(order):
                pg = p[i][0]
                if pg.geom_type != "Polygon":
                    continue
                best, best_j = float("-inf"), -1
                for j in sorted(int(k) for k in tree.query(pg)):
                    if j in matched:
                        continue
                    ok, s = calc._match_quality(pg, gg[j])
                    if ok and s > best + 1e-9:
                        best, best_j = s, j
                if best_j >= 0:
                    matched.add(best_j)
                    hits[rank] = True
        curves[cls] = ClassCurve(conf=np.array([p[i][1] for i in order], dtype=np.float64), hit=hits,
                                 pred_region=[p[i][2] for i in order], n_gt=len(g),
                                 gt_regions=Counter(x[2] for x in g))
    return curves


def _f1(tp, fp, fn):
    """F1 ровно как в метрике: FP с весом 0,5 в точности, эпсилон 1e-6."""
    precision = tp / (tp + 0.5 * fp + EPS)
    recall = tp / (tp + fn + EPS)
    return 2 * precision * recall / (precision + recall + EPS)


def curve_table(c: ClassCurve) -> tuple[np.ndarray, np.ndarray]:
    """Все различимые пороги класса (оставляем conf ≥ τ) и F1 на каждом; первый — «ничего не выдавать»."""
    if c.conf.size == 0:
        return np.array([np.inf]), np.array([0.0])
    last = np.r_[np.flatnonzero(c.conf[1:] != c.conf[:-1]), c.conf.size - 1]
    tp = np.cumsum(c.hit)[last].astype(float)
    n = (last + 1).astype(float)
    return np.r_[np.inf, c.conf[last]], np.r_[0.0, _f1(tp, n - tp, c.n_gt - tp)]


def score_at(curves: dict[str, ClassCurve], thresholds: dict[str, float]) -> tuple[float, dict]:
    """Итоговый балл при порогах (классы без эталона из среднего исключаются, как в метрике)."""
    num = den = 0.0
    per = {}
    for cls, c in curves.items():
        tau = thresholds.get(cls) or 0.0
        kept = c.conf >= tau
        tp = int(c.hit[kept].sum())
        fp = int(kept.sum()) - tp
        f1 = float(_f1(tp, fp, c.n_gt - tp))
        per[cls] = {"f1": f1, "tp": tp, "fp": fp, "fn": c.n_gt - tp, "n_gt": c.n_gt}
        if c.n_gt > 0:
            num += CLASS_WEIGHTS.get(cls, 0.0) * f1
            den += CLASS_WEIGHTS.get(cls, 0.0)
    return (num / den if den else 0.0), per


def bootstrap(curves: dict[str, ClassCurve], n: int = 200, seed: int = 0) -> dict[str, dict]:
    """Устойчивость порогов: регионы выбираются с возвращением, оптимум ищется заново.

    Регионы в пространстве не пересекаются, поэтому попадания из общего прохода
    переиспользуются с весами (сколько раз регион попал в выборку).
    """
    regions = sorted({r for c in curves.values() for r in c.pred_region} |
                     {r for c in curves.values() for r in c.gt_regions})
    if not regions or n <= 0:
        return {}
    idx = {r: i for i, r in enumerate(regions)}
    rng = np.random.default_rng(seed)
    taus: dict[str, list[float]] = {c: [] for c in curves}
    for _ in range(n):
        w = np.bincount(rng.integers(0, len(regions), len(regions)), minlength=len(regions)).astype(float)
        for cls, c in curves.items():
            n_gt = sum(w[idx[r]] * k for r, k in c.gt_regions.items())
            if n_gt == 0 or c.conf.size == 0:
                continue
            pw = np.array([w[idx[r]] for r in c.pred_region])
            last = np.r_[np.flatnonzero(c.conf[1:] != c.conf[:-1]), c.conf.size - 1]
            tp, nn = np.cumsum(c.hit * pw)[last], np.cumsum(pw)[last]
            taus[cls].append(float(c.conf[last[int(np.argmax(_f1(tp, nn - tp, n_gt - tp)))]]))
    return {cls: {"p10": float(np.percentile(v, 10)), "p50": float(np.median(v)),
                  "p90": float(np.percentile(v, 90)), "samples": len(v)}
            for cls, v in taus.items() if v}


def apply_thresholds(fc: dict, thresholds: dict[str, float], metric_path: str | None = None) -> dict:
    """Оставляет предсказания с confidence ≥ порога своего класса."""
    norm = load_official(metric_path).normalize_class_name
    feats = [f for f in fc.get("features", [])
             if float((f.get("properties") or {}).get("confidence", 1.0))
             >= (thresholds.get(norm((f.get("properties") or {}).get("class_name"))) or 0.0)]
    return {**{k: v for k, v in fc.items() if k != "features"}, "features": feats}


def verify(pred: dict, gt: dict, thresholds: dict[str, float], metric_path: str | None = None,
           curves: dict[str, ClassCurve] | None = None) -> dict:
    """Сверка: быстрый балл против полного официального расчёта на отфильтрованных предсказаниях."""
    curves = curves or build_curves(pred, gt, metric_path)
    fast, _ = score_at(curves, thresholds)
    official = score(apply_thresholds(pred, thresholds, metric_path), gt, metric_path)["final_score"]
    return {"fast": fast, "official": official, "abs_diff": abs(fast - official)}


def sweep(pred: dict, gt: dict, metric_path: str | None = None, n_boot: int = 0) -> dict:
    """Лучший порог и F1 по каждому классу + итоговый балл с этими порогами.

    Порог ставится посередине между последним оставленным и первым отброшенным
    значением confidence — так он устойчивее переносится на новые данные.
    Классы без эталона пропускаются: по этим данным их порог не определить.
    """
    curves = build_curves(pred, gt, metric_path)
    best: dict[str, dict] = {}
    for cls in CLASSES:
        c = curves.get(cls)
        if c is None or c.n_gt == 0:
            continue
        taus, f1 = curve_table(c)
        k = int(np.argmax(f1))
        if np.isinf(taus[k]):
            tau = float(c.conf.max() + 1e-6) if c.conf.size else 1.0
        else:
            lower = c.conf[c.conf < taus[k]]
            tau = float((taus[k] + lower.max()) / 2) if lower.size else float(taus[k])
        best[cls] = {"threshold": round(tau, 6), "f1": round(float(f1[k]), 4),
                     "f1_all": round(float(f1[-1]), 4), "n_gt": c.n_gt, "n_pred": int(c.conf.size),
                     "curve": [(float(t), round(float(v), 5)) for t, v in zip(taus[1:], f1[1:])]}
    thresholds = {c: v["threshold"] for c, v in best.items()}
    est, _ = score_at(curves, thresholds)
    return {"thresholds": thresholds, "per_class": best, "estimated_score": round(est, 6),
            "bootstrap": bootstrap(curves, n_boot) if n_boot else {}, "_curves": curves}


def _filter(fc: dict, markup: set[str] | None, regions: set[str] | None) -> dict:
    feats = [f for f in fc["features"]
             if (markup is None or (f.get("properties") or {}).get("markup_type") in markup)
             and (regions is None or (f.get("properties") or {}).get("region_name") in regions)]
    return {**fc, "features": feats}


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Официальная метрика и точный подбор порогов")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("score", "sweep"):
        p = sub.add_parser(name)
        p.add_argument("--pred", required=True)
        p.add_argument("--gt", required=True)
        p.add_argument("--metric", default=None)
        p.add_argument("--markup", default=None, help="только эти markup_type, через запятую (например, li)")
        p.add_argument("--regions", default=None, help="файл со списком регионов (по одному в строке)")
        if name == "sweep":
            p.add_argument("--out", default=None, help="куда сохранить пороги (JSON)")
            p.add_argument("--bootstrap", type=int, default=0, help="повторов бутстрепа по регионам")
    args = ap.parse_args()
    pred = json.loads(Path(args.pred).read_text(encoding="utf-8-sig"))
    gt = json.loads(Path(args.gt).read_text(encoding="utf-8-sig"))
    markup = set(args.markup.split(",")) if args.markup else None
    regions = ({ln.strip() for ln in Path(args.regions).read_text(encoding="utf-8").splitlines() if ln.strip()}
               if args.regions else None)
    pred, gt = _filter(pred, markup, regions), _filter(gt, markup, regions)
    if args.cmd == "score":
        res = score(pred, gt, args.metric)
        print(json.dumps({"final_score": res["final_score"],
                          "per_class": {c: {k: v[k] for k in ("f1", "tp", "fp", "fn")}
                                        for c, v in res["per_class"].items()}},
                         ensure_ascii=False, indent=2))
        return
    res = sweep(pred, gt, args.metric, args.bootstrap)
    check = verify(pred, gt, res["thresholds"], args.metric, res["_curves"])
    all_in = score_at(res["_curves"], {c: 0.0 for c in res["_curves"]})[0]
    print(f"{'класс':24s} {'GT':>6s} {'pred':>6s} {'порог':>8s} {'F1':>6s} {'F1 всё':>7s}  бутстреп p10–p50–p90")
    for cls, v in res["per_class"].items():
        b = res["bootstrap"].get(cls)
        bt = f"{b['p10']:.3f}–{b['p50']:.3f}–{b['p90']:.3f}" if b else "—"
        print(f"{cls:24s} {v['n_gt']:6d} {v['n_pred']:6d} {v['threshold']:8.4f} {v['f1']:6.3f} {v['f1_all']:7.3f}  {bt}")
    print(f"балл при порогах {res['estimated_score']:.6f}; официальный расчёт {check['official']:.6f} "
          f"(расхождение {check['abs_diff']:.1e}); без порогов {all_in:.6f}")
    if args.out:
        Path(args.out).write_text(json.dumps(res["thresholds"], indent=2), encoding="utf-8")
        print(f"пороги → {args.out} (перенесите в configs/default.toml, секция [thresholds])")


if __name__ == "__main__":
    _cli()
