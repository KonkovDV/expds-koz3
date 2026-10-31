"""Ранг внутри участка и плотность соседей вместо сырой уверенности.

Глобальный порог забирает яркие пустые ландшафты раньше могильников.
Ранг внутри (участок, класс) превращает порог в «верхнюю долю каждого снимка».
Соседи в радиусе 120 м в EPSG:3857 — группа насыпей, а не одиночный пик.

Сопоставление каждый раз заново: жадный разбор метрики зависит от порядка.
"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, r"C:\Sloi\expds-koz3_v0.2.0\expds-koz3")
from expds.metric import build_curves, score_at  # noqa: E402
from expds.metric import curve_table  # noqa: E402

METRIC = r"C:\Sloi\files\materials\compute_metrics.py"
KEEP = {"kurgany_povrezhdennye", "kurgany_tselye"}
GT_PATH = r"F:\expds3\work\gt_train.geojson"
RADIUS_M = 120.0


def load_kurgan(path: str, markup: str) -> list[dict]:
    fc = json.loads(open(path, encoding="utf-8").read())
    out = []
    for f in fc["features"]:
        p = f.get("properties") or {}
        if p.get("class_name") not in KEEP:
            continue
        if p.get("markup_type") not in (None, markup):
            continue
        out.append(f)
    return out


def centers(feats: list[dict]) -> np.ndarray:
    xy = np.empty((len(feats), 2), np.float64)
    for i, f in enumerate(feats):
        ring = f["geometry"]["coordinates"][0]
        n = len(ring) - 1 if len(ring) > 1 else len(ring)
        n = max(n, 1)
        sx = sy = 0.0
        for pt in ring[:n]:
            sx += pt[0]
            sy += pt[1]
        xy[i, 0] = sx / n
        xy[i, 1] = sy / n
    return xy


def groups(keys: list[str]) -> dict[str, np.ndarray]:
    buckets: dict[str, list[int]] = {}
    for i, k in enumerate(keys):
        buckets.setdefault(k, []).append(i)
    return {k: np.asarray(v, np.int64) for k, v in buckets.items()}


def percentile(values: np.ndarray, index: dict[str, np.ndarray]) -> np.ndarray:
    """1 — лучший в группе, 1/n — худший. Ничья сохраняет прежний порядок."""
    out = np.empty(len(values), np.float64)
    for idx in index.values():
        v = values[idx]
        order = np.argsort(-v, kind="mergesort")
        rank = np.empty(len(idx), np.float64)
        rank[order] = np.arange(len(idx))
        out[idx] = 1.0 - rank / len(idx)
    return out


def neighbor_counts(xy: np.ndarray, by_region: dict[str, np.ndarray], radius: float) -> np.ndarray:
    from scipy.spatial import cKDTree

    out = np.zeros(len(xy), np.float64)
    for idx in by_region.values():
        if len(idx) < 2:
            continue
        counts = np.asarray(cKDTree(xy[idx]).query_ball_point(xy[idx], radius, return_length=True))
        out[idx] = counts - 1
    return out


def best_f1(curves) -> tuple[float, dict]:
    thresholds = {}
    per = {}
    for cls, c in curves.items():
        if c.n_gt == 0:
            continue
        taus, f1 = curve_table(c)
        k = int(np.argmax(f1))
        tau = float(taus[k])
        if np.isinf(tau):
            tau = float(c.conf.max() + 1e-6) if c.conf.size else 1.0
        else:
            lower = c.conf[c.conf < tau]
            tau = float((tau + lower.max()) / 2) if lower.size else tau
        thresholds[cls] = tau
        tp = int(c.hit[c.conf >= tau].sum()) if c.conf.size else 0
        kept = int((c.conf >= tau).sum()) if c.conf.size else 0
        per[cls] = (float(f1[k]), tau, c.n_gt, kept, tp)
    score, _ = score_at(curves, thresholds)
    return float(score), per


def apply(feats: list[dict], conf: np.ndarray) -> dict:
    slim = []
    for f, c in zip(feats, conf):
        p = f["properties"]
        slim.append({
            "type": "Feature",
            "geometry": f["geometry"],
            "properties": {
                "class_name": p["class_name"],
                "confidence": float(c),
                "region_name": p.get("region_name", "default_source"),
                "markup_type": p.get("markup_type") or "sp_or",
            },
        })
    return {"type": "FeatureCollection", "features": slim}


def report(name: str, pred_feats: list[dict], conf: np.ndarray, gt: dict) -> None:
    print(f"схема {name}: сопоставление {len(pred_feats)}", flush=True)
    curves = build_curves(apply(pred_feats, conf), gt, METRIC)
    score, per = best_f1(curves)
    for cls, (f1, tau, n_gt, kept, tp) in per.items():
        print(f"  {cls:24s} GT {n_gt:5d} оставлено {kept:6d} TP {tp:4d} "
              f"порог {tau:.4f} F1 {f1:.3f}", flush=True)
    print(f"  балл слоя {score:.4f}", flush=True)


def main() -> None:
    gt_feats = load_kurgan(GT_PATH, "sp_or")
    gt = {"type": "FeatureCollection", "features": gt_feats}
    print(f"эталон курганов sp_or: {len(gt_feats)}", flush=True)
    files = [
        (r"F:\expds3\work\blobs_spor_r8.geojson", "пятна 8/16/28"),
        (r"F:\expds3\work\spor_r1_oof.geojson", "сеть 1 м"),
    ]
    for path, title in files:
        feats = load_kurgan(path, "sp_or")
        print(f"\n{title}: {len(feats)}", flush=True)
        if not feats:
            continue
        raw = np.array([float(f["properties"].get("confidence") or 0.0) for f in feats])
        regions = [str(f["properties"].get("region_name") or "") for f in feats]
        classes = [f["properties"]["class_name"] for f in feats]
        by_region = groups(regions)
        by_rc = groups([r + "\0" + c for r, c in zip(regions, classes)])
        pct = percentile(raw, by_rc)
        nbr = neighbor_counts(centers(feats), by_region, RADIUS_M)
        nbr_pct = percentile(nbr, by_rc)
        print(f"  соседей в {RADIUS_M:.0f} м: медиана {np.median(nbr):.0f}, "
              f"доля без соседей {(nbr == 0).mean():.2f}", flush=True)
        report("сырая уверенность", feats, raw, gt)
        report("ранг в участке", feats, pct, gt)
        report("ранг числа соседей", feats, nbr_pct, gt)
        report("ранг, если соседей ≥ 2", feats, np.where(nbr >= 2, pct, 0.0), gt)


if __name__ == "__main__":
    main()
