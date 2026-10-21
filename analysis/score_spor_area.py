"""Площадь предсказания спутника как фильтр: мелкий шум и поле целиком не курган.

Эталон спутника: повреждённые p5–p95 примерно 150–8600 м². Порог уверенности
после фильтра подбирается заново.
"""
import json
import sys

from shapely.geometry import shape

sys.path.insert(0, r"C:\Sloi\expds-koz3_v0.2.0\expds-koz3")
from expds.metric import sweep  # noqa: E402

W = r"F:\expds3\work"
METRIC = r"C:\Sloi\files\materials\compute_metrics.py"
CLASSES = {"kurgany_povrezhdennye", "kurgany_tselye", "arkhitektury"}
WINDOWS = [(0, 1e12), (30, 30000), (80, 12000), (150, 8000)]


def main() -> None:
    print("читаю", flush=True)
    pred = json.loads(open(W + r"\spor_all_oof.geojson", encoding="utf-8").read())
    gt = json.loads(open(W + r"\gt_train.geojson", encoding="utf-8").read())
    gt = {"type": "FeatureCollection", "features": [
        f for f in gt["features"]
        if (f.get("properties") or {}).get("markup_type") == "sp_or"
        and (f.get("properties") or {}).get("class_name") in CLASSES]}
    base = []
    for f in pred["features"]:
        p = f.get("properties") or {}
        if p.get("class_name") not in CLASSES:
            continue
        g = shape(f["geometry"])
        base.append((f, g.area))
    print(f"предсказаний {len(base)}, эталон {len(gt['features'])}", flush=True)
    for lo, hi in WINDOWS:
        feats = [f for f, a in base if lo <= a <= hi]
        fc = {"type": "FeatureCollection", "features": feats}
        res = sweep(fc, gt, METRIC)
        bits = " ".join(f"{c[:6]} {v['f1']:.3f}@{v['threshold']:.3f} n={v['n_pred']}"
                        for c, v in res["per_class"].items() if v["n_gt"])
        print(f"площадь {lo:.0f}–{hi:.0f}: осталось {len(feats):6d}  балл {res['estimated_score']:.4f}  {bits}",
              flush=True)


if __name__ == "__main__":
    main()
