"""Балл пятен 8/16/28 м на слое спутника. Официальный расчёт — только на оставленных порогом."""
import json
import sys

sys.path.insert(0, r"C:\Sloi\expds-koz3_v0.2.0\expds-koz3")
from expds.metric import apply_thresholds, score, sweep  # noqa: E402

METRIC = r"C:\Sloi\files\materials\compute_metrics.py"
KEEP = {"kurgany_povrezhdennye", "kurgany_tselye"}


def filt(fc, markup=True):
    feats = []
    for f in fc["features"]:
        p = f.get("properties") or {}
        if p.get("class_name") not in KEEP:
            continue
        if markup and p.get("markup_type") not in (None, "sp_or"):
            continue
        feats.append(f)
    return {"type": "FeatureCollection", "features": feats}


def main() -> None:
    print("читаю", flush=True)
    pred = filt(json.loads(open(r"F:\expds3\work\blobs_spor_r8.geojson", encoding="utf-8").read()))
    gt = filt(json.loads(open(r"F:\expds3\work\gt_train.geojson", encoding="utf-8").read()))
    print(f"предсказаний {len(pred['features'])}, эталон {len(gt['features'])}", flush=True)
    res = sweep(pred, gt, METRIC)
    for cls, v in res["per_class"].items():
        if v["n_gt"]:
            print(f"{cls:24s} GT {v['n_gt']:5d} pred {v['n_pred']:6d} порог {v['threshold']:.4f} "
                  f"F1 {v['f1']:.3f}", flush=True)
    print(f"балл {res['estimated_score']:.4f}", flush=True)
    kept = apply_thresholds(pred, res["thresholds"], METRIC)
    print(f"оставлено {len(kept['features'])}", flush=True)
    if len(kept["features"]) <= 20000:
        official = score(kept, gt, METRIC)["final_score"]
        print(f"официальный {official:.4f}", flush=True)


if __name__ == "__main__":
    main()
