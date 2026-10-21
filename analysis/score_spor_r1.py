"""Балл спутника на шаге 1 м, только курганы, вне обучения."""
import json
import sys

sys.path.insert(0, r"C:\Sloi\expds-koz3_v0.2.0\expds-koz3")
from expds.metric import apply_thresholds, score, sweep  # noqa: E402

METRIC = r"C:\Sloi\files\materials\compute_metrics.py"
KEEP = {"kurgany_povrezhdennye", "kurgany_tselye"}


def filt(path: str) -> dict:
    fc = json.loads(open(path, encoding="utf-8").read())
    feats = [f for f in fc["features"]
             if (f.get("properties") or {}).get("class_name") in KEEP
             and (f.get("properties") or {}).get("markup_type") in (None, "sp_or")]
    return {"type": "FeatureCollection", "features": feats}


def main() -> None:
    print("читаю", flush=True)
    pred = filt(r"F:\expds3\work\spor_r1_oof.geojson")
    gt = filt(r"F:\expds3\work\gt_train.geojson")
    print(f"предсказаний {len(pred['features'])}, эталон {len(gt['features'])}", flush=True)
    res = sweep(pred, gt, METRIC)
    for cls, v in res["per_class"].items():
        if v["n_gt"]:
            print(f"{cls:24s} GT {v['n_gt']:5d} pred {v['n_pred']:6d} порог {v['threshold']:.4f} "
                  f"F1 {v['f1']:.3f}", flush=True)
    print(f"балл слоя {res['estimated_score']:.4f}", flush=True)
    kept = apply_thresholds(pred, res["thresholds"], METRIC)
    print(f"оставлено {len(kept['features'])}", flush=True)
    if len(kept["features"]) <= 25000:
        print(f"официальный {score(kept, gt, METRIC)['final_score']:.4f}", flush=True)


if __name__ == "__main__":
    main()
