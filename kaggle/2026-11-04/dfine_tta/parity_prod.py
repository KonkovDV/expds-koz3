#!/usr/bin/env python3
"""Сверка прод-пути TTA D-FINE с ядром koz3-dfine-tta на одном растре (PREREG, шаг 1 «если PASS»).

Прод: expds.sp_dfine.Scene + expds.orsp_tta.detect_dfine (виды d8, слияние nms), веса фолда из каталога HF.
Ядро: oof_T8N_f{k}.geojson сухого или Kaggle-прогона на том же растре и тех же весах.
Сравниваются центры и уверенности эллипсов того же класса (EPSG:3857). Печатает сводку, код 1 при расхождении.

    python kaggle/2026-11-04/dfine_tta/parity_prod.py --tif RASTER --crs EPSG:32644 --weights DIR/dfine_f2 \
        --kernel-geojson OUT/oof_T8N_f2.geojson --region 138_Курганы_9 [--fuse nms|wbf]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tif", required=True, type=Path)
    ap.add_argument("--crs", required=True)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--kernel-geojson", required=True, type=Path)
    ap.add_argument("--region", required=True)
    ap.add_argument("--fuse", default="nms", choices=("nms", "wbf"))
    ap.add_argument("--tol-m", type=float, default=0.05, help="допуск центра, м")
    ap.add_argument("--tol-score", type=float, default=1e-3)
    a = ap.parse_args()

    import torch
    from pyproj import Transformer
    from scipy.spatial import cKDTree
    from shapely.geometry import shape
    from transformers import DFineForObjectDetection

    from expds import orsp_tta, sp_dfine

    st = sp_dfine.settings({"sp_dfine": {"fp16": False}})
    plan = orsp_tta.dfine_plan({"orsp_tta": {"enabled": True, "dfine_views": "d8", "dfine_fuse": a.fuse}})
    assert plan is not None and len(plan["views"]) == 8, plan
    dev = torch.device("cpu")
    model = DFineForObjectDetection.from_pretrained(str(a.weights)).eval()
    scene = sp_dfine.Scene(a.tif, a.crs, float(st["res_m"]))
    try:
        fused, done, total = orsp_tta.detect_dfine([model], scene, a.tif, a.crs, st, dev, lambda: True, plan)
        B, L, C = fused[0]
        to3857 = Transformer.from_crs(a.crs, "EPSG:3857", always_xy=True).transform
        prod = []
        for (x0, y0, x1, y1), s, c in zip(B, L, C):
            gx, gy = scene.T * ((x0 + x1) / 2, (y0 + y1) / 2)
            prod.append((int(c), *to3857(gx, gy), float(s)))
    finally:
        scene.close()

    classes = ["kurgany_tselye", "kurgany_povrezhdennye", "gorodishcha", "fortifikatsii", "arkhitektury"]
    ker = []
    for f in json.loads(a.kernel_geojson.read_text(encoding="utf-8"))["features"]:
        q = f["properties"]
        if q["region_name"] != a.region:
            continue
        c = shape(f["geometry"]).centroid
        ker.append((classes.index(q["class_name"]), c.x, c.y, float(q["confidence"])))

    res = {"prod": len(prod), "kernel": len(ker), "windows": [done, total]}
    dd, ds = [], []
    for k in range(len(classes)):
        p = np.array([r[1:] for r in prod if r[0] == k]).reshape(-1, 3)
        q = np.array([r[1:] for r in ker if r[0] == k]).reshape(-1, 3)
        if len(p) and len(q):
            d, i = cKDTree(p[:, :2]).query(q[:, :2])
            dd += list(d)
            ds += list(np.abs(p[i, 2] - q[:, 2]))
    dd, ds = np.array(dd), np.array(ds)
    res.update({"centre_m_max": float(dd.max()) if dd.size else None, "centre_m_p99": float(np.percentile(dd, 99)) if dd.size else None,
                "score_max": float(ds.max()) if ds.size else None, "score_p99": float(np.percentile(ds, 99)) if ds.size else None})
    ok = res["prod"] == res["kernel"] and dd.size and res["centre_m_p99"] <= a.tol_m and res["score_p99"] <= a.tol_score
    res["ok"] = bool(ok)
    print(json.dumps(res, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
