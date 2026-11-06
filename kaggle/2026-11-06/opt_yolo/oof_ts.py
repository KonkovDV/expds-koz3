#!/usr/bin/env python3
"""OOF прод-кодом: веса ultralytics → TorchScript → expds.sp_yolo.Detector на окнах фолда (паритет с продом).

    python oof_ts.py WEIGHTS.pt DATASET_DIR FOLD OUT.jsonl [--imgsz 640] [--device cpu]

Окна выгрузки уже JPEG q95, как после jpeg_roundtrip прода. Выход — формат yolo_oof.py: окно, xyxy в пикселях окна,
conf, cls. Порог уверенности на выходе — 0,01 (conf_min), NMS по классам iou 0,6 — как DEFAULTS sp_yolo.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("weights", type=Path)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("fold", type=int)
    ap.add_argument("out", type=Path)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    from PIL import Image
    from ultralytics import YOLO

    from expds import sp_yolo as SY
    ts = a.weights.with_suffix(".torchscript")
    if not ts.exists():
        ts = Path(YOLO(str(a.weights)).export(format="torchscript", imgsz=a.imgsz, device=a.device))
    det = SY.Detector(ts, device=a.device, imgsz=a.imgsz)
    val = [a.dataset / l.strip()[2:] for l in open(a.dataset / f"val_f{a.fold}.txt", encoding="utf-8") if l.strip()]
    n = 0
    with open(a.out, "w", encoding="utf-8") as fo:
        for i in range(0, len(val), 8):
            part = val[i:i + 8]
            tiles = [np.asarray(Image.open(p).convert("RGB")) for p in part]
            for p, (xyxy, conf, cls) in zip(part, det.predict(tiles)):
                fo.write(json.dumps({"tile": p.stem, "xyxy": np.round(xyxy, 1).tolist(),
                                     "conf": np.round(conf, 5).tolist(), "cls": cls.astype(int).tolist()}) + "\n")
                n += len(conf)
    print(json.dumps({"tiles": len(val), "boxes": n, "torchscript": str(ts)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
