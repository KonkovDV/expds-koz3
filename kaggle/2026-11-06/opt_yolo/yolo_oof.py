#!/usr/bin/env python3
"""OOF детектора окон спутника → объекты участков в EPSG:3857 (план OPTICS_DETECTOR_PLAN §3.5–3.6).

    python yolo_oof.py DATASET_DIR OUT.geojson OOF1.jsonl [OOF2.jsonl ...] [--min-conf 0.01]

- Окно → снимок → мир: index.csv (окно, участок, снимок, x0, y0) и meta/<участок>__<снимок>.json (CRS, аффинное
  преобразование снимка при шаге выгрузки). Центр рамки → точка в CRS участка → EPSG:3857.
- Дубли с перекрывающихся окон и видов: по участку и классу жадно по убыванию уверенности снимаются детекции ближе
  радиуса класса (курганы 6 м, архитектура 6 м, форты 10 м, городища 40 м).
- Выход как у сдачи: курганы — круг 0,5 м, прочие классы — диск 0,05 м в центре (`[post.micro_circles]`).
  markup_type sp_or, threshold_group sp_yolo, поле fold — фолд участка.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

CLASSES = ["kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha", "arkhitektury", "fortifikatsii"]
RADIUS_M = {"kurgany_povrezhdennye": 6.0, "kurgany_tselye": 6.0, "arkhitektury": 6.0, "fortifikatsii": 10.0,
            "gorodishcha": 40.0}
CIRCLE_M = {"kurgany_povrezhdennye": 0.5, "kurgany_tselye": 0.5}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dataset", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("oof", type=Path, nargs="+")
    ap.add_argument("--min-conf", type=float, default=0.01)
    a = ap.parse_args()
    from affine import Affine
    from pyproj import Transformer
    from shapely.geometry import Point, mapping

    idx = {r["tile"]: r for r in csv.DictReader(open(a.dataset / "index.csv", encoding="utf-8"))}
    metas: dict = {}
    pts: dict = {}
    for p in a.oof:
        for line in open(p, encoding="utf-8"):
            rec = json.loads(line)
            r = idx.get(rec["tile"])
            if r is None:
                continue
            key = f"{r['site']}__{Path(r['image']).stem}"
            if key not in metas:
                metas[key] = json.loads((a.dataset / "meta" / f"{key}.json").read_text(encoding="utf-8"))
            m = metas[key]
            tr = Affine(*m["transform"])
            x0, y0 = float(r["x0"]), float(r["y0"])
            for (bx0, by0, bx1, by1), c, k in zip(rec["xyxy"], rec["conf"], rec["cls"]):
                if c < a.min_conf:
                    continue
                X, Y = tr * (x0 + (bx0 + bx1) / 2, y0 + (by0 + by1) / 2)
                pts.setdefault((r["site"], int(k)), []).append((float(c), X, Y, m["crs"], int(r["fold"])))
    feats = []
    tfs: dict = {}
    for (site, k), lst in sorted(pts.items()):
        name = CLASSES[k]
        lst.sort(key=lambda t: -t[0])
        keep: list = []
        rad = RADIUS_M[name]
        for c, X, Y, crs, fold in lst:
            if any((X - kx) ** 2 + (Y - ky) ** 2 < rad * rad for _, kx, ky, _, _ in keep):
                continue
            keep.append((c, X, Y, crs, fold))
        for c, X, Y, crs, fold in keep:
            if crs not in tfs:
                tfs[crs] = Transformer.from_crs(crs, "EPSG:3857", always_xy=True).transform
            gx, gy = tfs[crs](X, Y)
            r_m = CIRCLE_M.get(name, 0.05)
            g = Point(gx, gy).buffer(r_m / max(np.cos(np.arctan(np.sinh(gy / 6378137.0))), 1e-6), quad_segs=4)
            feats.append({"type": "Feature", "geometry": mapping(g),
                          "properties": {"class_name": name, "confidence": round(c, 5), "region_name": site,
                                         "markup_type": "sp_or", "threshold_group": "sp_yolo", "fold": fold,
                                         "crs": "EPSG:3857"}})
    a.out.write_text(json.dumps({"type": "FeatureCollection", "features": feats}, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"features": len(feats), "sites": len({f["properties"]["region_name"] for f in feats})}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
