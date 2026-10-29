#!/usr/bin/env python3
"""Raw zero-shot quality of ADAF detections vs GT (no stack): per class, GT restricted to lidar coverage
(GT centroid on a valid SLRM pixel). A detection hits a GT if its box centre is inside the GT polygon
(greedy one-to-one by score within class). Reports n_gt, recall@0.05/0.3/0.5, precision@0.5, AP (all-point).
Precision for a class uses all detections of the mapped model (barrow->povr|tselye, enclosure->gor|fort);
"union" counts a hit against either mapped class.
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import shapely
import shapely.ops
from shapely.geometry import shape
from pyproj import Transformer

MAP = {"OD_barrow_sd": ["kurgany_povrezhdennye", "kurgany_tselye"],
       "OD_enclosure_sd": ["gorodishcha", "fortifikatsii"]}


def ap_of(scores, hits, n_gt):
    if n_gt == 0 or len(scores) == 0:
        return 0.0
    o = np.argsort(-np.asarray(scores)); h = np.asarray(hits)[o].astype(float)
    tp = np.cumsum(h); prec = tp / np.arange(1, len(h) + 1); rec = tp / n_gt
    mprec = np.maximum.accumulate(prec[::-1])[::-1]
    r_prev = np.concatenate([[0], rec[:-1]])
    return float(np.sum((rec - r_prev) * mprec))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dets", required=True); ap.add_argument("--gt", required=True)
    ap.add_argument("--slrm", default=str(Path(__file__).resolve().parents[3] / "adaf" / "slrm"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    gt = defaultdict(list)
    for f in json.load(open(a.gt))["features"]:
        p = f["properties"]; gt[(p["region_name"], p["class_name"])].append(f["geometry"])
    rep = {}
    for mdl, classes in MAP.items():
        files = sorted((Path(a.dets) / mdl).glob("*.json"))
        if not files:
            continue
        acc = {c: {"scores": [], "hits": [], "n_gt": 0, "hit_any": defaultdict(int)} for c in classes + ["union"]}
        n_det = 0
        for fp in files:
            d = json.load(open(fp)); reg = d["region"]
            z = np.load(Path(a.slrm) / f"{reg}.npz")
            crs = d["crs"]; crs = crs if isinstance(crs, str) else crs.get("crs")
            epsg = "EPSG:" + crs.split(":")[-1]
            to_utm = Transformer.from_crs("EPSG:3857", epsg, always_xy=True)
            valid, left, top, res = z["valid"], float(z["left"]), float(z["top"]), float(z["res"])
            dets = sorted(d["dets"], key=lambda e: -e["score"]); n_det += len(dets)
            pts = shapely.points([e["x"] for e in dets], [e["y"] for e in dets]) if dets else []
            polys_by_c = {}
            for c in classes:
                ps = []
                for g in gt.get((reg, c), []):
                    s = shapely.ops.transform(lambda x, y, zz=None: to_utm.transform(x, y), shape(g))
                    cx, cy = s.representative_point().coords[0]
                    i, j = int((top - cy) / res), int((cx - left) / res)
                    if 0 <= i < valid.shape[0] and 0 <= j < valid.shape[1] and valid[i, j]:
                        ps.append(s)
                polys_by_c[c] = ps
            for c in classes + ["union"]:
                ps = polys_by_c[c] if c != "union" else polys_by_c[classes[0]] + polys_by_c[classes[1]]
                acc[c]["n_gt"] += len(ps)
                tree = shapely.STRtree(ps) if ps else None
                used = set()
                for k, e in enumerate(dets):
                    hit = 0
                    if tree is not None:
                        for gi in tree.query(pts[k], predicate="within"):
                            if gi not in used:
                                used.add(gi); hit = 1; break
                    acc[c]["scores"].append(e["score"]); acc[c]["hits"].append(hit)
        r = {"n_det": n_det}
        for c, v in acc.items():
            s = np.asarray(v["scores"]); h = np.asarray(v["hits"])
            row = {"n_gt": v["n_gt"], "AP": round(ap_of(s, h, v["n_gt"]), 4)}
            for t in (0.05, 0.3, 0.5):
                m = s >= t
                row[f"recall@{t}"] = round(float(h[m].sum() / max(v["n_gt"], 1)), 4)
                row[f"prec@{t}"] = round(float(h[m].mean()) if m.any() else 0.0, 4)
                row[f"n_det@{t}"] = int(m.sum())
            r[c] = row
        rep[mdl] = r
    Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(json.dumps(rep, ensure_ascii=False))


if __name__ == "__main__":
    main()
