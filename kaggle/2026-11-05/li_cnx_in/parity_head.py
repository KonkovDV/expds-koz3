#!/usr/bin/env python3
"""Паритет прода с ядром (поправка 1 PREREG c5-in): курганы `expds.lidar_cnx_head` на участке трейна против OOF ядра.

    python parity_head.py LAS_DIR OOF_DIR OUT.json SITE:FOLD [SITE:FOLD ...]

LAS_DIR/<участок>/**/*.las и UTM.json — из бакета организаторов; OOF_DIR — выходы ядер c5-in (`oof_cnx_c5-in_f*.geojson`,
`weights_cnx_c5-in_f*.pt`). Участок берёт веса своего фолда (он был в нём тестом). Ядро считало на GPU в fp16, здесь CPU
FP32 — поэтому сравнение с допуском: пара — ближайший прод-курган того же класса ≤ 0,25 м; |Δ уверенности| ≤ 0,01.
Критерий паритета (до LB-пробы): среди детекций ядра с уверенностью ≥ 0,15 (отсечка гейта) нашли пару ≥ 99 %, и среди
прод-детекций ≥ 0,15 без пары в ядре ≤ 1 %.
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

KUR = ("kurgany_tselye", "kurgany_povrezhdennye")


def _xy(feats):
    from shapely.geometry import shape
    out = []
    for f in feats:
        g = f.get("geom") or shape(f["geometry"])
        c = g.centroid
        q = f.get("properties", f)
        out.append((c.x, c.y, q["class_name"], float(q["confidence"])))
    return out


def compare(kern, prod, floor=0.15, tol_m=0.25):
    rep = {"kernel_n": len(kern), "prod_n": len(prod)}
    by_cls = {c: np.array([(x, y, s) for x, y, cc, s in prod if cc == c], float).reshape(-1, 3) for c in KUR}
    used = {c: np.zeros(len(by_cls[c]), bool) for c in KUR}
    matched = miss = 0
    dscore = []
    for x, y, c, s in sorted(kern, key=lambda t: -t[3]):
        P = by_cls[c]
        if not len(P):
            miss += s >= floor
            continue
        d = np.hypot(P[:, 0] - x, P[:, 1] - y)
        d[used[c]] = np.inf
        j = int(np.argmin(d))
        if d[j] <= tol_m:
            used[c][j] = True
            dscore.append(abs(P[j, 2] - s))
            matched += s >= floor
        else:
            miss += s >= floor
    n_k = sum(s >= floor for *_, s in kern)
    extra = sum(int(((~used[c]) & (by_cls[c][:, 2] >= floor)).sum()) for c in KUR)
    n_p = sum(int((by_cls[c][:, 2] >= floor).sum()) for c in KUR)
    ds = np.array(dscore) if dscore else np.zeros(1)
    rep.update(kernel_ge_floor=int(n_k), matched_ge_floor=int(matched), missed_ge_floor=int(miss),
               prod_ge_floor=int(n_p), prod_extra_ge_floor=int(extra),
               match_rate=matched / n_k if n_k else 1.0, extra_rate=extra / n_p if n_p else 0.0,
               dscore_median=float(np.median(ds)), dscore_p99=float(np.percentile(ds, 99)), dscore_max=float(ds.max()))
    rep["PASS"] = rep["match_rate"] >= 0.99 and rep["extra_rate"] <= 0.01 and rep["dscore_p99"] <= 0.01
    return rep


def main():
    las_dir, oof_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    from expds import lidar_cnx as LC
    from expds import lidar_cnx_head as LH
    res = {}
    for tok in sys.argv[4:]:
        site, fold = tok.rsplit(":", 1)
        w = glob.glob(str(oof_dir / "**" / f"weights_cnx_c5-in_f{fold}.pt"), recursive=True)
        oof = glob.glob(str(oof_dir / "**" / f"oof_cnx_c5-in_f{fold}.geojson"), recursive=True)
        las = sorted(glob.glob(str(las_dir / site / "**" / "*.las"), recursive=True))
        utm = json.loads((las_dir / site / "UTM.json").read_text(encoding="utf-8"))
        m = re.search(r"EPSG:*\s*(\d+)", str(utm.get("crs")), re.I)
        crs = f"EPSG:{m.group(1)}" if m else None
        seg = LC.CnxSegmenter([w[0]], device="cpu", opts={"classes": list(KUR)}, amp=False)
        prod = LH.site_feats(seg, las, crs, {"classes": list(KUR)}, site)
        kern = [f for f in json.load(open(oof[0], encoding="utf-8"))["features"]
                if f["properties"]["region_name"] == site and f["properties"]["class_name"] in KUR]
        res[site] = compare(_xy(kern), _xy(prod))
        print(site, json.dumps(res[site], ensure_ascii=False), flush=True)
    res["PASS_all"] = all(v["PASS"] for v in res.values() if isinstance(v, dict))
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("PASS_all", res["PASS_all"])


if __name__ == "__main__":
    main()
