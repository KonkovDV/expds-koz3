#!/usr/bin/env python3
"""Offline proxy for the li "peak/point" decoding: rewrite ONLY lidar-net candidates in the family JSONL stack.

Raw per-pixel probability maps of our li nets are not on the box, so the map-level decoder
(expds/li_decode.py) cannot be evaluated here. What can be evaluated from existing polygons:

  D1 gf_point  gor/fort polygons of lidar-net groups -> circle of R_OUT m (true metres) at the polygon's
               representative point; confidence unchanged.
  D2 kur_dual  povr/tselye twins (same centre, same region, same group) of lidar-net groups: keep the
               larger-confidence class; keep the other only if its share conf/(sum) >= DUAL.
  D3 combo     D2 + area floor on gor/fort polygons (true m^2) + D1.

Everything else (other groups, all non-lidar sources) is copied byte-for-byte.
Thresholds are NOT touched here: they are refit per fold by honest_counts downstream.

    python scripts/li_decode_stack.py --src FAM_DIR --dst OUT_DIR --variant D1
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from shapely.geometry import mapping, shape

R_EARTH = 6378137.0
KUR = ("kurgany_povrezhdennye", "kurgany_tselye")
GF = ("gorodishcha", "fortifikatsii")
KUR_GROUPS = {"li", "li_dev", "li_mosaic"}           # lidar nets that emit kurgan twins
GF_GROUPS = {"li", "li_coarse", "li_mask", "li_fort"}  # lidar nets that emit gor/fort masks (gor_phys is physics, not a net)
R_OUT = 1.0
DUAL = 0.3
MIN_AREA = {"gorodishcha": 150.0, "fortifikatsii": 7.5}   # = 600 / 30 px at 0.5 m


def group(pr: dict) -> str:   # same rule as expds.metric.threshold_group
    return str(pr.get("threshold_group") or pr.get("markup_type") or "")


def merc_k(y: float) -> float:
    """EPSG:3857 scale factor at northing y (1 true metre = k map units)."""
    lat = 2.0 * math.atan(math.exp(y / R_EARTH)) - math.pi / 2.0
    return 1.0 / math.cos(lat)


def true_area(g) -> float:
    return g.area / merc_k(g.centroid.y) ** 2


def to_point_circle(f: dict) -> dict:
    g = shape(f["geometry"])
    p = g.representative_point()
    c = p.buffer(R_OUT * merc_k(p.y), quad_segs=8)
    return {**f, "geometry": mapping(c)}


def transform(feats: list[dict], variant: str) -> tuple[list[dict], dict]:
    do_point = variant in ("D1", "D3")
    do_dual = variant in ("D2", "D3")
    do_floor = variant == "D3"
    st = {"gf_point": 0, "gf_floor_drop": 0, "kur_drop": 0, "kur_twins": 0}
    out: list[dict] = []
    twins: dict[tuple, list[int]] = {}
    for f in feats:
        pr = f.get("properties") or {}
        g, c = group(pr), pr.get("class_name")
        if c in GF and g in GF_GROUPS and (do_point or do_floor):
            if do_floor and true_area(shape(f["geometry"])) < MIN_AREA[c]:
                st["gf_floor_drop"] += 1
                continue
            if do_point:
                f = to_point_circle(f)
                st["gf_point"] += 1
        if do_dual and c in KUR and g in KUR_GROUPS:
            ctr = shape(f["geometry"]).centroid
            key = (g, pr.get("region_name"), round(ctr.x, 2), round(ctr.y, 2))
            twins.setdefault(key, []).append(len(out))
        out.append(f)
    if do_dual:
        drop = set()
        for idx in twins.values():
            cls = {out[i]["properties"]["class_name"] for i in idx}
            if len(idx) != 2 or cls != set(KUR):
                continue
            st["kur_twins"] += 1
            a, b = idx
            ca, cb = (float(out[i]["properties"]["confidence"]) for i in idx)
            lo = b if cb < ca else a          # tie -> drop the first (povr/tselye order is arbitrary but fixed)
            if min(ca, cb) / max(ca + cb, 1e-12) < DUAL:
                drop.add(lo)
        st["kur_drop"] = len(drop)
        out = [f for i, f in enumerate(out) if i not in drop]
    return out, st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--variant", choices=["D1", "D2", "D3"], required=True)
    a = ap.parse_args()
    dst = Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    tot: dict[str, int] = {}
    for src in sorted(Path(a.src).glob("*.jsonl")):
        with open(src, encoding="utf-8") as fh:
            feats = [json.loads(line) for line in fh]
        out, st = transform(feats, a.variant)
        for k, v in st.items():
            tot[k] = tot.get(k, 0) + v
        tmp = dst / (src.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fo:
            for f in out:
                fo.write(json.dumps(f, ensure_ascii=False) + "\n")
        os.replace(tmp, dst / src.name)
        print(f"{src.stem}\t{len(feats)}->{len(out)}\t{st}", flush=True)
        del feats, out
    (dst / "_transform.json").write_text(json.dumps({"variant": a.variant, "src": a.src, **tot}, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
    print(json.dumps(tot))


if __name__ == "__main__":
    main()
