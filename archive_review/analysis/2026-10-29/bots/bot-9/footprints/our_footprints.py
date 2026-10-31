#!/usr/bin/env python3
"""Охваты НАШИХ участков и семей honest-10 (EPSG:3857) по /workspace/data/train-expds-3.

Охват участка = union(границы растров по заголовку, bbox LAS/LAZ по заголовку, выпуклые оболочки разметки).
Семья = site_family() из bot-10/repo/expds/naming.py:108 (первое слово названия).
Фолды honest-10 = family_folds(regions, site_family, 10) из bot-10/repo/expds/metric.py:554 (тот же жадный алгоритм).
Читаются только заголовки. Запуск: OMP_NUM_THREADS=2 nice -n 10 bot-9/venv/bin/python bot-9/scan/our_footprints.py
"""
from __future__ import annotations
import argparse, json, os, re, sys, unicodedata
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "kaggle", "scan_inv"))
import importlib.util
spec = importlib.util.spec_from_file_location("inv", os.path.join(HERE, "..", "kaggle", "scan_inv", "koz3-scan-inv.py"))
inv = importlib.util.module_from_spec(spec); spec.loader.exec_module(inv)
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union


def parse_site_name(d):
    m = re.match(r"^(\d{3})_(.+)$", d)
    return m.group(2) if m else d


def site_family(d):  # копия bot-10/repo/expds/naming.py:108
    return re.split(r"[_\s]", parse_site_name(d), maxsplit=1)[0]


def family_folds(regions, family_of, k=None):  # копия bot-10/repo/expds/metric.py:554
    fams = {}
    for r in regions:
        fams.setdefault(family_of(r), set()).add(r)
    names = sorted(fams)
    if not k or k >= len(names):
        return [fams[f] for f in names]
    bins = [set() for _ in range(k)]; sizes = [0] * k
    for f in sorted(names, key=lambda f: (-len(fams[f]), f)):
        i = int(np.argmin(sizes)); bins[i] |= fams[f]; sizes[i] += len(fams[f])
    return [b for b in bins if b]


def region_footprint(rdir):
    utm = os.path.join(rdir, "UTM.json")
    rcrs = inv.crs_from_utm_json(utm)[0] if os.path.exists(utm) else None
    geoms, srcs, nobj, errs = [], defaultdict(list), defaultdict(int), []
    for dp, _, fn in os.walk(rdir):
        for f in fn:
            p = os.path.join(dp, f); kind = inv.file_kind(p)
            if kind not in ("raster", "pointcloud", "markup"):
                continue
            mod = inv.modality_of(p)
            try:
                if kind == "raster":
                    info = inv.read_raster(p)
                elif kind == "pointcloud":
                    info = inv.read_las_header(p); info["crs"] = rcrs
                else:
                    info = inv.read_markup(p, rcrs); nobj[info["canon_class"]] += info["n_features"]
                hull = info.pop("_hull", None)
                if kind == "raster" and not info.get("crs"):
                    info["crs"] = rcrs
                crs, bn = info.get("crs"), info.get("bounds_native")
                if not (crs and bn):
                    continue
                if kind == "markup" and hull is not None and not hull.is_empty:
                    hb = hull if hull.geom_type == "Polygon" else hull.buffer(0.5)
                    ring = list(hb.exterior.coords)
                else:
                    ring = inv.poly_ring(bn)
                g = Polygon(inv.ring_to(ring, crs, "EPSG:3857")).buffer(0)
                geoms.append(g); srcs[inv.SOURCE.get(mod) or "unknown"].append(g)
            except Exception as e:
                errs.append(f"{p}: {type(e).__name__}: {e}")
    fp = unary_union(geoms) if geoms else None
    return fp, sorted(srcs), dict(nobj), errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/workspace/data/train-expds-3")
    ap.add_argument("--out", default=os.path.join(HERE, "our_footprints_3857.geojson"))
    a = ap.parse_args()
    regs = sorted(d for d in os.listdir(a.data) if os.path.isdir(os.path.join(a.data, d)))
    regs = [unicodedata.normalize("NFC", d) for d in regs]
    folds = family_folds(regs, site_family, 10)
    fold_of = {r: i for i, s in enumerate(folds) for r in s}
    feats, fam_geoms, errs = [], defaultdict(list), []
    for d in regs:
        fp, srcs, nobj, e = region_footprint(os.path.join(a.data, d)); errs += e
        fam = site_family(d)
        props = dict(feature_type="our_region", id=d, family=fam, honest10_fold=fold_of[d], sources=";".join(srcs),
                     n_objects=sum(nobj.values()), **{f"n_{k}": v for k, v in nobj.items()}, footprint_known=fp is not None)
        feats.append({"type": "Feature", "properties": props, "geometry": mapping(fp) if fp is not None else None})
        if fp is not None:
            fam_geoms[fam].append(fp)
        print(d, fam, fold_of[d], None if fp is None else round(fp.area / 1e6, 3), len(e), flush=True)
    for fam in sorted({site_family(d) for d in regs}):
        members = [d for d in regs if site_family(d) == fam]
        g = unary_union(fam_geoms[fam]) if fam_geoms[fam] else None
        feats.append({"type": "Feature", "properties": dict(feature_type="our_family", id=fam, family=fam,
                      honest10_fold=fold_of[members[0]], n_regions=len(members), regions=members,
                      footprint_known=g is not None, n_regions_with_footprint=len(fam_geoms[fam])),
                      "geometry": mapping(g) if g is not None else None})
    json.dump({"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::3857"}},
               "features": feats}, open(a.out, "w"), ensure_ascii=False)
    open(a.out.replace(".geojson", "_errors.log"), "w").write("\n".join(errs) + "\n")
    print("regions", len(regs), "families", len(fam_geoms), "folds", len(folds), "errors", len(errs), "->", a.out)


if __name__ == "__main__":
    main()
