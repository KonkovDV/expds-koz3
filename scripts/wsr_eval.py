#!/usr/bin/env python3
"""Within-site rank stop metrics (PROMPT_WITHIN_SITE_RANK_20261031, 5fd33ce).

Peaks = output of `scripts/nn_oof.py --weights <map> --markup sp_or` (circles of the centre map; confidence = peak
height; floor 0.05, min distance 12 m). GT = gt_train_v2 sp_or kurgans. Site = region. Family = expds.naming.site_family.

For every GT kurgan of class c on a site:
  * in-poly peak = peak of class c whose centre lies inside the GT polygon; best-in-GT = loudest of them;
  * "kept" if best-in-GT >= submission threshold (0.3645 damaged / 0.3725 intact), "quiet" if below, "none" if no in-poly peak;
  * louder FP = same-class peaks of the site louder than best-in-GT that lie in NO GT polygon of class c;
  * louder TP = same-class louder peaks inside some GT polygon of class c.
Quiet set (598-like) is defined on the REFERENCE map (old spor_focal_r18) and re-measured on the new map for the same GTs.
On the new map a GT without in-poly peak counts all same-class FPs of the site as louder (conservative)."""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import shapely
from shapely.geometry import shape

K = ("kurgany_povrezhdennye", "kurgany_tselye")
TAU = {"kurgany_povrezhdennye": 0.3645, "kurgany_tselye": 0.3725}


def site_family(region: str) -> str:
    try:
        from expds.naming import site_family as sf
        return sf(region)
    except Exception:
        import re
        name = region.split("_", 1)[1] if region[:3].isdigit() else region
        return re.split(r"[_\s]", name, maxsplit=1)[0]


def load_gt(path, regions=None):
    g = json.load(open(path, encoding="utf-8"))
    out = defaultdict(list)  # (region, cls) -> [polygon 3857]
    for f in g["features"]:
        p = f["properties"]
        if p.get("markup_type") != "sp_or" or p.get("class_name") not in K: continue
        if regions is not None and p.get("region_name") not in regions: continue
        out[(p["region_name"], p["class_name"])].append(shape(f["geometry"]))
    return out


def load_peaks(path):
    g = json.load(open(path, encoding="utf-8"))
    out = defaultdict(lambda: ([], []))  # (region, cls) -> (points, conf)
    for f in g["features"]:
        p = f["properties"]
        if p.get("class_name") not in K or p.get("markup_type", "sp_or") != "sp_or": continue
        c = shape(f["geometry"]).centroid
        out[(p["region_name"], p["class_name"])][0].append((c.x, c.y)); out[(p["region_name"], p["class_name"])][1].append(float(p["confidence"]))
    return {k: (np.asarray(v[0], float).reshape(-1, 2), np.asarray(v[1], float)) for k, v in out.items()}


def per_gt(gt, peaks):
    """-> list of dict per GT (key=(region, cls, idx))."""
    rows = []
    for (reg, cls), polys in sorted(gt.items()):
        xy, conf = peaks.get((reg, cls), (np.zeros((0, 2)), np.zeros(0)))
        pts = shapely.points(xy) if len(xy) else np.array([], dtype=object)
        inside = np.zeros((len(polys), len(xy)), bool)
        for i, poly in enumerate(polys):
            if len(xy): inside[i] = shapely.contains_xy(poly, xy[:, 0], xy[:, 1])
        in_any = inside.any(0) if len(polys) else np.zeros(len(xy), bool)
        site_max = float(conf.max()) if len(conf) else float("nan")
        for i in range(len(polys)):
            ix = np.flatnonzero(inside[i])
            best = float(conf[ix].max()) if len(ix) else None
            b = best if best is not None else -1.0
            louder = conf > b
            rows.append(dict(region=reg, family=site_family(reg), cls=cls, gi=i, n_peaks=int(len(conf)),
                             best=best, state=("none" if best is None else "kept" if best >= TAU[cls] else "quiet"),
                             fp_louder=int((louder & ~in_any).sum()), tp_louder=int((louder & in_any).sum()),
                             rank=int(louder.sum()) + 1, gap_site_max=(site_max - b) if best is not None else None))
    return rows


def summary(rows):
    st = defaultdict(int)
    for r in rows: st[r["state"]] += 1
    q = [r for r in rows if r["state"] == "quiet"]
    med = lambda k: float(np.median([r[k] for r in q])) if q else None
    return dict(n_gt=len(rows), inside=st["kept"] + st["quiet"], kept=st["kept"], quiet=st["quiet"], none=st["none"],
                quiet_median_fp_louder=med("fp_louder"), quiet_median_tp_louder=med("tp_louder"), quiet_median_rank=med("rank"),
                quiet_median_site_peaks=med("n_peaks"), quiet_median_gap=med("gap_site_max"))


def compare(ref_rows, new_rows):
    """Same GTs: quiet on the reference map -> their numbers on the new map."""
    new = {(r["region"], r["cls"], r["gi"]): r for r in new_rows}
    q = [r for r in ref_rows if r["state"] == "quiet"]
    was = [r["fp_louder"] for r in q]; now = [new[(r["region"], r["cls"], r["gi"])] for r in q]
    became = [n["fp_louder"] for n in now]
    with_peak = [n for n in now if n["state"] != "none"]
    return dict(n_quiet_ref=len(q), median_fp_louder_was=float(np.median(was)) if was else None,
                median_fp_louder_became=float(np.median(became)) if became else None,
                median_fp_louder_became_with_inpoly_peak=float(np.median([n["fp_louder"] for n in with_peak])) if with_peak else None,
                quiet_ref_now_no_inpoly_peak=sum(n["state"] == "none" for n in now),
                quiet_ref_now_kept=sum(n["state"] == "kept" for n in now),
                median_gap_to_site_max_was=float(np.median([r["gap_site_max"] for r in q])) if q else None,
                median_gap_to_site_max_became=float(np.median([n["gap_site_max"] for n in with_peak])) if with_peak else None,
                median_rank_was=float(np.median([r["rank"] for r in q])) if q else None,
                median_rank_became=float(np.median([n["rank"] for n in now])) if now else None)


def damaged_f1(pred_path, gt_path, regions, metric_path):
    from expds import metric as M
    g = json.load(open(gt_path, encoding="utf-8"))
    gt = {"type": "FeatureCollection", "features": [f for f in g["features"] if f["properties"].get("region_name") in regions
          and f["properties"].get("markup_type") == "sp_or" and f["properties"].get("class_name") in K]}
    fc = json.load(open(pred_path, encoding="utf-8")); fs = []
    for f in fc["features"]:
        p = dict(f["properties"]); p.pop("threshold_group", None); p["markup_type"] = "sp_or"
        if p.get("region_name") in regions and p.get("class_name") in K: fs.append({**f, "properties": p})
    cur = M.build_curves({"type": "FeatureCollection", "features": fs}, gt, metric_path); out = {}
    for c in K:
        cv = cur[c]; taus, f1 = M.curve_table(cv); i = int(np.argmax(f1)) if len(f1) else 0
        kept = cv.conf >= TAU[c]; tp = int(cv.hit[kept].sum())
        out[c] = dict(n_gt=int(cv.n_gt), thr=TAU[c], kept=int(kept.sum()), tp=tp,
                      f1_at_sub_thr=float(M._f1(tp, int(kept.sum()) - tp, cv.n_gt - tp)),
                      best_thr=float(taus[i]) if len(f1) else None, f1_best=float(f1[i]) if len(f1) else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True); ap.add_argument("--ref", required=True, help="peaks geojson of the old map")
    ap.add_argument("--new", default=None); ap.add_argument("--family", default=None, help="restrict to this family")
    ap.add_argument("--metric", default=None); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    gt = load_gt(a.gt)
    if a.family: gt = {k: v for k, v in gt.items() if site_family(k[0]) == a.family}
    regions = sorted({k[0] for k in gt})
    ref_rows = per_gt(gt, load_peaks(a.ref))
    fam = defaultdict(int)
    for r in ref_rows:
        if r["state"] == "quiet": fam[r["family"]] += 1
    res = dict(family=a.family, regions=len(regions), ref=summary(ref_rows),
               quiet_by_family=dict(sorted(fam.items(), key=lambda x: -x[1])))
    if a.new:
        new_rows = per_gt(gt, load_peaks(a.new)); res["new"] = summary(new_rows); res["compare"] = compare(ref_rows, new_rows)
    if a.metric:
        res["f1_ref"] = damaged_f1(a.ref, a.gt, set(regions), a.metric)
        if a.new: res["f1_new"] = damaged_f1(a.new, a.gt, set(regions), a.metric)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8"); print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
