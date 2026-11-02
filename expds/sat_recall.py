"""W1-SATREC (bot-W1, 02.11, docs/PREREG_W1_SATREC.md): satellite recall levers. All OFF by default.

sat_equal_fusion — equal-weight fusion of the satellite detectors into one group (default "sp_fuse"):
  per region and class, candidates in descending confidence; each unassigned candidate becomes an anchor and
  absorbs unassigned candidates whose centroid is within radius_m ground metres of the anchor centroid, or lies
  inside the anchor geometry, or whose geometry contains the anchor centroid. Fused confidence = mean over the
  detectors able to emit the class of each detector's max confidence in the cluster (absent detector = 0).
  Geometry/properties = anchor's. Members are removed; non-members pass through unchanged.
  Geometries must be EPSG:3857 (ground metres = units × cos φ, one cos φ per region).
sat_center_prior — post.frame_prior form on the satellite group (scene_drel; d0 width of the untouched centre,
  edge = 1 − strength at the scene edge). Applied before thresholds.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

DEFAULT_MEMBERS = {"sp_or": "r18", "sp_focal": "r18", "sp_arch": "r18", "sp_focal2": "f2", "sp_dfine": "dfine"}
DEFAULT_CLASS_MEMBERS = {
    "kurgany_povrezhdennye": ["r18", "f2", "dfine"],
    "kurgany_tselye": ["r18", "f2", "dfine"],
    "gorodishcha": ["r18", "dfine"],
    "arkhitektury": ["r18", "dfine"],
    "fortifikatsii": ["dfine"],
}


def _group_of(f: dict) -> str:
    return f.get("threshold_group") or f.get("markup_type") or ""


def _member_of(f: dict, members: dict) -> str | None:
    g = _group_of(f)
    if g in members:
        return members[g]
    # production r18 outputs carry markup_type sp_or without threshold_group
    if f.get("threshold_group") is None and f.get("markup_type") == "sp_or" and "sp_or" in members:
        return members["sp_or"]
    return None


def sat_equal_fusion(feats: list[dict], spec: dict | None = None) -> list[dict]:
    import math

    import shapely
    from scipy.spatial import cKDTree

    spec = spec or {}
    members = dict(spec.get("members") or DEFAULT_MEMBERS)
    cmem = {k: list(v) for k, v in (spec.get("class_members") or DEFAULT_CLASS_MEMBERS).items()}
    radius = float(spec.get("radius_m", 8.0))
    out_group = str(spec.get("group", "sp_fuse"))
    out, by = [], defaultdict(list)
    for f in feats:
        m = _member_of(f, members)
        if m is None or f.get("class_name") not in cmem or m not in cmem[f["class_name"]]:
            if m is None:
                out.append(f)
            continue   # member detector of a class it cannot vote for -> dropped (not expected)
        by[(f.get("region_name") or "", f["class_name"])].append((f, m))
    for (reg, cls), lst in sorted(by.items(), key=lambda kv: kv[0]):
        mlist = cmem[cls]
        conf = np.array([float(f.get("confidence", 0.0)) for f, _ in lst])
        geoms = np.array([f["geom"] for f, _ in lst], dtype=object)
        cents = shapely.centroid(geoms)
        cx, cy = shapely.get_x(cents), shapely.get_y(cents)
        lat = math.atan(math.sinh(float(np.median(cy)) / 6378137.0))
        s = max(math.cos(lat), 1e-6)
        kd = cKDTree(np.c_[cx * s, cy * s])
        t_pts = shapely.STRtree(cents)
        t_geo = shapely.STRtree(geoms)
        order = np.lexsort((np.arange(len(lst)), -conf))
        assigned = np.zeros(len(lst), bool)
        for i in order:
            if assigned[i]:
                continue
            assigned[i] = True
            near = set(kd.query_ball_point((cx[i] * s, cy[i] * s), radius))
            near |= set(t_pts.query(geoms[i], predicate="intersects").tolist())
            near |= set(t_geo.query(cents[i], predicate="intersects").tolist())
            best = {lst[i][1]: conf[i]}
            for j in near:
                if not assigned[j]:
                    assigned[j] = True
                    mj = lst[j][1]
                    best[mj] = max(best.get(mj, 0.0), conf[j])
            fused = sum(best.get(m, 0.0) for m in mlist) / len(mlist)
            f0 = lst[i][0]
            out.append(dict(f0, confidence=float(fused), threshold_group=out_group, markup_type=f0.get("markup_type") or "sp_or",
                            fuse_n=len(best), fuse_anchor=lst[i][1]))
    return out


def sat_center_prior(feats: list[dict], spec: dict | None = None) -> list[dict]:
    from . import post

    spec = dict(spec or {})
    spec.setdefault("groups", ["sp_dfine"])
    return post.frame_prior(feats, spec)
