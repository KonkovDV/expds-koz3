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
    """Scene-centre prior on satellite group(s) (default sp_dfine), before thresholds. Off unless [post].sat_center_prior.

    Legacy spec (d0, edge) → post.frame_prior unchanged (W1-SATREC). W1-SATREC2 (docs/PREREG_W1_SATREC2.md) adds:
      s            strength, m = max(0, 1 − s·clip((d − d0)/(1 − d0), 0, 1)); s may exceed 1 (m hits 0 before the edge)
      s_by_class   {class_name: s} overrides s per class (e.g. kurgany_tselye own strength)
      only_without_lidar  apply only in regions whose input has no markup "li" feature
      nl_group     relabel threshold_group of the prior groups to this name in lidar-free regions (own thresholds)
    Feature without scene_drel keeps its confidence.
    """
    from . import post

    spec = dict(spec or {})
    spec.setdefault("groups", ["sp_dfine"])
    if not any(k in spec for k in ("s", "s_by_class", "only_without_lidar", "nl_group")):
        return post.frame_prior(feats, spec)
    groups = set(spec["groups"])
    d0 = min(max(float(spec.get("d0", 0.0)), 0.0), 0.999)
    s_def = max(float(spec.get("s", 0.0)), 0.0)
    s_cls = {str(k): max(float(v), 0.0) for k, v in (spec.get("s_by_class") or {}).items()}
    only_nl = bool(spec.get("only_without_lidar"))
    nl_group = spec.get("nl_group") or None
    lidar = {f.get("region_name") for f in feats if f.get("markup_type") == "li"}
    out = []
    for f in feats:
        g = post._group_of(f)
        if g not in groups or (only_nl and f.get("region_name") in lidar):
            out.append(f)
            continue
        f2 = dict(f)
        if nl_group and f.get("region_name") not in lidar:
            f2["threshold_group"] = nl_group
        d = f.get("scene_drel")
        s = s_cls.get(f.get("class_name"), s_def)
        if d is not None and s > 0 and float(d) > d0:
            m = max(0.0, 1.0 - s * min((float(d) - d0) / (1.0 - d0), 1.0))
            f2["confidence"] = float(f.get("confidence", 0.0)) * m
        out.append(f2)
    return out
