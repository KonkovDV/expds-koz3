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
sat_frame_gate — W1-SATQ Q2 (docs/PREREG_W1_SATQ_AMEND1.md §C): on scenes >= min_scene_mpix (scene_mpix of the covering
  scene box that gives scene_drel), drop satellite detections with scene_drel > x and confidence < t2. Before thresholds.
W1-FRZ production flags (docs/W1_C1F_HANDOFF.md) — [post.w1_c1f] (main) / [post.w1_p2f] (comparison), enabled = false
  by default. resolve_final_flag(cfg) turns an enabled flag into [post].sat_center_prior + its own
  [thresholds_by_markup.sp_dfine] values (non-sp_dfine thresholds untouched); flag off → cfg returned as is.
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
    W1-COMBO (docs/PREREG_W1_COMBO.md) adds:
      frame_prior  {classes, d0, edge}: after the prior, post.frame_prior on the same groups (∪ nl_group) — FP1 on sp_dfine
    Feature without scene_drel keeps its confidence.
    """
    from . import post

    spec = dict(spec or {})
    spec.setdefault("groups", ["sp_dfine"])
    if not any(k in spec for k in ("s", "s_by_class", "only_without_lidar", "nl_group", "frame_prior")):
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
    fp = spec.get("frame_prior")
    if fp:
        out = post.frame_prior(out, {**dict(fp), "groups": sorted(groups | ({nl_group} if nl_group else set()))})
    return out


def box_mpix(box) -> float:
    """Mpix at 0.5 m of an EPSG:3857 scene box: (W·cos φ / 0.5)·(H·cos φ / 0.5)/1e6, φ = latitude of the box centre."""
    import math
    x0, y0, x1, y1 = box.bounds if hasattr(box, "bounds") else box
    c = math.cos(math.atan(math.sinh(((y0 + y1) / 2.0) / 6378137.0)))
    return (x1 - x0) * c / 0.5 * (y1 - y0) * c / 0.5 / 1e6


def annotate_scene_mpix(feats: list[dict], boxes_by_markup: dict[str, list]) -> list[dict]:
    """Field scene_mpix = box_mpix of the covering scene box with the smallest scene_drel (first on ties), per markup type.
    Same box choice as post.scene_drel; a feature outside all scenes gets no field."""
    for f in feats:
        boxes = boxes_by_markup.get(f.get("markup_type") or "")
        if not boxes or f.get("geom") is None or f["geom"].is_empty:
            continue
        pt = f["geom"].representative_point()
        best, bm = None, None
        for b in boxes:
            x0, y0, x1, y1 = b.bounds if hasattr(b, "bounds") else b
            hw, hh = (x1 - x0) / 2.0, (y1 - y0) / 2.0
            if hw <= 0 or hh <= 0 or not (x0 <= pt.x <= x1 and y0 <= pt.y <= y1):
                continue
            d = max(abs(pt.x - (x0 + hw)) / hw, abs(pt.y - (y0 + hh)) / hh)
            if best is None or d < best:
                best, bm = d, box_mpix((x0, y0, x1, y1))
        if bm is not None:
            f["scene_mpix"] = float(bm)
    return feats


SFG_CLASSES = ("kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha", "arkhitektury")


def sat_frame_gate(feats: list[dict], spec: dict | None = None) -> list[dict]:
    """W1-SATQ Q2, off unless [post].sat_frame_gate {groups=[sp_dfine], classes, min_scene_mpix=4, x, t2 | t2_cls}.
    t2_cls (COMP-FL, docs/bots/2026-11-02/coord-A/PREREG_COMP_FL.md) overrides t2 per class; a class absent from t2_cls
    falls back to t2, and without t2 it is left untouched.
    Removes a feature of the groups/classes with scene_mpix >= min_scene_mpix, scene_drel > x and confidence < t2.
    Features without scene_drel / scene_mpix are kept."""
    from . import post

    spec = dict(spec or {})
    groups = set(spec.get("groups") or ["sp_dfine"])
    classes = set(spec.get("classes") or SFG_CLASSES)
    a = float(spec.get("min_scene_mpix", 4.0)); x = float(spec["x"])
    t2_cls = {k: float(v) for k, v in (spec.get("t2_cls") or {}).items()}   # COMP-FL: per-class t2 (EN-relative form)
    t2 = float(spec["t2"]) if "t2" in spec else None
    if t2 is None and not t2_cls:
        raise ValueError("[post].sat_frame_gate needs t2 or t2_cls")
    out = []
    for f in feats:
        d, mp = f.get("scene_drel"), f.get("scene_mpix")
        tc = t2_cls.get(f.get("class_name"), t2)
        if (tc is not None and post._group_of(f) in groups and f.get("class_name") in classes and d is not None and mp is not None
                and float(mp) >= a and float(d) > x and float(f.get("confidence", 0.0)) < tc):
            continue
        out.append(f)
    return out


FINAL_FLAGS = ("w1_c1f", "w1_p2f")


def final_flag_on(cfg: dict) -> str | None:
    """Name of the enabled W1-FRZ production flag ([post.w1_c1f] / [post.w1_p2f]) or None. Both on → ValueError."""
    pc = (cfg or {}).get("post") or {}
    on = [k for k in FINAL_FLAGS if isinstance(pc.get(k), dict) and pc[k].get("enabled")]
    if len(on) > 1:
        raise ValueError(f"[post] {' and '.join(on)}: only one W1 sat-centre flag may be enabled")
    return on[0] if on else None


def resolve_final_flag(cfg: dict) -> dict:
    """cfg with the enabled W1-FRZ flag applied (shallow copies; the input is not modified); no flag → cfg itself.

    [post.<flag>]: enabled, groups (["sp_dfine"]), d0, s_by_class {class: s}, thresholds_sp_dfine {class: τ}.
    Sets [post].sat_center_prior = {groups, d0, s = 0, s_by_class} and replaces the listed classes of
    [thresholds_by_markup.sp_dfine]. A separate [post].sat_center_prior together with the flag → ValueError.
    The returned cfg carries the flag as enabled = false, resolved = true, so resolving it again is a no-op.
    """
    name = final_flag_on(cfg)
    if name is None:
        return cfg
    pc = dict(cfg.get("post") or {})
    if pc.get("sat_center_prior"):
        raise ValueError(f"[post.{name}] and [post].sat_center_prior are mutually exclusive")
    fl = pc[name]
    thr = {str(k): float(v) for k, v in (fl.get("thresholds_sp_dfine") or {}).items()}
    if not thr or not fl.get("s_by_class"):
        raise ValueError(f"[post.{name}] needs s_by_class and thresholds_sp_dfine")
    pc["sat_center_prior"] = {"groups": list(fl.get("groups") or ["sp_dfine"]), "d0": float(fl.get("d0", 0.0)), "s": 0.0,
                              "s_by_class": {str(k): float(v) for k, v in fl["s_by_class"].items()}}
    if fl.get("frame_prior"):   # optional FP1 after the centre prior (W1-COMBO; no flag ships it — COMBO FAILED)
        fp = dict(fl["frame_prior"])
        pc["sat_center_prior"]["frame_prior"] = {"classes": [str(c) for c in fp.get("classes") or []],
                                                 "d0": float(fp.get("d0", 0.5)), "edge": float(fp.get("edge", 0.5))}
    pc[name] = {**fl, "enabled": False, "resolved": True}   # idempotent: a resolved cfg is not resolved again
    by = dict(cfg.get("thresholds_by_markup") or {})
    by["sp_dfine"] = {**dict(by.get("sp_dfine") or {}), **thr}
    out = dict(cfg)
    out["post"], out["thresholds_by_markup"] = pc, by
    return out


# ---------------- W4 [post.w4_sat] (PREREG docs/bots/2026-11-02/bot-W4/c1fx_regate/PREREG.md), default OFF ----------------
W4_FLAG = "w4_sat"


def w4_flag_on(cfg: dict) -> bool:
    pc = (cfg or {}).get("post") or {}
    return isinstance(pc.get(W4_FLAG), dict) and bool(pc[W4_FLAG].get("enabled"))


def resolve_w4_flag(cfg: dict) -> dict:
    """cfg with an enabled [post.w4_sat] applied (shallow copies); off → cfg itself.

    [post.w4_sat]: enabled, groups (["sp_dfine"]), center {d0, s_by_class} → [post].sat_center_prior,
    frame {classes, d0, edge} → [post].sat_frame_prior (post.frame_prior on the groups), thresholds_sp_dfine {class: τ}
    replaces those classes of [thresholds_by_markup.sp_dfine]. Exclusive with the W1 flags / a separate sat_center_prior.
    """
    if not w4_flag_on(cfg):
        return cfg
    pc = dict(cfg.get("post") or {})
    if final_flag_on(cfg) or pc.get("sat_center_prior") or pc.get("sat_frame_prior"):
        raise ValueError("[post.w4_sat] is exclusive with w1_c1f / w1_p2f / sat_center_prior / sat_frame_prior")
    fl = pc[W4_FLAG]
    groups = list(fl.get("groups") or ["sp_dfine"])
    thr = {str(k): float(v) for k, v in (fl.get("thresholds_sp_dfine") or {}).items()}
    cen, fr = fl.get("center") or {}, fl.get("frame") or {}
    if not thr or not (cen or fr):
        raise ValueError("[post.w4_sat] needs thresholds_sp_dfine and center and/or frame")
    if cen:
        pc["sat_center_prior"] = {"groups": groups, "d0": float(cen.get("d0", 0.0)), "s": 0.0,
                                  "s_by_class": {str(k): float(v) for k, v in cen["s_by_class"].items()}}
    if fr:
        pc["sat_frame_prior"] = {"groups": groups, "classes": [str(c) for c in fr["classes"]],
                                 "d0": float(fr.get("d0", 0.5)), "edge": float(fr.get("edge", 0.5))}
    pc[W4_FLAG] = {**fl, "enabled": False, "resolved": True}
    by = dict(cfg.get("thresholds_by_markup") or {})
    by["sp_dfine"] = {**dict(by.get("sp_dfine") or {}), **thr}
    out = dict(cfg)
    out["post"], out["thresholds_by_markup"] = pc, by
    return out
