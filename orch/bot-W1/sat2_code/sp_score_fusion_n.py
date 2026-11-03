"""SIDE patch (not shipped): post.sp_score_fusion generalised from exactly-2 to N sat groups (PROMPT_SAT2 §4).
For a kurgan (povr|tselye) of group g ∈ groups: o_h = max kurgan conf (either class) of group h ≠ g within radius_m
ground metres on the same region, else 0 (absent member = 0); new conf = (conf + Σ_h o_h) / N.
N = len(groups) (denom="all", default → exactly equals the old 2-group function for groups of length 2), or
N = number of groups with ≥1 kurgan candidate in that region (denom="region_active": e.g. sp_focal2 only on no-LAS regions,
so it does not dilute LAS regions). No objects added/removed except sp_partner_only. Apply BEFORE thresholds (gen_floor)."""
from __future__ import annotations
from collections import defaultdict

SP_FUSION_GROUPS = ("sp_focal", "sp_focal2")
SP_FUSION_CLASSES = ("kurgany_povrezhdennye", "kurgany_tselye")


def _ground_scale(y: float) -> float:
    import math
    return math.cos(math.atan(math.sinh(y / 6378137.0)))


def sp_score_fusion_n(feats: list[dict], spec: dict | None = None) -> list[dict]:
    import numpy as np
    from scipy.spatial import cKDTree
    from shapely.geometry import shape
    cfg = spec or {}
    radius = float(cfg.get("radius_m", 8.0))
    groups = tuple(cfg.get("groups", SP_FUSION_GROUPS))
    denom = cfg.get("denom", "all")
    if len(groups) < 2 or len(set(groups)) != len(groups):
        raise ValueError("sp_score_fusion_n: need >= 2 distinct groups")
    if denom not in ("all", "region_active"):
        raise ValueError(f"denom {denom}")

    def _p(f):
        return f["properties"] if isinstance(f.get("properties"), dict) else f

    def _grp(p):
        g = p.get("threshold_group") or p.get("markup_type")
        return "sp_focal" if g == "sp_or" else g

    def _c(f):
        g = f.get("geom")
        return (g if g is not None else shape(f["geometry"])).centroid

    idx = []
    for i, f in enumerate(feats):
        p = _p(f); grp = _grp(p)
        if p.get("class_name") in SP_FUSION_CLASSES and grp in groups:
            c = _c(f); idx.append((i, p.get("region_name") or "", grp, c.x, c.y, float(p.get("confidence", 0.0))))
    if not idx:
        return [f for f in feats if not _p(f).get("sp_partner_only")]
    ys = defaultdict(list)
    for _, reg, _, _, y, _ in idx:
        ys[reg].append(y)
    scale = {reg: _ground_scale(float(np.median(v))) for reg, v in ys.items()}
    pts = defaultdict(list)
    for i, reg, grp, x, y, conf in idx:
        pts[(reg, grp)].append((x * scale[reg], y * scale[reg], conf))
    trees = {k: (cKDTree(np.asarray([(a, b) for a, b, _ in v])), np.asarray([c for _, _, c in v])) for k, v in pts.items()}
    active = defaultdict(set)
    for reg, grp in pts:
        active[reg].add(grp)
    new_conf = {}
    for i, reg, grp, x, y, conf in idx:
        s = conf
        for h in groups:
            if h == grp:
                continue
            t = trees.get((reg, h))
            if t is not None:
                hit = t[0].query_ball_point((x * scale[reg], y * scale[reg]), radius)
                if hit:
                    s += float(t[1][hit].max())
        n = len(groups) if denom == "all" else max(1, len(active[reg]))
        new_conf[i] = s / n
    out = []
    for i, f in enumerate(feats):
        if _p(f).get("sp_partner_only"):
            continue
        if i in new_conf:
            f = dict(f)
            if isinstance(f.get("properties"), dict):
                f["properties"] = dict(f["properties"], confidence=new_conf[i])
            else:
                f["confidence"] = new_conf[i]
        out.append(f)
    return out


if __name__ == "__main__":   # equivalence check vs expds.post.sp_score_fusion for N=2 + N=3 sanity
    import sys, random
    sys.path.insert(0, "/workspace/koz3/wt/sp-large-f0")
    from shapely.geometry import Point
    from expds.post import sp_score_fusion
    rnd = random.Random(0); feats = []
    for k in range(600):
        g = rnd.choice(["sp_focal", "sp_focal2", "sp_det2", "sp_or", "li"])
        feats.append({"geom": Point(4.2e6 + rnd.uniform(0, 300), 7.5e6 + rnd.uniform(0, 300)).buffer(4),
                      "class_name": rnd.choice(SP_FUSION_CLASSES), "confidence": rnd.random(),
                      "region_name": rnd.choice(["A", "B"]), "threshold_group": g, "sp_partner_only": rnd.random() < 0.05})
    a = sp_score_fusion(feats, {"groups": ["sp_focal", "sp_focal2"]})
    b = sp_score_fusion_n(feats, {"groups": ["sp_focal", "sp_focal2"]})
    assert len(a) == len(b) and all(abs(x["confidence"] - y["confidence"]) < 1e-12 for x, y in zip(a, b)), "N=2 mismatch"
    c = sp_score_fusion_n(feats, {"groups": ["sp_focal", "sp_focal2", "sp_det2"]})
    d = sp_score_fusion_n(feats, {"groups": ["sp_focal", "sp_focal2", "sp_det2"], "denom": "region_active"})
    assert len(c) == len(a) and all(0 <= f["confidence"] <= 1 for f in c + d)
    print("OK N=2 identical to post.sp_score_fusion on", len(a), "feats; N=3 runs (all/region_active)")
