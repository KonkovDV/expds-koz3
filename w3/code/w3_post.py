"""W3 post-processing levers (flags default OFF). Drop-in for expds/post.py on bots/bot7-candidate-v4-nomodels.

Feature dicts follow post.py convention: keys class_name, region_name, markup_type, threshold_group,
confidence, geom (shapely, EPSG:3857). Both functions only ADD features in their own threshold groups,
so base groups/thresholds are untouched (lesson of docs/REASONING_AND_DISCOVERY_LOG §4.2: plain relabel -0.0064).
"""
from __future__ import annotations

import math
from collections import defaultdict

POVR, TSEL, GOR = "kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha"


def _ground_scale(y: float) -> float:
    return math.cos(2 * math.atan(math.exp(y / 6378137.0)) - math.pi / 2)


def _grp(f: dict) -> str:
    return str(f.get("threshold_group") or f.get("markup_type") or "")


def gor_from_fort_arch(feats: list[dict], spec: dict | None = None) -> list[dict]:
    """[post.gor_from_fort_arch] enabled=false. Large fortifikatsii/arkhitektury -> extra gorodishcha twin.

    Why: 25 GT hillforts found with good geometry but written as fort/arch (other_class;
    15 sp_or, 4 li, 4 ae, 2 or). Twin goes to group gor_fa_<markup>, calibrated by honest sweep.
    """
    cfg = spec or {}
    if not cfg.get("enabled", False):
        return feats
    min_area = float(cfg.get("min_area_m2", 5000.0))
    src = set(cfg.get("classes", ["fortifikatsii", "arkhitektury"]))
    markups = set(cfg.get("markups", [])) or None
    skip_cov = float(cfg.get("skip_if_gor_iom", 0.5))
    gor_by = defaultdict(list)
    for f in feats:
        if f.get("class_name") == GOR:
            gor_by[(f.get("region_name"), f.get("markup_type"))].append(f["geom"])
    out = list(feats)
    for f in feats:
        if f.get("class_name") not in src:
            continue
        mk = f.get("markup_type")
        if markups is not None and mk not in markups:
            continue
        g = f["geom"]
        s = _ground_scale(g.centroid.y)
        if g.area * s * s < min_area:
            continue
        dup = False
        for h in gor_by.get((f.get("region_name"), mk), []):
            inter = g.intersection(h).area
            if inter >= skip_cov * min(g.area, h.area):
                dup = True
                break
        if dup:
            continue
        out.append(dict(f, class_name=GOR, threshold_group=f"gor_fa_{mk}", gor_fa_src=f.get("class_name")))
    return out


def kurgan_subclass_twins(feats: list[dict], spec: dict | None = None) -> list[dict]:
    """[post.kurgan_subclass_twins] enabled=false. Classifier-gated opposite-subclass twin for kurgans.

    Needs property subclass_p_povr (bot-9 OOF / final model, written by bot9_kurgan_subclass.py apply-style join).
    tselye with p>=hi -> povr twin (group <g>_sc_povr, conf*p); povr with p<=lo -> tselye twin
    (group <g>_sc_tsel, conf*(1-p)). No twin if an opposite-subclass feature of any group is within dedup_m.
    Originals are kept.
    """
    cfg = spec or {}
    if not cfg.get("enabled", False):
        return feats
    hi, lo = float(cfg.get("p_hi", 0.6)), float(cfg.get("p_lo", 0.2))
    dedup_m = float(cfg.get("dedup_m", 2.0))
    groups = set(cfg.get("groups", ["li", "li_mosaic", "li_coarse"]))
    scale_conf = bool(cfg.get("scale_conf", True))
    import numpy as np
    from scipy.spatial import cKDTree

    pts = {POVR: defaultdict(list), TSEL: defaultdict(list)}
    ys = defaultdict(list)
    for f in feats:
        if f.get("class_name") in (POVR, TSEL):
            ys[f.get("region_name") or ""].append(f["geom"].centroid.y)
    sc = {r: _ground_scale(sum(v) / len(v)) for r, v in ys.items()}
    for f in feats:
        c = f.get("class_name")
        if c in pts:
            r = f.get("region_name") or ""
            p = f["geom"].centroid
            pts[c][r].append((p.x * sc[r], p.y * sc[r]))
    trees = {c: {r: cKDTree(np.asarray(v)) for r, v in d.items() if v} for c, d in pts.items()}
    out = list(feats)
    for f in feats:
        c = f.get("class_name")
        if c not in (POVR, TSEL) or _grp(f) not in groups:
            continue
        p = f.get("subclass_p_povr")
        if p is None:
            continue
        p = float(p)
        if c == TSEL and p >= hi:
            new, w = POVR, p
        elif c == POVR and p <= lo:
            new, w = TSEL, 1.0 - p
        else:
            continue
        r = f.get("region_name") or ""
        q = f["geom"].centroid
        t = trees[new].get(r)
        if dedup_m > 0 and t is not None and t.query_ball_point((q.x * sc[r], q.y * sc[r]), dedup_m):
            continue
        conf = float(f.get("confidence", 0.0)) * (w if scale_conf else 1.0)
        tag = "sc_povr" if new == POVR else "sc_tsel"
        out.append(dict(f, class_name=new, confidence=conf, threshold_group=f"{_grp(f)}_{tag}",
                        class_name_base=c))
    return out
