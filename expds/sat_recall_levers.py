"""Satellite recall levers (bot-7): precision-guarded mid-band + cross-layer agree.

Default OFF. Distinct from CLOSED levers:
  - sat 8m cluster / [post.sp_score_fusion] radius_m=8 (score average, no new dets)
  - near_kept / ring, sp_large, peak-rerank, sat2 (other bots)

L1 [sat_band_verify]: lower thr (scale) but mid-band needs verify_p ≥ floor.
L2 [sat_agree_accept]: thr×agree_scale only when partner sat group fires within
    radius_m (default 16 m ≠ 8 m CLOSED cluster).
L3 [sat_cross_nms]: after thresholds, keep max-conf across sat groups within nms_m
    (dedup FP doubles; helps P so thr relax stays safe).
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

log = logging.getLogger(__name__)

KURGANS = ("kurgany_povrezhdennye", "kurgany_tselye")
SAT_GROUPS_DEFAULT = ("sp_or", "sp_focal", "sp_focal2", "sp_arch")
# Pair for agree: center-map (sp_or/sp_focal) ↔ wide (sp_focal2).
AGREE_PAIR_DEFAULT = ("sp_focal", "sp_focal2")


def _group(f: dict) -> str:
    g = f.get("threshold_group") or f.get("markup_type") or ""
    # Submit center map is markup sp_or, thr table often sp_focal.
    return "sp_focal" if g == "sp_or" else g


def _norm_group(g: str) -> str:
    return "sp_focal" if g == "sp_or" else g


def any_enabled(cfg: dict | None) -> bool:
    cfg = cfg or {}
    for key in ("sat_band_verify", "sat_agree_accept", "sat_cross_nms"):
        spec = cfg.get(key) or {}
        if isinstance(spec, dict) and spec.get("enabled"):
            return True
    return False


def band_settings(cfg: dict | None) -> dict[str, Any]:
    spec = dict((cfg or {}).get("sat_band_verify") or {})
    return {
        "enabled": bool(spec.get("enabled")),
        "scale": float(spec.get("scale", 0.90)),
        "verify_floor": float(spec.get("verify_floor", 0.40)),
        "groups": tuple(spec.get("groups") or SAT_GROUPS_DEFAULT),
        "classes": tuple(spec.get("classes") or KURGANS),
        # no verify_p: reject mid-band (safe) unless explicitly admit
        "no_verify_policy": str(spec.get("no_verify_policy") or "reject"),
    }


def agree_settings(cfg: dict | None) -> dict[str, Any]:
    spec = dict((cfg or {}).get("sat_agree_accept") or {})
    return {
        "enabled": bool(spec.get("enabled")),
        "radius_m": float(spec.get("radius_m", 16.0)),
        "scale": float(spec.get("scale", 0.85)),
        "groups": tuple(spec.get("groups") or AGREE_PAIR_DEFAULT),
        "classes": tuple(spec.get("classes") or KURGANS),
    }


def nms_settings(cfg: dict | None) -> dict[str, Any]:
    spec = dict((cfg or {}).get("sat_cross_nms") or {})
    return {
        "enabled": bool(spec.get("enabled")),
        "nms_m": float(spec.get("nms_m", 4.0)),
        "groups": tuple(spec.get("groups") or SAT_GROUPS_DEFAULT),
        "classes": tuple(spec.get("classes") or KURGANS),
    }


def _ground_scale(y: float) -> float:
    """EPSG:3857 → approx meters on ground (same idea as post._ground_scale)."""
    import math
    # WebMercator: meters_x ≈ cos(lat); y is northing.
    lat = (2.0 * math.atan(math.exp(y / 6378137.0)) - math.pi / 2.0)
    return max(0.2, abs(math.cos(lat)))


def mark_agree(feats: list[dict], spec: dict[str, Any] | None = None) -> list[dict]:
    """Stamp sat_agree=True on kurgans that have a partner from the other group within radius_m.

    Does not change confidence. radius_m default 16 — not the CLOSED 8 m score-fusion cluster.
    """
    st = spec if spec is not None else agree_settings({"sat_agree_accept": {"enabled": True}})
    if not st.get("enabled"):
        return feats
    groups = tuple(_norm_group(g) for g in st["groups"])
    if len(groups) != 2:
        raise ValueError(f"sat_agree_accept needs exactly 2 groups, got {groups}")
    classes = set(st["classes"])
    radius = float(st["radius_m"])
    if radius <= 0:
        raise ValueError(f"sat_agree_accept radius_m must be > 0, got {radius}")
    # Refuse accidental re-enable of CLOSED 8 m cluster semantics via wrong radius.
    if abs(radius - 8.0) < 1e-9:
        raise ValueError(
            "sat_agree_accept radius_m=8.0 overlaps CLOSED sat 8m cluster / sp_score_fusion; "
            "use e.g. 12–20 m"
        )

    from scipy.spatial import cKDTree
    import numpy as np

    idx: list[tuple[int, str, str, float, float]] = []
    for i, f in enumerate(feats):
        if f.get("class_name") not in classes:
            continue
        g = _group(f)
        if g not in groups:
            continue
        geom = f.get("geom")
        if geom is None or geom.is_empty:
            continue
        c = geom.centroid
        idx.append((i, f.get("region_name") or "", g, float(c.x), float(c.y)))
    if not idx:
        return feats

    ys: dict[str, list[float]] = defaultdict(list)
    for _, reg, _, _, y in idx:
        ys[reg].append(y)
    scale = {r: _ground_scale(float(np.median(v))) for r, v in ys.items()}

    pts: dict[tuple[str, str], list[tuple[float, float, int]]] = defaultdict(list)
    for i, reg, g, x, y in idx:
        s = scale[reg]
        pts[(reg, g)].append((x * s, y * s, i))

    trees: dict[tuple[str, str], Any] = {}
    for key, rows in pts.items():
        arr = np.asarray([(a, b) for a, b, _ in rows], dtype=np.float64)
        trees[key] = (cKDTree(arr), [i for _, _, i in rows])

    agree_at = set()
    g0, g1 = groups
    for i, reg, g, x, y in idx:
        other = g1 if g == g0 else g0
        t = trees.get((reg, other))
        if t is None:
            continue
        s = scale[reg]
        hits = t[0].query_ball_point((x * s, y * s), radius)
        if hits:
            agree_at.add(i)

    if not agree_at:
        return feats
    out = []
    for i, f in enumerate(feats):
        if i in agree_at:
            f = dict(f)
            f["sat_agree"] = True
        out.append(f)
    log.info("sat_agree_accept: marked %d / %d sat kurgans (r=%.1f m)", len(agree_at), len(idx), radius)
    return out


def _orig_tau(f: dict, thresholds: dict, default: float, by_markup: dict) -> float:
    own = by_markup.get(f.get("threshold_group") or f.get("markup_type"), {}) or {}
    # also try normalized group for sp_or ↔ sp_focal
    if f["class_name"] not in own:
        own = by_markup.get(_group(f), {}) or own
    return float(own.get(f["class_name"], thresholds.get(f["class_name"], default)))


def apply_thresholds(
    feats: list[dict],
    cfg: dict,
    by_markup: dict[str, dict[str, float]] | None,
) -> list[dict]:
    """Threshold pass with optional band-verify + agree scales; then optional cross-NMS."""
    thresholds = cfg.get("thresholds") or {}
    default = float(cfg.get("default_threshold", 0.15))
    by_markup = by_markup or {}
    band = band_settings(cfg)
    agree = agree_settings(cfg)

    if agree.get("enabled"):
        feats = mark_agree(feats, agree)

    band_groups = set(_norm_group(g) for g in band["groups"]) if band["enabled"] else set()
    band_classes = set(band["classes"]) if band["enabled"] else set()
    agree_groups = set(_norm_group(g) for g in agree["groups"]) if agree["enabled"] else set()
    agree_classes = set(agree["classes"]) if agree["enabled"] else set()

    if band["enabled"] and band["scale"] <= 0:
        raise ValueError(f"sat_band_verify scale must be > 0, got {band['scale']}")
    if agree["enabled"] and agree["scale"] <= 0:
        raise ValueError(f"sat_agree_accept scale must be > 0, got {agree['scale']}")

    kept: list[dict] = []
    for f in feats:
        conf = float(f.get("confidence", 1.0))
        thr = _orig_tau(f, thresholds, default, by_markup)
        g = _group(f)
        cls = f.get("class_name")

        # Effective thr: start from original; may lower via agree and/or band.
        eff = thr
        used_band = False
        if agree["enabled"] and f.get("sat_agree") and g in agree_groups and cls in agree_classes:
            eff = min(eff, thr * float(agree["scale"]))
        if band["enabled"] and g in band_groups and cls in band_classes:
            low = thr * float(band["scale"])
            if conf >= thr:
                pass  # full admit
            elif conf >= low:
                vp = f.get("verify_p")
                if vp is None:
                    if band["no_verify_policy"] == "admit":
                        eff = low
                        used_band = True
                    else:
                        continue  # reject mid-band without verify
                elif float(vp) >= float(band["verify_floor"]):
                    eff = low
                    used_band = True
                else:
                    continue
            # else conf < low → fall through to eff compare (will fail unless agree lowered more)

        if conf >= eff:
            if used_band or (agree["enabled"] and f.get("sat_agree") and conf < thr):
                f = dict(f)
                if used_band:
                    f["sat_band"] = True
            kept.append(f)

    nms = nms_settings(cfg)
    if nms["enabled"]:
        kept = cross_nms(kept, nms)
    return kept


def cross_nms(feats: list[dict], spec: dict[str, Any] | None = None) -> list[dict]:
    """Keep highest-conf detection among sat groups within nms_m (per region+class)."""
    st = spec if spec is not None else nms_settings({"sat_cross_nms": {"enabled": True}})
    if not st.get("enabled"):
        return feats
    nms_m = float(st["nms_m"])
    if nms_m <= 0:
        raise ValueError(f"sat_cross_nms nms_m must be > 0, got {nms_m}")
    groups = set(_norm_group(g) for g in st["groups"])
    classes = set(st["classes"])

    import numpy as np

    sat_idx = []
    other = []
    for i, f in enumerate(feats):
        if f.get("class_name") in classes and _group(f) in groups:
            geom = f.get("geom")
            if geom is None or geom.is_empty:
                other.append(f)
                continue
            c = geom.centroid
            sat_idx.append((i, f, f.get("region_name") or "", f["class_name"], float(c.x), float(c.y),
                            float(f.get("confidence", 0.0))))
        else:
            other.append(f)
    if len(sat_idx) <= 1:
        return feats

    # Greedy by confidence desc within region+class
    sat_idx.sort(key=lambda t: -t[6])
    kept_flags = [True] * len(sat_idx)
    # Build per (reg, cls) trees of already kept
    kept_pts: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    scales: dict[str, float] = {}
    ys: dict[str, list[float]] = defaultdict(list)
    for _, _, reg, _, _, y, _ in sat_idx:
        ys[reg].append(y)
    scales = {r: _ground_scale(float(np.median(v))) for r, v in ys.items()}

    out_sat = []
    for n, (_i, f, reg, cls, x, y, conf) in enumerate(sat_idx):
        if not kept_flags[n]:
            continue
        s = scales[reg]
        px, py = x * s, y * s
        conflict = False
        for qx, qy in kept_pts[(reg, cls)]:
            if (px - qx) ** 2 + (py - qy) ** 2 <= nms_m * nms_m:
                conflict = True
                break
        if conflict:
            kept_flags[n] = False
            continue
        kept_pts[(reg, cls)].append((px, py))
        out_sat.append(f)
    log.info("sat_cross_nms: %d → %d (nms=%.1f m)", len(sat_idx), len(out_sat), nms_m)
    return other + out_sat
