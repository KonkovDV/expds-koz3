"""Li post-processing levers: per-class τ floors + self-NMS (fragmentation).

Master [li_post] enabled (default false) + independent sub-flags class_tau /
self_nms (also false). OFF → byte-identical finish path (no-op).

Does NOT touch closed levers (r50c5, WSR, ring, peak-rerank, sp_large, …).
Does NOT duplicate bot-7 geophysics. Operates only on lidar-family features
(markup_type / threshold_group starting with ``li``).

Honest-10 residual (local s0 ConvNeXt-Li OOF vs fold GT, soft cov≥0.5 +
rep-point, base τ=0.15 = production fallback for li_cnx):
  off F1≈0.280; class_tau nested ΔF1≈+0.118; self_nms@8 ΔF1≈+0.003;
  both ≈+0.120. Numbers are Li-layer only — see reports/bot4_li_post.md.
"""
from __future__ import annotations

import logging
from collections import defaultdict

log = logging.getLogger(__name__)

#: Classes that may get an absolute floor under class_tau.
DEFAULT_FLOOR_CLASSES = (
    "kurgany_povrezhdennye",
    "kurgany_tselye",
    "gorodishcha",
    "fortifikatsii",
)

#: Nested-fit median floors from s0 leave-one-fold (base 0.15 grid).
#: Fort/gorodishcha stayed at 0.15 — do not raise (FN-heavy).
DEFAULT_CLASS_TAU_FLOOR = {
    "kurgany_povrezhdennye": 0.30,
    "kurgany_tselye": 0.30,
}


def enabled(cfg: dict | None) -> bool:
    return bool(((cfg or {}).get("li_post") or {}).get("enabled"))


def flag(cfg: dict | None, name: str) -> bool:
    """Master enabled AND sub-flag name (default false)."""
    if not enabled(cfg):
        return False
    return bool(((cfg or {}).get("li_post") or {}).get(name))


def spec(cfg: dict | None) -> dict:
    return dict((cfg or {}).get("li_post") or {})


def is_li_feat(feat: dict) -> bool:
    """True for lidar-family objects (li, li_cnx, li_mosaic, li_half, …)."""
    group = str(feat.get("threshold_group") or feat.get("markup_type") or "")
    markup = str(feat.get("markup_type") or "")
    return group.startswith("li") or markup.startswith("li")


def class_tau_floors(lo: dict | None) -> dict[str, float]:
    """Absolute confidence floors by class_name (empty → defaults)."""
    lo = lo or {}
    raw = lo.get("class_tau_floor")
    if isinstance(raw, dict) and raw:
        out: dict[str, float] = {}
        for k, v in raw.items():
            try:
                out[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
        return out
    return dict(DEFAULT_CLASS_TAU_FLOOR)


def apply_class_tau(feats: list[dict], floors: dict[str, float]) -> list[dict]:
    """Drop li-family feats below per-class absolute floor; leave others untouched.

    Floor is an extra gate on top of post.apply_thresholds (max with existing τ
    is implicit: if the object already passed apply_thresholds at a lower τ,
    this still drops it when conf < floor).
    """
    if not floors:
        return feats
    out: list[dict] = []
    for f in feats:
        if not is_li_feat(f):
            out.append(f)
            continue
        cls = f.get("class_name")
        floor = floors.get(cls)
        if floor is None:
            out.append(f)
            continue
        if float(f.get("confidence", 1.0)) >= float(floor):
            out.append(f)
    return out


def _xy(feat: dict) -> tuple[float, float] | None:
    geom = feat.get("geom")
    if geom is None or getattr(geom, "is_empty", True):
        return None
    try:
        c = geom.centroid
        return float(c.x), float(c.y)
    except Exception:
        return None


def self_nms(feats: list[dict], radius_m: float) -> list[dict]:
    """Greedy same-class NMS among li-family feats within radius_m (EPSG units ≈ m).

    Non-li feats pass through unchanged and keep their relative order among
    themselves; li survivors keep conf-descending order within the merged list
    by stable partition (non-li first in original order, then kept li).
    """
    if radius_m <= 0:
        return feats
    r2 = float(radius_m) * float(radius_m)
    li_idx = [i for i, f in enumerate(feats) if is_li_feat(f)]
    if len(li_idx) <= 1:
        return feats

    # Per class, greedy by confidence.
    by_cls: dict[str, list[int]] = defaultdict(list)
    for i in li_idx:
        by_cls[str(feats[i].get("class_name") or "")].append(i)
    keep_li: set[int] = set()
    for _cls, idxs in by_cls.items():
        idxs_sorted = sorted(idxs, key=lambda i: float(feats[i].get("confidence", 1.0)), reverse=True)
        taken: list[tuple[float, float]] = []
        for i in idxs_sorted:
            xy = _xy(feats[i])
            if xy is None:
                keep_li.add(i)
                continue
            x, y = xy
            if any((x - tx) * (x - tx) + (y - ty) * (y - ty) <= r2 for tx, ty in taken):
                continue
            taken.append((x, y))
            keep_li.add(i)

    out: list[dict] = []
    for i, f in enumerate(feats):
        if i in li_idx and i not in keep_li:
            continue
        out.append(f)
    return out


def apply(feats: list[dict], cfg: dict | None) -> list[dict]:
    """Run enabled li_post levers. No-op when master/sub-flags off."""
    if not enabled(cfg):
        return feats
    lo = spec(cfg)
    out = feats
    if flag(cfg, "class_tau"):
        out = apply_class_tau(out, class_tau_floors(lo))
    if flag(cfg, "self_nms"):
        radius = float(lo.get("self_nms_m", 8.0))
        out = self_nms(out, radius)
    return out
