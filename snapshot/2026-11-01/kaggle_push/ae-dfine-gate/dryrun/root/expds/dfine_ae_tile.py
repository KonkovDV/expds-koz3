"""Align Ae D-FINE infer tiling/fuse with Ae FT OOF (bot-3).

Mismatch audit (2026-10-29):
- res_m/tile/overlap/min_score/max_det/nms_iou/flip_tta already match FT (0.5/640/160/0.05/100/0.5/True).
- Window grid: dfine_aeor default "aeor" pins last tile to n-tile; OOF/sp_dfine use overhanging
  range(0, max(1,n-overlap), step) — different last-column starts (e.g. W=1000 → 360 vs 480).
- Fuse: infer default WBF@0.55; OOF used single batched_nms@0.5 across windows×flip.

Section [dfine_ae_tile] enabled=false by default. When ON, patches [dfine_aeor] in-memory:
windows_mode=train, fuse=nms (and optional scales/overlap overrides). Does not enable dfine_aeor.
"""
from __future__ import annotations

import copy
import logging
from typing import Any

log = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "windows_mode": "train",   # train | aeor
    "fuse": "nms",             # nms | wbf — match OOF when nms
    "fuse_iou": 0.55,          # used only if fuse=wbf
    "nms_iou": 0.5,
    # None = leave dfine_aeor defaults alone
    "scales": None,            # e.g. [1.0, 1.5] multi-scale lever
    "overlap_ae": None,        # optional int override for ae overlap
    "tile_ae": None,
    "budget_s": None,          # optional raise if large Ae rasters hit skip
}


def settings(cfg: dict | None) -> dict:
    out = copy.deepcopy(DEFAULTS)
    block = (cfg or {}).get("dfine_ae_tile") or {}
    if not isinstance(block, dict):
        return out
    for k, v in block.items():
        out[k] = v
    return out


def enabled(cfg: dict | None) -> bool:
    return bool(settings(cfg).get("enabled"))


def apply(cfg: dict) -> dict:
    """Patch cfg['dfine_aeor'] when [dfine_ae_tile] enabled. No-op if OFF."""
    st = settings(cfg)
    if not st.get("enabled"):
        return cfg
    out = dict(cfg)
    aeor = dict(out.get("dfine_aeor") or {})
    aeor["windows_mode"] = str(st.get("windows_mode") or "train")
    aeor["fuse"] = str(st.get("fuse") or "nms")
    if st.get("fuse_iou") is not None:
        aeor["fuse_iou"] = float(st["fuse_iou"])
    if st.get("nms_iou") is not None:
        aeor["nms_iou"] = float(st["nms_iou"])
    if st.get("scales") is not None:
        aeor["scales"] = list(st["scales"])
    if st.get("budget_s") is not None:
        aeor["budget_s"] = float(st["budget_s"])
    # per-mk overrides
    if st.get("tile_ae") is not None:
        tile = dict(aeor.get("tile") or {})
        if not isinstance(tile, dict):
            tile = {"ae": tile, "or": tile}
        tile["ae"] = int(st["tile_ae"])
        aeor["tile"] = tile
    if st.get("overlap_ae") is not None:
        ov = dict(aeor.get("overlap") or {})
        if not isinstance(ov, dict):
            ov = {"ae": ov, "or": ov}
        ov["ae"] = int(st["overlap_ae"])
        aeor["overlap"] = ov
    out["dfine_aeor"] = aeor
    log.info(
        "dfine_ae_tile: ON — windows_mode=%s fuse=%s nms_iou=%s scales=%s",
        aeor.get("windows_mode"),
        aeor.get("fuse"),
        aeor.get("nms_iou"),
        aeor.get("scales"),
    )
    return out
