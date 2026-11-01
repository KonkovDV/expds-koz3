#!/usr/bin/env python3
"""Rebuild KOZ-3 geo-200-217 train tree for pixgbm-cr gate (bot-6).

Sources (Kaggle /kaggle/input):
  koz3-w4-geo-s000  — rasters for regions 200–207
  koz3-w4-geo-s001  — rasters for regions 208–217
  koz3-xfer-s000    — small_files.tar.zst (labels/UTM for 200–217)
  koz3-gt-train-v2  — optional GT overlay

Layout expected inside w4-geo shards (PREREG W4):
  NNN_<site>/<folder *_MagGr_*|*_EtGr_*|*_TerFl_*>/...
plus UTM.json when present. Labels preferably come from xfer-s000 small tar.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

GEO_PREFIXES = tuple(f"{n:03d}_" for n in range(200, 218))


def find_inputs(root: Path | None = None) -> dict[str, Path]:
    root = root or Path("/kaggle/input")
    out: dict[str, Path] = {}
    if not root.is_dir():
        return out
    for p in sorted(root.iterdir()):
        if p.is_dir():
            out[p.name.lower()] = p
    return out


def find_named(inputs: dict[str, Path], *needles: str) -> Path | None:
    for n in needles:
        for k, p in inputs.items():
            if n in k:
                return p
    return None


def _is_geo_rel(rel: str) -> bool:
    rel = rel.replace("\\", "/").lstrip("./")
    top = rel.split("/", 1)[0]
    return any(top.startswith(p) for p in GEO_PREFIXES)


def link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return "exists"
    try:
        os.symlink(src, dst)
        return "symlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def overlay_tree(src_root: Path, dest: Path) -> dict:
    stats = {"src": src_root.name, "linked": 0, "copied": 0, "exists": 0, "skipped": 0, "files": 0}
    if not src_root.is_dir():
        stats["error"] = "missing"
        return stats
    for p in src_root.rglob("*"):
        if not p.is_file():
            continue
        try:
            rel = p.relative_to(src_root).as_posix()
        except ValueError:
            continue
        if not _is_geo_rel(rel):
            stats["skipped"] += 1
            continue
        stats["files"] += 1
        action = link_or_copy(p, dest / rel)
        stats[action] = stats.get(action, 0) + 1
    return stats


def rebuild(dest: Path | None = None, input_root: Path | None = None) -> dict:
    dest = dest or Path(os.environ.get("PIXGBM_TREE", "/kaggle/working/train-expds-3"))
    dest.mkdir(parents=True, exist_ok=True)
    inputs = find_inputs(input_root)
    report: dict = {"root": str(dest), "inputs": sorted(inputs), "overlays": []}

    # 1) labels/UTM from xfer-s000 via sibling rebuild_train_tree if present
    here = Path(__file__).resolve().parent
    rebuild_mod = here / "scripts" / "rebuild_train_tree.py"
    if not rebuild_mod.is_file():
        rebuild_mod = here / "rebuild_train_tree.py"
    xfer0 = find_named(inputs, "koz3-xfer-s000")
    if rebuild_mod.is_file() and xfer0 is not None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("rebuild_train_tree", rebuild_mod)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        # Prefer only s000 for labels; skip hollow s001–s009 large blobs
        os.environ.setdefault("PIXGBM_REBUILD_ALL", "")
        # Call lower-level pieces if available
        if hasattr(mod, "find_small_tar") and hasattr(mod, "extract_small_tar"):
            tar = mod.find_small_tar(xfer0)
            if tar:
                n = mod.extract_small_tar(tar, dest)
                report["xfer_s000_members"] = n
        gt = None
        if hasattr(mod, "find_gt_dataset"):
            gt = mod.find_gt_dataset(inputs)
        if gt is not None and hasattr(mod, "overlay_gt"):
            report["gt"] = mod.overlay_gt(gt, dest)

    # 2) geo rasters from w4-geo shards (authoritative for MagGr/EtGr/TerFl)
    for key in ("koz3-w4-geo-s000", "koz3-w4-geo-s001"):
        shard = find_named(inputs, key)
        if shard is None:
            report["overlays"].append({"src": key, "error": "not_attached"})
            continue
        # shard may nest one level
        roots = [shard]
        subdirs = [p for p in shard.iterdir() if p.is_dir() and any(p.name.startswith(pref) for pref in GEO_PREFIXES)]
        if subdirs:
            roots = [shard]
        st = overlay_tree(shard, dest)
        report["overlays"].append(st)

    geo_dirs = sorted(
        p.name for p in dest.iterdir()
        if p.is_dir() and any(p.name.startswith(pref) for pref in GEO_PREFIXES)
    ) if dest.is_dir() else []
    report["geo_region_dirs"] = geo_dirs
    report["n_geo_dirs"] = len(geo_dirs)

    # Count geo rasters roughly
    n_tif = 0
    for d in geo_dirs:
        for p in (dest / d).rglob("*"):
            if p.is_file() and p.suffix.lower() in {".tif", ".tiff", ".img"}:
                low = p.name.lower()
                if any(x in low for x in ("maggr", "etgr", "terfl", "маг", "эт", "тер")):
                    n_tif += 1
                elif any(x in str(p.parent).lower() for x in ("maggr", "etgr", "terfl")):
                    n_tif += 1
    report["n_geo_raster_files_guess"] = n_tif
    report["hollow"] = len(geo_dirs) == 0 or n_tif == 0
    return report


if __name__ == "__main__":
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    rep = rebuild(dest=dest)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    raise SystemExit(2 if rep.get("hollow") else 0)
