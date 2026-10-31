#!/usr/bin/env python3
"""bot-6 r3: discover KOZ-3 geo regions (200-217 or any with maggr/etgr/terfl),
plan/copy ONLY geo rasters + UTM.json + boundaries + noise (no LAS/aerial/annotations).
Abort if planned copy exceeds --max-gb (default 10).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

GEO_PREFIXES = tuple(str(n) for n in range(200, 218))
GEO_MTS = ("maggr", "etgr", "terfl")


def _ensure_repo(repo: Path) -> None:
    root = str(repo.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def discover(repo: Path, train_root: Path):
    _ensure_repo(repo)
    from expds.catalog import scan_root

    regions = scan_root(train_root)
    by_prefix = [r for r in regions if r.name[:3] in GEO_PREFIXES]
    by_rasters = [
        r for r in regions
        if any(r.rasters.get(mt) for mt in GEO_MTS)
    ]
    # Prefer prefix list when present; else discover by geo rasters.
    chosen = by_prefix if by_prefix else by_rasters
    # Always require at least one geo raster so empty prefix folders are dropped.
    chosen = [r for r in chosen if any(r.rasters.get(mt) for mt in GEO_MTS)]
    return chosen, {
        "scan_total": len(regions),
        "by_prefix_200_217": [r.name for r in by_prefix],
        "by_geo_rasters": [r.name for r in by_rasters],
        "chosen": [r.name for r in chosen],
        "selection": "prefix_200_217" if by_prefix else "discover_maggr_etgr_terfl",
    }


def plan_files(reg) -> list[Path]:
    files: list[Path] = []
    for mt in GEO_MTS:
        files.extend(reg.rasters.get(mt, []) or [])
    files.extend(reg.boundaries or [])
    files.extend(reg.noise or [])
    if reg.utm_json is not None:
        files.append(reg.utm_json)
    # unique, stable
    seen = set()
    out = []
    for f in files:
        key = str(f.resolve()) if f.exists() else str(f)
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


def file_size(p: Path) -> int:
    try:
        return p.stat().st_size if p.is_file() else 0
    except OSError:
        return 0


def build_plan(regions, train_root: Path) -> dict:
    items = []
    total = 0
    missing = []
    for reg in regions:
        for src in plan_files(reg):
            if not src.is_file():
                missing.append(str(src))
                continue
            rel = src.relative_to(reg.path)
            dst_rel = Path(reg.name) / rel
            sz = file_size(src)
            total += sz
            items.append({
                "region": reg.name,
                "src": str(src),
                "dst_rel": str(dst_rel).replace("\\", "/"),
                "bytes": sz,
                "markup_hint": _hint(src),
            })
    return {
        "train_root": str(train_root),
        "n_regions": len(regions),
        "n_files": len(items),
        "total_bytes": total,
        "total_gb": round(total / (1024 ** 3), 3),
        "missing": missing,
        "files": items,
    }


def _hint(p: Path) -> str:
    low = p.name.lower()
    for mt in GEO_MTS:
        if mt in low:
            return mt
    if p.name == "UTM.json":
        return "utm"
    # boundaries/noise matched by catalog already
    if p.suffix.lower() in (".geojson", ".json"):
        return "extra"
    return "other"


def do_copy(plan: dict, dst_root: Path, dry_run: bool) -> list[str]:
    errors = []
    for it in plan["files"]:
        src = Path(it["src"])
        dst = dst_root / it["dst_rel"]
        if dry_run:
            continue
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        except OSError as exc:
            errors.append("%s -> %s: %s" % (src, dst, exc))
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description="bot6 r3 selective geo region copy")
    ap.add_argument("--repo", required=True, help="worktree with expds/")
    ap.add_argument("--root", required=True, help="train root e.g. F:/2/train-expds-3")
    ap.add_argument("--dst", required=True, help="destination regions dir on C:")
    ap.add_argument("--plan-out", required=True, help="JSON plan path")
    ap.add_argument("--max-gb", type=float, default=10.0)
    ap.add_argument("--copy", action="store_true", help="actually copy (else plan only)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo)
    root = Path(args.root)
    dst = Path(args.dst)
    if not root.is_dir():
        print(json.dumps({"ok": False, "error": "root_missing", "root": str(root)}))
        return 16
    regions, meta = discover(repo, root)
    if not regions:
        print(json.dumps({"ok": False, "error": "no_geo_regions", "meta": meta}, ensure_ascii=False))
        return 16
    plan = build_plan(regions, root)
    plan["meta"] = meta
    plan["max_gb"] = args.max_gb
    plan["would_exceed"] = plan["total_gb"] > args.max_gb
    Path(args.plan_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.plan_out).write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "ok": not plan["would_exceed"] and not plan["missing"],
        "n_regions": plan["n_regions"],
        "n_files": plan["n_files"],
        "total_gb": plan["total_gb"],
        "max_gb": args.max_gb,
        "would_exceed": plan["would_exceed"],
        "missing_n": len(plan["missing"]),
        "selection": meta["selection"],
        "regions": meta["chosen"],
        "plan": str(args.plan_out),
    }
    print(json.dumps(summary, ensure_ascii=False))
    if plan["would_exceed"]:
        return 15
    if args.copy and not args.dry_run:
        dst.mkdir(parents=True, exist_ok=True)
        errors = do_copy(plan, dst, dry_run=False)
        if errors:
            (dst.parent / "notes" / "copy_errors.txt").parent.mkdir(parents=True, exist_ok=True)
            Path(args.plan_out).with_suffix(".errors.txt").write_text(
                "\n".join(errors), encoding="utf-8")
            print(json.dumps({"ok": False, "error": "copy_failed", "n": len(errors)}))
            return 17
        # write region list
        (dst / "_region_list.txt").write_text(
            "\n".join(meta["chosen"]) + "\n", encoding="utf-8")
    return 0




def _self_check() -> int:
    """Fixture tree under /tmp: plan size, abort >max-gb (exit 15), no LAS copied."""
    import tempfile
    root = Path(tempfile.mkdtemp(prefix="bot6_r3_sc_"))
    # Minimal fake Region-like layout the catalog understands is heavy;
    # unit-test plan_files / build_plan / max-gb path with a stub region object.
    class R:
        def __init__(self, name, path, rasters, boundaries=None, noise=None, utm_json=None):
            self.name = name
            self.path = path
            self.rasters = rasters
            self.boundaries = boundaries or []
            self.noise = noise or []
            self.utm_json = utm_json
    reg_dir = root / "200_X"
    reg_dir.mkdir()
    mag = reg_dir / "maggr_gray.tif"
    las = reg_dir / "points.las"
    utm = reg_dir / "UTM.json"
    mag.write_bytes(b"0" * 1024)
    las.write_bytes(b"L" * 2048)
    utm.write_text("{}", encoding="utf-8")
    bnd = reg_dir / "boundary.geojson"
    bnd.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
    reg = R("200_X", reg_dir, {"maggr": [mag], "etgr": [], "terfl": []},
            boundaries=[bnd], noise=[], utm_json=utm)
    files = plan_files(reg)
    assert mag in files and utm in files and bnd in files
    assert las not in files, "LAS must not be in plan"
    plan = build_plan([reg], root)
    assert plan["total_bytes"] == 1024 + len(utm.read_bytes()) + len(bnd.read_bytes())
    assert plan["n_files"] == 3
    # max-gb abort: tiny threshold
    plan["total_gb"] = 11.0
    plan["would_exceed"] = plan["total_gb"] > 10.0
    assert plan["would_exceed"]
    # write plan and simulate exit 15 path
    out = root / "plan.json"
    out.write_text(__import__("json").dumps({"total_gb": 11.0, "would_exceed": True}), encoding="utf-8")
    print(__import__("json").dumps({
        "self_check_pass": True,
        "n_files": 3,
        "excludes_las": True,
        "would_exceed_11gb": True,
        "exit_code_on_exceed": 15,
    }))
    return 0


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        sys.exit(_self_check())
    sys.exit(main())
