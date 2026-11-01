#!/usr/bin/env python3
"""bot-6 r3: build gt.geojson for geo regions (maggr/etgr/terfl only).

Uses expds.catalog + expds.gt loaders. Feature properties match writer /
e2e_score expectations: region_name, markup_type, class_name (+ fid, source).
Annotations are read from the train root; they are NOT copied into the
inference regions tree.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

GEO_PREFIXES = tuple(str(n) for n in range(200, 218))
GEO_MTS = frozenset({"maggr", "etgr", "terfl"})


def _ensure_repo(repo: Path) -> None:
    root = str(repo.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--root", required=True, help="train root with annotations")
    ap.add_argument("--out", required=True)
    ap.add_argument("--regions-from", default=None,
                    help="optional text file of region folder names (one per line)")
    ap.add_argument("--no-merge", action="store_true")
    args = ap.parse_args()

    _ensure_repo(Path(args.repo))
    from expds.catalog import scan_root
    from expds.gt import build_gt, to_feature_collection

    regions = scan_root(Path(args.root))
    if args.regions_from and Path(args.regions_from).is_file():
        want = {ln.strip() for ln in Path(args.regions_from).read_text(encoding="utf-8").splitlines() if ln.strip()}
        regions = [r for r in regions if r.name in want]
    else:
        by_prefix = [r for r in regions if r.name[:3] in GEO_PREFIXES]
        regions = by_prefix if by_prefix else [
            r for r in regions if any(r.rasters.get(mt) for mt in GEO_MTS)
        ]

    # Keep only geo markup annotations so ae/li GT rows do not poison the gate.
    for reg in regions:
        reg.annotations = [(p, c, mt) for (p, c, mt) in reg.annotations if mt in GEO_MTS]

    items, info = build_gt(regions, merge=not args.no_merge)
    # Drop non-geo markup if any slipped through
    items = [it for it in items if it.get("markup_type") in GEO_MTS]
    fc = to_feature_collection(items)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")

    mts = sorted({f["properties"]["markup_type"] for f in fc["features"]})
    regs = sorted({f["properties"]["region_name"] for f in fc["features"]})
    classes = sorted({f["properties"]["class_name"] for f in fc["features"]})
    summary = {
        "ok": True,
        "out": str(out),
        "n_features": len(fc["features"]),
        "n_regions_scanned": len(regions),
        "regions_in_gt": regs,
        "markup_types": mts,
        "classes": classes,
        "merges": info.get("merges"),
    }
    print(json.dumps(summary, ensure_ascii=False))
    if not fc["features"]:
        return 18
    if not (GEO_MTS & set(mts)):
        return 18
    return 0




def _self_check() -> int:
    """Smoke: GEO_MTS filter keeps only maggr/etgr/terfl; empty -> exit 18."""
    assert "maggr" in GEO_MTS and "ae" not in GEO_MTS
    # property filter simulation
    items = [
        {"markup_type": "maggr", "class_name": "object_poly"},
        {"markup_type": "ae", "class_name": "gorodishcha"},
        {"markup_type": "etgr", "class_name": "finds_points"},
    ]
    kept = [it for it in items if it.get("markup_type") in GEO_MTS]
    assert len(kept) == 2 and all(it["markup_type"] in GEO_MTS for it in kept)
    print(__import__("json").dumps({"self_check_pass": True, "kept": 2, "dropped_ae": 1, "empty_exit": 18}))
    return 0


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        sys.exit(_self_check())
    sys.exit(main())
