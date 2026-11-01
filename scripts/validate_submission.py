#!/usr/bin/env python3
"""Release submission validator: format/CRS/schema (LESSONS + LB) via check_compliance.

Rules enforced (see scripts/check_compliance.py + LESSONS.md):
  FeatureCollection; Polygon-only; EPSG:3857; ≤100 MB; ≤20k features; ≤3000 verts/ring;
  reject kit-zip (code/models without geojson).

Usage:
  python scripts/validate_submission.py path/to/result.geojson|submission.zip

Exit: 0 ok; 1 FAIL rules; 2 IO/args.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_compliance.py"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not SCRIPT.is_file():
        print(f"FAIL: missing {SCRIPT}", file=sys.stderr)
        return 2
    spec = importlib.util.spec_from_file_location("check_compliance_rel", SCRIPT)
    if spec is None or spec.loader is None:
        print(f"FAIL: cannot load {SCRIPT}", file=sys.stderr)
        return 2
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return int(mod.main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
