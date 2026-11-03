#!/usr/bin/env python3
"""Release v5 time-budget guard: soft_ceiling_s and optional measured wall ≤ max (default 1200 s)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_MAX_S = 1200.0  # 20 min


def soft_ceiling_from_toml(path: Path) -> float | None:
    try:
        import tomllib
    except ImportError:  # py<3.11
        import tomli as tomllib  # type: ignore
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    raw = (data.get("time") or {}).get("soft_ceiling_s")
    if raw is None or raw == "":
        return None
    return float(raw)


def check(soft_ceiling_s: float | None, wall_sec: float | None, max_sec: float) -> list[str]:
    errs: list[str] = []
    if soft_ceiling_s is not None and soft_ceiling_s > max_sec + 1e-6:
        errs.append(f"soft_ceiling_s={soft_ceiling_s} > max_sec={max_sec}")
    if wall_sec is not None and wall_sec > max_sec + 1e-6:
        errs.append(f"wall_sec={wall_sec} > max_sec={max_sec}")
    return errs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=None, help="TOML with [time].soft_ceiling_s")
    ap.add_argument("--wall-sec", type=float, default=None, help="measured inference wall seconds")
    ap.add_argument("--wall-file", type=Path, default=None, help="file with wall seconds (float)")
    ap.add_argument("--max-sec", type=float, default=DEFAULT_MAX_S, help="hard budget (default 1200)")
    ap.add_argument("--require-soft-ceiling", action="store_true",
                    help="FAIL if config missing soft_ceiling_s")
    args = ap.parse_args(argv)

    soft = None
    if args.config is not None:
        if not args.config.is_file():
            print(f"FAIL: config not found: {args.config}", file=sys.stderr)
            return 2
        soft = soft_ceiling_from_toml(args.config)
        if soft is None and args.require_soft_ceiling:
            print(f"FAIL: no [time].soft_ceiling_s in {args.config}", file=sys.stderr)
            return 1

    wall = args.wall_sec
    if args.wall_file is not None:
        wall = float(args.wall_file.read_text(encoding="utf-8").strip().split()[0])

    errs = check(soft, wall, float(args.max_sec))
    if errs:
        for e in errs:
            print(f"FAIL budget: {e}", file=sys.stderr)
        return 1
    parts = [f"max_sec={args.max_sec}"]
    if soft is not None:
        parts.append(f"soft_ceiling_s={soft}")
    if wall is not None:
        parts.append(f"wall_sec={wall}")
    print("PASS budget: " + ", ".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
