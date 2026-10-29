#!/usr/bin/env python3
"""Load each weight path with torch.load; exit 1 on any failure. ASCII-only."""
from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: torch_preflight.py PATH [PATH ...]", file=sys.stderr)
        return 2
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        print(f"FAIL import torch: {e}")
        return 1
    bad: list[str] = []
    for raw in argv:
        p = Path(raw)
        if not p.is_file():
            bad.append(f"{raw}: missing")
            continue
        if p.stat().st_size <= 0:
            bad.append(f"{raw}: empty")
            continue
        try:
            torch.load(str(p), map_location="cpu", weights_only=False)
        except TypeError:
            # older torch without weights_only
            try:
                torch.load(str(p), map_location="cpu")
            except Exception as e:  # noqa: BLE001
                bad.append(f"{raw}: {e}")
        except Exception as e:  # noqa: BLE001
            bad.append(f"{raw}: {e}")
    if bad:
        print("FAIL")
        for line in bad:
            print(line)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
