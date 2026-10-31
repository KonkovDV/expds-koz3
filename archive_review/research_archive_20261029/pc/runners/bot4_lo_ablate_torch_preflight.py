"""Host torch.load smoke for critical .pt weights. Fail fast on EOF/corrupt.

Usage: python bot4_lo_ablate_torch_preflight.py path1.pt [path2.pt ...]
Exit 0 on OK, 1 on any load failure. Skips missing paths with a WARN line.
"""
from __future__ import annotations

import sys


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: bot4_lo_ablate_torch_preflight.py path.pt [...]", file=sys.stderr)
        return 2
    try:
        import torch
    except Exception as e:
        print("FAIL import_torch: %s" % e)
        return 1
    bad = []
    checked = 0
    for p in sys.argv[1:]:
        try:
            import os
            if not os.path.isfile(p):
                print("WARN missing %s" % p)
                continue
            if os.path.getsize(p) <= 0:
                bad.append("%s: empty_file" % p)
                continue
            torch.load(p, map_location="cpu", weights_only=False)
            checked += 1
            print("OK %s" % p)
        except Exception as e:
            bad.append("%s: %s" % (p, e))
    if bad:
        print("FAIL")
        for x in bad:
            print(x)
        return 1
    if checked == 0:
        print("FAIL no_weights_checked")
        return 1
    print("OK checked=%d" % checked)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
