#!/usr/bin/env python3
"""Карты геофизики трейна, на которых в точках эталона нет сигнала, — по паспорту geo_probe.

На такой карте эталонные находки лежат в ровных местах: детектор их не находит, а сеть,
обученная на её тайлах, учится срабатывать на пустом месте. Пример — серая карта участка
211 (Гнёздово, Терраса 3): 409 находок, медиана |высоких частот| у эталона 0,21 MAD,
потолок полноты 0,005, тогда как обе цветные карты того же участка дают 0,37 и 0,50.

    python scripts/geo_silent_maps.py notes/geo_probe_train.json
    python -m expds.tiles ... --markup maggr --exclude-files <имена из вывода>

Печатает имена файлов (трейн; имена участков трейна в notes/ уже есть) и сводку.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def silent(rows: list[dict], max_median: float = 0.5, max_share: float = 0.3) -> list[dict]:
    """Карты с эталоном, где медиана силы ниже max_median MAD и сигнал ≥ 1 MAD у меньше max_share точек."""
    out = []
    for r in rows:
        s = r.get("gt_strength_mad") or {}
        if not r.get("gt_finds") or not s:
            continue
        if s.get("p50", 99) < max_median and s.get("share_ge_1.0", 1.0) < max_share:
            out.append({"region": r["region"], "file": r["file"], "gt_finds": r["gt_finds"],
                        "p50_mad": s.get("p50"), "share_ge_1": s.get("share_ge_1.0"),
                        "used_by_detector": r.get("used_by_detector")})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("probe_json")
    ap.add_argument("--max-median", type=float, default=0.5)
    ap.add_argument("--max-share", type=float, default=0.3)
    args = ap.parse_args()
    rows = json.loads(Path(args.probe_json).read_text(encoding="utf-8"))
    found = silent(rows, args.max_median, args.max_share)
    for r in found:
        print(json.dumps(r, ensure_ascii=False))
    print("--exclude-files " + " ".join(r["file"] for r in found))
    print(json.dumps({"silent_maps": len(found), "gt_finds_on_them": sum(r["gt_finds"] for r in found)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
