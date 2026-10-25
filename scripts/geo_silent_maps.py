#!/usr/bin/env python3
"""Карты геофизики трейна, на которых в точках эталона нет сигнала, — по паспорту geo_probe.

На такой карте эталонные находки лежат в ровных местах: детектор их не находит, а сеть,
обученная на её тайлах, учится срабатывать на пустом месте. Пример — серая карта участка
211 (Гнёздово, Терраса 3): 409 находок, медиана |высоких частот| у эталона 0,21 MAD,
потолок полноты 0,005, тогда как обе цветные карты того же участка дают 0,37 и 0,50.

    python scripts/geo_silent_maps.py notes/geo_probe_train.json
    python -m expds.tiles ... --markup maggr --exclude-files <имена из вывода>

Печатает имена файлов (трейн; имена участков трейна в notes/ уже есть) и сводку.

С --density 0,5,10,15,20 печатает ещё оценку опции [geophys.maggr.finds_points]
peak_density_min: потолок полноты и число пиков по картам, которые читает сдача, если
отсечку 1,0 MAD опускать (до 0,5) только на картах, где пиков меньше заданного числа
на 1000 м². Ступени отсечки паспорта дискретны, детектор берёт ровно нужное число пиков,
поэтому это оценка; порог rank-уверенности и маски она не учитывает.
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


FLOORS = ("2.0", "1.5", "1.0", "0.75", "0.5")


def density_sweep(rows: list[dict], densities: list[float], base: str = "1.0",
                  k_min: str = "0.5") -> list[dict]:
    """Потолок полноты и число пиков при опускании отсечки на картах с редкими пиками."""
    maps = [r for r in rows if r.get("used_by_detector") and r.get("gt_finds")
            and "peaks_per_1000m2" in r and "recall_ceiling" in r]
    steps = FLOORS[FLOORS.index(base):FLOORS.index(k_min) + 1]
    gt = sum(r["gt_finds"] for r in maps)
    out = []
    for dens in densities:
        hit = peaks = 0.0
        lowered = []
        for r in maps:
            floor = steps[0]
            for f in steps:
                floor = f
                if r["peaks_per_1000m2"][f] >= dens:
                    break
            if floor != base:
                lowered.append(f"{r['file']}:{floor}")
            hit += r["recall_ceiling"][floor] * r["gt_finds"]
            peaks += r["peaks"][floor]
        out.append({"peak_density_min": dens, "recall_ceiling": round(hit / max(gt, 1), 4),
                    "peaks": int(peaks), "lowered": lowered})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("probe_json")
    ap.add_argument("--max-median", type=float, default=0.5)
    ap.add_argument("--max-share", type=float, default=0.3)
    ap.add_argument("--density", default="", help="через запятую: пиков на 1000 м², например 0,5,10,15,20")
    args = ap.parse_args()
    rows = json.loads(Path(args.probe_json).read_text(encoding="utf-8"))
    found = silent(rows, args.max_median, args.max_share)
    for r in found:
        print(json.dumps(r, ensure_ascii=False))
    print("--exclude-files " + " ".join(r["file"] for r in found))
    print(json.dumps({"silent_maps": len(found), "gt_finds_on_them": sum(r["gt_finds"] for r in found)},
                     ensure_ascii=False))
    if args.density:
        for row in density_sweep(rows, [float(x) for x in args.density.split(",") if x.strip()]):
            print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
