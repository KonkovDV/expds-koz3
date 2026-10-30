#!/usr/bin/env python3
"""Совпадает ли эталон снимка с самим снимком: отклик «пятна» в точках эталона при сдвигах ±max.

Паспорт для оптики, как geo_probe для магниторазведки. Для каждого снимка (аэрофото,
спутник, ортофото) с эталонными курганами своего слоя:

1. снимок читается на шаге --res метров на земле, каналы усредняются;
2. отклик |DoG| на масштабе медианного кургана этого слоя (expds.register.blob_response);
3. средний отклик в центрах эталона при каждом сдвиге (dx, dy) до --max-shift метров.

Если пик в нуле — метки сидят на пятнах снимка. Если пик устойчиво в стороне (z ≥ 4,
сдвиг ≥ 1,5 м), метки сдвинуты относительно снимка: сеть этого слоя учит смещённую
точку, и кэш обучения надо собирать со сдвигом меток. Сравнение со сдвигом слоёв
эталона (scripts/layer_shift.py) говорит, что сдвинуто — снимок или разметка.

    python scripts/image_label_shift.py --root F:/expds3/train-expds-3 --gt F:/expds3/work/gt_train_v2.geojson \
        --markups ae sp_or --out notes/image_label_shift.json

С --labels вместо эталона берутся объекты файла (курганы лидара вне обучения, уверенность
от --labels-min): это сдвиг самого снимка относительно земли — тот, что [fusion] register
меряет на инференсе без эталона, и его можно сравнить со сдвигом слоёв эталона.
С --labels эталон не нужен (--gt можно не давать), поэтому так же меряется публичная
выборка по курганам лидара из своего result.geojson. Вывод с именами её участков в
notes/ и в git не класть.

    python scripts/image_label_shift.py --root <папка выборки> --labels <result.geojson> \
        --labels-min 0.5 --radius 5 --markups ae sp_or --out <рабочая папка>/public_photo_shift.json

dx — на восток, dy — на юг (строки растра), dn = −dy — на север, метры.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

KURGANS = ("kurgany_povrezhdennye", "kurgany_tselye")


def gt_by_region(path: str, classes: tuple[str, ...], markups: set[str],
                 min_conf: float | None = None) -> dict[str, dict[str, list]]:
    from shapely.geometry import shape

    fc = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for f in fc.get("features", []):
        p = f.get("properties") or {}
        if min_conf is not None and float(p.get("confidence", 1.0)) < min_conf:
            continue
        if p.get("class_name") in classes and p.get("markup_type") in markups:
            g = shape(f["geometry"])
            if not g.is_empty:
                out[p.get("region_name") or ""][p["markup_type"]].append(g)
    return out


def scan_raster(path: Path, geoms: list, res_m: float, max_shift_m: float, fallback_crs: str | None,
                radius_m: float | None = None) -> dict:
    """Сдвиг эталона относительно одного снимка (expds.register.raster_label_shift)."""
    from expds import register

    return register.raster_label_shift(path, geoms, res_m, max_shift_m, fallback_crs, radius_m)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="корень выборки (трейн или публичная)")
    ap.add_argument("--gt", default="", help="эталон; не нужен вместе с --labels")
    ap.add_argument("--markups", nargs="*", default=["ae", "sp_or", "or"])
    ap.add_argument("--classes", nargs="*", default=list(KURGANS))
    ap.add_argument("--res", type=float, default=1.0, help="шаг чтения снимка на земле, м")
    ap.add_argument("--max-shift", type=float, default=20.0, help="сдвиги до ±, м")
    ap.add_argument("--labels", default="",
                    help="вместо эталона снимка — объекты этого файла (например, курганы лидара вне обучения): "
                         "сдвиг самого снимка относительно земли, как его увидит [fusion] register")
    ap.add_argument("--labels-markup", default="li")
    ap.add_argument("--labels-min", type=float, default=0.99, help="только объекты не слабее этой уверенности")
    ap.add_argument("--radius", type=float, default=None,
                    help="радиус пятна, м (по умолчанию — медианный полигон; для кругов --labels задать 5)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if not args.gt and not args.labels:
        ap.error("нужен --gt или --labels")
    from expds.catalog import read_utm_json, scan_root

    if args.labels:
        lab = gt_by_region(args.labels, tuple(args.classes), {args.labels_markup}, args.labels_min)
        gt = {region: {mt: layers.get(args.labels_markup, []) for mt in args.markups}
              for region, layers in lab.items()}
    else:
        gt = gt_by_region(args.gt, tuple(args.classes), set(args.markups))
    regs = {r.name: r for r in scan_root(args.root)}
    rows = []
    for region, layers in sorted(gt.items()):
        reg = regs.get(region)
        if reg is None:
            continue
        for mt, geoms in layers.items():
            for path in reg.rasters.get(mt, []):
                try:
                    row = scan_raster(path, geoms, args.res, args.max_shift, read_utm_json(reg.utm_json),
                                      args.radius)
                except Exception as exc:
                    row = {"error": str(exc)}
                if row:
                    rows.append({"region": region, "markup": mt, "file": path.name, **row})
                    print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    for mt in args.markups:
        mine = [r for r in rows if r["markup"] == mt and "z_peak" in r]
        sysr = [r for r in mine if r["systematic"]]
        print(f"{mt}: снимков {len(mine)}, метки сдвинуты относительно снимка у {len(sysr)}"
              + (f", медиана {np.median([r['shift_m'] for r in sysr]):.1f} м" if sysr else "")
              + (f"; у остальных пик в нуле z {np.median([r['z_zero'] for r in mine if not r['systematic']]):.1f}"
                 if len(mine) > len(sysr) else ""), flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"args": vars(args), "rasters": rows}, ensure_ascii=False, indent=1),
                                  encoding="utf-8")


if __name__ == "__main__":
    main()
