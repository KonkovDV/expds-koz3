#!/usr/bin/env python3
"""Кандидаты городищ по форме рельефа на всех участках с LAS — файл для табло.

Обучения нет, поэтому файл «вне обучения» по построению: параметры генератора
(`expds.hillfort.HillfortParams`) не подбирались по эталону. Табло подбирает порог
группы `gor_phys`; если площадки попадают реже 7,8 %, порог выключит группу.

    python scripts/hillfort_oof.py --root F:/expds3/train-expds-3 --out F:/expds3/work/gor_phys.geojson
    python scripts/scoreboard.py ... --add F:/expds3/work/gor_phys.geojson:gor_phys --honest 10
    python scripts/hillfort_oof.py ... --set scarp_min_deg=15 --set min_rel_height_m=1.0 --out ...

Тот же файл годится источником для `oof_transform.py rescue` (лучшая площадка участка).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
log = logging.getLogger("hillfort_oof")

from expds import hillfort, post, writer
from expds.catalog import read_utm_json, scan_root
from expds.lidar import build_dem


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dem-res", type=float, default=0.5, help="шаг ЦМР из LAS, м (генератор работает на 1 м)")
    ap.add_argument("--set", action="append", default=[], help="параметр=значение HillfortParams (литерал TOML)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = {}
    for spec in args.set:
        key, _, raw = spec.partition("=")
        cfg[key.strip()] = tomllib.loads(f"v = {raw.strip()}")["v"]
    p = hillfort.params_from(cfg)
    feats: list[dict] = []
    n_las = 0
    for reg in scan_root(args.root):
        crs = read_utm_json(reg.utm_json)
        for las in reg.las:
            try:
                d = build_dem(las, res=args.dem_res, fill="pyramid", crs=crs)
                feats += hillfort.candidates(d, reg.name, las.name, p)
                n_las += 1
            except Exception:
                log.exception("%s: пропуск", las.name)
    out = post.finalize(feats)
    fc = writer.to_feature_collection(out)
    for f, src in zip(fc["features"], out):
        f["properties"]["threshold_group"] = src.get("threshold_group", "gor_phys")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"las": n_las, "written": len(out), "params": vars(p)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
