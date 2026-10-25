#!/usr/bin/env python3
"""Один кэш тайлов из нескольких типов снимков — для общей сети на аэрофото, спутник и ортофото.

Сейчас у каждого типа снимков своя сеть, а у ортофото своей сети нет. Общий кэш нужен, чтобы одна сеть видела и спутник, и ортофото: редкого класса в обучении становится больше.

    python -m expds.tiles --root F:/expds3/train-expds-3 --gt F:/expds3/work/gt_train_v2.geojson `
        --markup or --out F:/expds3/work/cache/or_arch_r03 --res 0.3 --tile 256 --classes arkhitektury
    (то же для sp_or и ae с теми же --res, --tile и --classes)
    python scripts/cache_merge.py --src F:/expds3/work/cache/or_arch_r03 F:/expds3/work/cache/spor_arch_r03 `
        F:/expds3/work/cache/ae_arch_r03 --out F:/expds3/work/cache/surf_arch_r03
    python scripts/nn_oof.py --markup or --predict-markups or sp_or ae --cache F:/expds3/work/cache/surf_arch_r03 ...

Классы, шаг, тайл, число каналов и единицы пикселя у источников обязаны совпадать.
Шарды не копируются, а связываются жёсткими ссылками (копия, если ссылки недоступны).
В индекс добавляется столбец markup; семьи те же, поэтому фолд держит всю семью
во всех типах снимков сразу.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

#: Поля meta.json, которые обязаны совпасть у всех источников.
SAME = ("classes", "res", "tile", "channels", "pixel_units", "li_source")


def _link(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def merge(sources: list[Path], out: Path) -> dict:
    """Сливает кэши в out. Возвращает meta нового кэша."""
    metas = [json.loads((s / "meta.json").read_text(encoding="utf-8")) for s in sources]
    for key in SAME:
        vals = {json.dumps(m.get(key), ensure_ascii=False) for m in metas}
        if len(vals) > 1:
            raise SystemExit(f"у кэшей разное {key}: {sorted(vals)}")
    if any(m.get("li_source") for m in metas):
        raise SystemExit("кэши лидара не сливаются: у них свои каналы рельефа")
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    offset = 0
    for src, meta in zip(sources, metas):
        shards = sorted(src.glob("images_*.npy"))
        for f in shards:
            n = int(f.stem.split("_")[1])
            _link(f, out / f"images_{n + offset:04d}.npy")
            _link(src / f"masks_{n:04d}.npy", out / f"masks_{n + offset:04d}.npy")
        with (src / "index.csv").open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                row["shard"] = str(int(row["shard"]) + offset)
                row["markup"] = row.get("markup") or meta.get("markup") or ""
                rows.append(row)
        offset += len(shards)
    fields = ["shard", "idx", "region", "family", "file", "cx", "cy", "positive", "markup"]
    with (out / "index.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    meta = dict(metas[0])
    meta["markup"] = "+".join(m.get("markup") or "?" for m in metas)
    meta["sources"] = [str(s) for s in sources]
    meta["flips"] = all(m.get("flips", False) for m in metas)
    meta["min_area_m2"] = min((m.get("min_area_m2") or 0) for m in metas)
    for key in ("tiles", "positive"):
        if all(key in m for m in metas):
            meta[key] = sum(int(m[key]) for m in metas)
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", nargs="+", required=True, help="кэши expds.tiles одного рецепта")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    meta = merge([Path(s) for s in args.src], Path(args.out))
    print(json.dumps({k: meta.get(k) for k in ("markup", "classes", "res", "tile", "tiles", "positive")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
