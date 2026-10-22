#!/usr/bin/env python3
"""Сборка из трейна папки, похожей на публичную выборку, — для хронометража.

По файлу структуры организаторов (data_structure.yaml) для каждого участка
выборки берётся участок трейна с теми же модальностями и ближайшими размерами
файлов. Копируются только снимки, LAS, UTM.json и extra/ нужных модальностей.
Разметка и готовые карты Li не копируются. Папки называются region_M001…;
имена участков выборки не используются. На одном диске файлы связываются
жёсткими ссылками, места не занимают.

    python scripts/make_mock_val.py --train F:/expds3/train-expds-3 \
        --structure C:/Sloi/files/materials/public_val_structure/data_structure.yaml \
        --out F:/expds3/mock_val

Выборку и эту папку в git не класть: это конкурсные данные (ТР 13.11).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from expds.catalog import scan_root  # noqa: E402
from expds.naming import markup_type_of  # noqa: E402

CODE_TO_MARKUP = {"La": "li", "SpOr": "sp_or", "Ae": "ae", "Or": "or", "MagGr": "maggr", "EtGr": "etgr",
                  "TerFl": "terfl"}
UNITS = {"KB": 1 / 1024, "MB": 1.0, "GB": 1024.0}


def parse_structure(path: Path) -> dict[str, dict[str, list[float]]]:
    """{участок: {markup: [размеры файлов, МБ]}} из data_structure.yaml или .md."""
    regions: dict[str, dict[str, list[float]]] = {}
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.search(r"(region_[0-9A-Z]{4})[:/]?\s*$", line)
        if m and ".tif" not in line and ".las" not in line:
            current = m.group(1)
            regions.setdefault(current, defaultdict(list))
            continue
        m = re.search(r"_(La|SpOr|Ae|Or|MagGr|EtGr|TerFl)(?:_[^()]*)?\.(las|laz|tif|tiff) \(([\d.]+) (KB|MB|GB)\)",
                      line)
        if m and current:
            regions[current][CODE_TO_MARKUP[m.group(1)]].append(float(m.group(3)) * UNITS[m.group(4)])
    return {r: dict(v) for r, v in regions.items() if v}


def train_files(reg) -> dict[str, list[Path]]:
    out = {mt: list(files) for mt, files in reg.rasters.items()}
    if reg.las:
        out["li"] = list(reg.las)
    return out


def size_mb(paths: list[Path]) -> float:
    return sum(p.stat().st_size for p in paths) / 2**20


def cost(want: dict[str, list[float]], have: dict[str, list[Path]]) -> float:
    c = 0.0
    for mt, sizes in want.items():
        files = have.get(mt)
        if not files:
            return math.inf
        c += abs(math.log((size_mb(files) + 0.1) / (sum(sizes) + 0.1)))
        c += 0.2 * abs(len(files) - len(sizes))
    return c


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--structure", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    want = parse_structure(Path(args.structure))
    train = [(reg, train_files(reg)) for reg in scan_root(args.train)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    used: dict[str, int] = defaultdict(int)
    manifest = []
    totals_want: dict[str, float] = defaultdict(float)
    totals_have: dict[str, float] = defaultdict(float)
    n = 0
    for name in sorted(want):
        spec = want[name]
        groups = [spec]
        best = min(train, key=lambda t: cost(spec, t[1]) + 0.5 * used[t[0].name])
        if math.isinf(cost(spec, best[1])):
            groups = [{mt: sizes} for mt, sizes in spec.items()]
        for part in groups:
            reg, files = min(train, key=lambda t: cost(part, t[1]) + 0.5 * used[t[0].name])
            used[reg.name] += 1
            n += 1
            dst = out / f"region_M{n:03d}"
            for mt in part:
                for f in files[mt]:
                    link(f, dst / f.relative_to(reg.path))
                totals_want[mt] += sum(part[mt])
                totals_have[mt] += size_mb(files[mt])
            for f in list(reg.boundaries) + list(reg.noise):
                if markup_type_of(f) in part:
                    link(f, dst / f.relative_to(reg.path))
            if reg.utm_json is not None:
                link(reg.utm_json, dst / "UTM.json")
            manifest.append({"mock": dst.name, "train": reg.name,
                             "modalities": {mt: {"want_mb": round(sum(part[mt]), 1),
                                                 "have_mb": round(size_mb(files[mt]), 1),
                                                 "files": len(files[mt])} for mt in part}})
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"участков выборки {len(want)}, собрано {n}")
    for mt in sorted(totals_want):
        print(f"  {mt:6s} нужно {totals_want[mt] / 1024:6.2f} ГБ, собрано {totals_have[mt] / 1024:6.2f} ГБ")


if __name__ == "__main__":
    main()
