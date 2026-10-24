#!/usr/bin/env python3
"""Состав файла предсказаний по слою, группе порога и классу.

Сверка табло и сдачи: группа объекта — threshold_group, иначе markup_type (post.apply_thresholds).
Копии курганов лидара в слой ae должны быть в той же группе, что в табло 0,353793:
копии мелкой сетки — ae, копии мозаики — li_ae (expds/fusion.py, COPY_GROUP).

    python scripts/check_groups.py F:/expds3/work/fusion/li_hard2_fused.geojson
    python scripts/check_groups.py out/result.geojson --config configs/default.toml

С --config печатается порог, который сдача применит к каждой группе, и сколько объектов он оставит.
"""
from __future__ import annotations

import argparse
import json
import sys
import tomllib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

KUR = ("kurgany_povrezhdennye", "kurgany_tselye")


def group_of(props: dict) -> str:
    return props.get("threshold_group") or props.get("markup_type") or "нет"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--config", default=None, help="configs/default.toml: показать порог группы и сколько он оставит")
    args = ap.parse_args()
    with Path(args.path).open(encoding="utf-8") as f:
        feats = json.load(f)["features"]
    cfg = tomllib.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else None
    rows: Counter = Counter()
    kept: Counter = Counter()
    for feat in feats:
        p = feat.get("properties") or {}
        key = (p.get("markup_type") or "нет", p.get("threshold_group") or "", p.get("class_name"))
        rows[key] += 1
        if cfg is not None:
            own = cfg.get("thresholds_by_markup", {}).get(group_of(p), {})
            tau = own.get(p.get("class_name"), cfg.get("thresholds", {}).get(p.get("class_name"),
                                                                              cfg.get("default_threshold", 0.15)))
            if float(p.get("confidence", 1.0)) >= float(tau):
                kept[key] += 1
    print(f"{'markup_type':10s} {'threshold_group':16s} {'класс':24s} {'объектов':>9s}" + ("  после порога" if cfg else ""))
    for key in sorted(rows):
        line = f"{key[0]:10s} {key[1] or '—':16s} {key[2]:24s} {rows[key]:9d}"
        print(line + (f"  {kept[key]:13d}" if cfg else ""))
    copies = {k: v for k, v in rows.items() if k[0] == "ae" and k[2] in KUR}
    if copies:
        groups = sorted({k[1] or "— (группа = ae)" for k in copies})
        print("\nкурганы слоя ae (копии лидара) лежат в группах:", ", ".join(groups))
        print("в табло 0,353793 копии мелкой сетки — группа ae (0,99053 / 0,802255), копии мозаики — li_ae")


if __name__ == "__main__":
    main()
