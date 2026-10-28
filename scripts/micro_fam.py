#!/usr/bin/env python3
"""Посемейный стек (split_stack_by_family.py) → тот же стек после post.micro_circles.

Преобразование до порогов: число объектов, уверенности и группы не меняются,
меняется только геометрия. Кривые потом строит family_curves.py.

    python scripts/micro_fam.py --fam ../bot-10/work/fam --out work/fam_bundle \
        --radius-map arkhitektury=0.25,kurgany_povrezhdennye=0.5,kurgany_tselye=0.5
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from shapely.geometry import mapping, shape

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from expds.post import micro_circles  # noqa: E402


def r3(o):
    if isinstance(o, (list, tuple)):
        return [r3(x) for x in o]
    return round(float(o), 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fam", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--radius-map", required=True)
    a = ap.parse_args()
    rmap = {k: float(v) for k, v in (kv.split("=") for kv in a.radius_map.split(","))}
    spec = {"radius_m": rmap}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stat = Counter()
    for src in sorted(Path(a.fam).glob("*.jsonl")):
        dst = out / src.name
        tmp = dst.with_suffix(".tmp")
        with open(src, encoding="utf-8") as fi, open(tmp, "w", encoding="utf-8") as fo:
            for line in fi:
                f = json.loads(line)
                p = f.get("properties") or {}
                cls = p.get("class_name")
                if cls in rmap and f.get("geometry"):
                    if p.get("crs", "EPSG:3857") != "EPSG:3857":
                        raise SystemExit(f"не EPSG:3857: {src.name} {p.get('crs')}")
                    g = shape(f["geometry"])
                    ng = micro_circles([{"class_name": cls, "geom": g}], spec)[0]["geom"]
                    if ng is not g:
                        f["geometry"] = {"type": "Polygon", "coordinates": r3(mapping(ng)["coordinates"])}
                        stat[cls] += 1
                fo.write(json.dumps(f, ensure_ascii=False) + "\n")
        tmp.replace(dst)
    for extra in Path(a.fam).glob("_*.json"):
        (out / extra.name).write_bytes(extra.read_bytes())
    print(json.dumps({"changed": stat, "spec": spec}, ensure_ascii=False))


if __name__ == "__main__":
    main()
