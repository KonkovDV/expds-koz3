#!/usr/bin/env python3
"""Посемейный стек (split_stack_by_family.py) → стек пакета-кандидата bot-7 до порогов.

Шаги в порядке watchdog.finish (порог потом подбирает честное табло по фолдам):
  sota_stack6 separate: post.multipole_object_poly(spec=None) + post.filter_finds_points_regimes
  [--twins] [post] kurgan_twins_only (bot-6): kurgan_cross_modal_twins, dedup_ae_vs_li_m = 0
  курганы 0,5 м: post.micro_circles (radius_m из --radius-map)
Реранк архитектуры здесь не делается (честно — только вложенно, bot-10/work4/rerank/nested_lofo5.py),
crop verify — уже в стеке, если стек снят табло с кропом.

    python scripts/pkg_fam.py --fam <fam> --out <fam_pkg> [--twins] [--no-stack6] [--radius-map ...]
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
from expds import post  # noqa: E402


def r3(o):
    if isinstance(o, (list, tuple)):
        return [r3(x) for x in o]
    return round(float(o), 3)


def key(d):
    return f'{d.get("class_name")}|{d.get("threshold_group") or d.get("markup_type")}'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fam", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--twins", action="store_true", help="[post] kurgan_twins_only = true")
    ap.add_argument("--no-stack6", action="store_true")
    ap.add_argument("--radius-map", default="kurgany_povrezhdennye=0.5,kurgany_tselye=0.5",
                    help="пусто — без кругов (курганы уже кругами во входе)")
    a = ap.parse_args()
    rmap = {k: float(v) for k, v in (kv.split("=") for kv in a.radius_map.split(","))} if a.radius_map else {}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tot_in, tot_out = Counter(), Counter()
    for src in sorted(Path(a.fam).glob("*.jsonl")):
        feats = []
        for line in open(src, encoding="utf-8"):
            f = json.loads(line)
            p = dict(f.get("properties") or {})
            if p.get("crs", "EPSG:3857") != "EPSG:3857":
                raise SystemExit(f"не EPSG:3857: {src.name} {p.get('crs')}")
            p["geom"] = shape(f["geometry"])
            feats.append(p)
        tot_in.update(map(key, feats))
        if not a.no_stack6:
            feats = post.multipole_object_poly(feats, None)
            feats = post.filter_finds_points_regimes(feats, None)
        if a.twins:
            feats = post.kurgan_cross_modal_twins(feats, {"dedup_ae_vs_li_m": 0.0})
        if rmap:
            feats = post.micro_circles(feats, {"radius_m": rmap})
        tot_out.update(map(key, feats))
        tmp = (out / src.name).with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fo:
            for d in feats:
                g = d.pop("geom")
                fo.write(json.dumps({"type": "Feature", "geometry": {"type": g.geom_type, "coordinates": r3(mapping(g)["coordinates"])},
                                     "properties": d}, ensure_ascii=False) + "\n")
        tmp.replace(out / src.name)
    for extra in Path(a.fam).glob("_*.json"):
        (out / extra.name).write_bytes(extra.read_bytes())
    diff = {k: [tot_in.get(k, 0), tot_out.get(k, 0)] for k in sorted(set(tot_in) | set(tot_out)) if tot_in.get(k, 0) != tot_out.get(k, 0)}
    print(json.dumps({"rows_in": sum(tot_in.values()), "rows_out": sum(tot_out.values()), "changed_groups[in,out]": diff,
                      "twins": a.twins, "stack6": not a.no_stack6, "radius_m": rmap}, ensure_ascii=False))


if __name__ == "__main__":
    main()
