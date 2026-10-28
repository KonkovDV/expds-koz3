#!/usr/bin/env python3
"""Склейка пофолдовых OOF разных ядер в один OOF на вариант (только на боксе/ПК).
python3 merge_oof.py OUT.geojson IN1.geojson [IN2 ...]
Проверяет: регион не встречается в двух входах, fid перенумеровываются 0..N-1. Пишет потоково."""
import json, sys
out, ins = sys.argv[1], sys.argv[2:]
seen, n = {}, 0
with open(out, "w") as fo:
    fo.write('{"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::3857"}}, "features": [\n')
    for p in ins:
        fc = json.load(open(p))
        regs = {f["properties"]["region_name"] for f in fc["features"]}
        for r in regs:
            if r in seen:
                sys.exit(f"регион {r} в {seen[r]} и {p}")
            seen[r] = p
        for f in fc["features"]:
            f["properties"]["fid"] = n
            fo.write((",\n" if n else "") + json.dumps(f, ensure_ascii=False))
            n += 1
        del fc
    fo.write("\n]}\n")
print(f"{n} признаков, {len(seen)} участков → {out}")
