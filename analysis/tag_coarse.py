"""Помечает лидарные предсказания облаков с крупной сеткой DEM (шаг > 0,1 м) группой li_coarse."""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Sloi\expds-koz3_v0.2.0\expds-koz3")
import laspy

from expds.catalog import scan_root

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
factor = {}
for reg in scan_root(r"F:\expds3\train-expds-3"):
    for las in reg.las:
        with laspy.open(las) as f:
            h = f.header
        w = math.ceil((h.maxs[0] - math.floor(h.mins[0] / 0.1) * 0.1) / 0.1) + 1
        hh = math.ceil((h.maxs[1] - math.floor(h.mins[1] / 0.1) * 0.1) / 0.1) + 1
        factor[(reg.name, las.name)] = math.ceil(math.sqrt(w * hh / 60e6)) if w * hh > 60e6 else 1
fc = json.loads(src.read_text(encoding="utf-8"))
n = 0
for f in fc["features"]:
    p = f["properties"]
    if p.get("markup_type") == "li" and factor.get((p.get("region_name"), p.get("source")), 1) > 1:
        p["threshold_group"] = "li_coarse"
        n += 1
dst.write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")
coarse = sorted({r for (r, _), k in factor.items() if k > 1})
print(f"крупных облаков {sum(k > 1 for k in factor.values())} на {len(coarse)} участках, помечено {n} предсказаний")
