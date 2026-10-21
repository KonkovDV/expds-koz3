"""Маленький набор участков для проверки в контейнере: жёсткие ссылки, без копий."""
import os
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Sloi\expds-koz3_v0.2.0\expds-koz3")
from expds.catalog import scan_root

SRC = Path(r"F:\expds3\train-expds-3")
DST = Path(r"F:\expds3\work\smoke_in")
regions = scan_root(SRC)
pick = ["179_Гнёздово_Лесная_1"]
geo = [r for r in regions if r.rasters.get("maggr") and not r.las]
geo.sort(key=lambda r: sum(p.stat().st_size for p in r.rasters["maggr"]))
pick.append(geo[0].name)
sp = [r for r in regions if r.rasters.get("sp_or") and not r.las]
sp.sort(key=lambda r: sum(p.stat().st_size for p in r.rasters["sp_or"]))
pick.append(sp[len(sp) // 2].name)
for name in pick:
    for f in (SRC / name).rglob("*"):
        if f.is_file():
            t = DST / f.relative_to(SRC)
            t.parent.mkdir(parents=True, exist_ok=True)
            if not t.exists():
                os.link(f, t)
print(pick)
