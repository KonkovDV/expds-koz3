"""bot-6 crsfix: по всем участкам train — CRS сцены D-FINE (utm_for) и EPSG шлюза городищ (gor_gate._epsg)
старая логика (885758e: fallback как есть) против новой. Ожидание: отличается только 008_Герите."""
import json, sys
from pathlib import Path
sys.path.insert(0, "/workspace/koz3_shared/bot-6/repo")
from expds import gor_gate, sp_dfine
from expds.catalog import read_utm_json, scan_root
regs = scan_root(Path("/workspace/data/train-expds-3"))
rows, diff = 0, []
for r in regs:
    fb = read_utm_json(r.utm_json)
    for mt in ("ae", "or", "sp_or"):
        for p in r.rasters.get(mt, []):
            rows += 1
            if fb:   # старая utm_for: fallback без проверки; новая читает растр только если fallback географическая
                old, new = fb, sp_dfine.utm_for(Path(p), fb)
                if old != new: diff.append([r.name, mt, Path(p).name, old, new])
    g_old = None
    try: g_old = int(str(fb).split(":")[-1]) if fb else None
    except ValueError: pass
    g_new = gor_gate._epsg(r)
    if g_old != g_new: diff.append([r.name, "gor_gate._epsg", "", g_old, g_new])
no_fb = sum(1 for r in regs if not read_utm_json(r.utm_json))
print(json.dumps({"regions": len(regs), "rasters_ae_or_spor": rows, "regions_without_utm_json_crs": no_fb, "diff": diff}, ensure_ascii=False, indent=1))
