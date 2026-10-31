"""bot-7: рамки сцен пакета (pipeline._with_scene_drel: map_footprint + region_crs при [crs] policy пакета)
против bot-10 work11/frame_boxes.json (read_utm_json), по всем участкам трейна. Только заголовки растров."""
import json, sys, tomllib
sys.path.insert(0, "/workspace/koz3_shared/bot-7/repo")
from expds.catalog import scan_root
from expds.geophys import map_footprint
from expds.pipeline import region_crs
cfg = tomllib.load(open("/workspace/koz3_shared/bot-7/repo/configs/default.toml", "rb"))
ref = json.load(open("/workspace/koz3_shared/bot-10/work11/frame_boxes.json"))["boxes"]
n = same = 0; diff = []
for reg in scan_root("/workspace/data/train-expds-3"):
    for mt in ("ae", "sp_or", "or"):
        mine = []
        for p in reg.rasters.get(mt, []):
            try:
                fp = map_footprint(p, region_crs(reg, cfg))
            except Exception:
                fp = None
            if fp is not None:
                mine.append(list(fp.bounds))
        theirs = ref.get(reg.name, {}).get(mt, [])
        if not mine and not theirs:
            continue
        n += 1
        ok = len(mine) == len(theirs) and all(max(abs(a - b) for a, b in zip(x, y)) < 1e-3 for x, y in zip(sorted(mine), sorted(theirs)))
        same += ok
        if not ok:
            diff.append([reg.name, mt, len(mine), len(theirs)])
print(json.dumps({"pairs": n, "same": same, "diff": diff[:30]}, ensure_ascii=False))
