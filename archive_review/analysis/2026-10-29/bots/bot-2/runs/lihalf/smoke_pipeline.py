"""Смоук li_half через pipeline.lidar_feats на реальных LAZ (только модель li, 1 поток)."""
import json, os, resource, sys, time, logging
from pathlib import Path
WT = Path(sys.argv[1]); sys.path.insert(0, str(WT))
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
from expds import pipeline
from expds.catalog import scan_root
cfg = pipeline.load_config(WT / "configs/default.toml")
cfg["models"] = {"li": cfg["models"]["li"]}
cfg["hillfort"] = {"enabled": False}
cfg.setdefault("fusion", {})["copy"] = []
cfg["fusion"]["li_to_ae"] = False
cfg["li_half"]["enabled"] = os.environ.get("SMOKE_HALF", "1") == "1"
cfg["thresholds_by_markup"]["li_half"] = {"kurgany_povrezhdennye": 0.0, "kurgany_tselye": 0.0}
cfg["lidar"]["prefetch"] = 0
out = {}
for reg in scan_root(sys.argv[2]):
    t = time.monotonic()
    feats = pipeline.lidar_feats(reg, cfg, pipeline.TimeGuard(10_000), {}, WT, None)
    groups = {}
    for f in feats:
        k = f"{f.get('threshold_group') or f.get('markup_type')}:{f['class_name']}"
        groups[k] = groups.get(k, 0) + 1
    out[reg.name] = {"seconds": round(time.monotonic() - t, 1), "groups": groups}
    print(reg.name, out[reg.name], flush=True)
out["maxrss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
print(json.dumps(out, ensure_ascii=False))
Path(sys.argv[3]).write_text(json.dumps(out, ensure_ascii=False, indent=1))
