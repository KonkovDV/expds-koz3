"""Сводка tt.sh: объекты по участкам и markup_type, фазы и предупреждения о бюджете из логов."""
import collections, json, re, sys
from pathlib import Path
W = Path(sys.argv[1])
for tag in ("new_full", "old_full", "new_cut", "old_cut"):
    p = W / f"res_{tag}.geojson"
    if not p.exists(): print(tag, "НЕТ ФАЙЛА ОТВЕТА"); continue
    fc = json.loads(p.read_text(encoding="utf-8"))
    c = collections.Counter(f["properties"].get("markup_type") for f in fc["features"])
    regs = {f["properties"].get("region_name") for f in fc["features"]}
    log = (W / f"run_{tag}.log").read_text(encoding="utf-8", errors="replace")
    warn = [l.split(": ", 1)[-1][:120] for l in log.splitlines() if re.search(r"бюджет|оптику времени|мягкий потолок|лидар оставит", l)]
    ph = re.findall(r"фазы: (.*)", log)
    print(f"{tag}: объектов {len(fc['features'])}, участков с объектами {len(regs)}, по типу {dict(sorted(c.items(), key=str))}")
    print("   фазы:", ph[-1][:300] if ph else "-")
    for w in warn[:8]: print("   !", w)
