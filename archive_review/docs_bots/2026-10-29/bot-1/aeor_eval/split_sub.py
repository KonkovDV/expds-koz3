import sys, json, glob
sys.path.insert(0, "/workspace/koz3_shared/bot-10/repo")
from expds.naming import site_family
from pathlib import Path
import importlib.util
spec = importlib.util.spec_from_file_location("sp", "/workspace/koz3_shared/bot-10/repo/scripts/split_stack_by_family.py"); sp = importlib.util.module_from_spec(spec); spec.loader.exec_module(sp)
CL = json.load(open("/workspace/koz3_shared/bot-10/work/family_clusters.json"))["clusters"]
UNIT = {m: "+".join(sorted(ms)) for ms in CL.values() for m in ms}
regs = {p.split("/")[-3] for m in ("Ae", "Or") for p in glob.glob(f"/workspace/data/train-expds-3/*/*/*_{m}_*")}
fams = {site_family(r) for r in regs}
units = sorted({UNIT.get(f, f) for f in fams}); members = sorted({m for u in units for m in u.split("+")})
json.dump({"regions": sorted(regs), "units": units, "member_families": members}, open("touched.json", "w"), ensure_ascii=False, indent=1)
out = Path("fam_base"); out.mkdir(exist_ok=True)
hs = {}
for feat, raw in sp.iter_features(Path("/workspace/rerank/stack_arch.geojson")):
    fam = site_family(str(feat["properties"].get("region_name", "default_source")))
    if fam not in members: continue
    if fam not in hs: hs[fam] = open(out / f"{fam}.jsonl", "w", encoding="utf-8")
    hs[fam].write(raw + "\n")
for h in hs.values(): h.close()
print(len(units), len(members), len(hs))
