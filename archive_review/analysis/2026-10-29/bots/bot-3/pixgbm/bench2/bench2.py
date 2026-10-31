"""Замер PIXGBM end-to-end через geo_pixgbm.region_feats (bot-3).
python bench2.py <участок> <потоки> <backend> <procs> <maps,через,запятую> <out.jsonl> [cand.pkl]
wall/cpu (cpu = родитель + дети пула, os.times), predict-only и feat — из PHASE["timing"]; кандидаты — в pkl для
сравнения точности между бэкендами."""
import os, sys, time, json, resource, pickle
reg_name, nt, backend, procs, maps, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5].split(","), sys.argv[6]
os.environ["OMP_NUM_THREADS"] = str(nt); os.environ["EXPDS_PIXGBM_THREADS"] = str(nt)
sys.path.insert(0, "/workspace/koz3_shared/bot-3/repo")
import tomllib
from expds import geo_pixgbm as G
from expds.catalog import scan_root
cfg = tomllib.load(open("/workspace/koz3_shared/bot-3/repo/configs/default.toml", "rb"))
cfg["geophys"]["pixgbm"].update(backend=backend, procs=procs, maps=maps, budget_s=10**9)
reg = [r for r in scan_root("/workspace/data/train-expds-3") if r.name.startswith(reg_name)][0]
md = "/workspace/koz3_shared/bot-3/pixgbm/models/all"
t0 = time.perf_counter(); c0 = os.times()
st = {}
feats = G.region_feats(reg, cfg, md, guard=None, status=st)
c1 = os.times(); wall = time.perf_counter() - t0
cpu = (c1.user - c0.user) + (c1.system - c0.system) + (c1.children_user - c0.children_user) + (c1.children_system - c0.children_system)
tm = G.PHASE["timing"]
row = dict(region=reg.name, threads=nt, backend=backend, procs=procs, maps=maps, wall=round(wall, 2), cpu=round(cpu, 2),
           cpu_children=round(c1.children_user - c0.children_user, 2),
           pred=round(sum(v for k, v in tm.items() if k.startswith("pred")), 2), feat_wait=round(tm.get("feat", 0), 2),
           timing={k: round(v, 2) for k, v in tm.items()}, n_cand=len(feats), steps=G.PHASE["steps"],
           maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024,
           maxrss_children_mb=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss // 1024, load1=os.getloadavg()[0])
with open(out, "a") as fh: fh.write(json.dumps(row, ensure_ascii=False) + "\n")
if len(sys.argv) > 7:
    pickle.dump([(f["class_name"], f["markup_type"], f["source"], f["confidence"], f["geom"].wkb) for f in feats], open(sys.argv[7], "wb"))
print(json.dumps(row, ensure_ascii=False))
