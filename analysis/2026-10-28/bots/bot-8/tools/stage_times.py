import sys, os, time, json
sys.path.insert(0, "/workspace/koz3_shared/bot-8/repo")
os.environ["EXPDS_GEOM_CACHE"] = sys.argv[3]
import orjson
import expds.metric as M, expds.geomcache as GC, expds.fastgeom as FG
METRIC = "/workspace/koz3_shared/bot-8/repo/materials/compute_metrics.py"
pred = orjson.loads(open(sys.argv[1], "rb").read()); gt = orjson.loads(open(sys.argv[2], "rb").read())
mod = M.load_official(METRIC); calc = mod.F1Calculator()
T = {}
def wrap(name, f):
    def g(*a, **k):
        t = time.perf_counter(); r = f(*a, **k); T[name] = T.get(name, 0) + time.perf_counter() - t; return r
    return g
GC.load = wrap("gc.load", GC.load); GC.batch_key = wrap("gc.key", GC.batch_key); FG.flatten = wrap("flatten", FG.flatten)
FG.extract_many = wrap("extract", FG.extract_many); GC.save = wrap("gc.save", GC.save)
M._collect = wrap("collect", M._collect); M._pair_quality = wrap("pair_q", M._pair_quality); M._gt_tree = wrap("gt_tree", M._gt_tree)
for rep in range(2):
    T.clear(); t = time.perf_counter(); M.build_curves(pred, gt, METRIC); T["total"] = time.perf_counter() - t
    print(rep, {k: round(v, 3) for k, v in T.items()})
