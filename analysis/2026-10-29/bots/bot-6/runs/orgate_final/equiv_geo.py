"""Эквивалентность на 4 участках по пути пайплайна: объекты EPSG:3857 из stack_arch.geojson → orgate.decide
(rerank_arch.table → UTM по центроиду, LAS из scan_root). gq/пул — глобальные по всему стеку A (как при обучении),
подставлены вместо пересчёта по 4 участкам. Сверка: массивы table() с table.pkl и удалённые id с offline_decisions."""
import json, pickle, resource, sys, time, gc
from pathlib import Path
import numpy as np
from shapely.geometry import shape
sys.path.insert(0, "/workspace/koz3_shared/bot-6/repo")
from expds import orgate, rerank_arch as R
from expds.catalog import scan_root
W = Path("/workspace/koz3_shared/bot-6/work/orgate_final")
tab = pickle.load(open("/workspace/rerank/work/table.pkl", "rb"))
T = {k: tab[k] for k in ("cls", "conf", "grp", "mk", "reg", "area", "perim", "cx", "cy", "bw", "bh", "nv")}
del tab; gc.collect()
ug, gcode = np.unique(T["grp"], return_inverse=True)
GQ = R.group_quantile(gcode, T["cls"], T["conf"]); POOL = R.pool_mask(gcode, T["cls"], T["conf"])
cur = {}
_gq, _pm = R.group_quantile, R.pool_mask
R.group_quantile = lambda grp, cls, conf: GQ[cur["ix"]]
R.pool_mask = lambda grp, cls, conf: POOL[cur["ix"]]
cat = orgate.catalog_info(scan_root("/workspace/data/train-expds-3"))
off = json.loads((W / "offline_decisions.json").read_text())
data = pickle.load(open(W / "equiv/regions4.pkl", "rb"))
model = orgate.load({"model": str(W / "orgate_g.txt"), "meta": str(W / "orgate_g.json")})
res = {}
for r, rows in sorted(data.items()):
    ids = np.array([i for i, _ in rows]); cur["ix"] = ids
    feats = [dict(f["properties"], geom=shape(f["geometry"])) for _, f in rows]
    Tp = R.table(feats)
    tdiff = {k: float(np.max(np.abs(Tp[k].astype(float) - T[k][ids].astype(float)))) for k in ("area", "perim", "cx", "cy", "bw", "bh", "nv", "conf", "cls")}
    teq = {k: bool((Tp[k].astype(str) == T[k][ids].astype(str)).all()) for k in ("grp", "mk", "reg")}
    t = time.time()
    rest, kept, info = orgate.decide(feats, {}, {r: cat[r]}, model=model)
    dt = time.time() - t
    rp = sorted(int(ids[i]) for i in info["removed_idx"]); kp = sorted(int(ids[i]) for i in info["kept_idx"])
    res[r] = dict(n_rows=len(feats), las=cat[r]["las"], n_arch=len(info["target"]), removed=len(rp), kept=len(kp),
                  removed_equal_offline=rp == sorted(off[r]["removed"]), kept_equal_offline=kp == sorted(off[r]["kept"]),
                  rest_plus_kept=len(rest) + len(kept), table_maxabs=tdiff, table_str_equal=teq, t_decide_s=round(dt, 3),
                  removed_ids=rp, kept_ids=kp)
    print(r, {k: v for k, v in res[r].items() if not k.endswith("_ids")}, flush=True)
res["_maxrss_kb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print("maxrss KB", res["_maxrss_kb"])
(W / "equiv/equiv_geo.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
