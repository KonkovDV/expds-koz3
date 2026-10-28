"""Сверка временного EXPDS_MATCH_SCOPE=site с официальным calculate_f1 по участкам (и global — по всему файлу)."""
import sys, os, json, importlib
from pathlib import Path
sys.path.insert(0, "repo")
pred = json.load(open(sys.argv[1], encoding="utf-8")); gt = json.load(open(sys.argv[2], encoding="utf-8"))
res = {}
for scope in ("global", "site"):
    os.environ["EXPDS_MATCH_SCOPE"] = scope
    from expds import metric as M
    M._UTM_GEOM_CACHE.clear(); M._SHM_CACHE_DIR = Path("/proc/expds_no_cache")
    cv = M.build_curves(pred, gt)
    res[scope] = {c: [int(v.hit.sum()), int(v.conf.size - v.hit.sum()), int(v.n_gt)] for c, v in cv.items()}
mod = M.load_official(None); calc = mod.F1Calculator()
def counts(r):
    return {c: [int(v["tp"]), int(v["fp"]), int(v["tp"] + v["fn"])] for c, v in r["per_class"].items()}
off_g = counts(calc.calculate_f1(pred, gt))
ps, gs = mod.split_merged_by_source_and_class(pred), mod.split_merged_by_source_and_class(gt)
off_s = {}
for site in set(ps) | set(gs):
    pf = [f for fc in ps.get(site, {}).values() for f in fc["features"]]
    gf = [f for fc in gs.get(site, {}).values() for f in fc["features"]]
    r = counts(calc.calculate_f1({"type": "FeatureCollection", "features": pf}, {"type": "FeatureCollection", "features": gf}))
    for c, v in r.items():
        a = off_s.setdefault(c, [0, 0, 0]); a[0] += v[0]; a[1] += v[1]; a[2] += v[2]
out = {"ours_global": res["global"], "official_global": off_g, "ours_site": res["site"], "official_site": off_s}
out["equal_global"] = all(res["global"].get(c, [0, 0, 0]) == v for c, v in off_g.items() if v != [0, 0, 0])
out["equal_site"] = all(res["site"].get(c, [0, 0, 0]) == v for c, v in off_s.items() if v != [0, 0, 0])
print(json.dumps(out, ensure_ascii=False))
Path(sys.argv[3]).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
