"""Совместный замер: реранкер архитектуры (/workspace/rerank nested3.py, вариант 'arch') → микрокруги.

Повторяет nested3.honest_with только для класса arkhitektury (sweep_groups идёт по классу отдельно,
остальные классы от варианта не зависят и берутся из counts_<layout>_arch.pkl реранкера).
Оценки реранкера (outer/inner) — готовые из nested_v2.npz / v3/nested_seedS.npz, без переобучения LightGBM.
Варианты: A (conf, круг 2 м), R (реранкер, 2 м), M (conf, микрокруг), RM (реранкер, микрокруг).
"""
import json, sys, pickle, copy, time, collections, numpy as np
from pathlib import Path
R = Path("/workspace/koz3_shared/bot-5/repo"); sys.path.insert(0, str(R))
from expds import metric as M, post
from expds.naming import site_family
from shapely.geometry import shape, mapping
from shapely.strtree import STRtree
M._SHM_CACHE_DIR = Path("/workspace/koz3_shared/bot-5/tmp/shm_off/x")
RR = Path("/workspace/rerank"); OUT = Path("/workspace/koz3_shared/bot-5/analysis/joint")
OUT.mkdir(parents=True, exist_ok=True)
A = "arkhitektury"; RAD = 0.25
mod = M.load_official(str(R / "materials" / "compute_metrics.py")); calc = mod.F1Calculator()
rows = np.load("arch_rows.npy")
st = json.load(open("stack_arch_only.geojson")); gtA = json.load(open("gt_train_v2_arch.geojson"))
assert len(st["features"]) == len(rows)
conf = np.array([mod._get_confidence(f) for f in st["features"]])
reg = np.array([str(f["properties"].get("region_name", "default_source")) for f in st["features"]], dtype=object)
grp = np.array([M.threshold_group(f["properties"]) for f in st["features"]], dtype=object)

gts = M._collect(mod, calc, gtA, True)[A]; G = [x[0] for x in gts]; GREG = collections.Counter(x[2] for x in gts)
tree = STRtree(G)
def pairs_for(fc):
    pr = M._collect(mod, calc, fc, False)[A]; M._UTM_GEOM_CACHE.clear(); assert len(pr) == len(rows)
    geo = [x[0] for x in pr]; PL = {}
    qp, qg = tree.query(geo)
    for a, b in zip(qp, qg):
        if geo[a].geom_type != "Polygon": continue
        ok, s = calc._match_quality(geo[a], G[b])
        if ok: PL.setdefault(int(a), []).append((int(b), float(s)))
    return {p: sorted(v) for p, v in PL.items()}
def micro_fc(fc):
    feats = [dict(f["properties"], geom=shape(f["geometry"]), class_name=A) for f in fc["features"]]
    feats = post.micro_circles(feats, {"classes": [A], "radius_m": RAD})
    return {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": mapping(f.pop("geom")), "properties": f} for f in feats]}
t = time.time(); PL2 = pairs_for(st); PLm = pairs_for(micro_fc(st)); del st
print("pairs", len(PL2), len(PLm), f"{time.time()-t:.0f}s", flush=True)

def curve(score, PL):  # = fast.curves для одного класса
    o = np.argsort(-score, kind="stable"); hit = np.zeros(len(o), bool); matched = set()
    for r, p in enumerate(o):
        best, bj = -np.inf, -1
        for j, s in PL.get(int(p), ()):
            if j in matched: continue
            if s > best + 1e-9: best, bj = s, j
        if bj >= 0: matched.add(bj); hit[r] = True
    return M.ClassCurve(conf=score[o].astype(np.float64), hit=hit, pred_region=list(reg[o]), n_gt=len(G),
                        gt_regions=GREG, pred_group=list(grp[o]))
def honest_arch(score_fn, PL, folds, regions):
    per = {}
    for k, held in enumerate(folds):
        cv = curve(score_fn(k), PL); res = M.sweep_groups({A: M.curve_where(cv, regions - held)})
        taus = {g: gv["threshold"] for g, gv in res[A]["groups"].items()}
        hc = M.curve_where(cv, held)
        for r, n in hc.gt_regions.items(): per.setdefault(r, [0, 0, 0])[2] += int(n)
        if hc.conf.size == 0: continue
        tau = np.array([taus.get(g, np.inf) for g in hc.pred_group]); kept = hc.conf >= tau
        for r, h in zip(np.asarray(hc.pred_region, dtype=object)[kept], hc.hit[kept]):
            per.setdefault(r, [0, 0, 0])[0 if h else 1] += 1
    return per

gt_regs = set(json.load(open("stack_regions.json"))["pred_regions"])
full_gt = json.load(open("/workspace/rerank/gt_train_v2.geojson"))
for f in full_gt["features"]:
    p = f.get("properties") or {}
    if mod.normalize_class_name(p.get("class_name", "unknown")) in set(calc.target_classes) and f.get("geometry"):
        gt_regs.add(str(p.get("region_name", "default_source")))
del full_gt
regions = gt_regs
boxes = json.loads((R / "notes" / "las_boxes_59.json").read_text(encoding="utf-8"))
new = {l.strip() for l in (R / "notes" / "regions_las_new.txt").read_text(encoding="utf-8").splitlines() if l.strip()}
optical = {r for r in regions if r} - set(boxes)
def random_family_folds(regions, k, seed):
    fams = {}
    for r in regions: fams.setdefault(site_family(r), set()).add(r)
    nm = sorted(fams); rng = np.random.default_rng(seed); lab = rng.permutation(len(nm)) % k
    return [set().union(*(fams[n] for n, l in zip(nm, lab) if l == i)) for i in range(k)]
def S(c): return [M.score_counts(c)[0], M.score_counts(c, new)[0], M.score_counts(c, optical)[0]]

layouts = sys.argv[1:] or ["canon", "seed0", "seed1", "seed2", "seed3", "seed4"]
summary = {}
for L in layouts:
    t = time.time()
    folds = M.family_folds(regions, site_family, 10) if L == "canon" else random_family_folds(regions, 10, int(L[4:]))
    fold_of = {r: i for i, f in enumerate(folds) for r in f}; fid = np.array([fold_of[r] for r in reg])
    z = np.load(RR / ("work/nested_v2.npz" if L == "canon" else f"work/v3/nested_{L}.npz"))
    outer = z["outer"][rows].copy(); inner = z["inner"][:, rows].copy(); del z
    np.savez_compressed(OUT / f"rerank_arch_scores_{L}.npz", outer=outer, inner=inner)
    def mix(k):
        s = conf.copy(); tr = fid != k; s[tr] = inner[k][tr]; s[~tr] = outer[~tr]; return s
    other = pickle.load(open(RR / f"work/v3/counts_{L}_arch.pkl", "rb"))
    res_ref = json.load(open(RR / f"work/v3/res_{L}.json"))
    cnt = {}
    for nm, fn, PL in (("A", lambda k: conf, PL2), ("R", mix, PL2), ("M", lambda k: conf, PLm), ("RM", mix, PLm)):
        c = copy.deepcopy(other); c[A] = honest_arch(fn, PL, folds, regions); cnt[nm] = c
    out = {"layout": L, "check": {"A_ours": S(cnt["A"]), "A_rerank_res": res_ref["base"],
                                  "R_ours": S(cnt["R"]), "R_rerank_res": res_ref["arch"]["s"],
                                  "R_arch_counts_equal": cnt["R"][A] == other[A]}}
    for nm in ("A", "R", "M", "RM"):
        a = M.score_counts(cnt[nm])[1][A]
        out[nm] = {"s": S(cnt[nm]), "arch": {k: a[k] for k in ("f1", "tp", "fp")}}
    for x, y in (("R", "A"), ("M", "A"), ("RM", "A"), ("RM", "R")):
        out[f"{x}-{y}"] = {"d": [p - q for p, q in zip(out[x]["s"], out[y]["s"])],
                           "boot": M.paired_bootstrap_counts(cnt[x], cnt[y], site_family, 1000)}
    pickle.dump(cnt, open(OUT / f"counts_{L}.pkl", "wb"))
    json.dump(out, open(OUT / f"res_{L}.json", "w"), indent=1, default=float)
    summary[L] = out
    print(L, f"{time.time()-t:.0f}s", json.dumps(out["check"], default=float), flush=True)
    for k in ("R-A", "M-A", "RM-A", "RM-R"):
        b = out[k]["boot"]; d = out[k]["d"]
        print(f"  {k:5s} all {d[0]:+.5f} new {d[1]:+.5f} opt {d[2]:+.5f} | boot {b['mean']:+.5f} [{b['p5']:+.5f};{b['p95']:+.5f}] P {b['p_gt0']:.3f}", flush=True)
json.dump(summary, open(OUT / ("summary.json" if len(layouts) > 1 else f"summary_{layouts[0]}.json"), "w"), indent=1, default=float)
