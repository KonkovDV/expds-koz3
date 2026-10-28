"""Честная проверка микрокругов архитектуры на стеке /workspace/rerank/stack_arch.geojson (= база honest-10 0,314615).

Класс считается отдельно (пороги групп подбираются внутри класса), счётчики архитектуры
подставляются в базовые счётчики всех классов (/workspace/rerank/work/base_honest.pkl, те же фолды семей).
"""
import json, sys, pickle, copy
from pathlib import Path
R = Path("/workspace/koz3_shared/bot-5/repo"); sys.path.insert(0, str(R))
from expds import metric as M, post
from expds.naming import site_family
from shapely.geometry import shape, mapping
M._SHM_CACHE_DIR = Path("/workspace/koz3_shared/bot-5/tmp/shm_off/x")
METRIC = str(R / "materials" / "compute_metrics.py")
A = "arkhitektury"
st = json.load(open("stack_arch_only.geojson")); gt = json.load(open("gt_train_v2_arch.geojson"))
base = pickle.load(open("/workspace/rerank/work/base_honest.pkl", "rb"))
folds = base["folds"]
boxes = json.loads((R / "notes" / "las_boxes_59.json").read_text(encoding="utf-8"))
new = {l.strip() for l in (R / "notes" / "regions_las_new.txt").read_text(encoding="utf-8").splitlines() if l.strip()}
regions_all = {r for by in base["counts"].values() for r in by}
optical = regions_all - set(boxes)

def micro(fc, r):
    feats = [dict(f["properties"], geom=shape(f["geometry"]), class_name=A) for f in fc["features"]]
    feats = post.finalize(post.micro_circles(feats, {"classes": [A], "radius_m": r}))
    return {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": mapping(f.pop("geom")),
            "properties": f} for f in feats]}

def run(fc, name):
    curves = {A: M.build_curves(fc, gt, METRIC)[A]}
    ins = M.sweep_groups(curves)[A]
    counts, info = M.honest_counts(curves, folds)
    full = copy.deepcopy(base["counts"]); full[A] = counts[A]
    s = [M.score_counts(full)[0], M.score_counts(full, new)[0], M.score_counts(full, optical)[0]]
    per = M.score_counts(full)[1][A]
    return {"name": name, "n_pred": len(fc["features"]), "insample_arch": {"f1": ins["f1"], "groups": ins["groups"]},
            "honest10": {"all": s[0], "new": s[1], "optical": s[2]}, "honest_arch": per, "info": info}, full

res, cnt = {}, {}
for name, fc in (("base_2m", st), ("micro_0.25", None), ("micro_0.05", None)):
    if fc is None:
        fc = micro(st, float(name.split("_")[1]))
    res[name], cnt[name] = run(fc, name); print(json.dumps(res[name], ensure_ascii=False), flush=True)
for name in ("micro_0.25", "micro_0.05"):
    res[name]["delta_vs_base"] = {k: res[name]["honest10"][k] - res["base_2m"]["honest10"][k] for k in ("all", "new", "optical")}
    res[name]["bootstrap_vs_base"] = M.paired_bootstrap_counts(cnt[name], cnt["base_2m"], site_family, n=1000)
res["check_base_matches_rerank"] = {"rerank_honest": base["scores"], "ours": res["base_2m"]["honest10"]}
json.dump(res, open("../analysis/arch_micro_honest.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: res[k].get("delta_vs_base") for k in ("micro_0.25", "micro_0.05")}),
      json.dumps({k: res[k].get("bootstrap_vs_base") for k in ("micro_0.25", "micro_0.05")}))
