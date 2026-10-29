"""Эквивалентность: решения expds.orgate (код пайплайна, объекты в EPSG:3857, LAS из входного каталога)
против офлайн-решений fit.py (offline_decisions.json) и офлайн-признаков (X2.npy + feats_gate.npz) на 4 участках."""
import json, pickle, resource, sys, time
from pathlib import Path
import numpy as np, shapely
from shapely.geometry import shape
sys.path.insert(0, "/workspace/koz3_shared/bot-6/repo")
from expds import orgate, rerank_arch as R
from expds.catalog import scan_root

W = Path("/workspace/koz3_shared/bot-6/work/orgate_final")
off = json.loads((W / "offline_decisions.json").read_text())
las_boxes = set(json.loads(Path("/workspace/expds-koz3/notes/las_boxes_59.json").read_text()))
data = pickle.load(open(W / "equiv/regions4.pkl", "rb"))
t0 = time.time()
regs = scan_root("/workspace/data/train-expds-3")
cat_all = orgate.catalog_info(regs)
las_cat = {r for r, v in cat_all.items() if v["las"]}
print("scan_root: участков", len(cat_all), "с LAS", len(las_cat), "== las_boxes_59:", las_cat == las_boxes,
      f"{time.time() - t0:.1f} s")
X2 = np.load("/workspace/rerank/work/X2.npy", mmap_mode="r")
gz = np.load("/workspace/koz3_shared/bot-6/work/orgate/feats_gate.npz")
gpos = {int(s): k for k, s in enumerate(gz["stack_idx"])}
pool_tr = np.load("/workspace/rerank/work/pool.npy", mmap_mode="r")
opts = {"model": str(W / "orgate_g.txt"), "meta": str(W / "orgate_g.json")}
model = orgate.load(opts)
res = {}
for r, rows in sorted(data.items()):
    ids = np.array([i for i, _ in rows])
    feats = [dict(f["properties"], geom=shape(f["geometry"])) for _, f in rows]
    t = time.time()
    rest, kept, info = orgate.decide(feats, opts, {r: cat_all[r]}, model=model)
    dt = time.time() - t
    # признаки, которые увидела модель (повтор внутри decide — для сравнения с офлайн-матрицей)
    T = R.table(feats); arch = T["cls"] == R.METRIC_CLASSES.index(R.CLASS)
    T["grp"] = np.array([{"sp_or": "sp_arch"}.get(g, g) if a else g for g, a in zip(T["grp"], arch)])
    tgt = arch & np.isin(T["grp"], ["or", "sp_arch"]); T["has_las"] = np.full(len(feats), cat_all[r]["las"])
    Xp = np.c_[R.features(T, tgt, model[1]["grp_vocab"], model[1]["mk_vocab"]),
               orgate.gate_features(T, tgt, {r: cat_all[r]["las"]}, {r: cat_all[r]["utm_json"]})]
    sid = ids[np.flatnonzero(tgt)]
    Xo = np.c_[np.asarray(X2[sid]), gz["X"][[gpos[int(s)] for s in sid]]]
    d = np.abs(Xp - Xo); d[np.isnan(Xp) & np.isnan(Xo)] = 0
    worst = sorted(((float(np.nanmax(d[:, j])), n) for j, n in enumerate(model[1]["names"])), reverse=True)[:3]
    sp = model[0].predict(Xp); so = model[0].predict(Xo)
    removed_pipe = sorted(int(ids[i]) for i in info["removed_idx"])
    kept_pipe = sorted(int(ids[i]) for i in info["kept_idx"])
    o = off[r]
    inpool_tr = pool_tr[sid].astype(bool)
    # офлайн: вне пула трейна → −1 (удалён). Пул в пайплайне считается по всему входу (как у R) —
    # на одном участке пул шире, поэтому решение сравниваем на строках пула трейна + проверяем, что вне пула всё удалено офлайн
    kp_on_trpool = sorted(int(s) for s in kept_pipe if pool_tr[s])
    res[r] = dict(n_rows=len(feats), n_target=int(tgt.sum()), las=cat_all[r]["las"], las_source=info["las_source"],
                  removed_pipe=len(removed_pipe), removed_off=len(o["removed"]),
                  removed_equal=removed_pipe == sorted(o["removed"]), kept_equal=kept_pipe == sorted(o["kept"]),
                  kept_equal_on_train_pool=kp_on_trpool == sorted(o["kept"]),
                  n_target_outside_train_pool=int((~inpool_tr).sum()),
                  feat_maxdiff_top3=worst, score_maxdiff=float(np.max(np.abs(sp - so))), t_decide_s=round(dt, 3),
                  removed_ids_pipe=removed_pipe, kept_ids_pipe=kept_pipe)
    print(r, {k: v for k, v in res[r].items() if not k.endswith("_ids_pipe")}, flush=True)
res["_maxrss_kb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
res["_total_s"] = round(time.time() - t0, 1)
print("maxrss KB", res["_maxrss_kb"], "total s", res["_total_s"])
(W / "equiv/equiv.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
