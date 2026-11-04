#!/usr/bin/env python3
"""W1-V3PROD процесс A: стек OOF (table.pkl + geom_wkb.pkl) + c19 (+dfx) → путь архива (копия x_v3on) по фолдам.
argv: arm(v2c|v3) mode(S0|PH|PC|PR) out.geojson. Код архива не меняется; только копия кода и конфиг архива."""
import copy, json, pickle, sys, time, tomllib, logging
from pathlib import Path
import numpy as np
X = Path("/workspace/koz3/w1/v3audit/x_v3on"); sys.path.insert(0, str(X))
from shapely import wkb as WKB
from shapely.geometry import shape
from expds import pipeline as PL, post, watchdog, writer, rerank_arch
logging.basicConfig(level=logging.WARNING)
arm, mode, OUT = sys.argv[1], sys.argv[2], Path(sys.argv[3])
CFGP = {"v3": X / "configs/default.toml", "v2c": Path("/workspace/koz3/w1/v3audit/x_v2c/configs/default.toml")}[arm]
cfg0 = tomllib.load(open(CFGP, "rb"))
# пути моделей реранка — от корня копии архива
cfg0["rerank_arch"] = dict(cfg0["rerank_arch"], model=str(X / cfg0["rerank_arch"]["model"]), meta=str(X / cfg0["rerank_arch"]["meta"]))
if mode == "PR":
    cfg0["rerank_arch"]["enabled"] = False
T = json.load(open("t_taus.json"))
folds = [set(f) for f in T["folds"]]; dreg = set(T["dreg"])
taus = T["taus_V3P" if arm == "v3" else "taus_B0"]
SP = {"sp_or", "sp_focal", "sp_arch", "sp_focal2"}
KEY = {"ae_fort": "ae", "sp_focal": "sp_or", "sp_arch": "sp_or"}          # гейт-группа → ключ порога архива
NOTG = {"li", "ae", "ae_fort", "sp_or", "sp_focal", "sp_arch", "or", "maggr", "terfl"}   # в архиве threshold_group=None
FLOORKEY = {"li": "li", "li_mosaic": "li", "li_coarse": "li_coarse", "li_mask": "li_mask", "li_fort": "li_fort", "li_dev": "li_dev",
            "sp_focal2": "sp_focal2", "ae": "ae", "ae_fort": "ae", "sp_or": "sp_or", "sp_focal": "sp_or", "sp_arch": "sp_or", "or": "or"}
t0 = time.time()
tab = pickle.load(open("/workspace/koz3/w5data/work/table.pkl", "rb"))
CL = list(tab["CL"]); cls = np.asarray(tab["cls"]); conf = np.asarray(tab["conf"], float); reg = np.asarray(tab["reg"]).astype(str)
grp = np.asarray(tab["grp"]).astype(str); mk = np.asarray(tab["mk"]).astype(str); del tab
G = pickle.load(open("/workspace/koz3/w5data/work/geom_wkb.pkl", "rb"))
assert len(G) == len(conf)
keep = grp != "etgr"
if arm == "v3":
    keep &= ~(np.isin(grp, list(SP)) & np.isin(reg, list(dreg)))
ex = pickle.load(open("/workspace/koz3/orch/bot-W1/work/j2nc/extras.pkl", "rb"))["ex"]
ex = [e for e in ex if arm == "v3" or e["threshold_group"] != "sp_dfine"]


def fold_cfg(fi):
    c = copy.deepcopy(cfg0)
    if mode in ("PH", "PR", "S0"):
        bm = c.setdefault("thresholds_by_markup", {})
        for g, d in taus[fi].items():
            k = KEY.get(g, g)
            for cl_, v in d.items():
                bm.setdefault(k, {})[cl_] = 1.01 if v is None else float(v)
        # (ключ, класс) с объектами, но без порога гейта в обучающих фолдах → как unseen=drop
        tset = {(KEY.get(g, g), cl_) for g, d in taus[fi].items() for cl_ in d}
        for g in set(grp[keep].tolist()) | {e["threshold_group"] for e in ex}:
            k = KEY.get(g, g)
            for cl_ in CL:
                if (k, cl_) not in tset:
                    bm.setdefault(k, {})[cl_] = 1.01
    return c


def feats_of(fi, c):
    held = folds[fi]
    m = keep & np.isin(reg, list(held))
    idx = np.flatnonzero(m)
    if mode != "S0":   # ранние полы прохода (output_floors) по ключам генерации
        fl = {k: PL.output_floors(c, k) for k in set(FLOORKEY.values())}
        ok = np.array([conf[i] >= fl[FLOORKEY[grp[i]]][CL[cls[i]]] if grp[i] in FLOORKEY else True for i in idx], bool)
        idx = idx[ok]
    out = []
    for i in idx:
        g = grp[i]
        out.append(dict(geom=WKB.loads(G[i]), class_name=CL[cls[i]], confidence=float(conf[i]), region_name=reg[i],
                        markup_type=("ae" if g == "ae_fort" else ("sp_or" if g in SP else mk[i])),
                        threshold_group=None if g in NOTG else g, source=None))
    for e in ex:
        if e["region_name"] in held:
            out.append(dict(geom=shape(e["geometry"]), class_name=e["class_name"], confidence=e["confidence"], region_name=e["region_name"],
                            markup_type=e["markup_type"], threshold_group=e["threshold_group"], source=None))
    return out


def sigma_rules(feats):
    """Порядок архива: sp_focal2 (без LAS) против якорей оптики; затем канал σ li_dev против якорей не-li_dev участка."""
    f2 = [f for f in feats if f["threshold_group"] == "sp_focal2"]
    if f2:
        base = [f for f in feats if f["threshold_group"] not in ("sp_focal2", "li_dev")]
        ids = {id(f) for f in PL.drop_near_kurgans(f2, PL.kept_kurgan_anchors(base), 12.0)}
        feats = [f for f in feats if f["threshold_group"] != "sp_focal2" or id(f) in ids]
    dev = [f for f in feats if f["threshold_group"] == "li_dev"]
    if dev:
        pool = [f for f in feats if f["threshold_group"] != "li_dev"]
        ids = {id(f) for f in PL.drop_near_kurgans(dev, PL.kept_kurgan_anchors(pool), 12.0)}
        feats = [f for f in feats if f["threshold_group"] != "li_dev" or id(f) in ids]
    return feats


allf, stat = [], []
for fi in range(len(folds)):
    c = fold_cfg(fi)
    fs = feats_of(fi, c); n0 = len(fs)
    if mode == "S0":
        res = post.apply_thresholds(fs, c.get("thresholds", {}), c.get("default_threshold", 0.15), c.get("thresholds_by_markup", {}))
        res = post.finalize(res); ncap = len(res)
    else:
        fs = sigma_rules(fs); n1 = len(fs)
        pre = post.apply_thresholds(post.only_known_classes(fs), c.get("thresholds", {}), 0.15, c.get("thresholds_by_markup", {}))
        ncap = len(pre)
        res = watchdog.finish(fs, c)
    stat.append(dict(fold=fi, regions=len(folds[fi]), n_in=n0, n_out=len(res), n_thr_nopost=ncap))
    print(f"[{arm} {mode}] fold {fi}: in {n0} → out {len(res)} (thr-only {ncap}) {time.time()-t0:.0f}s", flush=True)
    allf += res
fc = writer.to_feature_collection(allf)
OUT.write_text(json.dumps(fc, ensure_ascii=False))
Path(str(OUT) + ".stat.json").write_text(json.dumps(stat))
print("done", len(allf), f"{time.time()-t0:.0f}s")
