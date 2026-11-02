#!/usr/bin/env python3
"""W4 Q6 replay (copied from W1 s2_replay.py; prep = W4 q6 prep.pkl). Original doc: W1-SATREC step 2: v3_on production replay PH (as tools/v3prod a_replay.py, 9ea621d) for base / R1 / R2 / R3 / C1 / C2.
argv: variant out.geojson. Archive code = git archive b447649 (read-only copy), configs/v3_on.toml. Adds scene_drel
annotation (production pipeline._with_scene_drel; boxes = bot-10 frame_boxes.json) and stage snapshots of sat outputs."""
import copy, json, pickle, sys, time, tomllib, logging
from pathlib import Path
import numpy as np
X = Path("/workspace/koz3/w1/satrec/x_v3on"); sys.path.insert(0, str(X))
from shapely import wkb as WKB
from shapely.geometry import shape, mapping
from expds import pipeline as PL, post, watchdog, writer, rerank_arch
logging.basicConfig(level=logging.WARNING)
var, OUT = sys.argv[1], Path(sys.argv[2])
cfg0 = tomllib.load(open(X / "configs/v3_on.toml", "rb"))
cfg0["rerank_arch"] = dict(cfg0["rerank_arch"], model=str(X / cfg0["rerank_arch"]["model"]), meta=str(X / cfg0["rerank_arch"]["meta"]))
T = json.load(open("/workspace/koz3/orch/bot-W1/work/v3prod/t_taus.json"))
folds = [set(f) for f in T["folds"]]; dreg = set(T["dreg"])
P = pickle.load(open("/workspace/koz3/w4/q6/out/prep.pkl", "rb"))
R = P["res"]
if var == "base":
    taus = T["taus_V3P"]; assert taus == R["taus_base"], "base taus differ from t_taus.json"
    ex = P["ex_base"]; V = {"chosen": [None] * 10}
else:
    V = R["variants"][var]; taus = V["taus"]; ex = P["ex_" + V["extras"]]
del P
FB = json.load(open("/workspace/koz3/repo/docs/bots/2026-10-29/bot-10/work11/frame_boxes.json"))["boxes"]
SP = {"sp_or", "sp_focal", "sp_arch", "sp_focal2"}
KEY = {"ae_fort": "ae", "sp_focal": "sp_or", "sp_arch": "sp_or"}
NOTG = {"li", "ae", "ae_fort", "sp_or", "sp_focal", "sp_arch", "or", "maggr", "terfl"}
FLOORKEY = {"li": "li", "li_mosaic": "li", "li_coarse": "li_coarse", "li_mask": "li_mask", "li_fort": "li_fort", "li_dev": "li_dev",
            "sp_focal2": "sp_focal2", "ae": "ae", "ae_fort": "ae", "sp_or": "sp_or", "sp_focal": "sp_or", "sp_arch": "sp_or", "or": "or"}
t0 = time.time()
tab = pickle.load(open("/workspace/koz3/w5data/work/table.pkl", "rb"))
CL = list(tab["CL"]); cls = np.asarray(tab["cls"]); conf = np.asarray(tab["conf"], float); reg = np.asarray(tab["reg"]).astype(str)
grp = np.asarray(tab["grp"]).astype(str); mk = np.asarray(tab["mk"]).astype(str); del tab
G = pickle.load(open("/workspace/koz3/w5data/work/geom_wkb.pkl", "rb"))
keep = (grp != "etgr") & ~(np.isin(grp, list(SP)) & np.isin(reg, list(dreg)))


def fold_cfg(fi):
    c = copy.deepcopy(cfg0)
    bm = c.setdefault("thresholds_by_markup", {})
    for g, d in taus[fi].items():
        k = KEY.get(g, g)
        for cl_, v in d.items():
            bm.setdefault(k, {})[cl_] = 1.01 if v is None else float(v)
    tset = {(KEY.get(g, g), cl_) for g, d in taus[fi].items() for cl_ in d}
    for g in set(grp[keep].tolist()) | {e["threshold_group"] for e in ex}:
        k = KEY.get(g, g)
        for cl_ in CL:
            if (k, cl_) not in tset:
                bm.setdefault(k, {})[cl_] = 1.01
    return c


def feats_of(fi, c):
    held = folds[fi]
    idx = np.flatnonzero(keep & np.isin(reg, list(held)))
    fl = {k: PL.output_floors(c, k) for k in set(FLOORKEY.values())}
    ok = np.array([conf[i] >= fl[FLOORKEY[grp[i]]][CL[cls[i]]] if grp[i] in FLOORKEY else True for i in idx], bool)
    out = []
    for i in idx[ok]:
        g = grp[i]
        out.append(dict(geom=WKB.loads(G[i]), class_name=CL[cls[i]], confidence=float(conf[i]), region_name=reg[i],
                        markup_type=("ae" if g == "ae_fort" else ("sp_or" if g in SP else mk[i])),
                        threshold_group=None if g in NOTG else g, source=None))
    ext = []
    for e in ex:
        if e["region_name"] in held:
            ext.append(dict(geom=shape(e["geometry"]), class_name=e["class_name"], confidence=e["confidence"], region_name=e["region_name"],
                            markup_type=e["markup_type"], threshold_group=e["threshold_group"], source=None))
    if V.get("prior_group"):   # C1/C2: scene-centre prior, fold's nested (d0, s), = post.frame_prior form, before thresholds
        d0, s = V["chosen"][fi]
        if s > 0:
            byreg = {}
            for f in ext:
                byreg.setdefault(f["region_name"], []).append(f)
            ext = []
            for r, fs in byreg.items():
                post.annotate_scene_drel(fs, {mt: FB.get(r, {}).get(mt, []) for mt in ("ae", "sp_or", "or")})
                ext += post.frame_prior(fs, {"groups": [V["prior_group"]], "d0": d0, "edge": 1.0 - s})
            for f in ext:
                f.pop("scene_drel", None)
    out += ext
    # production: scene_drel for every feature of an imaged markup (pipeline._with_scene_drel)
    byreg = {}
    for f in out:
        byreg.setdefault(f["region_name"], []).append(f)
    for r, fs in byreg.items():
        post.annotate_scene_drel(fs, {mt: [tuple(b) for b in FB.get(r, {}).get(mt, [])] for mt in ("ae", "sp_or", "or")})
    return out


def sigma_rules(feats):
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


# stage snapshots of sat-origin outputs (markup sp_or), top-level calls inside finish only
SNAP, CUR, DEPTH = {}, [None], [0]
def is_sat(f):
    return (f.get("markup_type") == "sp_or")
def snap(name, feats):
    SNAP.setdefault(name, []).extend(dict(geometry=mapping(f["geom"]), class_name=f["class_name"], confidence=float(f.get("confidence", 0)),
                                          region_name=f["region_name"], fold=CUR[0], threshold_group=f.get("threshold_group"))
                                     for f in feats if is_sat(f))
def wrap(mod, name, tag, arg_snap=None):
    fn = getattr(mod, name)
    def w(*a, **k):
        DEPTH[0] += 1
        try:
            r = fn(*a, **k)
        finally:
            DEPTH[0] -= 1
        if DEPTH[0] == 0 and CUR[0] is not None:
            if arg_snap:
                snap(arg_snap, a[0])
            snap(tag, r[0] + r[1] if isinstance(r, tuple) else r)
        return r
    setattr(mod, name, w)
wrap(post, "apply_thresholds", "L2_thr")
wrap(post, "merge_within_markup", "L3_merge", arg_snap="L3_thr+rerank")
wrap(post, "multipole_object_poly", "L3b_multipole")
wrap(post, "filter_finds_points_regimes", "L3c_finds")
wrap(post, "kurgan_cross_modal_twins", "L3d_twins")
wrap(post, "micro_circles", "L3e_micro")
wrap(post, "finalize", "L3f_finalize")

allf, stat = [], []
for fi in range(len(folds)):
    c = fold_cfg(fi)
    fs = feats_of(fi, c); n0 = len(fs)
    fs = sigma_rules(fs)
    CUR[0] = fi
    snap("L1_in", fs)
    res = watchdog.finish(fs, c)
    snap("L4_final", res)
    CUR[0] = None
    stat.append(dict(fold=fi, regions=len(folds[fi]), n_in=n0, n_out=len(res), chosen=V["chosen"][fi]))
    print(f"[{var}] fold {fi}: in {n0} → out {len(res)} chosen {V['chosen'][fi]} {time.time()-t0:.0f}s", flush=True)
    allf += res
fc = writer.to_feature_collection(allf)
OUT.write_text(json.dumps(fc, ensure_ascii=False))
Path(str(OUT) + ".stat.json").write_text(json.dumps(stat))
pickle.dump(SNAP, open(str(OUT) + ".snap.pkl", "wb"))
print("done", len(allf), {k: len(v) for k, v in SNAP.items()}, f"{time.time()-t0:.0f}s")
