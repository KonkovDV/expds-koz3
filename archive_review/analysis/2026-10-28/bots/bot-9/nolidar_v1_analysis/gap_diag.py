"""Диагностика разрыва val/test: история эпох, val-семьи, семейный состав тестов, калибровка по фолдам, пространственные соседи."""
import csv, json, collections, numpy as np
from sklearn.metrics import roc_auc_score
from scipy.spatial import cKDTree
IN = "/workspace/kgl_out/nolidar_v/"
S = json.load(open(IN + "summary_verifier.json", encoding="utf-8"))
R = list(csv.DictReader(open(IN + "oof_verifier_nolidar.csv", encoding="utf-8")))
def A(y, s): y = np.asarray(y); return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None
out = {"jobs": [], "val_families": {}, "fold_calib": {}, "test_family_composition": {}, "spatial": {}}
for j in S["jobs"]:
    h = [x["val_auc"] for x in j.get("hist", []) if x["val_auc"] is not None]
    e = {k: j.get(k) for k in ("head", "fold", "val_family", "n_train", "pos_train", "n_val", "n_test", "pos_test", "best_val_auc", "test_auc", "test_auc_stack_conf")}
    if h: e.update(n_ep=len(h), argmax_ep=int(np.argmax(h)), val_last=round(h[-1], 4), val_mean=round(float(np.mean(h)), 4), val_min=round(min(h), 4), val_ep0=round(h[0], 4))
    out["jobs"].append(e)
for h in ("kurgan", "gor"):
    T = [r for r in R if r["group"] == h]
    fam = np.array([r["family"] for r in T]); y = np.array([int(r["label"]) for r in T]); c = np.array([float(r["confidence"]) for r in T])
    v = np.array([float(r["p_verifier"]) for r in T]); f = np.array([int(r["fold"]) for r in T])
    for j in S["jobs"]:
        if j["head"] != h or not j.get("val_family"): continue
        F = j["val_family"]; m = fam == F
        out["val_families"][f"{h}/fold{j['fold']}/{F}"] = {"family_test_fold": int(f[m][0]) if m.any() else None, "n": int(m.sum()), "pos": int(y[m].sum()),
            "prev": round(float(y[m].mean()), 4) if m.any() else None, "conf_auc_on_val_family": A(y[m], c[m]),
            "oof_verifier_auc_on_val_family(own test fold)": A(y[m], v[m]), "best_val_auc_in_fold": j["best_val_auc"]}
    for k in sorted(set(f)):
        m = f == k
        out["fold_calib"][f"{h}/fold{k}"] = {"v_mean_neg": round(float(v[m & (y == 0)].mean()), 4), "v_mean_pos": round(float(v[m & (y == 1)].mean()), 4) if (m & (y == 1)).any() else None,
            "v_median": round(float(np.median(v[m])), 4), "share_v>0.5": round(float((v[m] > .5).mean()), 4), "prev": round(float(y[m].mean()), 4)}
        rows = []
        for F in sorted(set(fam[m])):
            mm = m & (fam == F)
            rows.append({"family": F, "n": int(mm.sum()), "pos": int(y[mm].sum()), "prev": round(float(y[mm].mean()), 4), "v_mean": round(float(v[mm].mean()), 4),
                         "c_mean": round(float(c[mm].mean()), 4), "auc_v": A(y[mm], v[mm]), "auc_c": A(y[mm], c[mm])})
        out["test_family_composition"][f"{h}/fold{k}"] = sorted(rows, key=lambda r: -r["n"])
    # пространственные соседи: ближайший кандидат ДРУГОЙ семьи и др. фолда (м, EPSG:3857)
    xy = np.array([[float(r["cx"]), float(r["cy"])] for r in T]); tree = cKDTree(xy)
    d, i = tree.query(xy, k=min(30, len(xy)))
    cross = np.full(len(T), np.inf)
    for a in range(len(T)):
        for dd, ii in zip(d[a][1:], i[a][1:]):
            if f[ii] != f[a]: cross[a] = dd; break
    dup_same = int((d[:, 1] < 1.0).sum())
    out["spatial"][h] = {"cand_with_otherfold_neighbor_<64m(3857 units, among 30 NN)": int((cross < 64).sum()),
                         "<400m": int((cross < 400).sum()), "<2000m": int((cross < 2000).sum()),
                         "cand_with_any_neighbor_<1m(dup)": dup_same,
                         "median_NN_dist_any": round(float(np.median(d[:, 1])), 2)}
    # регион→семья: одна ли семья на регион; пересечение bbox регионов разных фолдов
    bb = {}
    for r in T:
        x, yy = float(r["cx"]), float(r["cy"]); g = r["region"]
        b = bb.get(g, [x, yy, x, yy, r["fold"], r["family"]]); bb[g] = [min(b[0], x), min(b[1], yy), max(b[2], x), max(b[3], yy), b[4], b[5]]
    ov = []
    gs = sorted(bb)
    for a in range(len(gs)):
        for b in range(a + 1, len(gs)):
            A1, B1 = bb[gs[a]], bb[gs[b]]
            if A1[4] != B1[4] and A1[0] <= B1[2] and B1[0] <= A1[2] and A1[1] <= B1[3] and B1[1] <= A1[3]:
                ov.append([gs[a], gs[b], A1[5], B1[5], A1[4], B1[4]])
    out["spatial"][h]["region_bbox_overlaps_across_folds"] = ov
json.dump(out, open("gap_diag.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for e in out["jobs"]: print(e)
for k, v in out["val_families"].items(): print(k, v)
for k, v in out["fold_calib"].items(): print(k, v)
for k, v in out["test_family_composition"].items():
    print(k); [print("   ", r) for r in v[:8]]
print(json.dumps(out["spatial"], ensure_ascii=False))
