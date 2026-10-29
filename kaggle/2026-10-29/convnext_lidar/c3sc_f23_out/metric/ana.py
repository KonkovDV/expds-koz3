# f23: 1) сверка базы с bot-10/work/curves; 2) каталоги c_A/c_X/c_Xx (f23) и c4_X/c4_Xx (f0–f3: + curves_X_t/Xx_t Гнёздово/Тимерево из c3sc_f01_out), 100 юнитов, симлинки;
# 3) li-анализ на gt_li (затронутые юниты): in-sample и LOFO по семьям (family_folds k=None); 4) li_cnx один: TP/FP по порогам, лучший F1 по фолду CNN
import sys, os, json, pickle
from pathlib import Path
import numpy as np
B10 = Path("/workspace/koz3_shared/bot-10"); sys.path.insert(0, str(B10 / "repo"))
from expds import metric as M
from expds.naming import site_family
D = Path(__file__).resolve().parent; os.chdir(D); A = B10 / "work/curves"
F01 = Path("/workspace/koz3_shared/bot-9/kaggle/convnext_lidar/c3sc_f01_out/metric"); U01 = ["Гнёздово", "Тимерево"]
FAMS = json.load(open("gt.log"))["fams"]  # затронутые семьи (prep.py gt)
U = sorted(p.stem for p in (D / "curves_base_t").glob("*.pkl") if set(p.stem.split("+")) & set(FAMS))  # family_curves пишет и пустой юнит кластера Иссык+Каджы — исключить
rep = {"units_f23": U, "units_skipped": sorted(p.stem for p in (D / "curves_base_t").glob("*.pkl") if p.stem not in U)}
rep["base_check"] = {u: (A / f"{u}.pkl").read_bytes() == (D / "curves_base_t" / f"{u}.pkl").read_bytes() for u in U}
allu = sorted(p.stem for p in A.glob("*.pkl")); rep["A_units"] = len(allu)
assert not set(U) & set(U01)
for v, src in (("A", None), ("X", "X"), ("Xx", "Xx"), ("4_X", "X"), ("4_Xx", "Xx")):
    d = D / f"c_{v}" if not v.startswith("4") else D / f"c{v}"; d.mkdir(exist_ok=True)
    for p in d.glob("*.pkl"): p.unlink()
    for u in allu:
        if src and u in U: t = D / f"curves_{src}_t" / f"{u}.pkl"
        elif src and v.startswith("4") and u in U01: t = F01 / f"curves_{src}_t" / f"{u}.pkl"
        else: t = A / f"{u}.pkl"
        assert t.exists(), t; os.symlink(t, d / f"{u}.pkl")
def load(v, units):
    out = {}
    for u in units:
        for cls, c in pickle.loads((D / f"curves_{v}" / f"{u}.pkl").read_bytes()).items():
            if cls not in out: out[cls] = c
            else:
                o = out[cls]; conf = np.concatenate([o.conf, c.conf]); hit = np.concatenate([o.hit, c.hit]); idx = np.argsort(-conf, kind="stable")
                out[cls] = M.ClassCurve(conf[idx], hit[idx], list(np.asarray(o.pred_region + c.pred_region, dtype=object)[idx]), o.n_gt + c.n_gt,
                                        o.gt_regions + c.gt_regions, list(np.asarray(o.pred_group + c.pred_group, dtype=object)[idx]))
    return out
KUR = ("kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha", "fortifikatsii")
rf = json.load(open("/workspace/kaggle_out/koz3-cnx-li-c3sc-f23/region_fold.json"))["region_fold"]
li = {}
for v in ("base_li", "cnx_li", "cnxx_li", "X_li", "Xx_li"):
    cv = load(v, U); regions = set().union(*(set(c.pred_region) | set(c.gt_regions) for c in cv.values()))
    row = {}
    for nm, folds in (("insample", None), ("lofo", M.family_folds(regions, site_family, None))):
        if folds is None:
            res = M.sweep_groups(cv); taus = {}
            for cls, x in res.items():
                for g, gv in x["groups"].items(): taus.setdefault(g, {})[cls] = gv["threshold"]
            sc, per = M.score_group_thresholds(cv, taus); row["taus"] = taus
        else:
            c, info = M.honest_counts(cv, folds, unseen="keep"); sc, per = M.score_counts(c)
            for fo in (2, 3):
                rs = {r for r in regions if rf.get(r) == fo or (rf.get(r) is None and any(rf.get(q) == fo and site_family(q) == site_family(r) for q in rf))}
                s2, p2 = M.score_counts(c, rs)
                row[f"lofo_f{fo}"] = {"score": s2, "per": {k: {kk: p2[k][kk] for kk in ("tp", "fp", "f1") if kk in p2[k]} for k in p2 if k in KUR}}
        row[nm] = {"score": sc, "per": {k: {kk: per[k][kk] for kk in ("tp", "fp", "f1") if kk in per[k]} for k in per if k in KUR}}
    li[v] = row
rep["li"] = li
tab = {}
for v in ("cnx_li", "cnxx_li"):
    cv = load(v, U)
    for fo in (2, 3):
        for cls in KUR:
            if cls not in cv: continue
            c = cv[cls]; regs = {r for r in rf if rf[r] == fo}
            m = np.array([r in regs for r in c.pred_region], bool); conf, hit = c.conf[m], c.hit[m]
            ngt = sum(n for r, n in c.gt_regions.items() if r in regs); ent = {"n_gt": ngt, "n_pred": int(conf.size)}
            for t in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
                k = conf >= t; tp = int(hit[k].sum()); fp = int(k.sum()) - tp
                ent[str(t)] = [tp, fp, round(2 * tp / max(1, 2 * tp + fp + (ngt - tp)), 4)]
            if conf.size:
                tp = np.cumsum(hit); n = np.arange(1, conf.size + 1); f1 = 2 * tp / (n + ngt); i = int(np.argmax(f1))
                ent["best"] = {"thr": float(conf[i]), "tp": int(tp[i]), "fp": int(n[i] - tp[i]), "f1": round(float(f1[i]), 4)}
            tab[f"{v}|f{fo}|{cls}"] = ent
rep["cnx_thr_table"] = tab
json.dump(rep, open("ana.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(rep["base_check"]), all(rep["base_check"].values()), rep["A_units"])
for v, r in li.items(): print(v, "insample %.4f lofo %.4f" % (r["insample"]["score"], r["lofo"]["score"]), {f: round(r[f"lofo_f{f}"]["score"], 4) for f in (2, 3)}, {k: round(x["f1"], 3) for k, x in r["lofo"]["per"].items()})
for k, e in tab.items(): print(k, e)
