# 1) сверка базы с bot-10/work/curves; 2) каталоги c_A/c_X/c_Xx (100 юнитов, симлинки); 3) li-анализ на gt_li (2 семьи)
import sys, os, json, pickle
from pathlib import Path
import numpy as np
B10 = Path("/workspace/koz3_shared/bot-10"); sys.path.insert(0, str(B10 / "repo"))
from expds import metric as M
from expds.naming import site_family
D = Path(__file__).resolve().parent; os.chdir(D); A = B10 / "work/curves"
U = ["Гнёздово", "Тимерево"]; rep = {}
rep["base_check"] = {u: (A / f"{u}.pkl").read_bytes() == (D / "curves_base_t" / f"{u}.pkl").read_bytes() for u in U}
allu = sorted(p.stem for p in A.glob("*.pkl")); rep["A_units"] = len(allu)
for v in ("A", "X", "Xx"):
    d = D / f"c_{v}"; d.mkdir(exist_ok=True)
    for p in d.glob("*.pkl"): p.unlink()
    for u in allu:
        os.symlink(A / f"{u}.pkl" if (v == "A" or u not in U) else D / f"curves_{v}_t" / f"{u}.pkl", d / f"{u}.pkl")
def load(v):
    out = {}
    for u in U:
        for cls, c in pickle.loads((D / f"curves_{v}" / f"{u}.pkl").read_bytes()).items():
            if cls not in out: out[cls] = c
            else:
                o = out[cls]; conf = np.concatenate([o.conf, c.conf]); hit = np.concatenate([o.hit, c.hit]); idx = np.argsort(-conf, kind="stable")
                out[cls] = M.ClassCurve(conf[idx], hit[idx], list(np.asarray(o.pred_region + c.pred_region, dtype=object)[idx]), o.n_gt + c.n_gt,
                                        o.gt_regions + c.gt_regions, list(np.asarray(o.pred_group + c.pred_group, dtype=object)[idx]))
    return out
KUR = ("kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha")
li = {}
for v in ("base_li", "sli_li", "cnx_li", "cnxx_li", "slix_li", "X_li", "Xx_li"):
    cv = load(v); regions = set().union(*(set(c.pred_region) | set(c.gt_regions) for c in cv.values()))
    row = {}
    # in-sample: пороги (группа,класс) по обеим семьям (оптимистично); LOFO: порог семьи из другой семьи (честно, но 1 семья на подбор)
    for nm, folds in (("insample", None), ("lofo", M.family_folds(regions, site_family, None))):
        if folds is None:
            res = M.sweep_groups(cv); taus = {}
            for cls, x in res.items():
                for g, gv in x["groups"].items(): taus.setdefault(g, {})[cls] = gv["threshold"]
            sc, per = M.score_group_thresholds(cv, taus); row["taus"] = taus
        else:
            c, info = M.honest_counts(cv, folds, unseen="keep"); sc, per = M.score_counts(c)
        row[nm] = {"score": sc, "per": {k: {kk: per[k][kk] for kk in ("tp", "fp", "f1") if kk in per[k]} for k in per if k in KUR}}
        # по семьям
        if folds is not None:
            for fam in U:
                rs = {r for r in regions if site_family(r) == fam}; s2, p2 = M.score_counts(c, rs)
                row[nm][fam] = {"score": s2, "per": {k: {kk: p2[k][kk] for kk in ("tp", "fp", "f1") if kk in p2[k]} for k in p2 if k in KUR}}
    li[v] = row
rep["li"] = li
# li_cnx в одиночку: TP/FP по семье и порогу, лучший F1 (in-sample) по классу
tab = {}
for v in ("cnx_li", "cnxx_li"):
    for u in U:
        cv = pickle.loads((D / f"curves_{v}" / f"{u}.pkl").read_bytes())
        for cls in KUR:
            if cls not in cv: continue
            c = cv[cls]; ent = {"n_gt": int(c.n_gt), "n_pred": int(c.conf.size)}
            for t in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
                k = c.conf >= t; tp = int(c.hit[k].sum()); fp = int(k.sum()) - tp
                ent[str(t)] = [tp, fp, round(2 * tp / max(1, 2 * tp + fp + (c.n_gt - tp)), 4)]
            if c.conf.size:
                tp = np.cumsum(c.hit); n = np.arange(1, c.conf.size + 1); f1 = 2 * tp / (n + c.n_gt); i = int(np.argmax(f1))
                ent["best"] = {"thr": float(c.conf[i]), "tp": int(tp[i]), "fp": int(n[i] - tp[i]), "f1": round(float(f1[i]), 4)}
            tab[f"{v}|{u}|{cls}"] = ent
rep["cnx_thr_table"] = tab
json.dump(rep, open("ana.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(rep["base_check"]), rep["A_units"])
for v, r in li.items(): print(v, "insample %.4f lofo %.4f" % (r["insample"]["score"], r["lofo"]["score"]), {f: round(r["lofo"][f]["score"], 4) for f in U}, {k: round(x["f1"], 3) for k, x in r["lofo"]["per"].items()})
for k, e in tab.items(): print(k, e)
