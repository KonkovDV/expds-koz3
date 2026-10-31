"""Анализ OOF верификатора nolidar v1 (Kaggle). Только чтение /workspace/kgl_out/nolidar_v/."""
import csv, json, collections, numpy as np, sys
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from scipy.stats import rankdata, spearmanr
IN = "/workspace/kgl_out/nolidar_v/"
R = list(csv.DictReader(open(IN + "oof_verifier_nolidar.csv", encoding="utf-8")))
out = {"n_rows_csv": len(R)}
out["by_group"] = dict(collections.Counter(r["group"] for r in R))
out["has_crop0"] = sum(r["has_crop"] != "1" for r in R); out["p_missing"] = sum(r["p_verifier"] == "" for r in R)
out["markup"] = dict(collections.Counter(r["markup_type"] for r in R))
def A(y, s): return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None
def AP(y, s): return float(average_precision_score(y, s)) if y.sum() > 0 else None
def logit(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
res = {}
for h in ("kurgan", "gor"):
    S = [r for r in R if r["group"] == h and r["p_verifier"] != ""]
    y = np.array([int(r["label"]) for r in S]); c = np.array([float(r["confidence"]) for r in S])
    v = np.array([float(r["p_verifier"]) for r in S]); f = np.array([int(r["fold"]) for r in S]).tolist(); f = np.array(f, dtype=object)
    fam = np.array([r["family"] for r in S])
    # стеки
    ra = (rankdata(c) + rankdata(v)) / 2                       # глобальный rank-average
    ra_f = np.zeros(len(S))                                    # rank-average внутри фолда (нормированный)
    for k in sorted(set(f)): m = f == k; ra_f[m] = (rankdata(c[m]) / m.sum() + rankdata(v[m]) / m.sum()) / 2
    X = np.c_[c, logit(v)]; lr = np.full(len(S), np.nan); coefs = {}
    for k in sorted(set(f)):                                   # LOFO: обучаем на остальных фолдах
        tr, te = f != k, f == k
        if y[tr].sum() == 0: continue
        m = LogisticRegression(C=1.0, max_iter=1000).fit(X[tr], y[tr]); lr[te] = m.predict_proba(X[te])[:, 1]
        coefs[k] = [round(float(x), 4) for x in (*m.coef_[0], m.intercept_[0])]
    scores = {"verifier": v, "conf": c, "rankavg_global": ra, "rankavg_infold": ra_f, "logreg_LOFO": lr}
    d = {"n": len(S), "pos": int(y.sum()), "spearman_conf_vs_verifier": round(float(spearmanr(c, v)[0]), 4),
         "logreg_coef[conf,logit_v,b]_by_testfold": coefs, "all": {}, "by_fold": {}, "by_family": {}}
    for nm, s in scores.items(): d["all"][nm] = {"auc": A(y, s), "ap": AP(y, s)}
    d["all"]["prevalence"] = float(y.mean())
    for k in sorted(set(f)):
        m = f == k
        d["by_fold"][k] = {"n": int(m.sum()), "pos": int(y[m].sum()), "n_fam": len(set(fam[m])),
                           **{nm: {"auc": A(y[m], s[m]), "ap": AP(y[m], s[m])} for nm, s in scores.items()}}
    # макро по семьям (≥3 позитивов), чтобы отделить «внутрисемейное» ранжирование от межсемейного
    mac = collections.defaultdict(list)
    for F in sorted(set(fam)):
        m = fam == F
        if y[m].sum() >= 1 and y[m].sum() < m.sum():
            row = {"fold": int(f[m][0]), "n": int(m.sum()), "pos": int(y[m].sum()),
                   "prev": round(float(y[m].mean()), 4),
                   "v_mean": round(float(v[m].mean()), 4), "c_mean": round(float(c[m].mean()), 4),
                   **{nm: A(y[m], s[m]) for nm, s in scores.items()}}
            d["by_family"][F] = row
            if y[m].sum() >= 3:
                for nm in scores: mac[nm].append((row[nm], int(f[m][0])))
    d["macro_family_auc_pos>=3"] = {nm: round(float(np.mean([a for a, _ in l])), 4) for nm, l in mac.items()}
    d["macro_family_auc_pos>=3_by_fold"] = {k: {nm: round(float(np.mean([a for a, ff in l if ff == k])), 4)
                                                for nm, l in mac.items() if any(ff == k for _, ff in l)} for k in sorted(set(f))}
    # бутстрап по семьям: ΔAUC (verifier − conf) и (logreg − conf), пары
    rng = np.random.default_rng(0); fams = sorted(set(fam)); idx = {F: np.where(fam == F)[0] for F in fams}; bs = collections.defaultdict(list)
    for _ in range(300):
        ii = np.concatenate([idx[F] for F in rng.choice(fams, len(fams))])
        if 0 < y[ii].sum() < len(ii):
            ac = A(y[ii], c[ii])
            for nm in ("verifier", "rankavg_infold", "logreg_LOFO"): bs[nm].append(A(y[ii], scores[nm][ii]) - ac)
    d["family_bootstrap_dAUC_vs_conf_p2.5_p50_p97.5"] = {nm: [round(float(x), 4) for x in np.percentile(l, [2.5, 50, 97.5])] for nm, l in bs.items()}
    res[h] = d
out["heads"] = res
json.dump(out, open("oof_metrics.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
for h, d in res.items():
    print(h, d["n"], d["pos"], "spearman", d["spearman_conf_vs_verifier"])
    for nm, x in d["all"].items(): print("  ", nm, x)
    for k, x in d["by_fold"].items(): print("  fold", k, x["n"], x["pos"], x["n_fam"], {nm: (None if x[nm]["auc"] is None else round(x[nm]["auc"], 3)) for nm in ("verifier", "conf", "rankavg_infold", "logreg_LOFO")})
    print("  macro", d["macro_family_auc_pos>=3"]); print("  macro_by_fold", d["macro_family_auc_pos>=3_by_fold"])
    print("  boot", d["family_bootstrap_dAUC_vs_conf_p2.5_p50_p97.5"]); print("  coefs", d["logreg_coef[conf,logit_v,b]_by_testfold"])
print({k: v for k, v in out.items() if k != "heads"})
