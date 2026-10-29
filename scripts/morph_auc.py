"""Задача 10, диагностика: одномерный AUC (по рангам) каждого признака.
(а) tselye против прочих лидарных кандидатов; (б) курган (TP по оф. парам) против ложного."""
import sys, json, numpy as np, pyarrow.parquet as pq
from scipy.stats import rankdata
C = np.load(sys.argv[1], allow_pickle=False); L = np.load(sys.argv[2]); t = pq.read_table(sys.argv[3])
sidx = t["stack_idx"].to_numpy(); assert (sidx == C["stack_idx"]).all()
ya = np.isin(sidx, L["tse_pos"]); yb = np.isin(sidx, L["kur_true"])
def auc(x, y):
    m = np.isfinite(x); x, y = x[m], y[m]
    n1 = y.sum(); n0 = (~y).sum()
    if n1 == 0 or n0 == 0: return float("nan"), int(m.sum())
    r = rankdata(x); return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)), int(m.sum())
res = {"n": int(sidx.size), "pos_a_tselye": int(ya.sum()), "pos_b_kurgan_true": int(yb.sum()), "feat": {}}
for c in t.column_names[1:]:
    x = t[c].to_numpy(zero_copy_only=False).astype(np.float64)
    a, na = auc(x, ya); b, nb = auc(x, yb)
    res["feat"][c] = {"auc_tselye": round(a, 4), "auc_kurgan_vs_false": round(b, 4), "nan_share": round(float(np.isnan(x).mean()), 4)}
json.dump(res, open(sys.argv[4], "w"), ensure_ascii=False, indent=1)
for c, v in sorted(res["feat"].items(), key=lambda kv: -abs(kv[1]["auc_kurgan_vs_false"] - .5)):
    print(f"{c:24s} a={v['auc_tselye']:.3f} b={v['auc_kurgan_vs_false']:.3f} nan={v['nan_share']:.3f}")
print(res["n"], res["pos_a_tselye"], res["pos_b_kurgan_true"])
