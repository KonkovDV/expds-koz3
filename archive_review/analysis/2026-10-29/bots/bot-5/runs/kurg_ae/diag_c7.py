"""bot-5. Диагностика на кандидатах bot-7 (src2): одномерные AUC признаков (pooled и средняя по участкам),
AUC OOF-p (c7g) по участкам и на подмножестве с виньетками (для сравнения с DINO). Только чтение готовых файлов."""
import numpy as np, glob, json, collections
from scipy.stats import rankdata
OD = '/workspace/koz3_shared/bot-5/analysis/kurg_ae'
def auc(s, t):
    t = np.asarray(t, bool); s = np.asarray(s, float)
    if t.all() or (~t).all(): return float('nan')
    r = rankdata(s); return float((r[t].sum() - t.sum() * (t.sum() + 1) / 2) / (t.sum() * (~t).sum()))
D = dict(np.load(f'{OD}/diag_c7g.npz')); P = np.load(f'{OD}/p_oof_fam10_c7g.npy')
g = (D['SRC'] == 2) & ~D['IGN']; y = D['LAB'] > 0
pos = {int(c): k for k, c in enumerate(D['SID']) if D['SRC'][k] == 2}
fn = json.load(open('/workspace/koz3_shared/bot-5/tmp/kurg_ae/feat7/feat_names.json'))
X = np.full((len(y), len(fn)), np.nan, np.float32); W = np.zeros(len(y), bool)
for f in sorted(glob.glob('/workspace/koz3_shared/bot-5/tmp/kurg_ae/feat7/*.npz')):
    z = dict(np.load(f)); rec = z['rec']; FF = z['F']
    for k_, (r, w) in enumerate(zip(rec, z['widx'])):
        k = pos[int(r[3])]; X[k] = FF[k_]; W[k] = w
C7 = dict(np.load('/workspace/koz3_shared/bot-5/tmp/kurg_ae/cand7.npz')); p7 = dict(zip(C7['cand_id'].tolist(), C7['score'].tolist()))
b7 = np.array([p7.get(int(s), np.nan) if D['SRC'][k] == 2 else np.nan for k, s in enumerate(D['SID'])])
regs = sorted(set(D['CREG'][g & y]))
out = {'n_gen': int(g.sum()), 'n_pos': int((g & y).sum()), 'pos_by_reg': dict(collections.Counter(D['CREG'][g & y].tolist())),
       'neg_by_reg': {r: int((g & ~y & (D['CREG'] == r)).sum()) for r in regs}}
def both(s):
    pr = [auc(s[g & (D['CREG'] == r)], y[g & (D['CREG'] == r)]) for r in regs]
    return [auc(s[g], y[g]), float(np.nanmean(pr))]
out['p_oof_c7g'] = both(P[:, 1] + P[:, 2]); out['b7_score'] = both(b7); out['p_oof_c7g_povr_tsel_sep'] = [both(P[:, 1]), both(P[:, 2])]
out['p_oof_by_reg'] = {r: auc((P[:, 1] + P[:, 2])[g & (D['CREG'] == r)], y[g & (D['CREG'] == r)]) for r in regs}
out['b7_score_by_reg'] = {r: auc(b7[g & (D['CREG'] == r)], y[g & (D['CREG'] == r)]) for r in regs}
fa = {n: both(X[:, i]) for i, n in enumerate(fn)}
out['feat_auc_top10_by_within_reg'] = sorted(fa.items(), key=lambda kv: -abs(kv[1][1] - 0.5))[:10]
gw = g & W; out['window_subset'] = dict(n=int(gw.sum()), n_pos=int((gw & y).sum()), auc_hand_p_oof=auc((P[gw, 1] + P[gw, 2]), y[gw]))
json.dump(out, open(f'{OD}/diag_c7.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False)[:4000])
