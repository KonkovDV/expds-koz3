"""bot-6 ORSP v3: почему v2 = v1 на стеке. Счёт sp_focal/sp_focal2 курганов в table.pkl, их уверенности и близость 12 м к сохранённым якорям."""
import pickle, json, numpy as np
from scipy.spatial import cKDTree
T = pickle.load(open('/workspace/rerank/work/table.pkl', 'rb')); CL = T['CL']
cls, conf, grp, reg, mk = T['cls'], T['conf'], T['grp'], T['reg'], T['mk']; XY = np.c_[T['cx'], T['cy']]
K = {'kurgany_povrezhdennye': CL.index('kurgany_povrezhdennye'), 'kurgany_tselye': CL.index('kurgany_tselye')}
KEPT = {"ae": (0.99053, 0.802255), "li": (0.06415, 0.19422), "li_ae": (0.993061, 1.000001), "li_mosaic": (0.959005, 0.17042), "sp_focal": (0.364475, 0.372525), "sp_or": (0.364475, 0.372525)}
out = {}
for g in ('sp_focal', 'sp_focal2', 'sp_or'):
    for cn, c in K.items():
        m = (grp == g) & (cls == c)
        if m.any(): out[f'{g}/{cn}'] = dict(n=int(m.sum()), mk=sorted(set(mk[m].tolist())), q=[float(x) for x in np.quantile(conf[m], [0, .1, .5, .9, 1])], n_ge_out=int((conf[m] >= (0.501645 if cn.endswith('povrezhdennye') else 0.645505)).sum()) if g == 'sp_focal2' else None)
# 12 м: sp_focal2 кандидаты рядом с якорями (тот же участок и класс)
near = {}
for cn, c in K.items():
    ci = 0 if cn.endswith('povrezhdennye') else 1
    anc = np.zeros(len(cls), bool)
    for g, t in KEPT.items(): anc |= (grp == g) & (cls == c) & (conf >= t[ci])
    anc |= (mk == 'li') & (grp == 'li') & (cls == c) & (conf >= KEPT['li'][ci])
    cand = np.flatnonzero((grp == 'sp_focal2') & (cls == c)); n = 0; nhi = 0
    for r in np.unique(reg[cand]):
        a = np.flatnonzero(anc & (reg == r)); cc = cand[reg[cand] == r]
        if not a.size: continue
        d, _ = cKDTree(XY[a]).query(XY[cc]); w = d < 12; n += int(w.sum()); nhi += int((w & (conf[cc] >= (0.501645 if ci == 0 else 0.645505))).sum())
    near[cn] = dict(sp_focal2=int(cand.size), within12_of_anchor=n, within12_and_above_out_floor=nhi)
out['near12'] = near
print(json.dumps(out, ensure_ascii=False, indent=1)); json.dump(out, open('stackcheck.json', 'w'), ensure_ascii=False, indent=1)
