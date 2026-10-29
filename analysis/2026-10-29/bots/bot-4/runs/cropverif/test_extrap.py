"""Оценка числа кропов на тестовой (публичной) выборке: пул = max(15 %, 1000) на (группа, класс), но не больше всех кандидатов.
Масштаб: кандидаты группы × (участков с этой разметкой в публичной выборке / в обучающей). Публичная выборка: 48 участков, спутник 30, аэрофото 4 (expds docs, notes/public_val_2026-10-22.json)."""
import pickle, json, numpy as np
from collections import Counter
T = pickle.load(open('/workspace/rerank/work/table.pkl', 'rb')); cls = T['cls']; grp = np.asarray(T['grp']); mk = np.asarray(T['mk']); reg = np.asarray(T['reg'])
pool = np.load('/workspace/rerank/work/pool.npy')
G = [('ae_fort', 5), ('sp_arch', 3), ('or', 3), ('sp_focal', 0), ('sp_focal', 1), ('sp_focal2', 0), ('sp_focal2', 1), ('ae', 0), ('ae', 1)]
nreg = {m: len(set(reg[mk == m])) for m in ('sp_or', 'ae', 'or')}
pub = {'sp_or': 30, 'ae': 4, 'or': 1}
out = {'train_regions_by_mk': nreg, 'public_regions_by_mk': pub, 'groups': {}}; tot = 0
for g, c in G:
    m = (grp == g) & (cls == c); allc = int(m.sum()); mm = Counter(mk[m]).most_common(1)[0][0]
    t_all = allc * pub[mm] / nreg[mm]; t_pool = min(t_all, max(0.15 * t_all, 1000))
    out['groups'][f'{g}/{T["CL"][c]}'] = dict(train_all=allc, train_pool=int((m & pool).sum()), mk=mm, test_all_est=round(t_all), test_pool_est=round(t_pool)); tot += t_pool
out['test_pool_total_est'] = round(tot); out['test_all_total_est'] = round(sum(v['test_all_est'] for v in out['groups'].values()))
json.dump(out, open('test_extrap.json', 'w'), indent=1, ensure_ascii=False); print(json.dumps(out, ensure_ascii=False, indent=1))
