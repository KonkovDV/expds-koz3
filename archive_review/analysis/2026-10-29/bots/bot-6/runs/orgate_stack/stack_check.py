"""bot-6: на каком стеке обучен orgate_g — состав кандидатов arch в /workspace/rerank/work/table.pkl
и сколько из них выбросил бы D-FINE drop_groups (R) на покрытых регионах (COV из bot-10 work14/dfine_gor.json)."""
import json, pickle, collections
import numpy as np
t = pickle.load(open('/workspace/rerank/work/table.pkl', 'rb'))
COV = set(json.load(open('/workspace/koz3_shared/bot-10/work14/dfine_gor.json', encoding='utf-8'))['report']['regions'])
DROP = {'sp_or', 'sp_focal', 'sp_arch', 'sp_focal2'}
cid = t['CL'].index('arkhitektury')
a = np.flatnonzero(t['cls'] == cid)
grp, mk, reg = t['grp'][a], t['mk'][a], t['reg'][a]
P = t['pairs'][('arkhitektury', 'arkhitektury')][0]
hit = np.isin(a, P)
out = {'rows_total': int(len(t['cls'])), 'arch_rows': int(len(a)), 'cov_regions': len(COV),
       'all_groups': sorted(set(map(str, t['grp']))),
       'arch_by_grp': {}, 'arch_dropped_by_R': 0, 'arch_kept_after_R': 0,
       'arch_pairs_ok_total': int(hit.sum())}
for g in sorted(set(grp)):
    m = grp == g
    d = m & np.isin(reg, list(COV)) & (g in DROP)
    out['arch_by_grp'][str(g)] = {'rows': int(m.sum()), 'mk': sorted(set(map(str, mk[m]))),
                                  'in_cov': int((m & np.isin(reg, list(COV))).sum()),
                                  'dropped_by_R': int(d.sum()), 'ok_pairs': int((m & hit).sum()),
                                  'ok_pairs_dropped_by_R': int((d & hit).sum())}
    out['arch_dropped_by_R'] += int(d.sum())
out['arch_kept_after_R'] = out['arch_rows'] - out['arch_dropped_by_R']
print(json.dumps(out, ensure_ascii=False, indent=1))
