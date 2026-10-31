"""Порог ≥0,45 для группы or (архитектура ортофото) на стеке A (/workspace/rerank/work/table.pkl), honest-10 / LOFO / 5 семян.
conf всех or-строк архитектуры < 0,45 (макс. 0,391) ⇒ вариант = убрать or-строки архитектуры. База — A (не A+R: rerank_arch меняет arch)."""
import json
src = open('/workspace/koz3_shared/bot-5/work/hillfort/gor_scene.py', encoding='utf-8').read()
exec(src.split("chk = gor_counts")[0].replace("\ndel curves\n", "\n"))
from expds.metric import ClassCurve
KA = 'arkhitektury'
cv = curves[KA]; g = np.asarray(cv.pred_group); keep = g != 'or'
print('or rows in arch curve', int((~keep).sum()), 'max conf', float(cv.conf[~keep].max()))
var = dict(curves); var[KA] = ClassCurve(conf=cv.conf[keep], hit=cv.hit[keep], pred_region=list(np.asarray(cv.pred_region, dtype=object)[keep]),
                                        n_gt=cv.n_gt, gt_regions=cv.gt_regions, pred_group=list(g[keep]))
ORR = ['008_Герите', '014_Лешки', '056_Фалхан', '093_Старый_Барах']
out = {'base_A': A}
lays = {'fam10': folds10, 'LOFO': family_folds(regions, site_family, None)}
from importlib import import_module
try:
    rff = getattr(M, 'random_family_folds')
except AttributeError:
    rff = None
for k, f in lays.items():
    b, _ = honest_counts(curves, f); v, _ = honest_counts(var, f)
    out[k] = {'delta': score_counts(v)[0] - score_counts(b)[0],
              'or_regions_base_arch': {r: b[KA].get(r) for r in ORR}, 'or_regions_var_arch': {r: v[KA].get(r) for r in ORR}}
    if k == 'fam10':
        out[k]['bootstrap'] = paired_bootstrap_counts(v, b, site_family, n=1000, seed=0)
        out[k]['delta_optical'] = score_counts(v, optical)[0] - score_counts(b, optical)[0]
    print(k, out[k], flush=True)
json.dump(out, open('/workspace/koz3_shared/bot-5/analysis/or_arch/or_thr_honest.json', 'w'), ensure_ascii=False, indent=1, default=float)
