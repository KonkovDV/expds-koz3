"""Граница эффекта [raster_fix] по честным счётчикам базы (honest-10, /workspace/rerank/work/base_honest.pkl).
Единственная сеть ae в сдаче — маска фортификаций, значит правки меняют только класс fortifikatsii
на участках, где они срабатывают: 015, 029 (3857), 106 (int16 ×3,04), 017, 036 (чёрные поля)."""
import copy, json, pickle, sys
sys.path.insert(0, '/workspace/koz3_shared/bot-6/repo')
from expds.metric import score_counts
b = pickle.load(open('/workspace/rerank/work/base_honest.pkl', 'rb'))
cn = b['counts']
AFF = ('015', '029', '106', '017', '036')
base, per = score_counts(cn)
out = {'base': base, 'fort_total': per['fortifikatsii'],
       'fort_affected': {r: v for r, v in cn['fortifikatsii'].items() if r[:3] in AFF}}
def sc(fn):
    c = copy.deepcopy(cn); fn(c['fortifikatsii']); return score_counts(c)[0] - base
def kill_fp(d):
    for r in d:
        if r[:3] in AFF: d[r][1] = 0
def lose_tp(d):
    for r in d:
        if r[:3] in AFF: d[r][1] += d[r][0]; d[r][0] = 0   # попадание уходит, кандидат остаётся ложным
def ideal(d):
    for r in d:
        if r[:3] in AFF: d[r][0] = d[r][2]; d[r][1] = 0
out['delta_all_fp_removed'] = sc(kill_fp)
out['delta_tp_lost'] = sc(lose_tp)
out['delta_ideal_affected'] = sc(ideal)
def one_fp(d):
    for r in d:
        if r[:3] == '106': d[r][1] -= 1
out['delta_per_fp_removed'] = sc(one_fp)
print(json.dumps(out, ensure_ascii=False, indent=1))
json.dump(out, open('bound.json', 'w'), ensure_ascii=False, indent=1)
