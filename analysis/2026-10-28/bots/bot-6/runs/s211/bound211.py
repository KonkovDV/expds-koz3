"""Оценка (не честная) для 211 с maps="all": подставить локальные счётчики local211.log в honest-10 базы."""
import copy, json, pickle, sys
sys.path.insert(0, '/workspace/koz3_shared/bot-6/repo')
from expds.metric import score_counts
b = pickle.load(open('/workspace/rerank/work/base_honest.pkl', 'rb')); cn = b['counts']
R = [r for r in cn['finds_points'] if r.startswith('211')][0]
base, per = score_counts(cn)
out = {'base': base, 'finds_211': cn['finds_points'].get(R), 'opoly_211': cn['object_poly'].get(R), 'finds_total': per['finds_points']}
c = copy.deepcopy(cn); c['finds_points'][R] = [56, 7, 409]; c['object_poly'][R] = [0, 2, cn['object_poly'][R][2]]
out['delta_local_all'] = score_counts(c)[0] - base
c2 = copy.deepcopy(cn); c2['finds_points'][R] = [56, 7, 409]
out['delta_finds_only'] = score_counts(c2)[0] - base
print(json.dumps(out, ensure_ascii=False, indent=1)); json.dump(out, open('bound211.json', 'w'), ensure_ascii=False, indent=1)
