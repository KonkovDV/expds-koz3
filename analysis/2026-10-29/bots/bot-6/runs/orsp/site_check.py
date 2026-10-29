"""Site-scope проверка SPF (bot-6): (1) есть ли в table.pkl пары кандидат–эталон разных участков (если нет — global ≡ site для этого кэша);
(2) парный бутстреп по участкам (а не семьям) на сохранённых canon-счётчиках counts_spf_canon.pkl."""
import sys, json, pickle
import numpy as np
sys.path.insert(0, '/workspace/rerank'); sys.path.insert(0, '/workspace/expds-koz3')
from common import paired_bootstrap_counts, score_counts
T = pickle.load(open('/workspace/rerank/work/table.pkl', 'rb'))
cross = {}
for (pc, gc), (P, J, S) in T['pairs'].items():
    gr = np.asarray(T['greg'][gc], dtype=object)[J]; cross[f'{pc}->{gc}'] = [int((T['reg'][P] != gr).sum()), int(len(P))]
cnt = pickle.load(open('counts_spf_canon.pkl', 'rb'))
res = {'cross_region_pairs': cross, 'd_canon': score_counts(cnt['SPF'])[0] - score_counts(cnt['A'])[0],
       'site_boot': paired_bootstrap_counts(cnt['SPF'], cnt['A'], lambda r: r, 1000)}
json.dump(res, open('res_site.json', 'w'), ensure_ascii=False, indent=1); print(json.dumps(res, ensure_ascii=False))
