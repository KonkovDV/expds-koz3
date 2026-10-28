import pickle, numpy as np, json, collections, sys
sys.path.insert(0,'/workspace/rerank'); sys.path.insert(0,'/workspace/expds-koz3')
from expds.naming import site_family
T = pickle.load(open('/workspace/rerank/work/table.pkl','rb')); T.pop('pairs')
reg, cls, grp = T['reg'], T['cls'], T['grp'].astype(str); CL = T['CL']
L = json.load(open('lost.json'))
for r in [x['region'] for x in L['zero_regions'] if x['oracle_all'] >= 0.005]:
    m = reg == r
    print(r, site_family(r), 'cands', int(m.sum()), dict(collections.Counter((CL[c], g) for c, g in zip(cls[m], grp[m])).most_common(6)))
# etgr/terfl GT regions & families
for c in ('object_poly',):
    print({g: collections.Counter(site_family(r) for r, gg in zip(reg[cls==CL.index(c)], grp[cls==CL.index(c)]) if gg==g) for g in ('etgr','terfl','maggr')})
