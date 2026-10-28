"""bot-7. Независимая проверка recall по экспортированному файлу (UTM-точки -> круг 10 м, офиц. _match_quality)."""
import json, collections, numpy as np, shapely, sys
from common import *
T,G,greg,gmk,lid,mod=load_gt(); calc=mod.F1Calculator(); del T
R=json.load(open(sys.argv[1]))
by=collections.defaultdict(list)
for r in R: by[r['reg']].append(r)
hit=collections.defaultdict(set)
for j in range(len(G)):
    g=G[j]; b=g.buffer(12)
    for r in by.get(greg[j],[]):
        p=Point(r['ux'],r['uy'])
        if b.contains(p) and calc._match_quality(p.buffer(10),g)[0]: hit[j].add(r['gen'].split('_')[0])
nl=[j for j in range(len(G)) if not lid[j]]
print('all hit',len(hit),'/',len(G),' nonlidar hit',sum(1 for j in nl if j in hit),'/',len(nl))
print('nonlidar by mk', {m:(sum(1 for j in nl if gmk[j]==m and j in hit),sum(1 for j in nl if gmk[j]==m)) for m in set(gmk[nl])})
print('gens that hit (nonlidar):',collections.Counter(g for j in nl for g in hit.get(j,())))
print('missed nonlidar:',[(greg[j],gmk[j],round(2*np.sqrt(G[j].area/np.pi))) for j in nl if j not in hit])
