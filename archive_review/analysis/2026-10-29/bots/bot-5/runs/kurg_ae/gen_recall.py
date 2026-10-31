# recall of generator candidates vs Ae barrow GT (inside polygon), per region
import glob, json, pickle, numpy as np, shapely, collections, sys
T = pickle.load(open('/workspace/rerank/work/table.pkl', 'rb'))
W = pickle.load(open('/workspace/rerank/work/gt_wkb.pkl', 'rb'))
gt = json.load(open('/workspace/rerank/gt_train_v2.geojson'))
CLS = ('kurgany_povrezhdennye', 'kurgany_tselye')
G = {}; 
for C in CLS:
    mks = np.array([str(f['properties'].get('markup_type')) for f in gt['features'] if f.get('geometry') and (f['properties'].get('class_name') or '') == C])
    g = shapely.from_wkb(W[C]); greg = np.array(T['greg'][C])
    G[C] = (g[mks == 'ae'], greg[mks == 'ae'])
by = collections.defaultdict(list)
for f in sorted(glob.glob('/workspace/koz3_shared/bot-5/tmp/kurg_ae/gen/*.npz')):
    z = np.load(f); by[str(z['reg'])].append(z['cands'])
tot = collections.Counter()
for reg, L in sorted(by.items()):
    c = np.concatenate(L); pts = shapely.points(c[:, 0], c[:, 1]); tree = shapely.STRtree(pts)
    area = sum(x[0, 5] if len(x) else 0 for x in L)
    s = f'{reg}: n={len(c)}'
    for C in CLS:
        g, gr = G[C]; g = g[gr == reg]
        if not len(g): continue
        hit = np.zeros(len(g), bool); hit3 = np.zeros(len(g), bool)
        q = tree.query(g, predicate='intersects'); hit[np.unique(q[0])] = True
        q = tree.query(shapely.buffer(g, 3.0), predicate='intersects'); hit3[np.unique(q[0])] = True
        s += f' | {C[8:12]} {hit.sum()}/{len(g)} (+3m {hit3.sum()})'
        tot[C, 'n'] += len(g); tot[C, 'h'] += hit.sum(); tot[C, 'h3'] += hit3.sum()
    print(s)
print(dict(tot))
