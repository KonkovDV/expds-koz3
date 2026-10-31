"""Шаг 1б (bot-3): разбор 514 FN object_poly «рядом, но не проходит геометрию» (S, honest-10 global keep).
Расстояние от центра ближайшего кандидата S до полигона GT; пересекает ли GT компонента P≥0,5 / P≥0,3 (oof, LOFO);
сколько GT на одну компоненту (слияние); попадает ли argmax-P-подобная точка — нет растра, поэтому только геометрия."""
from common import *
import glob
c = 'object_poly'; P = SV[c]; G = GTS[c]; gg = [x[0] for x in G]
keep = kept_mask(P, c, 'keep', 'global'); kid = [i for i in range(len(P)) if keep[i]]
m = greedy(P, c, kid); tpj = set(m.values())
cen = [p[0].centroid for p in P]; pt = STRtree([p[0] for p in P])
feats = []
for f in sorted(glob.glob('/workspace/koz3_shared/bot-3/pixgbm/oof/2*.geojson')):
    for ft in json.load(open(f))['features']:
        pr = ft['properties']
        feats.append({'type': 'Feature', 'geometry': ft['geometry'], 'properties': {
            'class_name': 'object_poly', 'confidence': pr['score'], 'region_name': pr['region'],
            'threshold_group': f"{pr['markup_type']}|{pr['t_pix']}|{pr['max_p']:.4f}"}})
comp = M._collect(mod, calc, {'type': 'FeatureCollection', 'features': feats}, False)['object_poly']
CO = {t: [x for x in comp if x[3].split('|')[1] == t] for t in ('0.3', '0.5', '0.7')}
CT = {t: STRtree([x[0] for x in v]) for t, v in CO.items()}
print({t: len(v) for t, v in CO.items()}, flush=True)
rec = collections.Counter(); dist = []; ngt_comp = collections.Counter(); a_comp = []; maxp_in = []
for j in range(len(G)):
    if j in tpj: continue
    b5 = gg[j].buffer(5.0)
    near = [int(i) for i in pt.query(b5) if P[int(i)][2] == G[j][2] and b5.contains(cen[int(i)])]
    if not near or any(calc._match_quality(P[i][0], gg[j])[0] for i in near): continue
    rec['geomfail'] += 1
    d = min(gg[j].distance(cen[i]) for i in near); dist.append(d)
    for t in ('0.3', '0.5', '0.7'):
        hits = [int(k) for k in CT[t].query(gg[j]) if CO[t][int(k)][2] == G[j][2] and CO[t][int(k)][0].intersects(gg[j])]
        if hits:
            rec[f'GT_пересекает_компоненту_t{t}'] += 1
            if t == '0.5':
                k = max(hits, key=lambda k: CO[t][k][0].intersection(gg[j]).area); cg = CO[t][k][0]
                a_comp.append(cg.area)
                n = sum(1 for jj in STRtree(gg).query(cg) if G[int(jj)][2] == G[j][2] and gg[int(jj)].intersects(cg))
                ngt_comp[min(n, 5)] += 1
                if cg.intersection(gg[j]).area / max(gg[j].area, 1e-9) >= 0.5: rec['GT_покрыт_компонентой05_≥50%'] += 1
q = lambda a: np.percentile(a, [10, 25, 50, 75, 90]).round(2).tolist()
dist = np.array(dist)
out = dict(counts=dict(rec), dist_center_to_gt_p10_90=q(dist),
           dist_hist={'≤0,5': int((dist <= .5).sum()), '0,5–1': int(((dist > .5) & (dist <= 1)).sum()),
                      '1–2': int(((dist > 1) & (dist <= 2)).sum()), '2–5': int((dist > 2).sum())},
           comp05_area_p10_90=q(a_comp) if a_comp else None, n_gt_per_comp05=dict(ngt_comp))
print(json.dumps(out, ensure_ascii=False)); json.dump(out, open(f'{W}/diag2_S.json', 'w'), ensure_ascii=False, indent=1, default=float)
