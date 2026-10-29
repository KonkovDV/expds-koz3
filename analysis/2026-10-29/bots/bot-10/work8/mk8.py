"""work8: (a) кандидаты городищ bot-7 (+ OOF p bot-5), (b) convex hull object_poly etgr/terfl (bot-6, bot6-geo 67f84bd).
Каждый вариант = каталог fam_<v>/ : симлинки на все базовые jsonl + переписанные затронутые семьи; _affected.json — затронутые семьи."""
import json, sys, os, collections
from pathlib import Path
sys.path.insert(0, '/workspace/koz3_shared/bot-10/repo')
from expds.naming import site_family
from shapely.geometry import shape, mapping
W = Path(__file__).resolve().parent; G7 = Path('/workspace/koz3_shared/bot-7/work/gor')
P5 = {x['cand_id']: x['p'] for x in json.load(open('/workspace/koz3_shared/bot-5/analysis/gor_patch/scores_cand7.json'))}

def write(v, src, add_by_fam, transform=None):
    d = W/f'fam_{v}'; d.mkdir(exist_ok=True)
    aff = set(add_by_fam) | (set(transform[0]) if transform else set())
    for f in Path(src).glob('*.jsonl'):
        t = d/f.name
        if t.exists() or t.is_symlink(): t.unlink()
        if f.stem not in aff: os.symlink(f.resolve(), t)
    stat = {}
    for fam in aff:
        out = open(d/f'{fam}.jsonl', 'w'); c = collections.Counter()
        sp = Path(src)/f'{fam}.jsonl'
        if sp.exists():
            for l in open(sp):
                if transform and fam in transform[0]:
                    ft = json.loads(l); ft, ch = transform[1](ft); c['hull'] += ch; l = json.dumps(ft, ensure_ascii=False) + '\n'
                out.write(l); c['base'] += 1
        for ft in add_by_fam.get(fam, []): out.write(json.dumps(ft, ensure_ascii=False) + '\n'); c['add'] += 1
        out.close(); stat[fam] = dict(c)
    json.dump(sorted(aff), open(d/'_affected.json', 'w'), ensure_ascii=False)
    json.dump(stat, open(d/'_counts.json', 'w'), ensure_ascii=False, indent=0)
    print(v, len(aff), {k: sum(s.get(k, 0) for s in stat.values()) for k in ['add', 'hull']}, flush=True)

def gor(v, fn, conf):
    by = collections.defaultdict(list)
    for f in json.load(open(G7/fn))['features']:
        p = f['properties']; cid = p['cand_id']
        if conf == 'p':
            if cid not in P5: continue
            c = P5[cid]
        else: c = p['confidence']
        by[site_family(p['region'])].append({'type': 'Feature', 'geometry': f['geometry'], 'properties': {
            'class_name': 'gorodishcha', 'confidence': float(c), 'region_name': p['region'], 'crs': 'EPSG:3857',
            'markup_type': p['markup_type'], 'threshold_group': 'gor7_' + p['markup_type'], 'cand_id': cid, 'gen': p['gen']}})
    write(v, '/workspace/koz3_shared/bot-10/work/fam', by)

def hullf(ft):
    p = ft['properties']
    if p.get('class_name') == 'object_poly' and p.get('markup_type') in ('etgr', 'terfl'):
        g = shape(ft['geometry']).convex_hull
        if g.geom_type == 'Polygon': ft['geometry'] = mapping(g); return ft, 1
    return ft, 0

which = sys.argv[1]
if which == 'hull': write('hull', '/workspace/koz3_shared/bot-10/work/fam', {}, (['Гнёздово'], hullf))
if which == 'hulls6': write('hulls6', '/workspace/koz3_shared/bot-10/work4/fam_sep', {}, (['Гнёздово'], hullf))
if which == 'gor_noc': gor(which, 'cands_gor_oof_D50_nocentre.geojson', 'p')
if which == 'gor_full': gor(which, 'cands_gor_oof_D50.geojson', 'p')
if which == 'gor_cs': gor(which, 'cands_gor_oof_D50_centre_small.geojson', 'p')
if which == 'gor_noc_score': gor(which, 'cands_gor_oof_D50_nocentre.geojson', 'score')

# --- 23:50 МСК: PREREG_gor.md — файлы *_comb, conf = score_comb, NMS 192 м ---
def nms192(rows):
    """Жадный NMS внутри региона: по убыванию score_comb, гасим кандидатов с |dx|<=192 и |dy|<=192 м (одна виньетка bot-5)."""
    import numpy as np
    keep = []; byreg = collections.defaultdict(list)
    for r in rows: byreg[r['reg']].append(r)
    for reg, rs in byreg.items():
        rs.sort(key=lambda r: (-r['score_comb'], r['cand_id']))
        kx = []; ky = []
        for r in rs:
            if kx:
                ax = np.abs(np.asarray(kx) - r['ux']); ay = np.abs(np.asarray(ky) - r['uy'])
                if ((ax <= 192) & (ay <= 192)).any(): continue
            kx.append(r['ux']); ky.append(r['uy']); keep.append(r)
    return keep

def gor_comb(v, base_fn, comb_from=None):
    rows = json.load(open(G7/(base_fn + '.json')))
    if comb_from:  # centre_small: score_comb по cand_id из полного _comb
        sc = {r['cand_id']: r['score_comb'] for r in json.load(open(G7/(comb_from + '.json')))}
        rows = [dict(r, score_comb=sc[r['cand_id']]) for r in rows if r['cand_id'] in sc]
    n0 = len(rows); kept = nms192(rows); ids = {r['cand_id']: r['score_comb'] for r in kept}
    print(v, 'кандидатов', n0, 'после NMS192', len(kept), flush=True)
    geo = base_fn + '.geojson'
    by = collections.defaultdict(list); nf = 0
    for f in json.load(open(G7/geo))['features']:
        p = f['properties']; cid = p['cand_id']
        if cid not in ids: continue
        nf += 1
        by[site_family(p['region'])].append({'type': 'Feature', 'geometry': f['geometry'], 'properties': {
            'class_name': 'gorodishcha', 'confidence': float(ids[cid]), 'region_name': p['region'], 'crs': 'EPSG:3857',
            'markup_type': p['markup_type'], 'threshold_group': 'gor7_' + p['markup_type'], 'cand_id': cid, 'gen': p['gen']}})
    assert nf == len(ids), (nf, len(ids))
    json.dump({'n_in': n0, 'n_nms': len(kept)}, open(W/f'nms_{v}.json', 'w'))
    write(v, '/workspace/koz3_shared/bot-10/work/fam', by)

if which == 'gorc_noc': gor_comb(which, 'cands_gor_oof_D50_nocentre_comb')
if which == 'gorc_full': gor_comb(which, 'cands_gor_oof_D50_comb')
if which == 'gorc_cs': gor_comb(which, 'cands_gor_oof_D50_centre_small', 'cands_gor_oof_D50_comb')
