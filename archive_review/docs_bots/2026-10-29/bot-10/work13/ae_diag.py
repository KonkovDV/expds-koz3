"""bot-10 work13 задача A: разбивка v2 (eb268d6) на честном OOF (honest-10, раскладка k10) в разрезе лога табло:
P/R по типам данных (F1Calculator._build_by_data_type) и полнота по размерам GT (_build_size_categories).
Модель v2 на OOF: стек A (work11/fam0) + S (геофизика: pix_lofo bot-3 вместо стека, правило mkfam_pix 'S')
-> фолдовые пороги (taus_S.json из work7/curves_g_S; unseen=keep|drop) -> post v2 (код из архива eb268d6:
multipole_object_poly, filter_finds_points_regimes, kurgan_cross_modal_twins c dedup_ae_vs_li_m=0).
merge_within_markup не моделируется (в fam0 нет ключа source -> merge_across_sources ничего не сливал бы).
Сопоставление: копия M.match_collected (global, жадно по conf, один-к-одному, official _match_quality),
дополнительно запоминаем пары. Пишем покомпонентные счётчики по участкам (для бутстрепа по семьям).
    python lbv2_oof.py <keep|drop> <out.pkl> [variants=v2,A]"""
import sys, os, json, pickle, time, gc, importlib.util
from collections import defaultdict, Counter
from pathlib import Path
sys.path.insert(0, '/workspace/koz3_shared/bot-10/repo')
from expds import metric as M
from expds.naming import site_family
from shapely.geometry import shape, mapping
from shapely.strtree import STRtree
import numpy as np
M._SHM_CACHE_DIR = Path('/proc/expds_no_cache')
W = Path('/workspace/koz3_shared/bot-10')
V2 = W / 'work13/v2src/expds'
spec = importlib.util.spec_from_file_location('expdsV2', V2 / '__init__.py', submodule_search_locations=[str(V2)])
eV = importlib.util.module_from_spec(spec); sys.modules['expdsV2'] = eV; spec.loader.exec_module(eV)
import expdsV2.post as PV
POL = sys.argv[1]; OUT = sys.argv[2]; VARIANTS = (sys.argv[3] if len(sys.argv) > 3 else 'v2,A').split(',')
MISS = float('-inf') if POL == 'keep' else float('inf')
TAUS = {'v2': json.load(open(W / 'work13/taus_S.json')), 'A': json.load(open(W / 'work6/taus_A.json'))}
LAYOUT = 'k10'
GEOF = ('Гнёздово', 'Семибратнее', 'Судьбищи'); GEOC = ('finds_points', 'object_poly'); KCL = ('kurgany_povrezhdennye', 'kurgany_tselye')
mod = M.load_official(None); calc = mod.F1Calculator()
DTG = mod.DATA_TYPE_GROUPS if hasattr(mod, 'DATA_TYPE_GROUPS') else None
if DTG is None:
    import re
    src = open(os.environ['EXPDS_METRIC'], encoding='utf-8').read()
    DTG = eval(re.search(r'DATA_TYPE_GROUPS = (\{.*?\})', src, re.S).group(1))
DTD = 'прочее'
BINS = list(getattr(calc, 'size_bins', [100, 500, 1000, 5000, 10000]))
print('DTG', DTG, 'BINS', BINS, flush=True)
def dtg(m): return DTG.get(m, DTD) if m else DTD

def need_all(T):
    """(группа,класс) -> мин. порог по фолдам k10; если в каком-то фолде ключа нет — None при keep (берём всё)."""
    fl = T['layouts'][LAYOUT]; keys = set()
    for f in fl:
        for g, d in f['taus'].items():
            for c in d: keys.add((g, c))
    mn = {}
    for k in keys:
        v = [f['taus'].get(k[0], {}).get(k[1]) for f in fl]
        mn[k] = None if (POL == 'keep' and any(x is None for x in v)) else min(x for x in v if x is not None)
    return mn
PIX = defaultdict(list)
for f in json.load(open('/workspace/koz3_shared/bot-3/pixgbm/pix_lofo.geojson'))['features']:
    PIX[f['properties']['region_name'].split('_')[1]].append(f)
feats = {}
for v in VARIANTS:
    mn = need_all(TAUS[v]); feats[v] = {}; n = 0; t = time.time()
    for p in sorted((W / 'work11/fam0').glob('*.jsonl')):
        fam = p.stem; lst = []
        pix = PIX.get(fam, []) if (v == 'v2' and fam in GEOF) else []
        cov = {c: {(f['properties']['region_name'], f['properties']['markup_type']) for f in pix if f['properties']['class_name'] == c} for c in GEOC}
        src = [json.loads(l) for l in open(p, encoding='utf-8')] if True else []
        if pix:
            src = [f for f in src if not (f['properties']['class_name'] in GEOC and (f['properties']['region_name'], M.threshold_group(f['properties'])) in cov[f['properties']['class_name']])] + [f for f in pix if f['properties']['class_name'] in GEOC]
        for f in src:
            pr = f['properties']; g = M.threshold_group(pr); k = (g, pr.get('class_name'))
            if k in mn:
                t0 = mn[k]
                if t0 is not None and float(pr.get('confidence', 0)) < t0: continue
            elif POL == 'drop': continue
            f['_g'] = g; lst.append(f)
        del src
        feats[v][fam] = lst; n += len(lst)
    print('prefilter', v, n, round(time.time() - t), 's', flush=True); gc.collect()

gt_by_fam = defaultdict(list)
for f in json.load(open('/workspace/rerank/gt_train_v2.geojson', encoding='utf-8'))['features']:
    gt_by_fam[site_family(str(f['properties'].get('region_name', 'default_source')))].append(f)
UNIT = {}
for members in json.load(open(W / 'work/family_clusters.json', encoding='utf-8'))['clusters'].values():
    for m in members: UNIT[m] = '+'.join(sorted(members))
MEM = defaultdict(list)
for fam in set(gt_by_fam) | set(feats[VARIANTS[0]]): MEM[UNIT.get(fam, fam)].append(fam)
SEP = '\x1f'
def enc(fc_feats):
    out = []
    for f in fc_feats:
        pr = dict(f['properties']); pr['region_name'] = f"{pr.get('region_name', 'default_source')}{SEP}{pr.get('markup_type') or ''}{SEP}{M.threshold_group(pr) or ''}"
        out.append({'type': 'Feature', 'geometry': f['geometry'], 'properties': pr})
    return out
GTC = {u: M._collect(mod, calc, {'type': 'FeatureCollection', 'features': enc([f for m in ms for f in gt_by_fam.get(m, [])])}, True) for u, ms in MEM.items()}
print('gt collected', flush=True)

def post_v2(kept):
    dd = []
    for f in kept:
        d = dict(f['properties']); d['geom'] = shape(f['geometry']); dd.append(d)
    n0 = len(dd)
    dd = PV.multipole_object_poly(dd, None); n1 = len(dd)
    dd = PV.filter_finds_points_regimes(dd, None); n2 = len(dd)
    dd = PV.kurgan_cross_modal_twins(dd, {'dedup_ae_vs_li_m': 0.0}); n3 = len(dd)
    out = []
    for d in dd:
        g = d.pop('geom'); out.append({'type': 'Feature', 'geometry': mapping(g), 'properties': d})
    return out, (n0, n1, n2, n3)

def match_pairs(p, g):
    """как M.match_collected (global): возвращает hits по p (в исходном порядке p) и множество индексов g."""
    order = sorted(range(len(p)), key=lambda i: p[i][1], reverse=True)
    hit = np.zeros(len(p), dtype=bool); matched = set(); PAIR.clear()
    if p and g:
        gg = [x[0] for x in g]
        tree = STRtree([x.buffer(calc.r_tolerance) if x.geom_type == 'Point' else x for x in gg])
        pg_ord = [p[i][0] for i in order]
        q_pred, q_gt = tree.query(pg_ord)
        if len(q_pred):
            splits = np.flatnonzero(np.diff(q_pred)) + 1
            for rank, cand in zip(q_pred[np.r_[0, splits]], np.split(q_gt, splits)):
                pgm = pg_ord[int(rank)]
                if pgm.geom_type != 'Polygon': continue
                best, bj = float('-inf'), -1
                for j in sorted(int(k) for k in cand):
                    if j in matched: continue
                    ok, s = calc._match_quality(pgm, gg[j])
                    if ok and s > best + 1e-9: best, bj = s, j
                if bj >= 0: matched.add(bj); hit[order[int(rank)]] = True; PAIR[bj] = order[int(rank)]
    return hit, matched

def nb(area):
    for i, u in enumerate(BINS):
        if area <= u: return i
    return len(BINS)
R = {}
PAIR = {}
LIREG = defaultdict(set)
for fam, L in gt_by_fam.items():
    for f in L: LIREG[f['properties']['region_name']].add(f['properties']['markup_type'])
AE = Counter(); AEP = Counter()
for v in VARIANTS:
    t = time.time()
    # region -> счётчики
    cls_cnt = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))   # reg -> cls -> [tp, fp, n_gt]
    dt = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))       # reg -> dtgroup -> [gt, tp(gt grp), pred, matched_pred]
    sz = defaultdict(lambda: defaultdict(lambda: [0, 0]))            # reg -> bin -> [gt, tp]
    szc = defaultdict(lambda: defaultdict(lambda: [0, 0]))           # reg -> 'cls|bin' -> [gt, tp]
    xt = Counter()                                                  # (pred dtg, gt dtg) пар TP
    st = defaultdict(int)
    dup = defaultdict(lambda: defaultdict(lambda: [0, 0])); dupc = defaultdict(Counter)
    for fold in TAUS[v]['layouts'][LAYOUT]:
        held = set(fold['held']); taus = fold['taus']
        for unit in sorted({UNIT.get(site_family(r), site_family(r)) for r in held}):
            kept = [f for m in MEM[unit] for f in feats[v].get(m, []) if f['properties'].get('region_name') in held]
            kept = [f for f in kept if float(f['properties'].get('confidence', 0)) >= taus.get(f['_g'], {}).get(f['properties']['class_name'], MISS)]
            st['kept'] += len(kept)
            if v == 'v2' and kept:
                kept, ns = post_v2(kept); st['post_in'] += ns[0]; st['post_mp'] += ns[1] - ns[0]; st['post_ff'] += ns[2] - ns[1]; st['post_tw'] += ns[3] - ns[2]
            preds = M._collect(mod, calc, {'type': 'FeatureCollection', 'features': enc(kept)}, False)
            gts = GTC.get(unit, {})
            for c in calc.target_classes:
                p = preds.get(c, []); g = gts.get(c, [])
                hit, mt = match_pairs(p, g)
                for j, x in enumerate(g):
                    reg, mk = x[2].split(SEP)[:2]
                    if reg not in held: continue
                    cls_cnt[reg][c][2] += 1; d = dt[reg][dtg(mk)]; d[0] += 1
                    if j in mt: d[1] += 1
                    if x[0].geom_type != 'Point' and x[0].area > 0:
                        b = nb(x[0].area); sz[reg][b][0] += 1; sz[reg][b][1] += int(j in mt)
                        szc[reg][f'{c}|{b}'][0] += 1; szc[reg][f'{c}|{b}'][1] += int(j in mt)
                for j, x in enumerate(g):
                    reg, mk = x[2].split(SEP)[:2]
                    if reg not in held or mk not in ('ae', 'or'): continue
                    hl = 'li' in LIREG[reg]
                    AE[(mk, c, 'li' if hl else 'noli', p[PAIR[j]][2].split(SEP)[2] + '/' + p[PAIR[j]][2].split(SEP)[1] if j in mt else 'MISS')] += 1
                for i, x in enumerate(p):
                    rr, mm, gg = x[2].split(SEP)
                    if mm in ('ae', 'or'): AEP[(mm, c, gg, 'li' if 'li' in LIREG[rr] else 'noli', 'tp' if hit[i] else 'fp')] += 1
                for i, x in enumerate(p):
                    reg, mk = x[2].split(SEP)[:2]
                    cls_cnt[reg][c][0 if hit[i] else 1] += 1
                    d = dt[reg][dtg(mk)]; d[2] += 1; d[3] += int(hit[i])
                # Задача C: FP-дубли — непопавший pred, чья реп. точка внутри попавшего pred того же класса (same/cross markup)
                hi = [i for i in range(len(p)) if hit[i]]
                if hi and len(hi) < len(p):
                    hg = [p[i][0] if p[i][0].geom_type == 'Polygon' else p[i][0].buffer(calc.r_tolerance) for i in hi]
                    tr = STRtree(hg)
                    for i, x in enumerate(p):
                        if hit[i]: continue
                        rp = x[0].representative_point() if x[0].geom_type == 'Polygon' else x[0]
                        reg, mk = x[2].split(SEP)[:2]
                        cand = [hi[int(k)] for k in tr.query(rp) if hg[int(k)].covers(rp)]
                        if not cand: continue
                        same = any(p[k][2].split(SEP)[:2][1] == mk for k in cand)
                        dup[reg][dtg(mk)][0 if same else 1] += 1
                        if same: dupc[reg][c] += 1
    tojs = lambda d: {k: {kk: list(vv) for kk, vv in vv_.items()} for k, vv_ in d.items()}
    R[v] = {'cls': tojs(cls_cnt), 'dt': tojs(dt), 'sz': tojs(sz), 'szc': tojs(szc), 'st': dict(st), 'dup': tojs(dup), 'dupc': {r: dict(c) for r, c in dupc.items()}}
    tot = defaultdict(lambda: [0, 0, 0, 0])
    for reg, dd in dt.items():
        for gname, a in dd.items():
            for i in range(4): tot[gname][i] += a[i]
    print(v, POL, 'by_dt', {k: (a[0], a[1], a[2] - a[3], round(a[1] / (a[1] + 0.5 * (a[2] - a[3]) + 1e-9), 4), round(a[1] / (a[0] + 1e-9), 4)) for k, a in tot.items()}, flush=True)
    ts = defaultdict(lambda: [0, 0])
    for reg, dd in sz.items():
        for b, a in dd.items(): ts[b][0] += a[0]; ts[b][1] += a[1]
    print(v, POL, 'size', {b: (ts[b][0], ts[b][1], round(ts[b][1] / max(ts[b][0], 1), 4)) for b in sorted(ts)}, dict(st), round(time.time() - t), 's', flush=True)
    counts = {c: {} for c in calc.target_classes}
    for reg, cc in cls_cnt.items():
        for c, a in cc.items(): counts[c][reg] = list(a)
    print(v, POL, 'score', round(M.score_counts(counts)[0], 6), flush=True)
    td = defaultdict(lambda: [0, 0])
    for reg, dd in dup.items():
        for gname, a in dd.items(): td[gname][0] += a[0]; td[gname][1] += a[1]
    cnt2 = {c: {r: [a[0], a[1] - dupc.get(r, {}).get(c, 0), a[2]] for r, a in rr.items()} for c, rr in counts.items()}
    print(v, POL, 'dupFP same/cross', dict(td), 'score_without_same_dups', round(M.score_counts(cnt2)[0], 6), flush=True)
    json.dump({'gt': [[*k, n] for k, n in sorted(AE.items())], 'pred': [[*k, n] for k, n in sorted(AEP.items())]}, open(OUT, 'w'), ensure_ascii=False, indent=0)
print('done', flush=True)
