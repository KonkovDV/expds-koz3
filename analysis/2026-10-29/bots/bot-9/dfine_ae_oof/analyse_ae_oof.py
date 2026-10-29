#!/usr/bin/env python3
"""OOF D-FINE Ae (koz3-dfine-ae): P/R по классам при порогах, recall по площади, FP по участкам, крупные городища.
Геометрии — через cm._extract_geometry (официальный загрузчик: пустые координаты/None отбрасываются, MultiPolygon ->
крупнейшая часть, make_valid). Матчинг = официальный F1Calculator._soft_match (koz3-1.5.0, bot-2/repo/materials/compute_metrics.py), точное
эквивалентное воспроизведение: пары pred–GT (hit, cov) считаются официальным _match_polygon_gt в UTM
(_to_utm 'webmercator'), только для пар с пересечением bbox в 3857 (+20 м) — пары без пересечения не могут дать hit
(cov>0 требует пересечения). Затем жадный проход по conf (стабильная сортировка в порядке файла, выбор GT с макс. cov,
tie-break score > best+1e-9 в порядке индекса GT) внутри (участок, класс). FP = n_pred - TP, FN = n_gt - TP.
Геометрии предсказаний читаются из исходного файла по смещениям (oof_index.npz), без json.load всего файла."""
import json, glob, sys, collections, time, csv
import numpy as np, shapely
sys.path.insert(0, '/workspace/koz3_shared/bot-2/repo/materials')
import compute_metrics as cm
from shapely.geometry import shape, Polygon
from shapely.strtree import STRtree
SRC = '/workspace/kaggle_out/koz3-dfine-ae/oof_dfine_ae.geojson'
IDX = '/workspace/koz3_shared/bot-9/dfine_ae_oof/oof_index.npz'
CLS = ["kurgany_tselye", "kurgany_povrezhdennye", "gorodishcha", "fortifikatsii", "arkhitektury"]
RU = {'архитектуры': 'arkhitektury', 'городища': 'gorodishcha', 'фортификации': 'fortifikatsii',
      'курганы_целые': 'kurgany_tselye', 'курганы_поврежденные': 'kurgany_povrezhdennye'}
THR = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MARGIN = 20.0
calc = cm.F1Calculator(); utm = lambda g: cm._to_utm(g, 'webmercator')
t0 = time.time(); log = lambda *a: print(f'[{time.time()-t0:7.1f}s]', *a, file=sys.stderr, flush=True)
D = np.load(IDX); REG = [str(r) for r in D['regions']]; TIFS = [str(t) for t in D['tifs']]
off, ln, reg, cls, conf, fold, tif = (D[k] for k in ('off', 'ln', 'reg', 'cls', 'conf', 'fold', 'tif'))
bb = np.stack([D['x0'], D['y0'], D['x1'], D['y1']], 1)
N = len(conf); VALID = ~np.isnan(bb[:, 0]); log('preds', N, 'valid', int(VALID.sum()))
fold_of_reg = {r: int(np.bincount(fold[reg == i]).argmax()) for i, r in enumerate(REG)}
dr = json.load(open('/workspace/kaggle_out/koz3-dfine-ae/dfine_report.json'))
for f, v in dr['folds'].items():
    for r in v['test_regions']: fold_of_reg.setdefault(r, int(f))
ALLREG = sorted(fold_of_reg)

def load_gt(markup, regions):
    gt = []
    for r in regions:
        for f in sorted(glob.glob(f'/workspace/data/train-expds-3/{r}/06_*/*_{markup}_*.geojson')):
            c = RU[f.split(f'_{markup}_')[-1][:-8]]
            for i, ft in enumerate(json.load(open(f))['features']):
                g3 = cm._extract_geometry(ft)  # как в официальном загрузчике: пустые/None -> пропуск, Multi -> крупнейшая часть
                if g3 is None: gt.append(dict(region=r, cls=c, geom=None, i=i, file=f.split('/')[-1])); continue
                g = utm(g3)
                gt.append(dict(region=r, cls=c, g3=g3, geom=g, area=g.area, i=i, file=f.split('/')[-1]))
    return gt
gt = load_gt('Ae', ALLREG); log('gt', len(gt), 'with geom', sum(g['geom'] is not None for g in gt))

# --- кандидатные пары по bbox ---
pairs = []  # (pred_idx, gt_k)
for ri, r in enumerate(REG):
    G = [k for k, g in enumerate(gt) if g['region'] == r and g['geom'] is not None and not g['g3'].is_empty]
    if not G: continue
    tree = STRtree([shapely.box(*(np.array(gt[k]['g3'].bounds) + [-MARGIN, -MARGIN, MARGIN, MARGIN])) for k in G])
    pi = np.nonzero((reg == ri) & ~np.isnan(bb[:, 0]))[0]
    for s in range(0, len(pi), 200000):
        sub = pi[s:s+200000]
        boxes = shapely.box(bb[sub, 0], bb[sub, 1], bb[sub, 2], bb[sub, 3])
        a, b = tree.query(boxes, predicate='intersects')
        pairs += list(zip(sub[a].tolist(), [G[j] for j in b]))
        del boxes
log('bbox pairs', len(pairs))
cand = sorted({p for p, _ in pairs}); log('cand preds', len(cand))
PG = {}
with open(SRC, 'rb') as f:
    for n, p in enumerate(cand):
        f.seek(int(off[p])); bt = f.read(int(ln[p])).rstrip()
        if bt.endswith(b']}'): bt = bt[:-2].rstrip()
        if bt.endswith(b','): bt = bt[:-1]
        ft = json.loads(bt); pr = ft['properties']
        assert abs(float(pr['confidence']) - conf[p]) < 1e-6 and pr['region_name'] == REG[reg[p]]
        PG[p] = utm(cm._extract_geometry(ft))
        if n % 50000 == 0: log('read', n)
HIT = {}  # (p,k) -> (hit, cov)
for p, k in pairs:
    pg = PG[p]
    if not isinstance(pg, Polygon): continue  # официальный матчинг пропускает не-Polygon
    h, c = calc._match_polygon_gt(pg, gt[k]['geom'])
    if c > 0: HIT[(p, k)] = (bool(h), float(c))
log('pairs with cov>0', len(HIT), 'hits', sum(v[0] for v in HIT.values()))
hits_of = collections.defaultdict(list)  # p -> [(k,cov)] только hit
for (p, k), (h, c) in HIT.items():
    if h: hits_of[p].append((k, c))

order = np.lexsort((np.arange(N), -conf.astype(np.float64)))  # conf desc, стабильно по порядку файла
def greedy(thr):
    """-> matched: dict gt_k -> pred p (по классам внутри участка)"""
    matched = {}; used = set()
    for p in order:
        if conf[p] < thr: break
        hl = hits_of.get(int(p))
        if not hl: continue
        best, bk = float('-inf'), -1
        for k, c in sorted(hl):  # порядок GT = порядок индекса (внутри участок/класс порядок k совпадает)
            if gt[k]['cls'] != CLS[cls[p]] or k in used: continue
            if c > best + 1e-9: best, bk = c, k
        if bk >= 0: used.add(bk); matched[bk] = int(p)
    return matched
def agnostic(thr):
    s = set()
    for (p, k), (h, c) in HIT.items():
        if h and conf[p] >= thr: s.add(k)
    return s
bins = lambda a: '<500' if a < 500 else ('500-5000' if a <= 5000 else '>5000')
f1k = lambda tp, fp, fn: round(2 * (tp / (tp + .5 * fp + 1e-9)) * (tp / (tp + fn + 1e-9)) / ((tp / (tp + .5 * fp + 1e-9)) + (tp / (tp + fn + 1e-9)) + 1e-9), 4)
def row(tp, fp, fn): return dict(tp=tp, fp=fp, fn=fn, P=round(tp/(tp+fp), 4) if tp+fp else None, R=round(tp/(tp+fn), 4) if tp+fn else None, F1_koz=f1k(tp, fp, fn))
out = dict(oof=SRC, n_pred=int(N), n_gt_total=len(gt), n_gt_geom=sum(g['geom'] is not None for g in gt),
           n_empty_pred_geom=int(np.isnan(bb[:, 0]).sum()), bbox_pairs=len(pairs), cand_preds=len(cand), fold_of_region=fold_of_reg)
out['gt_by_region_cls'] = dict(collections.Counter(f"{g['region']}|{g['cls']}" for g in gt if g['geom'] is not None))
out['gt_empty_after_buffer0'] = [(g['region'], g['file'], g['i']) for g in gt if g['geom'] is not None and g['g3'].is_empty]
out['gt_dropped_by_extract_geometry'] = dict(collections.Counter(f"{g['region']}|{g['cls']}" for g in gt if g['geom'] is None))
out['gt_none_geom'] = dict(collections.Counter(g['region'] for g in gt if g['geom'] is None))
ngt = collections.Counter((g['region'], g['cls']) for g in gt if g['geom'] is not None)
out['per_threshold'] = {}; out['per_region'] = {}; out['per_fold'] = {}; M = {}
for t in THR:
    m = greedy(t); M[t] = m
    sel = (conf >= t) & VALID
    npred = collections.Counter(zip(reg[sel].tolist(), cls[sel].tolist()))
    tpc = collections.Counter((gt[k]['region'], gt[k]['cls']) for k in m)
    keys = set((REG[r], CLS[c]) for r, c in npred) | set(ngt)
    per = collections.defaultdict(lambda: [0, 0, 0]); perr = collections.defaultdict(lambda: [0, 0, 0]); perf = collections.defaultdict(lambda: [0, 0, 0])
    for (r, c) in keys:
        np_ = npred.get((REG.index(r), CLS.index(c)), 0) if r in REG else 0
        tp = tpc.get((r, c), 0); v = (tp, np_ - tp, ngt.get((r, c), 0) - tp)
        for d, kk in ((per, c), (perr, r), (perf, fold_of_reg[r])):
            for i in range(3): d[kk][i] += v[i]
        if t in (0.05, 0.3, 0.5): out['per_region'].setdefault(f'{r}@{t}', {})[c] = dict(tp=v[0], fp=v[1], fn=v[2])
    rowd = {c: row(*v) for c, v in sorted(per.items())}
    tot = [sum(v[i] for v in per.values()) for i in range(3)]; rowd['_all'] = row(*tot)
    rowd['_macroF1_present_classes'] = round(np.mean([rowd[c]['F1_koz'] for c in per if per[c][0] + per[c][2] > 0]), 4)
    ab = collections.defaultdict(lambda: [0, 0]); abg = collections.defaultdict(lambda: [0, 0]); ag = agnostic(t)
    abc = collections.defaultdict(lambda: [0, 0])
    for k, g in enumerate(gt):
        if g['geom'] is None: continue
        b = bins(g['area']); ab[b][1] += 1; ab[b][0] += k in m; abg[b][1] += 1; abg[b][0] += k in ag
        abc[f"{g['cls']}|{b}"][1] += 1; abc[f"{g['cls']}|{b}"][0] += k in m
    rowd['_recall_by_area_same_class'] = {b: f'{v[0]}/{v[1]}' for b, v in sorted(ab.items())}
    rowd['_recall_by_area_any_class'] = {b: f'{v[0]}/{v[1]}' for b, v in sorted(abg.items())}
    rowd['_recall_by_cls_area'] = {b: f'{v[0]}/{v[1]}' for b, v in sorted(abc.items())}
    out['per_threshold'][str(t)] = rowd
    out['per_region'][f'_totals@{t}'] = {r: dict(tp=v[0], fp=v[1], fn=v[2]) for r, v in sorted(perr.items())}
    out['per_fold'][str(t)] = {str(f): row(*v) for f, v in sorted(perf.items())}
    log('thr', t, tot)
# GT >5000 м² Ae — каждый
def best_inter(k):
    l = [(round(float(conf[p]), 4), round(c, 3), CLS[cls[p]], h) for (p, kk), (h, c) in HIT.items() if kk == k]
    return dict(n_intersecting=len(l), max_conf=max(l)[0] if l else None,
                best_cov=max(l, key=lambda x: (x[1], x[0])) if l else None,
                best_hit_conf=max([x for x in l if x[3]], default=None))
Kby = collections.defaultdict(list)
for (p, k) in HIT: Kby[k].append(p)
out['gt_gt5000_ae'] = []
for k, g in enumerate(gt):
    if g['geom'] is not None and g['area'] > 5000:
        hl = [(round(float(conf[p]), 4), round(HIT[(p, k)][1], 3), CLS[cls[p]], HIT[(p, k)][0]) for p in Kby[k]]
        out['gt_gt5000_ae'].append(dict(region=g['region'], cls=g['cls'], file=g['file'], idx=g['i'], area=round(g['area'], 1),
            matched_same_class={str(t): k in M[t] for t in (0.05, 0.3)},
            n_intersecting=len(hl), max_conf=max(hl)[0] if hl else None,
            best_cov=max(hl, key=lambda x: (x[1], x[0])) if hl else None,
            best_hit_any_class=max([x for x in hl if x[3]], default=None),
            best_gor_any=max([x for x in hl if x[2] == 'gorodishcha'], default=None)))
# 45 крупных городищ bot 2 без кандидатов
rows = [r for r in csv.DictReader(open('/workspace/koz3_shared/bot-2/work/bigobj/A/gt_rows.csv'))
        if r['cls'] == 'gorodishcha' and float(r['area']) > 5000 and r['reason'] == 'no_cand']
out['bot2_45_n'] = len(rows); out['bot2_45_markups'] = dict(collections.Counter(r['markup'] for r in rows))
MK = {'ae': 'Ae', 'or': 'Or', 'sp_or': 'SpOr', 'li': 'Li'}
other_gt = {}
res45 = []
for r in rows:
    if r['region'] not in REG and r['region'] not in fold_of_reg:
        res45.append(dict(region=r['region'], markup=r['markup'], area=float(r['area']), in_ae_oof=False)); continue
    mk = MK.get(r['markup'], r['markup'])
    if mk not in other_gt: other_gt[mk] = load_gt(mk, ALLREG) if mk != 'Ae' else gt
    cands = [g for g in other_gt[mk] if g['region'] == r['region'] and g['cls'] == 'gorodishcha' and g['geom'] is not None and abs(g['area'] - float(r['area'])) < 1.0]
    d = dict(region=r['region'], markup=r['markup'], area=float(r['area']), in_ae_oof=True, gt_found=len(cands))
    if len(cands) == 1:
        g = cands[0]; d['file'] = g['file']; d['idx'] = g['i']
        ri = REG.index(g['region']) if g['region'] in REG else -1
        sel = np.nonzero((reg == ri) & ~np.isnan(bb[:, 0]) & (bb[:, 2] >= g['g3'].bounds[0] - MARGIN) & (bb[:, 0] <= g['g3'].bounds[2] + MARGIN)
                         & (bb[:, 3] >= g['g3'].bounds[1] - MARGIN) & (bb[:, 1] <= g['g3'].bounds[3] + MARGIN))[0]
        l = []
        with open(SRC, 'rb') as f:
            for p in sel:
                if p in PG: pg = PG[p]
                else:
                    f.seek(int(off[p])); bt = f.read(int(ln[p])).rstrip()
                    bt = bt[:-2].rstrip() if bt.endswith(b']}') else bt
                    bt = bt[:-1] if bt.endswith(b',') else bt
                    pg = utm(cm._extract_geometry(json.loads(bt)))
                if not pg.intersects(g['geom']): continue
                h, c = calc._match_polygon_gt(pg, g['geom'])
                l.append((round(float(conf[p]), 4), round(c, 3), CLS[cls[p]], bool(h)))
        d.update(n_intersecting=len(l), max_conf=max(l)[0] if l else None,
                 best_cov=max(l, key=lambda x: (x[1], x[0])) if l else None,
                 hit_any_class=max([x for x in l if x[3]], default=None),
                 hit_gor=max([x for x in l if x[3] and x[2] == 'gorodishcha'], default=None),
                 max_conf_gor=max([x for x in l if x[2] == 'gorodishcha'], default=None))
    res45.append(d)
out['bot2_45'] = res45
log('done')
json.dump(out, open('/workspace/koz3_shared/bot-9/dfine_ae_oof/ae_oof_analysis.json', 'w'), ensure_ascii=False, indent=1)
