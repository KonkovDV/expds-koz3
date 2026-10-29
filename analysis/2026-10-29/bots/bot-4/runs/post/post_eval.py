"""POST (bot-4): (3) neighbour prior, (4) курганы внутри своих городищ, (5) оба класса; поверх R. Объявление: bot-4/POST.md.
Запуск: OMP_NUM_THREADS=1 nice -n 10 venv/bin/python work/post/post_eval.py --layouts canon"""
import sys
_argv = sys.argv[1:]; sys.argv = [sys.argv[0], '--layouts', 'none']
exec(open('/workspace/koz3_shared/bot-4/work/thr/thr_eval.py').read().split('t0 = time.time(); res =')[0])
OUT = Path('/workspace/koz3_shared/bot-4/work/post')
import os as _os
_SFX = ('_' + _os.environ['POST_VARS'].replace(',', '_')) if _os.environ.get('POST_VARS') else ''
if _os.environ.get('KOZ_OUT'): OUT = Path(_os.environ['KOZ_OUT']); OUT.mkdir(parents=True, exist_ok=True)
import shapely
from scipy.spatial import cKDTree
from expds.metric import ClassCurve
ap2 = argparse.ArgumentParser(); ap2.add_argument('--layouts', default='canon'); ap2.add_argument('--boot', type=int, default=1000); a2 = ap2.parse_args(_argv)
KUR = ['kurgany_povrezhdennye', 'kurgany_tselye']; K0, K1 = fast.cid[KUR[0]], fast.cid[KUR[1]]; GOR = fast.cid['gorodishcha']
grp = T['grp']; mk = T['mk']; cx = T['cx']; cy = T['cy']
lat = np.arctan(np.sinh(cy / 6378137.0)); gx = cx * np.cos(lat); gy = cy * np.cos(lat)   # наземные метры (локально)
iskur = (cls == K0) | (cls == K1); isli = iskur & (mk == 'li') & np.array([str(g).startswith('li') for g in grp])
t0 = time.time()
# --- места LiDAR и сиблинги (одно место выдано в обоих классах)
loc_key = {}; sib = np.full(len(cls), -1)
for i in np.flatnonzero(isli):
    loc_key.setdefault((reg[i], grp[i], float(cx[i]), float(cy[i])), []).append(i)
for v in loc_key.values():
    if len(v) == 2 and cls[v[0]] != cls[v[1]]: sib[v[0]] = v[1]; sib[v[1]] = v[0]
print('sib pairs', int((sib >= 0).sum() // 2), f'{time.time()-t0:.0f}s', flush=True)
# --- (3) neighbour prior
nb = np.zeros(len(cls), int)
locs = list(loc_key.items()); L_reg = np.array([k[0] for k, _ in locs], dtype=object); L_grp = np.array([k[1] for k, _ in locs], dtype=object)
L_s = np.array([max(conf[i] for i in v) for _, v in locs]); L_x = np.array([gx[v[0]] for _, v in locs]); L_y = np.array([gy[v[0]] for _, v in locs])
strong = np.zeros(len(locs), bool)
for g in set(L_grp):
    m = L_grp == g; strong[m] = L_s[m] >= np.quantile(L_s[m], 0.99)
L_n = np.zeros(len(locs), int)
for key in set(zip(L_reg, L_grp)):
    m = np.flatnonzero((L_reg == key[0]) & (L_grp == key[1])); ms = m[strong[m]]
    if not len(ms): continue
    tr = cKDTree(np.c_[L_x[ms], L_y[ms]]); cnt = np.array([len(x) for x in tr.query_ball_point(np.c_[L_x[m], L_y[m]], 200.0)])
    L_n[m] = cnt - strong[m].astype(int)
for (k, v), n in zip(locs, L_n): nb[v] = n
conf_nb = conf.copy(); conf_nb[isli] = np.minimum(1.0, conf[isli] * (1 + 0.2 * np.minimum(nb[isli], 3) / 3))
print('nb done', np.bincount(np.minimum(L_n, 5)), f'{time.time()-t0:.0f}s', flush=True)
# --- (4) курганы в полигонах городищ: пары (gor cand, kurgan cand) того же участка
_GW = pickle.load(open(OUT.parent / 'post' / 'gor_wkb.pkl' if (OUT.parent / 'post' / 'gor_wkb.pkl').exists() else '/workspace/koz3_shared/bot-4/work/post/gor_wkb.pkl', 'rb'))
gidx = np.flatnonzero(cls == GOR); assert (gidx == _GW['idx']).all(); gpoly = shapely.from_wkb(_GW['wkb']); del _GW
kidx = np.flatnonzero(iskur); pts = shapely.points(cx[kidx], cy[kidx])
tree = shapely.STRtree(gpoly); pi, gi = tree.query(pts, predicate='within')
same = reg[kidx[pi]] == reg[gidx[gi]]; IN_K = kidx[pi[same]]; IN_G = gidx[gi[same]]
print('kurgan-in-gor pairs', len(IN_K), f'{time.time()-t0:.0f}s', flush=True)
# эталон: курганы внутри эталонных городищ
GT = pickle.load(open(W / 'work/gt_wkb.pkl', 'rb')); gtg = shapely.from_wkb(GT['gorodishcha']); gtr_g = np.array(T['greg']['gorodishcha'], dtype=object)
gt_in = {}
for c in KUR:
    kp = shapely.centroid(shapely.from_wkb(GT[c])); kr = np.array(T['greg'][c], dtype=object)
    a_, b_ = shapely.STRtree(gtg).query(kp, predicate='within'); ok = kr[a_] == gtr_g[b_]
    gt_in[c] = [int(len(set(a_[ok]))), int(len(kp))]
print('GT kurgans inside GT gor', gt_in, flush=True)
# --- (5б) спутниковый hedge: sp_focal vs sp_focal2 расходятся в классе
hedge = []  # (cand, new class id)
spm = iskur & np.isin(grp, ['sp_focal', 'sp_focal2'])
for r in set(reg[spm]):
    m = np.flatnonzero(spm & (reg == r))
    A_ = m[grp[m] == 'sp_focal']; B_ = m[grp[m] == 'sp_focal2']
    if not len(A_) or not len(B_): continue
    for P_, Q_ in ((A_, B_), (B_, A_)):
        tq = cKDTree(np.c_[gx[Q_], gy[Q_]])
        for i, nbr in zip(P_, tq.query_ball_point(np.c_[gx[P_], gy[P_]], 4.0)):
            if any(cls[Q_[j]] != cls[i] for j in nbr): hedge.append((int(i), K1 if cls[i] == K0 else K0))
print('hedge copies', len(hedge), f'{time.time()-t0:.0f}s', flush=True)
# --- кривые курганов с произвольной оценкой и копиями
def kur_curves(score, copies=()):
    out, ids = {}, {}
    for c in KUR:
        ci = fast.cid[c]; base_ix = np.flatnonzero(cls == ci); ex = np.array([i for i, nc in copies if nc == ci], dtype=int)
        ix = np.r_[base_ix, ex]; isc = np.r_[np.zeros(len(base_ix), bool), np.ones(len(ex), bool)]
        o = np.argsort(-score[ix], kind='stable'); ix, isc = ix[o], isc[o]
        hit = np.zeros(len(ix), bool); matched = set()
        for r_, p in enumerate(ix):
            lst = fast.PL.get((CL[cls[p]], c), {}).get(int(p))
            if not lst: continue
            best, bj = -np.inf, -1
            for j, s in lst:
                if j in matched: continue
                if s > best + 1e-9: best, bj = s, j
            if bj >= 0: matched.add(bj); hit[r_] = True
        gl = [grp[p] + ('_hedge' if h else '') for p, h in zip(ix, isc)]
        out[c] = ClassCurve(conf=score[ix].astype(np.float64), hit=hit, pred_region=list(reg[ix]), n_gt=T['n_gt'][c], gt_regions=fast.GREG[c], pred_group=gl)
        ids[c] = (ix, isc)
    return out, ids
def base_ids():
    ids = {}
    for c in CL:
        ci = fast.cid[c]; ix = np.flatnonzero(cls == ci); ix = ix[np.argsort(-conf[ix], kind='stable')]; ids[c] = (ix, np.zeros(len(ix), bool))
    return ids
BID = base_ids()
for c in CL: assert np.allclose(base_cv[c].conf, conf[BID[c][0]])
KC_nb, ID_nb = kur_curves(conf_nb); KC_h, ID_h = kur_curves(conf, hedge); KC_nbh, ID_nbh = kur_curves(conf_nb, hedge)
print('curves', f'{time.time()-t0:.0f}s', flush=True)
PERSIST = {id(x) for d in (base_cv, KC_nb, KC_h, KC_nbh) for x in d.values()}  # кэш кодов только для постоянных кривых (иначе утечка памяти по фолдам)
def subc(v, keep):
    pr = np.asarray(v.pred_region, dtype=object); pg = np.asarray(v.pred_group, dtype=object)
    return ClassCurve(conf=v.conf[keep], hit=v.hit[keep], pred_region=pr[keep].tolist(), n_gt=v.n_gt, gt_regions=v.gt_regions, pred_group=pg[keep].tolist())
def tau_vec(t, c, v):
    tc = t.get(c, {}); return np.array([tc.get(g, np.inf) for g in v.pred_group])
STAT = {}
RID = {r: i for i, r in enumerate(sorted(regions | {r for c in CL for r in fast.GREG[c]}))}
_codes = {}
def held_mask(v, held):
    k = id(v)
    if k not in PERSIST: return np.isin(np.asarray(v.pred_region, dtype=object), list(held))
    if k not in _codes or _codes[k][0] is not v: _codes[k] = (v, np.array([RID[r] for r in v.pred_region], dtype=np.int32))
    hm = np.zeros(len(RID), bool); hm[[RID[r] for r in held]] = True
    return hm[_codes[k][1]]
def fold_eval(counts, cv, ids, train, held, rm, dual, score, tag):
    tb = taus(cv, train); removed = {}
    if rm:
        vg = cv['gorodishcha']; gix = ids['gorodishcha'][0]; kg = vg.conf >= tau_vec(tb, 'gorodishcha', vg)
        kept_g = set(gix[kg].tolist()); bad = set(IN_K[np.isin(IN_G, list(kept_g))].tolist())
        cv = dict(cv)
        for c in KUR:
            ix = ids[c][0]; keep = ~np.isin(ix, list(bad)); removed[c] = keep; cv[c] = subc(cv[c], keep)
        tb = taus(cv, train)
    nk = 0
    for c, v in cv.items():
        hc_mask = held_mask(v, held); per = counts[c]
        for r_, n in v.gt_regions.items():
            if r_ in held: per.setdefault(r_, [0, 0, 0])[2] += int(n)
        tau = tau_vec(tb, c, v); kept = v.conf >= tau
        if dual and c in KUR:
            ix = ids[c][0][removed[c]] if c in removed else ids[c][0]; isc = ids[c][1][removed[c]] if c in removed else ids[c][1]
            s = sib[ix]; ok = (s >= 0) & ~isc
            sv = score[np.where(ok, s, 0)]; own = v.conf
            sec = ok & (own < sv)
            oc = np.where(cls[np.where(ok, s, 0)] == K0, 0, 1)
            # τ сиблинга (тот же grp, другой класс)
            tsib = np.array([tb.get(KUR[o], {}).get(g, np.inf) for o, g in zip(oc, v.pred_group)])
            sib_kept = sv >= tsib
            if rm: sib_kept &= ~np.isin(np.where(ok, s, -1), list(bad))
            kept |= sec & sib_kept & (own >= 0.05)
        sel = kept & hc_mask; nk += int(sel.sum())
        for r_, h in zip(np.asarray(v.pred_region, dtype=object)[sel], v.hit[sel]):
            per.setdefault(r_, [0, 0, 0])[0 if h else 1] += 1
        if rm and c in KUR and tag == 'canon':
            ix = ids[c][0][removed[c]] if c in removed else ids[c][0]
            # точность оставленных курганов внутри/снаружи своих городищ — до удаления считаем отдельно ниже
    return nk
def inside_stats(cv, ids, train, held):
    tb = taus(cv, train); vg = cv['gorodishcha']; kg = vg.conf >= tau_vec(tb, 'gorodishcha', vg)
    kept_g = ids['gorodishcha'][0][kg]; bad = set(IN_K[np.isin(IN_G, kept_g)].tolist()); st = {}
    for c in KUR:
        v = cv[c]; ix = ids[c][0]; kept = (v.conf >= tau_vec(tb, c, v)) & held_mask(v, held)
        ins = np.isin(ix, list(bad))
        st[c] = dict(in_n=int((kept & ins).sum()), in_tp=int((kept & ins & v.hit).sum()), out_n=int((kept & ~ins).sum()), out_tp=int((kept & ~ins & v.hit).sum()))
    return st
VARS = {'R': (None, False, False, False), 'NB': ('nb', False, False, False), 'IN': (None, True, False, False),
        'DUALli': (None, False, True, False), 'DUALsp': ('h', False, False, False), 'DUAL': ('h', False, True, False), 'PKG': ('nbh', True, True, False)}
import os as _os
_SFX = ('_' + _os.environ['POST_VARS'].replace(',', '_')) if _os.environ.get('POST_VARS') else ''
if _os.environ.get('POST_VARS'):  # напр. POST_VARS=NBDli — доп. пакет NB+DUALli (составлен после canon/seeds → помечен post-hoc)
    VARS['NBDli'] = ('nb', False, True, False); VARS = {k: VARS[k] for k in ['R'] + _os.environ['POST_VARS'].split(',')}
def run(layout):
    folds = canon if layout == 'canon' else (family_folds(regions, site_family, None) if layout == 'lofo' else rff(regions, 10, int(layout[4:])))
    outer, inner_of = load_rr(layout); fold_of = {r: i for i, f in enumerate(folds) for r in f}; fid = np.array([fold_of[r] for r in reg])
    counts = {v: {c: {} for c in CL} for v in ['A'] + list(VARS)}; nobj = {v: 0 for v in counts}; nobj_reg = {v: Counter() for v in counts}; every = set().union(*folds); ins = []
    for k, held in enumerate(folds):
        train = every - held
        cvR = dict(base_cv); cvR['arkhitektury'] = arch_curve(mixed(outer, inner_of(k), fid != k))
        nobj['A'] += fold_eval(counts['A'], base_cv, BID, train, held, False, False, conf, layout)
        for v, (kc, rm, dual, _) in VARS.items():
            cv = dict(cvR); ids = dict(BID); score = conf
            if kc == 'nb': cv.update(KC_nb); ids.update(ID_nb); score = conf_nb
            if kc == 'h': cv.update(KC_h); ids.update(ID_h)
            if kc == 'nbh': cv.update(KC_nbh); ids.update(ID_nbh); score = conf_nb
            nobj[v] += fold_eval(counts[v], cv, ids, train, held, rm, dual, score, layout)
        if layout == 'canon': ins.append(inside_stats(cvR, BID, train, held))
        if layout == 'lofo' and k % 10 == 0: print('lofo', k, f'{time.time()-t0:.0f}s', flush=True)
    return counts, nobj, ins
from collections import Counter
res = {'gt_in_gor': gt_in, 'hedge_copies': len(hedge), 'sib_pairs': int((sib >= 0).sum() // 2), 'nb_hist': np.bincount(np.minimum(L_n, 5)).tolist()}
for lay in a2.layouts.split(','):
    cnt, nobj, ins = run(lay); Lr = {'n_objects_kept_sum_over_heldout': nobj}
    for v in cnt:
        Lr[v] = summary(cnt[v], cnt['R']); Lr[v]['vsA'] = summary(cnt[v], cnt['A'])['d']
        if lay == 'canon' and v not in ('A', 'R'): Lr[v]['boot'] = paired_bootstrap_counts(cnt[v], cnt['R'], site_family, a2.boot)
    if lay == 'canon':
        tot = {c: {k: sum(x[c][k] for x in ins) for k in ins[0][c]} for c in KUR}; Lr['inside_stats'] = tot
        pickle.dump(cnt, open(OUT / f'counts_post_canon{_SFX}.pkl', 'wb'))
    res[lay] = Lr
    print(lay, {v: (round(Lr[v]['score'][0], 6), round(Lr[v]['d'], 5)) for v in cnt}, nobj, f'{time.time()-t0:.0f}s', flush=True)
    json.dump(res, open(OUT / f'res_post_{a2.layouts.replace(",", "_")}{_SFX}.json', 'w'), ensure_ascii=False, indent=1, default=float)
print('done', time.time() - t0)
