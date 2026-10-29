"""bot-5. Фильтр городищ v3 (patch gate, NOLIDAR §12 «clean gate», +0,005057): финальная голова и применение.
  fit:   обучить голову на ВСЕХ обучающих сценах (та же процедура, что в gor_patch_eval.py: PCA-64 без меток,
         logreg C=1, чистка Landauer 0,85/0,05 по внутреннему OOF 5 семейных фолдов) -> HEAD.npz;
         пороги групп: sweep_groups по обучению, где sp_or-строки умножены на OOF max-p сцены (как в каждом фолде eval) -> TAUS.json.
  apply: признаки виньеток новой сцены (extract_feats.py) + кандидаты GeoJSON -> GeoJSON с conf_gated и scores JSON.
Гиперпараметры заданы заранее (не подбирались). Env: MODE=fit|apply; см. FOR_BOT7_GOR_V3.md."""
import os, sys, json, hashlib, collections, time
from pathlib import Path
import numpy as np
MODE = os.environ.get('MODE', 'apply'); HERE = os.path.dirname(os.path.abspath(__file__))
def md5(p): return hashlib.md5(open(p, 'rb').read()).hexdigest()
if MODE == 'fit':
    src = Path(os.environ.get('GOR_PATCH_EVAL', os.path.join(HERE, 'gor_patch_eval.py'))).read_text(encoding='utf-8').split('def run_layout(')[0]
    exec(compile(src, 'gor_patch_eval_head', 'exec'))
    HEAD = os.environ.get('HEAD', str(OD / 'gor_v3_head.npz')); TAUS = os.environ.get('TAUS', str(OD / 'gor_v3_taus.json'))
    every = set().union(*layouts['fam10']); tri = ALLIDX[np.isin(PREG, list(every))]
    set_proj(tri)
    Xt = F[tri]; mu = Xt.mean(0); sd = Xt.std(0) + 1e-6   # повтор set_proj для сохранения параметров
    sub = tri[np.random.default_rng(0).permutation(len(tri))[:4000]]
    W = np.linalg.svd((F[sub] - mu) / sd, full_matrices=False)[2][:PC].T; qsd = (((F[tri] - mu) / sd) @ W).std(0) + 1e-6
    mdl, nrem = fit_clean(tri, True); assert mdl[0] == 'lr'
    np.savez(HEAD, mu=mu, sd=sd, W=W, qsd=qsd, w=mdl[1], feat_dim=F.shape[1])
    Qc = np.c_[np.ones(len(tri)), (((F[tri] - mu) / sd) @ W) / qsd]
    assert np.allclose(1 / (1 + np.exp(-Qc @ mdl[1])), pred(mdl, tri), atol=1e-6), 'saved head != in-memory head'
    # p обучающих сцен — внутренний OOF (5 семейных фолдов, каждая модель с чисткой) -> gate -> пороги
    regs = sorted(set(PREG[tri])); fo = family_folds(regs, site_family, 5); ptr = np.zeros(len(tri))
    for h in fo:
        te = np.isin(PREG[tri], list(h))
        if te.any(): ptr[te] = pred(fit_clean(tri[~te], True)[0], tri[te])
    gp = {r: float(ptr[PREG[tri] == r].max()) for r in sorted(set(PREG[tri]) & set(SCN))}
    out = {}
    for nm, g in (('gate', gp), ('base_no_gate', None)):
        res = sweep_groups({G: curve_where(build_curve_rows([], g), every)})
        out[nm] = {k: v['threshold'] for k, v in res[G]['groups'].items()}
    json.dump(dict(head=HEAD, head_md5=md5(HEAD), n_train_patches=int(len(tri)), removed_by_cleaning=nrem, class_G=str(G),
                   taus=out, scene_p_oof_train=gp), open(TAUS, 'w'), ensure_ascii=False, indent=1, default=float)
    log('fit done', HEAD, md5(HEAD), 'taus', out)
else:
    from pyproj import Transformer
    from shapely.geometry import shape
    t0 = time.time()
    H = dict(np.load(os.environ['HEAD'])); F = np.load(os.environ['FEATS'])['F'].astype(np.float32); META = json.load(open(os.environ['META'], encoding='utf-8'))
    assert F.shape[1] == int(H['feat_dim']), 'признаки не той модели (DINOv2-S, 1536)'
    Q = np.c_[np.ones(len(F)), (((F - H['mu']) / H['sd']) @ H['W']) / H['qsd']]; P = 1 / (1 + np.exp(-Q @ H['w']))
    PREG = np.array([m['reg'] for m in META]); PUX = np.array([m['ux'] for m in META]); PUY = np.array([m['uy'] for m in META])
    pscene = {r: float(P[PREG == r].max()) for r in sorted(set(PREG))}
    PC_, PR, PM, MKV = (os.environ.get(k, d) for k, d in (('PROP_CONF', 'conf'), ('PROP_REG', 'reg'), ('PROP_MK', 'mk'), ('MK_VALUE', 'sp_or')))
    CRS_IN = os.environ.get('CRS_IN', 'EPSG:4326'); DATA = os.environ.get('DATA', '/workspace/data/train-expds-3')
    gj = json.load(open(os.environ['CANDS'], encoding='utf-8')); trs = {}; scores = []; st = collections.Counter()
    for i, f in enumerate(gj['features']):
        pr = f.setdefault('properties', {}); r = pr.get(PR); c = shape(f['geometry']).representative_point(); x, y = c.x, c.y
        if CRS_IN != 'utm' and r is not None:
            if r not in trs: trs[r] = Transformer.from_crs(CRS_IN, int(json.load(open(f'{DATA}/{r}/UTM.json'))['crs'].split(':')[-1]), always_xy=True)
            x, y = trs[r].transform(x, y)
        ps = pscene.get(r); pr['gor_p_scene'] = ps
        ii = np.flatnonzero(PREG == r) if r in pscene else np.zeros(0, int)
        w = ii[(np.abs(PUX[ii] - x) <= 192) & (np.abs(PUY[ii] - y) <= 192)] if len(ii) else ii
        k = int(w[np.argmin(np.hypot(PUX[w] - x, PUY[w] - y))]) if len(w) else -1
        pr['gor_p'] = float(P[k]) if k >= 0 else None
        gate = ps is not None and (MKV == '*' or pr.get(PM) == MKV) and pr.get(PC_) is not None
        pr['conf_gated'] = float(pr[PC_]) * ps if gate else pr.get(PC_); st['gated' if gate else ('no_scene' if ps is None else 'not_gated')] += 1
        scores.append(dict(cand_id=pr.get('cand_id', i), p=pr['gor_p'], p_cov_max=float(P[w].max()) if len(w) else None, patch=k, p_scene=ps))
    json.dump(gj, open(os.environ['OUT_GEOJSON'], 'w'), ensure_ascii=False)
    json.dump(scores, open(os.environ['OUT_SCORES'], 'w'))
    print('apply done', dict(st), 'patches', len(F), 'scenes', len(pscene), 'cands', len(scores), f'{time.time()-t0:.2f}s', flush=True)
