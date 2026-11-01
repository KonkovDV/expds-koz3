"""Parity of expds.w5cf_infer against the W5 training table (offline).

For each region: the run objects are rebuilt from stack_arch.geojson (same GeoJSON the training table was made of),
region_features() is run on the region's real LAS (or the cf_* of X4 are injected with --cf-from-x4 / when no LAS
is given), then score() and features(). Compared with X4 (cnx_* → NaN by design): per-feature max |diff|, NaN
pattern, pool mask; the score vs the same final model applied to the X4 row (implementation parity) and vs the
outer OOF score (nested2 --tag w5cf / w5cfnc; different models by design → Spearman / mean |diff|).

  W5_DATA=/workspace/koz3/w1/cf/w5d python tools/w5cf_infer_parity.py --infer-dir DIR \
      --region 016_Хога:/path/016.las:/path/UTM.json --region 158_Курганы_29 --oof work/nested_w5cfnc.npz --json-out p.json
"""
import argparse, json, os, pickle, resource, sys, time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from expds import w5cf_infer as W  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--infer-dir', required=True); ap.add_argument('--region', action='append', required=True,
                                                             help='NAME[:LAS[,LAS...]:UTM.json]')
ap.add_argument('--oof', default=None); ap.add_argument('--cf-from-x4', action='store_true')
ap.add_argument('--tol', type=float, default=1e-5); ap.add_argument('--json-out'); a = ap.parse_args()
D = Path(os.environ.get('W5_DATA', '/workspace/koz3/w1/cf/w5d'))
T = pickle.load(open(D / 'work/table.pkl', 'rb')); reg_all = np.asarray(T['reg']).astype(str)
X4 = np.load(D / 'work/X4.npy', mmap_mode='r'); names = json.load(open(D / 'work/X4_names.json'))
pool_tr = np.load(D / 'work/pool.npy'); oof = np.load(a.oof)['outer'] if a.oof else None
cfg = {'w5cf': {'infer_dir': a.infer_dir, 'cf_deadline_s': 1e9}}
art = W.load_artifacts(cfg); assert art['names'] == names
spec = {}
for r in a.region:
    p = r.split(':'); spec[p[0]] = (p[1].split(',') if len(p) > 1 and p[1] else [], Path(p[2]) if len(p) > 2 else None)

import ijson, shapely.geometry  # noqa: E402
want = set(spec); objs = {k: [] for k in want}; idx = {k: [] for k in want}
t = time.time()
with open(D / 'stack_arch.geojson', 'rb') as fh:
    for i, f in enumerate(ijson.items(fh, 'features.item', use_float=True)):
        p = f['properties']; rn = str(p.get('region_name', 'default_source'))
        if rn in want:
            o = {k: p.get(k) for k in ('class_name', 'confidence', 'markup_type', 'threshold_group', 'region_name')}
            o['geom'] = shapely.geometry.shape(f['geometry']); objs[rn].append(o); idx[rn].append(i)
print(f'stream {time.time()-t:.0f}s', {k: len(v) for k, v in objs.items()}, flush=True)

cfi = [names.index(n) for n in names if n.startswith('cf_')]; cnx = [i for i, n in enumerate(names) if n.startswith('cnx_')]
res = {'infer_dir': a.infer_dir, 'variant': art['meta']['variant'], 'tol': a.tol, 'regions': {}}
for rn, feats in objs.items():
    ii = np.asarray(idx[rn]); assert (reg_all[ii] == rn).all()
    W.reset(); las, utm = spec[rn]
    reg = SimpleNamespace(name=rn, las=[Path(x) for x in las], utm_json=utm)
    li = [f for f in feats if f.get('markup_type') == 'li' and (f.get('threshold_group') or 'li') in
          ('li', 'li_dev', 'li_mosaic', 'li_coarse', 'li_fort', 'li_mask', 'gor_phys')]
    t = time.time()
    if las and not a.cf_from_x4:
        W.region_features(reg, li, cfg)
    else:
        if X4[ii[0], names.index('has_las')] == 1:
            W._STATE['las_regions'].add(rn)
        for f, j in zip(feats, ii):
            v = np.asarray(X4[j, cfi], np.float32)
            if np.isfinite(v).any() and f.get('markup_type') == 'li':
                f[W.KEY] = v
    t_cf = time.time() - t; t = time.time()
    s = W.score(feats, cfg); t_sc = time.time() - t
    tb = W.table(feats, art); Xi = W.features(tb, art); assert (tb['rows'] == np.arange(len(feats))).all()
    Xr = np.array(X4[ii], np.float64); Xr[:, cnx] = np.nan
    li_m = np.array([f.get('markup_type') == 'li' for f in feats])
    per = {}
    for k, n in enumerate(names):
        x, y = Xi[:, k].astype(np.float64), Xr[:, k]
        if n.startswith('cf_'):    # train: cf_* on pool rows only; inference: li objects only (W4 passes li to region_features)
            x, y = x[li_m & pool_tr[ii]], y[li_m & pool_tr[ii]]
        nm = np.isnan(x) != np.isnan(y); fin = np.isfinite(x) & np.isfinite(y)
        d = np.abs(x[fin] - y[fin]); both_inf = np.isinf(x) & np.isinf(y) & (np.sign(x) == np.sign(y))
        per[n] = dict(nan_mismatch=int(nm.sum()), inf_mismatch=int((np.isinf(x) != np.isinf(y)).sum() - 0),
                      max_abs=float(d.max()) if len(d) else 0.0, n_over_tol=int((d > a.tol).sum()))
    pool_i = W.in_pool(tb['grp'], tb['cls'], tb['conf'], art)
    sc = np.array([np.nan if v is None else v for v in s])
    fam = np.where(tb['cls'] <= 1, 0, tb['cls'])
    ref = np.full(len(feats), np.nan)
    Xr32 = Xr.astype(np.float32)
    for f_, m in art['models'].items():
        jj = np.flatnonzero(pool_i & (fam == f_))
        if len(jj):
            ref[jj] = m.predict(Xr32[jj])
    pm = np.isfinite(sc); dd = np.abs(sc[pm] - ref[pm])
    out = dict(rows=len(feats), li=int(li_m.sum()), las=[str(x) for x in las], cf_from_x4=bool(a.cf_from_x4 or not las),
               cf_rows=int(np.isfinite(tb['cf'][:, 0]).sum()), sec_region_features=round(t_cf, 1), sec_score=round(t_sc, 1),
               maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024,
               pool_equal=bool((pool_i == pool_tr[ii]).all()), pool_mismatch=int((pool_i != pool_tr[ii]).sum()), pool_n=int(pool_i.sum()), scored=int(pm.sum()),
               score_vs_model_on_X4_max_abs=float(dd.max()) if len(dd) else 0.0,
               score_vs_model_on_X4_n_over_tol=int((dd > a.tol).sum()),
               features_max_abs=max(v['max_abs'] for v in per.values()),
               features_nan_mismatch=sum(v['nan_mismatch'] for v in per.values()),
               features_over_tol={k: v for k, v in per.items() if v['n_over_tol'] or v['nan_mismatch'] or v['inf_mismatch']})
    lm = pm & li_m; dl = np.abs(sc[lm] - ref[lm])
    out['li_score_vs_model_on_X4_max_abs'] = float(dl.max()) if len(dl) else 0.0
    out['li_score_vs_model_on_X4_n_over_tol'] = int((dl > a.tol).sum())
    # exact mode: train gq (stable rank among ties) and train pool mask injected → pure implementation parity
    g0, p0 = W.gq_of, W.in_pool
    W.gq_of = lambda g, c, cf_, art_, _q=Xr[:, names.index('gq')]: _q.copy()
    W.in_pool = lambda g, c, cf_, art_, margin=0.0, _p=pool_tr[ii]: _p.copy()
    try:
        s2 = np.array([np.nan if v is None else v for v in W.score(feats, cfg)]); X2 = W.features(W.table(feats, art), art)
    finally:
        W.gq_of, W.in_pool = g0, p0
    cmp = [k for k, n in enumerate(names) if not n.startswith(('cnx_', 'cf_'))]
    a2, b2 = X2[:, cmp].astype(np.float64), Xr[:, cmp]
    fin = np.isfinite(a2) & np.isfinite(b2)
    pm2 = np.isfinite(s2); ref2 = np.full(len(feats), np.nan)
    for f_, m in art['models'].items():
        jj = np.flatnonzero(pool_tr[ii] & (fam == f_))
        if len(jj):
            ref2[jj] = m.predict(Xr32[jj])
    d2 = np.abs(s2[pm2] - ref2[pm2]); d2l = np.abs(s2[pm2 & li_m] - ref2[pm2 & li_m])
    out['exact_mode'] = dict(features_max_abs=float(np.abs(a2[fin] - b2[fin]).max()) if fin.any() else 0.0,
                             features_nonfinite_mismatch=int((np.isfinite(a2) != np.isfinite(b2)).sum()),
                             score_max_abs_all=float(d2.max()) if len(d2) else 0.0, score_max_abs_li=float(d2l.max()) if len(d2l) else 0.0)
    if oof is not None:
        from scipy.stats import spearmanr
        o = oof[ii]; m2 = pm & (o >= 0)
        out['vs_oof'] = dict(n=int(m2.sum()), spearman=float(spearmanr(sc[m2], o[m2])[0]) if m2.sum() > 2 else None,
                             mean_abs=float(np.abs(sc[m2] - o[m2]).mean()) if m2.any() else None,
                             spearman_li=float(spearmanr(sc[m2 & li_m], o[m2 & li_m])[0]) if (m2 & li_m).sum() > 2 else None)
    res['regions'][rn] = out
    print(rn, json.dumps(out, ensure_ascii=False), flush=True)
if a.json_out:
    Path(a.json_out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
