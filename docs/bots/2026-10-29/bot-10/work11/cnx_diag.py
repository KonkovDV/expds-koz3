"""bot-10 work11 C2, диагностика −0,0105 (in-sample, только объяснение, не приёмка). Курганы, 38 участков фолдов 0–3.
(а) пороги-оракул: per-site (sweep_groups на каждом участке отдельно) против single (одни пороги на все 38 участков, in-sample);
    разрыв per-site − single = цена переноса порога между участками. Для A, R, RC, X.
(б) ранжирование внутри участка без порогов: только объекты группы g (li в A, li_cnx в R/RC), AP и доля TP в top-N (N = эталонов класса на участке).
    Попадания взяты из общих кривых (группы делят эталон: объект, чей эталон забрала группа с большей уверенностью, — не TP) — оговорка.
    python cnx_diag.py > cnx_diag.log  → cnx_diag.json"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/workspace/koz3_shared/bot-10/repo'); sys.path.insert(0, '/workspace/koz3_shared/bot-10/repo/scripts')
from honest_bootstrap import load_curves
from expds import metric as M
W = Path('/workspace/koz3_shared/bot-10/work11')
RF = json.load(open('/workspace/koz3_shared/bot-9/kaggle/convnext_lidar/c3sc_f01_out/region_fold.json', encoding='utf-8'))['region_fold']
import os
FOLDS = tuple(int(x) for x in os.environ.get('CNX_FOLDS', '0,1,2,3').split(','))
COV = sorted(r for r, f in RF.items() if f in FOLDS)
KUR = ('kurgany_povrezhdennye', 'kurgany_tselye')
V = {'A': 'curves_A_s27', 'R': 'curves_cnxR_s27', 'RC': 'curves_cnxRC_s27', 'X': 'curves_cnxX_s27'}
if os.environ.get('CNX_V'): V = json.loads(os.environ['CNX_V'])
F1 = lambda tp, n, g: 2 * tp / (n + g) if n + g else 0.0
out = {'regions': COV, 'oracle': {}, 'rank': {}}
for v, d in V.items():
    cur = load_curves(W / d)
    o = {}
    for cls in KUR:
        c = cur[cls]
        allc = M.curve_where(c, set(COV))
        sg = M.sweep_groups({cls: allc})[cls]
        s_tp = sum(x['tp'] for x in sg['groups'].values()); s_n = sum(x['kept'] for x in sg['groups'].values())
        # single in-sample thresholds → по участкам
        taus = {g: x['threshold'] for g, x in sg['groups'].items()}
        p_tp = p_n = 0; per = {}
        for r in COV:
            cr = M.curve_where(c, {r})
            if cr.n_gt == 0 and cr.conf.size == 0: continue
            if cr.n_gt == 0:
                kept = 0; tp = 0     # оракул участка без эталона: всё выключить
            else:
                x = M.sweep_groups({cls: cr})[cls]
                tp = sum(y['tp'] for y in x['groups'].values()); kept = sum(y['kept'] for y in x['groups'].values())
            th = np.fromiter((taus.get(g, np.inf) for g in cr.pred_group), float, len(cr.pred_group))
            k = cr.conf >= th
            per[r] = {'n_gt': cr.n_gt, 'site_tp': int(tp), 'site_fp': int(kept - tp), 'single_tp': int(cr.hit[k].sum()), 'single_fp': int(k.sum() - cr.hit[k].sum())}
            p_tp += tp; p_n += kept
        o[cls] = {'n_gt': allc.n_gt, 'single_f1': F1(s_tp, s_n, allc.n_gt), 'single_tp': s_tp, 'single_fp': s_n - s_tp,
                  'persite_f1': F1(p_tp, p_n, allc.n_gt), 'persite_tp': p_tp, 'persite_fp': p_n - p_tp, 'single_taus': taus, 'per_region': per}
        print(v, cls, 'n_gt', allc.n_gt, 'single F1 %.4f (TP %d FP %d)' % (o[cls]['single_f1'], s_tp, s_n - s_tp),
              'per-site F1 %.4f (TP %d FP %d)' % (o[cls]['persite_f1'], p_tp, p_n - p_tp), flush=True)
    out['oracle'][v] = o
    g = {'A': 'li', 'R': 'li_cnx', 'RC': 'li_cnx'}.get(v)
    if g:
        rk = {}
        for cls in KUR:
            c = cur[cls]; ga = np.array(c.pred_group); ra = np.array(c.pred_region)
            per = {}; aps = []; top = [0, 0]
            for r in COV:
                n_gt = c.gt_regions.get(r, 0)
                m = (ga == g) & (ra == r)
                h = c.hit[m]   # уже по убыванию уверенности
                if n_gt == 0: per[r] = {'n_gt': 0, 'n_pred': int(m.sum())}; continue
                prec = np.cumsum(h) / np.arange(1, h.size + 1) if h.size else np.zeros(0)
                ap = float((prec * h).sum() / n_gt) if h.size else 0.0
                tN = int(h[:n_gt].sum())
                per[r] = {'n_gt': n_gt, 'n_pred': int(m.sum()), 'tp_all': int(h.sum()), 'ap': round(ap, 4), 'tp_topN': tN, 'topN_share': round(tN / n_gt, 4)}
                aps.append((ap, n_gt)); top[0] += tN; top[1] += n_gt
            rk[cls] = {'mean_ap': float(np.mean([a for a, _ in aps])) if aps else None,
                       'gt_weighted_ap': float(sum(a * n for a, n in aps) / sum(n for _, n in aps)) if aps else None,
                       'topN_share_pooled': top[0] / top[1] if top[1] else None, 'n_sites_with_gt': len(aps), 'per_region': per}
            print(v, g, cls, 'sites', len(aps), 'meanAP %.4f gtwAP %.4f topN %.4f' % (rk[cls]['mean_ap'] or 0, rk[cls]['gt_weighted_ap'] or 0, rk[cls]['topN_share_pooled'] or 0), flush=True)
        out['rank'][v] = rk
json.dump(out, open(W / os.environ.get('CNX_OUT', 'cnx_diag.json'), 'w'), ensure_ascii=False, indent=1)
