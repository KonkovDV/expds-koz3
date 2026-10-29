"""Пункт 2: сдвиг распределения P на выборочных пикселях трейна (pixgbm/feats): fold-модель M_{−F} на F (OOF)
против модели all на F (in-sample) и против M_{−F} на своих обучающих семьях (in-sample). Выход out/calib_shift_px.json."""
import sys, json
sys.path.insert(0, '/workspace/koz3_shared/bot-3/audit/wt/scripts')
import pixgbm as P
import lightgbm as lgb, numpy as np
from pathlib import Path
MF = Path('/workspace/koz3_shared/bot-3/pixgbm/models')
parts = P.load_feats('/workspace/koz3_shared/bot-3/pixgbm/feats')
FAMS = sorted({p['family'] for p in parts})
def auc(s, y):
    o = np.argsort(s, kind='stable'); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else None
def summ(s, y):
    q = np.quantile(s, [0.5, 0.9, 0.99]).tolist()
    return dict(n=int(len(s)), pos=int(y.sum()), q50=q[0], q90=q[1], q99=q[2], ge05=float((s >= 0.5).mean()), ge09=float((s >= 0.9).mean()),
                mean_pos=float(s[y == 1].mean()) if y.sum() else None, mean_neg=float(s[y == 0].mean()), auc=auc(s, y))
out = {}
for name, tgt in (('poly', 'yp'), ('finds', 'yf')):
    mall = lgb.Booster(model_file=str(MF / 'all' / f'{name}.txt'))
    for F in FAMS:
        mf = lgb.Booster(model_file=str(MF / F / f'{name}.txt'))
        for where, sel in (('F', lambda p: p['family'] == F), ('notF', lambda p: p['family'] != F)):
            X, y = [], []
            for p in parts:
                if not sel(p): continue
                if name == 'finds' and p['mt'] != 'maggr': continue
                m = p[tgt] >= 0
                if m.any(): X.append(p['X'][m]); y.append(p[tgt][m])
            if not X: continue
            X = np.concatenate(X); y = np.concatenate(y).astype(int)
            out[f'{name}|{F}|{where}|fold'] = summ(mf.predict(X, num_threads=2), y)
            out[f'{name}|{F}|{where}|all'] = summ(mall.predict(X, num_threads=2), y)
            print(name, F, where, 'fold', json.dumps(out[f'{name}|{F}|{where}|fold']), '\n   all', json.dumps(out[f'{name}|{F}|{where}|all']), flush=True)
            del X, y
json.dump(out, open('/workspace/koz3_shared/bot-3/audit/out/calib_shift_px.json', 'w'), indent=1)
