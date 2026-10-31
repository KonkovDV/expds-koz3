"""bot-5. DINOv2-S: ТОЛЬКО замер наличия сигнала, предобученные веса, в сдачу не идёт.
Подвыборка кандидатов bot-7 с виньеткой: все позитивы + до 300 негативов на участок (seed 0).
Сравнение на одной и той же подвыборке при одинаковых фолдах (honest-10 fold кандидата, leave-fold-out):
  (a) логистическая регрессия на DINO (768),
  (b) LightGBM на ручных признаках (64),
  (c) готовый OOF p c7g.
Метрики: pooled AUC и средний AUC внутри участка."""
import glob, sys, os, json, time, numpy as np, torch
import torch.nn.functional as Fn
from scipy.stats import rankdata
torch.set_num_threads(2)
OD = '/workspace/koz3_shared/bot-5/analysis/kurg_ae'
def auc(s, t):
    t = np.asarray(t, bool); s = np.asarray(s, float)
    if t.all() or (~t).all(): return float('nan')
    r = rankdata(s); return float((r[t].sum() - t.sum() * (t.sum() + 1) / 2) / (t.sum() * (~t).sum()))
D = dict(np.load(f'{OD}/diag_c7g.npz')); P = np.load(f'{OD}/p_oof_fam10_c7g.npy')
C7 = dict(np.load('/workspace/koz3_shared/bot-5/tmp/kurg_ae/cand7.npz')); f7 = dict(zip(C7['cand_id'].tolist(), C7['fold'].tolist()))
pos = {int(c): k for k, c in enumerate(D['SID']) if D['SRC'][k] == 2}
rng = np.random.default_rng(0)
WIN, HF, ROW = [], [], []
for f in sorted(glob.glob('/workspace/koz3_shared/bot-5/tmp/kurg_ae/feat7/*.npz')):
    z = dict(np.load(f)); wi = np.flatnonzero(z['widx'])
    if not len(wi): continue
    ks = np.array([pos[int(c)] for c in z['rec'][wi, 3]])
    ok = ~D['IGN'][ks]; y = D['LAB'][ks] > 0
    pi = np.flatnonzero(ok & y); ni = np.flatnonzero(ok & ~y)
    ni = rng.choice(ni, min(len(ni), 300 * 1), replace=False) if len(ni) else ni
    sel = np.concatenate([pi, ni])
    WIN.append(z['wins'][sel]); HF.append(z['F'][wi[sel]].astype(np.float32)); ROW.append(ks[sel])
WIN = np.concatenate(WIN); HF = np.concatenate(HF); ROW = np.concatenate(ROW)
y = D['LAB'][ROW] > 0; reg = D['CREG'][ROW]; fold = np.array([f7[int(D['SID'][k])] for k in ROW])
print('subset', len(ROW), 'pos', int(y.sum()), 'regions', len(set(reg)), 'folds', sorted(set(fold.tolist())), flush=True)
MD = glob.glob('/workspace/koz3_shared/bot-5/tmp/hf/hub/models--facebook--dinov2-small/snapshots/*/')[0]
from transformers import Dinov2Model
m = Dinov2Model.from_pretrained(MD).eval()
MEAN = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]; STD = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
E = np.zeros((len(WIN), 768), np.float32); t0 = time.time()
CACHE = os.path.exists(f'{OD}/dino_sub_rows.npy') and np.array_equal(np.load(f'{OD}/dino_sub_rows.npy'), ROW)
if CACHE: E = np.load(f'{OD}/dino_sub_emb.npy').astype(np.float32); print('emb from cache', flush=True)
for s in (range(0) if CACHE else range(0, len(WIN), 64)):
    x = torch.from_numpy(WIN[s:s + 64]).float()[:, None] / 255.0
    x = Fn.interpolate(x, size=(112, 112), mode='bilinear', align_corners=False).repeat(1, 3, 1, 1)
    with torch.no_grad(): o = m(pixel_values=(x - MEAN) / STD).last_hidden_state
    E[s:s + 64] = torch.cat([o[:, 0], o[:, 1:].mean(1)], 1).numpy()
print('dino done', f'{time.time()-t0:.0f}s', flush=True)
np.save(f'{OD}/dino_sub_emb.npy', E.astype(np.float16)); np.save(f'{OD}/dino_sub_rows.npy', ROW)
import lightgbm as lgb
def fitp(Xa, ya, Xb, cs):
    prm = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_data_in_leaf=20, bagging_fraction=0.8, bagging_freq=1,
               feature_fraction=cs, is_unbalance=True, num_threads=2, verbose=-1, seed=0)
    return lgb.train(prm, lgb.Dataset(Xa, ya.astype(float)), num_boost_round=300).predict(Xb)
pd_, ph, pg = np.full(len(y), np.nan), np.full(len(y), np.nan), np.full(len(y), np.nan)
for fo in sorted(set(fold.tolist())):
    te = fold == fo; tr = ~te
    if y[tr].sum() < 5: continue
    mu, sd = E[tr].mean(0), E[tr].std(0) + 1e-6; Xtr = (E[tr] - mu) / sd; Xte = (E[te] - mu) / sd
    wt = np.where(y[tr], 0.5 / y[tr].mean(), 0.5 / (1 - y[tr].mean())); t = np.where(y[tr], 1.0, -1.0)
    A_ = (Xtr * wt[:, None]).T @ Xtr + 1000.0 * np.eye(Xtr.shape[1]); w = np.linalg.solve(A_, (Xtr * wt[:, None]).T @ t)
    pd_[te] = Xte @ w  # взвешенная ridge-классификация (sklearn нет в venv), lambda=1000 фиксирована заранее
    pg[te] = fitp(E[tr], y[tr], E[te], 0.3)
    ph[te] = fitp(HF[tr], y[tr], HF[te], 0.7)
pc = P[ROW, 1] + P[ROW, 2]
def both(s):
    ok = ~np.isnan(s); rs = sorted(set(reg[y & ok]))
    pr = {r: auc(s[ok & (reg == r)], y[ok & (reg == r)]) for r in rs}
    return dict(pooled=auc(s[ok], y[ok]), within_mean=float(np.nanmean(list(pr.values()))), by_reg=pr, n=int(ok.sum()))
out = dict(n=len(y), n_pos=int(y.sum()), dino_ridge=both(pd_), dino_lgbm=both(pg), hand_lgbm_same_subset=both(ph), c7g_oof_p=both(pc),
           note='Только замер. DINOv2-S предобученный, в сдачу не идёт без письма организаторов. Фолды = honest-10 fold кандидата bot-7.')
json.dump(out, open(f'{OD}/dino_sub.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != 'by_reg'}) for k, v in out.items()}, ensure_ascii=False))
