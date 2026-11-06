""". Экспорт плотной головы gor_dense (вариант 'all', гиперпараметры dense_probe — заданы заранее, не подбирались)
в формате gor_v3_head (mu, sd, W, qsd, w, feat_dim) для expds/gor_dense.py.
FOLD не задан — обучение на ВСЕХ токенах трейна (голова сдачи) + s приора центра по заранее заданному правилу
dense_probe (внутренний OOF 5 семейных фолдов, top-1 попадание в sp_or, ничья → больший s).
FOLD=i (LAYOUT, по умолч. fam10) — голова i-го внешнего фолда (обучение = всё, кроме отложенного семейства) для теста
воспроизведения peaks.json; s берётся из peaks.json (cp_s фолда), не пересчитывается.
Env: DOUT (analysis/gor_dense), OUT (npz), FOLD, LAYOUT. Печатает md5 и JSON-сводку."""
import os, sys, json, hashlib
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import dense_probe as DP
DOUT = os.environ.get('DOUT', '/workspace/koz3_shared/bot-5/analysis/gor_dense'); OUT = Path(os.environ['OUT'])
FOLD = os.environ.get('FOLD'); LAYOUT = os.environ.get('LAYOUT', 'fam10'); VAR = 'all'
P = DP.Probe(DOUT, '')
def proj(idx):
    """= Probe.set_proj (тот же код), но возвращает mu, W, qsd; P.Q заполняется так же."""
    DF = P.DF; n = 0; s1 = np.zeros(DF.shape[1]); s2 = np.zeros(DF.shape[1])
    for a in range(0, len(idx), 20000):
        x = DF[idx[a:a + 20000]].astype(np.float64); s1 += x.sum(0); s2 += (x ** 2).sum(0); n += len(x)
    mu = s1 / n; sd = np.sqrt(np.maximum(s2 / n - mu ** 2, 0)) + 1e-6
    sub = idx[np.random.default_rng(0).permutation(len(idx))[:20000]]
    _, _, Vt = np.linalg.svd(((DF[sub].astype(np.float64) - mu) / sd).astype(np.float32), full_matrices=False)
    W = (Vt[:DP.PC].T / sd[:, None]).astype(np.float32); m32 = mu.astype(np.float32)
    Q = np.zeros((len(DF), DP.PC), np.float32)
    for a in range(0, len(DF), 20000): Q[a:a + 20000] = (DF[a:a + 20000].astype(np.float32) - m32) @ W
    qsd = Q[idx].std(0) + 1e-6; Q /= qsd; P.Q = np.c_[np.ones(len(Q), np.float32), Q]
    return m32, W, qsd.astype(np.float32)
summ = dict(variant=VAR, dout=DOUT)
if FOLD is None:
    dtr = P.ALL; summ['train'] = 'all'
else:
    folds = [set(f) for f in P.J['layouts'][LAYOUT]]; held = folds[int(FOLD)]; train = set().union(*folds) - held
    dtr = P.ALL[np.isin(P.REG, list(train))]; summ.update(layout=LAYOUT, fold=int(FOLD), held=sorted(held))
mu, W, qsd = proj(dtr)
md = P.fit(dtr, VAR); assert md[0] == 'lr'
summ.update(n_train_tokens=int(len(dtr)), n_regions=len(set(P.REG[dtr])), hn_log=dict(P.hn_log))
if FOLD is None:   # s по правилу dense_probe.run_fold на всём трейне
    dfo = DP.family_folds(sorted(set(P.REG[dtr])), DP.site_family, 5); gtr = P.groups(dtr); pin = np.zeros(len(dtr))
    for h in dfo:
        te = np.isin(P.REG[dtr], list(h))
        if te.any(): pin[te] = P.pred(P.fit(dtr[~te], VAR), dtr[te])
    proj(dtr)   # те же Q (детерминированно), на случай если fit что-то поменял — не меняет, но явно
    trs = [r for r in sorted(gtr) if r in P.SPR]; best = (-1, 0.0); hits = {}
    for cp in DP.CP_GRID:
        h = 0
        for r in trs:
            mm = gtr[r]; pk = P.peaks(pin[mm], dtr[mm], cp); h += int(bool(pk) and P.in_sp_or(r, pk[0][1], pk[0][2]))
        hits[str(cp)] = h
        if h > best[0] or (h == best[0] and cp > best[1]): best = (h, cp)
    summ.update(centre_s=best[1], inner_oof_top1_hits=hits, n_spr_train=len(trs))
else:
    pkj = json.load(open(Path(DOUT) / 'peaks.json', encoding='utf-8'))
    summ['centre_s'] = pkj['layouts'][LAYOUT][int(FOLD)]['cp_s'][VAR]
OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez(OUT, mu=mu, sd=np.ones_like(mu), W=W, qsd=qsd, w=np.asarray(md[1], np.float64), feat_dim=np.int64(P.DF.shape[1]))
# самопроверка: формула expds.gor_dense.head_p совпадает с Probe.pred на токенах
sys.path.insert(0, str(HERE.parents[1])); from expds.gor_dense import head_p
H = dict(np.load(OUT)); chk = P.ALL[np.random.default_rng(1).permutation(len(P.ALL))[:5000]]
summ['max_abs_diff_headp_vs_probe'] = float(np.abs(head_p(H, P.DF[chk]) - P.pred(md, chk)).max())
summ.update(out=str(OUT), md5=hashlib.md5(OUT.read_bytes()).hexdigest(), bytes=OUT.stat().st_size)
print(json.dumps(summ, ensure_ascii=False))
