"""bot-5. ТОЛЬКО ЧЕСТНЫЙ ЗАМЕР. Плотный локализатор городищ на спутнике, шаг 2: линейная проба на токенах dense_feats.py,
сглаживание, пики. Чистый numpy/scipy/shapely — одинаково работает на боксе и на Kaggle. Метки и фолды — из
gor_dense_inputs.json (export_inputs.py на боксе), семьи для внутренних фолдов — expds.naming.site_family (репо).
Всё задано ЗАРАНЕЕ, не подбиралось: PCA 64 (на токенах обучения внешнего фолда, без меток, общая для внутренних фитов),
L2-логрег C=1, веса классов сбалансированы, не больше 20 000 негативов на фит (случайно, seed 0); трудные негативы — верхний
1 % негативов обучения по p первой пробы, вес ×5, переобучение; сглаживание гаусс σ=1 токен (норм. свёртка по валидным);
K=2 пика на участок с подавлением 150 м.
Метка токена: pos — центр внутри эталона городища sp_or; ignore — в 30 м от городища sp_or вне его, внутри или в 30 м от
городища другой разметки, доля валидных пикселей < 0,5. Вариант 'pu' (основной): токены участков без городищ sp_or — ignore
(«нет файла разметки» ≠ «нет городища», NOLIDAR §12; OSINT_2026-10-29 §2 п.7); 'pu_nohn' — без трудных негативов;
'all' — такие токены neg; 'pu_edge' — как 'pu', но негативы дальше D_EDGE=300 м (Чебышёв) от центра кадра SpOr тоже ignore
(приор кадра: размечен только памятник в центре кадра, у края похожие структуры — не надёжный негатив; аудит
bot-10/work11/wt_audit docs/AUDIT_CLAUDE_2026-10-29.md:114 (коммит 6e26062); 300 м задано заранее = q90 размера городища 307 м, STATUS (E));
трудные негативы при этом берутся только из центральной части кадра.
Приор центра на выводе ('<v>+cp'): сглаженная карта × exp(-d²/2s²), d — расстояние до центра кадра, s из сетки CP_GRID
выбирается ВНУТРИ внешнего фолда по доле попаданий top-1 пика во внутреннем OOF обучающих участков (ничья → больший s);
отложенное семейство в выборе s не участвует. Варианты без '+cp' — те же карты без приора (для сравнения).
Данные этапа «Сканирование» ЗАПРЕЩЕНЫ организаторами и не используются нигде (никакого хука).
Строгая вложенность: пики обучающих участков — из внутреннего OOF (5 семейных фолдов обучения), отложенных — модель на всём
обучении. Выход: peaks{TAG}.json: layouts -> [по фолду: {held, variant -> {reg: [[p, x, y], ...]}}] + token AUC + попадание
top-1 точки в эталон sp_or отложенных участков.
Env: DOUT (папка с dense{TAG}.npz, files{TAG}.json, gor_dense_inputs.json), TAG, VARIANTS (pu,pu_edge,all; есть и pu_nohn), LAYOUTS, REPO.
Чекпойнты (peaks_ckpt.py): после КАЖДОГО фолда — CKPT_DIR (по умолч. DOUT/ckpt{TAG})/<layout>/f###.json+.npz (атомарно).
RESUME=1 (по умолч.) — готовые фолды не пересчитываются; FOLDS='0-29' или '0,3,7' — считать только эти фолды (нужен один LAYOUTS).
Раскрой попадает в peaks{TAG}.json, когда в CKPT_DIR есть все его фолды; иначе склейка позже: peaks_ckpt.py CKPT_DIR OUT.json
(результат совпадает с одним процессом: фиты детерминированы — rng(0) на каждый вызов, — а счётчики hn_log/cp_s_chosen считаются по фолдам)."""
import os, sys, json, math, collections, time
from pathlib import Path
import numpy as np, shapely
from scipy.ndimage import gaussian_filter
REPO = os.environ.get('REPO', str(Path(__file__).resolve().parents[2])); sys.path.insert(0, REPO)
from expds.metric import family_folds
from expds.naming import site_family
t0 = time.time()
def log(*a): print(f'[{time.time()-t0:.0f}s]', *a, flush=True)
PC, CREG, SIG, K_PK, SUPR, HN_Q, HN_W, BUF = 64, 1.0, 1.0, 2, 150.0, 0.99, 5.0, 30.0
NEG_CAP = int(os.environ.get('NEG_CAP', '20000'))
D_EDGE = 300.0; CP_GRID = [150.0, 250.0, 400.0, 700.0, float('inf')]
class Probe:
    def __init__(self, dout, tag):
        dout = Path(dout); Z = np.load(dout / f'dense{tag}.npz'); self.FI = json.load(open(dout / f'files{tag}.json', encoding='utf-8'))
        J = json.load(open(dout / 'gor_dense_inputs.json', encoding='utf-8')); self.J = J
        self.DF = Z['F']; self.fid = Z['fid']; self.row = Z['row'].astype(int); self.col = Z['col'].astype(int)
        self.ux = Z['ux']; self.uy = Z['uy']; self.vf = Z['vf'].astype(np.float32)
        self.FREG = np.array([f['reg'] for f in self.FI['files']]); self.REG = self.FREG[self.fid]
        log('tokens', len(self.DF), 'dim', self.DF.shape[1], 'files', len(self.FI['files']), 'regions', len(set(self.REG)), self.FI['model'], self.FI['gsds'])
        self.HF = {r: [(shapely.from_wkt(w), s) for w, s in v] for r, v in J['hf'].items()}
        self.SPR = {r for r, v in self.HF.items() if any(s for _, s in v)}
        lab = np.zeros(len(self.DF), np.int8)
        for r in sorted(set(self.REG)):
            ii = np.flatnonzero(self.REG == r)
            for g, s in self.HF.get(r, []):
                g = g if g.geom_type != 'Point' else g.buffer(5)
                near = shapely.contains_xy(g.buffer(BUF), self.ux[ii], self.uy[ii])
                if s:
                    ins = shapely.contains_xy(g, self.ux[ii], self.uy[ii]); lab[ii[ins]] = 1
                    lab[ii[near & ~ins & (lab[ii] != 1)]] = -1
                else: lab[ii[near & (lab[ii] != 1)]] = -1
        lab[(self.vf < 0.5) & (lab == 0)] = -1; self.lab = lab
        self.NEGSCENE = ~np.isin(self.REG, list(self.SPR))
        cx = np.zeros(len(self.FI['files'])); cy = np.zeros(len(cx))   # центр кадра = середина охвата токенов файла
        for k in range(len(cx)):
            ii = np.flatnonzero(self.fid == k)
            if len(ii): cx[k] = (self.ux[ii].min() + self.ux[ii].max()) / 2; cy[k] = (self.uy[ii].min() + self.uy[ii].max()) / 2
        self.dcx = self.ux - cx[self.fid]; self.dcy = self.uy - cy[self.fid]
        self.DCEN = np.hypot(self.dcx, self.dcy); self.EDGE = np.maximum(np.abs(self.dcx), np.abs(self.dcy)) > D_EDGE
        pd_ = self.DCEN[self.lab == 1]
        log('frame prior: pos tokens dist to frame centre m q50/q90', np.round(np.quantile(pd_, [0.5, 0.9]), 1).tolist(), 'edge share of neg', round(float(self.EDGE[self.lab == 0].mean()), 3))
        log('token labels', dict(collections.Counter(lab.tolist())), 'regions with pos tokens', len(set(self.REG[lab == 1])))
        self.FIDX = {k: np.flatnonzero(self.fid == k) for k in range(len(self.FI['files']))}
        self.ALL = np.arange(len(self.DF)); self.hn_log = collections.Counter(); self._cache = {}
        self.RLIST = sorted(set(self.REG.tolist())); self.RC = np.searchsorted(np.array(self.RLIST), self.REG)
    def groups(self, idx):   # {регион: позиции в idx} в порядке возрастания — то же, что np.flatnonzero(self.REG[idx] == r)
        c = self.RC[idx]; o = np.argsort(c, kind='stable'); u, st = np.unique(c[o], return_index=True)
        return {self.RLIST[k]: o[a:b] for k, a, b in zip(u, st, list(st[1:]) + [len(o)])}
    def set_proj(self, idx):
        DF = self.DF; n = 0; s1 = np.zeros(DF.shape[1]); s2 = np.zeros(DF.shape[1])
        for a in range(0, len(idx), 20000):
            x = DF[idx[a:a + 20000]].astype(np.float64); s1 += x.sum(0); s2 += (x ** 2).sum(0); n += len(x)
        mu = s1 / n; sd = np.sqrt(np.maximum(s2 / n - mu ** 2, 0)) + 1e-6
        sub = idx[np.random.default_rng(0).permutation(len(idx))[:20000]]
        _, _, Vt = np.linalg.svd(((DF[sub].astype(np.float64) - mu) / sd).astype(np.float32), full_matrices=False)
        W = (Vt[:PC].T / sd[:, None]).astype(np.float32); m32 = mu.astype(np.float32)
        Q = np.zeros((len(DF), PC), np.float32)
        for a in range(0, len(DF), 20000): Q[a:a + 20000] = (DF[a:a + 20000].astype(np.float32) - m32) @ W
        Q /= Q[idx].std(0) + 1e-6; self.Q = np.c_[np.ones(len(Q), np.float32), Q]
    def _fit(self, idx, yt, ew=None):
        if len(idx) == 0 or yt.min() == yt.max(): return ('const', float(yt.mean()) if len(idx) else 0.0)
        Q = self.Q[idx].astype(np.float64); sw = np.where(yt == 1, 0.5 / yt.mean(), 0.5 / (1 - yt.mean()))
        if ew is not None: sw = sw * ew
        w = np.zeros(Q.shape[1]); lam = np.r_[0, np.ones(Q.shape[1] - 1)] / CREG
        for _ in range(50):
            p = 1 / (1 + np.exp(-Q @ w)); gr = Q.T @ (sw * (p - yt)) + lam * w
            H = (Q * (sw * p * (1 - p))[:, None]).T @ Q + np.diag(lam) + 1e-9 * np.eye(len(w)); st = np.linalg.solve(H, gr); w -= st
            if np.abs(st).max() < 1e-8: break
        return ('lr', w)
    def pred(self, mdl, idx):
        if mdl[0] == 'const': return np.full(len(idx), mdl[1])
        return 1 / (1 + np.exp(-self.Q[idx].astype(np.float64) @ mdl[1]))
    def fit(self, idx, var):
        lab_ = self.lab[idx].copy()
        if var.startswith('pu'): lab_[self.NEGSCENE[idx] & (lab_ == 0)] = -1
        if var == 'pu_edge': lab_[self.EDGE[idx] & (lab_ == 0)] = -1
        li = idx[lab_ >= 0]; yt = (lab_[lab_ >= 0] == 1).astype(float)
        ng = np.flatnonzero(yt == 0)
        if len(ng) > NEG_CAP:
            keep = np.sort(np.r_[np.flatnonzero(yt == 1), np.random.default_rng(0).choice(ng, NEG_CAP, replace=False)]); li, yt = li[keep], yt[keep]
        m1 = self._fit(li, yt)
        if var.endswith('nohn') or m1[0] == 'const': return m1
        p = self.pred(m1, li); neg = np.flatnonzero(yt == 0)
        if len(neg) < 100: return m1
        top = neg[p[neg] >= np.quantile(p[neg], HN_Q)]; ew = np.ones(len(li)); ew[top] = HN_W
        self.hn_log['fits'] += 1; self.hn_log['hard_neg'] += len(top); self.hn_log['neg'] += len(neg)
        return self._fit(li, yt, ew)
    def peaks(self, p_tok, idx, cp=float('inf')):
        pos = np.full(len(self.DF), -1); pos[idx] = np.arange(len(idx)); cand = []
        for f_ in sorted(set(self.fid[idx].tolist())):
            ii = self.FIDX[f_]; H, W = self.FI['files'][f_]['H'], self.FI['files'][f_]['W']
            pm = np.zeros((H, W)); vm = np.zeros((H, W)); km = np.full((H, W), -1)
            pm[self.row[ii], self.col[ii]] = p_tok[pos[ii]]; vm[self.row[ii], self.col[ii]] = self.vf[ii] >= 0.5; km[self.row[ii], self.col[ii]] = ii
            num = gaussian_filter(pm * vm, SIG, mode='constant'); den = gaussian_filter(vm, SIG, mode='constant')
            sm = np.where(vm > 0, num / np.maximum(den, 1e-6), -1.0)
            if np.isfinite(cp):
                dm = np.zeros((H, W)); dm[self.row[ii], self.col[ii]] = self.DCEN[ii]
                sm = np.where(sm >= 0, sm * np.exp(-0.5 * (dm / cp) ** 2), sm)
            for q in np.argsort(-sm.ravel())[:200]:
                a, b = divmod(int(q), W); k = km[a, b]
                if k >= 0 and sm[a, b] >= 0: cand.append((float(sm[a, b]), float(self.ux[k]), float(self.uy[k])))
        cand.sort(key=lambda t: -t[0]); out = []
        for c in cand:
            if all(math.hypot(c[1] - o[1], c[2] - o[2]) > SUPR for o in out): out.append(c)
            if len(out) == K_PK: break
        return out
    def in_sp_or(self, r, x, y):
        return any(shapely.contains_xy(g if g.geom_type != 'Point' else g.buffer(5), x, y) for g, s in self.HF.get(r, []) if s)
    def run_fold(self, folds, fi, variants):
        """Один внешний фолд → (fr, hit, tok, hn). tok = {v: (idx отложенных, p)}; hn — счётчик фитов этого фолда (с учётом кэша)."""
        every = set().union(*folds); held = folds[fi]; train = every - held
        dtr = self.ALL[np.isin(self.REG, list(train))]; dhe = self.ALL[np.isin(self.REG, list(held))]
        fr = {'held': sorted(held)}; hit = {w: [0, 0] for v in variants for w in (v, v + '+cp')}; tok = {v: (dhe, np.zeros(0)) for v in variants}
        key = (frozenset(self.REG[dtr].tolist()), tuple(variants))
        if not len(dhe) and key in self._cache:   # отложенный фолд без сцен SpOr: обучение то же, что уже посчитано
            c, hn = self._cache[key]; fr.update(c); return fr, hit, tok, collections.Counter(hn)
        hn0 = collections.Counter(self.hn_log)
        if len(dtr):
            self.set_proj(dtr); dfo = family_folds(sorted(set(self.REG[dtr])), site_family, 5)
            gtr = self.groups(dtr); ghe = self.groups(dhe) if len(dhe) else {}
            for v in variants:
                pin = np.zeros(len(dtr))
                for h in dfo:
                    te = np.isin(self.REG[dtr], list(h))
                    if te.any(): pin[te] = self.pred(self.fit(dtr[~te], v), dtr[te])
                md = self.fit(dtr, v); phd = self.pred(md, dhe) if len(dhe) else np.zeros(0); tok[v] = (dhe, phd)
                # выбор s приора центра — только по внутреннему OOF обучающих участков с городищем sp_or
                trs = [r for r in sorted(gtr) if r in self.SPR]; best = (-1, 0.0)
                for cp in CP_GRID:
                    h = 0
                    for r in trs:
                        mm = gtr[r]; pk = self.peaks(pin[mm], dtr[mm], cp)
                        h += int(bool(pk) and self.in_sp_or(r, pk[0][1], pk[0][2]))
                    if h > best[0] or (h == best[0] and cp > best[1]): best = (h, cp)
                cps = best[1]
                for vv, cp in ((v, float('inf')), (v + '+cp', cps)):
                    pk_ = {}
                    for idx_set, pp, part, g in ((dtr, pin, 'tr', gtr), (dhe, phd, 'he', ghe)):
                        for r in sorted(g):
                            mm = g[r]; pk = self.peaks(pp[mm], idx_set[mm], cp); pk_[r] = [list(t) for t in pk]
                            if part == 'he' and r in self.SPR and pk: hit[vv][0] += int(self.in_sp_or(r, pk[0][1], pk[0][2])); hit[vv][1] += 1
                    fr[vv] = pk_
                fr.setdefault('cp_s', {})[v] = cps
        hn = collections.Counter(self.hn_log); hn.subtract(hn0); hn = +hn
        if not len(dhe): self._cache[key] = ({k: v for k, v in fr.items() if k != 'held'}, dict(hn))
        return fr, hit, tok, hn
def auc(s, t):
    s = np.asarray(s, float); t = np.asarray(t, bool)
    if t.all() or not t.any(): return None
    o = np.argsort(s, kind='mergesort'); rk = np.empty(len(s)); rk[o] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True); sums = np.bincount(inv, rk); rk = (sums / cnt)[inv]
    nP = t.sum(); return float((rk[t].sum() - nP * (nP + 1) / 2) / (nP * (len(s) - nP)))
if __name__ == '__main__':
    import peaks_ckpt as PCK
    DOUT = os.environ.get('DOUT', '/workspace/koz3_shared/bot-5/analysis/gor_dense'); TAG = os.environ.get('TAG', '')
    VAR = [v for v in os.environ.get('VARIANTS', 'pu,pu_edge,all').split(',') if v]
    P = Probe(DOUT, TAG); LAY = [l for l in os.environ.get('LAYOUTS', ','.join(P.J['layouts'])).split(',') if l]
    CK = Path(os.environ.get('CKPT_DIR', str(Path(DOUT) / f'ckpt{TAG}'))); RESUME = os.environ.get('RESUME', '1') == '1'
    SEL = PCK.parse_folds(os.environ.get('FOLDS', ''))
    if SEL is not None: assert len(LAY) == 1, 'FOLDS требует один LAYOUTS'
    cen = [int(P.in_sp_or(r, float(P.ux[P.REG == r].mean()), float(P.uy[P.REG == r].mean()))) for r in sorted(P.SPR & set(P.REG.tolist()))]
    META = dict(model=P.FI['model'], gsds=P.FI['gsds'], tag=TAG, n_tokens=int(len(P.DF)), token_labels={str(k): int(v) for k, v in collections.Counter(P.lab.tolist()).items()},
                centre_hit=[int(sum(cen)), len(cen)], variants=VAR, nfolds={nm: len(P.J['layouts'][nm]) for nm in P.J['layouts']})
    PCK.write_meta(CK, META)
    for nm in LAY:
        folds = [set(f) for f in P.J['layouts'][nm]]; todo = range(len(folds)) if SEL is None else [i for i in SEL if i < len(folds)]
        for fi in todo:
            if RESUME and PCK.has_fold(CK, nm, fi): continue
            fr, hit, tok, hn = P.run_fold(folds, fi, VAR)
            PCK.save_fold(CK, nm, fi, fr, hit, tok, hn, P.lab, P.NEGSCENE); log(nm, 'fold', fi, '/', len(folds), 'saved')
        if PCK.complete(CK, nm, len(folds)): log(nm, PCK.layout_stats(CK, nm, len(folds), VAR), 'centre', META['centre_hit'])
        else: log(nm, 'partial: folds in ckpt', PCK.n_done(CK, nm), '/', len(folds))
        R = PCK.assemble(CK); json.dump(R, open(Path(DOUT) / f'peaks{TAG}.json', 'w'), ensure_ascii=False)
    import resource; log('done; peak RSS MB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
