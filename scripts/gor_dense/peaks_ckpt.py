"""Чекпойнты dense_probe по фолдам и склейка в peaks{TAG}.json (bot-5).
Раскладка: CKPT_DIR/meta.json, CKPT_DIR/<layout>/f###.json (fr, hit, hn) + f###.npz (idx, p_<v>, lab, negscene отложенных токенов).
Склейка: python peaks_ckpt.py CKPT_DIR[,CKPT_DIR2,...] OUT.json — каталоги разных кернелов сливаются пофолдно
(дубликат фолда допустим только с тем же содержимым json). В OUT попадают только раскрои, где есть все фолды."""
import json, os, sys, collections, hashlib
from pathlib import Path
import numpy as np
def parse_folds(s):
    if not s.strip(): return None
    out = []
    for part in s.split(','):
        a, _, b = part.partition('-'); out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return sorted(set(out))
def _atomic(path, write):
    tmp = Path(str(path) + '.tmp'); write(tmp); os.replace(tmp, path)
def write_meta(ck, meta):
    ck = Path(ck); ck.mkdir(parents=True, exist_ok=True); f = ck / 'meta.json'
    if f.exists():
        old = json.load(open(f, encoding='utf-8'))
        for k in ('model', 'gsds', 'tag', 'n_tokens', 'token_labels', 'variants'): assert old[k] == meta[k], f'ckpt meta {k} differs: {old[k]} vs {meta[k]}'
    _atomic(f, lambda t: json.dump(meta, open(t, 'w', encoding='utf-8'), ensure_ascii=False))
def _fp(ck, nm, fi): return Path(ck) / nm / f'f{fi:03d}'
def has_fold(ck, nm, fi):
    b = _fp(ck, nm, fi); return Path(str(b) + '.json').exists() and Path(str(b) + '.npz').exists()
def n_done(ck, nm): return len(list((Path(ck) / nm).glob('f*.json'))) if (Path(ck) / nm).exists() else 0
def complete(ck, nm, n): return all(has_fold(ck, nm, i) for i in range(n))
def save_fold(ck, nm, fi, fr, hit, tok, hn, lab, negscene):
    b = _fp(ck, nm, fi); b.parent.mkdir(parents=True, exist_ok=True)
    arr = {}
    for v, (idx, p) in tok.items(): arr['idx'] = np.asarray(idx, np.int64); arr[f'p_{v}'] = np.asarray(p, np.float64)
    idx = arr.get('idx', np.zeros(0, np.int64)); arr['lab'] = lab[idx]; arr['negscene'] = negscene[idx]
    _atomic(Path(str(b) + '.npz'), lambda t: np.savez(open(t, 'wb'), **arr))   # npz первым: json — маркер готовности
    _atomic(Path(str(b) + '.json'), lambda t: json.dump(dict(fi=fi, fr=fr, hit=hit, hn=dict(hn)), open(t, 'w', encoding='utf-8'), ensure_ascii=False))
def _load(ck, nm, fi):
    b = _fp(ck, nm, fi); return json.load(open(str(b) + '.json', encoding='utf-8')), np.load(str(b) + '.npz')
def auc(s, t):
    s = np.asarray(s, float); t = np.asarray(t, bool)
    if t.all() or not t.any(): return None
    o = np.argsort(s, kind='mergesort'); rk = np.empty(len(s)); rk[o] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True); sums = np.bincount(inv, rk); rk = (sums / cnt)[inv]
    nP = t.sum(); return float((rk[t].sum() - nP * (nP + 1) / 2) / (nP * (len(s) - nP)))
def layout(ck, nm, n, variants):
    res, hit, hn, cps = [], collections.defaultdict(lambda: [0, 0]), collections.Counter(), collections.defaultdict(collections.Counter)
    I, L, N, Pv = [], [], [], collections.defaultdict(list)
    for fi in range(n):
        J, Z = _load(ck, nm, fi); res.append(J['fr']); hn.update(J['hn'])
        for k, (a, b) in J['hit'].items(): hit[k][0] += a; hit[k][1] += b
        for v, sv in J['fr'].get('cp_s', {}).items(): cps[v][str(float(sv))] += 1
        I.append(Z['idx']); L.append(Z['lab']); N.append(Z['negscene'])
        for v in variants: Pv[v].append(Z[f'p_{v}'] if f'p_{v}' in Z else np.zeros(len(Z['idx'])))
    lab = np.concatenate(L); neg = np.concatenate(N).astype(bool); st = {'top1_hit_held': {k: hit[k] for k in sorted(hit)}}
    for v in variants:
        p = np.concatenate(Pv[v]); okp = (lab >= 0) & ~neg; oka = lab >= 0
        st.setdefault('token_auc_pos_scenes', {})[v] = auc(p[okp], lab[okp] == 1); st.setdefault('token_auc_all', {})[v] = auc(p[oka], lab[oka] == 1)
    return res, st, hn, cps
def layout_stats(ck, nm, n, variants): return layout(ck, nm, n, variants)[1]
def assemble(cks):
    cks = [Path(c) for c in (cks if isinstance(cks, (list, tuple)) else [cks])]; meta = json.load(open(cks[0] / 'meta.json', encoding='utf-8'))
    for c in cks[1:]:
        m = json.load(open(c / 'meta.json', encoding='utf-8'))
        for k in ('model', 'gsds', 'tag', 'n_tokens', 'token_labels', 'variants'): assert m[k] == meta[k], f'{c}: meta {k} differs'
    if len(cks) > 1:   # слить в первый? нет — во временный общий вид: пофолдно выбираем источник
        merged = Path(str(cks[0]) + '_merged'); merged.mkdir(exist_ok=True); json.dump(meta, open(merged / 'meta.json', 'w', encoding='utf-8'), ensure_ascii=False)
        for c in cks:
            for f in c.glob('*/f*.json'):
                dst = merged / f.parent.name / f.name; dst.parent.mkdir(exist_ok=True)
                if dst.exists():
                    assert hashlib.sha256(dst.read_bytes()).digest() == hashlib.sha256(f.read_bytes()).digest(), f'fold conflict {f}'; continue
                (merged / f.parent.name / (f.stem + '.npz')).write_bytes(Path(str(f)[:-5] + '.npz').read_bytes()); dst.write_bytes(f.read_bytes())
        ck = merged
    else: ck = cks[0]
    R = {k: meta[k] for k in ('model', 'gsds', 'tag', 'n_tokens', 'token_labels', 'centre_hit')}; R.update(layouts={}, stats={}, hn_log=collections.Counter(), cp_s_chosen={})
    for nm, n in meta['nfolds'].items():
        if not complete(ck, nm, n): continue
        res, st, hn, cps = layout(ck, nm, n, meta['variants']); R['layouts'][nm] = res; R['stats'][nm] = st; R['hn_log'].update(hn)
        for v, c in cps.items(): R['cp_s_chosen'].setdefault(v, collections.Counter()).update(c)
    R['hn_log'] = dict(R['hn_log']); R['ckpt_dirs'] = [str(c) for c in cks]
    return R
if __name__ == '__main__':
    cks, out = sys.argv[1].split(','), sys.argv[2]; R = assemble(cks)
    json.dump(R, open(out, 'w', encoding='utf-8'), ensure_ascii=False); print(out, 'complete layouts:', sorted(R['layouts']))
