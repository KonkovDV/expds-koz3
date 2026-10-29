"""bot-5. Потоковая конвертация bot-7 cands_kur_ae.json (123 МБ) -> компактный npz (без json.load всего файла)."""
import re, numpy as np, json, glob, collections
SRC = '/workspace/koz3_shared/bot-7/work/kur/cands_kur_ae.json'
OUT = '/workspace/koz3_shared/bot-5/tmp/kurg_ae/cand7.npz'
pat = re.compile(r'\{[^{}]*\}')
cols = collections.defaultdict(list); buf = ''
with open(SRC, encoding='utf-8') as fh:
    while True:
        ch = fh.read(1 << 22)
        if not ch: break
        buf += ch; last = 0
        for mm in pat.finditer(buf):
            d = json.loads(mm.group(0)); last = mm.end()
            for k in ('cand_id', 'reg', 'file', 'ux', 'uy', 'epsg', 'gsd', 'gen', 'score', 'rank', 'D', 'fold'):
                cols[k].append(d[k])
        buf = buf[last:]
A = {k: np.array(v) for k, v in cols.items()}
A['gen'] = np.array([float(g[3:]) for g in cols['gen']], np.float32)
for k in ('ux', 'uy'): A[k] = A[k].astype(np.float64)
for k in ('gsd', 'score', 'D'): A[k] = A[k].astype(np.float32)
for k in ('cand_id', 'epsg', 'rank', 'fold'): A[k] = A[k].astype(np.int64)
A['tif'] = A.pop('file'); np.savez(OUT, **A)
print('n', len(A['cand_id']), 'unique ids', len(np.unique(A['cand_id'])))
for r in np.unique(A['reg']):
    e = set(A['epsg'][A['reg'] == r].tolist())
    u = json.load(open(glob.glob(f'/workspace/data/train-expds-3/{r}/*UTM*.json')[0]))['crs'].split('::')[-1]
    fs = set(A['tif'][A['reg'] == r].tolist())
    print(r, int((A['reg'] == r).sum()), 'epsg', e, 'regUTM', u, 'files', fs, 'folds', set(A['fold'][A['reg'] == r].tolist()))
