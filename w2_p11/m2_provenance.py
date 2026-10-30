"""P-W2-11 M2: провенанс 15 единиц c3-sc (s0/s1/s2 × f0..f4): seed/fold/n_det из лога ядра против OOF, merge_summary,
md5 весов (стейдж koz3-w2-weights, demfill/w, p11/s1f4_v1) и Kaggle-карт (w2/maps). Дубли md5 весов между сидами = ошибка."""
import json, glob, hashlib, os, re, collections
W = '/workspace/koz3/w2'
def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()
MS = json.load(open(f'{W}/oof/merged/merge_summary.json'))
out = {'units': [], 'weights': {}, 'dup_weights': [], 'maps_dirs': {}}
for sd in ('s0', 's1', 's2'):
    for d in sorted(glob.glob(f'{W}/oof/{sd}/*/')):
        k = d.rstrip('/').split('/')[-1]; L = json.load(open(glob.glob(d + '*.log')[0]))
        cfg = json.loads([e['data'] for e in L if 'CFG {' in e['data']][0].split('CFG ', 1)[1])
        done = {int(m.group(1)): int(m.group(2)) for e in L for m in [re.search(r'c3-sc_f(\d)\] готово: .*"n_det": (\d+)', e['data'])] if m}
        for f in done:
            oof = json.load(open(f'{d}oof_cnx_c3-sc_f{f}.geojson'))['features']
            out['units'].append(dict(seed=sd, fold=f, kernel=k, log_seed=cfg['seed'], jobs=cfg['jobs'], iters=cfg['iters'], n_det_log=done[f], n_det_oof=len(oof),
                                     n_det_merge=MS[f'c3-sc_{sd}']['n_det'][str(f)], ok=(cfg['seed'] == int(sd[1]) and done[f] == len(oof) == MS[f'c3-sc_{sd}']['n_det'][str(f)])))
for p in sorted(glob.glob(f'{W}/bk/stage/weights/s*/*/weights_*.pt') + glob.glob(f'{W}/demfill/w/*/*/weights_*.pt') + glob.glob(f'{W}/p11/s1f4_v1/weights_*.pt') + glob.glob(f'{W}/oof/s2/*/weights_*.pt')):
    out['weights'][p.replace(W + '/', '')] = md5(p)
by = collections.defaultdict(set)
for p, h in out['weights'].items():
    m = re.search(r'/(s\d)/.*_f(\d)\.pt$', p)
    if m: by[h].add(f'{m.group(1)}f{m.group(2)}')
    if 'p11/s1f4_v1' in p: by[h].add('s1f4(v1)')
out['dup_weights'] = [dict(md5=h, labels=sorted(v)) for h, v in by.items() if len({x[:4] for x in v}) > 1]
for sd in ('s0', 's1', 's2'):
    for d in sorted(glob.glob(f'{W}/maps/{sd}/*/')):
        fs = glob.glob(d + '**/maps_*.npz', recursive=True)
        out['maps_dirs'][d.replace(W + '/', '')] = dict(n=len(fs), mtime=sorted({os.path.getmtime(x) // 3600 * 3600 for x in fs})[:3])
json.dump(out, open(f'{W}/p11/m2_provenance.json', 'w'), ensure_ascii=False, indent=1)
for u in out['units']: print(u)
print('DUP', out['dup_weights']); print(json.dumps(out['maps_dirs'], ensure_ascii=False))
