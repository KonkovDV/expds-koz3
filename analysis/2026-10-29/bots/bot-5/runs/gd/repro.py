"""bot-5: воспроизведение peaks.json (fam10 фолд 4, 'all+cp') модулем expds.gor_dense из сырых TIF + сверка токенов с dense.npz.
python repro.py REG [REG...]  → JSON в stdout."""
import sys, json, time, os
from pathlib import Path
import numpy as np
REPO = Path('/workspace/koz3_shared/bot-5/repo'); sys.path.insert(0, str(REPO))
import torch; torch.set_num_threads(2)
from expds import gor_dense as GD, gor_gate, catalog
ROOT = Path(__file__).resolve().parent / 'root'; DATA = Path('/workspace/data/train-expds-3')
A = Path('/workspace/koz3_shared/bot-5/analysis/gor_dense')
st = GD.settings({'gor_dense': {'centre_s': 250.0}}); dev = torch.device('cpu')
t0 = time.time(); model = gor_gate.load_backbone(st, ROOT, dev); tl = time.time() - t0
H = GD.load_dense_head(ROOT / 'models/gor_dense/fam10_f4_head.npz', None)
Z = np.load(A / 'dense.npz'); FI = json.load(open(A / 'files.json', encoding='utf-8'))
PK = json.load(open(A / 'peaks.json', encoding='utf-8'))['layouts']['fam10'][4]
cp = PK['cp_s']['all']; out = dict(load_backbone_s=tl, cp=cp, regions={})
for r in sys.argv[1:]:
    reg = catalog.scan_region(DATA / r); epsg = gor_gate._epsg(reg); cands = []; fl = []
    for p in sorted(reg.rasters['sp_or']):
        t1 = time.time(); tok = GD.dense_tokens(model, Path(p), epsg, dev); dt = time.time() - t1
        k = [i for i, f in enumerate(FI['files']) if f['reg'] == r and f['file'] == Path(p).name][0]
        ii = np.flatnonzero(Z['fid'] == k)
        same_grid = bool(len(ii) == len(tok['row']) and (Z['row'][ii] == tok['row']).all() and (Z['col'][ii] == tok['col']).all())
        Fr = Z['F'][ii].astype(np.float32); Fm = tok['F'].astype(np.float32)
        pr, pm = GD.head_p(H, Fr), GD.head_p(H, Fm)
        fl.append(dict(file=Path(p).name, s=round(dt, 2), tiles=tok['tiles'], tokens=len(tok['row']), same_grid=same_grid,
                       max_abs_F=float(np.abs(Fr - Fm).max()), max_abs_ux=float(np.abs(Z['ux'][ii] - tok['ux']).max()),
                       max_abs_uy=float(np.abs(Z['uy'][ii] - tok['uy']).max()), vf_eq=bool((Z['vf'][ii].astype(np.float32) == tok['vf']).all()),
                       max_abs_p=float(np.abs(pr - pm).max())))
        cands += GD.file_candidates(pm, tok, cp)
    pk = GD.region_peaks(cands); ref = PK['all+cp'].get(r, [])
    out['regions'][r] = dict(files=fl, module=[list(t) for t in pk], ref=ref,
                             max_d_m=max((float(np.hypot(a[1] - b[1], a[2] - b[2])) for a, b in zip(pk, ref)), default=None),
                             max_d_score=max((abs(a[0] - b[0]) for a, b in zip(pk, ref)), default=None), n_eq=len(pk) == len(ref))
print(json.dumps(out, ensure_ascii=False, indent=1))
