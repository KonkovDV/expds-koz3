"""Кривые сабсета: юниты honest-10 с участками ConvNeXt заданных фолдов — симлинки на pkl.  python cnx_subset2.py <folds> <curves_dir> <out_dir>"""
import json, os, sys
from pathlib import Path
sys.path.insert(0, '/workspace/koz3_shared/bot-10/repo')
from expds.naming import site_family
folds = {int(x) for x in sys.argv[1].split(',')}
RF = json.load(open('/workspace/koz3_shared/bot-9/kaggle/convnext_lidar/c3sc_f01_out/region_fold.json', encoding='utf-8'))['region_fold']
UNIT = {}
for ms in json.load(open('/workspace/koz3_shared/bot-10/work/family_clusters.json', encoding='utf-8'))['clusters'].values():
    for m in ms: UNIT[m] = '+'.join(sorted(ms))
units = sorted({UNIT.get(site_family(r), site_family(r)) for r, f in RF.items() if f in folds})
src, out = Path(sys.argv[2]).resolve(), Path(sys.argv[3]); out.mkdir(exist_ok=True)
miss = []
for u in units:
    p = src / f'{u}.pkl'
    if not p.exists(): miss.append(u); continue
    t = out / p.name
    if t.is_symlink() or t.exists(): t.unlink()
    os.symlink(p, t)
print(len(units), 'units; missing', miss)
