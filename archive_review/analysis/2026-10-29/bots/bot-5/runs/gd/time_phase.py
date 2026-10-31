"""bot-5: реальное CPU-время фазы gor_dense (GD.apply, как в pipeline) по ВСЕМ участкам трейна с SpOr
(тестового набора на машине нет). torch 2 потока; запуск через watch.py (nice 10, RSS-сторож 2000 МБ)."""
import sys, json, time, tomllib, logging
from pathlib import Path
T0 = time.monotonic()
import torch; torch.set_num_threads(2)
ON = Path('/workspace/koz3_shared/bot-5/work/gd/e2e/on'); sys.path.insert(0, str(ON))
from expds import gor_dense as GD, catalog
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
DATA = Path('/workspace/data/train-expds-3')
cfg = tomllib.load(open(ON / 'configs/default.toml', 'rb'))
t = time.monotonic(); regions = {}
for d in sorted(DATA.iterdir()):
    if d.is_dir():
        try:
            r = catalog.scan_region(d)
        except Exception:
            continue
        if (getattr(r, 'rasters', {}) or {}).get('sp_or'):
            regions[r.name] = r
scan_s = time.monotonic() - t
timing = {}; orig = GD.region_points
def rp(*a, **k):
    k['timing'] = timing; return orig(*a, **k)
GD.region_points = rp
t = time.monotonic(); out, n = GD.apply([], regions, cfg, ON); phase_s = time.monotonic() - t
files = list(timing.values())
res = dict(n_regions=len(regions), n_files=len(files), n_points=n, phase_s=round(phase_s, 1),
           files_sum_s=round(sum(f['s'] for f in files), 1), load_s=round(phase_s - sum(f['s'] for f in files), 1),
           tiles=sum(f['tiles'] for f in files), vignettes=sum(f['vignettes'] for f in files), tokens=sum(f['tokens'] for f in files),
           max_file_s=round(max(f['s'] for f in files), 2), scan_s=round(scan_s, 1), torch_threads=torch.get_num_threads(),
           total_s=round(time.monotonic() - T0, 1))
json.dump(dict(summary=res, files=timing), open('time_phase.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False))
