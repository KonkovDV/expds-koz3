"""bot-5: та же фаза gor_dense, но бэкбон заменён нулевой заглушкой — время НЕ-модельной части (чтение TIF,
перепроецирование/ресэмплинг, виньетки, головы, пики). Нижняя граница фазы на любом GPU."""
import sys, json, time, tomllib, types
from pathlib import Path
import torch; torch.set_num_threads(2)
ON = Path('/workspace/koz3_shared/bot-5/work/gd/e2e/on'); sys.path.insert(0, str(ON))
from expds import gor_dense as GD, gor_gate, catalog
class Dummy:
    config = types.SimpleNamespace(hidden_size=384)
    n = 0
    def __call__(self, pixel_values):
        b, _, h, w = pixel_values.shape; Dummy.n += b
        return types.SimpleNamespace(last_hidden_state=torch.zeros(b, 1 + (h // 14) * (w // 14), 384))
gor_gate.load_backbone = lambda st, root, dev: Dummy()
DATA = Path('/workspace/data/train-expds-3'); cfg = tomllib.load(open(ON / 'configs/default.toml', 'rb'))
regions = {}
for d in sorted(DATA.iterdir()):
    if d.is_dir():
        try: r = catalog.scan_region(d)
        except Exception: continue
        if (getattr(r, 'rasters', {}) or {}).get('sp_or'): regions[r.name] = r
t = time.monotonic(); out, n = GD.apply([], regions, cfg, ON); s = time.monotonic() - t
print(json.dumps(dict(n_regions=len(regions), phase_nomodel_s=round(s, 1), forward_images=Dummy.n)))
