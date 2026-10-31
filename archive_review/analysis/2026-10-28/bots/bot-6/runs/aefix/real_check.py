"""[raster_fix] на настоящих снимках Ae трейна: CRS/шаг, растяжка int16, доля «поля нулей», время (CPU коробки).
Сеть не запускается (нет инференса). Шаг сети ae_fort_mask: seg.res 0,5 (BUDGET.md §3.1), max_upsample 2."""
import glob, json, resource, sys, time
sys.path.insert(0, '/workspace/koz3_shared/bot-6/repo')
import numpy as np
from rasterio.enums import Resampling
from rasterio.windows import Window
from expds.raster import configure_raster_fix, open_metric, zero_field_mask
from expds.tiles import band_stats, working_res
ROOT = '/workspace/data/train-expds-3'
def tif(code): return sorted(glob.glob(f'{ROOT}/{code}_*/03_*Ae*/*.tif'))
out = {}
def scan(path, seg_res=0.5, block=4096, zero=False, read=True):
    """Как _raster_with_model: рабочий шаг, окна block×block, чтение; без сети. Время чтения (и маски)."""
    t0 = time.process_time(); w0 = time.time(); rec = {}
    with open_metric(path) as ds:
        native = abs(ds.res[0]); r = working_res(native, seg_res, 2.0); scale = r / native
        w, h = max(1, int(ds.width / scale)), max(1, int(ds.height / scale)); n = min(ds.count, 3)
        rec.update(crs=str(ds.crs), native=round(native, 4), work=round(r, 4), px=[w, h], nodata=ds.nodata, dtype=ds.dtypes[0])
        step = block - 256; nwin = 0; zt = 0.0; zpx = 0; tot = 0
        if read:
            for y0 in range(0, max(h - 256, 1), step):
                for x0 in range(0, max(w - 256, 1), step):
                    bw, bh = min(block, w - x0), min(block, h - y0)
                    a = ds.read(list(range(1, n + 1)), window=Window(x0 * scale, y0 * scale, bw * scale, bh * scale),
                                out_shape=(n, bh, bw), resampling=Resampling.average if scale > 1 else Resampling.bilinear)
                    nwin += 1
                    if not a.any(): continue
                    if zero:
                        z0 = time.process_time(); m = zero_field_mask(a, ds.nodata, 5); zt += time.process_time() - z0
                        zpx += int(m.sum()); tot += m.size
    rec.update(windows=nwin, cpu_s=round(time.process_time() - t0, 2), wall_s=round(time.time() - w0, 2))
    if zero: rec.update(zero_mask_cpu_s=round(zt, 2), zero_field_share=round(zpx / max(tot, 1), 4))
    return rec
# 1) 3857 → UTM
for code in ('015', '029'):
    for p in tif(code):
        configure_raster_fix(None); off = scan(p)
        configure_raster_fix({'reproject_3857': True}); on = scan(p)
        out[f'{code} reproject'] = {'file': p.split('/')[-1], 'off': off, 'on': on}; print(code, json.dumps(out[f'{code} reproject'], ensure_ascii=False), flush=True)
# 2) int16 identity
for code in ('017', '036', '065', '105', '106', '183', '184'):
    for p in tif(code):
        configure_raster_fix(None); t0 = time.process_time(); s_off = band_stats(p, 'ae'); t1 = time.process_time() - t0
        configure_raster_fix({'int16_identity': True}); t0 = time.process_time(); s_on = band_stats(p, 'ae'); t2 = time.process_time() - t0
        out[f'{code} int16 {p.split("/")[-1]}'] = {'off': s_off, 'on': s_on, 'cpu_off': round(t1, 2), 'cpu_on': round(t2, 2)}
        print(code, s_off, '→', s_on, flush=True)
# 3) поле нулей (017, 036) — полное чтение на рабочем шаге
configure_raster_fix(None)
for code in ('017', '036'):
    for p in tif(code):
        out[f'{code} zero'] = scan(p, zero=True); print(code, out[f'{code} zero'], flush=True)
out['maxrss_mb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
json.dump(out, open('real_check.json', 'w'), ensure_ascii=False, indent=1, default=str)
print('maxrss_mb', out['maxrss_mb'])
