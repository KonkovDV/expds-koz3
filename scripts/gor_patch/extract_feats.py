"""bot-5. ТОЛЬКО ЧЕСТНЫЙ ЗАМЕР (не для сдачи). Виньетки SpOr 384 м (шаг 192 м) + контекст 768 м с тем же центром
-> замороженный DINOv2 (Apache-2.0): [CLS, mean(patch tokens)] каждой шкалы. Метки здесь НЕ используются.
Выход: $OUT/feats$TAG.npz (F float16) + meta$TAG.json (участок, файл, центр и углы виньетки в UTM участка).
Окружение: DATA (train-expds-3), OUT, MODEL (локальная папка снапшота HF или id, по умолчанию facebook/dinov2-small),
DEVICE (cpu|cuda), TAG, LIMIT_FILES. На боксе: CPU, 2 потока. Оценка — scripts/gor_patch/gor_patch_eval.py на боксе."""
import os, sys, json, glob, math, time, resource
import numpy as np, rasterio, torch
from rasterio.windows import Window
from rasterio.enums import Resampling
from pyproj import Transformer
torch.set_num_threads(int(os.environ.get('THREADS', '2')))
D = os.environ.get('DATA', '/workspace/data/train-expds-3'); OUT = os.environ.get('OUT', '/workspace/koz3_shared/bot-5/analysis/gor_patch')
MD = os.environ.get('MODEL', 'facebook/dinov2-small'); DEV = os.environ.get('DEVICE', 'cpu'); TAG = os.environ.get('TAG', '')
from transformers import Dinov2Model
m = Dinov2Model.from_pretrained(MD).eval()
bad = [n for n, t in list(m.named_parameters()) + list(m.named_buffers()) if t.device.type == 'meta']
assert not bad, bad[:5]
m = m.to(DEV)
PS, ST, CTX, PX = 384.0, 192.0, 768.0, 224
MEAN = np.array([0.485, 0.456, 0.406])[:, None, None]; STD = np.array([0.229, 0.224, 0.225])[:, None, None]
LIMIT = int(os.environ.get('LIMIT_FILES', '0'))
t0 = time.time(); meta = []; F = []
files = sorted(glob.glob(f'{D}/*/*SpOr*/*.tif*'))
if LIMIT: files = files[:LIMIT]
def stretch(a, valid):
    a = a.astype(np.float32); out = np.zeros_like(a)
    for b in range(3):
        v = a[b][valid]
        if v.size < 100: return None
        lo, hi = np.percentile(v, [2, 98]); out[b] = np.clip((a[b] - lo) / max(hi - lo, 1e-6), 0, 1)
    return out
for fi, f in enumerate(files):
    reg = f.split('/')[-3]
    epsg = int(json.load(open(f'{D}/{reg}/UTM.json'))['crs'].split(':')[-1])
    with rasterio.open(f) as d:
        tr = Transformer.from_crs(d.crs, epsg, always_xy=True)
        cx0, cy0 = (d.bounds.left + d.bounds.right) / 2, (d.bounds.bottom + d.bounds.top) / 2
        # наземный метр на пиксель: через перевод соседних пикселей в UTM
        xa, ya = tr.transform([cx0, cx0 + d.transform.a * 100], [cy0, cy0])
        gsd = math.hypot(xa[1] - xa[0], ya[1] - ya[0]) / 100
        # статистика для растяжки по всему файлу (по уменьшенной копии)
        sc = max(1, int(max(d.width, d.height) / 1024))
        small = d.read(out_shape=(d.count, d.height // sc, d.width // sc), resampling=Resampling.average)
        if d.colorinterp[3].name == 'alpha': vs = small[3] > 0
        else: vs = np.all(small[:3] != (d.nodata if d.nodata is not None else -1e9), 0)
        st = []
        for b in range(3):
            v = small[b][vs].astype(np.float32); st.append(np.percentile(v, [2, 98]) if v.size > 100 else (0, 255))
        pps, pst = PS / gsd, ST / gsd
        # быстрый путь: весь файл читается ОДИН раз с усреднением до шага шкалы (PS/PX и CTX/PX м/пкс), затем кропы
        DEC = {}
        for size in (PS, CTX):
            f_ = (size / PX) / gsd; Hh, Ww = max(1, int(round(d.height / f_))), max(1, int(round(d.width / f_)))
            arr = d.read(out_shape=(d.count, Hh, Ww), resampling=Resampling.average)
            if d.colorinterp[3].name == 'alpha': val = arr[3] > 127
            else: val = np.all(arr[:3] != (d.nodata if d.nodata is not None else -1e9), 0) & np.any(arr[:3] != 0, 0)
            x = np.zeros((3, Hh, Ww), np.float32)
            for b_ in range(3): x[b_] = np.clip((arr[b_] - st[b_][0]) / max(st[b_][1] - st[b_][0], 1e-6), 0, 1)
            x = ((x * val) - MEAN) / STD
            pad = PX  # нулевое (после нормировки — «чёрное») поле вокруг для кропов у края
            xp = np.zeros((3, Hh + 2 * pad, Ww + 2 * pad), np.float32) + ((0 - MEAN) / STD).astype(np.float32)
            xp[:, pad:pad + Hh, pad:pad + Ww] = x; vp = np.zeros((Hh + 2 * pad, Ww + 2 * pad), bool); vp[pad:pad + Hh, pad:pad + Ww] = val
            DEC[size] = (xp, vp, f_, pad); del arr
        nx = max(1, int(math.ceil((d.width - pps) / pst)) + 1); ny = max(1, int(math.ceil((d.height - pps) / pst)) + 1)
        for iy in range(ny):
            for ix in range(nx):
                cpx, cpy = ix * pst + pps / 2, iy * pst + pps / 2
                feats = []; ok = True
                for size in (PS, CTX):
                    xp, vp, f_, pad = DEC[size]
                    c0, r0 = int(round(cpx / f_ - PX / 2)) + pad, int(round(cpy / f_ - PX / 2)) + pad
                    x = xp[:, r0:r0 + PX, c0:c0 + PX]; valid = vp[r0:r0 + PX, c0:c0 + PX]
                    if size == PS and valid.mean() < 0.5: ok = False; break
                    with torch.no_grad():
                        o = m(pixel_values=torch.from_numpy(np.ascontiguousarray(x)[None]).to(DEV)).last_hidden_state[0].float().cpu()
                    feats.append(torch.cat([o[0], o[1:].mean(0)]).numpy())
                if not ok: continue
                X0, Y0 = d.xy(cpy, cpx)  # row, col -> map coords (центр)
                ux, uy = tr.transform(X0, Y0)
                # углы 384-м патча в UTM
                cs = [d.xy(cpy + sy * pps / 2, cpx + sx * pps / 2) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
                cu = [tr.transform(*c) for c in cs]
                meta.append(dict(reg=reg, file=os.path.basename(f), ux=ux, uy=uy, corners=cu, gsd=gsd))
                F.append(np.concatenate(feats).astype(np.float16))
    print(f'[{time.time()-t0:.0f}s] {fi+1}/{len(files)} {reg} gsd {gsd:.2f} patches {len(meta)} rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f}MB', flush=True)
    if (fi + 1) % 20 == 0 or fi + 1 == len(files):
        np.savez(f'{OUT}/feats{TAG}{"_test" if LIMIT else ""}.npz', F=np.stack(F))
        json.dump(meta, open(f'{OUT}/meta{TAG}{"_test" if LIMIT else ""}.json', 'w'), ensure_ascii=False)
print('done', len(meta), 'patches', f'{time.time()-t0:.0f}s')
