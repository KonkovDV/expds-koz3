"""bot-5. Плотный локализатор городищ без лидара, шаг 1: плотные патч-токены замороженного DINOv2 по ВСЕЙ сцене SpOr.
Идея (пересказ своими словами, код свой): сцена читается целиком с усреднением до фиксированного наземного шага
(GSDS, м/пкс; по умолчанию 2 и 4), режется на перекрывающиеся тайлы TILE пкс (шаг STRIDE), токены 14x14 пкс
усредняются по перекрытиям -> карта признаков на сетке первой шкалы; признаки второй шкалы (контекст) берутся
билинейно в центрах токенов первой и приклеиваются. Метки здесь НЕ используются.
Выход: $OUT/dense$TAG.npz: F (N x C*n_scales, float16), fid, row, col (сетка токенов файла), ux, uy (центр токена,
UTM участка), vf (доля валидных пикселей); files$TAG.json (по файлу: reg, file, grid H,W, gsd).
Env: DATA, OUT, MODEL (папка снапшота HF или id; поддерживаются dinov2 и dinov2-with-registers), DEVICE, TAG,
REVISION, GSDS ("2,4"), TILE (518), STRIDE (392), THREADS (2), LIMIT_FILES, ONLY_REGS (через запятую)."""
import os, sys, json, glob, math, time, resource
import numpy as np, rasterio, torch
from rasterio.enums import Resampling
from pyproj import Transformer
torch.set_num_threads(int(os.environ.get('THREADS', '2')))
D = os.environ.get('DATA', '/workspace/data/train-expds-3'); OUT = os.environ.get('OUT', '/workspace/koz3_shared/bot-5/analysis/gor_dense')
MD = os.environ.get('MODEL', 'facebook/dinov2-small'); DEV = os.environ.get('DEVICE', 'cpu'); TAG = os.environ.get('TAG', '')
GSDS = [float(s) for s in os.environ.get('GSDS', '2,4').split(',')]; TILE = int(os.environ.get('TILE', '518')); STRIDE = int(os.environ.get('STRIDE', '392'))
LIMIT = int(os.environ.get('LIMIT_FILES', '0')); ONLY = [s for s in os.environ.get('ONLY_REGS', '').split(',') if s]
assert TILE % 14 == 0 and STRIDE % 14 == 0
os.makedirs(OUT, exist_ok=True)
from transformers import AutoConfig
REV = os.environ.get('REVISION') or None   # пин ревизии HF (для id), для локальной папки не нужен
cfg = AutoConfig.from_pretrained(MD, revision=REV)
if getattr(cfg, 'num_register_tokens', 0):
    from transformers import Dinov2WithRegistersModel as MC
else:
    from transformers import Dinov2Model as MC
m = MC.from_pretrained(MD, revision=REV).eval()
bad = [n for n, t in list(m.named_parameters()) + list(m.named_buffers()) if t.device.type == 'meta']; assert not bad, bad[:5]
m = m.to(DEV); NPRE = 1 + int(getattr(cfg, 'num_register_tokens', 0)); C = cfg.hidden_size
if DEV == 'cuda': m = m.half()
MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]; STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]
files = sorted(glob.glob(f'{D}/*/*SpOr*/*.tif*'))
if ONLY: files = [f for f in files if f.split('/')[-3] in ONLY]
if LIMIT: files = files[:LIMIT]
def run_tiles(x):
    """x: 3xHxW нормированный. Возвращает карту токенов (H//14, W//14, C), усреднённую по перекрытиям."""
    H, W = x.shape[1:]; Ht, Wt = H // 14, W // 14; acc = np.zeros((Ht, Wt, C), np.float32); cnt = np.zeros((Ht, Wt), np.float32)
    th, tw = min(TILE, Ht * 14), min(TILE, Wt * 14)
    ys = list(range(0, max(Ht * 14 - th, 0) + 1, STRIDE)); xs = list(range(0, max(Wt * 14 - tw, 0) + 1, STRIDE))
    if ys[-1] != Ht * 14 - th: ys.append(Ht * 14 - th)
    if xs[-1] != Wt * 14 - tw: xs.append(Wt * 14 - tw)
    for y0 in ys:
        for x0 in xs:
            t = torch.from_numpy(np.ascontiguousarray(x[:, y0:y0 + th, x0:x0 + tw]))[None].to(DEV)
            if DEV == 'cuda': t = t.half()
            with torch.no_grad():
                o = m(pixel_values=t).last_hidden_state[0, NPRE:].float().cpu().numpy().reshape(th // 14, tw // 14, C)
            acc[y0 // 14:(y0 + th) // 14, x0 // 14:(x0 + tw) // 14] += o; cnt[y0 // 14:(y0 + th) // 14, x0 // 14:(x0 + tw) // 14] += 1
    return acc / np.maximum(cnt, 1)[..., None], len(ys) * len(xs)
def bilinear(Fm, r, c):
    """Fm (h,w,C); r,c — дробные индексы токенов (центры). Края — клип."""
    h, w = Fm.shape[:2]; r = np.clip(r, 0, h - 1); c = np.clip(c, 0, w - 1)
    r0 = np.floor(r).astype(int); c0 = np.floor(c).astype(int); r1 = np.minimum(r0 + 1, h - 1); c1 = np.minimum(c0 + 1, w - 1)
    a = (r - r0)[:, None]; b = (c - c0)[:, None]
    return (Fm[r0, c0] * (1 - a) * (1 - b) + Fm[r0, c1] * (1 - a) * b + Fm[r1, c0] * a * (1 - b) + Fm[r1, c1] * a * b)
t0 = time.time(); FL, FID, ROW, COL, UX, UY, VF = [], [], [], [], [], [], []; finfo = []; ntiles = 0
for fi, f in enumerate(files):
    reg = f.split('/')[-3]
    epsg = int(json.load(open(f'{D}/{reg}/UTM.json'))['crs'].split(':')[-1])
    with rasterio.open(f) as d:
        tr = Transformer.from_crs(d.crs, epsg, always_xy=True)
        cx0, cy0 = (d.bounds.left + d.bounds.right) / 2, (d.bounds.bottom + d.bounds.top) / 2
        xa, ya = tr.transform([cx0, cx0 + d.transform.a * 100], [cy0, cy0]); gsd = math.hypot(xa[1] - xa[0], ya[1] - ya[0]) / 100
        sc = max(1, int(max(d.width, d.height) / 1024))
        small = d.read(out_shape=(d.count, max(1, d.height // sc), max(1, d.width // sc)), resampling=Resampling.average)
        alpha = d.count >= 4 and d.colorinterp[3].name == 'alpha'
        vs = small[3] > 0 if alpha else np.all(small[:3] != (d.nodata if d.nodata is not None else -1e9), 0)
        st = []
        for b in range(3):
            v = small[b][vs].astype(np.float32); st.append(np.percentile(v, [2, 98]) if v.size > 100 else (0, 255))
        maps = []
        for g in GSDS:
            fct = g / gsd; Hh, Ww = max(14, int(round(d.height / fct))), max(14, int(round(d.width / fct)))
            Hh, Ww = Hh // 14 * 14, Ww // 14 * 14
            arr = d.read(out_shape=(d.count, Hh, Ww), resampling=Resampling.average)
            val = arr[3] > 127 if alpha else (np.all(arr[:3] != (d.nodata if d.nodata is not None else -1e9), 0) & np.any(arr[:3] != 0, 0))
            x = np.zeros((3, Hh, Ww), np.float32)
            for b_ in range(3): x[b_] = np.clip((arr[b_] - st[b_][0]) / max(st[b_][1] - st[b_][0], 1e-6), 0, 1)
            x = ((x * val) - MEAN) / STD; del arr
            Fm, nt = run_tiles(x); ntiles += nt
            vfrac = val[:Hh // 14 * 14, :Ww // 14 * 14].reshape(Hh // 14, 14, Ww // 14, 14).mean((1, 3))
            maps.append((Fm, vfrac, d.height / Hh, d.width / Ww))  # пикс. исходника на пиксель шкалы
        F0, v0, sy0, sx0 = maps[0]; h0, w0 = F0.shape[:2]
        rr, cc = np.meshgrid(np.arange(h0), np.arange(w0), indexing='ij'); rr = rr.ravel(); cc = cc.ravel()
        # центр токена в пикселях исходника
        py = (rr + 0.5) * 14 * sy0; px = (cc + 0.5) * 14 * sx0
        parts = [F0[rr, cc]]
        for Fm, vf_, syk, sxk in maps[1:]:
            parts.append(bilinear(Fm, py / (14 * syk) - 0.5, px / (14 * sxk) - 0.5))
        X0, Y0 = rasterio.transform.xy(d.transform, py, px, offset='ul')
        ux, uy = tr.transform(np.asarray(X0), np.asarray(Y0))
        FL.append(np.concatenate(parts, 1).astype(np.float16)); FID.append(np.full(len(rr), fi, np.int32)); ROW.append(rr.astype(np.int16)); COL.append(cc.astype(np.int16))
        UX.append(np.asarray(ux, np.float64)); UY.append(np.asarray(uy, np.float64)); VF.append(v0[rr, cc].astype(np.float16))
        finfo.append(dict(reg=reg, file=os.path.basename(f), H=int(h0), W=int(w0), gsd=gsd, epsg=epsg))
    print(f'[{time.time()-t0:.0f}s] {fi+1}/{len(files)} {reg} gsd {gsd:.2f} grid {h0}x{w0} tiles {ntiles} rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f}MB', flush=True)
np.savez(f'{OUT}/dense{TAG}.npz', F=np.concatenate(FL), fid=np.concatenate(FID), row=np.concatenate(ROW), col=np.concatenate(COL),
         ux=np.concatenate(UX), uy=np.concatenate(UY), vf=np.concatenate(VF))
json.dump(dict(files=finfo, model=MD, gsds=GSDS, tile=TILE, stride=STRIDE, C=C, nprefix=NPRE), open(f'{OUT}/files{TAG}.json', 'w'), ensure_ascii=False)
print('done tokens', sum(len(a) for a in FID), 'tiles', ntiles, f'{time.time()-t0:.0f}s')
