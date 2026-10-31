"""bot-6 ORGATE_S: копия bot-4/work/cropverif/embed.py; изменены только модель (DINOv2-S/reg, локальная копия HF facebook/dinov2-with-registers-small@0d9846e) и пути.
Эмбеддинги DINOv2-S/reg (facebook/dinov2-with-registers-small, Apache-2.0) для оптических кандидатов пула.
Кроп: то же чтение, что crop_verify.extract_crops (expds.tiles.iter_sources), шаг 0,5 м, окно 112 px = 56 м, центр = центроид кандидата.
Выход: emb_dinov2b_reg_112.npz: stack_idx, ok, fill, emb (float16, N×1536 = CLS(pooler) ‖ среднее патч-токенов)."""
import os, sys, time, json, resource, numpy as np, torch
from collections import defaultdict
sys.path.insert(0, '/workspace/expds-koz3')
torch.set_num_threads(2)
from shapely.geometry import Point
from expds.catalog import scan_root
from expds.crs import WEB_MERCATOR, to_crs
from expds.tiles import fix_channels, iter_sources
from transformers import AutoModel
ROOT = '/workspace/data/train-expds-3'; RES = 0.5; TILE = 112; OUT = 'emb_dinov2s_reg_112.npz'; CV = '/workspace/koz3_shared/bot-4/work/cropverif/'
LIMIT = int(os.environ.get('LIMIT', '0'))
z = np.load(CV + 'optical_pool_cands.npz', allow_pickle=False)
sidx, region, mk, x, y = z['stack_idx'], z['region'], z['mk'], z['x3857'], z['y3857']
N = len(sidx) if not LIMIT else LIMIT
model = AutoModel.from_pretrained('dinov2s_reg', local_files_only=True).eval()
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
emb = np.zeros((N, 2 * model.config.hidden_size), np.float16); ok = np.zeros(N, bool); fill = np.zeros(N, np.int64)
regions = {r.name: r for r in scan_root(ROOT)}
groups = defaultdict(list)
for k in range(N): groups[(str(region[k]), str(mk[k]))].append(k)
t0 = time.time(); t_emb = 0.0; n_emb = 0
def embed(X):
    global t_emb, n_emb
    t = time.perf_counter()
    with torch.inference_mode():
        xb = (torch.from_numpy(X).float() / 255.0 - MEAN) / STD
        o = model(pixel_values=xb)
        cls_ = o.pooler_output; patch = o.last_hidden_state[:, 1 + model.config.num_register_tokens:].mean(1)
        e = torch.cat([cls_, patch], 1).numpy().astype(np.float16)
    t_emb += time.perf_counter() - t; n_emb += len(X); return e
for gi, ((rn, m), ks) in enumerate(sorted(groups.items())):
    reg = regions.get(rn)
    if reg is None: print('нет участка', rn, flush=True); continue
    X = np.zeros((len(ks), 3, TILE, TILE), np.uint8); got = np.zeros(len(ks), bool); fl = np.zeros(len(ks), np.int64)
    pts = [Point(x[k], y[k]) for k in ks]
    try:
        for _, crs, bnd, read in iter_sources(reg, m, [], 'las', RES, TILE, None):
            for j, pt in enumerate(pts):
                q = to_crs(pt, WEB_MERCATOR, crs)
                if not bnd.contains(q): continue
                img, _ = read(q.x, q.y)
                sp = np.any(img, axis=0) if getattr(img, 'ndim', 0) == 3 else img
                n = int(np.count_nonzero(sp))
                if n <= fl[j]: continue
                X[j] = fix_channels(np.asarray(img), 3); got[j] = True; fl[j] = n
    except Exception as e:
        print('ошибка чтения', rn, m, repr(e), flush=True)
    jj = np.flatnonzero(got)
    for s in range(0, len(jj), 32):
        b = jj[s:s + 32]; e = embed(X[b])
        for j, row in zip(b, e): emb[ks[j]] = row
    for j in jj: ok[ks[j]] = True; fill[ks[j]] = fl[j]
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    print(f'{gi+1}/{len(groups)} {rn} {m} n={len(ks)} ok={got.sum()} t={time.time()-t0:.0f}s emb_ms={1000*t_emb/max(1,n_emb):.1f} maxrss={rss}MB', flush=True)
    if gi % 10 == 0: np.savez(OUT + '.part.npz', stack_idx=sidx[:N], ok=ok, fill=fill, emb=emb)
meta = dict(model='facebook/dinov2-with-registers-small@0d9846e56b43a21fa46d7f3f5070f0506a5795a9', sha256='78e7f952c5af83cf6da38f08ff52b062bbe00c1adf39c4c7e0361969f27abae7', license='Apache-2.0', res=RES, tile=TILE, feat='pooler(CLS)||mean(patch)',
            n=int(N), ok=int(ok.sum()), total_s=round(time.time() - t0, 1), embed_ms_per_crop=round(1000 * t_emb / max(1, n_emb), 2),
            threads=torch.get_num_threads(), maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
np.savez(OUT if not LIMIT else 'emb_smoke_s.npz', stack_idx=sidx[:N], ok=ok, fill=fill, emb=emb, meta=json.dumps(meta))
if os.path.exists(OUT + '.part.npz'): os.remove(OUT + '.part.npz')
print(json.dumps(meta), flush=True)
