"""P-W2-10: CPU-инференс c3-sc (centerhead) по готовому рельефу — копия predict_region/save_maps ядра s2f23 (вендор w2/vendor = вендор ядра).
usage: infer_cpu.py --relief X.npy --meta X.json --weights weights_cnx_c3-sc_fK.pt --fold K --out maps.npz [--tiles-only N]"""
import sys, json, argparse, time
import numpy as np, torch
V = '/workspace/koz3/w2/vendor'; sys.path.insert(0, V)
import cnx_head as HD
ap = argparse.ArgumentParser(); ap.add_argument('--relief'); ap.add_argument('--meta'); ap.add_argument('--weights'); ap.add_argument('--fold', type=int)
ap.add_argument('--out'); ap.add_argument('--tiles-only', type=int, default=0); ap.add_argument('--threads', type=int, default=3)
A = ap.parse_args(); torch.set_num_threads(A.threads)
CFG = json.load(open('/workspace/koz3/w2/oof/s2/koz3-w2-cnx-c3sc-s2f23/report_cnx_full.json'))['cfg']
HEAD = HD.get_head('centerhead', CFG); HD.check_head(HEAD)
TAGS5 = ("lrm_s", "lrm_l", "slope", "svf_r", "opp_r"); ch = 3
c = torch.load(A.weights, map_location='cpu', weights_only=True)
assert c['in_channels'] == ch and c['train']['fold'] == A.fold, (c['in_channels'], c['train'])
model = HEAD.build(ch, 'tu-convnext_tiny', None)
sd = model.state_dict(); missing = set(sd) ^ set(c['state_dict'])
assert not missing, sorted(missing)[:10]
model.load_state_dict({k: v.to(sd[k].dtype) for k, v in c['state_dict'].items()}); model.eval()
m = json.load(open(A.meta)); a = np.load(A.relief, mmap_mode='r')
def norm(x): return (x.float() - 128.0) / 127.0
def read_win(arr, x0, y0, n):
    C, H, W = arr.shape; out = np.zeros((C, n, n), arr.dtype)
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + n), min(H, y0 + n)
    if xb > xa and yb > ya: out[:, ya - y0:yb - y0, xa - x0:xb - x0] = arr[:, ya:yb, xa:xb]
    return out
@torch.no_grad()
def predict(a):
    C, H, W = a.shape; S, st, s = CFG["crop"], CFG["infer_stride"], HEAD.stride
    h4, w4, n4 = -(-H // s), -(-W // s), S // s
    tiles = [t for t in HD.tile_origins(H, W, S, st, s) if (a[0, t[0]:t[0] + S:8, t[1]:t[1] + S:8] > 0).any()]
    if A.tiles_only: tiles = tiles[:A.tiles_only]
    acc, cnt, cnt4 = {}, np.zeros((H, W), np.float32), np.zeros((h4, w4), np.float32)
    tf = [(k, f) for f in (False, True) for k in range(4)][:max(1, CFG["tta"])]; t0 = time.time()
    for b in range(0, len(tiles), CFG["infer_bs"]):
        bt = tiles[b:b + CFG["infer_bs"]]
        x = norm(torch.from_numpy(np.stack([read_win(a, xx, yy, S)[:ch] for yy, xx in bt]))); o = {}
        for k, fl in tf:
            xi = torch.rot90(x, k, (2, 3)); xi = xi.flip(3) if fl else xi
            yo = model(xi.contiguous(memory_format=torch.channels_last))
            yo = {kk: v.float() for kk, v in yo.items() if v is not None}
            ym = HD.untta_map(yo.pop("mask"), k, fl); yo = HEAD.untta(yo, k, fl); yo["mask"] = ym
            for kk, v in yo.items(): o[kk] = o[kk] + v if kk in o else v
        for kk, v in o.items():
            v = (v / len(tf)).numpy(); full = kk == "mask"; Hh, Ww, n, d = (H, W, S, 1) if full else (h4, w4, n4, s)
            if kk not in acc: acc[kk] = np.zeros((v.shape[1], Hh, Ww), np.float32)
            for (yy, xx), oi in zip(bt, v):
                y0, x0 = yy // d, xx // d; hh, ww = min(n, Hh - y0), min(n, Ww - x0); acc[kk][:, y0:y0 + hh, x0:x0 + ww] += oi[:, :hh, :ww]
        for yy, xx in bt:
            hh, ww = min(S, H - yy), min(S, W - xx); cnt[yy:yy + hh, xx:xx + ww] += 1
            y0, x0 = yy // s, xx // s; hh, ww = min(n4, h4 - y0), min(n4, w4 - x0); cnt4[y0:y0 + hh, x0:x0 + ww] += 1
        print(f'{b + len(bt)}/{len(tiles)} tiles {time.time() - t0:.0f}s', flush=True)
    va = (a[0] > 0) & (cnt > 0); va4 = HD.block_any(va, s) & (cnt4 > 0)
    for kk in acc:
        full = kk == "mask"; c_, v_ = (cnt, va) if full else (cnt4, va4)
        acc[kk] /= np.maximum(c_, 1)[None]
        if full or kk in HEAD.prob_keys: acc[kk] = (1 / (1 + np.exp(-acc[kk]))) * v_[None]
    acc["_valid"], acc["_valid_h"] = va, va4
    return acc, len(tiles), time.time() - t0
maps, nt, dt = predict(a); d = {}
for kk, v in maps.items():
    if kk.startswith("_"): continue
    d[kk] = np.round(v * 255).astype(np.uint8) if kk == "mask" else v.astype(np.float32 if kk in HEAD.prob_keys else np.float16)
meta = dict(region=m['region'], fold=A.fold, family=None, tag='c3-sc', head=HEAD.name, head_api=HD.HEAD_API, stride=HEAD.stride,
            left=m["left"], top=m["top"], step=m["step"], step_head=m["step"] * HEAD.stride, crs=m["crs"],
            H=int(maps["mask"].shape[1]), W=int(maps["mask"].shape[2]), tags=list(TAGS5[:ch]), weights=A.weights, relief=A.relief, n_tiles=nt, t_s=round(dt, 1))
np.savez_compressed(A.out, valid=maps["_valid"].astype(np.uint8), valid_h=maps["_valid_h"].astype(np.uint8), meta=np.array(json.dumps(meta, ensure_ascii=False)), **d)
print('wrote', A.out, nt, 'tiles', round(dt), 's')
