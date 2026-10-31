# W1 V8: box CPU re-inference of ae OOF with max_det 300 (and its first 100 = max_det 100) — code copied from koz3-w1-dfine-ae-final.
import os, sys, json, re, math, time, collections, glob
import numpy as np, torch
from pathlib import Path
torch.set_num_threads(int(os.environ.get("NT", 5)))
T0 = time.time()
def log(*a): print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)
CFG = dict(res=0.5, tile=640, overlap=160, bs=4, min_score=0.05, nms_iou=0.5, max_det=300)
CLASSES = ["kurgany_tselye", "kurgany_povrezhdennye", "gorodishcha", "fortifikatsii", "arkhitektury"]
MARKUP_TYPE = "ae"; SP_GROUP = "ae_dfine"
DATA = Path("/workspace/koz3/w1/aedata")
def fetch(key, size): return DATA / key
def raster_path(key, size): return DATA / key
def target_crs(inv, ds):
    if inv["utm"]:
        d = json.load(open(fetch(*inv["utm"]), encoding="utf-8")); m = re.search(r"EPSG::?(\d+)", d.get("crs", ""))
        if m: return f"EPSG:{m.group(1)}"
    from pyproj import Transformer
    lon, lat = Transformer.from_crs(ds.crs, "EPSG:4326", always_xy=True).transform(*ds.xy(ds.height // 2, ds.width // 2))
    return f"EPSG:{(32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1}"

class Scene:
    """Растр SpOr, приведённый на лету к UTM 0,5 м (WarpedVRT; при уменьшении — average, иначе bilinear), RGB uint8."""
    def __init__(self, inv, key, size):
        import rasterio
        from rasterio.vrt import WarpedVRT
        from rasterio.enums import Resampling
        from rasterio.warp import calculate_default_transform
        self.src = rasterio.open(raster_path(key, size)); self.crs = target_crs(inv, self.src)
        tr, w, h = calculate_default_transform(self.src.crs, self.crs, self.src.width, self.src.height, *self.src.bounds,
                                               resolution=CFG["res"])
        try: src_m = abs(calculate_default_transform(self.src.crs, self.crs, self.src.width, self.src.height, *self.src.bounds)[0].a)
        except Exception: src_m = CFG["res"]
        rs = Resampling.average if src_m < CFG["res"] else Resampling.bilinear
        self.vrt = WarpedVRT(self.src, crs=self.crs, transform=tr, width=w, height=h, resampling=rs, nodata=0)
        self.W, self.H, self.T = w, h, tr; self.bands = list(range(1, min(3, self.src.count) + 1))
        self.stretch = None
        if self.src.dtypes[0] != "uint8":
            ov = self.src.read(1, out_shape=(min(1024, self.src.height), min(1024, self.src.width)))
            v = ov[ov > 0]; self.stretch = tuple(np.percentile(v, [1, 99])) if v.size else (0, 1)
    def read(self, x0, y0, n):
        from rasterio.windows import Window
        # WarpedVRT не умеет boundless — читаем пересечение и дополняем нулями вручную
        a = np.zeros((len(self.bands), n, n), dtype=self.src.dtypes[0])
        cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(self.W, x0 + n), min(self.H, y0 + n)
        if cx1 > cx0 and cy1 > cy0:
            a[:, cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0] = self.vrt.read(self.bands, window=Window(cx0, cy0, cx1 - cx0, cy1 - cy0))
        if a.shape[0] < 3: a = np.repeat(a[:1], 3, 0)
        if self.stretch:
            lo, hi = self.stretch; a = np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1e-6) * 255, 0, 255)
        return np.ascontiguousarray(a.astype(np.uint8).transpose(1, 2, 0))
    def to_px(self, b):
        inv = ~self.T; (x0, y1), (x1, y0) = inv * (b[0], b[1]), inv * (b[2], b[3])
        return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)
    def close(self): self.vrt.close(); self.src.close()
def _valid(g): return g if g.is_valid else g.buffer(0)
OOF_SAVE_MIN = float(os.environ.get("OOF_SAVE_MIN", 0.2))
VARIANTS = ("noflip", "flipnms", "flipwbf")
def _wins(W, H, tile, overlap):  # как dfine_aeor.windows: последний ряд/столбец прижат к краю
    step = tile - overlap
    def axis(n):
        if n <= tile: return [0]
        xs = list(range(0, n - tile + 1, step))
        if xs[-1] + tile < n: xs.append(n - tile)
        return xs
    return [(x, y) for y in axis(int(H)) for x in axis(int(W))]
def wbf_fast(B, L, C, n_views=2, thr=0.55, cell=128.0):
    """Тот же алгоритм, что dfine_aeor.wbf (по классу, по убыванию уверенности, слитый бокс — среднее с весами,
    уверенность = сумма / max(n_views, членов)), но кандидаты ищутся через сеточный индекс (миллионы боксов)."""
    B = np.asarray(B, np.float64).reshape(-1, 4); L = np.asarray(L, np.float64).reshape(-1); C = np.asarray(C, int).reshape(-1)
    oB, oS, oL = [], [], []
    for c in np.unique(C):
        idx = np.where(C == c)[0]; idx = idx[np.argsort(-L[idx], kind="stable")]
        fused, wsum, ssum, cnt, grid = [], [], [], [], collections.defaultdict(list)
        def cells(b):
            return [(i, j) for i in range(int(b[0] // cell), int(b[2] // cell) + 1) for j in range(int(b[1] // cell), int(b[3] // cell) + 1)]
        for i in idx:
            b = B[i]; best, bj = -1.0, -1
            cand = {j for ce in cells(b) for j in grid.get(ce, ())}
            for j in cand:
                f = fused[j]; w = max(0.0, min(b[2], f[2]) - max(b[0], f[0])); h = max(0.0, min(b[3], f[3]) - max(b[1], f[1]))
                inter = w * h; ua = (b[2] - b[0]) * (b[3] - b[1]) + (f[2] - f[0]) * (f[3] - f[1]) - inter
                iou = inter / ua if ua > 0 else 0.0
                if iou > best: best, bj = iou, j
            if bj >= 0 and best >= thr:
                wsum[bj] = wsum[bj] + b * L[i]; ssum[bj] += L[i]; cnt[bj] += 1
                fused[bj] = wsum[bj] / max(ssum[bj], 1e-12)
                for ce in cells(fused[bj]):
                    if bj not in grid[ce]: grid[ce].append(bj)
            else:
                j = len(fused); fused.append(b.copy()); wsum.append(b * L[i]); ssum.append(float(L[i])); cnt.append(1)
                for ce in cells(b): grid[ce].append(j)
        for f, s, n in zip(fused, ssum, cnt):
            oB.append(f); oS.append(min(1.0, s / max(n_views, n))); oL.append(int(c))
    if not oB: return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
    return np.asarray(oB), np.asarray(oS), np.asarray(oL, int)
FOLDMAP = {"001_Гнёздово_Нивленское":0,"101_Гнёздово_Днепровская":0,"102_Гнёздово_Заольшанская":0,"103_Гнёздово_Ольшанская":0,
 "179_Гнёздово_Лесная_1":0,"180_Гнёздово_Лесная_2":0,"181_Гнёздово_Лесная_3":0,"036_Тимерево":1,"015_Демидовка":2,"189_Макеево":3,
 "012_Сож":4,"019_Селяне":4,"029_Вышегжа_3":4,"058_Перынь":4,"043_Гочево":3,"067_Чудово":3,"183_Сарское":3,"184_Шурскол":3,"037_Никола_Ленивец":2,"110_Саратов":2,"065_Шниткино":1}
def inv_of(region):
    tifs = sorted((str(p.relative_to(DATA)), p.stat().st_size) for p in (DATA / region).glob("03_*/*_Ae_*.tif"))
    u = DATA / region / "UTM.json"
    return {"region": region, "tifs": tifs, "utm": (f"{region}/UTM.json", u.stat().st_size) if u.exists() else None}
@torch.no_grad()
def infer_region(model, inv, dev, win_filter=None):
    from torchvision.ops import batched_nms
    from shapely.geometry import Polygon, mapping
    from shapely.ops import transform as st
    from pyproj import Transformer
    model.eval(); T = CFG["tile"]; out = {f"{v}_m{m}": [] for v in ("noflip", "flipwbf") for m in (100, 300)}; wstats = []
    def nms_t(B, L, C):
        if not len(B): return B, L, C
        k = batched_nms(B, L, C, CFG["nms_iou"]); return B[k], L[k], C[k]
    for key, size in inv["tifs"]:
        sc = Scene(inv, key, size); views = {m: [([], [], []), ([], [], [])] for m in (100, 300)}
        wins = _wins(sc.W, sc.H, T, CFG["overlap"])
        if win_filter is not None: wins = [w for w in wins if win_filter(key, w)]
        t_r = time.time(); log(f"[infer] {inv['region']} {key.rsplit('/', 1)[-1]}: {sc.W}x{sc.H} px, окон {len(wins)}")
        for b in range(0, len(wins), CFG["bs"]):
            chunk, imgs = [], []
            for x, y in wins[b:b + CFG["bs"]]:
                im = sc.read(x, y, T)
                if (im.max(2) == 0).mean() > 0.95: continue
                chunk.append((x, y)); imgs.append(im)
            if not imgs: continue
            px = torch.from_numpy(np.stack(imgs).transpose(0, 3, 1, 2).astype(np.float32) / 255).to(dev)
            for vi, flip in enumerate((False, True)):
                inp = px.flip(3) if flip else px
                o = model(pixel_values=inp)
                prob = o.logits.float().sigmoid(); bx = o.pred_boxes.float()
                if flip: bx = bx.clone(); bx[..., 0] = 1 - bx[..., 0]
                Q, C = prob.shape[1:]
                sc_all, idx_all = prob.flatten(1).topk(min(300, Q * C), 1)
                for i, (x, y) in enumerate(chunk):
                    s3 = sc_all[i]
                    wstats.append({"tif": key.rsplit('/', 1)[-1], "x": x, "y": y, "view": vi, "n05": int((s3 >= 0.05).sum()), "n04": int((s3 >= 0.4).sum()),
                                   "n07": int((s3 >= 0.7).sum()), "s100": float(s3[99]), "s300": float(s3[-1])})
                    for m in (100, 300):
                        sc_, idx = s3[:m], idx_all[i][:m]
                        keep = sc_ >= CFG["min_score"]; q = idx[keep] // C; c = idx[keep] % C; bb = bx[i][q] * T
                        xyxy = torch.stack([bb[:, 0] - bb[:, 2] / 2 + x, bb[:, 1] - bb[:, 3] / 2 + y, bb[:, 0] + bb[:, 2] / 2 + x, bb[:, 1] + bb[:, 3] / 2 + y], 1)
                        views[m][vi][0].append(xyxy.cpu()); views[m][vi][1].append(sc_[keep].cpu()); views[m][vi][2].append(c.cpu())
        cat = lambda v: (torch.cat(v[0]), torch.cat(v[1]), torch.cat(v[2])) if v[0] else (torch.zeros(0, 4), torch.zeros(0), torch.zeros(0, dtype=torch.long))
        res = {}
        for m in (100, 300):
            v0, v1 = cat(views[m][0]), cat(views[m][1]); n0, n1 = nms_t(*v0), nms_t(*v1)
            res[f"noflip_m{m}"] = n0
            res[f"flipwbf_m{m}"] = wbf_fast(torch.cat([n0[0], n1[0]]).numpy(), torch.cat([n0[1], n1[1]]).numpy(), torch.cat([n0[2], n1[2]]).numpy(), 2, 0.55)
        log(f"[infer] {inv['region']}: готово {len(wins)} окон за {time.time() - t_r:.0f} с;", {k: len(v[1]) for k, v in res.items()})
        to3857 = Transformer.from_crs(sc.crs, "EPSG:3857", always_xy=True).transform; t = np.linspace(0, 2 * np.pi, 33)[:-1]
        for vn, (B, L, Cc) in res.items():
            B, L, Cc = (np.asarray(B), np.asarray(L), np.asarray(Cc))
            for (x0, y0, x1, y1), s, c in zip(B, L, Cc):
                if s < 0.2: continue
                cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, max((x1 - x0) / 2, .5), max((y1 - y0) / 2, .5)
                ring = [sc.T * (cx + rx * math.cos(a), cy + ry * math.sin(a)) for a in t]
                out[vn].append({"type": "Feature", "geometry": mapping(_valid(st(to3857, Polygon(ring)))),
                                "properties": {"class_name": CLASSES[int(c)], "confidence": round(float(s), 5), "region_name": inv["region"],
                                               "crs": "EPSG:3857", "markup_type": MARKUP_TYPE, "original_crs": sc.crs, "group": SP_GROUP, "threshold_group": SP_GROUP,
                                               "fold": FOLDMAP[inv["region"]], "tif": key.rsplit("/", 1)[-1], "tta": vn.split("_")[0], "max_det": int(vn[-3:])}})
        sc.close()
    return out, wstats
if __name__ == "__main__":
    from transformers import DFineForObjectDetection
    regions = sys.argv[1].split(","); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
    W = Path("/workspace/koz3/out/w1-dfine-ae-final"); dev = torch.device("cpu"); models = {}
    for r in regions:
        if (OUT / f"{r}.json").exists(): log("уже есть", r); continue
        k = FOLDMAP[r]
        if k not in models:
            models.clear(); m = DFineForObjectDetection.from_pretrained(str(W / f"dfine_f{k}"), device_map=None)
            assert not any(p.is_meta for p in m.parameters()); assert all(torch.isfinite(p).all() for p in m.parameters())
            models[k] = m.eval(); log("модель", k, round(sum(p.numel() for p in m.parameters()) / 1e6, 2), "M")
        out, ws = infer_region(models[k], inv_of(r), dev)
        json.dump({"region": r, "fold": k, "wstats": ws, "features": out}, open(OUT / f"{r}.json.part", "w", encoding="utf-8"), ensure_ascii=False)
        os.replace(OUT / f"{r}.json.part", OUT / f"{r}.json"); log("сохранено", r, {a: len(b) for a, b in out.items()})
