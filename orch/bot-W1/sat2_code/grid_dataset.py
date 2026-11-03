"""sp_det2 GRID-cache Dataset for centerhead (CenterNet) — 3-ch SpOr, kurgans only, stride 4, markup sp_or.
Input = packed split dir (pack_split): images.npy uint8 (N,3,S,S), tiles.csv, objs.npy (M,5) [tile, x, y, diam_px, cls]
with cls in centerhead HM_CLASSES order (0 kurgany_tselye, 1 kurgany_povrezhdennye). Folds are pinned in tiles.csv
(from sat2_dataset_manifest.json, spor_focal family folds) — never recomputed from grid positives.
Augs: D8 (centerhead.d8) + quality aug (down-up, blur, JPEG q30–90, noise, haze, sharpen) + brightness/contrast.
Sampler: site-first p ∝ n_obj^0.5 for positives (split per region), framing negatives share `neg_frac`."""
from __future__ import annotations
import csv, json, math
from pathlib import Path
import numpy as np

IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None] * 255
IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None] * 255


def norm(img_u8: np.ndarray) -> np.ndarray:
    return (img_u8.astype(np.float32) - IMNET_MEAN) / IMNET_STD


# ---------------------------------------------------------------- packing (CPU, box)
def pack_split(caches: list[str], out: str, folds: set[int] | None, framing_only: bool, neg_cap: float | None,
               seed: int = 0) -> dict:
    """Grid caches (expds.tiles + grid_labels.py) → one compact split dir. neg_cap: max negatives per region =
    max(4, neg_cap·positives) (None = all). framing_only: drop negatives outside labelled framing."""
    rng = np.random.default_rng(seed)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    sel = []
    for c in caches:
        c = Path(c); rows = list(csv.DictReader(open(c / 'tiles.csv', encoding='utf-8')))
        objs = np.load(c / 'objs.npy')
        by_reg = {}
        for i, r in enumerate(rows):
            if folds is not None and (r['fold'] in ('', 'None') or int(r['fold']) not in folds):
                continue
            by_reg.setdefault(r['region'], []).append(i)
        for reg, idx in by_reg.items():
            pos = [i for i in idx if int(rows[i]['n_obj']) > 0]
            neg = [i for i in idx if int(rows[i]['n_obj']) == 0 and (not framing_only or rows[i]['framing'] == '1')]
            if neg_cap is not None:
                k = max(4, int(neg_cap * len(pos)))
                if len(neg) > k:
                    neg = sorted(rng.choice(neg, k, replace=False).tolist())
            for i in sorted(pos + neg):
                sel.append((c, rows[i], objs[objs[:, 0] == i]))
    S = None; n = len(sel); shards = {}
    for k, (c, r, o) in enumerate(sel):
        key = (str(c), int(r['shard']))
        if key not in shards:
            shards[key] = np.load(c / f"images_{int(r['shard']):04d}.npy", mmap_mode='r')
        S = shards[key].shape[-1]; break
    imgs = np.lib.format.open_memmap(out / 'images.npy', 'w+', np.uint8, (n, 3, S, S))
    rows_out, objs_out = [], []
    for k, (c, r, o) in enumerate(sel):
        key = (str(c), int(r['shard']))
        if key not in shards:
            shards[key] = np.load(c / f"images_{int(r['shard']):04d}.npy", mmap_mode='r')
        imgs[k] = shards[key][int(r['idx'])][:3]
        rows_out.append({**r, 'src_cache': Path(c).name})
        if len(o):
            o = o.copy(); o[:, 0] = k; objs_out.append(o)
    imgs.flush(); del imgs
    with open(out / 'tiles.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0])); w.writeheader(); w.writerows(rows_out)
    np.save(out / 'objs.npy', np.concatenate(objs_out) if objs_out else np.zeros((0, 5), np.float32))
    s = {'out': str(out), 'tiles': n, 'pos': sum(int(r['n_obj']) > 0 for r in rows_out),
         'objs_inside': sum(int(r['n_obj']) for r in rows_out), 'regions': len({r['region'] for r in rows_out}),
         'folds': sorted({r['fold'] for r in rows_out}), 'GB': round(n * 3 * S * S / 1e9, 3), 'tile': S}
    (out / 'pack.json').write_text(json.dumps(s, ensure_ascii=False, indent=1))
    return s


# ---------------------------------------------------------------- dataset (torch)
class GridSplit:
    def __init__(self, d: str, mmap: bool = False):
        d = Path(d)
        self.images = np.load(d / 'images.npy', mmap_mode='r' if mmap else None)
        self.rows = list(csv.DictReader(open(d / 'tiles.csv', encoding='utf-8')))
        o = np.load(d / 'objs.npy')
        self.objs = [o[o[:, 0] == i, 1:5].astype(np.float64) for i in range(len(self.rows))]
        self.S = self.images.shape[-1]

    def __len__(self):
        return len(self.rows)


def quality_aug(x, rng):
    """x torch uint8 (3,S,S) → uint8. Each op with own prob (Altai/EN balq recipe)."""
    import torch, torch.nn.functional as F
    import torchvision.transforms.functional as TF
    from torchvision.io import decode_jpeg, encode_jpeg
    S = x.shape[-1]
    if rng.random() < 0.3:                                   # down-up (resolution jitter)
        f = rng.uniform(0.4, 0.85); s = max(8, int(S * f))
        y = F.interpolate(x[None].float(), size=(s, s), mode='area')
        x = F.interpolate(y, size=(S, S), mode='bilinear', align_corners=False)[0].clamp(0, 255).to(torch.uint8)
    if rng.random() < 0.2:
        x = TF.gaussian_blur(x, 5, float(rng.uniform(0.3, 1.5)))
    if rng.random() < 0.15:
        x = TF.adjust_sharpness(x, float(rng.uniform(1.5, 3.0)))
    if rng.random() < 0.25:
        x = decode_jpeg(encode_jpeg(x.contiguous(), quality=int(rng.integers(30, 91))))
    xf = x.float()
    if rng.random() < 0.5:                                   # brightness/contrast/gamma
        xf = (xf - 128) * rng.uniform(0.75, 1.3) + 128 + rng.uniform(-25, 25)
    if rng.random() < 0.15:                                  # haze
        a = rng.uniform(0.1, 0.35); xf = xf * (1 - a) + 255 * a * rng.uniform(0.7, 1.0)
    if rng.random() < 0.2:
        xf = xf + torch.randn_like(xf) * float(rng.uniform(2, 10))
    return xf.clamp(0, 255)


class CenterGridDataset:
    """torch Dataset: idx → dict(image f32 (3,S,S) ImageNet-norm, hm (2,S/4,S/4), wh, off, reg_mask)."""

    def __init__(self, split: GridSplit, train: bool, cfg: dict, seed: int = 0):
        self.sp, self.train, self.cfg, self.seed = split, train, cfg, seed
        self.epoch = 0

    def __len__(self):
        return len(self.sp)

    def __getitem__(self, i):
        import torch
        from centerhead.d8 import d8_array, d8_objs
        from centerhead.targets import render_targets_strided
        rng = np.random.default_rng((self.seed, self.epoch, int(i), np.random.randint(1 << 30)))
        img = np.asarray(self.sp.images[i]); o = self.sp.objs[i]; S = self.sp.S
        if self.train:
            k, fl = int(rng.integers(4)), bool(rng.integers(2))
            img = d8_array(img, k, fl); o = d8_objs(o, S, k, fl)
            x = quality_aug(torch.from_numpy(np.ascontiguousarray(img)), rng).numpy()
        else:
            x = img.astype(np.float32)
        x = (x - IMNET_MEAN) / IMNET_STD
        t = render_targets_strided(o, S, self.cfg['stride'], 2, self.cfg)
        return {'image': torch.from_numpy(np.ascontiguousarray(x, np.float32)), **{k: torch.from_numpy(v) for k, v in t.items()}}


def site_first_weights(split: GridSplit, neg_frac: float = 0.3, power: float = 0.5) -> np.ndarray:
    """Sampling weights: positives ∝ n_obj^power, each region's positive mass ∝ (region n_obj)^power (site-first);
    negatives share neg_frac uniformly per region."""
    rows = split.rows; n = np.array([int(r['n_obj']) for r in rows], np.float64)
    reg = np.array([r['region'] for r in rows]); w = np.zeros(len(rows))
    pos = n > 0
    regs = sorted(set(reg[pos]))
    rmass = {r: n[(reg == r)].sum() ** power for r in regs}; Z = sum(rmass.values()) or 1
    for r in regs:
        m = (reg == r) & pos; ww = n[m] ** power
        w[m] = (1 - neg_frac) * rmass[r] / Z * ww / ww.sum()
    neg = ~pos
    if neg.any():
        nregs = sorted(set(reg[neg]))
        for r in nregs:
            m = (reg == r) & neg; w[m] = neg_frac / len(nregs) / m.sum()
    return w / w.sum()
