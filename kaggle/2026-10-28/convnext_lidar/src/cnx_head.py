"""Сменная голова курганов для convnext_lidar (bot-9). Контракт — HEAD_INTERFACE.md (версия HEAD_API = 1).

Ядро (convnext_main.py) владеет: энкодером/декодером smp.Unet, масочной головой (gor/fort/arch, BCE+Dice),
окнами/аугментациями, тайлингом/TTA участка, декодированием масок, экспортом GeoJSON.
Голова владеет: модулем(ями) над признаками декодера, целями по точкам курганов, своим лоссом, обратным TTA
своих карт и декодированием пиков в кандидаты (метры родной СК участка).

Выбор: env CNX_HEAD = "heat" (встроенная, по умолчанию) | "centerhead" (пакет bot-2, вендорится build.py)
| "pkg.module:factory" (любая фабрика, возвращающая объект с тем же API).
"""
from __future__ import annotations

import importlib
import math

import numpy as np
import torch
import torch.nn as nn

HEAD_API = 1
HM_CLASSES = ("kurgany_tselye", "kurgany_povrezhdennye")     # канал 0, 1 ключа "hm"
MASK_CLASSES = ("gorodishcha", "fortifikatsii", "arkhitektury")
CAND_COLS = ("X", "Y", "cls", "score")                      # обязательные колонки decode(); radius_m — необязательная


def d8_points(xy: np.ndarray, S: int, k: int, fl: bool) -> np.ndarray:
    """Непрерывные пиксельные (x=столбец, y=строка) координаты в кропе S×S → после
    a = np.rot90(a, k, axes=(-2,-1)); a = a[..., ::-1] if fl (ровно так ядро поворачивает растры)."""
    p = np.asarray(xy, np.float64).reshape(-1, 2).copy()
    for _ in range(k % 4):
        p = np.stack([p[:, 1], S - p[:, 0]], 1)
    if fl:
        p[:, 0] = S - p[:, 0]
    return p


def untta_map(t: torch.Tensor, k: int, fl: bool) -> torch.Tensor:
    """Скалярная карта (B,C,H,W), посчитанная на входе rot90(k)+flip(3) → в исходную систему."""
    t = t.flip(3) if fl else t
    return torch.rot90(t, -k, (2, 3))


def tile_origins(H: int, W: int, S: int = 512, step: int = 384, stride: int = 1):
    """Начала тайлов (y0, x0) — та же формула, что centerhead.decode.tile_origins (bot-2): шаг `step`, последний тайл
    выровнен вниз на кратное stride. Хвост < stride пикселей не покрыт (ядро зануляет его в valid)."""
    if step % stride or S % stride:
        raise ValueError("step и S должны делиться на stride")
    def ax(n):
        last = max(0, (n - S) // stride * stride)
        return sorted(set(list(range(0, max(1, n - S), step)) + [last]))
    return [(y, x) for y in ax(H) for x in ax(W)]


def _blocks(a: np.ndarray, s: int) -> np.ndarray:
    H, W = a.shape; h, w = -(-H // s), -(-W // s)
    p = np.zeros((h * s, w * s), np.float32); p[:H, :W] = a
    return p.reshape(h, s, w, s)


def block_any(a: np.ndarray, s: int) -> np.ndarray:
    """(H,W) bool → (ceil(H/s),ceil(W/s)): клетка валидна, если валиден хоть один пиксель (= valid_to_stride bot-2)."""
    return a.astype(bool) if s == 1 else _blocks(a, s).max(axis=(1, 3)) > 0


def block_mean(a: np.ndarray, s: int) -> np.ndarray:
    return a.astype(np.float32) if s == 1 else _blocks(a, s).mean(axis=(1, 3))


class Dec2Dict(nn.Module):
    """smp.Unet(classes=2+3) → dict (маски — ключ ядра, см. HEAD_INTERFACE.md). Голова "heat" = штатная segmentation_head smp (Conv3×3 16→5) на выходе декодера."""

    def __init__(self, unet: nn.Module, hm_bias: float):
        super().__init__()
        self.unet = unet
        with torch.no_grad():
            unet.segmentation_head[0].bias.fill_(hm_bias)         # априор p≈0,1 (CenterNet) — и для масок

    def forward(self, x):
        y = self.unet(x)
        return {"hm": y[:, :2], "mask": y[:, 2:5]}


class HeatHead:
    """Встроенная голова (как в первой версии ядра): гауссианы σ=max(1, 0,15·D) px, penalty-reduced focal (α2, β4),
    пики — совместный максимум kt/kp, maxfilter peak_k, класс = argmax, подавление внутри городищ."""
    name = "heat"
    stride = 1                    # все карты головы — на полном разрешении
    circle_r_m = None             # круг-выход — CFG["circle_r_m"] ядра (1,0 м)
    out_keys = ("hm", "mask")
    prob_keys = ("hm",)
    vec_keys = ()

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def build(self, ch: int, encoder: str, weights) -> nn.Module:
        import segmentation_models_pytorch as smp
        unet = smp.Unet(encoder, encoder_weights=weights, in_channels=ch, classes=5)
        return Dec2Dict(unet, self.cfg.get("hm_bias", -2.19))

    def targets(self, objs: np.ndarray, S: int) -> dict:
        c = self.cfg
        hm = np.zeros((2, S, S), np.float32)
        for x, y, d, k in np.asarray(objs, np.float64).reshape(-1, 4):
            s = max(c["sigma_min_px"], c["sigma_k"] * d); r = int(math.ceil(3 * s))
            cx, cy = int(math.floor(x)), int(math.floor(y))
            x0, x1, y0, y1 = max(0, cx - r), min(S, cx + r + 1), max(0, cy - r), min(S, cy + r + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            xx = np.arange(x0, x1) - cx; yy = np.arange(y0, y1) - cy
            g = np.exp(-(yy[:, None] ** 2 + xx[None, :] ** 2) / (2 * s * s)).astype(np.float32)
            np.maximum(hm[int(k), y0:y1, x0:x1], g, out=hm[int(k), y0:y1, x0:x1])
        return {"hm": hm}

    def loss(self, out: dict, tgt: dict, valid: torch.Tensor):
        x = out["hm"].float(); g = tgt["hm"].float(); v = valid[:, None].float()
        log_p, log_1mp, p = -nn.functional.softplus(-x), -nn.functional.softplus(x), torch.sigmoid(x)
        pos = (g >= 1.0).float()
        l = -(pos * (1 - p) ** 2 * log_p + (1 - pos) * (1 - g).clamp(min=0) ** 4 * p ** 2 * log_1mp) * v
        lh = l.sum() / (pos * v).sum().clamp(min=1)
        return lh, {"heat": float(lh.detach())}

    def untta(self, o: dict, k: int, fl: bool) -> dict:
        return {kk: untta_map(v, k, fl) for kk, v in o.items()}

    def decode(self, maps: dict, left: float, top: float, step: float, valid: np.ndarray, suppress: np.ndarray) -> dict:
        from scipy import ndimage as ndi
        c = self.cfg; hm = maps["hm"]
        joint = np.maximum(hm[0], hm[1])
        pk = (joint == ndi.maximum_filter(joint, size=c["peak_k"])) & (joint >= c["min_score"]) & valid & ~suppress
        ys, xs = np.nonzero(pk); sc = joint[ys, xs]; o = np.argsort(-sc, kind="stable")[:c["max_det"]]
        ys, xs, sc = ys[o], xs[o], sc[o]
        cls = np.where(hm[0, ys, xs] >= hm[1, ys, xs], 0, 1)
        return {"X": left + (xs + 0.5) * step, "Y": top - (ys + 0.5) * step, "cls": cls.astype(np.int64),
                "score": sc.astype(np.float64)}


class CenterHeadAdapter:
    """Адаптер пакета bot-2 `centerhead` (bot-2/repo/centerhead, его INTERFACE.md; вендорится build.py как vendor/centerhead).
    UnetCenterNet: CenterHead на признаке декодера шага cfg["stride"] (по умолчанию 4 → 64 канала, 128×128 при 512)
    + seg-голова smp (3 маски, полное разрешение). Цели render_targets_strided (hm/wh/off/reg_mask), лосс center_loss
    (focal + L1 wh/off, valid max-pool до шага), decode (пики pool×pool, NMS по расстоянию в классе).
    Конфиг: centerhead.config.DEFAULTS ← общие ключи ядра (sigma_k, sigma_min_px, default_diam_m, min_score, topk=max_det)
    ← env CNX_CH_<KEY> (например CNX_CH_STRIDE=1, CNX_CH_SIGMA_MODE=linear)."""
    name = "centerhead"
    out_keys = ("hm", "mask", "wh", "off")
    prob_keys = ("hm",)
    vec_keys = ("off",)

    def __init__(self, cfg: dict):
        import os
        from centerhead import config as C
        over = {k: cfg[k] for k in ("sigma_k", "sigma_min_px", "default_diam_m", "min_score") if k in cfg and k in C.DEFAULTS}
        if "max_det" in cfg and "topk" in C.DEFAULTS:
            over["topk"] = int(cfg["max_det"])
        for k, v in C.DEFAULTS.items():
            e = os.environ.get("CNX_CH_" + k.upper())
            if e is None:
                continue
            over[k] = (e.lower() in ("1", "true")) if isinstance(v, bool) else (type(v)(e) if isinstance(v, (int, float, str)) else v)
        self.cfg = C.cfg_with(**over); self.kcfg = cfg
        self.stride = int(self.cfg.get("stride", 1))
        self.circle_r_m = float(self.cfg["circle_r_m"]) if "circle_r_m" in self.cfg else None

    def build(self, ch, encoder, weights):
        from centerhead.head import build_unet_centernet
        return build_unet_centernet(ch, encoder, weights, self.cfg)

    def targets(self, objs, S):
        from centerhead.targets import render_targets_strided
        t = render_targets_strided(np.asarray(objs, np.float64).reshape(-1, 4), S, self.stride, 2, self.cfg)
        t["reg_mask"] = t["reg_mask"].astype(np.float32)
        return t

    def loss(self, out, tgt, valid):
        from centerhead.losses import center_loss, valid_to_stride
        v = valid_to_stride(valid, self.stride) if self.stride > 1 else valid.float()
        o = {k: (v_.float() if v_ is not None else None) for k, v_ in out.items() if k != "mask"}
        r = center_loss(o, {**tgt, "valid": v}, self.cfg)
        return r["loss_center"], {k: float(x) for k, x in r.items() if k != "loss_center"}

    def untta(self, o, k, fl):
        from centerhead.d8 import torch_untta_offsets
        r = {kk: untta_map(v, k, fl) for kk, v in o.items() if kk != "off" and v is not None}
        if o.get("off") is not None:
            r["off"] = torch_untta_offsets(o["off"], k, fl)
        return r

    def decode(self, maps, left, top, step, valid, suppress):
        from centerhead.decode import decode
        t = decode(maps["hm"], left, top, step, wh=maps.get("wh"), off=maps.get("off"), valid=valid & ~suppress, cfg=self.cfg)
        n = self.kcfg.get("max_det")
        if n and len(t["score"]) > n:
            o = np.argsort(-np.asarray(t["score"]), kind="stable")[:n]; t = {k: np.asarray(v)[o] for k, v in t.items()}
        return t


def get_head(name: str, cfg: dict):
    if name == "heat":
        return HeatHead(cfg)
    if name == "centerhead":
        return CenterHeadAdapter(cfg)
    mod, _, fn = name.partition(":")
    h = getattr(importlib.import_module(mod), fn or "make_head")(cfg)
    check_head(h)
    return h


def check_head(h) -> None:
    """Статическая проверка API (до обучения — ранний фейл). Динамическая — _head_selftest() ядра."""
    for a in ("name", "out_keys", "prob_keys", "vec_keys", "build", "targets", "loss", "untta", "decode"):
        if not hasattr(h, a):
            raise TypeError(f"голова {h!r}: нет атрибута {a} (HEAD_INTERFACE.md)")
    if "hm" not in h.out_keys or "mask" not in h.out_keys:
        raise TypeError("out_keys головы обязаны содержать 'hm' (B,2,S/s,S/s) и 'mask' (B,3,S,S)")
    if "hm" not in h.prob_keys or "mask" in h.prob_keys:
        raise TypeError("prob_keys: обязан содержать 'hm'; 'mask' — ключ ядра, не головы")
    s = getattr(h, "stride", 1)
    if not isinstance(s, int) or s < 1 or 384 % s or 512 % s:
        raise TypeError(f"stride головы {s!r}: целое ≥1, делитель 512 и 384 (кроп и шаг тайлов)")
