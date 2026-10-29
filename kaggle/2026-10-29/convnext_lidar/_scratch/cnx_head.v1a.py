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


class Dec2Dict(nn.Module):
    """smp.Unet(classes=2+3) → dict. Голова "heat" = штатная segmentation_head smp (Conv3×3 16→5) на выходе декодера."""

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
    out_keys = ("hm", "mask")
    prob_keys = ("hm", "mask")
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
    """Адаптер пакета bot-2 `centerhead` (bot-2/repo/centerhead, вендорится как vendor/centerhead):
    UnetCenterNet (CenterHead на выходе декодера + seg-голова smp на 3 маски), render_targets (hm/size/off/reg_mask),
    center_loss (focal + L1 size/off), decode (пики pool×pool + NMS по расстоянию). Конфиг — centerhead.config.DEFAULTS,
    поверх — общие с ядром ключи (sigma_k, sigma_min_px, default_diam_m, min_score) и env CNX_CH_* (см. HEAD_INTERFACE.md)."""
    name = "centerhead"
    out_keys = ("hm", "mask", "size", "off")
    prob_keys = ("hm", "mask")
    vec_keys = ("off",)

    def __init__(self, cfg: dict):
        import os
        from centerhead import config as C
        over = {k: cfg[k] for k in ("sigma_k", "sigma_min_px", "default_diam_m", "min_score") if k in cfg}
        for k, v in C.DEFAULTS.items():
            e = os.environ.get("CNX_CH_" + k.upper())
            if e is not None:
                over[k] = type(v)(e) if not isinstance(v, (bool, tuple)) else (e.lower() in ("1", "true") if isinstance(v, bool) else v)
        over["topk"] = int(cfg.get("max_det", C.DEFAULTS["topk"]))
        self.cfg = C.cfg_with(**over); self.kcfg = cfg

    def build(self, ch, encoder, weights):
        from centerhead.head import build_unet_centernet
        return build_unet_centernet(ch, encoder, weights, self.cfg)

    def targets(self, objs, S):
        from centerhead.targets import render_targets
        return render_targets(np.asarray(objs, np.float64).reshape(-1, 4), S, S, 2, self.cfg)

    def loss(self, out, tgt, valid):
        from centerhead.losses import center_loss
        return center_loss(out, {**tgt, "valid": valid}, self.cfg)

    def untta(self, o, k, fl):
        from centerhead.d8 import torch_untta_offsets
        r = {kk: untta_map(v, k, fl) for kk, v in o.items() if kk != "off" and v is not None}
        if o.get("off") is not None:
            r["off"] = torch_untta_offsets(o["off"], k, fl)
        return r

    def decode(self, maps, left, top, step, valid, suppress):
        from centerhead.decode import decode
        t = decode(maps["hm"], left, top, step, maps.get("size"), maps.get("off"), valid & ~suppress, self.cfg)
        n = self.kcfg.get("max_det")
        if n and len(t["score"]) > n:
            o = np.argsort(-t["score"], kind="stable")[:n]; t = {k: np.asarray(v)[o] for k, v in t.items()}
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
    """Статическая проверка API (до обучения — ранний фейл)."""
    for a in ("name", "out_keys", "prob_keys", "vec_keys", "build", "targets", "loss", "untta", "decode"):
        if not hasattr(h, a):
            raise TypeError(f"голова {h!r}: нет атрибута {a} (HEAD_INTERFACE.md)")
    if "hm" not in h.out_keys or "mask" not in h.out_keys:
        raise TypeError("out_keys головы обязаны содержать 'hm' (B,2,H,W) и 'mask' (B,3,H,W)")
