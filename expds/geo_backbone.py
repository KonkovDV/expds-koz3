"""Замороженный предобученный бэкбон (ImageNet ResNet18 / DINOv2 ViT-S/14) как источник доп. признаков PIXGBM (bot-3, opt-in).

Вход: 3 канала [d5, d15, z] карты load_map (z — робастно нормирован), clip ±4 → x/8+0,5 → нормировка ImageNet;
сетка bb_gsd = max(gsd, 0,1 м) (etgr — среднее f×f). Карта признаков считается тайлами с полем (центральная часть тайла),
тайлы без валидных пикселей пропускаются. Пиксель карты (r, c) → билинейно из клеток шага s входной сетки.
PCA (среднее + компоненты) — на обучающих строках фолда, применяется к клеткам до интерполяции (линейно, коммутирует).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from . import geo_pixgbm as G

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
SPEC = {"B1": dict(tile=512, margin=64, stride=8, dim=384),
        "B2": dict(tile=518, margin=42, stride=14, dim=384)}
BB_GSD = 0.1
HUB = Path(os.environ.get("EXPDS_BB_HUB", "/workspace/koz3_shared/bot-3/backbone/hub"))
RESNET18_PTH = Path(os.environ.get("EXPDS_BB_RESNET18", str(Path.home() / ".cache/torch/hub/checkpoints/resnet18-f37072fd.pth")))


def bb_input(d: dict):
    """→ x (3, Hb, Wb) float32 (нормированный вход), vb (Hb, Wb) bool, f (целый шаг укрупнения)."""
    z, valid, gsd = d["z"], d["valid"], d["gsd"]
    f = max(1, int(round(max(gsd, BB_GSD) / gsd)))
    (c5, c15), b = G.trends(z, valid, gsd)
    H, W = z.shape
    ch = np.zeros((3, H, W), np.float32)
    ch[0] = np.where(valid, z - G._up(c5, b, 0, 0, H, W), 0)
    ch[1] = np.where(valid, z - G._up(c15, b, 0, 0, H, W), 0)
    ch[2] = np.where(valid, z, 0)
    vb = valid
    if f > 1:
        hb, wb = -(-H // f), -(-W // f)
        p = np.zeros((3, hb * f, wb * f), np.float32); p[:, :H, :W] = ch
        v = np.zeros((hb * f, wb * f), np.float32); v[:H, :W] = valid
        s = p.reshape(3, hb, f, wb, f).sum((2, 4)); c = v.reshape(hb, f, wb, f).sum((1, 3))
        ch = s / np.maximum(c, 1)[None]; vb = c > 0
        del p, v, s
    ch = np.clip(ch, -4, 4) / 8 + 0.5
    ch = (ch - MEAN[:, None, None]) / STD[:, None, None]
    return ch.astype(np.float32), vb, f


class Backbone:
    def __init__(self, name: str, threads: int = 2):
        import torch
        torch.set_num_threads(int(threads))
        self.name, self.spec, self.torch = name, SPEC[name], torch
        if name == "B1":
            import torchvision
            m = torchvision.models.resnet18()
            m.load_state_dict(torch.load(str(RESNET18_PTH), map_location="cpu", weights_only=True))
        elif name == "B2":
            m = torch.hub.load(str(HUB / "facebookresearch_dinov2_main"), "dinov2_vits14", source="local", pretrained=False)
            m.load_state_dict(torch.load(str(HUB / "checkpoints/dinov2_vits14_pretrain.pth"), map_location="cpu", weights_only=True))
        else:
            raise ValueError(name)
        self.m = m.eval()

    def forward(self, tile: np.ndarray) -> np.ndarray:
        """tile (3, T, T) → (C, T/s, T/s) float32."""
        torch = self.torch
        with torch.inference_mode():
            x = torch.from_numpy(tile[None])
            m = self.m
            if self.name == "B1":
                x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
                l2 = m.layer2(m.layer1(x))
                l3 = m.layer3(l2)
                l3 = torch.nn.functional.interpolate(l3, size=l2.shape[-2:], mode="bilinear", align_corners=False)
                y = torch.cat([l2, l3], 1)[0]
            else:
                t = m.forward_features(x)["x_norm_patchtokens"][0]
                n = tile.shape[-1] // 14
                y = t.T.reshape(-1, tile.shape[-2] // 14, n)
            return y.numpy()


def bb_map(x: np.ndarray, vb: np.ndarray, bk: Backbone) -> np.ndarray:
    """Карта признаков (C, hc, wc) float16 на шаге s входной сетки; невычисленные клетки — 0."""
    sp = bk.spec
    T, mg, s, C = sp["tile"], sp["margin"], sp["stride"], sp["dim"]
    core = T - 2 * mg
    _, Hb, Wb = x.shape
    hc, wc = -(-Hb // s), -(-Wb // s)
    F = np.zeros((C, hc, wc), np.float16)
    pad = (0.5 - MEAN) / STD          # нормированный «0» (как вне съёмки)
    for ty in range(0, Hb, core):
        for tx in range(0, Wb, core):
            if not vb[max(ty - s, 0):ty + core + s, max(tx - s, 0):tx + core + s].any():   # +1 клетка: билинейка у края
                continue
            tile = np.empty((3, T, T), np.float32); tile[:] = pad[:, None, None]
            y0, x0 = ty - mg, tx - mg
            sy0, sx0 = max(y0, 0), max(x0, 0)
            sy1, sx1 = min(y0 + T, Hb), min(x0 + T, Wb)
            tile[:, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = x[:, sy0:sy1, sx0:sx1]
            out = bk.forward(tile)
            k0, k1 = mg // s, (mg + core) // s
            gy, gx = ty // s, tx // s
            ny, nx = min(k1 - k0, hc - gy), min(k1 - k0, wc - gx)
            F[:, gy:gy + ny, gx:gx + nx] = out[:, k0:k0 + ny, k0:k0 + nx]
    return F


def sample(F: np.ndarray, rows: np.ndarray, cols: np.ndarray, f: int, s: int) -> np.ndarray:
    """Билинейно F (C, hc, wc) в центрах пикселей карты (rows, cols) → (N, C) float32."""
    C, hc, wc = F.shape
    u = np.clip((rows + 0.5) / f / s - 0.5, 0, hc - 1)
    v = np.clip((cols + 0.5) / f / s - 0.5, 0, wc - 1)
    y0 = np.floor(u).astype(np.int64); x0 = np.floor(v).astype(np.int64)
    y1 = np.minimum(y0 + 1, hc - 1); x1 = np.minimum(x0 + 1, wc - 1)
    wy = (u - y0).astype(np.float32); wx = (v - x0).astype(np.float32)
    out = (F[:, y0, x0].astype(np.float32) * ((1 - wy) * (1 - wx))
           + F[:, y0, x1].astype(np.float32) * ((1 - wy) * wx)
           + F[:, y1, x0].astype(np.float32) * (wy * (1 - wx))
           + F[:, y1, x1].astype(np.float32) * (wy * wx))
    return np.ascontiguousarray(out.T)


def project(F: np.ndarray, mean: np.ndarray, comps: np.ndarray, rows_per: int = 64) -> np.ndarray:
    """PCA клеток: (C, hc, wc) → (k, hc, wc) float32, по полосам (память)."""
    C, hc, wc = F.shape
    k = comps.shape[1]
    out = np.zeros((k, hc, wc), np.float32)
    for y in range(0, hc, rows_per):
        blk = F[:, y:y + rows_per].astype(np.float32).reshape(C, -1).T
        out[:, y:y + rows_per] = ((blk - mean) @ comps).T.reshape(k, -1, wc)
    return out
