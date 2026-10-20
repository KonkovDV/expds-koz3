"""Инференс U-Net скользящим окном и превращение вероятностей в объекты.

Вероятности по классам → связные области выше нижнего порога → разделение
слипшихся курганов по пикам вероятности → полигоны с уверенностью = максимум
вероятности в объекте. Окончательный отбор — порогами по классам (post).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from . import accel
from .raster import mask_to_polygons

log = logging.getLogger(__name__)

#: Радиус разделения экземпляров по пикам (м); 0 — не делить (крупные объекты).
SPLIT_RADIUS_M = {"kurgany_povrezhdennye": 4.0, "kurgany_tselye": 4.0, "arkhitektury": 0.0,
                  "gorodishcha": 0.0, "fortifikatsii": 0.0, "object_poly": 0.0, "finds_points": 1.0}


def fp16_ok(device) -> bool:
    """FP16 (AMP) только на CUDA и не на GTX 16xx: у этих карт половинная точность в cuDNN даёт NaN."""
    import torch

    return device.type == "cuda" and "GTX 16" not in torch.cuda.get_device_name(device)


class Segmenter:
    """Загруженная модель из чекпойнта expds.train."""

    def __init__(self, path: str | Path, device: str | None = None):
        import torch
        import segmentation_models_pytorch as smp

        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.meta = {k: v for k, v in ck.items() if k != "state_dict"}
        self.classes: list[str] = ck["classes"]
        self.res: float = ck["res"]
        self.tile: int = ck["tile"]
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = smp.Unet(encoder_name=ck["encoder"], encoder_weights=None,
                              in_channels=ck["in_channels"], classes=len(self.classes))
        self.model.load_state_dict(ck["state_dict"])
        self.model.to(self.device).eval()
        self.amp = fp16_ok(self.device)

    def predict(self, img: np.ndarray, overlap: float = 0.25, batch: int = 16) -> np.ndarray:
        """img: C×H×W uint8 на рабочем шаге модели → вероятности K×H×W float32."""
        import torch

        c, h, w = img.shape
        t = self.tile
        step = max(1, int(t * (1 - overlap)))
        ys = list(range(0, max(h - t, 0) + 1, step)) or [0]
        xs = list(range(0, max(w - t, 0) + 1, step)) or [0]
        if ys[-1] + t < h:
            ys.append(h - t)
        if xs[-1] + t < w:
            xs.append(w - t)
        pad_h, pad_w = max(0, t - h), max(0, t - w)
        x_all = np.pad(img, ((0, 0), (0, pad_h), (0, pad_w)))
        ramp = np.minimum(np.arange(t) + 1, np.arange(t)[::-1] + 1).astype(np.float32)
        weight = np.minimum.outer(ramp, ramp)
        weight /= weight.max()
        coords = [(y, x) for y in ys for x in xs]
        need = x_all.nbytes + (len(self.classes) + 1) * x_all.shape[1] * x_all.shape[2] * 4
        if self.device.type == "cuda" and need < 0.6 * torch.cuda.mem_get_info(self.device)[0]:
            return self._predict_on_device(x_all, coords, weight, batch)[:, :h, :w]
        acc = np.zeros((len(self.classes), x_all.shape[1], x_all.shape[2]), np.float32)
        wsum = np.zeros(x_all.shape[1:], np.float32)
        with torch.inference_mode():
            for s in range(0, len(coords), batch):
                chunk = coords[s:s + batch]
                xb = np.stack([x_all[:, y:y + t, x:x + t] for y, x in chunk]).astype(np.float32)
                xb = torch.from_numpy((xb / 255.0 - 0.5) / 0.25).to(self.device)
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
                    pr = self.model(xb).float().sigmoid().cpu().numpy()
                for (y, x), p in zip(chunk, pr):
                    acc[:, y:y + t, x:x + t] += p * weight
                    wsum[y:y + t, x:x + t] += weight
        return (acc / np.maximum(wsum, 1e-6))[:, :h, :w]

    def _predict_on_device(self, x_all: np.ndarray, coords: list[tuple[int, int]], weight: np.ndarray,
                           batch: int) -> np.ndarray:
        """Как predict, но снимок, окна и накопление — в памяти устройства; на CPU — только итог."""
        import torch

        t = self.tile
        dev = self.device
        x = torch.from_numpy(x_all).to(dev)
        acc = torch.zeros((len(self.classes), *x.shape[1:]), device=dev)
        wsum = torch.zeros(x.shape[1:], device=dev)
        wt = torch.from_numpy(weight).to(dev)
        with torch.inference_mode():
            for s in range(0, len(coords), batch):
                chunk = coords[s:s + batch]
                xb = torch.stack([x[:, y:y + t, x0:x0 + t] for y, x0 in chunk]).float()
                xb = xb.div_(255.0).sub_(0.5).div_(0.25)
                with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=self.amp):
                    pr = self.model(xb).float().sigmoid()
                for (y, x0), p in zip(chunk, pr):
                    acc[:, y:y + t, x0:x0 + t] += p * wt
                    wsum[y:y + t, x0:x0 + t] += wt
        return (acc / wsum.clamp_min(1e-6)).cpu().numpy()


def split_by_peaks(prob: np.ndarray, mask: np.ndarray, radius_px: int, min_peak: float) -> np.ndarray:
    """Экземпляры: связные области маски, разделённые по ближайшему пику вероятности."""
    from scipy.ndimage import distance_transform_edt, find_objects, label

    lab, n = label(mask)
    if n == 0 or radius_px <= 0:
        return lab
    rows, cols, _ = accel.find_peaks(np.where(mask, prob, np.nan).astype(np.float32), radius_px, min_peak)
    if len(rows) < 2:
        return lab
    peaks_of: dict[int, list[tuple[int, int]]] = {}
    for r, c in zip(rows, cols):
        peaks_of.setdefault(int(lab[r, c]), []).append((int(r), int(c)))
    inst = lab.copy()
    next_id = n + 1
    for region, sl in enumerate(find_objects(lab), start=1):
        pk = peaks_of.get(region, [])
        if sl is None or len(pk) < 2:
            continue  # область с одним пиком (или без пиков) остаётся целой
        sub = lab[sl] == region
        markers = np.zeros(sub.shape, np.int32)
        for k, (r, c) in enumerate(pk, start=1):
            markers[r - sl[0].start, c - sl[1].start] = k
        _, (ir, ic) = distance_transform_edt(markers == 0, return_indices=True)
        nearest = markers[ir, ic]
        block = inst[sl]
        block[sub] = nearest[sub] + next_id - 1
        next_id += len(pk)
    return inst


def probs_to_features(probs: np.ndarray, classes: list[str], transform, crs: str | None, res: float,
                      region: str, markup: str, source: str, low: float = 0.05,
                      min_area_px: int = 3) -> list[dict]:
    """Вероятности K×H×W → объекты пайплайна (полигоны EPSG:3857 с уверенностью)."""
    out = []
    for k, cls in enumerate(classes):
        p = probs[k]
        mask = p >= low
        if not mask.any():
            continue
        r = SPLIT_RADIUS_M.get(cls, 0.0)
        inst = split_by_peaks(p, mask, int(round(r / res)) if r else 0, low * 2)
        n = int(inst.max())
        if n == 0:
            continue
        sizes = np.bincount(inst.ravel(), minlength=n + 1)
        peak = np.zeros(n + 1, np.float32)
        np.maximum.at(peak, inst.ravel(), p.ravel())
        inst = np.where(sizes[inst] >= min_area_px, inst, 0)
        for poly, lab in mask_to_polygons(inst, transform, crs, simplify=res * 0.5):
            out.append({"geom": poly, "class_name": cls, "confidence": float(peak[lab]), "region_name": region,
                        "markup_type": markup, "source": source, "original_crs": crs})
    return out
