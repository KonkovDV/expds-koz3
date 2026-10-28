"""Лоссы (torch). Фокальный CenterNet по логитам: log σ(x) = −softplus(−x), log(1−σ(x)) = −softplus(x) — без clamp,
устойчиво в fp16/AMP (считается в float32)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import DEFAULTS


def focal_loss_logits(logits: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor | None = None,
                      alpha: float = 2.0, beta: float = 4.0, pos_eps: float = 1e-4) -> torch.Tensor:
    """logits, gt: (B,C,H,W); valid: (B,H,W) или (B,1,H,W) {0,1}. Норма — число положительных (≥1)."""
    x = logits.float()
    g = gt.float()
    log_p = -F.softplus(-x)
    log_1mp = -F.softplus(x)
    p = torch.sigmoid(x)
    pos = (g >= 1.0 - pos_eps).float()
    pos_l = pos * (1 - p).pow(alpha) * log_p
    neg_l = (1 - pos) * (1 - g).clamp(min=0).pow(beta) * p.pow(alpha) * log_1mp
    l = pos_l + neg_l
    if valid is not None:
        v = valid.float()
        v = v[:, None] if v.dim() == 3 else v
        l = l * v
        pos = pos * v
    return -l.sum() / pos.sum().clamp(min=1.0)


def reg_l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """pred/target (B,K,H,W), mask (B,H,W) — L1 только в центр-пикселях, среднее по ним и K."""
    m = mask.float()[:, None]
    l = (pred.float() - target.float()).abs() * m
    return l.sum() / (m.sum() * pred.shape[1]).clamp(min=1.0)


def center_loss(out: dict, batch: dict, cfg: dict = DEFAULTS) -> dict:
    """out: {"hm": (B,2,h,w) логиты, "wh": (B,1,h,w), "off": (B,2,h,w)} (wh/off необязательны), h = H/stride;
    batch: {"hm", "wh", "off", "reg_mask", "valid"(необяз., (B,h,w) на сетке выхода)} — как render_targets_strided
    + батч-ось. → {"loss_center": тензор с градиентом (взвешенная сумма), "loss_hm", "loss_wh", "loss_off": float}."""
    valid = batch.get("valid")
    parts = {"hm": focal_loss_logits(out["hm"], batch["hm"], valid, cfg["focal_alpha"], cfg["focal_beta"])}
    total = cfg["w_hm"] * parts["hm"]
    if valid is not None and valid.dim() == 4:
        valid = valid[:, 0]
    rm = batch["reg_mask"]
    if valid is not None:
        rm = rm * valid.to(rm.dtype)
    if out.get("wh") is not None and cfg["w_wh"] > 0:
        parts["wh"] = reg_l1_loss(out["wh"], batch["wh"], rm)
        total = total + cfg["w_wh"] * parts["wh"]
    if out.get("off") is not None and cfg["w_off"] > 0:
        parts["off"] = reg_l1_loss(out["off"], batch["off"], rm)
        total = total + cfg["w_off"] * parts["off"]
    res = {"loss_center": total}
    res.update({f"loss_{k}": float(v.detach()) for k, v in parts.items()})
    return res


def valid_to_stride(valid: torch.Tensor, stride: int) -> torch.Tensor:
    """valid (B,H,W) {0,1} → (B,H/s,W/s): клетка валидна, если валиден хоть один пиксель (max-pool)."""
    import torch.nn.functional as F2
    return F2.max_pool2d(valid.float()[:, None], stride)[:, 0]
