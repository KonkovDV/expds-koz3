"""Голова CenterNet и обёртка над smp.Unet bot-9. Канал-агностично: in_channels (3/5/12) задаёт энкодер."""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import DEFAULTS


def _branch(cin: int, ch: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, ch, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(ch, cout, 1))


class CenterHead(nn.Module):
    """f (B,in_ch,h,w) → {"hm": логиты (B,n_cls,h,w), "wh": (B,wh_ch,h,w) код радиуса, "off": (B,2,h,w)}."""

    def __init__(self, in_ch: int, n_cls: int = 2, head_ch: int = 64, wh: bool = True, offset: bool = True,
                 hm_bias: float = DEFAULTS["hm_bias"], wh_ch: int = 1):
        super().__init__()
        self.hm = _branch(in_ch, head_ch, n_cls)
        self.wh = _branch(in_ch, head_ch, wh_ch) if wh else None
        self.off = _branch(in_ch, head_ch, 2) if offset else None
        with torch.no_grad():
            self.hm[-1].bias.fill_(hm_bias)
            for b in (self.wh, self.off):
                if b is not None:
                    nn.init.zeros_(b[-1].bias)
                    nn.init.normal_(b[-1].weight, std=1e-3)

    def forward(self, f: torch.Tensor) -> dict:
        return {"hm": self.hm(f), "wh": self.wh(f) if self.wh is not None else None,
                "off": self.off(f) if self.off is not None else None}


class UnetCenterNet(nn.Module):
    """smp.Unet (encoder+decoder bot-9, classes=3 на маски) + CenterHead на признаке декодера шага `stride`.

    forward(x (B,C,H,W)) → {"hm" (B,2,H/s,W/s) логиты, "wh" (B,1,H/s,W/s), "off" (B,2,H/s,W/s),
    "mask" (B,3,H,W) логиты [gor, fort, arch] на полном разрешении}.
    stride=1 — признак полного разрешения (выход декодера, 16 каналов у smp по умолчанию)."""

    def __init__(self, unet: nn.Module, in_ch: int, stride: int = DEFAULTS["stride"], n_cls: int = 2,
                 head_ch: int = DEFAULTS["head_ch"], hm_bias: float = DEFAULTS["hm_bias"], probe: int = 64):
        super().__init__()
        self.encoder, self.decoder, self.mask_head = unet.encoder, unet.decoder, unet.segmentation_head
        self.stride = int(stride)
        self._feat = None
        self._block = None
        if self.stride > 1:
            blocks = self.decoder.blocks
            with torch.no_grad():                               # проба: какой блок декодера даёт шаг stride
                outs = []
                hooks = [b.register_forward_hook(lambda m, i, o, k=k: outs.append((k, o.shape))) for k, b in
                         enumerate(blocks)]
                was = self.training
                self.eval()
                self._decode(torch.zeros(1, in_ch, probe, probe))
                self.train(was)
                for h in hooks:
                    h.remove()
            hit = [(k, sh) for k, sh in outs if sh[-1] * self.stride == probe]
            if not hit:
                raise ValueError(f"нет блока декодера с шагом {stride}: {[(k, tuple(s)) for k, s in outs]}")
            self._block, sh = hit[0]
            feat_ch = int(sh[1])
            blocks[self._block].register_forward_hook(self._grab)
        else:
            feat_ch = int(self.mask_head[0].in_channels)
        self.feat_ch = feat_ch
        self.center = CenterHead(feat_ch, n_cls, head_ch, True, True, hm_bias)
        with torch.no_grad():
            conv = self.mask_head[0]
            if getattr(conv, "bias", None) is not None:
                conv.bias.fill_(hm_bias)                         # как build_model bot-9

    def _grab(self, m, i, o):
        self._feat = o

    def _decode(self, x):
        feats = self.encoder(x)
        try:
            return self.decoder(feats)                           # smp ≥ 0.5: decoder(features: list)
        except TypeError:
            return self.decoder(*feats)                          # smp ≤ 0.4

    def forward(self, x: torch.Tensor) -> dict:
        self._feat = None
        dec = self._decode(x)
        f = dec if self.stride == 1 else self._feat
        self._feat = None
        o = self.center(f)
        o["mask"] = self.mask_head(dec)
        return o


def build_unet_centernet(ch: int, encoder: str = "tu-convnext_tiny", encoder_weights=None, cfg: dict = DEFAULTS,
                         **unet_kw) -> UnetCenterNet:
    import segmentation_models_pytorch as smp
    unet = smp.Unet(encoder, encoder_weights=encoder_weights, in_channels=ch, classes=3, **unet_kw)
    return UnetCenterNet(unet, ch, cfg["stride"], 2, cfg["head_ch"], cfg["hm_bias"])
