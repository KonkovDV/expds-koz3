"""Загрузчик весов bot-9 convnext_lidar в формате контракта bot-8 (bot-9/CNX_CONTRACT_FOR_BOT8.md).

Зависимости: torch, segmentation_models_pytorch==0.6.0, timm==1.0.29 (сеть строится офлайн, encoder_weights=None).
Код свой (bot-9); CnxUnetCenter повторяет граф bot-2 centerhead.head.UnetCenterNet @788202a (те же имена параметров),
равенство выходов проверено tests/test_contract.py.

    c = load_contract("weights_cnx_c5-sc_f0.pt")          # torch.load(weights_only=True)
    net = build_from_arch(c["arch"]); net.load_state_dict({k: v.float() for k, v in c["state_dict"].items()}); net.eval()
    flat = CnxFlat(net, c)                                 # (B,C,H,W) → (B,5,H,W) логиты [hm_t, hm_p, gor, fort, arch] на полном разрешении
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_contract(path, map_location="cpu"):
    c = torch.load(path, map_location=map_location, weights_only=True)
    assert c.get("format") == "koz3-weights-contract-bot8", c.get("format")
    return c


def _branch(cin, ch, cout):
    return nn.Sequential(nn.Conv2d(cin, ch, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(ch, cout, 1))


class _Center(nn.Module):
    def __init__(self, cin, n_cls, head_ch):
        super().__init__()
        self.hm, self.wh, self.off = _branch(cin, head_ch, n_cls), _branch(cin, head_ch, 1), _branch(cin, head_ch, 2)


class CnxUnetCenter(nn.Module):
    """smp.Unet (encoder/decoder/segmentation_head=mask_head, classes=3) + CenterNet-ветки на выходе decoder.blocks[feat_block].
    forward → dict hm (B,2,H/s,W/s), wh (B,1,..), off (B,2,..) [сырые], mask (B,3,H,W) [логиты]."""

    def __init__(self, encoder_name="tu-convnext_tiny", encoder_weights=None, in_channels=3, classes=3, stride=4,
                 feat_block=2, feat_ch=64, n_cls=2, head_ch=64):
        super().__init__()
        import segmentation_models_pytorch as smp
        u = smp.Unet(encoder_name, encoder_weights=encoder_weights, in_channels=in_channels, classes=classes)
        self.encoder, self.decoder, self.mask_head = u.encoder, u.decoder, u.segmentation_head
        self.stride, self.feat_block = int(stride), feat_block
        self._feat = None
        if self.stride > 1:
            self.decoder.blocks[feat_block].register_forward_hook(lambda m, i, o: setattr(self, "_feat", o))
        self.center = _Center(feat_ch, n_cls, head_ch)

    def forward(self, x):
        self._feat = None
        dec = self.decoder(self.encoder(x))
        f = dec if self.stride == 1 else self._feat
        self._feat = None
        c = self.center
        return {"hm": c.hm(f), "wh": c.wh(f), "off": c.off(f), "mask": self.mask_head(dec)}


def build_from_arch(arch: dict) -> nn.Module:
    name, kw = arch["name"], dict(arch["kwargs"])
    if name == "cnx_contract.CnxUnetCenter":
        return CnxUnetCenter(**kw)
    if name == "segmentation_models_pytorch.Unet":
        import segmentation_models_pytorch as smp
        return smp.Unet(**kw)
    raise ValueError(f"неизвестная arch {name}")


class CnxFlat(nn.Module):
    """Адаптер под tiling bot-8 (один тензор, sigmoid снаружи): (B,5,H,W) логиты [hm_t, hm_p, mask_gor, mask_fort, mask_arch];
    hm шага s повторяется nearest до полного разрешения (wh/off отбрасываются: центр — по клетке s·0,5 м)."""

    def __init__(self, net: nn.Module, contract: dict):
        super().__init__()
        self.net = net
        self.s = int(contract["out_stride"][0])

    def forward(self, x):
        o = self.net(x)
        if torch.is_tensor(o):
            return o
        hm = o["hm"] if self.s == 1 else F.interpolate(o["hm"], scale_factor=self.s, mode="nearest")
        return torch.cat([hm, o["mask"]], 1)
