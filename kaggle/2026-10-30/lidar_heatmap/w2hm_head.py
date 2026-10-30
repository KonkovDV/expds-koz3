"""P-W2-7 li_hm: голова-теплокарта курганов шага 1 (CenterNet-style, без wh/off) с QFL-лоссом.
Выбор только флагом: CNX_HEAD="w2hm_head:make_head" (по умолчанию ядро использует CNX_HEAD_DEFAULT — без изменений).
Отличия от centerhead (c3-sc, P-W2-5/6): шаг 1 вместо 4 (0,5 м против 2 м), Quality-Focal (β=2) по гауссиане с
нормировкой max(Σ g·valid, 50) вместо penalty-reduced focal по числу пиков, расчёт в float32 (стабильно при AMP fp16),
маска nodata. Цели/декод — как встроенная HeatHead (σ=max(1, 0,15·D) px, пики maxfilter peak_k, класс = argmax)."""
import os
import torch
from torch import nn
import cnx_head as HD


class QflHeatHead(HD.HeatHead):
    name = "li_hm"
    stride = 1
    circle_r_m = 0.5            # как у cnx/centerhead — разница только в детекторе, не в геометрии выхода

    def __init__(self, cfg):
        super().__init__(cfg)
        self.beta = float(os.environ.get("W2HM_QFL_BETA", "2.0"))
        self.norm_min = float(os.environ.get("W2HM_NORM_MIN", "50"))

    def loss(self, out, tgt, valid):
        x = out["hm"].float(); g = tgt["hm"].float().clamp(0, 1); v = valid[:, None].float()
        p = torch.sigmoid(x)
        bce = nn.functional.binary_cross_entropy_with_logits(x, g, reduction="none")
        l = bce * (g - p).abs().pow(self.beta) * v
        mass = (g * v).sum().clamp(min=self.norm_min)
        lh = l.sum() / mass
        return lh, {"qfl": float(lh.detach()), "mass": float(mass.detach())}


def make_head(cfg):
    return QflHeatHead(cfg)
