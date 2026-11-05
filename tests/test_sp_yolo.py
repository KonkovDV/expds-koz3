"""expds.sp_yolo: разбор выхода TorchScript YOLO, снятие дублей (вариант K), выключено по умолчанию. Без весов и данных."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from expds import sp_yolo as SY  # noqa: E402


class OneBox(torch.nn.Module):
    """[B, 4 + 5, 3]: якорь 0 — рамка класса 1 (cx 320, cy 160, 40×20) с уверенностью 0,9; якоря 1–2 — шум 0,005."""

    def forward(self, x):
        B = x.shape[0]
        y = torch.full((B, 9, 3), 0.005)
        y[:, 0, 0], y[:, 1, 0], y[:, 2, 0], y[:, 3, 0] = 320.0, 160.0, 40.0, 20.0
        y[:, 4 + 1, 0] = 0.9
        return y


def test_detector_decodes_and_rescales_to_window(tmp_path):
    p = tmp_path / "one.torchscript"
    torch.jit.script(OneBox()).save(str(p))
    det = SY.Detector(p, device="cpu", imgsz=640, conf_min=0.01)
    out = det.predict([np.zeros((1024, 1024, 3), np.uint8)])
    xyxy, conf, cls = out[0]
    assert len(conf) == 1 and int(cls[0]) == 1 and abs(float(conf[0]) - 0.9) < 1e-6
    k = 1024 / 640
    assert np.allclose(xyxy[0], [(320 - 20) * k, (160 - 10) * k, (320 + 20) * k, (160 + 10) * k], atol=1e-3)


def test_dedupe_against_drops_same_class_near_kept_object():
    pts = {0: [(0.8, 500000.0, 6000000.0, "EPSG:32637")], 3: [(0.7, 500100.0, 6000000.0, "EPSG:32637")]}
    fresh = SY.dedup_features(pts, "001_x", "sp_yolo")
    assert {f["class_name"] for f in fresh} == {"kurgany_povrezhdennye", "arkhitektury"}
    kur = next(f for f in fresh if f["class_name"] == "kurgany_povrezhdennye")
    anchor = {**kur, "threshold_group": "sp_focal", "confidence": 0.99}
    kept = SY.dedupe_against(fresh, [anchor], passes=lambda f: True, dist_m=8.0)
    assert [f["class_name"] for f in kept] == ["arkhitektury"]
    assert len(SY.dedupe_against(fresh, [anchor], passes=lambda f: False, dist_m=8.0)) == 2   # якорь не прошёл порог


def test_defaults_off_and_classes():
    assert SY.DEFAULTS["enabled"] is False and SY.cfg_of({})["threshold_group"] == "sp_yolo"
    assert SY.CLASSES[:2] == ["kurgany_povrezhdennye", "kurgany_tselye"]
    win = SY.windows(1500, 900, 1024, 0.25)
    assert win[0] == (0, 0) and (1500 - 1024, 0) in win
