"""expds.lidar_cnx_head: курганы cnx «как в ядре» (поправка 1 PREREG c5-in). Синтетика, без весов и данных."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from expds import lidar_cnx as LC  # noqa: E402
from expds import lidar_cnx_head as LH  # noqa: E402

CONTRACT = {
    "out_stride": [4, 4, 4, 4, 4, 1, 1, 1],
    "thresholds": {"mask_thr": 0.5, "circle_r_m": 0.5, "max_det": 20000, "head_min_score": 0.05,
                   "head_peak_pool": 3, "head_topk": 20000, "head_nms_min_m": 1.0, "head_nms_frac": 0.8,
                   "head_nms_cross_class": False, "head_size_mode": "log_r"},
    "infer": {"tile": 64, "stride": 48, "tta": 8},
}


class PeakNet(torch.nn.Module):
    """hm: гауссов пик класса `cls` в пикселе входа (py, px) — инвариантно к D8 (пик задан во входе); маски — 0."""

    def __init__(self, cls: int, gor: bool = False):
        super().__init__()
        self.cls, self.gor = cls, gor

    def forward(self, x):
        B, C, H, W = x.shape
        s = 4
        # «пик» там, где канал 0 входа максимален (так D8 входа корректно отражается в выходе)
        a = x[:, 0]
        hm = torch.full((B, 2, H // s, W // s), -6.0)
        pooled = torch.nn.functional.avg_pool2d(a[:, None], s)[:, 0]
        hm[:, self.cls] = -6.0 + 12.0 * (pooled > 0.9).float()
        mask = torch.full((B, 3, H, W), -6.0)
        if self.gor:
            mask[:, 0] = 6.0
        return {"hm": hm, "wh": torch.zeros((B, 1, H // s, W // s)), "off": torch.zeros((B, 2, H // s, W // s)),
                "mask": mask}


class FakeSeg:
    def __init__(self, net):
        self.nets = [type("Flat", (), {"net": net})()]
        self.device = torch.device("cpu")
        self.amp = False


def _input(H=100, W=90, py=42, px=54):
    x = np.full((5, H, W), -0.5, np.float32)
    x[0, py - 2:py + 2, px - 2:px + 2] = 1.0          # блок 4×4 по сетке шага 4 (строки 40–43, столбцы 52–55) → клетка (10, 13)
    valid = np.ones((H, W), bool)
    return x, valid


def test_tile_origins_match_centerhead():
    from centerhead.decode import tile_origins as ch_tiles
    for H, W in ((100, 90), (512, 700), (1300, 513)):
        assert LH.tile_origins(H, W, 512, 384, 4) == ch_tiles(H, W, 512, 384, 4)


def test_params_from_contract_and_missing_keys():
    p = LH.params(CONTRACT)
    assert p["tile"] == 64 and p["step"] == 48 and p["tta"] == 8 and p["stride"] == 4
    assert p["cfg"]["min_score"] == 0.05 and p["cfg"]["nms_cross_class"] is False and p["circle_r_m"] == 0.5
    bad = {**CONTRACT, "thresholds": {k: v for k, v in CONTRACT["thresholds"].items() if k != "head_peak_pool"}}
    with pytest.raises(LH.HeadError):
        LH.params(bad)


@pytest.mark.parametrize("cls", [0, 1])
def test_peak_decoded_at_right_place_and_class_with_d8(cls):
    x, valid = _input()
    p = LH.params(CONTRACT)
    maps = LH.predict_maps(FakeSeg(PeakNet(cls)), x, valid, p)
    assert maps["hm"].shape == (2, 25, 23) and maps["mask"].shape == (3, 100, 90)
    geo = {"left": 1000.0, "top": 2000.0, "step": 0.5, "crs": "EPSG:32637"}
    dets = LH.decode_kurgans(maps, geo, p)
    assert len(dets) == 1
    X, Y, c, s = dets[0]
    assert c == cls and s > 0.9
    # клетка головы (10, 13), шаг клетки 4 × 0,5 = 2 м: центр (13,5·2; 10,5·2) м от угла
    assert abs(X - (1000.0 + 13.5 * 2.0)) < 1e-6 and abs(Y - (2000.0 - 10.5 * 2.0)) < 1e-6
    feats = LH.to_features(dets, geo, p, "001_Участок", "a.las")
    assert feats[0]["class_name"] == LH.HM_CLASSES[cls] and feats[0]["markup_type"] == "li"
    assert abs(feats[0]["geom"].area - np.pi * 0.25) / (np.pi * 0.25) < 0.2     # круг 0,5 м в EPSG:3857 (масштаб)


def test_kurgans_inside_hillfort_mask_are_suppressed():
    x, valid = _input()
    p = LH.params(CONTRACT)
    maps = LH.predict_maps(FakeSeg(PeakNet(1, gor=True)), x, valid, p)
    assert LH.decode_kurgans(maps, {"left": 0.0, "top": 0.0, "step": 0.5, "crs": "EPSG:32637"}, p) == []


def test_outside_survey_has_no_detections():
    x, valid = _input()
    valid[:] = False
    p = LH.params(CONTRACT)
    maps = LH.predict_maps(FakeSeg(PeakNet(0)), x, valid, p)
    assert not maps["_valid"].any() and float(maps["hm"].max()) == 0.0


def test_replace_predicate_and_defaults():
    assert LC.DEFAULTS["decode"] == "rescore" and LC.DEFAULTS["replace_li_kurgans"] is False
    r = "001_Участок"
    li = {"region_name": r, "class_name": "kurgany_tselye", "markup_type": "li"}
    coarse = {**li, "threshold_group": "li_coarse"}
    dev = {**li, "threshold_group": "li_dev"}                      # σ — не снимается
    cnx = {**li, "threshold_group": "li_cnx"}
    other_site = {**li, "region_name": "002_Другой"}
    gor = {**li, "class_name": "gorodishcha"}
    sp = {**li, "markup_type": "sp_or"}
    assert LC.is_li_kurgan(li, r) and LC.is_li_kurgan(coarse, r)
    assert not any(LC.is_li_kurgan(f, r) for f in (dev, cnx, other_site, gor, sp))


def test_build_packs_centerhead_only_for_head_decode():
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("build_submission_t", root / "scripts" / "build_submission.py")
    bs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bs)
    on = {"li_cnx": {"enabled": True, "weights": ["models/cnx/w.pt"], "decode": "head"}}
    assert bs.cnx_head_on(on)
    assert not bs.cnx_head_on({"li_cnx": {"enabled": True, "weights": ["models/cnx/w.pt"]}})          # rescore
    assert not bs.cnx_head_on({"li_cnx": {"enabled": False, "weights": ["x"], "decode": "head"}})
    names = {p.name for p in bs.centerhead_files()}
    assert {"__init__.py", "config.py", "d8.py", "decode.py"} <= names and not any("test" in n for n in names)
