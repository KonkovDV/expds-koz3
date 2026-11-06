"""Флаги [orsp_tta]: выключены по умолчанию; U-Net спутника и D-FINE не смешиваются."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from expds import orsp_tta, pipeline

ROOT = Path(__file__).resolve().parents[1]


def _cfg(**over):
    base = {"orsp_tta": {"enabled": False, "unet_tta": 8, "unet_scales": [0.75, 1.0, 1.25],
                         "dfine_views": "d8", "dfine_gsd": [0.75, 1.0, 1.25]}}
    base["orsp_tta"].update(over)
    return base


def test_default_toml_flag_off():
    cfg = pipeline.load_config(ROOT / "configs" / "default.toml")
    st = orsp_tta.settings(cfg)
    assert st["enabled"] is False
    assert orsp_tta.dfine_plan(cfg) is None
    assert orsp_tta.cost(cfg)["unet_factor"] == 1
    assert orsp_tta.cost(cfg)["dfine_eval"] == "не оценено"


def test_disabled_ignores_knobs():
    cfg = _cfg()
    assert orsp_tta.apply_unet(cfg, "sp_or", 1, [1.0]) == (1, [1.0])
    assert orsp_tta.dfine_plan(cfg) is None


def test_unet_only_does_not_arm_dfine():
    cfg = _cfg(enabled=True, dfine_views="off", dfine_gsd=[1.0], unet_tta=4, unet_scales=[0.75, 1.0, 1.25])
    assert orsp_tta.dfine_plan(cfg) is None
    assert orsp_tta.apply_unet(cfg, "sp_or", 1, [1.0]) == (4, [0.75, 1.0, 1.25])
    assert orsp_tta.apply_unet(cfg, "sp_focal2", 1, [1.0])[0] == 4
    # аэрофото и ортофото не подменяются
    assert orsp_tta.apply_unet(cfg, "ae", 1, [1.0]) == (1, [1.0])
    assert orsp_tta.apply_unet(cfg, "or", 1, [1.0]) == (1, [1.0])
    assert orsp_tta.cost(cfg)["unet_factor"] == 12


def test_dfine_plan_marks_unevaluated():
    cfg = _cfg(enabled=True, dfine_views="d8", dfine_gsd=[1.0], unet_tta=0, unet_scales=[1.0])
    plan = orsp_tta.dfine_plan(cfg)
    assert plan is not None and len(plan["views"]) == 8
    assert orsp_tta.DFINE_EVAL == "не оценено"
    # d8 заменяет штатный flip (2 прохода), а не умножает его: 8/2 = 4
    assert orsp_tta.cost(cfg)["dfine_factor_vs_flip_tta"] == 4


def test_wbf_merges_same_class_only():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [0, 0, 10, 10], [100, 100, 110, 110]], dtype=np.float64)
    scores = np.array([0.9, 0.8, 0.5, 0.7])
    labels = np.array([1, 1, 2, 1])
    B, S, L = orsp_tta.wbf(boxes, scores, labels, n_views=4, iou=0.5)
    assert sorted(L.tolist()) == [1, 1, 2]
    # класс 2 — один вид из четырёх
    assert S[L == 2][0] == pytest.approx(0.5 / 4)
    # дальний бокс класса 1 не слился
    assert len(L) == 3


def test_fuse_does_not_mix_separate_calls():
    a = (np.array([[0.0, 0, 10, 10]]), np.array([0.9]), np.array([0]))
    b = (np.array([[0.0, 0, 10, 10]]), np.array([0.9]), np.array([0]))
    empty = (np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int))
    _, s1, _ = orsp_tta.fuse_views([a], "wbf", 0.55, 0.5)
    two, s2, _ = orsp_tta.fuse_views([a, b], "wbf", 0.55, 0.5)
    _, s_solo, _ = orsp_tta.fuse_views([a, empty], "wbf", 0.55, 0.5)
    assert float(s1[0]) == pytest.approx(0.9)          # один вид не делится
    assert len(two) == 1 and float(s2[0]) == pytest.approx(0.9)  # виден на обоих — без штрафа
    assert float(s_solo[0]) == pytest.approx(0.45)     # один вид из двух — уверенность / 2


def test_boxes_to_base_scales_gsd():
    class T:
        def __init__(self, a, e, c, f):
            self.a, self.b, self.c = a, 0.0, c
            self.d, self.e, self.f = 0.0, e, f

    base = T(0.5, -0.5, 100.0, 200.0)
    coarse = T(1.0, -1.0, 100.0, 200.0)  # вдвое грубее
    out = orsp_tta.boxes_to_base([[0, 0, 10, 10]], coarse, base)
    assert out[0] == pytest.approx([0, 0, 20, 20])


def test_view_inverse_matches_torch_rot90():
    torch = pytest.importorskip("torch")
    img = torch.zeros(1, 1, 32, 32)
    img[0, 0, 5, 8] = 1.0
    for rot in range(4):
        for hflip in (False, True):
            t = torch.rot90(img, rot, dims=(2, 3)) if rot else img
            if hflip:
                t = t.flip(3)
            ys, xs = torch.where(t[0, 0] > 0.5)
            xi, yi = orsp_tta._inv_point(float(xs[0]) + 0.5, float(ys[0]) + 0.5, rot, hflip, 32.0)
            assert abs(xi - 8.5) < 1e-4 and abs(yi - 5.5) < 1e-4


def test_cached_segmenters_respects_flag(monkeypatch, tmp_path):
    weight = tmp_path / "m.pt"
    weight.write_bytes(b"not-a-real-weight")

    class Seg:
        def __init__(self, path):
            self.path = path
            self.tta = None
            self.ms_scales = None
            self.classes = ["gorodishcha"]
            self.meta = {}

    monkeypatch.setattr("expds.infer_nn.Segmenter", Seg)
    models = {"sp_or": str(weight), "ae": str(weight)}
    off = pipeline._cached_segmenters({}, {"models": models, "orsp_tta": {"enabled": False, "unet_tta": 8,
                                        "unet_scales": [0.5, 2.0]}}, "sp_or", tmp_path)
    assert off[0].tta == 1 and off[0].ms_scales == [1.0]
    on = pipeline._cached_segmenters({}, {"models": models, "orsp_tta": {"enabled": True, "unet_tta": 4,
                                       "unet_scales": [0.75, 1.0, 1.25]}}, "sp_or", tmp_path)
    assert on[0].tta == 4 and on[0].ms_scales == [0.75, 1.0, 1.25]
    ae = pipeline._cached_segmenters({}, {"models": models, "orsp_tta": {"enabled": True, "unet_tta": 8,
                                      "unet_scales": [0.5, 2.0]}}, "ae", tmp_path)
    assert ae[0].tta == 1 and ae[0].ms_scales == [1.0]


def test_request_refuses_without_data():
    import importlib.util

    spec = importlib.util.spec_from_file_location("orsp_req", ROOT / "scripts" / "orsp_tta_honest10.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    text = mod.request_text()
    assert "не оценено" in text
    assert "tta_multiscale" in text and "baseline" in text
    assert "не проверены система координат и порог" in text
    assert mod.main(["--request"]) == 0
    assert mod.main(["--root", str(ROOT / "нет_такого_корня"), "--work", str(ROOT / "нет_такого_work")]) == 3


def test_compliance_report_lists_six_checks(tmp_path):
    import importlib.util

    script = ROOT / "scripts" / "check_compliance.py"
    spec = importlib.util.spec_from_file_location("b8cc", script)
    cc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cc)
    ring = [[4_200_000.0, 7_500_000.0], [4_200_050.0, 7_500_000.0],
            [4_200_050.0, 7_500_050.0], [4_200_000.0, 7_500_050.0], [4_200_000.0, 7_500_000.0]]
    fc = {"type": "FeatureCollection", "features": [{
        "type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]},
        "properties": {"class_name": "kurgany_tselye", "crs": "EPSG:3857", "confidence": 0.5},
    }]}
    raw = json.dumps(fc)
    p = tmp_path / "result.geojson"
    p.write_text(raw, encoding="utf-8")
    loaded, text, size = cc._load(p)
    rep = cc.check(loaded, text, size)
    assert rep["ok"], rep["errors"]
    text_out = cc.format_report(rep)
    for name in ("size", "features", "vertices", "remarks", "crs", "polygon"):
        assert f"[PASS] {name}" in text_out
    assert text_out.strip().splitlines()[-1] == "ИТОГ PASS" or "ИТОГ PASS" in text_out
    bare = json.loads(raw)
    bare["features"][0]["properties"].pop("crs")
    bad = cc.check(bare, json.dumps(bare), size)
    assert not bad["ok"]
    assert bad["summary"]["checks"]["crs"] is False
