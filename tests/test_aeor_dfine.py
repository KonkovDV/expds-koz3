"""bot-1 (кандидат v4): D-FINE всех классов на Ae/Or — флаг выкл., WBF внутри снимка, strict preflight, лимиты."""
import tomllib
from pathlib import Path

import numpy as np
import pytest
from affine import Affine

from expds import aeor_dfine as A

ROOT = Path(__file__).resolve().parents[1]


def _cfg():
    return tomllib.loads((ROOT / "configs" / "default.toml").read_text(encoding="utf-8"))


def test_default_off():
    cfg = _cfg()
    assert cfg["aeor_dfine"] == {"enabled": False}
    assert not A.enabled(cfg)
    st = A.settings(cfg)
    assert st["enabled"] is False and st["markups"] == ["ae", "or"]
    assert st["weights"] == {"ae": [], "or": []}


def test_default_config_has_no_dfine_thresholds():
    cfg = _cfg()
    tb = cfg.get("thresholds_by_markup", {})
    assert "ae_dfine" not in tb and "or_dfine" not in tb   # включение — только после honest-10 приёмки


def test_settings_parsing():
    st = A.settings({"aeor_dfine": {"weights": ["m/x"], "res_m": 0.3, "scales": [1.0, 1.5]}})
    assert st["weights"] == {"ae": ["m/x"], "or": ["m/x"]}
    assert st["res_m"] == {"ae": 0.3, "or": 0.3} and st["scales"] == [1.0, 1.5]
    st = A.settings({"aeor_dfine": {"weights": {"or": ["m/o"]}, "res_m": {"or": 0.2}}})
    assert st["weights"] == {"ae": [], "or": ["m/o"]} and st["res_m"] == {"ae": 0.5, "or": 0.2}


def test_thresholds_ready():
    ok, miss = A.thresholds_ready({}, "ae")
    assert not ok and set(miss) == set(A.CLASSES)
    cfg = {"thresholds_by_markup": {"or_dfine": {c: 0.5 for c in A.CLASSES}}}
    assert A.thresholds_ready(cfg, "or")[0] and not A.thresholds_ready(cfg, "ae")[0]


def test_preflight_strict_raises(tmp_path):
    with pytest.raises(A.AeOrUnavailable):
        A.preflight({"aeor_dfine": {"enabled": True}}, tmp_path)
    info = A.preflight({"aeor_dfine": {"enabled": True, "strict": False}}, tmp_path)
    assert info["problems"]


def test_wbf_merges_views_and_penalizes_missing():
    B = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], float)
    L = np.array([0.9, 0.7, 0.8])
    C = np.array([2, 2, 2])
    b, s, c = A.wbf(B, L, C, n_views=2, iou=0.55)
    assert len(s) == 2
    i = int(np.argmax(s))
    assert s[i] == pytest.approx(0.8) and b[i][0] == pytest.approx((0 * 0.9 + 1 * 0.7) / 1.6)
    assert sorted(s)[0] == pytest.approx(0.4)   # один вид из двух → 0.8 / 2


def test_wbf_keeps_classes_apart():
    B = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], float)
    b, s, c = A.wbf(B, np.array([0.9, 0.8]), np.array([0, 1]), n_views=1)
    assert sorted(c.tolist()) == [0, 1]


def test_fuse_views_single_view_passthrough():
    v = (np.array([[0, 0, 4, 4]], float), np.array([0.5]), np.array([1]))
    b, s, c = A.fuse_views([v], A.settings({}))
    assert s.tolist() == [0.5] and c.tolist() == [1]


def test_image_feats_markup_group():
    T = Affine(0.5, 0, 400000, 0, -0.5, 6200000)
    B = np.array([[100, 100, 116, 116], [101, 101, 117, 117], [400, 400, 416, 416]], float)
    fs = A.image_feats(B, np.array([0.9, 0.6, 0.7]), np.array([4, 4, 2]), T, "EPSG:32636", "r", "x_Or.tif", "or",
                       A.settings({}))
    assert len(fs) == 2   # первые два склеены (один снимок, один класс)
    assert {f["markup_type"] for f in fs} == {"or"} and {f["threshold_group"] for f in fs} == {"or_dfine"}
    assert {f["class_name"] for f in fs} == {A.CLASSES[4], A.CLASSES[2]}


def test_cap_per_group():
    fs = [{"threshold_group": "ae_dfine", "confidence": i / 10} for i in range(5)] + [{"threshold_group": "li", "confidence": 0}]
    out = A.cap(fs, 2)
    assert len(out) == 3 and sorted(f["confidence"] for f in out if f["threshold_group"] == "ae_dfine") == [0.3, 0.4]


def test_pipeline_gated_by_flag():
    src = (ROOT / "expds" / "pipeline.py").read_text(encoding="utf-8")
    assert src.count('if (cfg.get("aeor_dfine") or {}).get("enabled"):') == 2
