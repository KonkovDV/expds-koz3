"""W4 v4: флаг w5_cloudfeat (RR1s-cf). Выкл. — ничего не меняется; вкл. без модуля — понятная ошибка; шкала S."""
import sys
import tomllib
import types
from pathlib import Path

import numpy as np
import pytest

from expds import pipeline, w5_cloudfeat as W, watchdog

ROOT = Path(__file__).resolve().parents[1]


def _f(cls, conf, mk="li", grp=None):
    f = {"class_name": cls, "confidence": conf, "markup_type": mk, "region_name": "r1",
         "geom": None}
    if grp:
        f["threshold_group"] = grp
    return f


@pytest.fixture(autouse=True)
def _clean_cache():
    W._CACHE.clear()
    yield
    W._CACHE.clear()


@pytest.fixture
def ref(tmp_path):
    conf_li = np.sort(np.linspace(0.01, 0.99, 99))
    rb_pool = np.sort(np.linspace(0.1, 0.9, 9))
    r_pool = np.sort(np.linspace(0.0, 0.8, 9))
    p = tmp_path / "ref.npz"
    np.savez(p, alpha=0.6, classes=np.array(["kurgany_tselye"]),
             conf_li__kurgany_tselye=conf_li, rb_pool__kurgany_tselye=rb_pool, r_pool__kurgany_tselye=r_pool)
    return p


def test_default_flag_off_in_default_toml():
    cfg = tomllib.loads((ROOT / "configs" / "default.toml").read_text(encoding="utf-8"))
    assert cfg.get("w5_cloudfeat") is False
    assert W.enabled(cfg) is False


def test_v3_on_has_no_flag():
    cfg = tomllib.loads((ROOT / "configs" / "v3_on.toml").read_text(encoding="utf-8"))
    assert not W.enabled(cfg)


def test_off_is_identity_and_no_io(monkeypatch):
    feats = [_f("kurgany_tselye", 0.5), _f("arkhitektury", 0.7, mk="ae")]
    monkeypatch.setattr(W, "_plugin", lambda cfg: (_ for _ in ()).throw(AssertionError("no plugin when off")))
    monkeypatch.setattr(W, "_ref", lambda cfg: (_ for _ in ()).throw(AssertionError("no ref when off")))
    cfg = {"w5_cloudfeat": False}
    assert W.apply(feats, cfg) is feats
    assert W.region_features(None, feats, cfg) is feats
    W.preflight(cfg)
    assert W.li_floor(cfg, "li", "kurgany_tselye", 0.3) == 0.3


def test_off_output_floors_unchanged():
    cfg = tomllib.loads((ROOT / "configs" / "v3_on.toml").read_text(encoding="utf-8"))
    a = pipeline.output_floors(cfg, "li")
    cfg2 = dict(cfg, w5_cloudfeat=False)
    assert pipeline.output_floors(cfg2, "li") == a


def test_off_finish_same_as_without_key():
    cfg = tomllib.loads((ROOT / "configs" / "v3_on.toml").read_text(encoding="utf-8"))
    from shapely.geometry import Point
    feats = [dict(_f("kurgany_tselye", c, grp="li"), geom=Point(37 + i * 1e-3, 55).buffer(1e-4))
             for i, c in enumerate((0.1, 0.5, 0.999))]
    a = watchdog.finish([dict(f) for f in feats], cfg)
    b = watchdog.finish([dict(f) for f in feats], dict(cfg, w5_cloudfeat=False))
    assert [(f["class_name"], f["confidence"]) for f in a] == [(f["class_name"], f["confidence"]) for f in b]


def test_on_without_module_clear_error(ref):
    cfg = {"w5_cloudfeat": True, "w5cf": {"module": "expds.__no_such_w5cf_module__", "ref": str(ref)}}
    with pytest.raises(W.W5CloudFeatMissing, match="W1"):
        W.preflight(cfg)


def test_on_without_ref_clear_error(monkeypatch, tmp_path):
    mod = types.ModuleType("w5cf_fake_ok"); mod.score = lambda feats, cfg: [None] * len(feats)
    monkeypatch.setitem(sys.modules, "w5cf_fake_ok", mod)
    cfg = {"w5_cloudfeat": True, "w5cf": {"module": "w5cf_fake_ok", "ref": str(tmp_path / "none.npz")}}
    with pytest.raises(W.W5CloudFeatMissing, match="референс"):
        W.preflight(cfg)


def test_on_blend_scale(monkeypatch, ref):
    mod = types.ModuleType("w5cf_fake"); mod.score = lambda feats, cfg: [0.8, None, 0.0, None][:len(feats)]
    monkeypatch.setitem(sys.modules, "w5cf_fake", mod)
    cfg = {"w5_cloudfeat": True, "w5cf": {"module": "w5cf_fake", "ref": str(ref), "alpha": 0.6}}
    W.preflight(cfg)
    feats = [_f("kurgany_tselye", 0.50, grp="li_dev"), _f("kurgany_tselye", 0.50, grp="li_dev"),
             _f("kurgany_tselye", 0.50), _f("kurgany_tselye", 0.50, mk="ae")]
    out = W.apply(feats, cfg)
    rb = W._cdf(np.sort(np.linspace(0.01, 0.99, 99)), np.array([0.5]))[0]
    assert out[0]["confidence"] == pytest.approx(0.6 * 0.9 + 0.4 * rb)       # top score → top RB of pool
    assert out[1]["confidence"] == pytest.approx(rb)                           # no score → RB
    assert out[2]["confidence"] == pytest.approx(0.6 * 0.1 + 0.4 * rb)       # lowest score → lowest RB of pool
    assert out[3]["confidence"] == 0.50                                        # not li → untouched
    assert feats[0]["confidence"] == 0.50                                      # input not mutated


def test_on_score_length_mismatch(monkeypatch, ref):
    mod = types.ModuleType("w5cf_bad"); mod.score = lambda feats, cfg: []
    monkeypatch.setitem(sys.modules, "w5cf_bad", mod)
    cfg = {"w5_cloudfeat": True, "w5cf": {"module": "w5cf_bad", "ref": str(ref)}}
    with pytest.raises(ValueError):
        W.apply([_f("kurgany_tselye", 0.5)], cfg)


def test_li_floor_conservative(ref):
    cfg = {"w5_cloudfeat": True, "w5cf": {"module": "x", "ref": str(ref), "alpha": 0.6}}
    conf_li = np.sort(np.linspace(0.01, 0.99, 99))
    for thr in (0.5, 0.7, 0.9, 0.99):
        fl = W.li_floor(cfg, "li", "kurgany_tselye", thr)
        need = max(0.0, (thr - 0.6) / 0.4)
        # every conf with RB ≥ need survives the floor
        ok = conf_li[W._cdf(conf_li, conf_li) >= need]
        assert (ok >= fl).all()
    assert W.li_floor(cfg, "ae", "kurgany_tselye", 0.9) == 0.9


def test_v4_on_config_if_present():
    p = ROOT / "configs" / "v4_on.toml"
    if not p.exists():
        pytest.skip("v4_on.toml нет")
    v4 = tomllib.loads(p.read_text(encoding="utf-8")); v3 = tomllib.loads((ROOT / "configs" / "v3_on.toml").read_text(encoding="utf-8"))
    assert v4["w5_cloudfeat"] is True and v4["w5cf"]["alpha"] == 0.6
    for k in v3:
        if k in ("thresholds_by_markup",):
            continue
        assert v4[k] == v3[k], k
    for g, d in v3["thresholds_by_markup"].items():
        if g not in W.LI_GROUPS:
            assert v4["thresholds_by_markup"][g] == d, g


def test_weights_manifest_mismatch_raises(tmp_path, monkeypatch):
    import hashlib
    import json

    import pytest

    from expds import w5_cloudfeat as W
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "a.txt").write_bytes(b"x")
    good = {"w5cf": {"m/a.txt": {"sha256": hashlib.sha256(b"x").hexdigest(), "bytes": 1}}}
    (tmp_path / "WEIGHTS_MANIFEST.json").write_text(json.dumps(good))
    monkeypatch.setattr(W, "_ROOT", tmp_path)
    W._verify_manifest()
    (tmp_path / "m" / "a.txt").write_bytes(b"y")
    with pytest.raises(W.W5CloudFeatMissing):
        W._verify_manifest()
