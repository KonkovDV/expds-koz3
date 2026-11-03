"""W4 [post.w4_sat] (PREREG c1fx_regate): off by default; center + frame priors on sp_dfine before thresholds."""
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
from shapely.geometry import Point

from expds import sat_recall as SR

CFG = Path(__file__).resolve().parents[1] / "configs" / "v3_on.toml"
FL = {"enabled": True, "groups": ["sp_dfine"], "center": {"d0": 0.25, "s_by_class": {"gorodishcha": 0.5, "arkhitektury": 0.5}},
      "frame": {"classes": ["kurgany_tselye", "arkhitektury"], "d0": 0.5, "edge": 0.5},
      "thresholds_sp_dfine": {"gorodishcha": 0.11, "arkhitektury": 0.12, "kurgany_tselye": 0.13}}


def _cfg(fl=None):
    c = tomllib.loads(CFG.read_text(encoding="utf-8"))
    if fl:
        c["post"]["w4_sat"] = dict(fl)
    return c


def _f(cls, conf, d=1.0, grp="sp_dfine"):
    return dict(geom=Point(0, 0).buffer(3), class_name=cls, confidence=conf, region_name="r1", markup_type="sp_or",
                threshold_group=grp, scene_drel=d)


def test_off_by_default_identity():
    c = _cfg()
    assert "w4_sat" not in c["post"] and SR.resolve_w4_flag(c) is c


def test_resolve():
    c = _cfg(FL); r = SR.resolve_w4_flag(c)
    assert r["post"]["sat_center_prior"]["s_by_class"] == {"gorodishcha": 0.5, "arkhitektury": 0.5}
    assert r["post"]["sat_frame_prior"] == {"groups": ["sp_dfine"], "classes": ["kurgany_tselye", "arkhitektury"], "d0": 0.5, "edge": 0.5}
    assert r["thresholds_by_markup"]["sp_dfine"]["kurgany_tselye"] == 0.13
    assert r["thresholds_by_markup"]["sp_dfine"]["fortifikatsii"] == c["thresholds_by_markup"]["sp_dfine"]["fortifikatsii"]
    assert "sat_center_prior" not in c["post"] and SR.resolve_w4_flag(r) is r
    c["post"]["w1_c1f"]["enabled"] = True
    with pytest.raises(ValueError):
        SR.resolve_w4_flag(c)


@pytest.mark.parametrize("fn", ["finish", "safe_finish"])
def test_finish_applies_both_priors(monkeypatch, fn):
    from expds import post, watchdog
    seen = {}
    real = post.apply_thresholds

    def spy(feats, thr, default, by_markup):
        seen["conf"] = {(f["class_name"], f["threshold_group"]): f["confidence"] for f in feats}
        seen["by"] = by_markup
        return real(feats, thr, default, by_markup)
    monkeypatch.setattr(post, "apply_thresholds", spy)
    feats = [_f("gorodishcha", 0.8), _f("arkhitektury", 0.8), _f("kurgany_tselye", 0.8), _f("fortifikatsii", 0.8),
             _f("kurgany_tselye", 0.8, grp=None)]
    getattr(watchdog, fn)(feats, _cfg(FL))
    c = seen["conf"]
    assert abs(c[("gorodishcha", "sp_dfine")] - 0.4) < 1e-9          # center: s 0.5 at the edge
    assert abs(c[("arkhitektury", "sp_dfine")] - 0.2) < 1e-9         # center 0.5 × frame 0.5
    assert abs(c[("kurgany_tselye", "sp_dfine")] - 0.4) < 1e-9       # frame only
    assert c[("fortifikatsii", "sp_dfine")] == 0.8
    assert seen["by"]["sp_dfine"]["gorodishcha"] == 0.11


def test_sp_dfine_phase_scene_drel_under_w4_sat(monkeypatch):
    from expds import pipeline, sp_dfine

    def fake_region_feats(reg, cfg, guard, cache, root, crs_for=None, soft_ok=None):
        cache["models"] = [object()]
        return [dict(_f("gorodishcha", 0.8)) for _ in range(1)], True
    monkeypatch.setattr(sp_dfine, "region_feats", fake_region_feats)
    monkeypatch.setattr(pipeline, "_with_scene_drel", lambda reg, cfg, feats: [dict(f, w4_mark=1) for f in feats])
    monkeypatch.setattr(pipeline, "_soft_ok", lambda *a, **k: True)
    reg = SimpleNamespace(name="r1", rasters={"sp_or": ["x.tif"]}, utm_json=None)
    guard = SimpleNamespace(ok=lambda s: True, elapsed=lambda: 0.0)
    for fl in (None, FL):
        cfg = _cfg(fl); cfg["sp_dfine"] = dict(cfg["sp_dfine"], drop_groups=[])
        got = []
        st = SimpleNamespace(add=got.extend, snapshot=lambda: list(got), replace=lambda x: None)
        pipeline._sp_dfine_phase([reg], cfg, guard, st, Path("."))
        assert len(got) == 1 and (("w4_mark" in got[0]) == (fl is not None))
