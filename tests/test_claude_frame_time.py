"""[post] frame_prior, scene_drel и порядок прохода оптики (аудит 29.10, часть 2)."""
import math
from types import SimpleNamespace

from shapely.geometry import Point, box

from expds import pipeline, post, watchdog


def _f(x, y, conf=0.5, grp="sp_or", cls="gorodishcha"):
    return {"class_name": cls, "threshold_group": grp, "markup_type": "sp_or", "confidence": conf,
            "region_name": "r", "geom": Point(x, y).buffer(1.0)}


def test_scene_drel_centre_edge_and_overlap():
    scene = box(0, 0, 1000, 500)
    assert post.scene_drel(Point(500, 250), [scene]) == 0.0
    assert math.isclose(post.scene_drel(Point(1000, 250), [scene]), 1.0)
    assert math.isclose(post.scene_drel(Point(750, 250), [scene]), 0.5)
    assert post.scene_drel(Point(2000, 250), [scene]) is None
    # вторая сцена с центром ближе к точке — берётся меньшее значение
    assert math.isclose(post.scene_drel(Point(900, 250), [scene, box(800, 150, 1000, 350)]), 0.0)


def test_frame_prior_demotes_only_far_objects_of_its_groups():
    feats = post.annotate_scene_drel(
        [_f(500, 250), _f(950, 250), _f(950, 250, grp="li"), _f(3000, 0)], {"sp_or": [box(0, 0, 1000, 500)]})
    out = post.frame_prior(feats)
    assert out[0]["confidence"] == 0.5                       # центр
    assert math.isclose(out[1]["confidence"], 0.5 * (1 - 0.5 * (0.9 - 0.5) / 0.5))
    assert out[2]["confidence"] == 0.5                       # чужая группа
    assert out[3]["confidence"] == 0.5 and "scene_drel" not in out[3]   # вне сцен


def test_finish_applies_frame_prior_before_thresholds():
    feats = post.annotate_scene_drel([_f(500, 250, 0.4), _f(990, 250, 0.4)], {"sp_or": [box(0, 0, 1000, 500)]})
    cfg = {"thresholds": {}, "default_threshold": 0.3, "thresholds_by_markup": {},
           "post": {"frame_prior": {"d0": 0.5, "edge": 0.5}}}
    for fn in (watchdog.finish, watchdog.safe_finish):
        out = fn([dict(f) for f in feats], cfg)
        assert len(out) == 1 and out[0]["scene_drel"] < 0.01   # центр кадра; край опущен ниже порога


def test_optics_order_small_first_is_opt_in(tmp_path):
    def reg(name, size):
        p = tmp_path / f"{name}.tif"
        p.write_bytes(b"0" * size)
        return SimpleNamespace(name=name, rasters={"sp_or": [p]}, las=[])

    regions = [reg("a_big", 5000), reg("b_small", 10), reg("c_none", 0)]
    assert [r.name for r in pipeline.optics_order(regions, {})] == ["a_big", "b_small", "c_none"]
    cfg = {"time": {"optics_order": "small_first"}}
    assert [r.name for r in pipeline.optics_order(regions, cfg)] == ["c_none", "b_small", "a_big"]
    assert pipeline.optical_bytes(regions[0]) == 5000


def _k(region, s, grp="li_cnx", cls="kurgany_povrezhdennye"):
    return {"class_name": cls, "threshold_group": grp, "markup_type": "li", "confidence": s,
            "region_name": region, "geom": Point(0, 0).buffer(1.0)}


def test_site_calibration_aligns_differently_scaled_sites():
    """Две семьи с одной формой распределения и разной шкалой (×5): после rank/quantile шкалы совпадают."""
    import numpy as np

    base = np.linspace(0.06, 0.2, 40)
    feats = [_k("under", float(s)) for s in base] + [_k("over", float(min(s * 5, 1.0))) for s in base]
    for mode in ("rank", "quantile"):
        out = post.site_calibration(feats, {"groups": ["li_cnx"], "mode": mode})
        a = [f["confidence"] for f in out if f["region_name"] == "under"]
        b = [f["confidence"] for f in out if f["region_name"] == "over"]
        assert np.allclose(a, b, atol=1e-6)
        assert all("conf_raw" in f for f in out)


def test_site_calibration_respects_floor_min_n_and_groups():
    feats = [_k("r", 0.5)] * 10 + [_k("r", 0.01)] + [_k("q", 0.5, grp="li")] * 40
    out = post.site_calibration(feats, {"groups": ["li_cnx"], "min_n": 30})
    assert [f["confidence"] for f in out] == [f["confidence"] for f in feats]      # мало объектов, чужая группа
    cfg = {"thresholds_by_markup": {"li_cnx": {"kurgany_povrezhdennye": 0.3}},
           "post": {"site_calibration": {"groups": ["li_cnx"], "floor": 0.05}}}
    assert pipeline.output_floors(cfg, "li_cnx")["kurgany_povrezhdennye"] == 0.05
    assert pipeline.output_floors(cfg, "li")["kurgany_povrezhdennye"] == 0.15


def test_finish_runs_site_calibration_first():
    feats = [_k("under", 0.06 + 0.001 * i) for i in range(40)]
    cfg = {"thresholds": {}, "default_threshold": 0.15,
           "thresholds_by_markup": {"li_cnx": {"kurgany_povrezhdennye": 0.9}},
           "post": {"site_calibration": {"groups": ["li_cnx"], "mode": "rank"}}}
    for fn in (watchdog.finish, watchdog.safe_finish):
        out = fn([dict(f) for f in feats], cfg)
        assert len(out) == 5      # ранг ≥ 0,9: 36/40 … 40/40, хотя сырая уверенность < 0,1
