from shapely.geometry import Point
from expds import post, watchdog


def _f(cls, grp, x, conf=0.9, markup=None):
    return {"class_name": cls, "threshold_group": grp, "markup_type": markup or grp, "confidence": conf,
            "region_name": "r", "geom": Point(4.1e6 + x, 7.4e6).buffer(2.0)}  # EPSG:3857


def test_twins_only_keeps_ae_near_li_and_adds_copies():
    feats = [_f("kurgany_povrezhdennye", "li", 0.0, 0.5), _f("kurgany_povrezhdennye", "ae", 0.1),
             _f("kurgany_tselye", "ae", 50.0), _f("kurgany_tselye", None, 90.0, markup="sp_or")]
    out = post.kurgan_cross_modal_twins(feats, {"dedup_ae_vs_li_m": 0.0})
    assert sum(1 for f in out if f["threshold_group"] == "ae" and f["class_name"] == "kurgany_povrezhdennye") == 1
    tw = [f for f in out if f["threshold_group"] in ("ae_twin", "sp_focal_twin")]
    assert {f["threshold_group"] for f in tw} == {"ae_twin", "sp_focal_twin"}
    assert all(f["class_name"] == "kurgany_povrezhdennye" for f in tw)
    assert {f["markup_type"] for f in tw} == {"ae", "sp_or"}
    assert len(out) == len(feats) + 2  # ничего не удалено, исходные целые на месте


def test_watchdog_flag_forces_no_dedup(monkeypatch):
    seen = {}
    monkeypatch.setattr(post, "kurgan_cross_modal_twins", lambda out, spec: seen.setdefault("spec", spec) and out)
    cfg = {"post": {"kurgan_twins_only": {"dedup_ae_vs_li_m": 3.0}}, "thresholds": {}, "default_threshold": 0.0}
    watchdog.finish([], cfg)
    assert seen["spec"]["dedup_ae_vs_li_m"] == 0.0


def test_scale_is_isometric_per_region():
    """Регресс (REPORT6_twins.md §2): масштаб на точку, (x·s(y), y·s(y)), не изометрия. На 55° смещение
    (на восток 0,53·d, на север d) сжимается примерно в 24 раза, и li-повреждённый в 22,6 м «гасил» ae-двойника
    (в сдаче 5 вместо 85). С одним cos φ на участок все 20 двойников на месте; старый код даёт 0."""
    import math
    y0 = 7.4e6
    m = 1.0 / math.cos(math.atan(math.sinh(y0 / 6378137.0)))  # единиц 3857 на метр
    feats = []
    for k in range(20):
        x, y = 4.1e6 + k * 100.0 * m, y0
        a = _f("kurgany_tselye", "ae", 0.0)
        a["geom"] = Point(x, y).buffer(1.0)
        b = _f("kurgany_povrezhdennye", "li", 0.0, 0.5)
        b["geom"] = Point(x + 10.56 * m, y + 20.0 * m).buffer(1.0)  # 22,6 м на земле, дальше li_dedup_m = 4
        feats += [a, b]
    out = post.kurgan_cross_modal_twins(feats, {"dedup_ae_vs_li_m": 0.0})
    assert sum(1 for f in out if f["threshold_group"] == "ae_twin") == 20
