"""[post] kurgans_outside_hillforts и kurgan_neighbour_prior (аудит 29.10): синтетика в EPSG:3857 на 55° с. ш."""
import math

from shapely.geometry import Point

from expds import pipeline, post, watchdog

Y0 = 7.4e6
M = 1.0 / math.cos(math.atan(math.sinh(Y0 / 6378137.0)))  # единиц EPSG:3857 на метр земли


def _k(cls, grp, x_m, y_m=0.0, conf=0.5, region="r"):
    return {"class_name": cls, "threshold_group": grp, "markup_type": grp, "confidence": conf,
            "region_name": region, "geom": Point(4.1e6 + x_m * M, Y0 + y_m * M).buffer(0.5 * M)}


def _gor(x_m, r_m, conf=0.99, region="r", grp="li_mask"):
    return {"class_name": "gorodishcha", "threshold_group": grp, "markup_type": "li", "confidence": conf,
            "region_name": region, "geom": Point(4.1e6 + x_m * M, Y0).buffer(r_m * M)}


def test_hillfort_mask_drops_only_kurgans_inside():
    gor = _gor(0.0, 60.0)                                   # ≈ 11 300 м²
    inside = _k("kurgany_povrezhdennye", "li", 10.0)
    outside = _k("kurgany_povrezhdennye", "li", 80.0)
    other_region = _k("kurgany_tselye", "li", 10.0, region="q")
    arch = dict(_k("kurgany_tselye", "li", 5.0), class_name="arkhitektury")
    out = post.kurgans_outside_hillforts([gor, inside, outside, other_region, arch])
    assert inside not in out
    assert all(f in out for f in (gor, outside, other_region, arch))


def test_hillfort_mask_area_guard_buffer_and_conf():
    tiny = _gor(0.0, 5.0)                                   # ≈ 80 м²: ядро-круг, не маска
    k = _k("kurgany_povrezhdennye", "li", 2.0)
    assert k in post.kurgans_outside_hillforts([tiny, k])
    gor = _gor(0.0, 60.0, conf=0.4)
    near = _k("kurgany_povrezhdennye", "li", 65.0)          # 5 м за кромкой
    assert near in post.kurgans_outside_hillforts([gor, near])
    assert near not in post.kurgans_outside_hillforts([gor, near], {"buffer_m": 10.0})
    assert near in post.kurgans_outside_hillforts([gor, near], {"buffer_m": 10.0, "min_gor_conf": 0.5})


def test_neighbour_prior_boosts_cemetery_and_demotes_isolated():
    # могильник: 6 мест по 30 м, у каждого оба класса; одиночка в 2 км
    cem = [_k(c, "li", 30.0 * i, conf=0.4) for i in range(6) for c in post.KURGANS]
    lone = _k("kurgany_povrezhdennye", "li", 2000.0, conf=0.4)
    sp = _k("kurgany_povrezhdennye", "sp_or", 2000.0, conf=0.4)  # чужая группа не трогается
    out = post.kurgan_neighbour_prior(cem + [lone, sp], {"support_conf": 0.3})
    by_x = {(round(f["geom"].centroid.x), f["class_name"], f["threshold_group"]): f for f in out}
    edge = by_x[(round(cem[0]["geom"].centroid.x), "kurgany_povrezhdennye", "li")]
    assert edge["neighbours"] == 5                           # двойник того же места не считается
    assert math.isclose(edge["confidence"], 0.4 * (0.5 + 0.5 * 5 / 3))
    lone_out = next(f for f in out if f.get("threshold_group") == "li" and f.get("neighbours") == 0)
    assert math.isclose(lone_out["confidence"], 0.2)
    assert next(f for f in out if f["threshold_group"] == "sp_or")["confidence"] == 0.4


def test_neighbour_prior_support_by_group_threshold():
    feats = [_k("kurgany_povrezhdennye", "li", 0.0, conf=0.5), _k("kurgany_povrezhdennye", "li", 50.0, conf=0.05)]
    tau = post.threshold_fn({}, 0.15, {"li": {"kurgany_povrezhdennye": 0.1}})
    out = post.kurgan_neighbour_prior(feats, None, tau)
    assert [f["neighbours"] for f in out] == [0, 1]          # слабый сосед ниже порога опорой не служит


def test_finish_runs_prior_before_thresholds_and_mask_after():
    cem = [_k("kurgany_povrezhdennye", "li", 30.0 * i, conf=0.12) for i in range(8)]
    lone = _k("kurgany_povrezhdennye", "li", 5000.0, conf=0.12)
    gor = _gor(5000.0, 60.0)
    cfg = {"thresholds": {}, "default_threshold": 0.15,
           "thresholds_by_markup": {"li": {"kurgany_povrezhdennye": 0.15}, "li_mask": {"gorodishcha": 0.5}},
           "post": {"kurgan_neighbour_prior": {"support_conf": 0.1}, "kurgans_outside_hillforts": True}}
    out = watchdog.finish(cem + [lone, gor], cfg)
    kur = [f for f in out if f["class_name"] == "kurgany_povrezhdennye"]
    assert len(kur) == 8                                     # могильник поднят над порогом, одиночка — нет
    cfg["post"]["kurgan_neighbour_prior"] = False
    assert not [f for f in watchdog.finish(cem + [lone, gor], cfg) if f["class_name"] == "kurgany_povrezhdennye"]
    lone["confidence"] = 0.9                                 # уверенный, но внутри городища
    cfg["post"]["kurgans_outside_hillforts"] = True
    out = watchdog.safe_finish(cem + [lone, gor], cfg)
    assert not [f for f in out if f["class_name"] == "kurgany_povrezhdennye"]


def test_output_floors_lowered_only_for_prior_groups():
    cfg = {"thresholds_by_markup": {"li": {"kurgany_povrezhdennye": 0.3}, "sp_or": {"kurgany_povrezhdennye": 0.3}}}
    assert pipeline.output_floors(cfg, "li")["kurgany_povrezhdennye"] == 0.3
    cfg["post"] = {"kurgan_neighbour_prior": True}
    assert math.isclose(pipeline.output_floors(cfg, "li")["kurgany_povrezhdennye"], 0.2)
    assert pipeline.output_floors(cfg, "sp_or")["kurgany_povrezhdennye"] == 0.3
    assert pipeline.output_floors(cfg, "li")["gorodishcha"] == 0.15
