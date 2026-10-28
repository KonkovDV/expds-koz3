from shapely.geometry import Point

from expds import post


def _f(grp, cls, x, conf, reg="r"):
    return {"class_name": cls, "threshold_group": grp, "region_name": reg,
            "confidence": conf, "geom": Point(x, 7_000_000.0)}


def test_sp_score_fusion_averages_with_other_group_max():
    feats = [_f("sp_focal", "kurgany_povrezhdennye", 0.0, 0.4),
             _f("sp_focal2", "kurgany_tselye", 3.0, 0.8),
             _f("sp_focal2", "kurgany_povrezhdennye", 4.0, 0.6),
             _f("sp_focal", "kurgany_tselye", 1000.0, 0.5),          # нет соседа -> o = 0
             _f("sp_focal2", "kurgany_povrezhdennye", 0.0, 0.9, reg="q"),  # другой участок
             _f("li", "kurgany_povrezhdennye", 0.0, 0.7)]             # не sp — не трогаем
    out = post.sp_score_fusion(feats)
    assert len(out) == len(feats)
    c = [f["confidence"] for f in out]
    assert abs(c[0] - (0.4 + 0.8) / 2) < 1e-9
    assert abs(c[1] - (0.8 + 0.4) / 2) < 1e-9
    assert abs(c[2] - (0.6 + 0.4) / 2) < 1e-9
    assert abs(c[3] - 0.25) < 1e-9
    assert abs(c[4] - 0.45) < 1e-9
    assert c[5] == 0.7
    assert feats[0]["confidence"] == 0.4  # вход не меняется


def test_sp_score_fusion_off_by_default_in_finish():
    from expds import watchdog
    feats = [_f("sp_focal", "kurgany_povrezhdennye", 0.0, 0.4)]
    cfg = {"thresholds": {}, "default_threshold": 0.0}
    assert "sp_score_fusion" not in (cfg.get("post") or {})
    out = post.sp_score_fusion(feats, {"radius_m": 8.0})
    assert out[0]["confidence"] == 0.2
    assert callable(watchdog.finish)
