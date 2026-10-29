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


def test_sp_fusion_raw_partners_alias_and_thresholds():
    # sp_or (markup сдачи) = sp_focal; сосед sp_focal2 ниже старого порога выдачи 0,501645 и
    # помеченный sp_partner_only (выкинут отбором 12 м) всё равно поднимает уверенность и не выдаётся.
    a = {"class_name": "kurgany_povrezhdennye", "markup_type": "sp_or", "region_name": "r",
         "confidence": 0.30, "geom": Point(0.0, 7_000_000.0)}
    b = _f("sp_focal2", "kurgany_povrezhdennye", 2.0, 0.40)
    b["sp_partner_only"] = True
    out = post.sp_score_fusion([a, b])
    assert len(out) == 1 and abs(out[0]["confidence"] - 0.35) < 1e-9
    spec = {"thresholds": {"sp_focal": {"kurgany_povrezhdennye": 0.27822},
                           "sp_focal2": {"kurgany_povrezhdennye": 0.514733}}}
    bm = post.sp_fusion_thresholds({"sp_or": {"kurgany_povrezhdennye": 0.3645, "gorodishcha": 0.99}}, spec)
    assert bm["sp_or"] == {"kurgany_povrezhdennye": 0.27822, "gorodishcha": 0.99}
    assert bm["sp_focal"]["kurgany_povrezhdennye"] == 0.27822
    assert bm["sp_focal2"]["kurgany_povrezhdennye"] == 0.514733
    kept = post.apply_thresholds(out, {}, 0.15, bm)
    assert len(kept) == 1  # 0,35 ≥ 0,27822; со старым 0,3645 был бы отрезан
    fl = post.sp_fusion_gen_floors({"kurgany_povrezhdennye": 0.501645, "gorodishcha": 0.99}, {})
    assert fl == {"kurgany_povrezhdennye": 0.05, "gorodishcha": 0.99}


def test_pipeline_sp_floors_flag_off_is_unchanged():
    import pytest
    pytest.importorskip("rasterio")
    from expds import pipeline
    cfg = {"thresholds_by_markup": {"sp_focal2": {"kurgany_povrezhdennye": 0.501645}}}
    assert pipeline._sp_floors(cfg, "sp_focal2") == pipeline.output_floors(cfg, "sp_focal2")
    cfg["post"] = {"sp_score_fusion": {"gen_floor": 0.05}}
    assert pipeline._sp_floors(cfg, "sp_focal2")["kurgany_povrezhdennye"] == 0.05
    assert pipeline._sp_floors(cfg, "ae") == pipeline.output_floors(cfg, "ae")
