"""Выбор карт MagGr участка: серая или все карты; находки разных карт — один объект."""
import shutil
from pathlib import Path
from types import SimpleNamespace

from conftest import UTM, write_mag_tif
from shapely.geometry import Point

from expds import geophys, pipeline


def test_select_rasters_modes():
    files = [Path("01_x_MagGr_серый.tif"), Path("02_x_MagGr_цвет.tif"), Path("03_x_MagGr_цвет_2.tif")]
    assert geophys.select_rasters(files, "maggr") == files[:1]                  # как в сдаче
    assert geophys.select_rasters(files, "maggr", "all") == files
    only_color = [Path("02_x_MagGr_цвет.tif")]
    assert geophys.select_rasters(only_color, "maggr") == only_color            # серой нет — цветная
    deep = [Path("x_EtGr_карта_1_0.75.tif"), Path("x_EtGr_карта_2_2.5.tif")]
    assert geophys.select_rasters(deep, "etgr", "all") == deep[:1]              # глубже 2 м — всё равно нет


def test_dedupe_points_merges_only_across_maps():
    def f(x, conf, src, cls="finds_points"):
        return {"geom": Point(x, 6_100_000.0).buffer(0.3), "class_name": cls, "confidence": conf, "source": src}

    from expds.crs import WEB_MERCATOR, to_crs

    feats = [f(500_000.0, 0.9, "gray"), f(500_000.5, 0.8, "color"),        # одна находка на двух картах
             f(500_001.0, 0.7, "gray"),                                      # соседняя на той же карте — своя
             f(500_010.0, 0.6, "color"), f(500_000.2, 0.5, "color", "object_poly")]
    feats = [dict(x, geom=to_crs(x["geom"], UTM, WEB_MERCATOR)) for x in feats]
    out = geophys.dedupe_points(feats, 2.0, UTM)
    kept = sorted((x["class_name"], x["source"], x["confidence"]) for x in out)
    assert kept == sorted([("finds_points", "gray", 0.9), ("finds_points", "gray", 0.7),
                           ("finds_points", "color", 0.6), ("object_poly", "color", 0.5)])


def test_dedupe_keeps_the_stronger_map_not_the_higher_rank():
    def f(x, conf, strength, src):
        return {"geom": Point(x, 6_100_000.0).buffer(0.3), "class_name": "finds_points",
                "confidence": conf, "_strength": strength, "source": src}

    from expds.crs import WEB_MERCATOR, to_crs

    feats = [f(500_000.0, 0.4, 0.85, "gray"), f(500_000.4, 1.0, 0.15, "color")]
    feats = [dict(x, geom=to_crs(x["geom"], UTM, WEB_MERCATOR)) for x in feats]
    out = geophys.dedupe_points(feats, 2.0, UTM)
    assert [(x["source"], x["confidence"]) for x in out] == [("gray", 0.4)]


def test_pipeline_all_maps_does_not_double_the_finds(tmp_path):
    src, _ = write_mag_tif(tmp_path / "src.tif")
    gray = tmp_path / "01_r_MagGr_серый.tif"
    color = tmp_path / "02_r_MagGr_цвет.tif"
    shutil.copy(src, gray)
    shutil.copy(src, color)
    reg = SimpleNamespace(name="r", rasters={"maggr": [gray, color]}, boundaries=[], noise=[], utm_json=None,
                          las=[])
    cfg = {"geophys": {"maggr": {"finds_points": {"min_confidence": 0.0}}}}
    one = [f for f in pipeline.geophys_feats(reg, cfg, pipeline.TimeGuard(1000)) if f["class_name"] == "finds_points"]
    cfg["geophys"]["maggr"]["maps"] = "all"
    both = [f for f in pipeline.geophys_feats(reg, cfg, pipeline.TimeGuard(1000)) if f["class_name"] == "finds_points"]
    assert len(one) > 0 and len(both) == len(one)                    # те же точки на двух картах — по разу
    assert {f["source"] for f in one} == {gray.name}


def test_tiles_exclude_files():
    from expds.tiles import sources_for

    reg = SimpleNamespace(rasters={"maggr": [Path("01_a_MagGr_серый.tif"), Path("02_a_MagGr_цвет.tif")]},
                          li_rasters=[])
    assert sources_for(reg, "maggr", []) == [[Path("01_a_MagGr_серый.tif")], [Path("02_a_MagGr_цвет.tif")]]
    assert sources_for(reg, "maggr", [], ("01_a_MagGr_серый.tif",)) == [[Path("02_a_MagGr_цвет.tif")]]
