"""Выбор карт MagGr участка: серая или все карты; находки разных карт — один объект."""
import shutil
from pathlib import Path
from types import SimpleNamespace

from conftest import UTM, write_mag_tif
from shapely.geometry import Point

from expds import geophys, pipeline


def test_broken_etgr_boundary_keeps_maggr_finds(tmp_path, monkeypatch):
    """Сильная магнитка не пропадает, если граница электроразведки того же участка не читается."""
    from types import SimpleNamespace

    from expds.catalog import Region

    mag = write_mag_tif(tmp_path / "a_MagGr_серый.tif")[0]
    et = write_mag_tif(tmp_path / "b_EtGr_карта_01_0.5.tif", seed=2)[0]
    reg = Region("r", tmp_path, rasters={"maggr": [mag], "etgr": [et]})
    real = geophys.survey_boundary

    def boom(files, markup, *args, **kwargs):
        if markup == "etgr":
            raise RuntimeError("битая граница")
        return real(files, markup, *args, **kwargs)

    monkeypatch.setattr(geophys, "survey_boundary", boom)
    guard = SimpleNamespace(ok=lambda *_a, **_k: True, elapsed=lambda: 0.0, left=lambda: 999.0)
    out = pipeline.geophys_feats(reg, {}, guard)
    assert any(f["markup_type"] == "maggr" and f["class_name"] == "finds_points" for f in out)
    assert all(f["markup_type"] != "etgr" for f in out)


def test_broken_finds_keep_the_polygon_of_the_same_map(tmp_path, monkeypatch):
    """Находки и контуры одной карты считаются отдельно: сбой находок не стирает контур."""
    from expds.catalog import Region

    mag = write_mag_tif(tmp_path / "a_MagGr_серый.tif")[0]
    reg = Region("r", tmp_path, rasters={"maggr": [mag]})
    poly = {"geom": Point(0, 0).buffer(2), "class_name": "object_poly", "confidence": 0.9,
            "markup_type": "maggr", "region_name": "r", "source": mag.name}

    def boom(*_a, **_k):
        raise RuntimeError("находки")

    monkeypatch.setattr(geophys, "finds_points", boom)
    monkeypatch.setattr(geophys, "object_polys", lambda *_a, **_k: [dict(poly)])
    guard = SimpleNamespace(ok=lambda *_a, **_k: True, elapsed=lambda: 0.0, left=lambda: 999.0)
    out = pipeline.geophys_feats(reg, {}, guard)
    assert any(f["class_name"] == "object_poly" for f in out)
    assert all(f["class_name"] != "finds_points" for f in out)


def test_geo_model_failure_keeps_optical_objects(monkeypatch):
    """Сеть на картах не должна стирать снимки, которые оптика уже нашла."""
    found = {"class_name": "kurgany_tselye", "geom": Point(0, 0).buffer(1), "confidence": 0.8}
    monkeypatch.setattr(pipeline, "_wide_without_lidar", lambda *a, **k: list(a[-1]) + [found])
    monkeypatch.setattr(pipeline, "geo_model_feats", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("сеть")))
    reg = SimpleNamespace(name="r", rasters={})
    guard = SimpleNamespace(ok=lambda *_a, **_k: True, elapsed=lambda: 0.0, left=lambda: 999.0)
    out = pipeline.optical_feats(reg, {}, guard, {}, Path("."))
    assert out == [found]


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
