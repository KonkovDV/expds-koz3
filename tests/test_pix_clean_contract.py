"""Контракт expds/pix_clean.py (чистая перепись PIXGBM по docs/PIXGBM_SPEC_CLEANROOM.md).

Синтетические растры и маленькие модели: без данных соревнования.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
lgb = pytest.importorskip("lightgbm")
from rasterio.transform import from_origin  # noqa: E402
from shapely.geometry import Point  # noqa: E402

from expds import pix_clean as pc  # noqa: E402
from expds.catalog import Region  # noqa: E402

CRS = "EPSG:32636"
X0, Y0 = 400000.0, 6070000.0


def _raster(path: Path, res: float = 0.05, n: int = 400, spots=((100, 100), (250, 300))) -> Path:
    rng = np.random.default_rng(0)
    img = np.full((3, n, n), 128, np.float32) + rng.normal(0, 4, (3, n, n))
    yy, xx = np.mgrid[0:n, 0:n]
    for r, c in spots:
        img -= 60 * np.exp(-((yy - r) ** 2 + (xx - c) ** 2) / (2 * 6.0 ** 2))
    img = np.clip(img, 5, 250).astype(np.uint8)
    with rasterio.open(path, "w", driver="GTiff", width=n, height=n, count=3, dtype="uint8", crs=CRS,
                       transform=from_origin(X0, Y0, res, res)) as ds:
        ds.write(img)
    return path


def _models(d: Path, classes=("finds_points", "object_poly"), n_feat: int | None = None) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    nf = n_feat or len(pc.feature_names())
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, nf))
    y = (X[:, 0] > 0).astype(int)
    for c in classes:
        b = lgb.train({"objective": "binary", "verbose": -1, "num_threads": 1, "seed": 0, "deterministic": True},
                      lgb.Dataset(X, y), num_boost_round=5)
        b.save_model(str(d / pc.MODEL_FILES[c]))
    (d / "meta.json").write_text(json.dumps({"feature_version": pc.FEATURE_VERSION}), encoding="utf-8")
    return d


def _region(tmp: Path, files: dict) -> Region:
    return Region(name="299_Тест", path=tmp, rasters={k: list(v) for k, v in files.items()})


def _cfg(**pix) -> dict:
    base = {"enabled": True, "impl": "clean", "budget_s": 1e9, "procs": 1, "p_min": 0.0, "t_pix": 0.0}
    base.update(pix)
    return {"geophys": {"pixgbm": base}}


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    monkeypatch.delenv("PIXGBM_BUDGET_S", raising=False)
    monkeypatch.delenv("EXPDS_PIXGBM_STRICT", raising=False)
    pc._MODEL_CACHE.clear()
    pc._RATE.clear()
    yield
    pc._MODEL_CACHE.clear()
    pc._RATE.clear()


def test_feature_names_unique_and_stable():
    names = pc.feature_names()
    assert len(names) == len(set(names)) == 60


def test_region_feats_contract(tmp_path):
    r1 = _raster(tmp_path / "01_Т_MagGr_серый.tif")
    r2 = _raster(tmp_path / "01_Т_EtGr_карта_01_0.5.tif")
    md = _models(tmp_path / "m")
    reg = _region(tmp_path, {"maggr": [r1], "etgr": [r2]})
    st: dict = {}
    out = pc.region_feats(reg, _cfg(), md, status=st)
    assert out
    for f in out:
        assert set(f) >= {"geom", "class_name", "confidence", "region_name", "markup_type", "source",
                          "original_crs", "threshold_group"}
        assert f["threshold_group"] == f"{f['markup_type']}_pix"
        assert 0.0 <= f["confidence"] <= 1.0
        assert f["region_name"] == reg.name
        assert "_xy" not in f
        assert f["geom"].is_valid and not f["geom"].is_empty
        assert f["class_name"] != "finds_points" or f["markup_type"] == "maggr"
    assert st["done"][("finds_points", "maggr")] == {r1.name}
    assert st["done"][("object_poly", "etgr")] == {r2.name}
    assert not any(st["fell"].values())


def test_deterministic(tmp_path):
    r1 = _raster(tmp_path / "01_Т_MagGr_серый.tif")
    md = _models(tmp_path / "m")
    reg = _region(tmp_path, {"maggr": [r1]})
    a = pc.region_feats(reg, _cfg(), md)
    b = pc.region_feats(reg, _cfg(), md)
    assert [(f["class_name"], round(f["confidence"], 9), f["geom"].wkt) for f in a] == \
           [(f["class_name"], round(f["confidence"], 9), f["geom"].wkt) for f in b]


def test_budget_fallback_to_classic(tmp_path, monkeypatch):
    r1 = _raster(tmp_path / "01_Т_MagGr_серый.tif")
    md = _models(tmp_path / "m")
    reg = _region(tmp_path, {"maggr": [r1]})
    monkeypatch.setenv("PIXGBM_BUDGET_S", "0")
    st: dict = {}
    out = pc.region_feats(reg, _cfg(), md, status=st)
    assert out == []
    assert st["fell"][("finds_points", "maggr")] and st["fell"][("object_poly", "maggr")]
    assert not st["done"]


def test_budget_ladder_finds_only(tmp_path, monkeypatch):
    r1 = _raster(tmp_path / "01_Т_MagGr_серый.tif")
    md = _models(tmp_path / "m")
    reg = _region(tmp_path, {"maggr": [r1]})
    # прогноз полного прохода не влезает в срок, прогноз только finds влезает
    monkeypatch.setattr(pc, "forecast_s", lambda mode, mpix: 100.0 if mode == "full" else 1.0)

    class Guard:
        def elapsed(self):
            return 0.0

        def left(self):
            return 50.0
    st: dict = {}
    out = pc.region_feats(reg, _cfg(budget_s=1e9), md, guard=Guard(), status=st)
    assert {f["class_name"] for f in out} <= {"finds_points"}
    assert st["done"].get(("finds_points", "maggr")) == {r1.name}
    assert st["fell"][("object_poly", "maggr")]


def test_missing_model_soft_and_strict(tmp_path, monkeypatch):
    r1 = _raster(tmp_path / "01_Т_MagGr_серый.tif")
    md = _models(tmp_path / "m", classes=("finds_points",))
    reg = _region(tmp_path, {"maggr": [r1]})
    st: dict = {}
    out = pc.region_feats(reg, _cfg(), md, status=st)
    assert {f["class_name"] for f in out} <= {"finds_points"}
    assert st["fell"][("object_poly", "maggr")]
    pc._MODEL_CACHE.clear()
    monkeypatch.setenv("EXPDS_PIXGBM_STRICT", "1")
    with pytest.raises(FileNotFoundError):
        pc.region_feats(reg, _cfg(), md)


def test_wrong_feature_count_strict(tmp_path, monkeypatch):
    md = _models(tmp_path / "m", n_feat=7)
    monkeypatch.setenv("EXPDS_PIXGBM_STRICT", "1")
    with pytest.raises(ValueError):
        pc.load_models(md, ("finds_points",))
    monkeypatch.delenv("EXPDS_PIXGBM_STRICT")
    models, _ = pc.load_models(md, ("finds_points",))
    assert models == {}


def _f(cls, mt, geom, src="a.tif", conf=0.5):
    return {"geom": geom, "class_name": cls, "markup_type": mt, "source": src, "confidence": conf}


def test_dedupe_union_class_and_method():
    # EPSG:3857 на широте ~54°: 1 м на местности ≈ 1,7 единицы
    x, y = 3_550_000.0, 7_200_000.0
    old = [_f("finds_points", "maggr", Point(x, y)),
           _f("object_poly", "maggr", Point(x, y).buffer(5))]
    new = [_f("finds_points", "maggr", Point(x + 1.0, y)),       # ≈0,6 м — дубль
           _f("finds_points", "maggr", Point(x + 4.0, y)),       # ≈2,3 м — новый
           _f("object_poly", "maggr", Point(x, y).buffer(1)),    # центр внутри — дубль
           _f("object_poly", "terfl", Point(x, y).buffer(1)),    # другой метод — остаётся
           _f("object_poly", "maggr", Point(x + 20, y).buffer(1))]
    out = pc.dedupe_union(old, new)
    assert [o["geom"].centroid.x - x for o in out] == pytest.approx([4.0, 0.0, 20.0], abs=1e-6)
    assert [o["markup_type"] for o in out] == ["maggr", "terfl", "maggr"]


def test_dedupe_union_etgr_layers_not_merged():
    x, y = 3_550_000.0, 7_200_000.0
    old = [_f("object_poly", "etgr", Point(x, y).buffer(5), src="01_EtGr_карта_01.tif")]
    new = [_f("object_poly", "etgr", Point(x, y).buffer(1), src="01_EtGr_карта_01.tif"),
           _f("object_poly", "etgr", Point(x, y).buffer(1), src="02_EtGr_карта_02.tif")]
    out = pc.dedupe_union(old, new)
    assert [o["source"] for o in out] == ["02_EtGr_карта_02.tif"]


def test_cross_maps_only_maggr_different_sources():
    x, y = 3_550_000.0, 7_200_000.0
    feats = [_f("finds_points", "maggr", Point(x, y), "серый.tif", 0.9),
             _f("finds_points", "maggr", Point(x + 0.5, y), "цвет.tif", 0.4),
             _f("finds_points", "maggr", Point(x + 0.5, y + 0.2), "серый.tif", 0.3)]
    out = pc.dedupe_cross_maps(feats, 1.0)
    assert sorted(o["confidence"] for o in out) == [0.3, 0.9]


def test_pipeline_dispatch_strict_reraises(tmp_path, monkeypatch):
    from expds import pipeline

    src = Path(pipeline.__file__).read_text(encoding="utf-8")
    assert 'impl == "clean"' in src and "pix_clean" in src and "EXPDS_PIXGBM_STRICT" in src
