"""Сеть на картах геофизики в сдаче: включается только строкой [models], своя группа порога."""
from pathlib import Path
from types import SimpleNamespace

from shapely.geometry import Point

from expds import pipeline


def _region(tmp_path):
    gray = tmp_path / "01_x_MagGr_серый.tif"
    color = tmp_path / "01_x_MagGr_цвет.tif"
    return SimpleNamespace(name="region_0001", las=[], rasters={"maggr": [gray, color]},
                           boundaries=[], noise=[], utm_json=None)


def test_no_model_line_means_no_network_branch(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(pipeline, "_cached_segmenters", lambda *a, **k: called.append(a[2]) or [])
    out = pipeline.geo_model_feats(_region(tmp_path), {"models": {}}, pipeline.TimeGuard(1000), {}, Path("."))
    assert out == [] and called == []


def test_network_uses_gray_map_and_its_own_group(monkeypatch, tmp_path):
    seen = []

    def segs(cache, cfg, markup, root, tta=1):
        return [SimpleNamespace(keep=None)] if markup == "maggr" else []

    def raster(seg, path, region, markup, block=4096, overlap=256, min_conf=None, fallback_crs=None):
        seen.append((Path(path).name, markup, min_conf["finds_points"]))
        return [{"class_name": "finds_points", "markup_type": markup, "confidence": 0.7,
                 "geom": Point(0, 0).buffer(0.3), "region_name": region}]

    monkeypatch.setattr(pipeline, "_cached_segmenters", segs)
    monkeypatch.setattr(pipeline, "_raster_with_model", raster)
    cfg = {"models": {"maggr": {"path": "m.pt"}},
           "thresholds_by_markup": {"maggr_nn": {"finds_points": 0.42}}}
    out = pipeline.geo_model_feats(_region(tmp_path), cfg, pipeline.TimeGuard(1000), {}, Path("."))
    assert seen == [("01_x_MagGr_серый.tif", "maggr", 0.42)]
    assert [f["threshold_group"] for f in out] == ["maggr_nn"]


def test_classic_switch_skips_only_that_modality(monkeypatch, tmp_path):
    reg = _region(tmp_path)
    reg.rasters["terfl"] = [tmp_path / "01_x_TerFl.tif"]
    ran = []
    monkeypatch.setattr(pipeline.geophys, "finds_points", lambda path, *a, **k: ran.append(Path(path).name) or [])
    monkeypatch.setattr(pipeline.geophys, "object_polys", lambda path, *a, **k: ran.append(Path(path).name) or [])
    monkeypatch.setattr(pipeline.geophys, "load_mask", lambda *a, **k: None)
    cfg = {"geophys": {"maggr": {"classic": False}}}
    pipeline.geophys_feats(reg, cfg, pipeline.TimeGuard(1000))
    assert ran == ["01_x_TerFl.tif"]
    ran.clear()
    pipeline.geophys_feats(reg, {"geophys": {}}, pipeline.TimeGuard(1000))
    assert "01_x_MagGr_серый.tif" in ran and "01_x_MagGr_цвет.tif" not in ran
