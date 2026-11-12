"""Аудит 12.11: сбой мозаики и мёртвое чтение окон не стирают уже доступные объекты.

На целом входе список тот же: живой фоновый поток по-прежнему отдаёт свои пиксели,
а одна сеть на грубой сетке по-прежнему даёт свои курганы.
"""
from types import SimpleNamespace

import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

from expds.catalog import Region
from expds.pipeline import _raster_with_model, lidar_feats


def _tif(path, value: int):
    data = np.full((3, 8, 8), value, np.uint8)
    with rasterio.open(path, "w", driver="GTiff", height=8, width=8, count=3, dtype="uint8",
                       crs="EPSG:32637", transform=from_origin(500000, 6100000, 1, 1)) as dst:
        dst.write(data)


def _seg():
    seen = []

    def predict(img):
        seen.append(float(img.mean()))
        return np.zeros((1, img.shape[1], img.shape[2]), np.float32)

    seg = SimpleNamespace(meta={"in_channels": 3, "target": "mask"}, classes=["gorodishcha"],
                          res=1.0, predict=predict)
    return seg, seen


def test_dead_readahead_still_reads_every_window(tmp_path, monkeypatch):
    path = tmp_path / "a.tif"
    _tif(path, 3)
    seg, seen = _seg()

    def dead(*_a, **_k):
        if False:
            yield None

    monkeypatch.setattr("expds.pipeline._read_ahead_private", dead)
    _raster_with_model(seg, path, "r", "sp_or", block=4, overlap=0)
    assert seen == [3.0, 3.0, 3.0, 3.0]


def test_readahead_exception_does_not_stop_the_file(tmp_path, monkeypatch):
    path = tmp_path / "a.tif"
    _tif(path, 4)
    seg, seen = _seg()

    def boom(*_a, **_k):
        yield (0, 0, 0, 0), RuntimeError("второй дескриптор не открылся")
        return
        yield

    monkeypatch.setattr("expds.pipeline._read_ahead_private", boom)
    _raster_with_model(seg, path, "r", "sp_or", block=4, overlap=0)
    assert seen == [4.0, 4.0, 4.0, 4.0]


def test_live_readahead_still_uses_its_own_pixels(tmp_path, monkeypatch):
    path = tmp_path / "a.tif"
    _tif(path, 0)
    seg, seen = _seg()

    def sentinel(opener, _path, jobs, read_with):
        with opener(_path) as dataset:
            for job in jobs:
                bw, bh, win, a = read_with(dataset, job)
                yield job, (bw, bh, win, np.full_like(a, 9))

    monkeypatch.setattr("expds.pipeline._read_ahead_private", sentinel)
    _raster_with_model(seg, path, "r", "sp_or", block=4, overlap=0)
    assert seen == [9.0, 9.0, 9.0, 9.0]


def test_failed_mosaic_keeps_kurgans_from_every_network(tmp_path, monkeypatch):
    las = tmp_path / "a.las"
    las.write_bytes(b"x")
    reg = Region("r", tmp_path, las=[las])
    segs = [SimpleNamespace(meta={}, name="a"), SimpleNamespace(meta={}, name="b")]

    def li_with_model(seg, *_a, **_k):
        return [{"class_name": "kurgany_tselye", "confidence": 0.9, "markup_type": "li",
                 "region_name": "r", "source": seg.name, "geom": Point(0, 0).buffer(1)}]

    monkeypatch.setattr("expds.lidar.would_coarsen", lambda *_a, **_k: True)
    monkeypatch.setattr("expds.lidar.build_dem", lambda *_a, **_k: SimpleNamespace(crs="EPSG:32637", info={}))
    monkeypatch.setattr("expds.lidar.build_dem_mosaic", lambda *_a, **_k: None)
    monkeypatch.setattr("expds.pipeline._li_with_model", li_with_model)
    monkeypatch.setattr("expds.pipeline._cached_segmenters", lambda cache, *_a, **_k: cache["li"])
    guard = SimpleNamespace(ok=lambda *_a, **_k: True)
    out = lidar_feats(reg, {"lidar": {"mosaic": True}, "models": {"li": "m.pt"}}, guard,
                      {"li": segs, "li_mask": [], "li_fort": []}, tmp_path)
    assert sorted(f["source"] for f in out) == ["a", "b"]
    assert {f["threshold_group"] for f in out} == {"li_mosaic"}


def test_rejected_fp_filter_is_not_on_the_submission_path():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    text = (root / "expds" / "pipeline.py").read_text(encoding="utf-8")
    assert "lidar_fpf" not in text
    assert "fp_filter" not in (root / "configs" / "default.toml").read_text(encoding="utf-8")
