"""expds.speed (харвест speedup-2026-10-28, 29.10.2026): всё выкл. по умолчанию, приоритеты, равенство путей."""
import tomllib
from pathlib import Path

import numpy as np
import pytest

from expds import speed

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.delenv("EXPDS_SPEEDUP", raising=False)
    for f in speed.FLAGS:
        monkeypatch.delenv(f"EXPDS_SPEED_{f.upper()}", raising=False)
    speed.configure(None)
    yield
    speed.configure(None)


def test_proven_paths_run_when_the_section_is_off():
    speed.configure({"enabled": False})
    assert speed.use_fast("batch_circles") and speed.use_fast("fast_writer")
    assert speed.use_fast("meta_init") and speed.use_fast("gdal_threads")
    assert speed.use_fast("read_ahead") and not speed.use_fast("slot_accum")


def test_default_config_off():
    cfg = tomllib.loads((ROOT / "configs" / "default.toml").read_text(encoding="utf-8"))
    assert cfg["speed"] == {"enabled": False}
    assert speed.configure(cfg["speed"]) == []
    assert speed.active() == []


def test_umbrella_and_overrides(monkeypatch):
    assert speed.configure({"enabled": True}) == list(speed.UMBRELLA)
    assert not speed.on("slot_accum")                     # слоты — только явно
    speed.configure({"enabled": True, "fast_writer": False, "slot_accum": True})
    assert not speed.on("fast_writer") and speed.on("slot_accum")
    monkeypatch.setenv("EXPDS_SPEED_FAST_WRITER", "1")
    assert speed.on("fast_writer")
    monkeypatch.setenv("EXPDS_SPEEDUP", "0")
    assert not speed.on("read_ahead")
    speed.configure({"bogus": True, "read_ahead": "yes"})  # мусор — пропуск, выкл.
    monkeypatch.delenv("EXPDS_SPEEDUP")
    monkeypatch.delenv("EXPDS_SPEED_FAST_WRITER")
    assert speed.active() == []
    with pytest.raises(KeyError):
        speed.on("nope")


def test_gdal_threads(monkeypatch):
    monkeypatch.delenv("GDAL_NUM_THREADS", raising=False)
    speed.configure(None)
    import os

    assert "GDAL_NUM_THREADS" not in os.environ
    speed.configure({"gdal_threads": True})
    assert os.environ["GDAL_NUM_THREADS"] == "ALL_CPUS"


@pytest.mark.parametrize("shape", [(1, 5, 7), (2, 4, 4), (3, 2, 2), (4, 3, 3)])
def test_fix_channels_same(monkeypatch, shape):
    from expds.tiles import fix_channels

    img = np.random.default_rng(0).integers(0, 255, shape, dtype=np.uint8)
    monkeypatch.setenv("EXPDS_SPEED_FAST_CHANNELS", "0")
    a = fix_channels(img, 3)
    monkeypatch.setenv("EXPDS_SPEED_FAST_CHANNELS", "1")
    b = fix_channels(img, 3)
    assert a.dtype == b.dtype and np.array_equal(a, b)


@pytest.mark.parametrize("depth", [0, 1, 3])
def test_read_ahead_order(depth):
    from expds.pipeline import _read_ahead

    jobs = list(range(7))
    assert list(_read_ahead(jobs, lambda j: j * j, depth)) == [(j, j * j) for j in jobs]


def test_read_ahead_close_early():
    from expds.pipeline import _read_ahead

    g = _read_ahead(list(range(10)), lambda j: j)
    assert next(g) == (0, 0)
    g.close()


def test_batch_circles_flag_same(monkeypatch):
    import shapely

    from expds.centers import centers_to_features
    from rasterio.transform import from_origin

    probs = np.zeros((2, 64, 64), np.float32)
    probs[1, 10, 10] = probs[1, 40, 50] = probs[1, 20, 55] = 0.9
    tr = from_origin(500000, 6000000, 1.0, 1.0)

    def run():
        fs = centers_to_features(probs, ["bg", "finds_points"], tr, "EPSG:32637", 1.0, "r", "Ae", "x.tif")
        return [shapely.to_wkb(f["geom"]) for f in fs]

    monkeypatch.setenv("EXPDS_SPEED_BATCH_CIRCLES", "0")
    a = run()
    monkeypatch.setenv("EXPDS_SPEED_BATCH_CIRCLES", "1")
    assert a == run()


def test_private_readahead_matches_the_calling_thread(tmp_path):
    """Фоновое окно со своим dataset совпадает с чтением в этом потоке."""
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.windows import Window

    from expds.pipeline import _read_ahead_private
    from expds.raster import open_metric

    path = tmp_path / "a.tif"
    data = np.arange(64, dtype=np.uint8).reshape(1, 8, 8)
    with rasterio.open(path, "w", driver="GTiff", height=8, width=8, count=1, dtype="uint8",
                       crs="EPSG:32637", transform=from_origin(500000, 6100000, 1, 1)) as dst:
        dst.write(data)

    def read_with(dataset, job):
        y, x = job
        return dataset.read(1, window=Window(x, y, 4, 4))

    jobs = [(0, 0), (0, 4), (4, 0), (4, 4)]
    with open_metric(path) as dataset:
        sync = [read_with(dataset, job) for job in jobs]
    ahead = list(_read_ahead_private(open_metric, path, jobs, read_with))
    assert [job for job, _ in ahead] == jobs
    for left, (_job, right) in zip(sync, ahead):
        assert np.array_equal(left, right)
    early = _read_ahead_private(open_metric, path, jobs, read_with)
    assert next(early)[0] == (0, 0)
    early.close()
