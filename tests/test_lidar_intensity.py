"""Канал int: интенсивность отражения точек земли из LAS (приём mlart, этап «Сканирование»)."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np
from conftest import UTM, X0, Y0

from expds import accel
from expds.lidar import Dem, build_dem, relief_channels


@contextmanager
def numpy_backend():
    saved = accel._rs
    accel._rs = None
    try:
        yield
    finally:
        accel._rs = saved


def write_bright_stripe(path: Path, n: int = 60_000, seed: int = 3) -> Path:
    """Ровная земля 60×60 м: полоса x ∈ [20, 40) м ярче остальной земли, растительность — ярче всех."""
    import laspy
    from pyproj import CRS

    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 60, n)
    y = rng.uniform(0, 60, n)
    keep = ~((x > 45) & (x < 48) & (y > 5) & (y < 8))      # дыра в покрытии
    x, y = x[keep], y[keep]
    z = 150.0 + 0.01 * x
    inten = np.where((x >= 20) & (x < 40), 3000, 1000).astype(np.uint16)
    cls = np.full(x.size, 2, np.uint8)
    veg = rng.random(x.size) < 0.1
    cls[veg] = 5
    z = np.where(veg, z + 5.0, z)
    inten[veg] = 60000
    hdr = laspy.LasHeader(point_format=6, version="1.4")
    hdr.scales = np.array([0.001, 0.001, 0.001])
    hdr.offsets = np.array([X0, Y0, 0.0])
    hdr.add_crs(CRS.from_user_input(UTM))
    las = laspy.LasData(hdr)
    las.x, las.y, las.z = x + X0, y + Y0, z
    las.classification = cls
    las.intensity = inten
    path.parent.mkdir(parents=True, exist_ok=True)
    las.write(str(path))
    return path


def test_int_channel_reads_ground_intensity(tmp_path):
    las = write_bright_stripe(tmp_path / "s.las")
    d = build_dem(las, res=0.1, fill="delaunay", crs=UTM)
    assert d.info["las_path"] == str(las)
    img, tr = relief_channels(d, 0.5, ("lrm_s", "lrm_l", "slope", "int"))
    base, tb = relief_channels(d, 0.5)
    assert img.shape[0] == 4 and np.array_equal(img[:3], base) and tr == tb
    ch = img[3].astype(int)
    x = tr.c + (np.arange(ch.shape[1]) + 0.5) * tr.a - X0     # центр столбца, м от начала участка
    bright = (x >= 21) & (x < 39)
    dark = (x < 18) | (x > 42)
    inside = ch > 0
    # растительность (60000) в канал не попадает: иначе шкала по перцентилям прижала бы полосу к нулю
    assert np.median(ch[:, bright][inside[:, bright]]) > 200
    assert np.median(ch[:, dark][inside[:, dark]]) < 60
    # нулей внутри контура съёмки нет: пустые ячейки заполнены соседями
    assert (img[3][img[0] > 0] > 0).all()


def test_int_channel_same_on_numpy_path(tmp_path):
    las = write_bright_stripe(tmp_path / "s.las")
    d = build_dem(las, res=0.1, fill="delaunay", crs=UTM)
    a, _ = relief_channels(d, 0.5, ("lrm_s", "int", "slope"))
    with numpy_backend():
        b, _ = relief_channels(d, 0.5, ("lrm_s", "int", "slope"))
    assert a.shape == b.shape == (3,) + a.shape[1:]
    assert np.array_equal(a[1], b[1])


def test_int_channel_without_las_is_constant():
    dem = np.full((40, 40), 100.0, np.float32)
    d = Dem(dem=dem, left=0.0, top=40.0, res=0.1, crs=UTM)
    img, _ = relief_channels(d, 0.5, ("lrm_s", "int"))
    assert set(np.unique(img[1]).tolist()) == {128}


def test_int_channel_inserted_after_fused_rust_channels(tmp_path):
    """Rust собирает каналы рельефа одним вызовом, int досчитывается и встаёт на своё место."""
    las = write_bright_stripe(tmp_path / "s.las")
    d = build_dem(las, res=0.1, fill="delaunay", crs=UTM)
    with numpy_backend():
        ref, tref = relief_channels(d, 0.5, ("lrm_s", "int", "slope"))

    class FakeRust:
        def __init__(self):
            self.calls = []

        def relief_channels(self, dem, src_res, res, tags, *params):
            self.calls.append(list(tags))
            with numpy_backend():
                img, tr = relief_channels(Dem(dem=dem, left=d.left, top=d.top, res=src_res), res, tuple(tags))
            return img, tr.a

        def box_mean(self, z, radius):
            with numpy_backend():
                return accel.box_mean(z, radius)

    fake = FakeRust()
    saved = accel._rs
    accel._rs = fake
    try:
        got, tgot = relief_channels(d, 0.5, ("lrm_s", "int", "slope"))
    finally:
        accel._rs = saved
    assert fake.calls == [["lrm_s", "slope"]]
    assert tgot == tref and np.array_equal(got, ref)
