"""Rust-накопление сетки LAS побитно равно numpy-эталону (все 3 режима, 1 и много кусков)
и полный DTM с ним равен dtm.build конкурента (только для сверки, код не копируется)."""
import os, sys, time
import numpy as np
import laspy
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "py"))
import las_grid_fast as lg  # noqa: E402

D = "/workspace/data/train-expds-3"
SITES = [s for s in os.environ.get("LG_SITES", "016_Хога,101_Гнёздово_Днепровская,061_Граборг").split(",")]


def laz(site):
    name = site.split("_", 1)[1]
    return f"{D}/{site}/01_{name}_La_точки/01_{name}_La_точки.laz"


def geom(path, res=0.25):
    with laspy.open(path) as r:
        mn, mx = r.header.mins, r.header.maxs
    x0, y1 = np.floor(mn[0]), np.ceil(mx[1])
    W = int(np.ceil((mx[0] - x0) / res)) + 1
    H = int(np.ceil((y1 - mn[1]) / res)) + 1
    return x0, y1, res, W, H


def ref_accumulate(path, x0, y1, res, W, H, mode, chunk):
    zsum = np.zeros(H * W, np.float64); cnt = np.zeros(H * W, np.int32)
    zmin = np.full(H * W, np.inf, np.float32)
    with laspy.open(path) as r:
        for p in r.chunk_iterator(chunk):
            x, y, z = np.asarray(p.x), np.asarray(p.y), np.asarray(p.z, np.float32)
            c = np.asarray(p.classification)
            m = (c == 2) if mode == 0 else ~np.isin(c, (7, 18))
            x, y, z = x[m], y[m], z[m]
            ix = ((x - x0) / res).astype(np.int64); iy = ((y1 - y) / res).astype(np.int64)
            ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
            k = iy[ok] * W + ix[ok]
            if mode < 2:
                zsum += np.bincount(k, z[ok], minlength=H * W)
                cnt += np.bincount(k, minlength=H * W).astype(np.int32)
            else:
                np.minimum.at(zmin, k, z[ok])
    return zsum, cnt, zmin


@pytest.mark.parametrize("site", SITES)
@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("chunk", [5_000_000, 77_777])
@pytest.mark.parametrize("raw", [True, False])
def test_accumulate_bitexact(site, mode, chunk, raw, monkeypatch):
    monkeypatch.setattr(lg, "RAW", raw)
    p = laz(site); g = geom(p)
    a = lg.accumulate(p, *g, mode, chunk=chunk)
    b = ref_accumulate(p, *g, mode, chunk)
    for u, v in zip(a, b):
        if u.size == 0:
            continue
        assert u.dtype == v.dtype and u.tobytes() == v.tobytes()


@pytest.mark.parametrize("site", SITES)
def test_full_dtm_equal(site):
    comp = os.environ.get("COMP", "/workspace/koz3_shared/competitors/EhimenNathan-EXpedition")
    sys.path.insert(0, comp)
    from koz3 import dtm
    from scipy import ndimage as ndi
    p = laz(site)
    ref = dtm.build(p)
    x0, y1, res, W, H = geom(p)
    use_class = dtm.ground_fraction(p) >= 0.05
    all_ground = (not use_class) and dtm.is_ground_only(p)
    g = lg.ground_grid(p, x0, y1, res, W, H, use_class, all_ground)
    if not (use_class or all_ground):
        g = dtm.pmf(g, res)
    has = np.isfinite(g)
    dist, (iy, ix) = ndi.distance_transform_edt(~has, return_indices=True)
    out = g[iy, ix]
    if (~has).any():
        out = np.where(~has, ndi.gaussian_filter(out, 1.0), out)
    assert out.astype(np.float32).tobytes() == ref[0].tobytes()
    assert np.array_equal(dist * res <= dtm.FILL_M, ref[1])


def test_grid_mean_bitexact():
    import expds_fast
    rng = np.random.default_rng(0)
    cnt = rng.integers(0, 5, 100_003).astype(np.int32)
    zsum = rng.normal(150, 40, cnt.size) * cnt
    ref = np.where(cnt > 0, zsum / np.maximum(cnt, 1), np.nan).astype(np.float32)
    assert expds_fast.grid_mean(zsum, cnt).tobytes() == ref.tobytes()
