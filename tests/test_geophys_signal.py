"""Режимы геофизики 25.10: каналы карты, аналитический сигнал, полярность, отсечка пиков.

По умолчанию все режимы совпадают со сдачей (first, abs, abs); новые включаются только конфигом.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from expds import accel, geophys

ROOT = Path(__file__).resolve().parents[1]


def dipole(n=200, res=0.1, sep_m=3.0, amp=10.0, seed=0):
    """Две гауссовы аномалии разного знака (не физический диполь) и слабый шум."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:n, :n] * res
    c = n * res / 2

    def g(x0, y0, s):
        return np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * s * s))

    f = amp * g(c, c + sep_m / 2, 0.5) - 0.6 * amp * g(c, c - sep_m / 2, 0.5)
    return (f + rng.normal(0, 0.05, f.shape)).astype(np.float32)


def test_defaults_are_the_shipped_detector():
    p = geophys.GeoParams()
    assert (p.band_mode, p.signal, p.polarity) == ("first", "abs", "abs")
    hp = dipole()
    assert np.array_equal(geophys.detection_image(hp, 0.1, "abs"), np.abs(hp))
    assert np.array_equal(geophys.polarity_image(hp, "abs"), np.abs(hp))


def gradiometer_dipole(n=300, res=0.1, depth=1.0, inc=70.0, h1=0.3, h2=1.3, noise=0.02, seed=0):
    """Вертикальный градиент ΔT точечного индуцированного диполя: два датчика на 0,3 и 1,3 м.

    Оси: x — север (вверх по растру), y — восток, z — вниз. Наклонение 70° — средняя полоса России.
    """
    i = np.radians(inc)
    t = np.array([np.cos(i), 0.0, np.sin(i)])
    rows, cols = np.mgrid[:n, :n] * res
    c = n * res / 2

    def total(hz):
        rx, ry, rz = c - rows, cols - c, -hz - depth
        r = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
        md = t[0] * rx + t[1] * ry + t[2] * rz
        b = [(3 * md * comp / r ** 2 - tc) / r ** 3 for comp, tc in ((rx, t[0]), (ry, t[1]), (rz, t[2]))]
        return b[0] * t[0] + b[1] * t[1] + b[2] * t[2]

    g = (total(h1) - total(h2)) / (h2 - h1)
    g = g / np.abs(g).max() * 10
    return (g + np.random.default_rng(seed).normal(0, noise * 10, g.shape)).astype(np.float32)


@pytest.mark.parametrize("depth", [0.3, 1.0, 1.5])
def test_gradiometer_dipole_peaks_over_the_source(depth):
    """Положительный полюс градиента при наклонении 70° доминирует: |hp| даёт один пик над предметом.

    Аналитический сигнал после продолжения вверх тоже ставит главный пик над источником;
    его шкала другая, поэтому порог для него подбирается отдельно (geo_tune --signals asa).
    """
    hp = gradiometer_dipole(depth=depth)
    r = round(2.0 / 0.1)
    rows, cols, v_abs = accel.find_peaks(geophys.detection_image(hp, 0.1, "abs"), r, 3.0)
    assert len(v_abs) == 1
    assert abs(rows[0] - 150) <= 0.3 * depth / 0.1 + 2 and abs(cols[0] - 150) <= 2
    rows, cols, v_asa = accel.find_peaks(geophys.detection_image(hp, 0.1, "asa"), r, 3.0)
    assert abs(rows[0] - 150) <= 3 and abs(cols[0] - 150) <= 3
    assert v_asa[0] > 10 * v_asa[1]


def test_polarity_keeps_one_sign():
    hp = dipole()
    pos, neg = geophys.polarity_image(hp, "pos"), geophys.polarity_image(hp, "neg")
    assert (pos >= 0).all() and (neg >= 0).all()
    assert np.allclose(pos - neg, hp)


def test_scalar_from_bands_gray_and_color():
    rng = np.random.default_rng(1)
    t = rng.random((50, 60)).astype(np.float32)
    gray = np.stack([t * 255] * 3)
    assert np.array_equal(geophys.scalar_from_bands(gray, "auto"), gray[0])
    # двухцветная шкала синий → красный: первый канал есть, но pca восстанавливает поле точнее при шуме
    color = np.stack([255 * t, 40 + 0 * t, 255 * (1 - t)])
    pca = geophys.scalar_from_bands(color, "pca")
    assert abs(np.corrcoef(pca.ravel(), t.ravel())[0, 1]) > 0.999
    auto = geophys.scalar_from_bands(color, "auto")
    assert np.allclose(auto, pca)
    color[:, 0, 0] = np.nan
    assert np.isnan(geophys.scalar_from_bands(color, "luma")[0, 0])
    with pytest.raises(ValueError):
        geophys.scalar_from_bands(color, "hue")


def test_peaks_above_k_do_not_depend_on_the_search_floor():
    """geo_tune ищет пики один раз с низкой отсечкой и режет по k: набор тот же."""
    img = np.abs(np.random.default_rng(2).normal(size=(120, 140))).astype(np.float32)
    lo = accel.find_peaks(img, 3, 0.5)
    hi = accel.find_peaks(img, 3, 1.5)
    keep = lo[2] >= np.float32(1.5)
    assert sorted(zip(lo[0][keep], lo[1][keep])) == sorted(zip(hi[0], hi[1]))


def _write_tif(path: Path, bands: np.ndarray, res=0.1, crs="EPSG:32636"):
    import rasterio
    from rasterio.transform import from_origin

    with rasterio.open(path, "w", driver="GTiff", height=bands.shape[1], width=bands.shape[2],
                       count=bands.shape[0], dtype=bands.dtype,
                       crs=crs, transform=from_origin(500000, 6100000, res, res)) as ds:
        ds.write(bands)


def test_geo_probe_runs_on_a_synthetic_survey(tmp_path):
    reg = tmp_path / "region_0001"
    reg.mkdir()
    f = dipole(n=300)
    u8 = np.clip(127 + f * 10, 0, 255).astype(np.uint8)
    _write_tif(reg / "01_x_MagGr_серый.tif", np.stack([u8] * 3))
    (reg / "UTM.json").write_text(json.dumps({"crs": "EPSG:32636"}), encoding="utf-8")
    out = tmp_path / "probe.json"
    res = subprocess.run([sys.executable, str(ROOT / "scripts" / "geo_probe.py"), "--root", str(tmp_path),
                          "--out", str(out)], capture_output=True, text=True, cwd=ROOT, check=False)
    assert res.returncode == 0, res.stderr
    rows = json.loads(out.read_text(encoding="utf-8"))
    assert len(rows) == 1 and rows[0]["bands"] == 3 and rows[0]["channels_differ"] == 0.0
    assert rows[0]["region"] == "region_000" and rows[0]["peaks"]["0.5"] >= rows[0]["peaks"]["2.0"] >= 1


def test_geo_predict_override_and_run(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("geo_predict", ROOT / "scripts" / "geo_predict.py")
    gp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gp)
    cfg = {"geophys": {"maggr": {"finds_points": {"peak_k": 1.0}}}}
    gp.apply_override(cfg, "geophys.maggr.finds_points.peak_k=0.75")
    gp.apply_override(cfg, 'geophys.band_mode="auto"')
    assert cfg["geophys"]["maggr"]["finds_points"]["peak_k"] == 0.75 and cfg["geophys"]["band_mode"] == "auto"
    with pytest.raises(ValueError):
        gp.apply_override(cfg, "без_знака")

    reg = tmp_path / "001_Тест_1"
    reg.mkdir()
    u8 = np.clip(127 + gradiometer_dipole(n=300) * 10, 0, 255).astype(np.uint8)
    _write_tif(reg / "01_x_MagGr_серый.tif", np.stack([u8]))
    (reg / "UTM.json").write_text(json.dumps({"crs": "EPSG:32636"}), encoding="utf-8")
    out = tmp_path / "geo.geojson"
    res = subprocess.run([sys.executable, str(ROOT / "scripts" / "geo_predict.py"), "--root", str(tmp_path),
                          "--out", str(out)], capture_output=True, text=True, cwd=ROOT, check=False)
    assert res.returncode == 0, res.stderr
    fc = json.loads(out.read_text(encoding="utf-8"))
    assert any(f["properties"]["class_name"] == "finds_points" for f in fc["features"])


def test_site_overlap_finds_the_neighbouring_train_site(tmp_path):
    train, val = tmp_path / "train", tmp_path / "val"
    near = train / "001_Гнёздово_1"
    far = train / "002_Далёкое_1"
    v = val / "region_0001"
    for d in (near, far, v):
        d.mkdir(parents=True)
        (d / "UTM.json").write_text(json.dumps({"crs": "EPSG:32636"}), encoding="utf-8")
    u8 = np.zeros((1, 50, 50), np.uint8)
    _write_tif(near / "01_x_MagGr_серый.tif", u8)
    import rasterio
    from rasterio.transform import from_origin

    def put(path, x0, y0):
        with rasterio.open(path, "w", driver="GTiff", height=50, width=50, count=1, dtype="uint8",
                           crs="EPSG:32636", transform=from_origin(x0, y0, 0.1, 0.1)) as ds:
            ds.write(u8)

    put(far / "01_x_MagGr_серый.tif", 560000, 6150000)
    put(v / "01_x_MagGr_серый.tif", 500010, 6100000)        # в 5 м от рамки «Гнёздово_1»
    out = tmp_path / "o.json"
    res = subprocess.run([sys.executable, str(ROOT / "scripts" / "site_overlap.py"), "--train", str(train),
                          "--val", str(val), "--out", str(out)], capture_output=True, text=True, cwd=ROOT,
                         check=False)
    assert res.returncode == 0, res.stderr
    rows = json.loads(out.read_text(encoding="utf-8"))
    assert rows[0]["region"] == "val_000" and rows[0]["family"] == "Гнёздово" and rows[0]["gap_m"] < 50


def test_layer_copies_fill_other_slices_once():
    from shapely.geometry import Point

    def f(x, src, conf=0.5, crs="EPSG:32636"):
        return {"geom": Point(x, 0).buffer(1.0), "class_name": "object_poly", "confidence": conf,
                "region_name": "r", "markup_type": "etgr", "source": src, "original_crs": crs}

    feats = [f(0, "карта_1_0.75.tif", 0.9), f(0, "карта_2_1.0.tif", 0.6), f(10, "карта_2_1.0.tif", 0.4),
             f(50, "профиль_1.tif", 0.7, crs=None)]
    sources = {"карта_1_0.75.tif": "EPSG:32636", "карта_2_1.0.tif": "EPSG:32636",
               "карта_3_1.25.tif": "EPSG:32636", "профиль_1.tif": None}
    out = geophys.layer_copies(feats, sources, decay=0.5, group="etgr_copy")
    where = sorted((c["source"], round(c["geom"].centroid.x)) for c in out)
    # x=0 уже есть в срезах 1 и 2 → копия только в срез 3 (одна, от самого уверенного);
    # x=10 из среза 2 → в срезы 1 и 3; профиль без CRS не смешивается с картами
    assert where == [("карта_1_0.75.tif", 10), ("карта_3_1.25.tif", 0), ("карта_3_1.25.tif", 10)]
    top = next(c for c in out if c["source"] == "карта_3_1.25.tif" and round(c["geom"].centroid.x) == 0)
    assert top["confidence"] == 0.45 and top["threshold_group"] == "etgr_copy" and top["copied_from"] == "карта_1_0.75.tif"
    assert geophys.layer_copies(feats, {"один.tif": "EPSG:32636"}) == []


def test_layer_copy_is_off_by_default_in_the_pipeline(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from shapely.geometry import Point

    from expds import pipeline

    files = [tmp_path / "x_EtGr_карта_1_0.75.tif", tmp_path / "x_EtGr_карта_2_1.0.tif"]
    reg = SimpleNamespace(name="r", las=[], rasters={"etgr": files}, boundaries=[], noise=[], utm_json=None)

    def polys(path, region, mt, p, **kw):
        return [{"geom": Point(0, 0).buffer(1), "class_name": "object_poly", "confidence": 0.9,
                 "region_name": region, "markup_type": mt, "source": Path(path).name,
                 "original_crs": "EPSG:32636"}] if "карта_1" in Path(path).name else []

    monkeypatch.setattr(pipeline.geophys, "object_polys", polys)
    monkeypatch.setattr(pipeline.geophys, "load_mask", lambda *a, **k: None)
    monkeypatch.setattr(pipeline.geophys, "source_crs", lambda path, *a, **k: "EPSG:32636")
    off = pipeline.geophys_feats(reg, {"geophys": {}}, pipeline.TimeGuard(1000))
    on = pipeline.geophys_feats(reg, {"geophys": {"etgr": {"layer_copy": True}}}, pipeline.TimeGuard(1000))
    assert len(off) == 1 and len(on) == 2 and on[1]["threshold_group"] == "etgr_copy"
