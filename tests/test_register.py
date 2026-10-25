"""Сдвиг слоя разметки относительно лидара (expds.register, scripts/layer_shift.py, oof_transform shift)."""
import importlib.util
import json

import numpy as np
from conftest import ROOT, UTM, X0, Y0
from shapely.geometry import Point

from expds import register
from expds.crs import WEB_MERCATOR, to_crs

SHIFT = (6.0, -4.0)


def field(seed: int = 0, n: int = 80, size: float = 300.0, spacing: float = 9.0) -> np.ndarray:
    """Густое поле курганов: точки не ближе spacing метров."""
    rng = np.random.default_rng(seed)
    pts: list[np.ndarray] = []
    while len(pts) < n:
        p = rng.uniform(0, size, 2)
        if all(np.hypot(*(p - q)) >= spacing for q in pts):
            pts.append(p)
    return np.array(pts)


def shifted_layer(ref: np.ndarray, seed: int = 1, keep: float = 0.7, noise: float = 0.7, clutter: int = 20):
    rng = np.random.default_rng(seed)
    sub = ref[rng.random(len(ref)) < keep]
    dst = sub + np.array(SHIFT) + rng.normal(0, noise, sub.shape)
    return np.vstack([dst, rng.uniform(0, 300, (clutter, 2))])


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_estimate_shift_recovers_systematic_offset_in_dense_field():
    ref = field()
    s = register.estimate_shift(ref, shifted_layer(ref))
    assert s.systematic and s.n_pairs >= 40
    assert abs(s.dx - SHIFT[0]) < 0.5 and abs(s.dy - SHIFT[1]) < 0.5


def test_no_offset_is_not_systematic():
    ref = field()
    rng = np.random.default_rng(2)
    s = register.estimate_shift(ref, ref + rng.normal(0, 0.7, ref.shape))
    assert not s.systematic and s.norm < 1.0


def test_contain_rate_counts_points_inside_after_shift():
    ref = field()
    polys = [Point(x + SHIFT[0], y + SHIFT[1]).buffer(4.0) for x, y in ref]
    assert register.contain_rate(ref, polys) < 0.1                # 7,2 м сдвига при радиусе 4 м
    assert register.contain_rate(ref, polys, SHIFT) == 1.0


def gt_collection(ref: np.ndarray, dst: np.ndarray, region: str = "001_Тест") -> dict:
    feats = []
    for mt, pts in (("li", ref), ("ae", dst)):
        for x, y in pts:
            g = to_crs(Point(X0 + x, Y0 + y).buffer(4.0), UTM, WEB_MERCATOR)
            feats.append({"type": "Feature", "geometry": g.__geo_interface__,
                          "properties": {"class_name": "kurgany_povrezhdennye", "markup_type": mt,
                                         "region_name": region}})
    return {"type": "FeatureCollection", "features": feats}


def test_layer_shift_script_reports_copies_inside_after_shift(tmp_path):
    ref = field()
    path = tmp_path / "gt.geojson"
    path.write_text(json.dumps(gt_collection(ref, shifted_layer(ref, clutter=0, noise=0.3))), encoding="utf-8")
    ls = load_script("layer_shift")
    layers = ls.load_layers(str(path), ls.KURGANS)
    row = ls.compare(layers["001_Тест"]["li"], layers["001_Тест"]["ae"], 30.0, 6.0)
    assert row["systematic"]
    assert abs(row["dx"] - SHIFT[0]) < 0.5 and abs(row["dy"] - SHIFT[1]) < 0.5
    assert row["inside_before"] < 0.1 and row["inside_after"] > 0.6   # доля ref, у которых есть пара в dst


def test_oof_shift_moves_only_the_chosen_layer():
    ot = load_script("oof_transform")
    g = to_crs(Point(X0 + 10, Y0 + 10).buffer(2.0), UTM, WEB_MERCATOR)
    feats = [{"geom": g, "markup_type": "ae", "region_name": "r", "class_name": "kurgany_tselye"},
             {"geom": g, "markup_type": "li", "region_name": "r", "class_name": "kurgany_tselye"}]
    out, moved = ot.shift_feats(feats, {"r": SHIFT}, "ae")
    assert moved == 1 and out[1]["geom"].equals(g)
    a = to_crs(out[0]["geom"], WEB_MERCATOR, UTM).centroid
    assert abs(a.x - (X0 + 10 + SHIFT[0])) < 0.05 and abs(a.y - (Y0 + 10 + SHIFT[1])) < 0.05


def blob_photo(path, centers, res: float = 0.5, size: float = 300.0, radius: float = 5.0, seed: int = 3):
    """Снимок UTM: светлые пятна распаханных курганов на шумном поле."""
    import rasterio
    from rasterio.transform import from_origin

    rng = np.random.default_rng(seed)
    n = int(size / res)
    yy, xx = (np.mgrid[0:n, 0:n] + 0.5) * res                  # yy — от верхнего края вниз
    img = 100 + rng.normal(0, 5, (n, n))
    for cx, cy in centers:                                      # cy — от нижнего края вверх
        img += 40 * np.exp(-((xx - cx) ** 2 + (yy - (size - cy)) ** 2) / (2 * (radius / 1.5) ** 2))
    with rasterio.open(path, "w", driver="GTiff", height=n, width=n, count=1, dtype="uint8",
                       transform=from_origin(X0, Y0 + size, res, res), crs=UTM) as ds:
        ds.write(np.clip(img, 1, 255).astype(np.uint8), 1)
    return path


def label_geoms(centers, shift=(0.0, 0.0)):
    return [to_crs(Point(X0 + x + shift[0], Y0 + y + shift[1]).buffer(5.0), UTM, WEB_MERCATOR) for x, y in centers]


def test_image_label_shift_finds_labels_off_the_photo(tmp_path):
    centers = field(seed=4, n=40, spacing=20.0)
    path = blob_photo(tmp_path / "ae.tif", centers)
    ils = load_script("image_label_shift")
    row = ils.scan_raster(path, label_geoms(centers, (-5.0, 3.0)), 1.0, 12.0, None)
    assert row["systematic"], row
    assert abs(row["dx_m"] - 5.0) <= 1.0 and abs(row["dy_m"] - 3.0) <= 1.0   # на восток и на юг


def test_image_label_shift_quiet_when_labels_sit_on_blobs(tmp_path):
    centers = field(seed=4, n=40, spacing=20.0)
    path = blob_photo(tmp_path / "ae.tif", centers)
    row = load_script("image_label_shift").scan_raster(path, label_geoms(centers), 1.0, 12.0, None)
    assert not row["systematic"] and row["shift_m"] <= 1.0 and row["z_zero"] > 4


def test_cache_label_shift_moves_east_and_south_in_ground_meters(tmp_path):
    from shapely.geometry import box

    from expds.tiles import load_label_shifts, shift_labels

    g = Point(X0, Y0).buffer(1.0)
    out = shift_labels([(g, "k")], (5.0, 3.0), UTM, box(X0 - 10, Y0 - 10, X0 + 10, Y0 + 10))
    c = out[0][0].centroid
    assert abs(c.x - (X0 + 5)) < 1e-6 and abs(c.y - (Y0 - 3)) < 1e-6
    g3 = to_crs(g, UTM, WEB_MERCATOR)
    out3 = shift_labels([(g3, "k")], (5.0, 0.0), "EPSG:3857", g3.buffer(50).envelope)
    back = to_crs(out3[0][0], WEB_MERCATOR, UTM).centroid
    assert abs(back.x - (X0 + 5)) < 0.1 and abs(back.y - Y0) < 0.1     # метры на земле, не единицы 3857
    gdeg = Point(37.6, 55.75)
    outdeg = shift_labels([(gdeg, "k")], (100.0, 0.0), "EPSG:4326", gdeg.buffer(0.02).envelope)
    from pyproj import Transformer
    fwd = Transformer.from_crs(4326, 32637, always_xy=True)
    backd = Transformer.from_crs(32637, 4326, always_xy=True)
    ux, uy = fwd.transform(37.6, 55.75)
    elon, elat = backd.transform(ux + 100.0, uy)
    assert abs(outdeg[0][0].x - elon) < 1e-5 and abs(outdeg[0][0].y - elat) < 1e-5
    p = tmp_path / "shifts.json"
    p.write_text(json.dumps({"rasters": [{"file": "a.tif", "dx_m": 5.0, "dy_m": 3.0, "systematic": True},
                                         {"file": "b.tif", "dx_m": 9.0, "dy_m": 0.0, "systematic": False}]}),
                 encoding="utf-8")
    assert load_label_shifts(str(p)) == {"a.tif": (5.0, 3.0)}


def li_kurgans(centers):
    out = []
    for x, y in centers:
        g = to_crs(Point(X0 + x, Y0 + y).buffer(1.5), UTM, WEB_MERCATOR)
        for cls, conf in (("kurgany_povrezhdennye", 0.995), ("kurgany_tselye", 0.004)):
            out.append({"geom": g, "class_name": cls, "confidence": conf, "markup_type": "li",
                        "threshold_group": None, "source": "a.las", "region_name": "r"})
    return out


def test_li_to_ae_register_moves_copies_onto_the_photo(tmp_path):
    from expds.fusion import FusionParams, li_to_ae

    centers = field(seed=5, n=40, spacing=20.0)
    photo_offset = (4.0, -5.0)                                  # снимок привязан на 4 м восточнее, 5 м южнее
    path = blob_photo(tmp_path / "ae.tif", [(x + photo_offset[0], y + photo_offset[1]) for x, y in centers])
    feats = li_kurgans(centers)
    plain = li_to_ae(feats, [path], FusionParams())
    moved = li_to_ae(feats, [path], FusionParams(register=True))
    assert len(plain) == len(moved) == 80
    a = to_crs(plain[0]["geom"], WEB_MERCATOR, UTM).centroid
    b = to_crs(moved[0]["geom"], WEB_MERCATOR, UTM).centroid
    assert abs((b.x - a.x) - photo_offset[0]) <= 1.0 and abs((b.y - a.y) - photo_offset[1]) <= 1.0


def test_li_to_ae_register_leaves_aligned_photo_alone(tmp_path):
    from expds.fusion import FusionParams, li_to_ae

    centers = field(seed=5, n=40, spacing=20.0)
    path = blob_photo(tmp_path / "ae.tif", centers)
    feats = li_kurgans(centers)
    plain = li_to_ae(feats, [path], FusionParams())
    moved = li_to_ae(feats, [path], FusionParams(register=True))
    assert all(p["geom"].equals(m["geom"]) for p, m in zip(plain, moved))


def test_oof_shift_reads_both_report_formats(tmp_path):
    ot = load_script("oof_transform")
    a = tmp_path / "layers.json"
    a.write_text(json.dumps({"report": {"li:ae": {"regions": [
        {"region": "r", "dx": 1.0, "dy": 2.0, "systematic": True},
        {"region": "q", "dx": 9.0, "dy": 9.0, "systematic": False}]}}}), encoding="utf-8")
    assert ot.load_shifts(a, "li:ae", "ae") == {"r": (1.0, 2.0)}
    b = tmp_path / "image.json"
    b.write_text(json.dumps({"rasters": [
        {"region": "r", "markup": "ae", "n": 10, "dx_m": 4.0, "dy_m": 5.0, "systematic": True},
        {"region": "r", "markup": "ae", "n": 30, "dx_m": 3.0, "dy_m": 6.0, "systematic": True},
        {"region": "r", "markup": "sp_or", "n": 50, "dx_m": 8.0, "dy_m": 0.0, "systematic": True}]}),
        encoding="utf-8")
    assert ot.load_shifts(b, "li:ae", "ae") == {"r": (3.0, -6.0)}          # юг → минус север


def test_image_label_shift_runs_on_a_set_without_ground_truth(tmp_path):
    """Публичная выборка: эталона нет, якоря — курганы лидара из своего result.geojson."""
    import subprocess
    import sys

    centers = field(seed=6, n=40, spacing=20.0)
    reg = tmp_path / "set" / "region_0001"
    reg.mkdir(parents=True)
    (reg / "UTM.json").write_text(json.dumps({"utm": "36N"}), encoding="utf-8")
    blob_photo(reg / "01_x_Ae_аэрофото.tif", [(x + 6.0, y - 3.0) for x, y in centers])
    feats = [{"type": "Feature", "geometry": g.__geo_interface__,
              "properties": {"class_name": "kurgany_povrezhdennye", "markup_type": "li", "confidence": 0.9,
                             "region_name": "region_0001"}} for g in label_geoms(centers)]
    labels = tmp_path / "result.geojson"
    labels.write_text(json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8")
    out = tmp_path / "shift.json"
    run = subprocess.run([sys.executable, str(ROOT / "scripts" / "image_label_shift.py"), "--root", str(tmp_path / "set"),
                          "--labels", str(labels), "--labels-min", "0.5", "--radius", "5", "--markups", "ae",
                          "--max-shift", "12", "--out", str(out)], capture_output=True, text=True, timeout=300, check=False)
    assert run.returncode == 0, run.stderr
    row = json.loads(out.read_text(encoding="utf-8"))["rasters"][0]
    assert row["systematic"] and abs(row["dx_m"] - 6.0) <= 1.0 and abs(row["dn_m"] + 3.0) <= 1.0
