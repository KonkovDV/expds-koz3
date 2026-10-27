"""Сборка эталона и кэша тайлов на синтетическом «трейне» с именами как в датасете."""
import csv
import json

import numpy as np
import pytest
from shapely.geometry import Point, mapping

from conftest import MOUNDS, UTM, X0, Y0, write_las
from expds import gt as gtmod
from expds import tiles
from expds.catalog import scan_root
from expds.crs import WEB_MERCATOR, to_crs
from expds.lidar import MODEL_DEM, RELIEF_TAGS, build_dem, hillshade_set, relief_channels, write_set


def make_train(root):
    site = root / "001_Тест_Курганы"
    las = write_las(site / "01_Тест_Курганы_La_точки" / "01_Тест_Курганы_La_точки.las")
    (site / "UTM.json").write_text(json.dumps({"utm": "36N"}), encoding="utf-8")
    d = build_dem(las, res=0.1)
    write_set(hillshade_set(d, tags=("g", "c")), d, site / "02_Тест_Курганы_Li_карты", "01_Тест_Курганы_Li")
    ann = site / "06_Тест_Курганы_разметка"
    ann.mkdir(parents=True)
    # Li: два кургана в UTM; Ае (кириллица): третий курган в EPSG:3857 + дубль первого
    li = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {}, "geometry": mapping(Point(X0 + mx, Y0 + my).buffer(r / 2))}
        for mx, my, _, r in MOUNDS[:2]]}
    (ann / "Тест_Курганы_Li_курганы_целые.geojson").write_text(json.dumps(li), encoding="utf-8")
    ae = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {},
         "geometry": mapping(to_crs(Point(X0 + mx, Y0 + my).buffer(r / 2), UTM, WEB_MERCATOR))}
        for mx, my, _, r in (MOUNDS[2], MOUNDS[0])]}
    (ann / "Тест_Курганы_Ае_курганы_поврежденные.geojson").write_text(json.dumps(ae), encoding="utf-8")
    return root


def test_gt_and_stats(tmp_path):
    regions = scan_root(make_train(tmp_path / "train"))
    assert len(regions) == 1 and len(regions[0].annotations) == 2
    assert {a[2] for a in regions[0].annotations} == {"li", "ae"}
    items, info = gtmod.build_gt(regions)
    assert len(items) == 4
    st = gtmod.stats(items)
    by = {(o["class_name"], o["markup_type"]): o["objects"] for o in st["objects"]}
    assert by[("kurgany_tselye", "li")] == 2 and by[("kurgany_povrezhdennye", "ae")] == 2
    # Разные классы — не дубли; дубли считаются внутри одного класса
    assert st["cross_modality_pairs"] == []
    fc = gtmod.to_feature_collection(items)
    xs = [Point(*f["geometry"]["coordinates"][0][0]).x for f in fc["features"]]
    assert min(abs(x) for x in xs) > 1e6      # всё в EPSG:3857


def _gt_file(root, tmp_path):
    items, _ = gtmod.build_gt(scan_root(root))
    gt_path = tmp_path / "gt.geojson"
    gt_path.write_text(json.dumps(gtmod.to_feature_collection(items)), encoding="utf-8")
    return gt_path


def test_tile_cache_li_tif(tmp_path):
    root = make_train(tmp_path / "train")
    meta = tiles.build_cache(str(root), str(_gt_file(root, tmp_path)), "li", str(tmp_path / "cache"),
                             res=0.5, tile=64, li_tags=["g", "c"], li_source="tif")
    assert meta["channels"] == 4 and meta["positive"] >= 2 and not meta["flips"]
    imgs = np.load(tmp_path / "cache" / "images_0000.npy")
    masks = np.load(tmp_path / "cache" / "masks_0000.npy")
    assert imgs.shape[1:] == (4, 64, 64) and imgs.dtype == np.uint8
    bit = meta["classes"].index("kurgany_tselye")
    assert ((masks >> bit) & 1).sum() > 0


def test_tile_cache_only_classes(tmp_path):
    root = make_train(tmp_path / "train")
    meta = tiles.build_cache(str(root), str(_gt_file(root, tmp_path)), "li", str(tmp_path / "cache"), res=0.5,
                             tile=64, only_classes=["kurgany_tselye"])
    assert meta["classes"] == ["kurgany_tselye"]
    masks = np.load(tmp_path / "cache" / "masks_0000.npy")
    assert masks.max() == 1                                   # единственный класс — бит 0
    with pytest.raises(ValueError):
        tiles.build_cache(str(root), str(_gt_file(root, tmp_path)), "li", str(tmp_path / "c2"), res=0.5, tile=64,
                          only_classes=["finds_points"])


def test_min_area_drops_small_mounds(tmp_path):
    root = make_train(tmp_path / "train")
    gt = str(_gt_file(root, tmp_path))
    # Два целых кургана лидара: радиусы 2,5 и 2 м, площади около 20 и 13 м².
    kept = tiles.build_cache(str(root), gt, "li", str(tmp_path / "big"), res=0.5, tile=64,
                             min_area_m2=16)
    none = tiles.build_cache(str(root), gt, "li", str(tmp_path / "none"), res=0.5, tile=64,
                             min_area_m2=10_000)
    assert kept["min_area_m2"] == 16 and kept["positive"] >= 1
    assert none["positive"] == 0


def test_ground_area_ignores_mercator_stretch():
    g = Point(X0, Y0).buffer(10)
    merc = to_crs(g, UTM, WEB_MERCATOR)
    utm_area = tiles.ground_area_m2(g, UTM)
    merc_area = tiles.ground_area_m2(merc, WEB_MERCATOR)
    assert abs(utm_area - 3.1416 * 100) < 5
    assert abs(merc_area - utm_area) / utm_area < 0.05
    assert merc.area > utm_area * 2


def test_shrink_separates_touching_mounds():
    a, b = Point(0, 0).buffer(5), Point(10, 0).buffer(5)
    assert a.buffer(1e-6).intersects(b)
    sa, sb = tiles.shrink_polygon(a, 0.4), tiles.shrink_polygon(b, 0.4)
    assert not sa.intersects(sb) and sa.within(a)
    assert abs(sa.area / a.area - 0.36) < 0.02
    assert tiles.shrink_polygon(Point(1, 1), 0.4).equals(Point(1, 1))
    assert tiles.shrink_polygon(a, 0.0) is a


def test_tile_cache_li_las_matches_inference(tmp_path):
    """Тайлы из LAS — точные куски тех каналов, которые инференс подаёт модели."""
    root = make_train(tmp_path / "train")
    cache = tmp_path / "cache"
    meta = tiles.build_cache(str(root), str(_gt_file(root, tmp_path)), "li", str(cache), res=0.5, tile=64)
    assert meta["li_source"] == "las" and meta["li_tags"] == list(RELIEF_TAGS)
    assert meta["channels"] == 3 and meta["flips"] and meta["positive"] >= 2
    imgs = np.load(cache / "images_0000.npy")
    masks = np.load(cache / "masks_0000.npy")
    bit = meta["classes"].index("kurgany_tselye")
    assert ((masks >> bit) & 1).sum() > 0
    las = next(root.rglob("*.las"))
    assert (meta["dem_res"], meta["dem_fill"]) == (MODEL_DEM["res"], MODEL_DEM["fill"])
    full, tr = relief_channels(build_dem(las, res=meta["dem_res"], fill=meta["dem_fill"], crs=UTM), 0.5)
    assert full.dtype == np.uint8 and full[:, full.any(axis=0)].min() >= 1
    rows = list(csv.DictReader((cache / "index.csv").open(encoding="utf-8")))
    for r in rows:
        tile = imgs[int(r["idx"])]
        col = round((float(r["cx"]) - tr.c) / tr.a - 32)
        row = round((tr.f - float(r["cy"])) / tr.a - 32)
        found = False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                ref, _ = tiles.cut_window(full, tr, tr.c + (col + dc + 32) * tr.a,
                                          tr.f - (row + dr + 32) * tr.a, 64)
                found |= np.array_equal(ref, tile)
        assert found, r


def test_lidar_center_checkpoint_emits_circles():
    """Чекпойнт target=centers на LAS отдаёт круг, а не маску пятна."""
    from expds.lidar import Dem
    from expds.pipeline import _li_with_model

    class Seg:
        meta = {"li_source": "las", "li_tags": ["lrm_s", "lrm_l", "slope"], "target": "centers",
                "center_radius_m": 12.0, "min_dist_m": 4.0}
        res = 0.5
        classes = ["gorodishcha", "kurgany_povrezhdennye"]

        def predict(self, img):
            p = np.zeros((2, img.shape[1], img.shape[2]), np.float32)
            p[0, img.shape[1] // 2, img.shape[2] // 2] = 0.8
            return p

    z = np.full((40, 40), 10.0, np.float32)
    d = Dem(z, left=500000.0, top=6000000.0, res=0.5, crs=UTM)
    feats = _li_with_model(Seg(), d, "001_Тест", "x.las", hillshade_set, cand_p=object())
    assert len(feats) == 1
    assert feats[0]["class_name"] == "gorodishcha" and feats[0]["markup_type"] == "li"
    assert feats[0]["confidence"] == pytest.approx(0.8)
    assert feats[0]["geom"].geom_type == "Polygon" and feats[0]["geom"].area > 100


def test_li_las_model_runs_in_pipeline(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")
    from expds.infer_nn import Segmenter
    from expds.pipeline import _li_with_model
    from expds.train import train

    root = make_train(tmp_path / "train")
    cache = tmp_path / "cache"
    tiles.build_cache(str(root), str(_gt_file(root, tmp_path)), "li", str(cache), res=0.5, tile=64)
    model = tmp_path / "li.pt"
    train(str(cache), str(model), epochs=1, batch=4, workers=0, pretrained=False)
    seg = Segmenter(model, device="cpu")
    assert seg.meta["li_source"] == "las"
    d = build_dem(next(root.rglob("*.las")), crs=UTM)
    feats = _li_with_model(seg, d, "001_Тест_Курганы", "x.las", hillshade_set)
    assert all(f["geom"].geom_type == "Polygon" and f["markup_type"] == "li" for f in feats)

    from expds import candidates

    p = candidates.CandidateParams()
    n = len(candidates.detect(d, p)["x"])
    feats = _li_with_model(seg, d, "001_Тест_Курганы", "x.las", hillshade_set, p)
    kur = [f for f in feats if f["class_name"].startswith("kurgany")]
    assert n > 0 and len(kur) == 2 * n
    assert all(0.0 <= f["confidence"] <= 1.0 for f in kur)


def _write_strip(path, data, step=1.0):
    import rasterio
    from rasterio.transform import from_origin

    n, h, w = data.shape
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=n, dtype=data.dtype.name,
                       crs=UTM, transform=from_origin(X0, Y0 + h * step, step, step),
                       tiled=False, blockxsize=w, blockysize=1) as dst:
        dst.write(data)


def test_strip_grid_matches_the_tiled_read_and_keeps_the_pixel_type(tmp_path):
    """Полосатый файл: окно в рабочем шаге совпадает с обычным чтением, тип пикселя тот же."""
    import rasterio

    rng = np.random.default_rng(0)
    data = rng.integers(0, 255, (3, 64, 96), dtype=np.uint8)
    path = tmp_path / "strip.tif"
    _write_strip(path, data, step=0.25)
    with rasterio.open(path) as ds:
        assert tiles.StripGrid.applies(ds)
        grid = tiles.StripGrid(ds, 1.0)
        img, x0, y0 = grid.window(X0 + 12.0, Y0 + 8.0, 8)
        assert img.dtype == np.uint8 and img.shape == (3, 8, 8)
        assert (x0, y0) == (X0 + 8.0, Y0 + 12.0)
        ref = ds.read([1, 2, 3], window=rasterio.windows.Window(32, 16, 32, 32), out_shape=(3, 8, 8),
                      resampling=rasterio.enums.Resampling.average)
        assert np.abs(img.astype(int) - ref.astype(int)).max() <= 1


def test_strip_grid_memory_is_one_window_and_edges_are_zero(tmp_path):
    import rasterio

    data = np.full((3, 48, 64), 200, np.uint8)
    path = tmp_path / "strip.tif"
    _write_strip(path, data)
    with rasterio.open(path) as ds:
        grid = tiles.StripGrid(ds, 1.0)
        corner, _, _ = grid.window(X0, Y0 + 48.0, 8)
        assert int(corner[:, 0, 0].max()) == 0 and int(corner[:, -1, -1].min()) == 200
        for cy in (Y0 + 40.0, Y0 + 30.0, Y0 + 20.0, Y0 + 10.0):
            grid.window(X0 + 32.0, cy, 8)
            assert len(grid.rows) <= 8


def test_tiled_file_is_not_a_strip(tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    path = tmp_path / "tiled.tif"
    with rasterio.open(path, "w", driver="GTiff", height=64, width=64, count=3, dtype="uint8", crs=UTM,
                       transform=from_origin(X0, Y0 + 64, 1, 1), tiled=True, blockxsize=16, blockysize=16) as dst:
        dst.write(np.zeros((3, 64, 64), np.uint8))
    with rasterio.open(path) as ds:
        assert not tiles.StripGrid.applies(ds)


def test_geographic_window_hanging_off_the_edge_is_padded(tmp_path):
    """Градусный снимок идёт через WarpedVRT. Окно с угла не должно падать."""
    import rasterio
    from rasterio.transform import from_bounds

    from expds.raster import open_metric

    path = tmp_path / "geo.tif"
    data = np.full((3, 24, 24), 180, np.uint8)
    transform = from_bounds(37.0, 55.0, 37.02, 55.02, 24, 24)
    with rasterio.open(path, "w", driver="GTiff", height=24, width=24, count=3, dtype="uint8",
                       crs="EPSG:4326", transform=transform) as dst:
        dst.write(data)
    with open_metric(path) as ds:
        cx, cy = ds.bounds.left, ds.bounds.top
        step = abs(float(ds.res[0]))
    img = tiles.read_window([path], cx, cy, step, 32)
    assert img.shape == (3, 32, 32) and img.dtype == np.uint8
    assert int(img.max()) > 0
    assert int(img[:, 0, 0].max()) == 0
