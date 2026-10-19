"""Сборка эталона и кэша тайлов на синтетическом «трейне» с именами как в датасете."""
import json

import numpy as np
from shapely.geometry import Point, mapping

from conftest import MOUNDS, UTM, X0, Y0, write_las
from expds import gt as gtmod
from expds import tiles
from expds.catalog import scan_root
from expds.crs import WEB_MERCATOR, to_crs
from expds.lidar import build_dem, hillshade_set, write_set


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


def test_tile_cache_li(tmp_path):
    root = make_train(tmp_path / "train")
    items, _ = gtmod.build_gt(scan_root(root))
    gt_path = tmp_path / "gt.geojson"
    gt_path.write_text(json.dumps(gtmod.to_feature_collection(items)), encoding="utf-8")
    meta = tiles.build_cache(str(root), str(gt_path), "li", str(tmp_path / "cache"), res=0.5, tile=64,
                             li_tags=["g", "c"])
    assert meta["channels"] == 4 and meta["positive"] >= 2
    imgs = np.load(tmp_path / "cache" / "images_0000.npy")
    masks = np.load(tmp_path / "cache" / "masks_0000.npy")
    assert imgs.shape[1:] == (4, 64, 64) and imgs.dtype == np.uint8
    bit = meta["classes"].index("kurgany_tselye")
    assert ((masks >> bit) & 1).sum() > 0
