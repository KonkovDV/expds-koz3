"""Кэш сплошной сеткой → COCO (PNG, папки RF-DETR) → «предсказания» = эталонные рамки → круги стека.

Круг должен лечь внутрь своего объекта, и объект, видный в нескольких окнах, — один круг.
"""
import importlib.util
import json

import pytest

from conftest import ROOT, UTM, X0, Y0
from shapely.geometry import box, shape


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_grid_cache_roundtrip_gives_one_circle_inside_each_object(val_dir, tmp_path):
    pytest.importorskip("cv2")  # W4: окружение без OpenCV — skip
    from expds.crs import WEB_MERCATOR, to_crs
    from expds.tiles import build_cache

    objs = [box(X0 + 6, Y0 + 6, X0 + 12, Y0 + 12), box(X0 + 22, Y0 + 26, X0 + 30, Y0 + 32)]
    gt = tmp_path / "gt.geojson"
    gt.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": to_crs(o, UTM, WEB_MERCATOR).__geo_interface__,
         "properties": {"class_name": "object_poly", "markup_type": "maggr", "region_name": "region_0002"}}
        for o in objs]}), encoding="utf-8")
    cache = tmp_path / "grid"
    build_cache(str(val_dir), str(gt), "maggr", str(cache), res=0.25, tile=64, grid_overlap=0.5, seed=0)
    out = tmp_path / "coco"
    summary = load("coco_from_cache").export(cache, out, layout="rfdetr", images=True, all_valid=True)
    assert summary["parts"]["valid"]["images"] == 16 and "warning" not in summary
    coco = json.loads((out / "valid" / "_annotations.coco.json").read_text(encoding="utf-8"))
    assert all((out / "valid" / im["file_name"]).exists() for im in coco["images"])
    assert (out / "test" / "_annotations.coco.json").exists()
    assert all(im["crs"] == UTM for im in coco["images"])
    # объект виден в нескольких окнах сетки — «детектор» дал рамку в каждом
    assert len(coco["annotations"]) > len(objs)
    preds = [{"image_id": a["image_id"], "category_id": a["category_id"], "bbox": a["bbox"], "score": 0.9}
             for a in coco["annotations"]]
    stack = load("coco_preds_to_stack").convert(coco, preds, "maggr_det", radius=0.3, merge_m=3.0)
    feats = stack["features"]
    assert len(feats) == len(objs)
    for f in feats:
        c = to_crs(shape(f["geometry"]).centroid, WEB_MERCATOR, UTM)
        assert any(o.contains(c) for o in objs)
        p = f["properties"]
        assert p["threshold_group"] == "maggr_det" and p["markup_type"] == "maggr" and p["views"] >= 1


def test_merge_distance_uses_ground_meters_in_web_mercator():
    """6 м EPSG:3857 на 55° с. ш. — около 3,4 м на земле и должны склеиться при пороге 4 м."""
    y0 = 7_500_000.0
    conv = load("coco_preds_to_stack")
    coco = {"images": [
        {"id": 1, "region": "r", "cx": 0.0, "cy": y0, "res": 1.0, "tile": 8, "crs": "EPSG:3857"},
        {"id": 2, "region": "r", "cx": 6.0, "cy": y0, "res": 1.0, "tile": 8, "crs": "EPSG:3857"},
    ], "categories": [{"id": 1, "name": "kurgany_tselye"}]}
    preds = [{"image_id": i, "category_id": 1, "bbox": [3, 3, 2, 2], "score": 0.9}
             for i in (1, 2)]
    out = conv.convert(coco, preds, "sp", radius=4.0, merge_m=4.0)
    assert len(out["features"]) == 1
    assert out["features"][0]["properties"]["box_w_m"] < 1.5


def test_old_cache_without_crs_is_refused(tmp_path):
    conv = load("coco_preds_to_stack")
    coco = {"images": [{"id": 1, "region": "r", "cx": 0.0, "cy": 0.0, "res": 1.0, "tile": 8, "crs": None}],
            "categories": [{"id": 1, "name": "kurgany_tselye"}]}
    import pytest

    with pytest.raises(SystemExit, match="CRS"):
        conv.convert(coco, [{"image_id": 1, "category_id": 1, "bbox": [1, 1, 2, 2], "score": 0.5}], "g", 1.0, 2.0)
