"""Второй этап по картинке (scripts/crop_verify.py): кроп с насыпью отличается от пустого места."""
import importlib.util
import json

import numpy as np
import pytest
from conftest import ROOT, UTM, X0, Y0
from shapely.geometry import Point

from expds.crs import WEB_MERCATOR, to_crs

FAMILIES = ["Альфа", "Бета", "Гамма", "Дельта"]


def load():
    spec = importlib.util.spec_from_file_location("crop_verify", ROOT / "scripts" / "crop_verify.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_spor(path, ox, mounds, seed):
    import rasterio
    from rasterio.transform import from_origin

    rng = np.random.default_rng(seed)
    n = 128
    a = rng.normal(70, 12, (3, n, n))
    yy, xx = np.mgrid[0:n, 0:n]
    for mx, my in mounds:
        a[:, ((xx + 0.5 - mx) ** 2 + (n - yy - 0.5 - my) ** 2) < 36] = 210
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=n, width=n, count=3, dtype="uint8",
                       transform=from_origin(X0 + ox, Y0 + n, 1.0, 1.0), crs=UTM) as ds:
        ds.write(np.clip(a, 0, 255).astype(np.uint8))


def feat(x, y, r, conf=None, region="", group=None):
    g = to_crs(Point(X0 + x, Y0 + y).buffer(r), UTM, WEB_MERCATOR)
    props = {"class_name": "kurgany_povrezhdennye", "region_name": region, "markup_type": "sp_or"}
    if conf is not None:
        props["confidence"] = conf
    if group:
        props["threshold_group"] = group
    return {"type": "Feature", "geometry": g.__geo_interface__, "properties": props}


def scene(tmp_path):
    rng = np.random.default_rng(0)
    root = tmp_path / "train"
    stack, gt = [], []
    for k, fam in enumerate(FAMILIES):
        region = f"{101 + k:03d}_{fam}_1"
        ox = k * 1000.0
        mounds = [(20 + 18 * j, 30 + (j % 2) * 50) for j in range(6)]
        write_spor(root / region / f"05_{region}_SpOr_снимок" / f"01_{region}_SpOr.tif", ox, mounds, k)
        (root / region / "UTM.json").write_text(json.dumps({"crs": UTM}), encoding="utf-8")
        for mx, my in mounds:
            gt.append(feat(ox + mx, my, 6.0, region=region))
            stack.append(feat(ox + mx, my, 4.0, float(rng.uniform(0.2, 0.8)), region, "sp_focal"))
        for _ in range(10):
            x, y = float(rng.uniform(15, 113)), float(rng.uniform(100, 118))
            stack.append(feat(ox + x, y, 4.0, float(rng.uniform(0.2, 0.8)), region, "sp_focal"))
        stack.append(feat(ox + 60, 60, 4.0, 0.01, region, "sp_focal"))       # ниже --min-conf
    return root, {"type": "FeatureCollection", "features": stack}, {"type": "FeatureCollection", "features": gt}


def test_crop_classifier_beats_stack_confidence(tmp_path):
    pytest.importorskip("segmentation_models_pytorch")
    cv = load()
    root, stack, gt = scene(tmp_path)
    out, rep = cv.run(str(root), stack, gt, "sp_or", {"kurgany_povrezhdennye"}, {"sp_focal"}, 0.05,
                      res=1.0, tile=24, li_tags=(), folds=2, mode="geo",
                      metric_path=str(ROOT / "materials" / "compute_metrics.py"),
                      epochs=10, batch=16, lr=1e-3, encoder="resnet18", pretrained=False, device="cpu")
    assert rep["candidates"] == 64 and rep["with_crop"] == 64 and rep["verified"] == 64 and rep["hits"] == 24
    assert rep["auc_crop"] > rep["auc_stack"] + 0.25
    props = [f["properties"] for f in out["features"]]
    assert all("confidence_base" in p for p in props)
    low = [p for p in props if p["confidence_base"] == 0.01]
    assert len(low) == 4 and all("verify_p" not in p and p["confidence"] == 0.01 for p in low)
    assert all(0.0 <= p["confidence"] <= 1.0 for p in props)
    assert rep["demote_with_gt"] is False


def test_default_output_does_not_read_the_ground_truth(tmp_path):
    """Без --demote-with-gt второй круг на эталоне не обнуляется: эталона на площадке нет."""
    pytest.importorskip("segmentation_models_pytorch")
    cv = load()
    root, stack, gt = scene(tmp_path)
    twin = json.loads(json.dumps(stack["features"][0]))
    twin["properties"]["confidence"] = 0.1
    stack["features"].append(twin)
    kw = dict(res=1.0, tile=24, li_tags=(), folds=2, mode="p",
              metric_path=str(ROOT / "materials" / "compute_metrics.py"),
              epochs=2, batch=16, lr=1e-3, encoder="resnet18", pretrained=False, device="cpu")
    out, _ = cv.run(str(root), stack, gt, "sp_or", {"kurgany_povrezhdennye"}, {"sp_focal"}, 0.05, **kw)
    assert float(out["features"][-1]["properties"]["verify_p"]) > 0.0
    leaked, rep = cv.run(str(root), stack, gt, "sp_or", {"kurgany_povrezhdennye"}, {"sp_focal"}, 0.05,
                         demote_with_gt=True, **kw)
    assert float(leaked["features"][-1]["properties"]["verify_p"]) == 0.0 and rep["demote_with_gt"]


def test_saved_weights_load_in_the_inference_hook(tmp_path):
    pytest.importorskip("segmentation_models_pytorch")
    from expds.crop_model import _network

    import torch

    cv = load()
    X = np.zeros((8, 3, 16, 16), np.uint8)
    X[:4] = 200
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    info = cv.save_model(str(tmp_path / "crop.pt"), X, y, "sp_or", 0.5, 16, epochs=1, batch=4,
                         pretrained=False, device="cpu")
    assert info["crops"] == 8 and info["positives"] == 4 and info["pretrained"] is False
    blob = torch.load(tmp_path / "crop.pt", weights_only=False)
    net = _network(blob["encoder"], blob["pretrained"])
    net.load_state_dict(blob["state_dict"])


def test_blend_modes():
    cv = load()
    c, p = np.array([0.25, 0.81]), np.array([1.0, 0.25])
    assert np.allclose(cv.blend(c, p, "geo"), [0.5, 0.45])
    assert np.allclose(cv.blend(c, p, "p"), p) and np.allclose(cv.blend(c, p, "none"), c)


def test_missing_crops_do_not_invent_a_median():
    cv = load()
    p = np.array([np.nan, np.nan, 0.2])
    keys = [("sp_focal", "a"), ("sp_focal", "a"), ("sp_focal", "b")]
    assert cv.group_median(p, keys) == {("sp_focal", "b"): 0.2}


def test_a_stronger_circle_outside_the_group_keeps_the_mound():
    cv = load()
    strong = feat(10, 10, 4.0, 0.9, "101_Альфа_1", "li")
    strong["properties"]["markup_type"] = "li"
    weak = feat(10, 10, 4.0, 0.4, "101_Альфа_1", "sp_focal")
    stack = {"type": "FeatureCollection", "features": [strong, weak]}
    gt = {"type": "FeatureCollection", "features": [feat(10, 10, 6.0, region="101_Альфа_1")]}
    y_any, y_first = cv.labels_of(cv.load_rerank(), stack, gt,
                                  str(ROOT / "materials" / "compute_metrics.py"), [1])
    assert int(y_any[0]) == 1 and int(y_first[0]) == 0


def test_the_fuller_image_wins(tmp_path):
    """Кромка первого снимка не забирает кроп, если второй снимок покрывает окно целиком."""
    import rasterio
    from rasterio.transform import from_origin

    cv = load()
    region = "101_Альфа_1"
    folder = tmp_path / region / "05_SpOr_снимок"
    folder.mkdir(parents=True)
    (tmp_path / region / "UTM.json").write_text(json.dumps({"crs": UTM}), encoding="utf-8")

    def write(path, left, top, size, value):
        with rasterio.open(path, "w", driver="GTiff", height=size, width=size, count=3, dtype="uint8",
                           transform=from_origin(left, top, 1.0, 1.0), crs=UTM) as ds:
            ds.write(np.full((3, size, size), value, np.uint8))

    write(folder / "01_edge_SpOr.tif", X0, Y0 + 8, 8, 1)
    write(folder / "02_full_SpOr.tif", X0 - 20, Y0 + 40, 64, 200)
    stack = {"type": "FeatureCollection", "features": [feat(4, 4, 2.0, 0.5, region, "sp_focal")]}
    X, ok = cv.extract_crops(str(tmp_path), stack, [0], "sp_or", 1.0, 16, ())
    assert ok[0] and float(X[0].mean()) > 100


def test_a_second_circle_on_the_same_mound_is_not_boosted():
    cv = load()
    p = np.array([0.9, 0.9, 0.2])
    y_any = np.array([1, 1, 0])
    y_first = np.array([1, 0, 0])
    assert np.allclose(cv.demote_duplicates(p, y_any, y_first), [0.9, 0.0, 0.2])


def test_lidar_crops_come_from_the_las_relief(tmp_path):
    from conftest import MOUNDS, write_las

    from expds.lidar import RELIEF_TAGS

    cv = load()
    region = "101_Альфа_1"
    write_las(tmp_path / region / f"01_{region}_La_точки" / f"01_{region}_La_точки.las")
    (tmp_path / region / "UTM.json").write_text(json.dumps({"crs": UTM}), encoding="utf-8")
    feats = [feat(x, y, 1.5, 0.5, region, "li") for x, y, _, _ in MOUNDS] + [feat(500.0, 500.0, 1.5, 0.5, region, "li")]
    for f in feats:
        f["properties"]["markup_type"] = "li"
    stack = {"type": "FeatureCollection", "features": feats}
    X, ok = cv.extract_crops(str(tmp_path), stack, list(range(len(feats))), "li", 0.5, 32, RELIEF_TAGS)
    assert ok[: len(MOUNDS)].all() and not ok[-1]           # последний кандидат вне облака
    assert X.shape == (len(feats), 3, 32, 32) and X[0].std() > 0
