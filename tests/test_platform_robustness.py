"""Устойчивость к платформе и данным выборки (23.10.2026): CRS, раскладки входа, запись, сигналы."""
import json
import os
import signal
import subprocess
import sys

import numpy as np
import pytest
import rasterio
from conftest import ROOT, UTM, X0, Y0, write_las, write_mag_tif
from rasterio.control import GroundControlPoint
from rasterio.transform import from_origin
from shapely.geometry import box

from expds import writer
from expds.catalog import read_utm_json, scan_root
from expds.crs import resolve_region_crs, to_crs, valid_crs


def test_utm_json_without_code_falls_back_instead_of_breaking(tmp_path):
    cases = {"EPSG": None, "EPSG:32636": "EPSG:32636", "urn:ogc:def:crs:EPSG::32637": "EPSG:32637",
             "nonsense": None}
    for i, (raw, want) in enumerate(cases.items()):
        p = tmp_path / f"u{i}.json"
        p.write_text(json.dumps({"crs": raw}), encoding="utf-8")
        assert read_utm_json(p) == want, raw
    p = tmp_path / "zone.json"
    p.write_text(json.dumps({"utm": "36N"}), encoding="utf-8")
    assert read_utm_json(p) == "EPSG:32636"
    assert valid_crs("EPSG:") is None


def _utm_raster(path, crs=UTM, x0=X0, y0=Y0, size=120, res=1.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=size, width=size, count=1, dtype="uint8",
                       transform=from_origin(x0, y0 + size * res, res, res), crs=crs) as ds:
        ds.write(np.full((size, size), 100, np.uint8), 1)
    return path


def test_region_crs_conflict_is_resolved_by_rasters(tmp_path):
    las = write_las(tmp_path / "a.las", n=5_000)                    # WKT в LAS: EPSG:32636
    assert resolve_region_crs(None, [las], []) == UTM               # нет UTM.json → LAS
    assert resolve_region_crs(UTM, [las], []) == UTM                # совпадают
    assert resolve_region_crs("EPSG:", [las], []) == UTM            # битый UTM.json → LAS
    wrong = "EPSG:32637"
    assert resolve_region_crs(wrong, [las], [], "vote") == wrong    # голосовать нечем → прежнее поведение
    assert resolve_region_crs(wrong, [las], [], "las") == UTM
    ortho = _utm_raster(tmp_path / "sp.tif")                        # снимок участка в 36N накрывает облако
    assert resolve_region_crs(wrong, [las], [ortho], "vote") == UTM
    assert resolve_region_crs(wrong, [las], [ortho], "utm_json") == wrong


def test_dem_keeps_las_coordinates_and_only_labels_crs(tmp_path):
    """ЦМР не переводится в зону подписи: левый край сетки — из заголовка файла."""
    from expds.lidar import build_dem

    las = write_las(tmp_path / "a.las", n=2_000)
    dem = build_dem(las, res=5.0, crs="EPSG:32633")
    assert dem.crs == "EPSG:32633"
    assert dem.left == pytest.approx(X0, abs=5.0)


def test_input_layouts_flat_nested_and_single_region(tmp_path):
    flat = tmp_path / "flat"
    write_las(flat / "x_La_точки.las", n=2_000)
    (flat / "UTM.json").write_text(json.dumps({"utm": "36N"}), encoding="utf-8")
    regs = scan_root(flat)
    assert len(regs) == 1 and regs[0].las and regs[0].utm_json is not None

    nested = tmp_path / "nested"
    for name in ("region_0001", "region_0002"):
        write_mag_tif(nested / "public_val" / name / f"07_{name}_MagGr_карты" / f"01_{name}_MagGr.tif")
    regs = scan_root(nested)
    assert sorted(r.name for r in regs) == ["region_0001", "region_0002"]

    single = tmp_path / "region_0003"
    write_las(single / "01_region_0003_La_точки" / "01_region_0003_La_точки.las", n=2_000)
    write_mag_tif(single / "07_region_0003_MagGr_карты" / "01_region_0003_MagGr.tif")
    (single / "UTM.json").write_text(json.dumps({"utm": "36N"}), encoding="utf-8")
    regs = scan_root(single)
    assert len(regs) == 1 and regs[0].name == "region_0003" and regs[0].las and regs[0].rasters.get("maggr")

    usual = tmp_path / "usual"
    write_las(usual / "region_0001" / "01_region_0001_La_точки" / "a.las", n=2_000)
    write_mag_tif(usual / "region_0002" / "07_region_0002_MagGr_карты" / "b.tif")
    assert sorted(r.name for r in scan_root(usual)) == ["region_0001", "region_0002"]


def test_circle_in_degrees_is_metres_on_the_ground():
    from expds.raster import circle_3857

    lon, lat = 33.0, 55.0
    c = circle_3857(lon, lat, 4.0, "EPSG:4326")
    utm = to_crs(c, "EPSG:3857", UTM)
    assert abs(utm.area - np.pi * 16) / (np.pi * 16) < 0.05
    m = circle_3857(X0 + 10, Y0 + 10, 4.0, UTM)                     # метрическая CRS — как раньше
    assert abs(to_crs(m, "EPSG:3857", UTM).area - np.pi * 16) / (np.pi * 16) < 0.05


def test_gcp_and_degree_rasters_are_read_in_metres(tmp_path):
    from expds.raster import needs_warp, open_metric

    gcp = tmp_path / "ae_gcp.tif"
    n = 200
    corners = [(0, 0, X0, Y0 + 100), (0, n, X0 + 100, Y0 + 100), (n, 0, X0, Y0), (n, n, X0 + 100, Y0)]
    gcps = [GroundControlPoint(row=r, col=c, x=x, y=y) for r, c, x, y in corners]
    with rasterio.open(gcp, "w", driver="GTiff", height=n, width=n, count=1, dtype="uint8") as ds:
        ds.write(np.full((n, n), 90, np.uint8), 1)
        ds.gcps = (gcps, rasterio.crs.CRS.from_string(UTM))
    with rasterio.open(gcp) as ds:
        assert needs_warp(ds) == "gcps"
    with open_metric(gcp) as ds:
        b = ds.bounds
        assert ds.crs.to_epsg() == 32636
        assert abs(b.left - X0) < 2 and abs(b.top - (Y0 + 100)) < 2 and 0.3 < abs(ds.res[0]) < 0.7

    deg = tmp_path / "sp_deg.tif"
    with rasterio.open(deg, "w", driver="GTiff", height=100, width=100, count=1, dtype="uint8",
                       transform=from_origin(33.0, 55.001, 1e-5, 1e-5), crs="EPSG:4326") as ds:
        ds.write(np.full((100, 100), 80, np.uint8), 1)
    with open_metric(deg) as ds:
        assert ds.crs.to_epsg() == 32636 and 0.3 < abs(ds.res[0]) < 1.5
    utm_file = _utm_raster(tmp_path / "plain.tif")
    with open_metric(utm_file) as ds:                               # метрический растр — без перепроецирования
        assert not hasattr(ds, "src_dataset") and ds.crs.to_epsg() == 32636


def test_fill_holes_and_cores_stay_inside():
    from shapely.geometry import Polygon

    from expds import post

    ring = box(0, 0, 100, 100).difference(box(20, 20, 80, 80))
    feats = [{"geom": ring, "class_name": "gorodishcha", "confidence": 0.9, "markup_type": "sp_or"},
             {"geom": ring, "class_name": "fortifikatsii", "confidence": 0.9, "markup_type": "ae"}]
    out = post.fill_holes(feats, ["gorodishcha"])
    assert not out[0]["geom"].interiors and out[1]["geom"].interiors
    ell = Polygon([(0, 0), (200, 0), (200, 40), (60, 40), (60, 160), (0, 160)])      # Г-образная маска
    core = post.core_of(ell, 0.3)
    assert core.within(ell.buffer(1e-6)) and core.area <= 0.3 * ell.area * 1.05
    assert ell.contains(core.representative_point())
    spec = {"gorodishcha": {"frac": 0.3, "groups": ["sp_or"]}}
    shaped = post.shape_cores([dict(feats[0], geom=ell), dict(feats[0], geom=ell, markup_type="li")], spec)
    assert shaped[0]["geom"].area < ell.area and shaped[1]["geom"] is ell


def test_copy_layers_follow_coverage_confidence_and_groups(tmp_path):
    from shapely.geometry import Point

    from expds import fusion

    a = np.zeros((200, 200), np.uint8)
    a[:, :100] = 120                                                 # снимок покрывает западную половину
    sp = tmp_path / "sp.tif"
    with rasterio.open(sp, "w", driver="GTiff", height=200, width=200, count=1, dtype="uint8",
                       transform=from_origin(X0, Y0 + 100, 0.5, 0.5), crs=UTM) as ds:
        ds.write(a, 1)

    def obj(x, y, cls, conf, group=None):
        f = {"geom": to_crs(Point(X0 + x, Y0 + y).buffer(2.0), UTM), "class_name": cls, "confidence": conf,
             "markup_type": "li", "region_name": "R", "source": "a.las"}
        return dict(f, threshold_group=group) if group else f

    feats = [obj(20, 50, fusion.POVR, 0.9), obj(25, 30, fusion.POVR, 0.2), obj(80, 50, fusion.POVR, 0.99),
             obj(15, 40, fusion.POVR, 0.95, "li_cpu"), obj(30, 60, "gorodishcha", 0.95, "li_mask")]
    rules = [{"src": "li", "dst": "sp_or", "group": "li_sp", "min": {fusion.POVR: 0.5}},
             {"src": "li", "dst": "sp_or", "group": "li_sp_gor", "src_groups": ["li_mask"], "min": {"gorodishcha": 0.9}}]
    out = fusion.copy_layers(feats, {"sp_or": [sp]}, rules)
    got = sorted((f["class_name"], f["threshold_group"], round(f["confidence"], 2)) for f in out)
    assert got == [("gorodishcha", "li_sp_gor", 0.95), (fusion.POVR, "li_sp", 0.9)]
    assert all(f["markup_type"] == "sp_or" for f in out)
    assert fusion.copy_layers(feats, {}, rules) == []


def test_checkpoint_writes_only_the_answer(tmp_path):
    from expds.watchdog import RunState, checkpoint

    st = RunState(tmp_path / "result.geojson", {"thresholds": {}, "default_threshold": 0.0})
    st.add([{"geom": box(X0, Y0, X0 + 10, Y0 + 10), "class_name": "kurgany_tselye", "confidence": 0.5,
             "markup_type": "li", "region_name": "r"}])
    checkpoint(st, "конец")
    assert [p.name for p in tmp_path.iterdir()] == ["result.geojson"]
    assert writer.validate(tmp_path / "result.geojson")["ok"]


def test_run_writes_result_only_at_the_end(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from expds import pipeline

    out = tmp_path / "o" / "result.geojson"
    seen = {}
    reg = SimpleNamespace(name="r", las=[], rasters={}, utm_json=None, boundaries=[], noise=[])
    monkeypatch.setattr(pipeline, "scan_root", lambda _p: [reg])
    monkeypatch.setattr(pipeline, "geophys_feats", lambda *_a, **_k: [
        {"geom": box(X0, Y0, X0 + 3, Y0 + 3), "class_name": "object_poly", "confidence": 0.9,
         "markup_type": "maggr", "region_name": "r"}])
    monkeypatch.setattr(pipeline, "lidar_feats", lambda *_a, **_k: seen.setdefault("li", out.exists()) and [] or [])
    monkeypatch.setattr(pipeline, "optical_feats", lambda *_a, **_k: [])
    monkeypatch.setenv("EXPDS_TMP", str(tmp_path / "tmp"))
    cfg = tmp_path / "t.toml"
    cfg.write_text("[time]\nwatchdog = false\nearly_checkpoint = \"side\"\n", encoding="utf-8")
    pipeline.run(tmp_path, out, cfg)
    assert seen["li"] is False                                       # к лидару result.geojson ещё нет
    assert writer.validate(out)["ok"] and not (tmp_path / "tmp" / "result.geojson.partial").exists()


def test_main_without_arguments_uses_env_and_cleans_tmp(val_dir, tmp_path):
    out = tmp_path / "res"
    env = dict(os.environ, EXPDS_INPUT=str(val_dir), EXPDS_OUTPUT=str(out))
    r = subprocess.run([sys.executable, str(ROOT / "inference" / "main.py")], capture_output=True, text=True,
                       timeout=600, env=env, check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    assert writer.validate(out / "result.geojson")["ok"]
    assert not (out / ".expds_tmp").exists()
    assert "среда:" in r.stdout and "опись входа" in r.stdout


def test_decoder_from_checkpoint_and_ensemble_by_class(tmp_path):
    torch = pytest.importorskip("torch")
    smp = pytest.importorskip("segmentation_models_pytorch")
    from expds.infer_nn import Ensemble, Segmenter
    from expds.pipeline import _segmenters

    classes = ["kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha"]
    paths = []
    for s in range(2):
        torch.manual_seed(s)
        net = smp.create_model("FPN", encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=3)
        p = tmp_path / f"fpn{s}.pt"
        torch.save({"state_dict": net.state_dict(), "encoder": "resnet18", "arch": "FPN", "in_channels": 3,
                    "classes": classes, "res": 0.5, "tile": 64}, p)
        paths.append(p)
    img = np.random.default_rng(1).integers(0, 255, (3, 70, 90), dtype=np.uint8)
    a, b = (Segmenter(p, device="cpu").predict(img) for p in paths)
    ens = Ensemble([Segmenter(p, device="cpu") for p in paths], only=classes[:2])
    got = ens.predict(img)
    assert np.allclose(got[:2], (a[:2] + b[:2]) / 2, atol=1e-6) and np.allclose(got[2], a[2], atol=1e-6)
    segs = _segmenters({"li": {"path": [str(p) for p in paths], "ensemble_classes": classes[:2]}}, "li", tmp_path)
    assert isinstance(segs[0], Ensemble) and segs[0].only == [0, 1]
    with pytest.raises(ValueError):
        Ensemble([Segmenter(p, device="cpu") for p in paths], only=["arkhitektury"])


def test_train_with_decoder_and_ema(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")
    import torch

    from expds.infer_nn import Segmenter
    from expds.train import train

    cache = tmp_path / "cache"
    cache.mkdir()
    rng = np.random.default_rng(0)
    n, t = 16, 64
    imgs = rng.integers(0, 255, (n, 1, t, t), dtype=np.uint8)
    masks = np.zeros((n, t, t), np.uint8)
    masks[:, 20:30, 20:30] = 1 << 1
    imgs[:, :, 20:30, 20:30] = 255
    np.save(cache / "images_0000.npy", imgs)
    np.save(cache / "masks_0000.npy", masks)
    (cache / "index.csv").write_text("shard,idx,region,family,file,cx,cy,positive\n" + "".join(
        f"0,{i},r{i % 2},{'A' if i % 2 else 'B'},f.tif,0,0,1\n" for i in range(n)), encoding="utf-8")
    meta = {"markup": "li", "classes": ["kurgany_povrezhdennye", "kurgany_tselye"], "res": 0.5, "tile": t,
            "li_tags": ["g"], "flips": False, "channels": 1}
    (cache / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    train(str(cache), str(tmp_path / "u.pt"), epochs=1, batch=4, val_families=["B"], workers=0, pretrained=False,
          arch="UnetPlusPlus", ema=0.9)
    ck = torch.load(tmp_path / "u.pt", map_location="cpu", weights_only=False)
    assert ck["arch"] == "UnetPlusPlus" and ck["ema"] == 0.9
    probs = Segmenter(tmp_path / "u.pt", device="cpu").predict(rng.integers(0, 255, (1, 80, 80), dtype=np.uint8))
    assert probs.shape == (2, 80, 80) and np.isfinite(probs).all()


def test_offline_scripts_on_a_synthetic_train(tmp_path):
    """oof_transform (copy, core), audit_crs и audit_rasters работают из чистого клона."""
    from shapely.geometry import Point, mapping

    from expds.crs import WEB_MERCATOR

    train = tmp_path / "train"
    reg = train / "001_Тест"
    write_las(reg / "01_Тест_La_точки" / "01_Тест_La_точки.las", n=3_000)
    (reg / "UTM.json").write_text(json.dumps({"utm": "37N"}), encoding="utf-8")   # неверная зона
    _utm_raster(reg / "04_Тест_SpOr_спутник" / "04_Тест_SpOr.tif")
    pts = [(20, 30), (60, 70)]
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": mapping(to_crs(Point(X0 + x, Y0 + y).buffer(3), UTM, WEB_MERCATOR)),
         "properties": {"class_name": "kurgany_povrezhdennye", "confidence": 0.7, "markup_type": "li",
                        "region_name": "001_Тест"}} for x, y in pts]}
    oof = tmp_path / "li_oof.geojson"
    oof.write_text(json.dumps(fc), encoding="utf-8")
    gt = tmp_path / "gt.geojson"
    gt.write_text(json.dumps(fc), encoding="utf-8")
    py = [sys.executable]
    run = lambda *a: subprocess.run(py + [str(ROOT / "scripts" / a[0]), *a[1:]], capture_output=True,
                                    text=True, timeout=300, check=False)
    r = run("oof_transform.py", "copy", "--train", str(train), "--src", str(oof), "--dst", "sp_or",
            "--classes", "kurgany_povrezhdennye", "--group", "li_sp", "--out", str(tmp_path / "copy.geojson"))
    assert r.returncode == 0, r.stdout + r.stderr
    copies = json.loads((tmp_path / "copy.geojson").read_text(encoding="utf-8"))["features"]
    assert len(copies) == 2 and {f["properties"]["threshold_group"] for f in copies} == {"li_sp"}
    assert {f["properties"]["markup_type"] for f in copies} == {"sp_or"}
    r = run("oof_transform.py", "core", "--src", str(oof), "--classes", "kurgany_povrezhdennye", "--frac", "0.2",
            "--out", str(tmp_path / "core.geojson"))
    assert r.returncode == 0, r.stdout + r.stderr
    r = run("audit_crs.py", "--train", str(train), "--gt", str(gt), "--out", str(tmp_path / "crs.json"))
    assert r.returncode == 0, r.stdout + r.stderr
    rep = json.loads((tmp_path / "crs.json").read_text(encoding="utf-8"))
    assert rep["summary"]["conflicts"] == 1 and rep["summary"]["conflicts_where_las_is_right"] == 1
    assert rep["regions"][0]["pick_vote"] == UTM
    r = run("audit_rasters.py", "--train", str(train), "--out", str(tmp_path / "ras.json"))
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads((tmp_path / "ras.json").read_text(encoding="utf-8"))["summary"]["sp_or"]["kinds"] == {"utm": 1}


def test_scale_and_photometric_augmentation_keep_shapes_and_labels():
    torch = pytest.importorskip("torch")
    from expds.train import augment_batch

    torch.manual_seed(0)
    x = torch.randn(4, 3, 64, 64)
    y = torch.zeros(4, 2, 64, 64)
    y[:, 1, 24:40, 24:40] = 1
    for _ in range(10):
        xa, ya = augment_batch(x, y, scale=0.33, photo=0.2)
        assert xa.shape == x.shape and ya.shape == y.shape
        assert set(torch.unique(ya).tolist()) <= {0.0, 1.0}
        assert 0.4 < ya[:, 1].sum() / y[:, 1].sum() < 2.5        # площадь объекта меняется вместе с масштабом
    xa, ya = augment_batch(x, y)
    assert torch.equal(xa, x) and torch.equal(ya, y)


def test_mosaic_uses_all_points_when_there_is_no_ground_class(tmp_path):
    from expds.lidar import build_dem_mosaic, has_class

    las = write_las(tmp_path / "noground.las", n=20_000, ground_class=1)   # «земля» сохранена классом 1
    assert not has_class(las, 2)
    d = build_dem_mosaic(las, res=0.1, work_res=0.5, tile_m=40.0, halo_m=5.0)
    assert np.isfinite(d.dem).mean() > 0.9
