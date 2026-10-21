"""Сквозной запуск inference/main.py на синтетической «валидации»."""
import json
import subprocess
import sys

import numpy as np
import pytest
from shapely.geometry import Point, mapping

from conftest import MOUNDS, ROOT, UTM, X0, Y0
from expds import writer
from expds.crs import WEB_MERCATOR, to_crs


def test_main_writes_valid_geojson(val_dir, tmp_path, official_metric):
    out = tmp_path / "out" / "result.geojson"
    r = subprocess.run([sys.executable, str(ROOT / "inference" / "main.py"), str(val_dir), str(out)],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    rep = writer.validate(out)
    assert rep["ok"], rep
    by = rep["summary"]["by_class"]
    assert by.get("kurgany_tselye", 0) >= 1 and by.get("finds_points", 0) >= 1
    if official_metric:
        from expds.metric import score

        gt = {"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {"class_name": "kurgany_tselye", "region_name": "region_0001"},
             "geometry": mapping(to_crs(Point(X0 + mx, Y0 + my).buffer(r / 2), UTM, WEB_MERCATOR))}
            for mx, my, _, r in MOUNDS]}
        res = score(json.loads(out.read_text(encoding="utf-8")), gt, official_metric)
        assert res["per_class"]["kurgany_tselye"]["tp"] == len(MOUNDS)


def test_main_accepts_output_dir_like_organizer_examples(tmp_path):
    bad = tmp_path / "in" / "region_0009"
    bad.mkdir(parents=True)
    (bad / "broken.las").write_bytes(b"not a las file")
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    for target in (out_dir, tmp_path / "output2"):
        r = subprocess.run([sys.executable, str(ROOT / "inference" / "main.py"), str(bad.parent), str(target)],
                           capture_output=True, text=True, timeout=300)
        assert r.returncode == 0 and writer.validate(target / "result.geojson")["ok"]


def test_exhausted_budget_still_visits_geophys_before_lidar(tmp_path, monkeypatch):
    """Нулевой бюджет не отменяет геофизику: она идёт раньше лидара и не ждёт запас в 120 с."""
    from types import SimpleNamespace

    from expds import pipeline

    calls = []
    reg = SimpleNamespace(name="r", las=[tmp_path / "a.las"], rasters={}, utm_json=None,
                          boundaries=[], noise=[])
    monkeypatch.setattr(pipeline, "scan_root", lambda _p: [reg])
    monkeypatch.setattr(pipeline, "geophys_feats", lambda *_a, **_k: calls.append("geo") or [])
    monkeypatch.setattr(pipeline, "lidar_feats", lambda *_a, **_k: calls.append("li") or [])
    monkeypatch.setattr(pipeline, "optical_feats", lambda *_a, **_k: calls.append("opt") or [])

    class Dummy:
        def __init__(self, *_a, **_k):
            pass

        def close(self):
            pass

    monkeypatch.setattr("expds.lidar.DemPrefetch", Dummy)
    cfg = tmp_path / "t.toml"
    cfg.write_text("[time]\nbudget_s = 0\nreserve_s = 120\n", encoding="utf-8")
    pipeline.run(tmp_path, tmp_path / "out.geojson", cfg)
    assert calls == ["geo"]


def test_missing_torch_still_keeps_geophysics(tmp_path, monkeypatch):
    """Сбой импорта torch не обнуляет файл: геофизика всё равно считается."""
    import builtins

    from types import SimpleNamespace

    from expds import pipeline

    calls = []
    real = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "torch" or str(name).startswith("torch."):
            raise ImportError("torch нет")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    reg = SimpleNamespace(name="r", las=[], rasters={}, utm_json=None, boundaries=[], noise=[])
    monkeypatch.setattr(pipeline, "scan_root", lambda _p: [reg])
    monkeypatch.setattr(pipeline, "geophys_feats", lambda *_a, **_k: calls.append("geo") or [])
    monkeypatch.setattr(pipeline, "lidar_feats", lambda *_a, **_k: calls.append("li") or [])
    monkeypatch.setattr(pipeline, "optical_feats", lambda *_a, **_k: calls.append("opt") or [])
    cfg = tmp_path / "t.toml"
    cfg.write_text("[time]\nbudget_s = 0\nwatchdog = false\n", encoding="utf-8")
    summary = pipeline.run(tmp_path, tmp_path / "out.geojson", cfg)
    assert calls == ["geo"]
    assert summary["written"] >= 0
    assert (tmp_path / "out.geojson").exists()


def test_coarse_lidar_has_its_own_threshold_group():
    from expds.pipeline import lidar_threshold_group, output_floors

    assert lidar_threshold_group({"factor": 1}) is None
    assert lidar_threshold_group({"factor": 3}) == "li_coarse"
    assert lidar_threshold_group({}) is None
    assert lidar_threshold_group({"factor": "нет"}) is None
    cfg = {"thresholds": {"kurgany_povrezhdennye": 0.15},
           "thresholds_by_markup": {"li": {"kurgany_povrezhdennye": 0.05},
                                    "li_coarse": {"kurgany_povrezhdennye": 0.97}}}
    assert output_floors(cfg, "li")["kurgany_povrezhdennye"] == 0.05
    assert output_floors(cfg, "li_coarse")["kurgany_povrezhdennye"] == 0.97


def test_model_specs_and_class_filter():
    from types import SimpleNamespace

    from expds.pipeline import _kept, model_specs

    models = {"li": "models/li.pt",
              "sp_or": ["models/a.pt", {"path": "models/b.pt", "classes": ["gorodishcha"]}],
              "ae": {"path": "models/c.pt"}}
    assert model_specs(models, "li") == [("models/li.pt", None)]
    assert model_specs(models, "sp_or") == [("models/a.pt", None), ("models/b.pt", ["gorodishcha"])]
    assert model_specs(models, "ae") == [("models/c.pt", None)]
    assert model_specs(models, "or") == []
    feats = [{"class_name": "gorodishcha"}, {"class_name": "kurgany_tselye"}]
    assert _kept(SimpleNamespace(keep={"gorodishcha"}), feats) == feats[:1]
    assert _kept(SimpleNamespace(keep=None), feats) == feats


def test_ensemble_averages_members(tmp_path):
    torch = pytest.importorskip("torch")
    smp = pytest.importorskip("segmentation_models_pytorch")
    from expds.infer_nn import Ensemble, Segmenter
    from expds.pipeline import _segmenters, model_files, model_specs

    classes = ["kurgany_povrezhdennye", "kurgany_tselye"]
    paths = []
    for s in range(2):
        torch.manual_seed(s)
        net = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=2)
        p = tmp_path / f"m{s}.pt"
        torch.save({"state_dict": net.state_dict(), "encoder": "resnet18", "in_channels": 3,
                    "classes": classes, "res": 0.5, "tile": 64}, p)
        paths.append(p)
    img = np.random.default_rng(0).integers(0, 255, (3, 70, 90), dtype=np.uint8)
    a, b = (Segmenter(p, device="cpu").predict(img) for p in paths)
    ens = Ensemble([Segmenter(p, device="cpu") for p in paths])
    assert np.allclose(ens.predict(img), (a + b) / 2, atol=1e-6)

    models = {"li": {"path": [str(p) for p in paths]}}
    assert model_files(model_specs(models, "li")[0][0]) == [str(p) for p in paths]
    segs = _segmenters(models, "li", tmp_path)
    assert len(segs) == 1 and isinstance(segs[0], Ensemble)

    torch.save({"state_dict": smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3,
                                       classes=1).state_dict(),
                "encoder": "resnet18", "in_channels": 3, "classes": classes[:1], "res": 0.5, "tile": 64},
               tmp_path / "other.pt")
    with pytest.raises(ValueError):
        Ensemble([Segmenter(paths[0], device="cpu"), Segmenter(tmp_path / "other.pt", device="cpu")])


def test_main_survives_bad_input(tmp_path):
    bad = tmp_path / "in" / "region_0009"
    bad.mkdir(parents=True)
    (bad / "broken.las").write_bytes(b"not a las file")
    out = tmp_path / "o" / "result.geojson"
    r = subprocess.run([sys.executable, str(ROOT / "inference" / "main.py"), str(bad.parent), str(out)],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and writer.validate(out)["ok"]


def test_train_and_infer_smoke(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")
    from expds.infer_nn import Segmenter, probs_to_features
    from expds.train import train

    cache = tmp_path / "cache"
    cache.mkdir()
    rng = np.random.default_rng(0)
    n, t = 24, 64
    imgs = rng.integers(0, 255, (n, 1, t, t), dtype=np.uint8)
    masks = np.zeros((n, t, t), np.uint8)
    masks[:, 20:30, 20:30] = 1 << 1          # класс kurgany_tselye
    imgs[:, :, 20:30, 20:30] = 255
    np.save(cache / "images_0000.npy", imgs)
    np.save(cache / "masks_0000.npy", masks)
    rows = "shard,idx,region,family,file,cx,cy,positive\n" + "".join(
        f"0,{i},r{i % 3},{'A' if i % 3 else 'B'},f.tif,0,0,1\n" for i in range(n))
    (cache / "index.csv").write_text(rows, encoding="utf-8")
    meta = {"markup": "li", "classes": ["kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha",
                                        "arkhitektury", "fortifikatsii"],
            "res": 0.5, "tile": t, "li_tags": ["g"], "flips": False, "channels": 1}
    (cache / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    res = train(str(cache), str(tmp_path / "m.pt"), epochs=1, batch=8, val_families=["B"], workers=0,
                pretrained=False)
    assert (tmp_path / "m.pt").exists() and res["train_tiles"] == 16
    seg = Segmenter(tmp_path / "m.pt", device="cpu")
    probs = seg.predict(rng.integers(0, 255, (1, 100, 150), dtype=np.uint8))
    assert probs.shape == (5, 100, 150) and np.isfinite(probs).all()
    from rasterio.transform import from_origin
    feats = probs_to_features(probs, seg.classes, from_origin(X0, Y0 + 50, 0.5, 0.5), UTM, 0.5,
                              "r", "li", "x", low=0.0)
    assert all(f["geom"].geom_type == "Polygon" for f in feats)


def test_broken_lidar_model_uses_cpu_detector(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from expds import pipeline

    dem = SimpleNamespace(crs="EPSG:32636", info={"factor": 1})
    monkeypatch.setattr(pipeline, "_cached_segmenters", lambda *a, **k: [SimpleNamespace(meta={})])
    monkeypatch.setattr(pipeline, "_li_with_model", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("oom")))
    monkeypatch.setattr("expds.lidar.build_dem", lambda *a, **k: dem)
    seen = {}

    def cpu(d, region, source, cfg, root, cand_p):
        seen["source"] = source
        return [{"class_name": "kurgany_povrezhdennye", "threshold_group": "li_cpu",
                 "confidence": 0.2, "markup_type": "li", "region_name": region, "source": source}]

    monkeypatch.setattr(pipeline, "cpu_kurgans", cpu)
    reg = SimpleNamespace(name="r", las=[tmp_path / "a.las"], rasters={}, utm_json=None)
    out = pipeline.lidar_feats(reg, {}, pipeline.TimeGuard(1000), {}, tmp_path, prefetch=None)
    assert seen["source"] == "a.las"
    assert out[0]["threshold_group"] == "li_cpu"


def test_watchdog_writes_a_valid_file_and_stops(tmp_path):
    import time

    from shapely.geometry import box

    from expds.watchdog import RunState, arm

    state = RunState(tmp_path / "result.geojson",
                     {"thresholds": {}, "default_threshold": 0.0, "thresholds_by_markup": {}})
    state.add([{"geom": box(0, 0, 10, 10), "class_name": "kurgany_tselye", "confidence": 0.5,
                "markup_type": "li", "region_name": "r"}])
    flag = {}
    arm(state, 0.15, exit_fn=lambda code: flag.setdefault("code", code))
    time.sleep(0.6)
    assert flag["code"] == 0
    assert writer.validate(tmp_path / "result.geojson")["ok"]


def test_inference_stays_fp32_unless_asked():
    torch = pytest.importorskip("torch")
    from expds.infer_nn import configure_runtime, inference_amp

    configure_runtime(amp=False, tf32=False, deterministic=True)
    assert inference_amp(torch.device("cpu")) is False
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cuda.matmul.allow_tf32 is False


def test_two_las_tiles_in_one_folder_are_both_kept():
    from pathlib import Path

    from expds.catalog import choose_las_boxes

    folder = Path("region/La")
    west = (folder / "west.las", (0.0, 0.0, 100.0, 100.0))
    east = (folder / "east.las", (500.0, 0.0, 600.0, 100.0))
    raw = (folder / "site.las", (200.0, 0.0, 300.0, 100.0))
    ground = (folder / "site_ground.las", (200.0, 0.0, 300.0, 100.0))
    kept = choose_las_boxes([west, east, raw, ground])
    names = sorted(p.name for p in kept)
    assert names == ["east.las", "site_ground.las", "west.las"]
    unknown = choose_las_boxes([(folder / "a.las", None), (folder / "b.las", None)])
    assert len(unknown) == 1
