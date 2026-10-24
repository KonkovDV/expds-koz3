"""Общий кэш тайлов нескольких типов снимков (scripts/cache_merge.py)."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("cache_merge", ROOT / "scripts" / "cache_merge.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def fake_cache(path: Path, markup: str, shards: int, fill: int, classes=("arkhitektury",), res=0.3) -> Path:
    """Кэш expds.tiles в миниатюре: по два тайла 3×8×8 на шард."""
    path.mkdir(parents=True)
    rows = []
    for s in range(shards):
        np.save(path / f"images_{s:04d}.npy", np.full((2, 3, 8, 8), fill + s, np.uint8))
        np.save(path / f"masks_{s:04d}.npy", np.full((2, 8, 8), s % 2, np.uint8))
        for i in range(2):
            rows.append([s, i, f"00{s}_{markup}", f"Семья{s}", f"{markup}.tif", 0.0, 0.0, s % 2])
    with (path / "index.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["shard", "idx", "region", "family", "file", "cx", "cy", "positive"])
        w.writerows(rows)
    meta = {"markup": markup, "classes": list(classes), "res": res, "tile": 8, "li_tags": None,
            "li_source": None, "flips": True, "channels": 3, "pixel_units": "native",
            "min_area_m2": 0.0, "tiles": 2 * shards, "positive": 2 * (shards // 2)}
    (path / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return path


def test_merge_renumbers_shards_and_keeps_rows(tmp_path):
    m = _mod()
    a = fake_cache(tmp_path / "or", "or", shards=2, fill=10)
    b = fake_cache(tmp_path / "sp", "sp_or", shards=3, fill=50)
    meta = m.merge([a, b], tmp_path / "joint")
    out = tmp_path / "joint"
    assert sorted(p.name for p in out.glob("images_*.npy")) == [f"images_{i:04d}.npy" for i in range(5)]
    from expds.train import load_index

    rows = load_index(out)
    assert len(rows) == 10
    assert [r["markup"] for r in rows] == ["or"] * 4 + ["sp_or"] * 6
    # строка указывает на тот же тайл, что в исходном кэше
    for r in rows:
        img = np.load(out / f"images_{int(r['shard']):04d}.npy")[int(r["idx"])]
        expect = (10 if r["markup"] == "or" else 50) + int(r["region"][2])
        assert int(img[0, 0, 0]) == expect
    assert meta["markup"] == "or+sp_or" and meta["tiles"] == 10 and meta["classes"] == ["arkhitektury"]


def test_merge_refuses_different_recipes(tmp_path):
    m = _mod()
    a = fake_cache(tmp_path / "or", "or", shards=1, fill=1, res=0.1)
    b = fake_cache(tmp_path / "sp", "sp_or", shards=1, fill=2, res=0.5)
    with pytest.raises(SystemExit, match="res"):
        m.merge([a, b], tmp_path / "joint")


def test_train_runs_on_a_merged_cache(tmp_path):
    """Обучение принимает общий кэш: лишний столбец markup и составное имя типа снимков."""
    pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")
    from expds import train as tr
    from expds.infer_nn import Segmenter

    m = _mod()
    srcs = []
    for name, fill in (("or", 40), ("sp_or", 90)):
        c = tmp_path / name
        c.mkdir()
        rng = np.random.default_rng(fill)
        np.save(c / "images_0000.npy", rng.integers(0, 255, (4, 3, 32, 32), dtype=np.uint8))
        masks = np.zeros((4, 32, 32), np.uint8)
        masks[:, 10:14, 10:14] = 1
        np.save(c / "masks_0000.npy", masks)
        (c / "index.csv").write_text("shard,idx,region,family,file,cx,cy,positive\n" + "".join(
            f"0,{i},{name}{i},{'A' if i % 2 else 'B'},f.tif,0,0,1\n" for i in range(4)), encoding="utf-8")
        meta = {"markup": name, "classes": ["arkhitektury"], "res": 0.3, "tile": 32, "flips": True,
                "channels": 3, "pixel_units": "native", "li_source": None}
        (c / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        srcs.append(c)
    m.merge(srcs, tmp_path / "joint")
    tr.train(str(tmp_path / "joint"), str(tmp_path / "m.pt"), epochs=1, batch=4, val_families=["B"], workers=0,
             pretrained=False, gpu_data=False)
    seg = Segmenter(tmp_path / "m.pt")
    assert seg.meta["markup"] == "or+sp_or" and seg.meta["classes"] == ["arkhitektury"]


def test_nn_oof_weights_mode_predicts_another_markup(tmp_path, monkeypatch, capsys):
    """nn_oof --weights: готовая сеть спутника на ортофото без обучения, файл для табло."""
    pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")
    import sys

    import rasterio
    from conftest import UTM, X0, Y0
    from rasterio.transform import from_origin

    from expds import train as tr

    cache = tmp_path / "cache"
    cache.mkdir()
    rng = np.random.default_rng(0)
    np.save(cache / "images_0000.npy", rng.integers(0, 255, (4, 3, 32, 32), dtype=np.uint8))
    np.save(cache / "masks_0000.npy", np.zeros((4, 32, 32), np.uint8))
    (cache / "index.csv").write_text("shard,idx,region,family,file,cx,cy,positive\n" + "".join(
        f"0,{i},r{i},{'A' if i % 2 else 'B'},f.tif,0,0,0\n" for i in range(4)), encoding="utf-8")
    (cache / "meta.json").write_text(json.dumps({"markup": "sp_or", "classes": ["arkhitektury"], "res": 0.5,
                                                 "tile": 32, "flips": True, "channels": 3}), encoding="utf-8")
    tr.train(str(cache), str(tmp_path / "m.pt"), epochs=1, batch=4, val_families=["B"], workers=0,
             pretrained=False, gpu_data=False)
    reg = tmp_path / "root" / "region_0003"
    tif = reg / "05_region_0003_Or_ортофото" / "05_region_0003_Or_ортофото.tif"
    tif.parent.mkdir(parents=True)
    with rasterio.open(tif, "w", driver="GTiff", height=200, width=200, count=3, dtype="uint8",
                       transform=from_origin(X0, Y0 + 20, 0.1, 0.1), crs=UTM) as ds:
        ds.write(rng.integers(0, 255, (3, 200, 200), dtype=np.uint8))
    (reg / "UTM.json").write_text(json.dumps({"utm": "36N"}), encoding="utf-8")
    sys.path.insert(0, str(ROOT / "scripts"))
    import nn_oof

    out = tmp_path / "or_from_spor.geojson"
    monkeypatch.setattr(sys, "argv", ["nn_oof.py", "--markup", "or", "--root", str(tmp_path / "root"),
                                      "--weights", str(tmp_path / "m.pt"), "--out", str(out)])
    nn_oof.main()
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["regions"] == 1 and printed["markups"] == ["or"]
    fc = json.loads(out.read_text(encoding="utf-8"))
    assert fc["type"] == "FeatureCollection"
    assert all(f["properties"]["markup_type"] == "or" for f in fc["features"])


def test_oof_transform_pick_splits_a_joint_file_by_markup(tmp_path, monkeypatch):
    import sys

    def feat(mt, cls, x):
        return {"type": "Feature", "geometry": {"type": "Point", "coordinates": [x, 0.0]},
                "properties": {"markup_type": mt, "class_name": cls, "confidence": 0.5, "region_name": "r"}}

    src = tmp_path / "joint.geojson"
    src.write_text(json.dumps({"type": "FeatureCollection", "features": [
        feat("or", "arkhitektury", 1), feat("sp_or", "arkhitektury", 2), feat("or", "gorodishcha", 3)]}),
        encoding="utf-8")
    spec = importlib.util.spec_from_file_location("oof_transform", ROOT / "scripts" / "oof_transform.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    out = tmp_path / "or.geojson"
    monkeypatch.setattr(sys, "argv", ["oof_transform.py", "pick", "--src", str(src), "--markups", "or",
                                      "--classes", "arkhitektury", "--out", str(out)])
    m.main()
    got = json.loads(out.read_text(encoding="utf-8"))["features"]
    assert [(f["properties"]["markup_type"], f["properties"]["class_name"]) for f in got] == [("or", "arkhitektury")]
