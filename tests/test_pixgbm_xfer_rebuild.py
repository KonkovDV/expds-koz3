"""bot-6: synthetic xfer-shard rebuild (no contest data, no network)."""
from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / "kaggle" / "pixgbm_cleanroom" / "rebuild_train_tree.py"


def _load():
    import importlib.util

    spec = importlib.util.spec_from_file_location("rebuild_train_tree", MOD)
    m = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(m)
    return m


def test_rebuild_geo_filter_and_symlink(tmp_path: Path):
    m = _load()
    inp = tmp_path / "input"
    # s000 with small tar (plain tar, no zst)
    s000 = inp / "koz3-xfer-s000"
    s000.mkdir(parents=True)
    tar_path = s000 / "small_files.tar"
    with tarfile.open(tar_path, "w") as tf:
        for rel in (
            "200_TestGeo/UTM.json",
            "200_TestGeo/06_разметка/200_TestGeo_MagGr_ObjectPoly.geojson",
            "001_Other/UTM.json",  # should be skipped by geo filter
        ):
            p = tmp_path / "_src" / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("{}", encoding="utf-8")
            tf.add(p, arcname=rel)

    # s001 large blobs
    s001 = inp / "koz3-xfer-s001"
    s001.mkdir()
    blob = s001 / "f0001.tif"
    blob.write_bytes(b"TIFFDATA")
    (s001 / "manifest.json").write_text(
        json.dumps(
            {
                "files": {
                    "f0001": {"rel": "200_TestGeo/02_карты/200_TestGeo_MagGr.tif"},
                    "f0002": {"rel": "001_Other/foo.tif"},  # filtered out
                }
            }
        ),
        encoding="utf-8",
    )
    # missing f0002 blob — ok

    # gt dataset with extra annotation
    gt = inp / "koz3-gt-train-v2"
    gtf = gt / "200_TestGeo" / "extra" / "note.geojson"
    gtf.parent.mkdir(parents=True)
    gtf.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")

    dest = tmp_path / "tree"
    report = m.rebuild_train_tree(dest=dest, input_root=inp)
    assert "200_TestGeo" in report["geo_region_dirs"]
    assert (dest / "200_TestGeo" / "UTM.json").is_file()
    assert not (dest / "001_Other").exists()
    raster = dest / "200_TestGeo" / "02_карты" / "200_TestGeo_MagGr.tif"
    assert raster.is_file() or raster.is_symlink()
    assert raster.read_bytes() == b"TIFFDATA"
    assert (dest / "200_TestGeo" / "extra" / "note.geojson").is_file()


def test_iter_blob_mappings_shapes():
    m = _load()
    pairs = list(
        m.iter_blob_mappings(
            {"files": {"f0010.tif": "200_X/a.tif", "f0011": {"rel": "200_X/b.tif"}}}
        )
    )
    assert ("f0010.tif", "200_X/a.tif") in pairs
    assert ("f0011", "200_X/b.tif") in pairs
