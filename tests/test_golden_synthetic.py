"""Сквозной прогон inference/main.py на синтетике всех типов съёмки с весами из configs/default.toml.

Ловит незаметную смену выхода по умолчанию (аудит 03.11: каждое слияние веток проверялось этим прогоном).
Синтетика: LAS + аэрофото (region_0001), магниторазведка (region_0002), спутник + ортофото (region_0003).
Сверяются число объектов по (markup_type, class_name) и валидность файла по Прил. 3 ТР, не sha256:
байты зависят от сборки BLAS, а состав выхода — нет.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from conftest import MOUNDS, UTM, X0, Y0, write_las, write_mag_tif

ROOT = Path(__file__).resolve().parents[1]

#: Состав выхода main на 03.11 (main 35de7a9c, CPU, OMP 1): sha256 d2a76542… того же прогона — docs/AUDIT_2026-11-03.md §4.
EXPECTED = {
    ("ae", "kurgany_povrezhdennye"): 3,
    ("ae", "kurgany_tselye"): 3,
    ("li", "kurgany_tselye"): 3,
    ("maggr", "finds_points"): 70,
    ("maggr", "object_poly"): 1,
    ("or", "arkhitektury"): 1,
}


def _write_rgb(path: Path, size_m: float, res: float, seed: int, bands: int) -> None:
    import rasterio
    from rasterio.transform import from_origin

    rng = np.random.default_rng(seed)
    n = int(size_m / res)
    yy, xx = np.mgrid[0:n, 0:n] * res
    base = rng.normal(120, 12, (n, n))
    for mx, my, _h, r in MOUNDS:
        base += 40 * np.exp(-((xx - mx) ** 2 + (yy - (size_m - my)) ** 2) / (2 * (r / 2) ** 2))
    base[(xx > 60) & (xx < 70) & (yy > 20) & (yy < 28)] += 60
    rr = np.hypot(xx - 40, yy - 50)
    base[(rr > 25) & (rr < 28)] -= 50
    a = np.clip(base, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=n, width=n, count=bands, dtype="uint8",
                       transform=from_origin(X0, Y0 + size_m, res, res), crs=UTM) as ds:
        for b in range(bands):
            ds.write(a, b + 1)


def _models_present() -> bool:
    import tomllib

    from expds.pipeline import model_files, model_specs

    cfg = tomllib.loads((ROOT / "configs" / "default.toml").read_text(encoding="utf-8"))
    models = cfg.get("models", {})
    return all((ROOT / f).is_file() for mt in models for p, _ in model_specs(models, mt) for f in model_files(p))


@pytest.mark.skipif(not _models_present(), reason="веса из configs/default.toml не на месте")
def test_default_output_on_synthetic_multimodal_input(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")
    inp = tmp_path / "input"
    r1 = inp / "region_0001"
    write_las(r1 / "01_region_0001_La_точки" / "01_region_0001_La_точки.las", n=200_000)
    (r1 / "UTM.json").write_text(json.dumps({"utm": "36N"}), encoding="utf-8")
    _write_rgb(r1 / "02_region_0001_Ae" / "01_region_0001_Ae.tif", 100.0, 0.5, 3, bands=1)
    r2 = inp / "region_0002"
    write_mag_tif(r2 / "07_region_0002_MagGr_карты" / "01_region_0002_MagGr.tif")
    (r2 / "UTM.json").write_text(json.dumps({"utm": "36N"}), encoding="utf-8")
    r3 = inp / "region_0003"
    _write_rgb(r3 / "03_region_0003_SpOr" / "01_region_0003_SpOr.tif", 120.0, 0.5, 5, bands=3)
    _write_rgb(r3 / "04_region_0003_Or" / "01_region_0003_Or.tif", 100.0, 0.25, 6, bands=3)
    (r3 / "UTM.json").write_text(json.dumps({"utm": "36N"}), encoding="utf-8")

    out = tmp_path / "out" / "result.geojson"
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([sys.executable, str(ROOT / "inference" / "main.py"), str(inp), str(out)],
                       capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=900)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "Traceback" not in r.stdout + r.stderr, (r.stdout + r.stderr)[-3000:]

    from expds.writer import validate

    rep = validate(out)
    assert rep["ok"], rep["errors"]
    fc = json.loads(out.read_text(encoding="utf-8"))
    got = Counter((f["properties"].get("markup_type"), f["properties"]["class_name"]) for f in fc["features"])
    assert dict(got) == EXPECTED
    assert sorted(p.name for p in out.parent.iterdir()) == ["result.geojson"]   # без хвостов .expds_tmp / writing.*
