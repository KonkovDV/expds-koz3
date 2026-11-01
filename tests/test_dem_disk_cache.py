"""Дисковый кэш DEM (bots/bot1-speedup-1101): побайтно тот же выход на ≥2 участках + pytest."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from conftest import ROOT, UTM, write_las
from expds import demcache, speed
from expds.lidar import DemPrefetch, build_dem, build_dem_mosaic

pytest.importorskip("laspy")


@pytest.fixture(autouse=True)
def _clean_speed(monkeypatch, tmp_path):
    monkeypatch.delenv("EXPDS_SPEEDUP", raising=False)
    for f in speed.FLAGS:
        monkeypatch.delenv(f"EXPDS_SPEED_{f.upper()}", raising=False)
    monkeypatch.delenv("EXPDS_DEM_DISK_CACHE", raising=False)
    monkeypatch.delenv("EXPDS_RUST_MATCH", raising=False)
    speed.configure(None)
    demcache.invalidate_code_fp()
    yield
    speed.configure(None)
    monkeypatch.delenv("EXPDS_DEM_DISK_CACHE", raising=False)


def _sha(dem) -> str:
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(dem, np.float32).tobytes()).hexdigest()


def test_dem_disk_cache_byte_identical_two_regions(tmp_path, monkeypatch):
    cache = tmp_path / "demc"
    monkeypatch.setenv("EXPDS_DEM_DISK_CACHE", str(cache))
    regions = []
    for i, seed in enumerate((7, 11)):
        las = write_las(tmp_path / f"r{i}.las", n=40_000, seed=seed)
        cold = build_dem(las, res=0.5, fill="delaunay", crs=UTM)
        assert cold.info.get("dem_cache") == "miss"
        t0 = time.perf_counter()
        warm = build_dem(las, res=0.5, fill="delaunay", crs=UTM)
        warm_s = time.perf_counter() - t0
        assert warm.info.get("dem_cache") == "hit"
        assert cold.dem.shape == warm.dem.shape
        assert cold.dem.tobytes() == warm.dem.tobytes()
        assert cold.left == warm.left and cold.top == warm.top and cold.res == warm.res
        # без кэша — та же сетка
        monkeypatch.setenv("EXPDS_DEM_DISK_CACHE", "off")
        ref = build_dem(las, res=0.5, fill="delaunay", crs=UTM)
        monkeypatch.setenv("EXPDS_DEM_DISK_CACHE", str(cache))
        assert ref.dem.tobytes() == cold.dem.tobytes()
        regions.append({"las": str(las), "sha": _sha(cold.dem), "warm_s": warm_s})
    assert len({r["sha"] for r in regions}) == 2
    assert all(r["warm_s"] < 1.5 for r in regions)


def test_dem_mosaic_cache_byte_identical(tmp_path, monkeypatch):
    cache = tmp_path / "demc"
    monkeypatch.setenv("EXPDS_DEM_DISK_CACHE", str(cache))
    las = write_las(tmp_path / "noclass2.las", n=25_000, seed=3, ground_class=1)
    a = build_dem_mosaic(las, res=0.1, work_res=0.5, tile_m=40.0, halo_m=4.0, fill="delaunay", crs=UTM)
    b = build_dem_mosaic(las, res=0.1, work_res=0.5, tile_m=40.0, halo_m=4.0, fill="delaunay", crs=UTM)
    assert a.dem.tobytes() == b.dem.tobytes()
    assert b.info.get("dem_cache") == "hit"


def test_speed_flag_enables_default_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("EXPDS_SPEED_DEM_DISK_CACHE", "1")
    assert "dem_disk_cache" in speed.configure({})
    assert os.environ.get("EXPDS_DEM_DISK_CACHE")
    monkeypatch.setenv("EXPDS_SPEED_RUST_MATCH", "1")
    speed.configure({"rust_match": True})
    assert os.environ.get("EXPDS_RUST_MATCH") == "1"
    # вне зонтика (явные флаги не входят в enabled=true)
    monkeypatch.delenv("EXPDS_SPEED_DEM_DISK_CACHE", raising=False)
    monkeypatch.delenv("EXPDS_DEM_DISK_CACHE", raising=False)
    monkeypatch.delenv("EXPDS_SPEED_RUST_MATCH", raising=False)
    monkeypatch.delenv("EXPDS_RUST_MATCH", raising=False)
    speed.configure({"enabled": True})
    assert "dem_disk_cache" not in speed.active()
    assert "rust_match" not in speed.active()


def test_prefetch_uses_disk_cache(tmp_path, monkeypatch):
    cache = tmp_path / "demc"
    monkeypatch.setenv("EXPDS_DEM_DISK_CACHE", str(cache))
    a = write_las(tmp_path / "a.las", n=15_000, seed=1)
    b = write_las(tmp_path / "b.las", n=15_000, seed=2)
    ref_a = build_dem(a, res=0.5, fill="delaunay", crs=UTM)
    ref_b = build_dem(b, res=0.5, fill="delaunay", crs=UTM)
    pf = DemPrefetch([(a, UTM), (b, UTM)], depth=1)
    try:
        got_a = pf.get(a, 0.5, "delaunay", UTM)
        got_b = pf.get(b, 0.5, "delaunay", UTM)
    finally:
        pf.close()
    assert got_a.dem.tobytes() == ref_a.dem.tobytes()
    assert got_b.dem.tobytes() == ref_b.dem.tobytes()


def test_result_geojson_byte_identical_with_dem_cache(val_dir, tmp_path, monkeypatch):
    """Два прогона inference/main.py на 2 синтетических участках: result.geojson побайтно равен
    с холодным и тёплым DEM-кэшем (кэш не меняет детекции)."""
    cache = tmp_path / "demc"
    out1 = tmp_path / "o1" / "result.geojson"
    out2 = tmp_path / "o2" / "result.geojson"
    env = os.environ.copy()
    env.pop("EXPDS_SPEEDUP", None)
    for f in speed.FLAGS:
        env.pop(f"EXPDS_SPEED_{f.upper()}", None)
    env["EXPDS_DEM_DISK_CACHE"] = str(cache)
    # укороченный бюджет сторожа не нужен — синтетика быстрая
    cmd = [sys.executable, str(ROOT / "inference" / "main.py"), str(val_dir)]
    r1 = subprocess.run(cmd + [str(out1)], capture_output=True, text=True, timeout=600, env=env)
    assert r1.returncode == 0, r1.stdout + r1.stderr
    r2 = subprocess.run(cmd + [str(out2)], capture_output=True, text=True, timeout=600, env=env)
    assert r2.returncode == 0, r2.stdout + r2.stderr
    b1, b2 = out1.read_bytes(), out2.read_bytes()
    assert b1 == b2 and b1.startswith(b"{")
    # и против прогона без кэша
    env["EXPDS_DEM_DISK_CACHE"] = "off"
    out0 = tmp_path / "o0" / "result.geojson"
    r0 = subprocess.run(cmd + [str(out0)], capture_output=True, text=True, timeout=600, env=env)
    assert r0.returncode == 0, r0.stdout + r0.stderr
    assert out0.read_bytes() == b1
    # val_dir — ≥2 региона
    regs = [p for p in Path(val_dir).iterdir() if p.is_dir()]
    assert len(regs) >= 2
