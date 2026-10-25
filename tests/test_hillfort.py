"""Кандидаты городищ по форме рельефа (expds/hillfort.py)."""
import numpy as np
from scipy import ndimage

from expds.hillfort import HillfortParams, candidates, find
from expds.lidar import Dem


def terrain(n: int = 400) -> np.ndarray:
    """Равнина 100 м: мыс 80×60 м на 6 м выше с обрывом 6 м на 4 м и пологий холм той же высоты."""
    rng = np.random.default_rng(0)
    z = np.full((n, n), 100.0) + rng.normal(0, 0.02, (n, n))
    cape = np.zeros((n, n), bool)
    cape[60:120, 60:140] = True
    outside = ndimage.distance_transform_edt(~cape)
    z += 6.0 * np.clip(1 - outside / 4.0, 0, 1)
    yy, xx = np.mgrid[0:n, 0:n]
    z += 6.0 * np.exp(-((xx - 290) ** 2 + (yy - 290) ** 2) / (2 * 50.0 ** 2))
    return z.astype(np.float32)


def test_steep_sided_plateau_wins_over_a_gentle_hill():
    found = find(terrain(), 1.0, HillfortParams())
    assert found, "площадка с обрывом не найдена"
    best = found[0]
    assert 60 <= best["row"] < 120 and 60 <= best["col"] < 140
    assert best["enclosure"] > 0.6 and best["height_m"] > 3
    assert all(not (240 < c["row"] < 340 and 240 < c["col"] < 340) for c in found)   # пологий холм — не городище


def test_candidates_are_small_cores_in_3857_with_own_group():
    d = Dem(dem=terrain(), left=500_000.0, top=6_100_400.0, res=1.0, crs="EPSG:32636")
    out = candidates(d, "r", "x.las")
    assert out and out[0]["class_name"] == "gorodishcha" and out[0]["threshold_group"] == "gor_phys"
    g = out[0]["geom"]
    assert g.area < 200 and abs(g.centroid.x) > 1e6            # круг в метрах веб-меркатора, не в пикселях
    assert 0 < out[0]["confidence"] <= 1


def test_flat_or_unreferenced_dem_gives_nothing():
    flat = Dem(dem=np.full((200, 200), 50.0, np.float32), left=0.0, top=200.0, res=1.0, crs="EPSG:32636")
    assert candidates(flat, "r", "x") == []
    assert candidates(Dem(dem=terrain(), left=0.0, top=400.0, res=1.0, crs=None), "r", "x") == []


def test_hillfort_oof_script_runs_over_a_root(val_dir, tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "gor_phys.geojson"
    r = subprocess.run([sys.executable, str(root / "scripts" / "hillfort_oof.py"), "--root", str(val_dir),
                        "--out", str(out), "--set", "min_rel_height_m=0.5"], capture_output=True, text=True, check=True)
    summary = json.loads(r.stdout.strip().splitlines()[-1])
    assert summary["las"] == 1 and summary["params"]["min_rel_height_m"] == 0.5
    fc = json.loads(out.read_text(encoding="utf-8"))
    assert fc["type"] == "FeatureCollection"
    assert all(f["properties"]["threshold_group"] == "gor_phys" for f in fc["features"])


def test_pipeline_hillfort_is_off_by_default_and_on_by_config(val_dir, monkeypatch):
    """Сдача зовёт генератор только при [hillfort] enabled = true; ЦМР — та же, что у модели лидара."""
    from expds import hillfort, pipeline
    from expds.catalog import scan_root

    class FakeSeg:
        def __init__(self):
            self.meta: dict = {}
            self.classes = ["kurgany_povrezhdennye"]

    calls = []
    monkeypatch.setattr(pipeline, "_cached_segmenters", lambda *a, **k: [FakeSeg()])
    monkeypatch.setattr(pipeline, "_li_with_model", lambda *a, **k: [])
    monkeypatch.setattr(hillfort, "candidates", lambda d, region, source, p=None: calls.append(p) or [
        {"class_name": "gorodishcha", "threshold_group": "gor_phys"}])
    reg = next(r for r in scan_root(val_dir) if r.las)
    off = pipeline.lidar_feats(reg, {}, pipeline.TimeGuard(1000), {}, val_dir)
    assert calls == [] and not any(f.get("threshold_group") == "gor_phys" for f in off)
    on = pipeline.lidar_feats(reg, {"hillfort": {"enabled": True, "scarp_min_deg": 15}},
                              pipeline.TimeGuard(1000), {}, val_dir)
    assert len(calls) == 1 and calls[0].scarp_min_deg == 15
    assert any(f.get("threshold_group") == "gor_phys" for f in on)
