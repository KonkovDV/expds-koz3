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
    """Сдача зовёт генератор только при [hillfort] enabled = true; ЦМР — своя, как в hillfort_oof.py."""
    from expds import hillfort, pipeline
    from expds.catalog import scan_root

    class FakeSeg:
        def __init__(self):
            self.meta: dict = {}
            self.classes = ["kurgany_povrezhdennye"]

    calls = []
    monkeypatch.setattr(pipeline, "_cached_segmenters", lambda *a, **k: [FakeSeg()])
    monkeypatch.setattr(pipeline, "_li_with_model", lambda *a, **k: [])
    dems = []
    monkeypatch.setattr(hillfort, "candidates", lambda d, region, source, p=None: calls.append(p) or dems.append(d) or [
        {"class_name": "gorodishcha", "threshold_group": "gor_phys"}])
    reg = next(r for r in scan_root(val_dir) if r.las)
    off = pipeline.lidar_feats(reg, {}, pipeline.TimeGuard(1000), {}, val_dir)
    assert calls == [] and not any(f.get("threshold_group") == "gor_phys" for f in off)
    on = pipeline.lidar_feats(reg, {"hillfort": {"enabled": True, "scarp_min_deg": 15}},
                              pipeline.TimeGuard(1000), {}, val_dir)
    assert len(calls) == 1 and calls[0].scarp_min_deg == 15
    assert any(f.get("threshold_group") == "gor_phys" for f in on)
    las = reg.las[0]
    same = hillfort.build_dem(las, pipeline.region_crs(reg, {}), calls[0])
    assert dems[0].res == same.res == HillfortParams().dem_res                  # не ЦМР модели курганов 0,1 м
    assert np.array_equal(np.nan_to_num(dems[0].dem, nan=-1), np.nan_to_num(same.dem, nan=-1))


def cape(size: int, n: int = 700) -> tuple[np.ndarray, np.ndarray]:
    """Мыс size×size м на 6 м выше поймы, обрыв 4 м с трёх сторон, с напольной — ров 6 м и коренной берег."""
    rng = np.random.default_rng(1)
    z = np.full((n, n), 100.0) + rng.normal(0, 0.03, (n, n))
    top = np.zeros((n, n), bool)
    r0 = c0 = (n - size) // 2
    top[r0:r0 + size, c0:c0 + size] = True
    top[: r0 - 60, :] = True
    top[r0 - 60: r0, c0 - 20: c0 + size + 20] = True
    z += 6.0 * np.clip(1 - ndimage.distance_transform_edt(~top) / 4.0, 0, 1)
    z[r0 - 3: r0 + 3, c0: c0 + size] -= 2.0
    site = np.zeros((n, n), bool)
    site[r0 + 3: r0 + size, c0: c0 + size] = True
    return z.astype(np.float32), site


def test_wide_cape_keeps_its_middle_and_its_score():
    """Площадка 3,2 га шире окна 120 м: второй проход держит её целиком, а не подковой у края."""
    z, site = cape(180)
    one = find(z, 1.0, HillfortParams(relief_radius_l_m=0))
    two = find(z, 1.0, HillfortParams(relief_radius_l_m=150))
    b1 = next(c for c in one if site[c["row"], c["col"]])
    b2 = next(c for c in two if site[c["row"], c["col"]])
    assert two[0] is b2
    assert b1["area_m2"] < 0.5 * site.sum() < b2["area_m2"]
    assert b2["enclosure"] > b1["enclosure"] + 0.15 and b2["confidence"] > b1["confidence"] + 0.1


def test_small_cape_is_not_swallowed_by_the_upland():
    """Широкое окно сливает мыс 0,16 га с коренным берегом через дно рва; остаётся более обрывистый мыс."""
    z, site = cape(40)
    found = find(z, 1.0, HillfortParams(relief_radius_l_m=150))
    assert site[found[0]["row"], found[0]["col"]] and found[0]["enclosure"] > 0.6
    assert sum(site[c["row"], c["col"]] for c in found) == 1
