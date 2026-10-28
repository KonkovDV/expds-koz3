"""Грубый проход городищ (hillfort.COARSE): крупная площадка, которую обычный проход отсекает."""
import numpy as np
from scipy import ndimage

from expds.hillfort import COARSE, HillfortParams, candidates, coarse_params, find
from expds.lidar import Dem


def big_terrain(step: float = 3.0, n: int = 260) -> np.ndarray:
    """Шаг 3 м, 780×780 м: площадка 300×270 м (8,1 га) на 6 м выше с обрывом ~9 м и пологий холм."""
    rng = np.random.default_rng(1)
    z = np.full((n, n), 100.0) + rng.normal(0, 0.02, (n, n))
    cape = np.zeros((n, n), bool)
    cape[40:130, 40:140] = True
    outside = ndimage.distance_transform_edt(~cape) * step
    z += 6.0 * np.clip(1 - outside / 9.0, 0, 1)
    yy, xx = np.mgrid[0:n, 0:n]
    z += 6.0 * np.exp(-((xx - 200) ** 2 + (yy - 200) ** 2) / (2 * 25.0 ** 2))
    return z.astype(np.float32)


def test_coarse_finds_a_large_plateau_that_the_fine_pass_rejects():
    z = big_terrain()
    fine = find(z, 3.0, HillfortParams())   # окно 60 м: от площадки 8 га остаётся «подкова» у края
    assert all(c["area_m2"] < 0.5 * 81_000 for c in fine)
    found = find(z, 3.0, coarse_params())
    assert found, "грубый проход не нашёл площадку 8 га"
    best = found[0]
    assert 40 <= best["row"] < 130 and 40 <= best["col"] < 140
    assert best["area_m2"] > 0.8 * 81_000 and best["confidence"] > max(c["confidence"] for c in fine)
    assert all(not (170 < c["row"] < 230 and 170 < c["col"] < 230) for c in found)   # пологий холм — нет


def test_coarse_candidates_have_their_own_group_and_override():
    d = Dem(dem=big_terrain(), left=500_000.0, top=6_100_780.0, res=3.0, crs="EPSG:32636")
    out = candidates(d, "r", "x.las", coarse_params())
    assert out and all(f["threshold_group"] == "gor_coarse_dem" for f in out)
    assert out[0]["geom"].area < 200                     # тот же круг ≤3 м, что и у gor_phys
    assert coarse_params({"max_per_dem": 1}).max_per_dem == 1 and COARSE["max_per_dem"] == 3
    assert HillfortParams().group == "gor_phys"          # обычный проход не изменился
