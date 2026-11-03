"""e3c4f41 (метры на земле, EPSG:3857) не двигает боксы D-FINE и не возвращает sp_or.

Патч bot-3 меняет только шаг геофизики: ``crs.ground_scale`` / ``[geophys] ground_metres``
/ ``EXPDS_GROUND_M``. По умолчанию множитель 1. ``sp_dfine.boxes_to_ellipses`` переводит
пиксели сцены (UTM, аффин) в EPSG:3857 своим Transformer и этот флаг не читает.

Регрессия v3on_0876090 (mock, F1 sp_or 0.2429 → 0.0) поэтому этим патчем не лечится.
v3 сама под подозрением: тест не объявляет её верной базой. Разбор mock-дампа —
``reports/bot2_dfine_a3_localization.md`` (другой бот); здесь только то, что считается
на синтетике без весов и без GPU.
"""
from __future__ import annotations

import math
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pyproj import Transformer
from shapely.geometry import Point

from expds import crs, post, sp_dfine
from expds import dfine_aeor

ROOT = Path(__file__).resolve().parents[1]
# UTM 36N, северная ориентация, 0,5 м/пиксель. Центр бокса (100, 100) → (500050, 6100000).
_T = SimpleNamespace(a=0.5, b=0.0, c=500_000.0, d=0.0, e=-0.5, f=6_100_050.0)
_BOX = np.array([[96.0, 96.0, 104.0, 104.0]])
_UTM = "EPSG:32636"


def _ellipse(crs_name: str = _UTM):
    return sp_dfine.boxes_to_ellipses(
        _BOX, np.array([0.8]), np.array([0]), _T, crs_name, "r", "a.tif", 0)[0]


def test_ground_metres_does_not_move_dfine_boxes(monkeypatch):
    """Вкл/выкл ground_metres и EXPDS_GROUND_M: WKT эллипса побайтно один и тот же."""
    monkeypatch.delenv("EXPDS_GROUND_M", raising=False)
    wkts = []
    for flag in (None, False, True):
        monkeypatch.setattr(crs, "GROUND_METRES", flag)
        wkts.append(_ellipse()["geom"].wkt)
    monkeypatch.setenv("EXPDS_GROUND_M", "1")
    monkeypatch.setattr(crs, "GROUND_METRES", None)
    wkts.append(_ellipse()["geom"].wkt)
    assert len(set(wkts)) == 1
    # сам патч при этом жив: для EPSG:3857 множитель не 1, когда его просят
    bounds = (3.6e6, 7.3e6, 3.6e6 + 100, 7.3e6 + 100)
    assert crs.ground_scale("EPSG:3857", bounds, enable=False) == 1.0
    y = (bounds[1] + bounds[3]) / 2.0
    lat = math.degrees(math.atan(math.sinh(y / 6378137.0)))
    assert crs.ground_scale("EPSG:3857", bounds, enable=True) == pytest.approx(math.cos(math.radians(lat)))
    src = Path(sp_dfine.__file__).read_text(encoding="utf-8")
    body = src.split("def boxes_to_ellipses", 1)[1].split("\ndef ", 1)[0]
    assert "GROUND_METRES" not in body and "ground_scale" not in body


def test_wrong_scene_crs_misses_the_utm_ellipse_ground_metres_does_not():
    """Ошибка CRS сцены уносит эллипс на километры. ground_metres такого сдвига не делает."""
    good = _ellipse(_UTM)
    # те же числа аффина, но как будто это уже EPSG:3857 (типичная путаница метров и веб-меркатора)
    bad = _ellipse("EPSG:3857")
    assert good["geom"].contains(good["geom"].centroid)
    assert not bad["geom"].intersects(good["geom"])
    assert good["geom"].centroid.distance(bad["geom"].centroid) > 1_000_000
    # верный эллипс сидит на перепроецированном центре UTM, не на «сырых» метрах
    e, n = 500_000 + 100 * 0.5, 6_100_050 - 100 * 0.5
    x, y = Transformer.from_crs(_UTM, "EPSG:3857", always_xy=True).transform(e, n)
    assert good["geom"].centroid.distance(Point(x, y)) < 0.05
    assert bad["geom"].centroid.distance(Point(e, n)) < 1.0


def test_drop_groups_removes_hitting_unet_regardless_of_ground_metres(monkeypatch):
    """Вариант R снимает U-Net sp_or/sp_focal. Флаг метров на земле объект не возвращает."""
    hit = {"geom": Point(0, 0).buffer(4), "class_name": "kurgany_tselye", "confidence": 0.99,
           "region_name": "mock", "markup_type": "sp_or"}
    df = dict(hit, threshold_group="sp_dfine", confidence=0.7)
    groups = ["sp_or", "sp_focal", "sp_arch", "sp_focal2"]
    monkeypatch.setattr(crs, "GROUND_METRES", True)
    monkeypatch.setenv("EXPDS_GROUND_M", "1")
    out = sp_dfine.drop_groups([hit, df], groups, ["mock"])
    assert out == [df]
    # участок без полного прохода D-FINE не трогается — как в коде фазы
    assert sp_dfine.drop_groups([hit], groups, []) == [hit]


def test_confidence_threshold_ignores_where_the_box_is():
    """482 → 13 на mock — отсечение по уверенности, не по координате. Патч 3857 сюда не входит."""
    near = {"geom": Point(0, 0).buffer(4), "class_name": "kurgany_tselye", "confidence": 0.20,
            "region_name": "mock", "markup_type": "sp_or", "threshold_group": "sp_dfine"}
    far = dict(near, geom=Point(50_000, 0).buffer(4), confidence=0.90)
    by = {"sp_dfine": {"kurgany_tselye": 0.50}}
    kept = post.apply_thresholds([near, far], {}, 0.15, by)
    assert kept == [far]


def test_sp_dfine_flag_is_independent_of_dfine_aeor_and_default_off():
    cfg = tomllib.loads((ROOT / "configs" / "default.toml").read_text(encoding="utf-8"))
    assert cfg["sp_dfine"]["enabled"] is False
    assert cfg["dfine_aeor"]["enabled"] is False
    assert cfg["gor_dense"]["enabled"] is False
    only_ae = {"sp_dfine": {"enabled": False}, "dfine_aeor": {"enabled": True}}
    assert sp_dfine.enabled(only_ae) is False
    assert dfine_aeor.enabled(only_ae) is True
    only_sp = {"sp_dfine": {"enabled": True}, "dfine_aeor": {"enabled": False}}
    assert sp_dfine.enabled(only_sp) is True
    assert dfine_aeor.enabled(only_sp) is False
    src = (ROOT / "expds" / "pipeline.py").read_text(encoding="utf-8")
    # фаза SpOr и фаза Ae/Or — разные if, включение одной не входит в условие другой
    sp_line = next(ln for ln in src.splitlines()
                   if 'if (cfg.get("sp_dfine") or {}).get("enabled"):' in ln)
    ae_line = next(ln for ln in src.splitlines()
                   if 'if (cfg.get("dfine_aeor") or {}).get("enabled"):' in ln)
    assert "dfine_aeor" not in sp_line
    assert "sp_dfine" not in ae_line
