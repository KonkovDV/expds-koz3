"""Регрессия сквозного v3 на mock (ПК 29.10, zip v3on_0876090).

F1 ALL 0,3125 → 0,2880, sp_or 0,2429 → 0. D-FINE: 482 сырых, drop_groups снял 46
объектов v2, после порога осталось 13 и ни один не попал в эталон (50–1500 м).

Проверено на синтетике, без весов:
* пиксельный бокс на растре EPSG:32636 и на растре EPSG:3857 (варп в UTM, как патч
  метров на земле e3c4f41) попадает в эталонную точку EPSG:3857 ближе 2 м;
* индекс класса 1 — kurgany_povrezhdennye, как id2label модели f0;
* обнуление sp_or даёт слепой drop_groups до порога, а не сдвиг CRS.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import Point

from expds import sp_dfine

Y = 7_361_866.0  # ~55° с.ш. в EPSG:3857


def _east(metres: float) -> float:
    """Сдвиг на восток в единицах EPSG:3857, чтобы на земле вышло ``metres`` метров."""
    return metres / sp_dfine.ground_scale(Y)


def _circ(x, y, cls="kurgany_povrezhdennye", group="sp_or", conf=0.9, region="101"):
    return {
        "geom": Point(x, Y).buffer(_east(4.0)),
        "class_name": cls,
        "confidence": conf,
        "region_name": region,
        "markup_type": "sp_or",
        "threshold_group": group,
    }


def test_pixel_box_utm_and_3857_match_gt_within_2m(tmp_path):
    """Известный пиксель → полигон в EPSG:3857 не дальше 2 м от эталонной точки."""
    import rasterio
    from pyproj import Transformer
    from rasterio.transform import from_origin

    # Растр уже в зоне участка (101/103 — EPSG:32636). Центр бокса — пиксель (64, 64).
    utm = tmp_path / "utm.tif"
    with rasterio.open(utm, "w", driver="GTiff", width=200, height=200, count=3, dtype="uint8",
                       crs="EPSG:32636", transform=from_origin(500000, 6100000, 0.5, 0.5)) as ds:
        ds.write(np.full((3, 200, 200), 100, np.uint8))
    with rasterio.open(utm) as ds:
        affine = ds.transform
    box = np.array([[60.0, 60.0, 68.0, 68.0]])
    ell = sp_dfine.boxes_to_ellipses(box, np.array([0.8]), np.array([1]), affine, "EPSG:32636", "r", "utm.tif", 0)
    east, north = 500000 + 64 * 0.5, 6100000 - 64 * 0.5
    gx, gy = Transformer.from_crs("EPSG:32636", "EPSG:3857", always_xy=True).transform(east, north)
    assert ell[0]["class_name"] == "kurgany_povrezhdennye"
    assert ell[0]["geom"].centroid.distance(Point(gx, gy)) < 2.0

    # Тот же мир, но исходный растр в EPSG:3857: сцена варпится в UTM с шагом 0,5 м.
    # Ошибочная подпись «это уже 3857» уводит точку на тысячи километров; код так не делает.
    e2, n2 = 500050.0, 6099950.0
    x2, y2 = Transformer.from_crs("EPSG:32636", "EPSG:3857", always_xy=True).transform(e2, n2)
    web = tmp_path / "web.tif"
    with rasterio.open(web, "w", driver="GTiff", width=80, height=80, count=3, dtype="uint8",
                       crs="EPSG:3857", transform=from_origin(x2 - 40, y2 + 40, 1.0, 1.0)) as ds:
        ds.write(np.full((3, 80, 80), 90, np.uint8))
    scene = sp_dfine.Scene(web, "EPSG:32636", 0.5)
    try:
        col, row = (~scene.T) * (e2, n2)
        boxed = np.array([[col - 2, row - 2, col + 2, row + 2]])
        got = sp_dfine.boxes_to_ellipses(
            boxed, np.array([0.8]), np.array([0]), scene.T, "EPSG:32636", "r", "web.tif", 0)
        assert got[0]["class_name"] == sp_dfine.CLASSES[0] == "kurgany_tselye"
        assert got[0]["geom"].centroid.distance(Point(x2, y2)) < 2.0
        mislabeled = sp_dfine.boxes_to_ellipses(
            boxed, np.array([0.8]), np.array([0]), scene.T, "EPSG:3857", "r", "web.tif", 0)
        assert mislabeled[0]["geom"].centroid.distance(Point(x2, y2)) > 1000.0
    finally:
        scene.close()


def test_class_index_matches_f0_id2label():
    """Порядок головы f0: 0 целые, 1 повреждённые, 2 городища, 3 фортификации, 4 архитектура."""
    assert sp_dfine.CLASSES == (
        "kurgany_tselye", "kurgany_povrezhdennye", "gorodishcha", "fortifikatsii", "arkhitektury")


def test_below_threshold_and_far_dfine_do_not_cover():
    """Порог povr 0,555 (как R_max f0). Бокс 0,10 на эталоне и бокс 0,90 в 200 м ничего не снимают."""
    cfg = {"thresholds_by_markup": {"sp_dfine": {"kurgany_povrezhdennye": 0.555265}}}
    v2 = [_circ(0, Y), _circ(_east(40), Y, region="101")]
    low = _circ(0, Y, group="sp_dfine", conf=0.10)
    far = _circ(_east(200), Y, group="sp_dfine", conf=0.90)
    assert sp_dfine.passes_group_threshold(low, cfg) is False
    assert sp_dfine.passes_group_threshold(far, cfg) is True
    survivors = [f for f in (low, far) if sp_dfine.passes_group_threshold(f, cfg)]
    out = sp_dfine.drop_groups_covered(v2 + [low, far], ["sp_or", "sp_focal", "sp_arch", "sp_focal2"],
                                       ["101"], 12.0, survivors)
    kept_v2 = [f for f in out if f.get("threshold_group") != "sp_dfine"]
    assert len(kept_v2) == 2
    assert all(f in out for f in (low, far))


def test_same_class_within_cover_replaces_only_that_object():
    cfg = {"thresholds_by_markup": {"sp_dfine": {"kurgany_povrezhdennye": 0.555265}}}
    near_v2 = _circ(0, Y)
    other = _circ(_east(40), Y)
    other_class = _circ(_east(80), Y, cls="arkhitektury")
    hit = _circ(_east(2), Y, group="sp_dfine", conf=0.80)
    assert sp_dfine._ground_dist((0, Y), (_east(2), Y)) < 12.0
    out = sp_dfine.drop_groups_covered(
        [near_v2, other, other_class, hit],
        ["sp_or", "sp_focal", "sp_arch", "sp_focal2"], ["101"], 12.0, [hit])
    assert near_v2 not in out
    assert other in out and other_class in out and hit in out


def test_phase_keeps_v2_when_thresholded_dfine_is_far(monkeypatch):
    """Как на mock: много сырых D-FINE, выше порога только далёкий FP — v2 на GT остаётся."""
    from expds import pipeline

    class St:
        def __init__(self, feats):
            self.feats = list(feats)
            self.note = {}
        def add(self, feats):
            self.feats += feats
        def snapshot(self):
            return list(self.feats)
        def replace(self, feats):
            self.feats = list(feats)

    class G:
        soft_ceiling = None
        def ok(self, _s=0):
            return True
        def elapsed(self):
            return 0.0

    v2 = [_circ(_east(i * 30), Y, region="101") for i in range(4)]
    # 3 ниже порога (в том числе один прямо на v2) и 1 выше порога в 200 м — аналог 482 → 13.
    raw = [_circ(0, Y, group="sp_dfine", conf=0.05, region="101"),
           _circ(_east(200), Y, group="sp_dfine", conf=0.90, region="101")]

    def fake(reg, cfg, guard, cache, root, crs_for=None, soft_ok=None):
        cache["models"] = ["m"]
        return list(raw), True

    monkeypatch.setattr(sp_dfine, "region_feats", fake)
    st = St(v2)
    cfg = {"sp_dfine": {"enabled": True, "drop_groups": ["sp_or", "sp_focal", "sp_arch", "sp_focal2"]},
           "thresholds_by_markup": {"sp_dfine": {"kurgany_povrezhdennye": 0.555265}}}
    assert pipeline._sp_dfine_phase(
        [SimpleNamespace(name="101", rasters={"sp_or": ["x"]})], cfg, G(), st, None) == 2
    kept = [f for f in st.feats if f.get("threshold_group") != "sp_dfine"]
    assert kept == v2


def test_phase_blind_r_still_available_when_cover_disabled(monkeypatch):
    from expds import pipeline

    class St:
        def __init__(self, feats):
            self.feats = list(feats)
            self.note = {}
        def add(self, feats):
            self.feats += list(feats)
        def snapshot(self):
            return list(self.feats)
        def replace(self, feats):
            self.feats = list(feats)

    class G:
        soft_ceiling = None
        def ok(self, _s=0):
            return True
        def elapsed(self):
            return 0.0

    def fake(reg, cfg, guard, cache, root, crs_for=None, soft_ok=None):
        cache["models"] = ["m"]
        return [_circ(_east(200), Y, group="sp_dfine", conf=0.9, region="101")], True

    monkeypatch.setattr(sp_dfine, "region_feats", fake)
    st = St([_circ(0, Y, region="101")])
    cfg = {"sp_dfine": {"enabled": True, "drop_groups": ["sp_or"], "drop_cover_m": 0}}
    pipeline._sp_dfine_phase([SimpleNamespace(name="101", rasters={"sp_or": ["x"]})], cfg, G(), st, None)
    assert all(f.get("threshold_group") == "sp_dfine" for f in st.feats)
