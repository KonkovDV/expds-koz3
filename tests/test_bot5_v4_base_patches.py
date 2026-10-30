"""bot-5 v4: патчи bot-3 уже в базе ветки — повторный cherry-pick не нужен.

История tip:
- 3e7e7c82 ← e3c4f41 (ground_metres opt-in для EPSG:3857)
- 26ab4187 ← e0ec0ff (dedupe_union правило U: класс+markup; EtGr ещё source)
Полные сценарии — tests/test_ground_metres.py, tests/test_union_layers.py.
Здесь только страж «дефолт OFF / ключ группы» для сборки v4.
"""
import pytest
from shapely.geometry import Point

from expds import crs


def test_v4_ground_metres_opt_in_default_off():
    assert crs.ground_metres_on() is False
    assert crs.ground_scale("EPSG:3857", bounds=(0, 0, 1, 1)) == 1.0


def test_v4_dedupe_union_rule_u_present():
    G = pytest.importorskip("expds.geo_pixgbm", reason="expds/geo_pixgbm.py is held out of main (EhimenNathan-derived lineage)")
    _union_key, dedupe_union = G._union_key, G.dedupe_union
    mag_a = {"geom": Point(0, 0).buffer(3), "class_name": "object_poly",
             "markup_type": "maggr", "source": "gray.tif"}
    mag_b = dict(mag_a, source="color.tif", geom=Point(0, 0).buffer(1))
    et_a = dict(mag_a, markup_type="etgr", source="et_01.tif")
    et_b = dict(mag_a, markup_type="etgr", source="et_02.tif", geom=Point(0, 0).buffer(1))
    assert _union_key(mag_a) == _union_key(mag_b)
    assert _union_key(et_a) != _union_key(et_b)
    assert dedupe_union([mag_a], [mag_b]) == []          # maggr: слой не разделяет
    assert dedupe_union([et_a], [et_b]) == [et_b]         # etgr: другой source сохраняется
