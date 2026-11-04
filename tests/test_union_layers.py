"""Правило U (dedupe_union) не гасит кандидата другого слоя EtGr и другого метода (правила 29.10, п. 4–5)."""
import pytest
from shapely.geometry import Point

G = pytest.importorskip("expds.geo_pixgbm", reason="expds/geo_pixgbm.py is held out of main (EhimenNathan-derived lineage)")
dedupe_union = G.dedupe_union


def _poly(x, y, mt, src, r=3.0):
    return {"geom": Point(x, y).buffer(r), "class_name": "object_poly", "markup_type": mt, "source": src}


def test_etgr_other_layer_kept():
    base = [_poly(0, 0, "etgr", "01_x_EtGr_карта_01_0.2.tif")]
    new = [_poly(0, 0, "etgr", "02_x_EtGr_карта_02_0.5.tif", 1.0)]
    assert dedupe_union(base, new) == new


def test_etgr_same_layer_dropped():
    base = [_poly(0, 0, "etgr", "01_x_EtGr_карта_01_0.2.tif")]
    assert dedupe_union(base, [_poly(0, 0, "etgr", "01_x_EtGr_карта_01_0.2.tif", 1.0)]) == []


def test_other_markup_kept():
    base = [_poly(0, 0, "maggr", "01_x_MagGr_серый.tif")]
    new = [_poly(0, 0, "terfl", "01_x_TerFl.tif", 1.0)]
    assert dedupe_union(base, new) == new


def test_maggr_gray_color_copy_dropped():
    """MagGr: серая и цветная копия — одна съёмка; правило U по-прежнему гасит дубль между копиями."""
    base = [_poly(0, 0, "maggr", "01_x_MagGr_серый.tif")]
    assert dedupe_union(base, [_poly(0, 0, "maggr", "01_x_MagGr_цвет.tif", 1.0)]) == []


def test_finds_radius():
    b = [{"geom": Point(0, 0), "class_name": "finds_points", "markup_type": "maggr", "source": "a"}]
    assert dedupe_union(b, [dict(b[0], geom=Point(0.5, 0))]) == []
    assert len(dedupe_union(b, [dict(b[0], geom=Point(1.5, 0))])) == 1
