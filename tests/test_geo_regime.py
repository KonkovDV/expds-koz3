"""Режим геофизики на площадке по сумме F1 и счёту выданного (scripts/geo_regime.py)."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("geo_regime", ROOT / "scripts" / "geo_regime.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def f1(t, f, g):
    return 2 * t / (g + t + 0.5 * f)


def test_census_pins_the_regime_where_the_log_alone_cannot():
    m = _mod()
    # находки почти пусты: object_poly 150 эталонов, 75 TP, 10 FP; находки 150 эталонов, 5 TP, 30 FP
    s = f1(75, 10, 150) + f1(5, 30, 150)
    alone = m.feasible(80, 40, 300, s, tol=2e-3)
    # по одной сумме F1 возможны и «находки пусты», и «object_poly пуст»
    assert alone["finds_tp"][0] <= 5 and alone["finds_tp"][1] >= 75
    pinned = m.feasible(80, 40, 300, s, tol=2e-3, n_op=85, n_f=35)
    assert pinned["solutions"] > 0
    assert pinned["finds_tp"][0] <= 5 <= pinned["finds_tp"][1] <= 35


def test_census_counts_only_geophysics(tmp_path):
    m = _mod()
    feats = [{"type": "Feature", "geometry": None, "properties": {"class_name": c, "markup_type": mt}}
             for c, mt in [("finds_points", "maggr"), ("object_poly", "etgr"), ("object_poly", "etgr"),
                           ("kurgany_tselye", "li")]]
    p = tmp_path / "r.geojson"
    p.write_text(json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8")
    c = m.census(p)
    assert c["by_class"] == {"finds_points": 1, "object_poly": 2}
    assert c["by_pair"] == {"finds_points/maggr": 1, "object_poly/etgr": 2}
