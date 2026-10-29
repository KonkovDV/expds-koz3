"""ae_layer_audit.build_gt: кириллический код «Ае» в имени файла учитывается, пустые фигуры отбрасываются (bot-6)."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("ae_layer_audit", ROOT / "scripts" / "ae_layer_audit.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _fc(features):
    return {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::3857"}},
            "features": features}


def test_cyrillic_ae_and_empty(tmp_path):
    d = tmp_path / "999_Тест" / "06_Тест_разметка"
    d.mkdir(parents=True)
    sq = {"type": "Polygon", "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]}
    empty = {"type": "Polygon", "coordinates": []}
    feats = [{"type": "Feature", "properties": {}, "geometry": sq},
             {"type": "Feature", "properties": {}, "geometry": empty}]
    (d / "Тест_Ae_курганы_целые.geojson").write_text(json.dumps(_fc(feats)), encoding="utf-8")       # латиница
    (d / "Тест_Ае_курганы_поврежденные.geojson").write_text(json.dumps(_fc(feats)), encoding="utf-8")  # кириллица «Ае»
    (d / "Тест_Li_архитектуры.geojson").write_text(json.dumps(_fc(feats[:1])), encoding="utf-8")
    got = _mod().build_gt(tmp_path)
    by = sorted((f["properties"]["markup_type"], f["properties"]["class_name"]) for f in got)
    assert by == [("ae", "kurgany_povrezhdennye"), ("ae", "kurgany_tselye"), ("li", "arkhitektury")]
