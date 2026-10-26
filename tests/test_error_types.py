"""Типы ошибок поверх официального сопоставления (scripts/error_types.py)."""
import importlib.util
import json

from conftest import ROOT, UTM, X0, Y0
from shapely.geometry import Point

from expds.crs import WEB_MERCATOR, to_crs

METRIC = str(ROOT / "materials" / "compute_metrics.py")


def load_script():
    spec = importlib.util.spec_from_file_location("error_types", ROOT / "scripts" / "error_types.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def feat(x, y, r, cls, conf=None, markup="li"):
    g = to_crs(Point(X0 + x, Y0 + y).buffer(r), UTM, WEB_MERCATOR)
    props = {"class_name": cls, "markup_type": markup, "region_name": "001_Тест"}
    if conf is not None:
        props["confidence"] = conf
    return {"type": "Feature", "geometry": g.__geo_interface__, "properties": props}


def scene():
    p, t = "kurgany_povrezhdennye", "kurgany_tselye"
    gt = [feat(0, 0, 5, p), feat(100, 0, 5, p), feat(200, 0, 5, t), feat(400, 0, 5, p)]
    pred = [feat(0, 0, 2, p, 0.9),          # попал
            feat(1, 0, 2, p, 0.8),          # второй на найденном — дубль
            feat(106, 0, 8, p, 0.7),        # центр вне эталона — место
            feat(200, 0, 2, p, 0.6),        # на целом кургане — путаница классов
            feat(300, 0, 2, p, 0.5)]        # фон
    return {"type": "FeatureCollection", "features": pred}, {"type": "FeatureCollection", "features": gt}


def test_each_error_gets_its_type_and_the_score_is_official():
    from expds.metric import score

    et = load_script()
    pred, gt = scene()
    res = et.classify(pred, gt, METRIC)
    assert abs(res["score"] - score(pred, gt, METRIC)["final_score"]) < 1e-9
    pv = res["per_class"]["kurgany_povrezhdennye"]
    assert (pv["tp"], pv["fp"], pv["fn"]) == (1, 4, 2)
    assert pv["fp_types"] == {"duplicate": 1, "localization": 1, "class_confusion": 1, "background": 1}
    assert pv["fn_types"] == {"covered": 1, "missed": 1}
    assert res["per_class"]["kurgany_tselye"]["fn_types"] == {"other_class": 1}
    assert pv["gain"]["localization"] > 0 and pv["gain"]["duplicate"] > 0
    tv = res["per_class"]["kurgany_tselye"]
    assert tv["gain"]["class_confusion"] > 0


def test_error_types_cli_writes_summary_and_qgis_layer(tmp_path):
    import subprocess
    import sys

    pred, gt = scene()
    (tmp_path / "pred.geojson").write_text(json.dumps(pred), encoding="utf-8")
    (tmp_path / "gt.geojson").write_text(json.dumps(gt), encoding="utf-8")
    run = subprocess.run([sys.executable, str(ROOT / "scripts" / "error_types.py"), "--pred", str(tmp_path / "pred.geojson"),
                          "--gt", str(tmp_path / "gt.geojson"), "--out", str(tmp_path / "e.json"),
                          "--geojson", str(tmp_path / "e.geojson")], capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr
    summary = json.loads((tmp_path / "e.json").read_text(encoding="utf-8"))
    assert summary["by_region"]["001_Тест"]["FP:kurgany_povrezhdennye:background"] == 1
    layer = json.loads((tmp_path / "e.geojson").read_text(encoding="utf-8"))
    assert len(layer["features"]) == 4 + 3
    assert {f["properties"]["kind"] for f in layer["features"]} == {"FP", "FN"}
