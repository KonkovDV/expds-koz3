"""Потолок геометрии выпуска: круг попадает в компактный объект и не попадает в узкий вал."""
import importlib.util

from shapely.geometry import box

from conftest import ROOT, UTM, X0, Y0
from expds.crs import WEB_MERCATOR, to_crs

METRIC = str(ROOT / "materials" / "compute_metrics.py")


def load_script():
    spec = importlib.util.spec_from_file_location("oracle_ceiling", ROOT / "scripts" / "oracle_ceiling.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def feat(geom, cls, markup="sp_or"):
    g = to_crs(geom, UTM, WEB_MERCATOR)
    return {"type": "Feature", "geometry": g.__geo_interface__,
            "properties": {"class_name": cls, "region_name": "101_Альфа_1", "markup_type": markup}}


def test_compact_circle_hits_and_a_thin_rampart_does_not():
    oc = load_script()
    metric = oc.load_metric(METRIC)
    compact = feat(box(X0, Y0, X0 + 80, Y0 + 80), "kurgany_povrezhdennye", "sp_or")
    thin = feat(box(X0, Y0 + 200, X0 + 200, Y0 + 201), "fortifikatsii", "ae")
    gt = {"type": "FeatureCollection", "features": [compact, thin]}

    shipped = oc.evaluate(gt, metric, False)
    assert shipped["per_class"]["kurgany_povrezhdennye"]["recall"] == 1.0
    assert shipped["per_class"]["fortifikatsii"]["recall"] == 1.0

    circles = oc.evaluate(gt, metric, True)
    assert circles["per_class"]["kurgany_povrezhdennye"]["recall"] == 1.0
    assert circles["per_class"]["fortifikatsii"]["recall"] == 0.0


def test_same_layer_circles_are_not_merged():
    """Два кургана одного снимка остаются двумя объектами: склейка идёт между снимками."""
    oc = load_script()
    gt = {"type": "FeatureCollection", "features": [
        feat(box(X0, Y0, X0 + 6, Y0 + 6), "kurgany_tselye", "li"),
        feat(box(X0 + 3, Y0, X0 + 9, Y0 + 6), "kurgany_tselye", "li"),
    ]}
    out = oc.as_submission(oc.emit(gt))
    assert len(out) == 2
    assert all(f["confidence"] == 1.0 for f in out)
