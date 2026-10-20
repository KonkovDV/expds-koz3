import numpy as np
import pytest

from conftest import MOUNDS, UTM, X0, Y0, write_las
from expds import accel, candidates
from expds.lidar import build_dem, hillshade_set


def test_las_grid_backends_agree(tmp_path):
    las = write_las(tmp_path / "a.las")
    a, ia = accel.las_grid(las, 0.25)
    with_np = accel._rs
    accel._rs = None
    try:
        b, ib = accel.las_grid(las, 0.25)
    finally:
        accel._rs = with_np
    assert a.shape == b.shape
    assert np.allclose(np.nan_to_num(a, nan=-1), np.nan_to_num(b, nan=-1), atol=1e-4)
    assert ia["points_used"] == ib["points_used"] and ia["left"] == pytest.approx(ib["left"])


def test_ground_fallback_to_all_points(tmp_path):
    las = write_las(tmp_path / "b.las", ground_class=1)   # нет класса 2
    dem, info = accel.las_grid(las, 0.5)
    assert info["used_all"] and np.isfinite(dem).any()


def test_dem_and_hillshades(tmp_path):
    las = write_las(tmp_path / "c.las")
    d = build_dem(las, res=0.1)
    assert d.crs == UTM
    assert d.info["holes"] > 0 and np.isfinite(d.dem).mean() > 0.95
    imgs = hillshade_set(d)
    assert set(imgs) == {"c", "ch", "g", "i"}
    assert imgs["g"].dtype == np.uint8 and imgs["c"].shape[0] == 3
    # i = 255 − g внутри съёмки
    v = np.isfinite(d.dem)
    assert np.array_equal(imgs["i"][v].astype(int), 255 - imgs["g"][v].astype(int))


def _train_root(root):
    import json

    from shapely.geometry import Point, mapping

    from expds.crs import WEB_MERCATOR, to_crs

    for i, name in enumerate(("001_Альфа", "002_Бета", "003_Гамма")):
        site = root / name
        write_las(site / f"01_{name[4:]}_La_точки" / f"01_{name[4:]}_La_точки.las", n=150_000, seed=i)
        (site / "UTM.json").write_text(json.dumps({"utm": "36N"}), encoding="utf-8")
        ann = site / f"06_{name[4:]}_разметка"
        ann.mkdir(parents=True)
        for cls, mounds in (("курганы_поврежденные", MOUNDS[:1]), ("курганы_целые", MOUNDS[1:])):
            fc = {"type": "FeatureCollection", "features": [
                {"type": "Feature", "properties": {},
                 "geometry": mapping(to_crs(Point(X0 + mx, Y0 + my).buffer(r / 2), UTM, WEB_MERCATOR))}
                for mx, my, _, r in mounds]}
            (ann / f"{name[4:]}_Li_{cls}.geojson").write_text(json.dumps(fc), encoding="utf-8")
    return root


def test_oof_by_family_and_class_scores(tmp_path):
    import json

    from expds import gt as gtmod
    from expds.catalog import scan_root

    root = _train_root(tmp_path / "train")
    items, _ = gtmod.build_gt(scan_root(root))
    gt_path = tmp_path / "gt.geojson"
    gt_path.write_text(json.dumps(gtmod.to_feature_collection(items)), encoding="utf-8")
    p = candidates.CandidateParams()
    recs = candidates.collect(root, gt_path, p)
    assert len(recs) == 3 and len({r["family"] for r in recs}) == 3
    for r in recs:
        assert set(np.unique(r["y"])) == {0, 1, 2}           # фон, повреждённый, целый
    fold = candidates.family_folds([r["family"] for r in recs], [int((r["y"] > 0).sum()) for r in recs], 3)
    assert sorted(fold.values()) == [0, 1, 2]
    feats = candidates.oof_predictions(recs, p, folds=3, multiclass=True, rounds=20)
    by_cls = {}
    for f in feats:
        by_cls.setdefault(f["class_name"], []).append(f["confidence"])
    assert set(by_cls) == {"kurgany_povrezhdennye", "kurgany_tselye"}
    assert len(by_cls["kurgany_povrezhdennye"]) == sum(len(r["y"]) for r in recs)
    assert not np.allclose(by_cls["kurgany_povrezhdennye"], by_cls["kurgany_tselye"])


def test_dem_params_follow_model_then_config():
    from expds.lidar import LEGACY_DEM, MODEL_DEM, dem_params

    assert dem_params(None, {}) == MODEL_DEM
    assert dem_params(None, {"lidar": {"dem_res": 0.25, "dem_fill": "delaunay"}}) == {"res": 0.25, "fill": "delaunay"}
    assert dem_params({"dem_res": None}) == LEGACY_DEM                      # модель обучена до MODEL_DEM
    assert dem_params({"dem_res": 0.5, "dem_fill": "pyramid"}, {"lidar": {"dem_res": 0.1}}) == {"res": 0.5,
                                                                                               "fill": "pyramid"}


def test_pyramid_dem_fills_holes_inside_survey(tmp_path):
    las = write_las(tmp_path / "e.las")
    d = build_dem(las, res=0.5, fill="pyramid")
    assert np.isfinite(d.dem).mean() > 0.95 and d.res == 0.5


def test_candidates_find_mounds(tmp_path):
    las = write_las(tmp_path / "d.las", n=200_000)
    d = build_dem(las, res=0.1)
    c = candidates.detect(d)
    assert len(c["x"]) >= len(MOUNDS)
    s = candidates.score(c["features"])
    top = np.argsort(-s)[: len(MOUNDS)]
    found = 0
    for mx, my, _, _ in MOUNDS:
        dist = np.hypot(c["x"][top] - (X0 + mx), c["y"][top] - (Y0 + my))
        found += dist.min() < 1.5
    assert found == len(MOUNDS)
