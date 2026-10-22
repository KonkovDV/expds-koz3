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


def test_mosaic_matches_full_dem_averaged_to_half_metre(tmp_path):
    """Куски 0,1 м с полем дают тот же шаг 0,5 м, что и целая сетка, кроме узкого шва."""
    from expds.candidates import downsample_mean
    from expds.lidar import build_dem_mosaic

    las = write_las(tmp_path / "mosaic.las", n=80_000)
    full = build_dem(las, res=0.1, max_pixels=10**12)
    whole = downsample_mean(full.dem, 5)
    mosaic = build_dem_mosaic(las, tile_m=40.0, halo_m=20.0)
    h = min(whole.shape[0], mosaic.dem.shape[0])
    w = min(whole.shape[1], mosaic.dem.shape[1])
    a, b = whole[:h, :w], mosaic.dem[:h, :w]
    both = np.isfinite(a) & np.isfinite(b)
    assert both.mean() > 0.5
    close = np.abs(a - b) <= 0.05
    assert close[both].mean() > 0.99
    assert mosaic.info["factor"] == 1 and mosaic.res == pytest.approx(0.5)
    las = write_las(tmp_path / "b.las", ground_class=1)   # нет класса 2
    dem, info = accel.las_grid(las, 0.5)
    assert info["used_all"] and np.isfinite(dem).any()


def test_full_cloud_matches_keep_one_and_thinning_drops_points(tmp_path):
    las = write_las(tmp_path / "thin.las", n=20_000)
    full, info = accel.las_grid(las, 0.5)
    same, info_same = accel.las_grid(las, 0.5, keep_prob=1.0)
    thin, info_thin = accel.las_grid(las, 0.5, keep_prob=0.4)
    assert np.array_equal(np.nan_to_num(full, nan=-1), np.nan_to_num(same, nan=-1))
    assert info["points_used"] == info_same["points_used"]
    assert 0 < info_thin["points_used"] < info["points_used"]
    assert np.isfinite(thin).any()


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


def test_prob_peaks_fill_gaps_and_skip_nearby():
    c = {"x": np.array([10.0, 0.0]), "y": np.array([10.0, 0.0]),
         "features": np.ones((2, 8), np.float32)}
    prob = np.zeros((20, 20), np.float32)
    # from_origin(0, 20, 1): центр пикселя (col, row) = (col+0.5, 19.5-row)
    prob[19, 0] = 0.95        # (0.5, 0.5) — ближе 2 м к (0, 0), не добавлять
    prob[2, 16] = 0.8         # (16.5, 17.5) — далеко, добавить
    from rasterio.transform import from_origin
    tr = from_origin(0, 20, 1, 1)
    out = candidates.add_prob_peaks(c, prob, tr, min_prob=0.5, radius_m=1.0, dedupe_m=2.0)
    assert len(out["x"]) == 3
    assert np.isnan(out["features"][-1]).all()
    assert abs(out["x"][-1] - 16.5) < 1e-6 and abs(out["y"][-1] - 17.5) < 1e-6


def test_relief_on_coarse_dem_stays_on_model_step():
    from expds.lidar import Dem, relief_channels

    dem = np.zeros((30, 30), np.float32)
    dem[5:25, 5:25] = 10
    img, tr = relief_channels(Dem(dem=dem, left=0, top=9, res=0.3, crs="EPSG:32636"), 0.5)
    assert tr.a == pytest.approx(0.5) and abs(tr.e) == pytest.approx(0.5)
    assert img.shape[1] == int(np.floor(30 * 0.3 / 0.5))


def test_dem_prefetch_matches_direct_build(tmp_path):
    from expds.lidar import DemPrefetch

    a = write_las(tmp_path / "a.las", n=40_000, seed=1)
    b = write_las(tmp_path / "b.las", n=40_000, seed=2)
    pf = DemPrefetch([(a, UTM), (b, UTM)])
    try:
        da = pf.get(a, 0.1, "delaunay", UTM)
        db = pf.get(b, 0.1, "delaunay", UTM)
        again = pf.get(a, 0.1, "delaunay", UTM)
    finally:
        pf.close()
    for got, path in ((da, a), (db, b), (again, a)):
        ref = build_dem(path, res=0.1, fill="delaunay", crs=UTM)
        assert np.array_equal(got.dem, ref.dem, equal_nan=True)
        assert (got.left, got.top, got.res) == (ref.left, ref.top, ref.res)


def test_mask_to_polygons_matches_shape_by_shape():
    from rasterio.features import shapes
    from rasterio.transform import from_origin
    from shapely.geometry import shape

    from expds.crs import WEB_MERCATOR, to_crs
    from expds.raster import mask_to_polygons

    rng = np.random.default_rng(5)
    lab = np.zeros((120, 160), np.int32)
    for k in range(1, 40):
        r, c = rng.integers(5, 110), rng.integers(5, 150)
        lab[r:r + rng.integers(2, 9), c:c + rng.integers(2, 9)] = k
    lab[50:60, 50:60] = 99
    lab[54:56, 54:56] = 0
    tr = from_origin(X0, Y0 + 100, 0.5, 0.5)
    got = mask_to_polygons(lab, tr, UTM, simplify=0.25)
    ref = []
    for geom, val in shapes(lab, mask=lab > 0, transform=tr, connectivity=8):
        ref.append((to_crs(shape(geom).simplify(0.25, preserve_topology=True), UTM, WEB_MERCATOR), int(val)))
    assert [v for _, v in got] == [v for _, v in ref]
    assert all(a.equals_exact(b, tolerance=0) for (a, _), (b, _) in zip(got, ref))
    assert any(len(p.interiors) for p, _ in got)


def test_probs_to_features_floor_is_post_threshold():
    from rasterio.transform import from_origin

    from expds.infer_nn import probs_to_features
    from expds.post import apply_thresholds

    rng = np.random.default_rng(6)
    probs = np.zeros((2, 90, 90), np.float32)
    for k in range(2):
        for _ in range(25):
            r, c = rng.integers(3, 85, 2)
            probs[k, r - 2:r + 3, c - 2:c + 3] = rng.uniform(0.06, 1.0)
    tr = from_origin(X0, Y0 + 45, 0.5, 0.5)
    classes = ["gorodishcha", "fortifikatsii"]
    floors = {"gorodishcha": 0.7, "fortifikatsii": 0.4}
    full = probs_to_features(probs, classes, tr, UTM, 0.5, "r", "li", "s")
    cut = probs_to_features(probs, classes, tr, UTM, 0.5, "r", "li", "s", min_conf=floors)
    kept = apply_thresholds(full, floors)
    assert 0 < len(cut) < len(full)
    key = lambda f: (f["class_name"], f["confidence"], f["geom"].wkb)
    assert sorted(map(key, cut)) == sorted(map(key, kept))


def test_circles_match_point_buffer():
    from shapely.geometry import Point

    from expds.candidates import CandidateParams, to_features
    from expds.crs import WEB_MERCATOR

    c = {"x": np.array([500_000.0, 500_030.0]), "y": np.array([6_000_000.0, 6_000_010.0])}
    scores = np.array([[0.0, 0.4, 0.2], [0.0, 0.1, 0.9]], np.float32)
    feats = to_features(c, scores, "EPSG:32636", "r", CandidateParams(circle_m=2.0))
    assert len(feats) == 4
    assert feats[0]["geom"].equals(feats[1]["geom"])
    one = Point(c["x"][0], c["y"][0]).buffer(2.0, quad_segs=4)
    from expds.crs import to_crs
    expect = to_crs(one, "EPSG:32636", WEB_MERCATOR)
    assert feats[0]["geom"].equals_exact(expect, tolerance=1e-4)


def test_work_grid_matches_blocks_and_hits_exact_step():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(20, 20)).astype(np.float32)
    src[0, 0] = np.nan
    blocked, step = candidates.to_work_grid(src, 0.1, 0.5)
    assert step == pytest.approx(0.5)
    assert np.allclose(blocked, candidates.downsample_mean(src, 5), equal_nan=True)
    coarse = np.full((10, 10), 3.0, np.float32)
    coarse[0, :] = np.nan
    out, step = candidates.to_work_grid(coarse, 0.3, 0.5)
    assert step == pytest.approx(0.5)
    assert out.shape == (6, 6)
    assert np.nanmax(out) == pytest.approx(3.0)


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
