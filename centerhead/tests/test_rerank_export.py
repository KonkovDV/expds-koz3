import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Point, mapping, shape

from centerhead import cfg_with
from centerhead.export import candidates_to_feats, to_fc
from centerhead.features import build_features
from centerhead.rerank import OOFReranker, check_no_leak, circle, folds_to_map, match_labels

REPO = Path(__file__).resolve().parents[2]
CM_DIR = Path("/workspace/koz3_shared/bot-10/repo/materials")
UTM = "EPSG:32637"
LEFT, TOP = 500_000.0, 6_200_000.0


def _gt():
    g = [Point(LEFT + 10, TOP - 10).buffer(4), Point(LEFT + 30, TOP - 10).buffer(3), Point(LEFT + 50, TOP - 10)]
    return g, np.array([0, 1, 0])


def _cand(X, Y, cls, s):
    n = len(X)
    return {"X": np.asarray(X, float), "Y": np.asarray(Y, float), "cls": np.asarray(cls), "score": np.asarray(s, float),
            "radius_m": np.full(n, 3.0), "row": np.zeros(n, int), "col": np.zeros(n, int)}


def test_match_labels_official():
    g, gc = _gt()
    t = _cand([LEFT + 10.5, LEFT + 10, LEFT + 30, LEFT + 30, LEFT + 51.5, LEFT + 53],
              [TOP - 10, TOP - 10, TOP - 10, TOP - 10, TOP - 10, TOP - 10],
              [0, 0, 0, 1, 0, 0], [0.9, 0.8, 0.95, 0.5, 0.7, 0.6])
    y, mg = match_labels(t, g, gc, 0.5)
    assert y.tolist() == [1, 0, 0, 1, 1, 0]          # дубликат; чужой класс; точка ≤ 2 м; 3 м > 2 м
    assert mg.tolist() == [0, -1, -1, 1, 2, -1]
    y2, _ = match_labels(t, g, gc, 0.5, mode="any")
    assert y2.tolist() == [1, 1, 0, 1, 1, 0]


@pytest.mark.skipif(not (CM_DIR / "compute_metrics.py").exists(), reason="нет официального compute_metrics.py")
def test_match_labels_agrees_with_official_metric():
    sys.path.insert(0, str(CM_DIR))
    import compute_metrics as CM
    rng = np.random.default_rng(0)
    cen = [(LEFT + 20 * i, TOP - 20 * j) for i in range(6) for j in range(5)]
    g = [Point(x, y).buffer(rng.uniform(1.5, 6)) for x, y in cen]
    gc = rng.integers(2, size=len(g))
    X = np.array([x for x, _ in cen] * 2) + rng.normal(0, 2, 60)
    Y = np.array([y for _, y in cen] * 2) + rng.normal(0, 2, 60)
    t = _cand(X, Y, rng.integers(2, size=60), rng.uniform(0, 1, 60))
    y, _ = match_labels(t, g, gc, 0.5)
    from centerhead.config import HM_CLASSES
    from pyproj import Transformer
    tr = Transformer.from_crs(UTM, "EPSG:3857", always_xy=True)
    from shapely.ops import transform
    gt_fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": mapping(transform(tr.transform, gg)),
         "properties": {"class_name": HM_CLASSES[c], "markup_type": "li"}} for gg, c in zip(g, gc)]}
    pred_fc = to_fc(candidates_to_feats(t, "r", UTM, cfg=cfg_with(circle_r_m=0.5)))
    res = CM.F1Calculator().calculate_f1(pred_fc, gt_fc)
    tp = sum(int(v.get("tp", 0)) for k, v in res["per_class"].items() if k in HM_CLASSES)
    assert tp == int(y.sum()), (tp, int(y.sum()))


def test_reranker_oof_no_leak():
    rng = np.random.default_rng(0)
    regions = np.array([f"fam{i // 2}_r{i}" for i in range(8)]).repeat(150)
    X = rng.normal(size=(len(regions), 5)).astype(np.float32)
    y = (X[:, 0] + 0.3 * rng.normal(size=len(X)) > 0.5).astype(int)
    fam = lambda r: r.split("_")[0]                                   # noqa: E731
    from collections import defaultdict
    d = defaultdict(set)
    for r in set(regions):
        d[fam(r)].add(r)
    folds = [d[f] for f in sorted(d)][:2] + [d["fam2"] | d["fam3"]]
    rr = OOFReranker(num_rounds=50, feature_names=[f"f{i}" for i in range(5)])
    oof = rr.fit_oof(X, y, regions, folds, family_of=fam)
    assert np.isfinite(oof).all() and len(rr.fold_models) == 3
    pos, neg = oof[y == 1], oof[y == 0]
    assert pos.mean() > neg.mean() + 0.3
    # модель фолда k не обучалась на фолде k: её прогноз на фолде k == oof
    k0 = folds_to_map(folds)[regions[0]]
    te = np.array([folds_to_map(folds)[r] == k0 for r in regions])
    np.testing.assert_allclose(rr.predict(X[te], fold=k0), oof[te])
    rr.fit_full(X, y)
    p = rr.predict(X[:10])
    assert p.shape == (10,) and ((p >= 0) & (p <= 1)).all()
    bad = [{"fam0_r0", "fam0_r1"}, {"fam1_r2"}, {"fam1_r3", "fam2_r4", "fam2_r5", "fam3_r6", "fam3_r7"}]
    with pytest.raises(ValueError):
        OOFReranker(num_rounds=5).fit_oof(X, y, regions, bad, family_of=fam)
    with pytest.raises(ValueError):
        check_no_leak({"a": 0, "b": 1}, cnn_fold_of={"a": 0, "b": 0})
    with pytest.raises(ValueError):
        check_no_leak({"a": 0}, cnn_fold_of={})


def test_reranker_save_load(tmp_path):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 3))
    y = (X[:, 0] > 0).astype(int)
    rr = OOFReranker(num_rounds=20).fit_full(X, y)
    rr.save(tmp_path / "m.json")
    np.testing.assert_allclose(OOFReranker.load(tmp_path / "m.json").predict(X), rr.predict(X))


def test_family_folds_input():
    sys.path.insert(0, str(REPO))
    metric = pytest.importorskip("expds.metric")
    regs = [f"f{i // 3}_{i}" for i in range(9)]
    folds = metric.family_folds(regs, lambda r: r.split("_")[0], k=2)
    m = folds_to_map(folds)
    check_no_leak(m, lambda r: r.split("_")[0])


def test_features_shapes_and_gor():
    rng = np.random.default_rng(0)
    H = W = 64
    hm = rng.uniform(0, 1, (2, H, W)).astype(np.float32)
    mask = rng.uniform(0, 1, (3, H, W)).astype(np.float32)
    rel = rng.integers(0, 256, (12, H, W)).astype(np.uint8)
    t = _cand([LEFT + 5, LEFT + 30, LEFT + 31.5], [TOP - 5, TOP - 30, TOP - 30], [0, 1, 0], [0.9, 0.5, 0.4])
    t["row"], t["col"] = np.array([10, 60, 63]), np.array([10, 60, 63])
    gor = [Point(LEFT + 30, TOP - 30).buffer(10)]
    F, names = build_features(t, hm, mask, rel, gor_polys=gor)
    # раздельные сетки: hm на шаге 2 м, рельеф/маски на 0,5 м — те же пиксели, если координаты согласованы
    t2 = dict(t, X=LEFT + (t["col"] + 0.5) * 0.5, Y=TOP - (t["row"] + 0.5) * 0.5)
    Fg, _ = build_features(t2, hm[:, ::4, ::4], mask, rel, gor_polys=None, hm_grid=(LEFT, TOP, 2.0),
                           full_grid=(LEFT, TOP, 0.5))
    j = names.index("rel_lrm_s_ctr")
    np.testing.assert_array_equal(Fg[:, j], F[:, j])
    assert Fg[0, names.index("hm_other_ctr")] == hm[1, 8, 8]
    F3, n3 = build_features(t, hm, mask, rel[:3])
    assert "rel_slope_ctr" in n3 and "rel_svf_r_ctr" not in n3
    assert F.shape == (3, len(names)) and F.dtype == np.float32 and len(set(names)) == len(names)
    i = names.index
    assert F[0, i("in_gor")] == 0 and F[0, i("gor_sdist_m")] > 0
    assert F[1, i("in_gor")] == 1 and F[1, i("gor_sdist_m")] == pytest.approx(-10, abs=0.1)
    assert F[1, i("nb_cnt_10")] == 1 and F[1, i("nn_dist")] == pytest.approx(1.5)
    assert F[0, i("score")] == pytest.approx(0.9) and F[1, i("hm_other_ctr")] == pytest.approx(hm[0, 60, 60])
    assert F[0, i("rel_lrm_s_ctr")] == (rel[0, 10, 10] if rel[0, 10, 10] else F[0, i("rel_lrm_s_ctr")])
    F0, n0 = build_features(_cand([], [], [], []), hm, mask, rel, gor_polys=gor)
    assert F0.shape == (0, len(n0)) and n0 == names


def test_export_writer_conventions():
    sys.path.insert(0, str(REPO))
    writer = pytest.importorskip("expds.writer")
    t = _cand([LEFT + 5, LEFT + 30], [TOP - 5, TOP - 30], [0, 1], [1.2, 0.123456789])
    feats = candidates_to_feats(t, "reg_A", UTM)
    fc = to_fc(feats)
    assert fc == writer.to_feature_collection(feats)
    rep = writer.validate(fc)
    assert rep["ok"], rep
    p = fc["features"][0]["properties"]
    assert p["class_name"] == "kurgany_tselye" and p["confidence"] == 1.0 and p["crs"] == "EPSG:3857"
    assert p["markup_type"] == "li" and p["original_crs"] == UTM and p["region_name"] == "reg_A"
    assert fc["features"][1]["properties"]["confidence"] == 0.12346
    g = shape(fc["features"][0]["geometry"])
    assert g.geom_type == "Polygon" and len(g.exterior.coords) == 33
    from pyproj import Transformer
    from shapely.ops import transform
    back = transform(Transformer.from_crs("EPSG:3857", UTM, always_xy=True).transform, g)
    assert back.centroid.distance(Point(LEFT + 5, TOP - 5)) < 0.02
    assert math.sqrt(back.area / math.pi) == pytest.approx(0.5, rel=0.02)   # истинный радиус в родной СК
    json.dumps(fc)
