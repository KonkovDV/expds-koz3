"""expds.w5cf_infer: plug-in contract of expds.w5_cloudfeat on tiny synthetic artifacts (no train data, no LAS)."""
import json

import numpy as np
import pytest

lgb = pytest.importorskip("lightgbm")
shapely = pytest.importorskip("shapely")
pytest.importorskip("scipy")

from expds import w5cf_infer as W  # noqa: E402
from expds.cloudfeat import NAMES as CF  # noqa: E402

CL = ["kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha", "arkhitektury", "object_poly", "fortifikatsii", "finds_points"]
NAMES = (["conf", "logit", "cls", "grp", "mk", "area", "compact", "bw", "aspect", "nv", "has_twin", "twin_conf", "twin_diff",
          "has_las", "rg_n", "rg_rank", "rg_pct", "rg_max", "conf_rel", "rg_mean", "gq", "nb_same10", "nb_same30", "nb_same100",
          "nb_strong30", "nb_strong100", "og_max5", "og_n5", "nn_dist", "nb_otherfam50", "d_nhigher_g", "d_maxq_other_g",
          "d_localmax", "d_maxq_othersub", "d_n8", "cnx_s0", "cnx_s1", "cnx_mean", "cnx_agree", "cnx_samecls"] + list(CF))


def _art(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.random((400, len(NAMES))); y = (X[:, 0] + 0.3 * rng.random(400) > 0.7).astype(float)
    m = lgb.train(dict(objective="binary", verbose=-1, num_threads=1, min_data_in_leaf=5), lgb.Dataset(X, y), 5)
    m.save_model(str(tmp_path / "fam0.txt"))
    meta = dict(variant="test", names=NAMES, CL=CL, grp_codes=["li", "or"], mk_codes=["li", "or"],
                pool_thr={"li|0": 0.5, "li|1": None, "or|0": None}, models={"0": {"file": "fam0.txt"}})
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    np.savez(tmp_path / "gq_ref.npz", keys=np.array(["li|0", "li|1"]), off=np.array([0, 3, 5]),
             val=np.array([0.1, 0.5, 0.9, 0.2, 0.8]), cnt=np.array([1, 2, 1, 1, 1]))
    return {"w5cf": {"infer_dir": str(tmp_path)}}


def _obj(x, cls, conf, mk="li", reg="R1", point=False):
    g = shapely.Point(x, 7_500_000.0) if point else shapely.box(x, 7_500_000.0, x + 10, 7_500_010.0)
    return dict(geom=g, class_name=cls, confidence=conf, markup_type=mk, threshold_group=None, region_name=reg)


def test_score_contract(tmp_path):
    cfg = _art(tmp_path); W.reset()
    feats = [_obj(4_200_000.0, CL[0], 0.9), _obj(4_200_000.0, CL[1], 0.95), _obj(4_200_500.0, CL[0], 0.2),
             _obj(4_201_000.0, CL[2], 0.99), _obj(4_200_100.0, "unknown", 0.9), dict(class_name=CL[0], confidence=0.9)]
    s = W.score(feats, cfg)
    assert len(s) == len(feats)
    assert s[0] is not None and 0 <= s[0] <= 1 and s[1] is not None        # pool: li|0 conf ≥ 0.5, li|1 whole group
    assert s[2] is None                                                      # below pool threshold
    assert s[3] is None and s[4] is None and s[5] is None                    # no model / unknown class / no geometry
    tb = W.table(feats, W.load_artifacts(cfg)); X = W.features(tb, W.load_artifacts(cfg))
    assert X.shape == (4, len(NAMES))
    assert np.isnan(X[:, [NAMES.index(n) for n in NAMES if n.startswith(("cnx_", "cf_"))]]).all()   # no LAS / no cnx
    assert (X[:, NAMES.index("has_las")] == 0).all()
    assert X[0, NAMES.index("has_twin")] == 1 and X[1, NAMES.index("has_twin")] == 1                 # P/T within 0.5 m
    assert X[0, NAMES.index("gq")] == pytest.approx(1.0)        # train li|0 conf [0.1, 0.5×2, 0.9]: 0.9 → position 3 of 0..3
    assert X[2, NAMES.index("gq")] == pytest.approx(0.5 / 3)    # unseen 0.2: between 0.1 (pos 0) and 0.5 (pos 1–2)
    assert s == W.score(feats, cfg)                                                                  # deterministic


def test_region_without_las(tmp_path):
    cfg = _art(tmp_path); W.reset()
    feats = [_obj(4_200_000.0, CL[0], 0.9)]
    class R:
        name, las, utm_json = "R1", [], None
    assert W.region_features(R(), feats, cfg) is None
    assert W.KEY not in feats[0] and "R1" not in W._STATE["las_regions"]
    assert W.score(feats, cfg)[0] is not None
