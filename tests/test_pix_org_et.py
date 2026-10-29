"""GEO-ORG3: порт EtGr-бейзлайна организаторов 02 (expds/pix_org_et.py)."""
import numpy as np
import pytest
from affine import Affine
from shapely.geometry import box

from expds import pix_org_et as E


def _blob_map(kind="map"):
    H = W = 120
    yy, xx = np.mgrid[:H, :W]
    z = np.exp(-((yy - 40) ** 2 + (xx - 40) ** 2) / 60.0) * 4 - np.exp(-((yy - 85) ** 2 + (xx - 80) ** 2) / 80.0) * 4
    z += np.random.default_rng(0).normal(0, 0.05, z.shape)
    return {"path": E.Path("08_x_EtGr_карта_01.tif"), "kind": kind, "tr": Affine(0.1, 0, 0, 0, -0.1, 12),
            "crs": "EPSG:32636", "z": z.astype(np.float32), "mask": np.ones((H, W), bool)}


def test_kind_and_boundary():
    assert E.kind_of(E.Path("08_Гнёздово_Селище_2_EtGr_профиль_03.tif")) == "prof"
    assert E.kind_of(E.Path("08_Гнёздово_Селище_2_EtGr_карта_01_0.25.tif")) == "map"
    b = [E.Path("extra/Гнёздово_Селище_2_EtGr_карта_01_0.25_границы.geojson")]
    assert E.boundary_for(E.Path("08_Гнёздово_Селище_2_EtGr_карта_01_0.25.tif"), b) == b[0]
    assert E.boundary_for(E.Path("08_Гнёздово_Селище_2_EtGr_карта_02_0.5.tif"), b) is None


def test_candidates_both_signs_and_features():
    d = _blob_map()
    polys, X = E.candidates(d)
    assert X.shape == (len(polys), E.N_BASE) and len(polys) > 0
    assert {1.0, -1.0} <= set(X[:, 0].tolist())
    assert set(X[:, 1].tolist()) <= {0.0, 20.0, 10.0}
    assert np.all((X[:, 9] > 0) & (X[:, 9] <= 1))
    kids = E.nested_children(polys, X)
    assert any(kids)


def test_label_and_platform_counts():
    g = [box(0, 0, 1, 1)]
    p = [box(0.1, 0.1, 0.9, 0.9), box(5, 5, 6, 6), box(0, 0, 10, 10)]
    y = E.label(p, g)
    assert y.tolist() == [True, False, False]   # третий: IoU 0.01 < 0.1
    assert E.platform_counts(p[:2], [0.9, 0.8], g) == (1, 1, 0)
    assert E.f1_platform(1, 1, 0) == pytest.approx(2 / 2.5)


def test_nms_and_split():
    polys = [box(0, 0, 4, 2), box(0, 0, 1.9, 2), box(2.1, 0, 4, 2), box(10, 10, 11, 11)]
    s = np.array([0.9, 0.4, 0.35, 0.2])
    assert E.nms(polys, s, 0.3) == [0]
    c = {"polys": polys, "kind": "map", "kids": [[1, 2], [], [], []]}
    assert [i for i, _ in E.predict_polys(c, s, 0.5)] == [1, 2]      # 0.4, 0.35 ≥ 0.25
    c["kind"] = "prof"
    assert [i for i, _ in E.predict_polys(c, s, 0.5)] == [0]


def test_fit_predict_and_threshold():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, E.N_BASE)).astype(np.float32)
    y = X[:, 5] > 0.8
    m = E.fit(X, y, 0, 1)
    p = E.predict(m, X)
    assert p[y].mean() > p[~y].mean() + 0.3
    assert E.fit(X, np.zeros(400, bool)) is None
    assert E.predict(None, X).sum() == 0


def test_flag_default_off():
    import tomllib
    from pathlib import Path
    cfg = tomllib.loads((Path(__file__).resolve().parents[1] / "configs" / "default.toml").read_text())
    assert cfg["geophys"]["pixgbm"]["org_etgr"] is False
