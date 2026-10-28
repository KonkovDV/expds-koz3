"""Synthetic checks of expds.li_decode (point decoding of lidar-net maps)."""
import numpy as np
from shapely.geometry import Point

from expds import li_decode as L

try:
    from affine import Affine
except ImportError:                      # minimal stand-in for boxes without rasterio/affine
    from collections import namedtuple

    Affine = namedtuple("Affine", "a b c d e f")

RES = 0.5
TR = Affine(RES, 0, 1000.0, 0, -RES, 2000.0)


def blob(h, w, r, c, sigma, amp):
    yy, xx = np.mgrid[0:h, 0:w]
    return amp * np.exp(-((yy - r) ** 2 + (xx - c) ** 2) / (2 * sigma ** 2))


def test_find_peaks_nms_keeps_strongest():
    s = np.maximum.reduce([blob(60, 60, 20, 20, 1, 0.9), blob(60, 60, 20, 23, 1, 0.6), blob(60, 60, 45, 45, 1, 0.5)])
    r, c, h = L.find_peaks(s, 0.1, 2.5 / RES)       # 5 px NMS: the 3-px neighbour is suppressed
    assert list(zip(r, c)) == [(20, 20), (45, 45)]
    assert h[0] > h[1]
    r2, _, _ = L.find_peaks(s, 0.1, 1.0)            # no NMS beyond the 3x3 local-max test
    assert len(r2) >= 2


def test_kurgan_class_and_dual():
    p = np.zeros((2, 60, 60), np.float32)
    p[0] += blob(60, 60, 15, 15, 2, 0.8)            # povr-dominated mound
    p[1] += blob(60, 60, 15, 15, 2, 0.1)
    p[0] += blob(60, 60, 45, 45, 2, 0.45)           # ambiguous mound: both classes
    p[1] += blob(60, 60, 45, 45, 2, 0.40)
    out = L.decode_kurgans(p, list(L.KURGANS), RES, L.params(None))
    at = {}
    for r, c, cls, conf in out:
        at.setdefault((r, c), {})[cls] = conf
    assert set(at[(15, 15)]) == {"kurgany_povrezhdennye"}
    assert set(at[(45, 45)]) == set(L.KURGANS)
    assert at[(45, 45)]["kurgany_povrezhdennye"] > at[(45, 45)]["kurgany_tselye"]


def test_components_area_floor_and_interior_point():
    p = np.zeros((200, 200), np.float32)
    yy, xx = np.mgrid[0:200, 0:200]
    ring = (np.hypot(yy - 100, xx - 100) < 40) & (np.hypot(yy - 100, xx - 100) > 25)   # gor-like ring
    p[ring] = 0.8
    p[5:8, 5:8] = 0.9                                  # 9 px = 2.25 m^2: below the gor floor
    out = L.decode_components(p, RES, 0.5, 150.0)
    assert len(out) == 1
    r, c, conf = out[0]
    assert ring[r, c] and abs(conf - 0.8) < 1e-6      # the point is inside the component, not in the hole


def test_decode_li_features_and_radius():
    classes = ["kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha", "fortifikatsii"]
    p = np.zeros((4, 80, 80), np.float32)
    p[0] = blob(80, 80, 20, 20, 2, 0.7)
    p[2, 40:75, 40:75] = 0.9                           # 35x35 px = 306 m^2 >= 150
    p[3, 2:4, 70:72] = 0.9                             # 4 px = 1 m^2 < 7.5: dropped
    f = L.decode_li(p, classes, TR, None, RES, "r", "li", "x.las",
                    {"decode": "peaks", "peaks": {"radius_m": {"gorodishcha": 3.0}}})
    by = {}
    for x in f:
        by.setdefault(x["class_name"], []).append(x)
    assert len(by["kurgany_povrezhdennye"]) == 1 and "fortifikatsii" not in by
    k = by["kurgany_povrezhdennye"][0]["geom"]
    assert k.contains(Point(TR.c + 20.5 * RES, TR.f - 20.5 * RES))
    assert abs(np.sqrt(k.area / np.pi) - 1.0) < 0.05
    g = by["gorodishcha"][0]["geom"]
    assert abs(np.sqrt(g.area / np.pi) - 3.0) < 0.1
    assert L.enabled({"decode": "peaks"}) and not L.enabled({}) and not L.enabled({"decode": "legacy"})


def test_dump_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("EXPDS_LI_DUMP", str(tmp_path))
    p = np.random.default_rng(0).random((2, 10, 12)).astype(np.float32)
    path = L.dump_probs(p, list(L.KURGANS), TR, "EPSG:32637", "001_X", "a b.las", {"x": [1.0], "y": [2.0]}, "mask")
    z = np.load(path)
    assert z["probs"].shape == (2, 10, 12) and z["probs"].dtype == np.float16
    assert list(z["classes"]) == list(L.KURGANS) and str(z["crs"]) == "EPSG:32637"
    assert tuple(z["transform"]) == tuple(TR)[:6] and z["cand_x"][0] == 1.0
    monkeypatch.delenv("EXPDS_LI_DUMP")
    assert L.dump_probs(p, list(L.KURGANS), TR, None, "r", "s", None) is None
