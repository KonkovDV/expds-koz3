"""li_half: второй проход сети li на 1 м — синтетика, без весов."""
import numpy as np
import pytest
from shapely.geometry import Point

from expds import candidates, li_half
from expds.crs import WEB_MERCATOR, to_crs
from expds.lidar import Dem

UTM = "EPSG:32637"
X0, Y0 = 500_000.0, 6_000_000.0


def _dem(res=0.1, size_m=200.0, mounds=((60.0, 60.0, 9.0, 0.8), (140.0, 140.0, 2.0, 0.3))):
    n = int(size_m / res)
    ax = (np.arange(n) + 0.5) * res
    xx, yy = np.meshgrid(ax, ax)                     # строки — сверху вниз от top
    z = np.zeros((n, n), np.float32)
    for mx, my, r, h in mounds:                      # my — от верхнего края
        d2 = (xx - mx) ** 2 + (yy - my) ** 2
        z += (h * np.clip(1 - d2 / r ** 2, 0, None)).astype(np.float32)
    return Dem(dem=z, left=X0, top=Y0 + size_m, res=res, crs=UTM, info={"factor": 1})


class FakeSeg:
    classes = ["kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha"]
    meta = {"li_source": "las", "li_tags": ["lrm_s", "lrm_l", "slope"]}
    keep = None

    def __init__(self):
        self.tta = 8
        self.seen_tta = None
        self.shapes = []

    def predict(self, img):
        self.seen_tta = self.tta
        self.shapes.append(img.shape)
        a = img[0].astype(np.float32) / 255.0                 # lrm_s как «вероятность»
        return np.stack([a, a * 0.5, np.zeros_like(a)])


def test_defaults_off_and_config_merge():
    assert li_half.enabled({}) is False
    hp = li_half.params({"li_half": {"enabled": True, "cand": {"relief_radius_m": 25.0}}})
    assert hp["enabled"] and hp["res"] == 1.0 and hp["tta"] == 1 and hp["group"] == "li_half"
    assert hp["cand"]["relief_radius_m"] == 25.0 and hp["cand"]["work_res"] == 1.0
    cp = li_half.cand_params(candidates.CandidateParams(circle_m=1.5), hp)
    assert cp.work_res == 1.0 and cp.relief_radius_m == 25.0 and cp.circle_m == 2.0


def test_default_toml_keeps_li_half_off():
    from conftest import ROOT
    from expds.pipeline import load_config

    cfg = load_config(ROOT / "configs" / "default.toml")
    assert li_half.enabled(cfg) is False
    assert "li_half" not in cfg.get("thresholds_by_markup", {})


def test_half_pass_grid_tta_group_and_dump(tmp_path):
    d = _dem()
    seg = FakeSeg()
    hp = li_half.params({})
    feats = li_half.half_pass(seg, d, "r1", "a.laz", candidates.CandidateParams(), hp, tmp_path)
    assert seg.seen_tta == 1 and seg.tta == 8                # tta прохода 1, потом возвращён
    assert seg.shapes[0][1:] == (200, 200)                   # вход сети на 1 м, не 0,5
    assert feats and {f["threshold_group"] for f in feats} == {"li_half"}
    assert {f["class_name"] for f in feats} == {"kurgany_povrezhdennye", "kurgany_tselye"}
    # крупная насыпь ⌀ 18 м найдена вторым детектором (центр в 1,5 м)
    c = to_crs(Point(X0 + 60, Y0 + 200 - 60), UTM, WEB_MERCATOR)
    near = [f for f in feats if f["geom"].centroid.distance(c) < 1.5 * 1.8]
    assert near
    files = list(tmp_path.glob("*.npz"))
    assert len(files) == 1
    z = np.load(files[0])
    assert z["probs"].dtype == np.float16 and z["probs"].shape == (2, 200, 200)
    assert z["scores"].shape == (len(z["x"]), 2) and str(z["region"]) == "r1"
    assert abs(float(z["transform"][0]) - 1.0) < 1e-9


def test_half_pass_max_px_guard():
    d = _dem()
    hp = li_half.params({"li_half": {"max_px": 100}})
    assert li_half.half_pass(FakeSeg(), d, "r1", "a.laz", None, hp) == []


def test_new_only_drops_near_kept_main_li():
    def circ(x, y, cls, conf, group=None):
        f = {"geom": to_crs(Point(X0 + x, Y0 + y).buffer(2), UTM, WEB_MERCATOR), "class_name": cls,
             "confidence": conf, "region_name": "r1", "markup_type": "li"}
        if group:
            f["threshold_group"] = group
        return f

    main = [circ(0, 0, "kurgany_tselye", 0.9),               # оставлен порогом li (0,19422)
            circ(100, 0, "kurgany_tselye", 0.01),            # ниже порога — не якорь
            circ(200, 0, "kurgany_tselye", 0.99, "li_dev")]  # чужая группа — не якорь
    main.append(circ(300, 0, "kurgany_povrezhdennye", 0.07))   # оставлен порогом li (0,06415), повреждённый
    half = [circ(5, 0, "kurgany_tselye", 0.5, "li_half"),   # 5 м от якоря — вон
            circ(1, 0, "kurgany_povrezhdennye", 0.5, "li_half"),  # другой класс в 1 м — тоже вон (поправка 29.10)
            circ(100, 5, "kurgany_tselye", 0.5, "li_half"),
            circ(200, 5, "kurgany_tselye", 0.5, "li_half"),
            circ(300, 4, "kurgany_tselye", 0.5, "li_half"),  # целый рядом с принятым повреждённым — вон
            circ(313, 0, "kurgany_povrezhdennye", 0.5, "li_half"),  # 13 м — остаётся
            circ(30, 0, "kurgany_tselye", 0.5, "li_half")]  # 30 м — остаётся
    out = li_half.new_only(half, main, li_half.params({}))
    kept = sorted((round(to_crs(f["geom"], WEB_MERCATOR, UTM).centroid.x - X0), f["class_name"]) for f in out)
    assert kept == [(30, "kurgany_tselye"), (100, "kurgany_tselye"), (200, "kurgany_tselye"),
                    (313, "kurgany_povrezhdennye")]
    other = dict(circ(1, 0, "kurgany_povrezhdennye", 0.5, "li_half"), region_name="r2")   # другой участок — остаётся
    assert li_half.new_only([other], main, li_half.params({})) == [other]
