"""bot-7, кандидат v2: PIXGBM S и sota_stack6 — независимые флаги, порядок pix → stack6, пороги *_pix, модели."""
import hashlib
from pathlib import Path

import pytest
from shapely.geometry import Point

from expds import geophys, pipeline, watchdog
from expds.catalog import Region

ROOT = Path(__file__).resolve().parents[1]
# bot-3/pixgbm/pc/thr_pix_full.json (bot-9)
THR_PIX = {"maggr_pix": {"finds_points": 0.902915, "object_poly": 0.684725},
           "etgr_pix": {"object_poly": 0.545825}, "terfl_pix": {"object_poly": 0.746421}}
SHA = {"poly.txt": "d1fe3efafcd4377d41430d8828e881576c9024f9d7a5c89f55e23554e02ec800",
       "finds.txt": "5fa8c09d77d0914cc925fb6bb925a0ab1b64ccecf3c823d7f3546e71c4ceb04a"}


@pytest.fixture(scope="module")
def cfg():
    return pipeline.load_config(ROOT / "configs" / "default.toml")


def test_flags_independent_and_as_documented(cfg):
    pix = cfg["geophys"]["pixgbm"]
    assert pix["enabled"] is True and pix["replace"] is True          # вариант S (bot-10 PREREG/REPORT7)
    assert set(pix["classes"]) == {"finds_points", "object_poly"} and pix["union_name_re"] == ""
    assert float(pix["budget_s"]) == 600 and pix.get("backend", "lightgbm") == "lightgbm"   # REPORT4 §9; без .so
    assert int(pix["procs"]) == 2                                      # решение bot 1: пул ≤ 2 (ОЗУ)
    assert float(cfg["time"]["soft_ceiling_s"]) == 1200 and pipeline.soft_ceiling_from(cfg, environ={}) == 1200.0   # bot-8
    post = cfg["post"]
    assert post["multipole_object_poly"] is True and post["filter_finds_points"] is True   # stack6 sep
    assert post.get("kurgan_twins_only") is True and not post.get("kurgan_twins")   # twins-only: bot-10 work8/twins78_stats.log
    assert "pixgbm" not in post and "multipole" not in str(pix)        # флаги в разных таблицах


def test_pix_thresholds_present(cfg):
    for g, d in THR_PIX.items():
        assert cfg["thresholds_by_markup"][g] == d


def test_pix_models_shipped(cfg):
    """Real weight SHA check — skip on clean clones (poly/finds.txt not in git)."""
    md = ROOT / cfg["geophys"]["pixgbm"]["model_dir"]
    missing = [n for n in SHA if not (md / n).is_file()]
    if missing:
        import pytest
        pytest.skip(f"pixgbm weights absent (not in git): {missing}")
    for name, h in SHA.items():
        assert hashlib.sha256((md / name).read_bytes()).hexdigest() == h


def test_build_submission_collects_pix_models(monkeypatch):
    """Config paths must be collected; do not require gitignored weight bytes on disk."""
    import importlib.util
    from conftest import allow_missing_submission_models
    allow_missing_submission_models(monkeypatch)
    spec = importlib.util.spec_from_file_location("bs", ROOT / "scripts" / "build_submission.py")
    bs = importlib.util.module_from_spec(spec); spec.loader.exec_module(bs)
    rel = {str(p.relative_to(ROOT)) for p in bs.collect()}
    assert {"models/pixgbm/all/poly.txt", "models/pixgbm/all/finds.txt"} <= rel


Y0 = 7.4e6


def _f(cls, mt, x, conf, grp=None, r=1.0, src="a.tif"):
    g = Point(4.1e6 + x, Y0)
    return {"class_name": cls, "markup_type": mt, "threshold_group": grp, "confidence": conf, "region_name": "r",
            "source": src, "geom": g.buffer(r) if cls == "object_poly" else g.buffer(0.5)}


def _patch_classic(monkeypatch, classic):
    monkeypatch.setattr(geophys, "select_rasters", lambda files, mt, maps: [Path(f"{mt}.tif")] if files else [])
    monkeypatch.setattr(geophys, "params_for", lambda g, mt, c: ({}, {"rank": False, "use_noise": False, "min_confidence": 0.0}))
    monkeypatch.setattr(geophys, "survey_boundary", lambda *a, **k: None)
    monkeypatch.setattr(geophys, "load_mask", lambda *a, **k: None)
    monkeypatch.setattr(geophys, "sane_masks", lambda *a, **k: (None, None))
    monkeypatch.setattr(geophys, "apply_masks", lambda found, b, n: found)
    pick = lambda path, cls: [f for f in classic if f["markup_type"] == path.stem and f["class_name"] == cls]
    monkeypatch.setattr(geophys, "finds_points", lambda path, *a, **k: pick(path, "finds_points"))
    monkeypatch.setattr(geophys, "object_polys", lambda path, *a, **k: pick(path, "object_poly"))
    monkeypatch.setattr(pipeline, "region_crs", lambda reg, cfg=None: "EPSG:3857")


def _region():
    return Region(name="r", path=Path("."), rasters={"maggr": [Path("maggr.tif")], "etgr": [Path("etgr.tif")]})


CLASSIC = [_f("finds_points", "maggr", 0, 0.9), _f("object_poly", "maggr", 30, 0.9), _f("object_poly", "etgr", 60, 0.9)]
PIX = [_f("finds_points", "maggr", 100, 0.95, "maggr_pix"), _f("finds_points", "maggr", 110, 0.85, "maggr_pix"),
       _f("object_poly", "maggr", 130, 0.80, "maggr_pix", r=3.0), _f("object_poly", "maggr", 160, 0.60, "maggr_pix", r=3.0)]


def test_replace_S_then_stack6_sees_pix(cfg, monkeypatch):
    from expds import geo_pixgbm
    _patch_classic(monkeypatch, CLASSIC)
    seen = {}

    def fake(reg, c, md, guard=None, status=None):
        seen["md"] = Path(md)
        status["done"] = {("finds_points", "maggr"): {"a.tif"}, ("object_poly", "maggr"): {"a.tif"}}
        return [dict(f) for f in PIX]

    monkeypatch.setattr(geo_pixgbm, "region_feats", fake)
    feats = pipeline.geophys_feats(_region(), cfg, pipeline.TimeGuard(1500))
    assert seen["md"] == ROOT / "models/pixgbm/all"                 # от корня репо, не от cwd
    groups = sorted((f["class_name"], f["markup_type"], f.get("threshold_group")) for f in feats)
    # классика maggr (оба класса) заменена pix; etgr — у pix нет кандидатов etgr — остаётся
    assert ("object_poly", "etgr", None) in groups
    assert not [f for f in feats if f["markup_type"] == "maggr" and f.get("threshold_group") != "maggr_pix"]
    out = watchdog.finish(feats, cfg)
    fp = [f for f in out if f["class_name"] == "finds_points"]
    assert len(fp) == 1                                              # 0,85 < 0,902915 отсечён порогом maggr_pix
    op_m = [f for f in out if f["class_name"] == "object_poly" and f["markup_type"] == "maggr"]
    assert op_m and all(f["threshold_group"] == "maggr_poly" for f in op_m)   # stack6 (multipole) прошёл по pix
    assert all(f["confidence"] >= 0.684725 or f["confidence"] < 0.8 for f in op_m)  # 0,60 отсечён до multipole
    assert len({round(f["geom"].centroid.x) for f in op_m if f["confidence"] == 0.8}) >= 1


def test_flags_off_independently(cfg, monkeypatch):
    from expds import geo_pixgbm
    _patch_classic(monkeypatch, CLASSIC)
    monkeypatch.setattr(geo_pixgbm, "region_feats", lambda *a, **k: [dict(f) for f in PIX])
    c = dict(cfg, geophys=dict(cfg["geophys"], pixgbm=dict(cfg["geophys"]["pixgbm"], enabled=False)))
    feats = pipeline.geophys_feats(_region(), c, pipeline.TimeGuard(1500))
    assert not [f for f in feats if f.get("threshold_group") == "maggr_pix"] and len(feats) == 3
    c2 = dict(cfg, post=dict(cfg["post"], multipole_object_poly=False, filter_finds_points=False))
    out = watchdog.finish(pipeline.geophys_feats(_region(), c2, pipeline.TimeGuard(1500)), c2)
    assert [f for f in out if f.get("threshold_group") == "maggr_pix" and f["class_name"] == "object_poly"]


def test_pix_fell_raster_keeps_classic(cfg, monkeypatch):
    """bot-3 5deb94a: растр, где pix не отработал (срок/ошибка, status["fell"]), сохраняет классику своего
    (класс, метод); где отработал (status["done"]) — классика заменена."""
    from expds import geo_pixgbm
    _patch_classic(monkeypatch, CLASSIC)

    def fake(reg, c, md, guard=None, status=None):
        status["done"] = {("finds_points", "maggr"): {"a.tif"}}
        status["fell"] = {("object_poly", "maggr"): {"a.tif"}}
        return [dict(f) for f in PIX if f["class_name"] == "finds_points"] + [dict(PIX[2])]

    monkeypatch.setattr(geo_pixgbm, "region_feats", fake)
    feats = pipeline.geophys_feats(_region(), cfg, pipeline.TimeGuard(1500))
    classic = sorted((f["class_name"], f["markup_type"]) for f in feats if f.get("threshold_group") is None)
    assert classic == [("object_poly", "etgr"), ("object_poly", "maggr")]   # finds maggr заменены, op maggr — нет


def _rf_env(monkeypatch, tmp_path, prob_calls):
    from expds import geo_pixgbm
    import numpy as np

    class M:
        backend = "lightgbm"

        def feature_name(self):
            return []

    monkeypatch.setitem(geo_pixgbm.PHASE, "level", 0)          # лестница липкая на процесс — не портить другим тестам
    monkeypatch.setitem(geo_pixgbm.PHASE, "rate", {})
    monkeypatch.setitem(geo_pixgbm.PHASE, "steps", dict.fromkeys(geo_pixgbm.PHASE["steps"], 0))
    monkeypatch.setattr(geo_pixgbm, "load_models", lambda md, be="lightgbm": {"poly": M()} if Path(md).exists() else {})
    monkeypatch.setattr(geo_pixgbm, "pick_rasters", lambda files: list(files))
    monkeypatch.setattr(geophys, "survey_boundary", lambda *a, **k: None)
    monkeypatch.setattr(geophys, "load_mask", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "region_crs", lambda reg, cfg=None: "EPSG:3857")
    v = np.ones((10, 10), bool)
    monkeypatch.setattr(geo_pixgbm, "load_map", lambda *a, **k: {"crs": "EPSG:3857", "valid": v, "noise": ~v})
    monkeypatch.setattr(geo_pixgbm, "prob_map", lambda d, mt, use, **k: prob_calls.append(mt) or {"poly": 1})
    monkeypatch.setattr(geo_pixgbm, "poly_candidates", lambda *a, **k: [{"class_name": "object_poly"}])
    return Region(name="r", path=Path("."), rasters={"etgr": [Path("a.tif"), Path("b.tif")]})


def test_region_feats_deadline_and_no_models(monkeypatch, tmp_path):
    """Срок вышел (budget 0) — pix не считается, оба растра в status["fell"]; в срок — full, оба в done;
    нет моделей — [] без чтения растров (bot-7)."""
    from expds import geo_pixgbm
    calls = []
    reg = _rf_env(monkeypatch, tmp_path, calls)
    monkeypatch.setenv("PIXGBM_BUDGET_S", "1e6")
    st = {}
    assert len(geo_pixgbm.region_feats(reg, {}, tmp_path, status=st)) == 2 and calls == ["etgr", "etgr"]
    assert st["done"] == {("object_poly", "etgr"): {"a.tif", "b.tif"}} and not st["fell"]
    reg = _rf_env(monkeypatch, tmp_path, calls)
    monkeypatch.setenv("PIXGBM_BUDGET_S", "-1")
    st, calls[:] = {}, []
    assert geo_pixgbm.region_feats(reg, {}, tmp_path, status=st) == [] and calls == []
    assert st["fell"] == {("object_poly", "etgr"): {"a.tif", "b.tif"}}
    assert geo_pixgbm.PHASE["steps"]["classic_budget"] == 2
    monkeypatch.setitem(geo_pixgbm.PHASE, "level", 0)
    assert geo_pixgbm.region_feats(reg, {}, tmp_path / "nope") == [] and calls == []


def test_check_submission_pixgbm_model_dir(tmp_path):
    """check_submission: model_dir PIXGBM — папка; при enabled нужны poly.txt и finds.txt, лишних WARN нет."""
    import importlib.util, zipfile
    spec = importlib.util.spec_from_file_location("cs", ROOT / "scripts" / "check_submission.py")
    cs = importlib.util.module_from_spec(spec); spec.loader.exec_module(cs)
    toml = '[geophys.pixgbm]\nenabled = true\nmodel_dir = "models/pixgbm/all"\n'
    for files, want in ((["poly.txt", "finds.txt"], "PASS"), (["poly.txt"], "FAIL")):
        zp = tmp_path / f"z{len(files)}.zip"
        with zipfile.ZipFile(zp, "w") as z:
            z.writestr("configs/default.toml", toml)
            for f in files:
                z.writestr(f"models/pixgbm/all/{f}", "x")
        cs.RESULTS.clear()
        with zipfile.ZipFile(zp) as z:
            cs.check_config_static(z)
        st = [s for s, c, m in cs.RESULTS if c == "config" and ("весам" in m or "отсутствуют" in m or "PIXGBM включён" in m)]
        # после cherry-pick bot-8 7878087 нехватка класса — отдельный FAIL check_pixgbm_static, путь к папке PASS
        assert ("FAIL" in st) == (want == "FAIL") and set(st) <= {"PASS", want}, cs.RESULTS
        assert not any(s == "WARN" and "pixgbm" in m for s, c, m in cs.RESULTS)
