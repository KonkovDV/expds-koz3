"""bot-8 bot8-li-half: li_half бота 2 в нашем пайплайне — мягкий потолок, enabled только bool, пороги, порядок
отборов с cnx, проверки check_submission. Без весов: сеть и проход подменены."""
import logging

from pathlib import Path

import pytest
from shapely.geometry import Point

from expds import hillfort, li_half, pipeline
from expds.pipeline import TimeGuard

ROOT = Path(__file__).resolve().parents[1]
THR = {"kurgany_povrezhdennye": 0.5, "kurgany_tselye": 0.5}


class Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


def _guard(soft_left, environ=None, cfg=None):
    clk = Clock(0.0)
    g = TimeGuard(1500, clock=clk).set_soft(soft_ceiling_s=soft_left,
                                            soft_need=pipeline.soft_need_from(cfg or {}, environ=environ or {}))
    return g


# ---------- enabled только bool, пороги, оценка ----------

@pytest.mark.parametrize("raw", ["true", 1, "yes", 1.0])
def test_enabled_only_toml_bool(raw, caplog):
    li_half._ENABLED_WARNED.clear()
    with caplog.at_level(logging.WARNING, logger="expds.li_half"):
        assert li_half.enabled({"li_half": {"enabled": raw}}) is False
        assert li_half.params({"li_half": {"enabled": raw}})["enabled"] is False
    assert sum("не bool" in r.getMessage() for r in caplog.records) == 1        # одна строка на значение
    assert li_half.enabled({"li_half": {"enabled": True}}) is True
    assert li_half.enabled({"li_half": {"enabled": False}}) is False and li_half.enabled({}) is False


def test_threshold_problem_like_cnx():
    tp = li_half.threshold_problem
    assert "нет [thresholds_by_markup.li_half]" in tp({"li_half": {"enabled": True}})
    assert "нет [thresholds_by_markup.li_half]" in tp({"thresholds_by_markup": {"li_half": {}}})
    assert "нет порогов" in tp({"thresholds_by_markup": {"li_half": {"kurgany_tselye": 0.5}}})
    assert "другого прохода" in tp({"li_half": {"group": "li"}, "thresholds_by_markup": {"li": THR}})
    assert tp({"thresholds_by_markup": {"li_half": THR}}) is None
    assert li_half.has_thresholds({"thresholds_by_markup": {"li_half": THR}})


def test_est_s():
    hp = li_half.params({})
    assert li_half.est_s(0.0, hp) == 5.0 and li_half.est_s(-3, hp) == 5.0
    assert li_half.est_s(100.0, hp) == pytest.approx(35.0)
    assert li_half.est_s(100.0, li_half.params({"li_half": {"tta": 8}})) == pytest.approx(280.0)


def test_default_config_li_half_off_and_soft_key():
    cfg = pipeline.load_config(ROOT / "configs" / "default.toml")
    assert li_half.enabled(cfg) is False and "soft_need_s" not in (cfg.get("li_half") or {})
    assert pipeline.SOFT_NEED_DEFAULT["li_half"] == 30.0
    assert pipeline.soft_need_from(cfg, environ={})["li_half"] == 30.0


# ---------- мягкий потолок ----------

def test_soft_need_li_half_config_and_env(capsys):
    assert pipeline.soft_need_from({"time": {"soft": {"li_half": 45}}}, environ={})["li_half"] == 45.0
    need = pipeline.soft_need_from({"time": {"soft": {"li_half": 45}}}, environ={"EXPDS_SOFT_NEED_LI_HALF_S": "7"})
    assert need["li_half"] == 7.0                                   # env важнее [time.soft]
    assert pipeline.soft_need_from({}, environ={"EXPDS_SOFT_NEED_LI_HALF_S": "x"})["li_half"] == 30.0
    assert "не принят" in capsys.readouterr().err


def test_li_half_skipped_when_soft_left_short():
    # порог: max(30, est 5) + wide 120 + sigma 60 + verify 60 = 270
    g = _guard(250)
    assert not pipeline._li_half_soft_ok(g, {}, "r/a.las", 5.0)
    rec = [s for s in g.soft_skipped if s["phase"] == "li_half"]
    assert len(rec) == 1 and rec[0]["what"] == "r/a.las" and rec[0]["need_s"] == 270.0
    assert pipeline._li_half_soft_ok(_guard(271), {}, "r/a.las", 5.0)
    # оценка больше 30 — берётся она: 100 + 240 = 340
    assert not pipeline._li_half_soft_ok(_guard(300), {}, "x", 100.0)
    assert pipeline._li_half_soft_ok(_guard(341), {}, "x", 100.0)
    # без потолка (тесты, скрипты) — всегда да
    assert pipeline._li_half_soft_ok(TimeGuard(1500), {}, "x", 10 ** 6)


def test_li_half_soft_env_override():
    env = {"EXPDS_SOFT_NEED_LI_HALF_S": "100"}
    assert not pipeline._li_half_soft_ok(_guard(300, env), {}, "x", 5.0)      # 100 + 240 = 340
    assert pipeline._li_half_soft_ok(_guard(341, env), {}, "x", 5.0)
    env = {"EXPDS_SOFT_NEED_LI_HALF_S": "0", "EXPDS_SOFT_NEED_SIGMA_S": "0", "EXPDS_SOFT_NEED_VERIFY_S": "0",
           "EXPDS_SOFT_NEED_WIDE_WINDOW_S": "0"}
    assert pipeline._li_half_soft_ok(_guard(6, env), {}, "x", 5.0)            # остаётся оценка 5 с


def test_li_half_is_first_in_skip_order():
    """При любом soft_left: если li_half ещё идёт, то и окно 256 м, cnx, σ (с резервом кропа) и кроп идут."""
    cnx_cfg = {"li_cnx": {"enabled": True, "weights": "models/cnx/w.pt", "soft_need_s": 120}}
    assert pipeline.lidar_cnx.enabled(cnx_cfg)
    for cfg in ({}, cnx_cfg, {"time": {"soft": {"sigma": 200, "li_half": 0}}}):
        for left in range(1, 800, 3):
            g = _guard(left, cfg=cfg)
            if not pipeline._li_half_soft_ok(g, cfg, "x", 5.0):
                continue
            assert pipeline._soft_ok(g, "wide_window"), (cfg, left)
            assert pipeline._soft_ok(g, "sigma", extra_s=g.soft_need_s("verify")), (cfg, left)
            assert pipeline._soft_ok(g, "verify"), (cfg, left)
            if cfg is cnx_cfg:
                assert g.soft_ok(120.0), left
    # cnx включён — его резерв тоже держится: 270 + 120
    assert not pipeline._li_half_soft_ok(_guard(300, cfg=cnx_cfg), cnx_cfg, "x", 5.0)
    assert pipeline._li_half_soft_ok(_guard(391, cfg=cnx_cfg), cnx_cfg, "x", 5.0)


# ---------- lidar_feats: подменённые сеть и проход ----------

class FakeSeg:
    def __init__(self):
        self.meta: dict = {}
        self.classes = ["kurgany_povrezhdennye", "kurgany_tselye"]
        self.keep = None


def _circ(x, conf, group=None, cls="kurgany_tselye", region="region_0001", mt="li"):
    f = {"geom": Point(x, 1000.0).buffer(2.0), "class_name": cls, "confidence": conf, "region_name": region,
         "markup_type": mt, "source": "t"}
    if group:
        f["threshold_group"] = group
    return f


@pytest.fixture
def lidar_env(val_dir, monkeypatch):
    from expds.catalog import scan_root

    calls = []
    monkeypatch.setattr(pipeline, "_cached_segmenters", lambda cache, cfg, mt, root: [FakeSeg()] if mt == "li" else [])
    monkeypatch.setattr(pipeline, "_li_with_model", lambda *a, **k: [_circ(0.0, 0.9)])     # принят li (0,19422)
    monkeypatch.setattr(hillfort, "candidates", lambda *a, **k: [])

    def half(seg, d, region, source, cand, hp, dump=None, extra=None):
        calls.append({"hp": hp, "dump": dump})
        return [_circ(5.0, 0.8, "li_half", region=region), _circ(50.0, 0.8, "li_half", region=region),
                _circ(90.0, 0.1, "li_half", region=region)]

    monkeypatch.setattr(li_half, "half_pass", half)
    monkeypatch.delenv(li_half.DUMP_ENV, raising=False)
    reg = next(r for r in scan_root(val_dir) if r.las)
    return reg, calls, val_dir


def _cfg(**lh):
    return {"li_half": {"enabled": True, **lh}, "thresholds_by_markup": {"li_half": THR},
            "hillfort": {"enabled": True}}


def _half(out):
    return sorted(round(f["geom"].centroid.x) for f in out if f.get("threshold_group") == "li_half")


def test_lidar_feats_li_half_new_only_and_hillfort_not_clobbering(lidar_env):
    """С [hillfort] enabled (как в default.toml) у бота 2 hp затирался HillfortParams → TypeError, лидар участка терялся."""
    reg, calls, root = lidar_env
    out = pipeline.lidar_feats(reg, _cfg(), TimeGuard(1000), {}, root)
    assert len(calls) == 1 and calls[0]["hp"]["group"] == "li_half"
    assert _half(out) == [50]                  # 5 м от принятого li — вон; 0,1 < порога 0,5 — вон
    assert any(f.get("markup_type") == "li" and not f.get("threshold_group") for f in out)


def test_lidar_feats_off_without_thresholds_or_non_bool(lidar_env, caplog):
    reg, calls, root = lidar_env
    base = pipeline.lidar_feats(reg, {"hillfort": {"enabled": True}}, TimeGuard(1000), {}, root)
    assert calls == [] and _half(base) == []
    cache: dict = {}
    with caplog.at_level(logging.WARNING):
        for _ in range(2):
            out = pipeline.lidar_feats(reg, {"li_half": {"enabled": True}, "hillfort": {"enabled": True}},
                                       TimeGuard(1000), cache, root)
            assert calls == [] and _half(out) == []          # без порогов прохода нет совсем
    assert sum("объекты li_half НЕ выдаются" in r.getMessage() for r in caplog.records) == 1
    out = pipeline.lidar_feats(reg, dict(_cfg(), li_half={"enabled": "true"}), TimeGuard(1000), {}, root)
    assert calls == [] and _half(out) == []


def test_lidar_feats_dump_only_when_disabled(lidar_env, monkeypatch, tmp_path):
    reg, calls, root = lidar_env
    monkeypatch.setenv(li_half.DUMP_ENV, str(tmp_path / "dump"))
    out = pipeline.lidar_feats(reg, {"li_half": {"enabled": False}, "thresholds_by_markup": {"li_half": THR}},
                               TimeGuard(1000), {}, root)
    assert len(calls) == 1 and calls[0]["dump"] == tmp_path / "dump" and _half(out) == []


def test_lidar_feats_soft_ceiling_skips_li_half_only(lidar_env):
    reg, calls, root = lidar_env
    g = TimeGuard(1000).set_soft(soft_ceiling_s=200)       # soft_left ≈ 200 < 270
    out = pipeline.lidar_feats(reg, _cfg(), g, {}, root)
    assert calls == [] and _half(out) == []
    assert [s["phase"] for s in g.soft_skipped] == ["li_half"]
    assert any(f.get("markup_type") == "li" for f in out)  # основной li не тронут
    g = TimeGuard(1000).set_soft(soft_ceiling_s=1000)
    assert _half(pipeline.lidar_feats(reg, _cfg(), g, {}, root)) == [50] and not g.soft_skipped


# ---------- порядок отборов li_half → cnx ----------

def test_cnx_does_not_recount_accepted_li_half():
    cfg = {"thresholds_by_markup": {"li_half": THR, "li_cnx": THR}}
    pool = [_circ(0.0, 0.9), _circ(50.0, 0.8, "li_half"), _circ(150.0, 0.2, "li_half")]
    fresh = [_circ(52.0, 0.9, "li_cnx"), _circ(152.0, 0.9, "li_cnx"), _circ(3.0, 0.9, "li_cnx")]
    kept = pipeline.dedupe_cnx(fresh, pool, cfg)
    # 52 — дубль принятого li_half (50), 3 — дубль принятого li; 152 — li_half там ниже своего порога
    assert [round(f["geom"].centroid.x) for f in kept] == [152]
