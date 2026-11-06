"""мягкий потолок 1200 с (EXPDS_SOFT_CEILING_S) — API guard и отключение необязательных шагов.

TimeGuard 1500 (budget_s от старта run()) и сторож 1620 (от watchdog.PROCESS_START) не меняются.
"""
import json
import time
from types import SimpleNamespace

import pytest
from shapely.geometry import Point

from expds import pipeline, watchdog, writer
from expds.pipeline import TimeGuard


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


# ---------- конфигурация потолка ----------

def test_soft_ceiling_default_env_and_config():
    assert pipeline.soft_ceiling_from({}, environ={}) == 1200.0
    assert pipeline.SOFT_CEILING_DEFAULT_S == 1200.0
    assert pipeline.soft_ceiling_from({"time": {"soft_ceiling_s": 900}}, environ={}) == 900.0
    # env важнее конфига
    assert pipeline.soft_ceiling_from({"time": {"soft_ceiling_s": 900}}, environ={"EXPDS_SOFT_CEILING_S": "1500"}) == 1500.0
    assert pipeline.soft_ceiling_from({}, environ={"EXPDS_SOFT_CEILING_S": " "}) == 1200.0     # пусто → как нет


@pytest.mark.parametrize("raw", ["0", "-5", "abc", "nan", "inf"])
def test_soft_ceiling_invalid_env_disables_with_warning(raw, capsys):
    assert pipeline.soft_ceiling_from({}, environ={"EXPDS_SOFT_CEILING_S": raw}) is None
    assert "мягкий потолок выключен" in capsys.readouterr().err


@pytest.mark.parametrize("raw", [0, -1, "x", True])
def test_soft_ceiling_invalid_config_disables_with_warning(raw, capsys):
    assert pipeline.soft_ceiling_from({"time": {"soft_ceiling_s": raw}}, environ={}) is None
    assert "мягкий потолок выключен" in capsys.readouterr().err


def test_soft_need_thresholds_configurable(capsys):
    base = ("wide_window", "sigma", "verify")
    need = pipeline.soft_need_from({}, environ={})
    assert {k: need[k] for k in base} == {"wide_window": 120.0, "sigma": 60.0, "verify": 60.0}
    need = pipeline.soft_need_from({"time": {"soft": {"sigma": 90, "verify": 30}}},
                                   environ={"EXPDS_SOFT_NEED_VERIFY_S": "45", "EXPDS_SOFT_NEED_WIDE_WINDOW_S": "bad"})
    assert {k: need[k] for k in base} == {"wide_window": 120.0, "sigma": 90.0, "verify": 45.0}
    assert "не принят" in capsys.readouterr().err


# ---------- API guard ----------

def test_guard_api_t0_elapsed_soft_left():
    clk = Clock(1000.0)
    g = TimeGuard(1500, clock=clk)
    assert g.t0 == g.run_t0 == 1000.0
    assert g.soft_ceiling is None and g.soft_left() == float("inf") and g.soft_ok(10 ** 9)
    g.set_soft(soft_ceiling_s=1200)
    clk.t += 1100
    assert g.elapsed() == 1100 and g.soft_left() == 100
    assert g.soft_ok(60) and not g.soft_ok(100)
    assert g.soft_skipped == []                              # без phase — без записи
    assert not g.soft_ok(120, phase="sigma", what="тест")
    assert g.soft_skipped[0]["phase"] == "sigma" and g.soft_skipped[0]["soft_left_s"] == 100


def test_t0_is_process_start_in_monotonic_base_budget_unchanged():
    """t0 = min(PROCESS_START, старт run()) — база time.monotonic; бюджет 1500 по-прежнему от старта run()."""
    g = TimeGuard(1500).set_soft(process_start=watchdog.PROCESS_START, soft_ceiling_s=1200)
    assert g.t0 == watchdog.PROCESS_START <= g.run_t0
    assert g.elapsed() >= g.run_elapsed()
    assert abs(g.elapsed() - (time.monotonic() - watchdog.PROCESS_START)) < 5
    # left() и run_elapsed() вызывают clock() отдельно — под нагрузкой suite
    # разница двух семплов ~1e-6; допуск 1 мс (семантика: бюджет от run_t0).
    assert abs(g.left() - (1500 - g.run_elapsed())) < 1e-3
    # 1620-сторож и мягкий потолок считают от одного начала
    assert abs((watchdog.PLATFORM_LIMIT_S - 180 - g.elapsed()) - watchdog.deadline_left(180)) < 1


def test_budget_1500_and_watchdog_1620_unchanged():
    assert pipeline.DEFAULT_BUDGET_S == 1500
    assert watchdog.PLATFORM_LIMIT_S - watchdog.GUARD_MARGIN_S == 1620
    clk = Clock(0.0)
    g = TimeGuard(1500, clock=clk).set_soft(process_start=-10 ** 6, soft_ceiling_s=1)   # чужая база часов — игнор
    assert g.t0 == 0.0
    clk.t = 1379
    assert g.ok(120) and not g.soft_ok(0)                   # потолок не трогает ok()/left()
    clk.t = 1381
    assert not g.ok(120)


# ---------- весь run() ----------

class FakeSeg:
    classes = ["kurgany_povrezhdennye"]
    keep = None


def _feat(region, group, x):
    return {"geom": Point(x, 0.0).buffer(5.0), "class_name": "kurgany_povrezhdennye", "confidence": 0.9,
            "markup_type": "sp_or", "region_name": region, "threshold_group": group, "source": "t"}


def _setup(tmp_path, monkeypatch):
    calls = []
    with_las = SimpleNamespace(name="a", las=[tmp_path / "a.las"], rasters={}, utm_json=None, boundaries=[], noise=[])
    no_las = SimpleNamespace(name="b", las=[], rasters={"sp_or": [tmp_path / "b.tif"]}, utm_json=None,
                             boundaries=[], noise=[])
    monkeypatch.setattr(pipeline, "scan_root", lambda _p: [with_las, no_las])
    monkeypatch.setattr(pipeline, "geophys_feats", lambda *_a, **_k: [])
    monkeypatch.setattr(pipeline, "lidar_feats", lambda *_a, **_k: calls.append("li") or [])
    monkeypatch.setattr(pipeline, "_cached_segmenters",
                        lambda cache, cfg, mt, root: [FakeSeg()] if mt in ("sp_or", "sp_focal2") else [])

    def raster(seg, path, region, markup, **_k):
        calls.append("opt")
        return [_feat(region, None, 100.0 * len(calls))]

    real_wide = pipeline._wide_without_lidar

    def wide(reg, cfg, guard, seg_cache, root, feats):
        out = real_wide(reg, cfg, guard, seg_cache, root, feats)
        if len(out) > len(feats):
            calls.append("wide")
        return out

    monkeypatch.setattr(pipeline, "_raster_with_model", raster)
    monkeypatch.setattr(pipeline, "_wide_without_lidar", wide)
    monkeypatch.setattr(pipeline, "li_dev_feats", lambda *_a, **_k: calls.append("sigma") or [])
    import expds.crop_model as cm

    monkeypatch.setattr(cm, "maybe_verify", lambda feats, regions, cfg: calls.append("verify") and None)

    class Dummy:
        def __init__(self, *_a, **_k):
            pass

        def close(self):
            pass

    monkeypatch.setattr("expds.lidar.DemPrefetch", Dummy)
    monkeypatch.setenv("EXPDS_TMP", str(tmp_path / "tmp"))
    cfg = tmp_path / "t.toml"
    cfg.write_text('[time]\nwatchdog = false\n[models]\nli_dev = {path = "x.pt"}\nsp_focal2 = {path = "y.pt"}\n'
                   '[verify]\nenabled = true\nweights = "none.pt"\n', encoding="utf-8")
    return calls, cfg


def test_run_default_ceiling_runs_all_optional_steps(tmp_path, monkeypatch):
    monkeypatch.delenv("EXPDS_SOFT_CEILING_S", raising=False)
    monkeypatch.setattr(watchdog, "PROCESS_START", time.monotonic())    # свежий «процесс»: до 1200 с далеко
    calls, cfg = _setup(tmp_path, monkeypatch)
    out = tmp_path / "o" / "result.geojson"
    summary = pipeline.run(tmp_path, out, cfg)
    assert "wide" in calls and "sigma" in calls and "verify" in calls
    assert "soft_skipped" not in summary
    assert writer.validate(out)["ok"]


@pytest.mark.parametrize("env", ["0.001", "1"])
def test_run_tiny_ceiling_skips_optional_still_writes_geojson(tmp_path, monkeypatch, caplog, env):
    monkeypatch.setenv("EXPDS_SOFT_CEILING_S", env)
    monkeypatch.setattr(watchdog, "PROCESS_START", time.monotonic() - 5)
    calls, cfg = _setup(tmp_path, monkeypatch)
    out = tmp_path / "o" / "result.geojson"
    with caplog.at_level("INFO"):
        summary = pipeline.run(tmp_path, out, cfg)
    assert "li" in calls and "opt" in calls                   # обязательное идёт как раньше
    assert "wide" not in calls and "sigma" not in calls and "verify" not in calls
    phases = [s["phase"] for s in summary["soft_skipped"]]
    assert phases == ["wide_window", "sigma", "verify"]
    assert all(s["soft_left_s"] <= s["need_s"] for s in summary["soft_skipped"])
    assert caplog.text.count("мягкий потолок: ") == 3
    assert "мягкий потолок отключил" in caplog.text          # строка протокола
    rep = writer.validate(out)
    assert rep["ok"], rep["errors"]
    assert len(json.loads(out.read_text(encoding="utf-8"))["features"]) >= 1


def test_run_disabled_ceiling_runs_all(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("EXPDS_SOFT_CEILING_S", "0")
    monkeypatch.setattr(watchdog, "PROCESS_START", time.monotonic() - 10 ** 5)   # «очень давно» — всё равно не режет
    calls, cfg = _setup(tmp_path, monkeypatch)
    summary = pipeline.run(tmp_path, tmp_path / "o" / "result.geojson", cfg)
    assert "wide" in calls and "sigma" in calls and "verify" in calls
    assert "soft_skipped" not in summary
    assert "мягкий потолок выключен" in capsys.readouterr().err


def test_pixgbm_deadline_uses_guard_t0_monotonic():
    """ geo_pixgbm.pix_deadline: T0 = min(старт процесса, guard.t0) — та же шкала time.monotonic."""
    import time as _time
    geo_pixgbm = pytest.importorskip("expds.geo_pixgbm", reason="expds/geo_pixgbm.py not in main (EN-similarity hold, docs/MERGE_2026-10-29.md)")
    from expds import watchdog
    g = pipeline.TimeGuard(1500).set_soft(process_start=watchdog.PROCESS_START, soft_ceiling_s=1200.0)
    assert g.clock is _time.monotonic and g.t0 <= _time.monotonic()
    t0 = min(geo_pixgbm.process_t0(), g.t0)
    assert abs(geo_pixgbm.pix_deadline(g, 300) - (t0 + 300)) < 1e-6
    assert geo_pixgbm.pix_deadline(g, 300) <= g.t0 + 300 + 1e-6
    g.t0 = g.t0 - 1000.0  # более ранний guard.t0 побеждает
    assert abs(geo_pixgbm.pix_deadline(g, 300) - (g.t0 + 300)) < 1e-6
