"""дедлайн внутри цикла окон снимков — медленные окна, поддельные часы, частичный результат.

Рецепт: competitors/NOTES_KateBuikina-expds-solution.md §5 (segmenter._deadline_hit в начале цикла окон).
У нас — те же бюджет [time] budget_s и дедлайн сторожа (watchdog.deadline_left), без новых часов.
"""
import json
import math
from types import SimpleNamespace

import numpy as np

from test_raster_infer import SPOTS, write_raster
from expds import pipeline, watchdog, writer
from expds.pipeline import TimeGuard, _raster_with_model


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class SlowSeg:
    """Яркость первого канала = вероятность; каждое окно «идёт» step секунд по поддельным часам."""
    classes = ["kurgany_tselye"]
    res = 0.5
    meta = {"in_channels": 3, "max_upsample": 2.0}

    def __init__(self, clock, step=100.0):
        self.clock, self.step, self.calls = clock, step, 0

    def predict(self, img):
        self.calls += 1
        self.clock.t += self.step
        return img[:1].astype(np.float32) / 255.0


def _centers(feats):
    return sorted((round(f["geom"].centroid.x, 1), round(f["geom"].centroid.y, 1)) for f in feats)


def test_window_loop_stops_near_deadline_with_partial_result(tmp_path):
    p = write_raster(tmp_path / "ae.tif")
    full = _raster_with_model(SlowSeg(Clock()), p, "r", "ae", block=64, overlap=16)
    assert len(full) == len(SPOTS)
    clk = Clock()
    seg = SlowSeg(clk)
    guard = TimeGuard(250, window_reserve_s=30, clock=clk)
    part = _raster_with_model(seg, p, "r", "ae", block=64, overlap=16, guard=guard)
    # 250 → 150 → 50 → −50 с: три окна, четвёртое не начинается (осталось ≤ 30 с)
    assert seg.calls == 3
    assert len(guard.partial) == 1
    info = guard.partial[0]
    assert info["windows_done"] == 3 and info["windows_total"] > 3 and info["file"] == "ae.tif"
    assert 0 < len(part) < len(full) and info["feats"] == len(part)
    assert set(_centers(part)) <= set(_centers(full))      # частичный результат — подмножество полного


def test_one_broken_window_keeps_objects_from_the_others(tmp_path):
    """Слабые спутник и аэрофото: сбой одного окна не выбрасывает объекты остальных окон файла."""
    p = write_raster(tmp_path / "ae.tif")
    full = _raster_with_model(SlowSeg(Clock()), p, "r", "ae", block=64, overlap=16)

    class OneBoom(SlowSeg):
        def __init__(self, clock):
            super().__init__(clock)
            self.hit = False

        def predict(self, img):
            self.calls += 1
            if not self.hit and float(img.max()) > 200:
                self.hit = True
                raise RuntimeError("окно")
            return img[:1].astype(np.float32) / 255.0

    part = _raster_with_model(OneBoom(Clock()), p, "r", "ae", block=64, overlap=16)
    assert 0 < len(part) < len(full)
    assert set(_centers(part)) < set(_centers(full))


def test_three_broken_windows_stop_the_file_and_keep_what_was_found(tmp_path):
    p = write_raster(tmp_path / "ae.tif")

    class AlwaysBoom(SlowSeg):
        def predict(self, img):
            self.calls += 1
            raise RuntimeError("окно")

    seg = AlwaysBoom(Clock())
    assert _raster_with_model(seg, p, "r", "ae", block=64, overlap=16) == []
    assert seg.calls == 3


def test_window_deadline_off_or_far_keeps_full_result(tmp_path):
    p = write_raster(tmp_path / "ae.tif")
    clk = Clock()
    off = TimeGuard(250, window_reserve_s=None, clock=clk)    # [time] window_deadline = false
    assert len(_raster_with_model(SlowSeg(clk), p, "r", "ae", block=64, overlap=16, guard=off)) == len(SPOTS)
    clk2 = Clock()
    far = TimeGuard(10 ** 6, window_reserve_s=30, clock=clk2)
    assert len(_raster_with_model(SlowSeg(clk2), p, "r", "ae", block=64, overlap=16, guard=far)) == len(SPOTS)
    assert not off.partial and not far.partial


def test_window_reserve_config():
    assert pipeline.window_reserve({}) == pipeline.DEFAULT_WINDOW_RESERVE_S
    assert pipeline.window_reserve({"time": {"window_reserve_s": 45}}) == 45.0
    assert pipeline.window_reserve({"time": {"window_deadline": False, "window_reserve_s": 45}}) is None


def test_watchdog_deadline_bounds_window_loop():
    """Если сторож (1800 − guard_margin_s от старта процесса) ближе бюджета — решает он."""
    clk = Clock(watchdog.PROCESS_START)
    assert math.isclose(watchdog.deadline_left(180, clock=clk), watchdog.PLATFORM_LIMIT_S - 180, abs_tol=1e-6)  # 1620 с
    clk.t += 1600
    assert math.isclose(watchdog.deadline_left(180, clock=clk), 20, abs_tol=1e-6)   # float: 19.999999999996362
    guard = TimeGuard(10 ** 6, window_reserve_s=30, hard_left=lambda: watchdog.deadline_left(180, clock=clk),
                      clock=clk)
    assert guard.ok(60) and not guard.window_ok()


def test_run_writes_valid_partial_output_and_exits_normally(tmp_path, monkeypatch):
    """Весь run(): окна медленные → частичный файл через обычный конец (checkpoint → safe_finish → writer)."""
    p = write_raster(tmp_path / "ae.tif")
    clk = Clock()
    seg = SlowSeg(clk)
    reg = SimpleNamespace(name="r", las=[], rasters={"ae": [p]}, utm_json=None, boundaries=[], noise=[])

    class FakeClockGuard(TimeGuard):
        def __init__(self, budget_s, window_reserve_s=None, hard_left=None, clock=None):
            assert hard_left is None                       # watchdog = false в конфиге теста
            super().__init__(budget_s, window_reserve_s, None, clock=clk)

    orig = pipeline._raster_with_model
    monkeypatch.setattr(pipeline, "_raster_with_model",
                        lambda *a, **k: orig(*a, **{**k, "block": 64, "overlap": 16}))
    monkeypatch.setattr(pipeline, "TimeGuard", FakeClockGuard)
    monkeypatch.setattr(pipeline, "scan_root", lambda _p: [reg])
    monkeypatch.setattr(pipeline, "_cached_segmenters", lambda cache, cfg, mt, root: [seg] if mt == "ae" else [])
    exits = []
    monkeypatch.setattr(watchdog.os, "_exit", lambda code: exits.append(code))
    monkeypatch.setenv("EXPDS_TMP", str(tmp_path / "tmp"))
    cfg = tmp_path / "t.toml"
    cfg.write_text("[time]\nbudget_s = 250\nreserve_s = 10\nwatchdog = false\nwindow_reserve_s = 30\n"
                   "[verify]\nenabled = false\n", encoding="utf-8")
    out = tmp_path / "o" / "result.geojson"
    summary = pipeline.run(tmp_path, out, cfg)
    assert exits == []                                     # обычный конец, не аварийный выход сторожа
    assert seg.calls == 3
    assert [x["windows_done"] for x in summary["partial_files"]] == [3]
    rep = writer.validate(out)
    assert rep["ok"], rep["errors"]
    n = len(json.loads(out.read_text())["features"])
    assert 0 < n < len(SPOTS)
    assert pipeline._WINDOW_GUARD is None                  # вне run() проверки в цикле окон нет
