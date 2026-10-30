"""bot-6: D-FINE Ae/Or ([dfine_aeor]) на v3 — флаг выкл., нарезка, слияние внутри растра, геопривязка,
сторож времени, группы dfine_ae/dfine_or в постобработке v3 (R, frame_prior, verify, merge_within_markup)."""
import importlib.util
import inspect
import json
import logging
import math
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from affine import Affine
from pyproj import Transformer
from shapely.geometry import Point

from expds import dfine_aeor as A
from expds import post, sp_dfine

ROOT = Path(__file__).resolve().parents[1]
UTM = "EPSG:32637"
X0, Y0 = 400000.0, 6200000.0


def _cfg():
    return tomllib.loads((ROOT / "configs" / "default.toml").read_text(encoding="utf-8"))


def _thr(v=0.3):
    return {g: {c: v for c in A.MODEL_CLASSES} for g in A.GROUPS.values()}


# ── флаг, имена, классы ──────────────────────────────────────────────────────────────────────────────────
def test_default_off_and_names():
    cfg = _cfg()
    assert cfg["dfine_aeor"] == {"enabled": False}
    assert "aeor_dfine" not in cfg
    assert not A.enabled(cfg)
    assert A.GROUPS == {"ae": "dfine_ae", "or": "dfine_or"}
    tbm = cfg.get("thresholds_by_markup") or {}
    assert "dfine_ae" not in tbm   # чисел без OOF нет
    assert tbm["dfine_or"] == {"gorodishcha": 1.01}   # городища D-FINE Or выкл. до замера bot-10 с gor_dense
    assert A.GROUP_ALIASES == {"ae_dfine": "dfine_ae", "or_dfine": "dfine_or"}


def test_classes_match_compute_metrics():
    spec = importlib.util.spec_from_file_location("cm_b6", ROOT / "materials" / "compute_metrics.py")
    cm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cm)
    assert set(A.COMPETITION_CLASSES) == set(cm.CLASS_WEIGHTS)
    assert set(A.MODEL_CLASSES) <= set(cm.CLASS_WEIGHTS)


def test_model_labels_id2label_and_class_map(tmp_path):
    d = tmp_path / "w"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"id2label": {str(i): c for i, c in enumerate(A.MODEL_CLASSES)}}))
    assert A.model_labels(d, {}) == list(A.MODEL_CLASSES)
    (d / "config.json").write_text(json.dumps({"id2label": {"0": "mound", "1": "kurgany_tselye"}}))
    with pytest.raises(A.AeOrUnavailable):
        A.model_labels(d, {})
    assert A.model_labels(d, {"mound": "kurgany_povrezhdennye"}) == ["kurgany_povrezhdennye", "kurgany_tselye"]


def test_settings_per_modality_as_training():
    st = A.settings({"dfine_aeor": {"enabled": True, "weights": ["w"]}})
    assert st["res_m"] == {"ae": 0.5, "or": 0.5}
    assert (st["tile"]["ae"], st["overlap"]["ae"]) == (640, 160)
    assert (st["tile"]["or"], st["overlap"]["or"]) == (1024, 288)
    assert st["weights"] == {"ae": ["w"], "or": ["w"]}
    st = A.settings({"dfine_aeor": {"tile": {"or": 800}, "batch": 2}})
    assert st["tile"] == {"ae": 640, "or": 800} and st["batch"] == {"ae": 2, "or": 2}


# ── preflight: веса, пороги, маркер ──────────────────────────────────────────────────────────────────────
def test_preflight_unready_logs_marker(tmp_path, caplog):
    cfg = {"dfine_aeor": {"enabled": True}}
    with caplog.at_level(logging.ERROR):
        info = A.preflight(cfg, tmp_path)
    assert not info["ok"]
    assert A.UNAVAILABLE in caplog.text and "weights.ae пуст" in caplog.text and "нет порогов dfine_or" in caplog.text


def test_preflight_missing_weight_files_named(tmp_path, caplog):
    cfg = {"dfine_aeor": {"enabled": True, "weights": {"ae": ["models/x"], "or": ["models/x"]}},
           "thresholds_by_markup": _thr()}
    with caplog.at_level(logging.ERROR):
        assert not A.preflight(cfg, tmp_path)["ok"]
    assert str(tmp_path / "models/x/model.safetensors") in caplog.text


def test_thresholds_file_fills_and_toml_wins(tmp_path):
    w = tmp_path / "models" / "x"
    w.mkdir(parents=True)
    (w / "config.json").write_text("{}")
    (w / "model.safetensors").write_bytes(b"0")
    f = tmp_path / "models" / "thr.json"
    data = _thr(0.4)
    data["sp_or"] = {"kurgany_tselye": 0.9}   # чужие группы файл не трогает
    f.write_text(json.dumps(data))
    cfg = {"dfine_aeor": {"enabled": True, "weights": {"ae": ["models/x"], "or": ["models/x"]},
                          "thresholds_file": "models/thr.json"},
           "thresholds_by_markup": {"dfine_ae": {"gorodishcha": 0.7}}}
    info = A.preflight(cfg, tmp_path)
    assert info["ok"], info
    assert cfg["thresholds_by_markup"]["dfine_ae"]["gorodishcha"] == 0.7
    assert cfg["thresholds_by_markup"]["dfine_or"]["gorodishcha"] == 0.4
    assert "sp_or" not in cfg["thresholds_by_markup"]
    cfg2 = {"dfine_aeor": {"enabled": True, "weights": {"ae": ["models/x"], "or": ["models/x"]},
                           "thresholds_file": "models/none.json"}}
    assert not A.preflight(cfg2, tmp_path)["ok"]


# ── нарезка ──────────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("W,H,T,ov", [(640, 640, 640, 160), (641, 100, 640, 160), (3000, 2100, 1024, 288),
                                      (5000, 777, 640, 160)])
def test_tiling_full_coverage_and_overlap(W, H, T, ov):
    wins = A.windows(W, H, T, ov)
    cov = np.zeros((H, W), bool)
    for x, y in wins:
        cov[y:y + T, x:x + T] = True
    assert cov.all()
    xs = sorted({x for x, _ in wins})
    for a, b in zip(xs, xs[1:]):
        assert T - (b - a) >= ov          # перекрытие не меньше обучающего
    assert all(b - a <= T - ov for a, b in zip(xs, xs[1:]))
    assert A.n_windows(W, H, T, ov) == len(wins)


# ── слияние ──────────────────────────────────────────────────────────────────────────────────────────────
def test_wbf_merges_views_within_image_only_same_class():
    B = [[0, 0, 10, 10], [1, 0, 11, 10], [0, 0, 10, 10], [50, 50, 60, 60]]
    S = [0.8, 0.6, 0.9, 0.5]
    C = [0, 0, 1, 0]
    b, s, c = A.wbf(B, S, C, n_views=2, iou=0.55)
    assert len(s) == 3
    i = int(np.argmax((c == 0) & (b[:, 0] < 5)))
    assert s[i] == pytest.approx((0.8 + 0.6) / 2)
    b2, s2, c2 = A.nms(B, S, C, 0.5)
    assert len(s2) == 3


class _Scene:
    """Заглушка Scene: W×H пикселей 0,5 м UTM, яркие окна."""
    W = H = 700

    def __init__(self, path, crs, res_m=0.5):
        self.path, self.crs, self.res = path, crs, res_m
        k = res_m / 0.5
        self.W = self.H = int(math.ceil(700 / k))
        self.T = Affine(res_m, 0, X0, 0, -res_m, Y0)

    def read(self, x, y, n):
        return np.full((n, n, 3), 128, np.uint8)

    def close(self):
        pass


class _Model:
    """Заглушка D-FINE: один бокс 20×20 px в центре окна (класс 0, score 0.9) + шумовой ниже пола."""
    calls = 0

    def __call__(self, pixel_values):
        import torch

        _Model.calls += 1
        n = pixel_values.shape[0]
        logits = torch.full((n, 2, 5), -10.0)
        logits[:, 0, 0] = math.log(0.9 / 0.1)
        T = pixel_values.shape[-1]
        boxes = torch.tensor([[0.5, 0.5, 20 / T, 20 / T], [0.1, 0.1, 0.01, 0.01]]).repeat(n, 1, 1)
        return SimpleNamespace(logits=logits, pred_boxes=boxes)


def _reg(tmp_path, rasters):
    for m, ps in rasters.items():
        for p in ps:
            Path(p).write_bytes(b"x" * 10)
    return SimpleNamespace(name="r1", rasters=rasters, utm_json=None)


def _run(tmp_path, rasters, cfg_extra=None, cache=None, phase_left=lambda: 1e9):
    pytest.importorskip("torch")
    cfg = {"dfine_aeor": dict({"enabled": True, "flip_tta": True, "tile_s": 1e-6}, **(cfg_extra or {}))}
    reg = _reg(tmp_path, rasters)
    loader = lambda st, mk, root, c: ([_Model()], list(A.MODEL_CLASSES))   # noqa: E731
    return A.region_feats(reg, cfg, None, {} if cache is None else cache, tmp_path, crs_for=lambda r: UTM,
                          phase_left=phase_left, scene_cls=_Scene, loader=loader)


def test_fusion_never_across_images_or_modalities(tmp_path, monkeypatch):
    calls = []
    real = A.fuse_views
    monkeypatch.setattr(A, "fuse_views", lambda views, st: calls.append(len(views)) or real(views, st))
    a, b, c = (str(tmp_path / n) for n in ("a.tif", "b.tif", "c.tif"))
    feats = _run(tmp_path, {"ae": [a, b], "or": [c]})
    assert len(calls) == 3 and all(n == 2 for n in calls)     # 3 растра × (2 вида: flip) — по одному вызову на растр
    by = {(f["markup_type"], f["source"]) for f in feats}
    assert by == {("ae", "a.tif"), ("ae", "b.tif"), ("or", "c.tif")}
    assert {f["threshold_group"] for f in feats} == {"dfine_ae", "dfine_or"}
    # в каждом растре одно и то же место: 700 px → окна (0,60) по каждой оси для Ae 640/160 → кругов на растр > 0
    n_by_src = {s: sum(f["source"] == s for f in feats) for s in ("a.tif", "b.tif", "c.tif")}
    assert all(n >= 1 for n in n_by_src.values())


def test_merge_within_markup_count(tmp_path):
    """Общий post.merge_within_markup: dfine_ae из двух снимков одного места → 1; dfine_or отдельно (другой
    markup_type) → итог = 2 на одно место; объекты одного снимка между собой не сливаются."""
    a, b, c = (str(tmp_path / n) for n in ("a.tif", "b.tif", "c.tif"))
    feats = _run(tmp_path, {"ae": [a, b], "or": [c]})
    na = sum(f["source"] == "a.tif" for f in feats)
    merged = post.merge_within_markup(feats)
    ae = [f for f in merged if f["markup_type"] == "ae"]
    orr = [f for f in merged if f["markup_type"] == "or"]
    assert len(ae) == na                    # a и b совпадают по месту → по одному на место
    assert len(orr) == sum(f["source"] == "c.tif" for f in feats)
    assert all(f["class_name"] == "kurgany_tselye" for f in merged)   # MODEL_CLASSES[0]


def test_georeference_roundtrip_3857():
    st = A.settings({})
    T = Affine(0.5, 0, X0, 0, -0.5, Y0)
    B = np.array([[100.0, 200.0, 120.0, 220.0]])
    fs = A.image_feats(B, np.array([0.9]), np.array([2]), T, UTM, "r1", "a.tif", "or", st)
    assert len(fs) == 1 and fs[0]["class_name"] == A.MODEL_CLASSES[2]
    g = fs[0]["geom"]
    back = Transformer.from_crs("EPSG:3857", UTM, always_xy=True).transform
    ux, uy = back(g.centroid.x, g.centroid.y)
    px, py = ~T @ (ux, uy)
    assert px == pytest.approx(110.0, abs=0.05) and py == pytest.approx(210.0, abs=0.05)
    assert fs[0]["markup_type"] == "or" and fs[0]["threshold_group"] == "dfine_or"
    fwd = Transformer.from_crs(UTM, "EPSG:3857", always_xy=True).transform
    cx, cy = fwd(*(T @ (110, 210)))
    assert g.contains(Point(cx, cy))


# ── сторож времени ───────────────────────────────────────────────────────────────────────────────────────
def test_plan_level_degrades_in_order():
    st = A.settings({"dfine_aeor": {"scales": [1.0, 1.5], "flip_tta": True, "safety": 8}})
    tiles = {1.0: 100, 1.5: 50}
    # full = (100+50)*2*1*t, no_flip = 150 t, one_scale = 100 t; × 8
    t = 0.01
    assert A.plan_level(tiles, t, 1, st, 8 * 300 * t + 1)[0] == "full"
    assert A.plan_level(tiles, t, 1, st, 8 * 150 * t + 1)[0] == "no_flip"
    assert A.plan_level(tiles, t, 1, st, 8 * 100 * t + 1)[0] == "one_scale"
    assert A.plan_level(tiles, t, 1, st, 8 * 100 * t - 1)[0] == "skip"
    assert A.plan_level(tiles, None, 1, st, 10)[0] == "one_scale"   # нет замера → дешёвый проб
    assert A.plan_level(tiles, t, 1, st, 0)[0] == "skip"
    assert A.views_at("no_flip", st) == ((False,), [1.0, 1.5])
    assert A.views_at("one_scale", st) == ((False,), [1.0])


def test_time_guard_skips_and_logs(tmp_path, caplog):
    a = str(tmp_path / "a.tif")
    _Model.calls = 0
    with caplog.at_level(logging.WARNING):
        feats = _run(tmp_path, {"ae": [a]}, {"tile_s": 10.0}, phase_left=lambda: 30.0)
    assert feats == [] and _Model.calls == 0
    assert "уровень skip" in caplog.text


def test_time_guard_measures_then_degrades(tmp_path, caplog):
    a, b = str(tmp_path / "a.tif"), str(tmp_path / "b.tif")
    cache = {"t_view": {"ae": 5.0}}
    with caplog.at_level(logging.WARNING):
        _run(tmp_path, {"ae": [a, b]}, {"tile_s": None}, cache=cache, phase_left=lambda: 5.0 * 4 * 8 + 1)
    # 4 окна (700 px, 640/160), 1 модель: full=8 видов-окон×5 с×8 > остатка, no_flip=4×5×8 ≤ → no_flip
    assert "уровень no_flip" in caplog.text


def test_phase_deadline_stops_windows(tmp_path):
    a = str(tmp_path / "a.tif")
    left = iter([1e9, 1e9] + [-1.0] * 100)
    _Model.calls = 0
    _run(tmp_path, {"ae": [a]}, {"batch": 1, "flip_tta": False}, phase_left=lambda: next(left))
    assert _Model.calls < 4


class _Clock:
    """Поддельные часы модуля dfine_aeor: чтение окна медленной сцены двигает их на read_s."""
    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t


def _slow_scene(clock, read_s):
    class Slow(_Scene):
        reads = 0

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.W = self.H = 4000                 # много окон и на Or (окно 1024 px)

        def read(self, x, y, n):
            Slow.reads += 1
            clock.t += read_s
            return super().read(x, y, n)
    return Slow


def test_deadline_checked_before_each_window_not_only_batch(tmp_path, monkeypatch):
    """bot-7 13:12: самый медленный Or (033 Елизаветинское: 4 окна за 89,2 с ≈ 22 с/окно чтения+warp (детектор — заглушка), work/aeor_rss/or_033.log).
    Батч 4 окна, бюджет фазы 30 с, окно читается 23 с: проверка только между батчами прочла бы все 4 (92 с,
    перерасход 62 с). Теперь — проверка перед каждым окном по замеру: 1 окно, фаза не переехала потолок."""
    pytest.importorskip("torch")
    clock = _Clock()
    monkeypatch.setattr(A, "time", clock)
    Slow = _slow_scene(clock, 23.0)
    budget = 30.0
    cfg = {"dfine_aeor": {"enabled": True, "flip_tta": False, "tile_s": 1e-6, "batch": 4}}
    reg = _reg(tmp_path, {"or": [str(tmp_path / "o.tif")]})
    _Model.calls = 0
    A.region_feats(reg, cfg, None, {}, tmp_path, crs_for=lambda r: UTM, phase_left=lambda: budget - clock.t,
                   scene_cls=Slow, loader=lambda st, mk, root, c: ([_Model()], list(A.MODEL_CLASSES)))
    assert Slow.reads == 1 and clock.t <= budget   # до правки: 4 чтения, 92 с
    assert _Model.calls == 1                       # прочитанное окно прогнано (на прямой проход время было)


def test_deadline_guard_window_left_also_bounds_reads(tmp_path, monkeypatch):
    """Срок сторожа (window_left − reserve) тоже проверяется перед каждым окном, а не только фазовый."""
    pytest.importorskip("torch")
    clock = _Clock()
    monkeypatch.setattr(A, "time", clock)
    Slow = _slow_scene(clock, 10.0)
    guard = SimpleNamespace(window_ok=lambda: True, window_left=lambda: 45.0 - clock.t, window_reserve_s=5.0)
    cfg = {"dfine_aeor": {"enabled": True, "flip_tta": False, "tile_s": 1e-6, "batch": 2}}
    reg = _reg(tmp_path, {"or": [str(tmp_path / "o.tif")]})
    A.region_feats(reg, cfg, guard, {}, tmp_path, crs_for=lambda r: UTM, phase_left=lambda: 1e9,
                   scene_cls=Slow, loader=lambda st, mk, root, c: ([_Model()], list(A.MODEL_CLASSES)))
    assert clock.t <= 40.0 and Slow.reads == 3   # 40 − 3·10 = 10 ≤ 10 → четвёртое окно не читается


def test_phase_skips_region_via_soft_ok_and_never_waits(monkeypatch, caplog):
    """Нехватка времени: участок пропускается по soft_ok (строка «мягкий потолок»), фаза сразу выходит —
    ни ожидания, ни вызова region_feats для пропущенных; проверка phase_left перед каждым участком."""
    from expds import pipeline

    calls = []
    monkeypatch.setattr(A, "region_feats", lambda reg, *a, **k: calls.append(reg.name) or [_df("ae", reg=reg.name)])
    regs = [SimpleNamespace(name=n, rasters={"ae": [f"/nonexistent/{n}.tif"]}, utm_json=None) for n in ("r1", "r2", "r3")]
    soft = iter([True, False, True])

    class G:
        soft_ceiling = 100.0
        soft_skipped = []

        def ok(self, need=0):
            return True

        def elapsed(self):
            return 0.0

        def soft_need_s(self, phase):
            return 0.0

        def soft_ok(self, need, phase=None, what=""):
            r = next(soft)
            if not r:
                self.soft_skipped.append(what)
                logging.getLogger("expds.pipeline").warning("мягкий потолок: %s пропущено (%s)", phase, what)
            return r
    state = SimpleNamespace(note={"dfine_aeor": {"ok": True}}, add=lambda fs: None)
    with caplog.at_level(logging.INFO):
        pipeline._dfine_aeor_phase(regs, {"dfine_aeor": {"enabled": True}}, G(), state, ROOT, {})
    assert calls == ["r1"] and "мягкий потолок: dfine_aeor" in caplog.text
    assert "dfine_aeor n_det=1 total" in caplog.text
    # срок фазы исчерпан до участка — выход до region_feats
    calls.clear()
    monkeypatch.setattr(pipeline.time, "monotonic", iter([0.0] + [1e6] * 50).__next__)
    pipeline._dfine_aeor_phase(regs, {"dfine_aeor": {"enabled": True}}, SimpleNamespace(ok=lambda n=0: True,
                               elapsed=lambda: 0.0), state, ROOT, {})
    assert calls == []


# ── постобработка v3 не трогает dfine_ae / dfine_or ──────────────────────────────────────────────────────
def _df(markup, cls="gorodishcha", conf=0.5, reg="r1"):
    return {"geom": Point(X0, Y0).buffer(4), "class_name": cls, "confidence": conf, "region_name": reg,
            "markup_type": markup, "threshold_group": A.GROUPS[markup], "source": "c.tif", "scene_drel": 0.99}


def test_variant_r_does_not_drop_dfine_or():
    sp = {"geom": Point(X0, Y0).buffer(4), "class_name": "gorodishcha", "confidence": 0.5, "region_name": "r1",
          "markup_type": "sp_or", "source": "s.tif"}
    out = sp_dfine.drop_groups([_df("or"), _df("ae"), sp], ["sp_or", "sp_focal", "sp_arch", "sp_focal2"], ["r1"])
    assert [f["markup_type"] for f in out] == ["or", "ae"]
    assert all(f["markup_type"] != sp_dfine.MARKUP for f in (_df("or"), _df("ae")))


def test_frame_prior_does_not_touch_dfine():
    fs = [_df("or"), _df("ae")]
    out = post.frame_prior([dict(f) for f in fs], {})
    assert [f["confidence"] for f in out] == [0.5, 0.5]


def test_verify_keeps_dfine_conf(tmp_path, monkeypatch):
    from expds import crop_model

    w = tmp_path / "w.pt"
    w.write_bytes(b"0")
    cfg = _cfg()
    cfg["verify"] = dict(cfg["verify"], weights=str(w))
    monkeypatch.setattr(crop_model, "_predict", lambda *a, **k: pytest.fail("verify выбрал dfine"))
    fs = [_df("or", "kurgany_tselye"), _df("ae", "kurgany_tselye")]
    assert crop_model.maybe_verify(fs, {"r1": object()}, cfg) is None
    assert [f["confidence"] for f in fs] == [0.5, 0.5]


def test_thresholds_by_group_apply_to_dfine():
    """Порог группы dfine_or главнее глобального по классу (pipeline/post: threshold_group раньше markup_type)."""
    assert len(post.apply_thresholds([_df("or")], {"gorodishcha": 0.99}, 0.15, {"dfine_or": {"gorodishcha": 0.4}})) == 1
    assert len(post.apply_thresholds([_df("or")], {"gorodishcha": 0.1}, 0.15, {"dfine_or": {"gorodishcha": 0.6}})) == 0


# ── пайплайн и сборка ────────────────────────────────────────────────────────────────────────────────────
def test_pipeline_gated_by_flag_lazy_import_and_order():
    from expds import pipeline

    src = inspect.getsource(pipeline.run)
    assert src.count('(cfg.get("dfine_aeor") or {}).get("enabled")') == 2
    i_sp, i_ae, i_gg = (src.index(s) for s in ("_sp_dfine_phase(", "_dfine_aeor_phase(", 'cfg.get("gor_gate")'))
    assert i_sp < i_ae < i_gg
    top = [l for l in (ROOT / "expds" / "pipeline.py").read_text(encoding="utf-8").splitlines()
           if l.startswith(("from .", "import ")) and "dfine_aeor" in l]
    assert not top
    assert "import_dfine" in inspect.getsource(sp_dfine.load_models)
    assert "transformers" not in (ROOT / "expds" / "dfine_aeor.py").read_text(encoding="utf-8").split('"""', 2)[2] \
        .replace("transformers 5.21.0", "")


def test_phase_unready_logs_total_zero(caplog):
    from expds import pipeline

    state = SimpleNamespace(note={"dfine_aeor": {"ok": False, "problems": ["нет весов"]}},
                            add=lambda fs: pytest.fail("add при неготовности"))
    with caplog.at_level(logging.INFO):
        assert pipeline._dfine_aeor_phase([], {"dfine_aeor": {"enabled": True}}, None, state, ROOT) == 0
    assert "dfine_aeor n_det=0 total" in caplog.text


def test_phase_logs_per_region_and_total(monkeypatch, caplog):
    from expds import pipeline

    added = []
    state = SimpleNamespace(note={"dfine_aeor": {"ok": True}}, add=added.extend)
    regs = [SimpleNamespace(name=n, rasters={"ae": [f"/nonexistent/{n}.tif"]}, utm_json=None) for n in ("r1", "r2")]
    monkeypatch.setattr(A, "region_feats", lambda reg, *a, **k: [_df("ae", reg=reg.name)])
    guard = SimpleNamespace(ok=lambda need=0: True, elapsed=lambda: 0.0)
    with caplog.at_level(logging.INFO):
        n = pipeline._dfine_aeor_phase(regs, {"dfine_aeor": {"enabled": True}}, guard, state, ROOT, {})
    assert n == 2 and len(added) == 2
    assert "dfine_aeor n_det=1 region=r1" in caplog.text and "dfine_aeor n_det=1 region=r2" in caplog.text
    assert "dfine_aeor n_det=2 total" in caplog.text


def test_not_packaged_when_off():
    spec = importlib.util.spec_from_file_location("b6_build", ROOT / "scripts" / "build_submission.py")
    bs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bs)
    assert "dfine_aeor" in bs.FLAG_MODULES
    rel = {p.relative_to(ROOT).as_posix() for p in bs.collect()}
    assert "expds/dfine_aeor.py" not in rel


def test_check_submission_knows_marker():
    src = (ROOT / "scripts" / "check_submission.py").read_text(encoding="utf-8")
    assert "DFINE_AEOR_UNAVAILABLE" in src and A.UNAVAILABLE == "DFINE_AEOR_UNAVAILABLE"


def test_cap_per_group():
    fs = [dict(_df("ae"), confidence=c) for c in (0.1, 0.9, 0.5)] + [_df("or")] + \
         [{"threshold_group": "sp_or", "confidence": 0.1}]
    out = A.cap(fs, 2)
    assert sorted(f["confidence"] for f in out if f.get("threshold_group") == "dfine_ae") == [0.5, 0.9]
    assert len(out) == 4


# ── правки bot-7 (12:31): ключ склейки, отдельная цепочка, check_submission ──────────────────────────────
def _poly(x, y, r=6.0):
    return Point(x, y).buffer(r, 32)


def _old_feats():
    """Прежние группы v3 на одном месте с dfine: ae (база), li, SpOr-архитектура, Ae-фортификация."""
    X, Y = 4_200_000.0, 7_500_000.0
    out = []
    for i in range(6):
        out.append({"region_name": "r1", "markup_type": "ae", "class_name": "kurgany_tselye", "source": f"ae{i % 2}.tif",
                    "confidence": 0.6 + 0.05 * i, "geom": _poly(X + 30 * i, Y)})
        out.append({"region_name": "r1", "markup_type": "li", "class_name": "kurgany_povrezhdennye", "source": "li.tif",
                    "confidence": 0.7, "geom": _poly(X + 30 * i, Y + 40)})
    out.append({"region_name": "r1", "markup_type": "sp_or", "threshold_group": "sp_or", "class_name": "arkhitektury",
                "source": "sp.tif", "confidence": 0.8, "geom": _poly(X, Y + 90, 10)})
    out.append({"region_name": "r1", "markup_type": "ae", "class_name": "fortifikatsii", "source": "ae0.tif",
                "confidence": 0.9, "geom": _poly(X + 60, Y + 140, 15)})
    return out


def _dfine_feats():
    X, Y = 4_200_000.0, 7_500_000.0
    out = []
    for i in range(6):   # те же места, что у старых ae/li/arch — склеились бы при общем ключе
        for src in ("ae0.tif", "ae1.tif"):
            out.append({"region_name": "r1", "markup_type": "ae", "threshold_group": "dfine_ae", "source": src,
                        "class_name": "kurgany_tselye", "confidence": 0.95, "geom": _poly(X + 30 * i + 2, Y)})
    out.append({"region_name": "r1", "markup_type": "or", "threshold_group": "dfine_or", "source": "or.tif",
                "class_name": "arkhitektury", "confidence": 0.9, "geom": _poly(X, Y + 90, 9)})
    return out


def _ser(fs):
    return [json.dumps({k: (v.wkb_hex if k == "geom" else v) for k, v in sorted(f.items())}, ensure_ascii=False,
                       sort_keys=True, default=str) for f in fs]


def test_merge_key_isolates_dfine_groups():
    old, new = _old_feats(), _dfine_feats()
    merged = post.merge_within_markup(old + new)
    d_ae = [f for f in merged if f.get("threshold_group") == "dfine_ae"]
    assert len(d_ae) == 6                                   # ae0+ae1 dfine → по одному на место, только между собой
    assert all(f["source"] in ("ae0.tif", "ae1.tif") and f["confidence"] == 0.95 for f in d_ae)
    rest = [f for f in merged if f.get("threshold_group") not in post.ISOLATED_GROUPS]
    assert _ser(rest) == _ser(post.merge_within_markup(old))   # прежние группы — как без dfine
    union = post.merge_within_markup(old + new, isolate=())  # межисточниковое объединение — только по опции
    assert len([f for f in union if f["markup_type"] == "ae" and f["class_name"] == "kurgany_tselye"]) == 6
    assert not any(f.get("threshold_group") == "dfine_ae" and f["confidence"] < 0.95 for f in union)
    assert A.DEFAULTS["merge_union"] is False and "merge_union" not in _cfg()["dfine_aeor"]


def test_flag_on_old_groups_byte_identical_to_off():
    """Флаг вкл. + объекты dfine: прежние объекты на выходе finish/safe_finish байт-в-байт как при флаге выкл."""
    from expds import watchdog

    old, new = _old_feats(), _dfine_feats()
    off = _cfg()
    on = _cfg()
    on["dfine_aeor"] = {"enabled": True}
    on.setdefault("thresholds_by_markup", {}).update(_thr(0.3))
    for fin in (watchdog.finish, watchdog.safe_finish):
        ref = fin(list(old), off)
        got = fin(list(old) + list(new), on)
        kept_old = [f for f in got if f.get("threshold_group") not in post.ISOLATED_GROUPS]
        assert _ser(kept_old) == _ser(ref)
        assert got[:len(ref)] == kept_old                   # порядок: v3, затем dfine
        assert sum(f.get("threshold_group") == "dfine_ae" for f in got) == 6
        assert sum(f.get("threshold_group") == "dfine_or" for f in got) == 1
    on["dfine_aeor"]["merge_union"] = True                  # опция объединения меняет общую цепочку — потому выкл.
    assert watchdog._aeor_split(old + new, on) is None


def test_aeor_tail_respects_feature_limit():
    from expds import watchdog

    cfg = _cfg()
    cfg["dfine_aeor"] = {"enabled": True}
    cfg.setdefault("thresholds_by_markup", {}).update(_thr(0.3))
    assert watchdog._aeor_finish(_dfine_feats(), cfg, 0) == []
    assert len(watchdog._aeor_finish(_dfine_feats(), cfg, 3)) == 3


def _cs():
    spec = importlib.util.spec_from_file_location("b6_cs", ROOT / "scripts" / "check_submission.py")
    cs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cs)
    cs.RESULTS.clear()
    return cs


def _zip(tmp_path, cfg_text, extra=()):
    import zipfile
    zp = tmp_path / "s.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("configs/default.toml", cfg_text)
        z.writestr("expds/sp_dfine.py", "")
        for n in extra:
            z.writestr(n, "{}")
    return zipfile.ZipFile(zp)


def _args(tmp_path, **kw):
    return SimpleNamespace(weights=True, dfine_aeor_root=str(tmp_path / "none"), workdir=str(tmp_path), python="python3",
                           cpus="0,1", allow_net=False, timeout=60, threads=2, dfine_region=None, **kw)


def test_check_submission_flag_off_requires_nothing_new(tmp_path):
    cs = _cs()
    text = (ROOT / "configs" / "default.toml").read_text(encoding="utf-8")
    cfg = tomllib.loads(text)
    z = _zip(tmp_path, text)
    cs.check_dfine_aeor_weights(z, cfg, _args(tmp_path))
    assert cs.RESULTS == []
    cs.dfine_lines(z, "sp_dfine n_det=5 region=x\n")
    assert not [r for r in cs.RESULTS if "dfine_aeor" in r[2]]
    cs.RESULTS.clear()
    cs.check_dfine(z, cfg, _args(tmp_path))                  # sp_dfine/gor_gate/dfine_aeor выкл. → проб нет
    assert cs.RESULTS == []
    cs.check_config_static(z)
    assert not [r for r in cs.RESULTS if "dfine_aeor" in r[2] or "DFINE_AEOR" in r[2]]


def test_check_submission_smoke_needs_weights(tmp_path):
    cs = _cs()
    cfg = {"dfine_aeor": {"enabled": True, "weights": {"ae": ["models/dfine_ae"], "or": ["models/dfine_or"]}}}
    z = _zip(tmp_path, "[dfine_aeor]\nenabled = true\n")
    cs.check_dfine_aeor_weights(z, cfg, _args(tmp_path))
    assert [r[0] for r in cs.RESULTS] == ["FAIL", "FAIL"]
    assert all("нужны веса" in r[2] for r in cs.RESULTS)
    assert cs.DFINE_AEOR_KNOWN["ae"]["region"] == "181_Гнёздово_Лесная_3"
    assert cs.DFINE_AEOR_KNOWN["or"]["region"] == "185_Изборск"
    cs.RESULTS.clear()
    cs.check_dfine_aeor_weights(z, cfg, SimpleNamespace(**{**vars(_args(tmp_path)), "weights": False}))
    assert [r[0] for r in cs.RESULTS] == ["WARN"] and "смоук не выполнен" in cs.RESULTS[0][2]


def test_check_submission_smoke_verdict_reaches_report(tmp_path, monkeypatch):
    """bot-6 v4: первый настоящий прогон с весами bot-9 упал TypeError — report() звался с (статус, текст) без
    раздела. Проба с данными и весами должна дойти до report(статус, "dfine", текст): n_det>0 — PASS, 0 — FAIL."""
    cs = _cs()
    reg = tmp_path / "train" / "185_Изборск" / "05_Изборск_Or_ортофото"
    reg.mkdir(parents=True)
    (reg / "01_Изборск_Or_ортофото.tif").write_bytes(b"")
    cfg = {"dfine_aeor": {"enabled": True, "markups": ["or"], "weights": {"or": ["models/dfine_or"]}}}
    z = _zip(tmp_path, "", [f"models/dfine_or/{f}" for f in ("config.json", "model.safetensors")])
    args = SimpleNamespace(**{**vars(_args(tmp_path)), "dfine_aeor_root": str(tmp_path / "train")})
    for n, want in ((94, "PASS"), (0, "FAIL")):
        out = {"ok": True, "n_det": n, "max": 0.46, "windows": [1, 1], "models": 1, "min_score": 0.05}
        monkeypatch.setattr(cs.subprocess, "run", lambda *a, _o=out, **k: SimpleNamespace(
            stdout="DFINE_AEOR_JSON=" + json.dumps(_o), stderr="", returncode=0))
        cs.RESULTS.clear()
        cs.check_dfine_aeor_weights(z, cfg, args)
        assert [(r[0], r[1]) for r in cs.RESULTS] == [(want, "dfine")] and "185_Изборск" in cs.RESULTS[0][2]


def test_check_submission_missing_train_data_warns_not_pass(tmp_path):
    """bot-7 13:12: веса в архиве есть, а данных трейна для смоука на машине нет — явный WARN «смоук не выполнен»
    на каждую модальность, ни одного PASS; итог без FAIL (не дефект архива), но WARN виден."""
    cs = _cs()
    cfg = {"dfine_aeor": {"enabled": True, "weights": {"ae": ["models/dfine_ae"], "or": ["models/dfine_or"]}}}
    z2 = _zip(tmp_path, "", [f"models/dfine_{m}/{f}" for m in ("ae", "or") for f in ("config.json", "model.safetensors")])
    cs.check_dfine_aeor_weights(z2, cfg, _args(tmp_path))
    assert [r[0] for r in cs.RESULTS] == ["WARN", "WARN"]
    assert all(r[2].startswith("смоук не выполнен") and "опорного участка" in r[2] for r in cs.RESULTS)
    cs.RESULTS.clear()
    cs.check_dfine_aeor_weights(z2, {"dfine_aeor": {**cfg["dfine_aeor"], "markups": ["xx"]}}, _args(tmp_path))
    assert [r[0] for r in cs.RESULTS] == ["WARN"] and "смоук не выполнен" in cs.RESULTS[0][2]


def test_check_submission_aeor_verdicts():
    cs = _cs()
    assert cs.dfine_aeor_verdict("ae", {"ok": True, "n_det": 0, "max": 0, "windows": [1, 1]})[0] == "FAIL"
    assert cs.dfine_aeor_verdict("or", {"ok": True, "n_det": 3, "max": 0.4, "windows": [1, 1]})[0] == "PASS"
    assert cs.dfine_aeor_verdict("or", {"ok": False, "err": "x"})[0] == "FAIL"
    log = "dfine_aeor n_det=0 region=181_Гнёздово_Лесная_3\ndfine_aeor n_det=0 region=other\n"
    assert [r[0] for r in cs.dfine_aeor_ndet_verdict(log)] == ["FAIL"]
    src = (ROOT / "scripts" / "check_submission.py").read_text(encoding="utf-8")
    assert '("expds/dfine_aeor.py", aeor)' in src             # проба импорта в чистом процессе — общая с sp_dfine


def test_scene_uses_overview_only_for_heavy_downsampling(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    p = tmp_path / "o.tif"
    a = np.full((3, 2048, 2048), 120, dtype=np.uint8)
    with rasterio.open(p, "w", driver="GTiff", width=2048, height=2048, count=3, dtype="uint8", crs=UTM,
                       transform=from_origin(X0, Y0, 0.02, 0.02), tiled=True) as d:
        d.write(a)
        d.build_overviews([2, 4, 8, 16], rasterio.enums.Resampling.average)
    sc = A.Scene(p, UTM, 0.5)
    assert sc.overview_level == 2 and sc.W == round(2048 * 0.02 / 0.5)       # 0,02 × 8 = 0,16 ≤ 0,25; × 16 = 0,32
    im = sc.read(0, 0, 64)
    assert im.shape == (64, 64, 3) and int(im[10, 10, 0]) == 120
    sc.close()
    sc = A.Scene(p, UTM, 0.04)                                                # слабое уменьшение — как sp_dfine.Scene
    assert sc.overview_level is None
    sc.close()


def test_scene_geographic_rgba_not_1x1(tmp_path):
    """bot-9 13:37: Or 008_Герите (EPSG:4326, пиксель ~1e-7°, RGBA, обзоры 2…511) у bot-9 прочитан 1×1 px.
    Scene считает размер через calculate_default_transform в UTM (метры), а не res в градусах: сцена = охват/0,5 м,
    альфа не читается, сетка окон покрывает сцену (замер на настоящем 008: work/aeor_v4/raster_probe.log)."""
    import rasterio
    from rasterio.transform import from_origin
    p = tmp_path / "g.tif"
    n, deg = 2048, 1e-6                                       # 57° с. ш.: ≈0,111 м по широте, ≈0,061 м по долготе
    a = np.full((4, n, n), 90, dtype=np.uint8)
    a[3] = 255
    with rasterio.open(p, "w", driver="GTiff", width=n, height=n, count=4, dtype="uint8", crs="EPSG:4326",
                       transform=from_origin(45.1, 57.0, deg, deg), tiled=True, photometric="RGB") as d:
        d.write(a)
        d.build_overviews([2, 4, 8, 16], rasterio.enums.Resampling.average)
    crs = sp_dfine.utm_for(p, None)
    assert crs == "EPSG:32638"
    sc = A.Scene(p, crs, 0.5)
    w_m, h_m = n * deg * 111320 * math.cos(math.radians(57)), n * deg * 110540
    assert sc.overview_level == 0 and sc.bands == [1, 2, 3]   # 0,08 м × 2 ≤ 0,25; альфа (4) не читается
    assert abs(sc.W - w_m / 0.5) / (w_m / 0.5) < 0.15 and abs(sc.H - h_m / 0.5) / (h_m / 0.5) < 0.15
    assert abs(abs(sc.T.a) - 0.5) < 1e-9 and min(sc.W, sc.H) > 100
    wins = A.windows(sc.W, sc.H, 128, 32)
    assert max(x for x, _ in wins) + 128 >= sc.W and max(y for _, y in wins) + 128 >= sc.H
    im = sc.read(sc.W // 2 - 32, sc.H // 2 - 32, 64)
    assert im.shape == (64, 64, 3) and int(im[32, 32, 0]) == 90
    sc.close()


def test_thresholds_file_aliases_and_gor_off(tmp_path):
    cfg = _cfg()
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"ae_dfine": {"kurgany_tselye": 0.4}, "or_dfine": {"gorodishcha": 0.2, "arkhitektury": 0.5}}))
    cfg["dfine_aeor"] = {"enabled": True, "thresholds_file": str(p)}
    A.load_thresholds_file(cfg, ROOT)
    tbm = cfg["thresholds_by_markup"]
    assert tbm["dfine_ae"]["kurgany_tselye"] == 0.4 and tbm["dfine_or"]["arkhitektury"] == 0.5
    assert tbm["dfine_or"]["gorodishcha"] == 1.01             # TOML главнее OOF
    f = {"class_name": "gorodishcha", "markup_type": "or", "threshold_group": "dfine_or", "confidence": 0.999}
    assert post.apply_thresholds([f], cfg.get("thresholds", {}), 0.15, tbm) == []


def test_recipe_autocast_only_no_half():
    """Вход как у sp_dfine/обучения bot-9: float32 RGB/255 без нормализации, flip-TTA, NMS 0,5; .half() нет."""
    import torch

    seen = []

    class Rec:
        def __call__(self, pixel_values):
            seen.append(pixel_values.clone())
            n = pixel_values.shape[0]
            return SimpleNamespace(logits=torch.full((n, 4, 5), -9.0), pred_boxes=torch.full((n, 4, 4), 0.5))

    class Sc:
        W = H = 64

        def read(self, x, y, n):
            a = np.zeros((n, n, 3), dtype=np.uint8)
            a[..., 0], a[..., 1], a[..., 2] = 255, 51, 0
            a[:, : n // 2, 0] = 102                   # левая половина другая — видно отражение
            return a

    st = A.settings({"dfine_aeor": {"tile": 64, "overlap": 0, "batch": 2}})
    assert st["nms_iou"] == 0.5 and st["flip_tta"] is True and st["fp16"] is True
    A.detect_scene([Rec()], Sc(), st, "ae", [False, True], torch.device("cpu"))
    assert len(seen) == 2 and all(t.dtype == torch.float32 for t in seen)
    px = seen[0][0]
    assert float(px[0, 0, 63]) == 1.0 and float(px[0, 0, 0]) == pytest.approx(0.4) and float(px[1, 0, 0]) == pytest.approx(0.2)
    assert float(px[2].max()) == 0.0
    assert torch.equal(seen[1], seen[0].flip(3))    # второй вид — горизонтальное отражение того же батча
    src = (ROOT / "expds" / "dfine_aeor.py").read_text(encoding="utf-8")
    assert ".half()" not in src and "torch.autocast(" in src
