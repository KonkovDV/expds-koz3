"""bot-8: инъекции отказов — один сбой не должен обнулять или урезать весь прогон."""
import json
import math
from types import SimpleNamespace

from conftest import X0, Y0
from shapely.geometry import box

from expds import writer


def _f(cls="kurgany_tselye", d=0.0, conf=0.9, mt="li"):
    return {"geom": box(X0 + d, Y0, X0 + d + 5, Y0 + 5), "class_name": cls, "confidence": conf,
            "markup_type": mt, "region_name": "r"}


def test_geophys_exception_in_one_region_does_not_skip_lidar(tmp_path, monkeypatch):
    """pipeline.py:862-867 — геофизика без try: сбой одного региона не должен отменять лидар/оптику."""
    from expds import pipeline

    out = tmp_path / "o" / "result.geojson"
    regs = [SimpleNamespace(name=n, las=[], rasters={}, utm_json=None, boundaries=[], noise=[]) for n in ("a", "b")]
    calls = {"li": 0}

    def geo(reg, *_a, **_k):
        if reg.name == "a":
            raise ValueError("битый регион")
        return [_f("object_poly", mt="maggr")]

    def li(*_a, **_k):
        calls["li"] += 1
        return [_f(d=100)]

    monkeypatch.setattr(pipeline, "scan_root", lambda _p: regs)
    monkeypatch.setattr(pipeline, "geophys_feats", geo)
    monkeypatch.setattr(pipeline, "lidar_feats", li)
    monkeypatch.setattr(pipeline, "optical_feats", lambda *_a, **_k: [])
    monkeypatch.setenv("EXPDS_TMP", str(tmp_path / "tmp"))
    cfg = tmp_path / "t.toml"
    cfg.write_text("[time]\nwatchdog = false\n[verify]\nenabled = false\n", encoding="utf-8")
    try:
        pipeline.run(tmp_path, out, cfg)
    except Exception:
        pass
    assert calls["li"] == 2, "лидар пропущен из-за сбоя геофизики одного региона"
    classes = sorted(f["properties"]["class_name"] for f in json.loads(out.read_text())["features"])
    assert classes == ["kurgany_tselye", "kurgany_tselye", "object_poly"]


def test_checkpoint_survives_failing_postprocessing(tmp_path, monkeypatch):
    """watchdog.py:140-147 — исключение в finish() не должно оставлять ответ пустым."""
    from expds import post
    from expds.watchdog import RunState, checkpoint

    def boom(_feats, *a, **k):
        raise RuntimeError("GEOS упал в склейке")

    monkeypatch.setattr(post, "merge_within_markup", boom)
    st = RunState(tmp_path / "result.geojson", {"thresholds": {}, "default_threshold": 0.0})
    st.add([_f(), _f(d=20)])
    try:
        checkpoint(st, "конец")
    except Exception:
        pass
    p = tmp_path / "result.geojson"
    assert p.exists() and writer.validate(p)["ok"]
    assert len(json.loads(p.read_text())["features"]) == 2


def test_writer_never_emits_nan(tmp_path):
    """writer.py:23-67 — NaN в координатах даёт невалидный JSON («NaN»); такой объект надо отбросить."""
    import numpy as np
    import shapely
    c = shapely.get_coordinates(box(X0, Y0, X0 + 3, Y0 + 3))
    c[1, 0] = math.nan                                   # средняя вершина, кольцо остаётся замкнутым
    g = shapely.set_coordinates(box(X0, Y0, X0 + 3, Y0 + 3), c)
    bad = {"geom": g, "class_name": "kurgany_tselye",
           "confidence": 0.5, "region_name": "r"}
    p = writer.write_geojson([bad, _f()], tmp_path / "result.geojson")
    txt = p.read_text()
    assert "NaN" not in txt
    json.loads(txt, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    assert writer.validate(p)["ok"]


def test_safe_finish_fallback_keeps_bot7_micro_circles(tmp_path, monkeypatch):
    """watchdog.safe_finish: при сбое finish() запасная цепочка тоже делает [post.micro_circles] (пакет bot-7)."""
    from shapely.geometry import Point

    from expds import post
    from expds.watchdog import safe_finish

    def boom(_feats, *a, **k):
        raise RuntimeError("GEOS упал в склейке")

    monkeypatch.setattr(post, "merge_within_markup", boom)
    big = Point(X0 + 50, Y0 + 50).buffer(2.0 / post._ground_scale(Y0), quad_segs=8)
    f = {"geom": big, "class_name": "kurgany_tselye", "confidence": 0.9, "markup_type": "li", "region_name": "r"}
    cfg = {"thresholds": {}, "default_threshold": 0.0,
           "post": {"micro_circles": {"radius_m": {"kurgany_tselye": 0.5}}}}
    out = safe_finish([f], cfg)
    assert len(out) == 1
    assert out[0]["geom"].area < big.area * 0.1        # 0,5 м вместо 2 м: площадь в 16 раз меньше
    assert out[0]["geom"].centroid.distance(big.centroid) < 1e-6


def test_safe_finish_fallback_keeps_bot7_rerank_arch(tmp_path, monkeypatch):
    """watchdog.safe_finish на bot7-candidate: при сбое finish() архитектура идёт через [rerank_arch] со своими порогами."""
    import importlib.util

    import pytest
    if importlib.util.find_spec("expds.rerank_arch") is None:
        pytest.skip("нет expds.rerank_arch (ветка bot7-bundle)")
    from test_bot7_rerank_arch import _cfg, _feats, _key, _model

    from expds import post, watchdog

    fs = _feats()
    rr = _model(tmp_path, {"sp_arch": 0.0})
    good = watchdog.finish(fs, _cfg(rerank_arch=rr))

    def boom(_feats, *a, **k):
        raise RuntimeError("GEOS упал в склейке")

    monkeypatch.setattr(post, "merge_within_markup", boom)
    fb = watchdog.safe_finish(fs, _cfg(rerank_arch=rr))
    arch = [f for f in fb if f["class_name"] == "arkhitektury"]
    assert len(arch) == 20 and all("conf_stack" in f for f in arch)   # базовый порог 1.01 не применён
    assert _key(fb) == _key(good)
