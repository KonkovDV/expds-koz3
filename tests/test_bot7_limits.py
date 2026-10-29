"""bot-7: последний рубеж лимитов выдачи в writer.write_geojson (≤ 20000 объектов, ≤ 3000 вершин, только Polygon)."""
import json
import math

from shapely.geometry import Point, Polygon

from expds import writer
from expds.constants import MAX_FEATURES, MAX_VERTICES


def _f(geom, cls="finds_points", conf=0.5):
    return {"geom": geom, "class_name": cls, "confidence": conf, "region_name": "r", "markup_type": "maggr"}


def _ring(n, r=50.0, x0=4_000_000.0, y0=7_000_000.0):
    return Polygon([(x0 + r * math.cos(2 * math.pi * k / n), y0 + r * math.sin(2 * math.pi * k / n)) for k in range(n)])


def test_normal_output_untouched(tmp_path):
    feats = [_f(Point(4_000_000 + 10 * i, 7_000_000).buffer(1.0)) for i in range(5)]
    p = writer.write_geojson(feats, tmp_path / "result.geojson")
    fc = json.loads(p.read_text(encoding="utf-8"))
    assert fc == writer.to_feature_collection(feats)
    assert writer.validate(p)["ok"]


def test_too_many_vertices_dropped(tmp_path):
    big = _ring(MAX_VERTICES + 50)            # в обход finalize/limit_vertices
    ok = _ring(100, x0=4_001_000.0)
    p = writer.write_geojson([_f(big), _f(ok)], tmp_path / "result.geojson")
    fc = json.loads(p.read_text(encoding="utf-8"))
    assert len(fc["features"]) == 1 and fc["features"][0]["properties"]["fid"] == 0
    assert writer.validate(p)["ok"]


def test_feature_cap(tmp_path):
    sq = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [list(map(list, sq.exterior.coords))]},
         "properties": {"fid": k, "class_name": "finds_points"}} for k in range(MAX_FEATURES + 7)]}
    out = writer.enforce_limits(fc)
    assert len(out["features"]) == MAX_FEATURES
    assert [f["properties"]["fid"] for f in out["features"][:3]] == [0, 1, 2]


def test_safe_finish_fallback_keeps_stack6(monkeypatch):
    """bot-7: при сбое finish() запасная цепочка тоже делает sota_stack6 (multipole + filter_finds_points); сбой шага не роняет её."""
    from expds import post, watchdog

    calls = []
    monkeypatch.setattr(post, "merge_within_markup", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(post, "multipole_object_poly", lambda fs, spec: calls.append("mp") or fs)
    def bad(fs, spec):
        calls.append("ff"); raise RuntimeError("GEOS")
    monkeypatch.setattr(post, "filter_finds_points_regimes", bad)
    f = _f(Point(4_000_000, 7_000_000).buffer(1.0), cls="kurgany_tselye", conf=0.9)
    cfg = {"thresholds": {}, "default_threshold": 0.0, "post": {"multipole_object_poly": True, "filter_finds_points": True}}
    out = watchdog.safe_finish([f], cfg)
    assert calls == ["mp", "ff"] and len(out) == 1


def test_candidate_twins_only_on_and_in_fallback(monkeypatch):
    """bot-7: в конфиге кандидата v2 kurgan_twins_only = true (приёмка bot-10), полный twins выкл.; запасная цепочка его тоже делает (dedup ae/li = 0)."""
    import tomllib
    from pathlib import Path

    from expds import post, watchdog

    cfg = tomllib.loads((Path(__file__).resolve().parents[1] / "configs" / "default.toml").read_text(encoding="utf-8"))
    assert cfg["post"].get("kurgan_twins_only") is True and not cfg["post"].get("kurgan_twins")
    seen = {}
    monkeypatch.setattr(post, "merge_within_markup", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(post, "kurgan_cross_modal_twins", lambda fs, spec: seen.setdefault("spec", spec) and fs)
    f = _f(Point(4_000_000, 7_000_000).buffer(1.0), cls="kurgany_tselye", conf=0.9)
    watchdog.safe_finish([f], {"thresholds": {}, "default_threshold": 0.0, "post": {"kurgan_twins_only": {"dedup_ae_vs_li_m": 3.0}}})
    assert seen["spec"]["dedup_ae_vs_li_m"] == 0.0
