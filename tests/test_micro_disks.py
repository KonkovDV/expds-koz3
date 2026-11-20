"""Микродиски [post.micro_disks]: отдельный шаг для архитектуры, площадных объектов и фортификаций.

Проверяется то, на чём держится решение включить шаг:
- без ключа выход finish() тот же до бита;
- курганы и число объектов не меняются, меняется только геометрия трёх классов;
- диск переживает запись (округление до 0,01) на широтах участков конкурса;
- по официальной метрике засчитанный полигон остаётся засчитанным, а промах
  «внутренняя точка внутри эталона, покрытие < 0,5» становится попаданием.
"""
import importlib.util
import json
import math
import tomllib

import pytest
from shapely.geometry import Point, Polygon, box

from conftest import ROOT
from expds import post, watchdog, writer

DISKS = {"radius_m": {"arkhitektury": 0.05, "object_poly": 0.05, "fortifikatsii": 0.05}, "anchor": "rp"}


def _merc(lon: float, lat: float) -> tuple[float, float]:
    return lon * 20037508.34 / 180, math.log(math.tan((90 + lat) * math.pi / 360)) * 6378137.0


def _feat(cls, geom, conf=0.9, mt="ae"):
    return {"class_name": cls, "geom": geom, "confidence": conf, "region_name": "r", "markup_type": mt,
            "threshold_group": mt, "source": "s"}


def _scene(lat: float):
    x, y = _merc(37.0, lat)
    k = 1 / math.cos(math.radians(lat))
    return [
        _feat("fortifikatsii", box(x - 20 * k, y - 1 * k, x + 20 * k, y + 1 * k)),
        _feat("arkhitektury", Point(x + 100 * k, y).buffer(3 * k, quad_segs=8), mt="sp_or"),
        _feat("object_poly", box(x + 200 * k, y, x + 201.5 * k, y + 0.8 * k), mt="etgr"),
        _feat("kurgany_povrezhdennye", Point(x + 300 * k, y).buffer(2 * k, quad_segs=8), mt="li"),
        _feat("gorodishcha", Point(x + 400 * k, y).buffer(30 * k, quad_segs=8), mt="li"),
    ], k


def _cfg() -> dict:
    """Финальная конфигурация (configs/default.toml, тот же файл, что в сданном архиве)."""
    return tomllib.loads((ROOT / "configs" / "default.toml").read_text(encoding="utf-8"))


def _without_disks() -> dict:
    """Та же конфигурация без шага [post.micro_disks] — контроль для сравнения."""
    cfg = _cfg()
    cfg["post"].pop("micro_disks")
    return cfg


def test_final_config_enables_disks_for_three_classes():
    assert _cfg()["post"]["micro_disks"] == DISKS


def test_without_key_finish_is_unchanged():
    cfg = _without_disks()
    assert "micro_disks" not in cfg["post"]
    feats, _ = _scene(55.0)
    a = writer.to_feature_collection(watchdog.finish([dict(f) for f in feats], cfg))
    cfg2 = _without_disks()
    cfg2["post"]["micro_disks"] = None          # не словарь — шаг не включается
    b = writer.to_feature_collection(watchdog.finish([dict(f) for f in feats], cfg2))
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


@pytest.mark.parametrize("lat", [40.0, 45.0, 52.0, 55.0, 60.0, 64.0, 68.0])
def test_only_three_classes_change_and_disks_survive_writing(lat):
    feats, k = _scene(lat)
    cfg = _cfg()
    base = _without_disks()
    for c in (cfg, base):
        c["thresholds_by_markup"] = {}
        c["default_threshold"] = 0.0
        c["rerank_arch"] = {"enabled": False}
    out_a = watchdog.finish([dict(f) for f in feats], base)
    out_b = watchdog.finish([dict(f) for f in feats], cfg)
    assert len(out_a) == len(out_b) == len(feats)
    by_a = {f["class_name"]: f for f in out_a}
    by_b = {f["class_name"]: f for f in out_b}
    for cls in ("kurgany_povrezhdennye", "gorodishcha"):
        assert by_a[cls]["geom"].equals_exact(by_b[cls]["geom"], 0)
        assert by_a[cls]["confidence"] == by_b[cls]["confidence"]
    fc = writer.to_feature_collection(out_b)
    rep = writer.validate(fc)
    assert rep["ok"] and rep["summary"]["tr_remarks"] == 0
    for f in fc["features"]:
        if f["properties"]["class_name"] in DISKS["radius_m"]:
            g = Polygon(f["geometry"]["coordinates"][0])
            assert g.is_valid
            r_m = math.sqrt(g.area / math.pi) / k
            assert 0.035 <= r_m <= 0.065, r_m
    # диск стоит во внутренней точке исходного полигона
    for cls in DISKS["radius_m"]:
        src = next(f["geom"] for f in feats if f["class_name"] == cls)
        assert by_b[cls]["geom"].contains(src.representative_point())


def _metric(path):
    spec = importlib.util.spec_from_file_location("koz3_compute_metrics", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fort_f1(mod, pred_geom, gt_geom):
    pred = json.loads(json.dumps(writer.to_feature_collection(post.finalize([_feat("fortifikatsii", pred_geom)]))))
    gt = {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": gt_geom.__geo_interface__,
          "properties": {"class_name": "fortifikatsii", "markup_type": "ae", "region_name": "r"}}]}
    gt = json.loads(json.dumps(gt, default=list))
    pc = mod.F1Calculator().calculate_f1(pred, gt)["per_class"]["fortifikatsii"]
    return pc["tp"], pc["fp"], pc["fn"]


def test_official_metric_weak_dominance(official_metric):
    if official_metric is None:
        pytest.skip("нет compute_metrics.py")
    mod = _metric(official_metric)
    x, y = _merc(37.0, 55.0)
    k = 1 / math.cos(math.radians(55.0))
    gt = box(x - 10 * k, y - 1 * k, x + 10 * k, y + 1 * k)                      # вал 20 x 2 м
    cases = {
        "покрытие 0,33, точка внутри": (box(x - 3 * k, y - 3 * k, x + 3 * k, y + 3 * k), (0, 1, 1), (1, 0, 0)),
        "засчитанный полигон": (box(x - 5 * k, y - 0.8 * k, x + 5 * k, y + 0.8 * k), (1, 0, 0), (1, 0, 0)),
        "точка вне эталона": (box(x - 3 * k, y + 1.5 * k, x + 3 * k, y + 6 * k), (0, 1, 1), (0, 1, 1)),
    }
    for name, (poly, want_poly, want_disk) in cases.items():
        disk = post.micro_circles([_feat("fortifikatsii", poly)], DISKS)[0]["geom"]
        assert _fort_f1(mod, poly, gt) == want_poly, name
        assert _fort_f1(mod, disk, gt) == want_disk, name
