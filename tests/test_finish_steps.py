"""Сбой одного шага сдачи не выключает остальные шаги цепочки."""
import tomllib

from shapely.geometry import Point

from expds import post, watchdog
from conftest import ROOT


def _feat():
    return {"geom": Point(500_000, 6_100_000).buffer(2), "class_name": "kurgany_tselye",
            "confidence": 0.9, "markup_type": "li", "region_name": "r",
            "threshold_group": "li", "source": "a.las"}


def _default():
    return tomllib.loads((ROOT / "configs" / "default.toml").read_text(encoding="utf-8"))


def test_default_finish_still_shrinks_the_kurgan():
    out = watchdog.finish([_feat()], _default())
    assert out
    assert min(f["geom"].area for f in out) < 2.0


def test_failed_merge_still_shrinks_the_kurgan(monkeypatch):
    monkeypatch.setattr(post, "merge_within_markup", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("склейка")))
    out = watchdog.finish([_feat()], _default())
    assert out
    assert min(f["geom"].area for f in out) < 2.0


def test_failed_micro_circle_keeps_the_kurgan_and_the_chain(monkeypatch):
    monkeypatch.setattr(post, "micro_circles", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("круг")))
    out = watchdog.finish([_feat()], _default())
    assert out
    assert min(f["geom"].area for f in out) > 5.0
