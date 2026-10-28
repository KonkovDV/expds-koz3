"""bot-7: флаг [rerank_arch] (по умолчанию выключен) и конфиг кандидата."""
import json
import tomllib

import numpy as np
import pytest
from conftest import ROOT
from shapely.geometry import Point

from expds import rerank_arch, watchdog

X3857, Y3857 = 4_118_890.0, 7_361_866.0      # ~37° в. д., 55° с. ш.


def _feats(n=40, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        cls = "arkhitektury" if i % 2 == 0 else "kurgany_tselye"
        x, y = X3857 + rng.uniform(0, 400), Y3857 + rng.uniform(0, 400)
        out.append({"geom": Point(x, y).buffer(3.0), "class_name": cls, "confidence": float(rng.uniform(0.3, 0.99)),
                    "markup_type": "sp_or", "region_name": "r1" if i < n // 2 else "r2"})
    return out


def _cfg(**extra):
    return {"thresholds": {}, "default_threshold": 0.15,
            "thresholds_by_markup": {"sp_or": {"arkhitektury": 1.01, "kurgany_tselye": 0.5}}, **extra}


def _key(fs):
    return sorted((f["class_name"], round(f["confidence"], 6), round(f["geom"].centroid.x, 3)) for f in fs)


def test_default_config_candidate():
    cfg = tomllib.loads((ROOT / "configs/default.toml").read_text(encoding="utf-8"))
    rr = cfg["rerank_arch"]
    assert isinstance(rr["enabled"], bool)
    if rr["enabled"]:                                        # включён — файлы обязаны быть в репозитории
        assert (ROOT / rr["model"]).is_file() and (ROOT / rr["meta"]).is_file()
        meta = json.loads((ROOT / rr["meta"]).read_text(encoding="utf-8"))
        assert meta["names"] == rerank_arch.NAMES
    assert cfg["post"]["micro_circles"]["radius_m"] == {"kurgany_povrezhdennye": 0.5, "kurgany_tselye": 0.5}
    assert cfg["thresholds_by_markup"]["or"]["arkhitektury"] == 0.065565
    assert cfg["verify"]["enabled"] is True


def test_flag_off_and_missing_files_are_noop(tmp_path):
    fs = _feats()
    base = watchdog.finish(fs, _cfg())
    off = watchdog.finish(fs, _cfg(rerank_arch={"enabled": False, "model": "x.txt", "meta": "x.json"}))
    miss = watchdog.finish(fs, _cfg(rerank_arch={"enabled": True, "model": str(tmp_path / "no.txt"),
                                                 "meta": str(tmp_path / "no.json")}))
    assert _key(base) == _key(off) == _key(miss)
    assert all(f["class_name"] != "arkhitektury" for f in base)   # порог 1.01 у архитектуры


def _model(tmp_path, taus):
    lgb = pytest.importorskip("lightgbm")
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, len(rerank_arch.NAMES)))
    y = (X[:, 0] > 0).astype(float)
    b = lgb.train({"objective": "binary", "verbose": -1, "num_threads": 1, "min_data_in_leaf": 5}, lgb.Dataset(X, y), 10)
    b.save_model(str(tmp_path / "m.txt"))
    meta = {"names": rerank_arch.NAMES, "classes": list(rerank_arch.METRIC_CLASSES), "grp_vocab": ["or", "sp_arch"],
            "mk_vocab": ["or", "sp_or"], "thresholds": taus, "arch_group_alias": {"sp_or": "sp_arch"}}
    (tmp_path / "m.json").write_text(json.dumps(meta), encoding="utf-8")
    return {"enabled": True, "model": str(tmp_path / "m.txt"), "meta": str(tmp_path / "m.json")}


def test_flag_on_rescores_only_arch_with_own_thresholds(tmp_path):
    fs = _feats()
    rr = _model(tmp_path, {"sp_arch": 0.0})
    on = watchdog.finish(fs, _cfg(rerank_arch=rr))
    base = watchdog.finish(fs, _cfg())
    arch = [f for f in on if f["class_name"] == "arkhitektury"]
    assert len(arch) == 20                                   # базовый порог 1.01 к ним не применён, все в пуле
    assert all("conf_stack" in f for f in arch)
    assert _key([f for f in on if f["class_name"] != "arkhitektury"]) == _key(base)
    rr_hi = _model(tmp_path, {"sp_arch": 1.01})
    assert not [f for f in watchdog.finish(fs, _cfg(rerank_arch=rr_hi)) if f["class_name"] == "arkhitektury"]


def test_features_shape_and_pool():
    fs = _feats(60)
    T = rerank_arch.table(fs)
    tgt = T["cls"] == rerank_arch.METRIC_CLASSES.index("arkhitektury")
    X = rerank_arch.features(T, tgt, ["sp_or"], ["sp_or"])
    assert X.shape == (tgt.sum(), 35) and np.isfinite(X).all()
    conf = np.linspace(0, 1, 6000)
    pool = rerank_arch.pool_mask(np.array(["g"] * 6000), np.full(6000, 3), conf)
    assert pool.sum() == 1000 and pool[-1000:].all()         # > 5000 → топ max(1000, 15 %)
