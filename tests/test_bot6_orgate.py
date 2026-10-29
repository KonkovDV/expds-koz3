"""bot-6 ORGATE G ([orgate], по умолчанию выключен): гейт arch вместо R. Флаг выкл. — побайтно база; модель и meta;
смоук на опорном входе (008_Герите); правило LAS по каталогу; dfine_*/sp_dfine не оцениваются; маркер и откат на R;
сборка и check_submission."""
import hashlib
import importlib.util
import json
import logging
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import Point, shape

from expds import orgate, rerank_arch, watchdog

ROOT = Path(__file__).resolve().parents[1]
FX = Path("/workspace/koz3_shared/bot-6/work/orgate_final/orgate_smoke_008.geojson")   # вне git (п. 13.11 ТР)
X3857, Y3857 = 4_118_890.0, 7_361_866.0


def _cfg():
    return tomllib.loads((ROOT / "configs" / "default.toml").read_text(encoding="utf-8"))


def _fixture():
    if not FX.is_file():
        pytest.skip("нет опорного входа смоука ORGATE (вне git, п. 13.11 ТР)")
    fx = json.loads(FX.read_text(encoding="utf-8"))
    return fx, [dict(f["properties"], geom=shape(f["geometry"])) for f in fx["features"]]


def _key(fs):
    return [(f["class_name"], round(float(f["confidence"]), 9), f["geom"].wkb_hex) for f in fs]


def test_default_off_model_and_meta():
    cfg = _cfg()
    og = cfg["orgate"]
    assert og == {"enabled": False, "model": "models/orgate_g.txt", "meta": "models/orgate_g.json"}
    m = (ROOT / og["model"]).read_bytes()
    assert hashlib.sha256(m).hexdigest() == "676f4c1bbc8ff9d71b0bb86b39c9b870c10dc5a327bcf38d81c08e9d64db697a"
    assert len(m) < 600_000
    meta = json.loads((ROOT / og["meta"]).read_text(encoding="utf-8"))
    assert meta["names"] == rerank_arch.NAMES + orgate.GATE_NAMES and meta["variant"] == "G"
    assert set(meta["thresholds"]) == {"or", "sp_arch"} and meta["arch_group_alias"] == {"sp_or": "sp_arch"}
    assert meta["las_rule"]
    bst, _ = orgate.load(og)
    assert bst.num_feature() == 57


def test_flag_off_is_noop_in_finish():
    fx, feats = _fixture()
    cfg = _cfg()
    base = {k: v for k, v in cfg.items() if k != "orgate"}
    a = watchdog.finish([dict(f) for f in feats], base)
    b = watchdog.finish([dict(f) for f in feats], cfg)
    assert _key(a) == _key(b)


def test_smoke_reference_region_removes_expected():
    fx, feats = _fixture()
    rest, kept, info = orgate.decide(feats, {}, fx["catalog"])
    assert len(info["target"]) == 43 and info["las_source"] == "catalog"
    assert info["removed_idx"] == fx["expected_removed_idx"] and len(info["removed_idx"]) >= 1
    assert len(rest) == 0 and len(kept) == 43 - len(info["removed_idx"])
    assert all("conf_stack" in k for k in kept)


def test_las_rule_from_catalog_and_li_fallback():
    fx, feats = _fixture()
    r = feats[0]["region_name"]
    _, kept, info = orgate.decide(feats, {}, {r: {"las": True, "utm_json": None}})
    assert kept == [] and len(info["removed_idx"]) == 43 and set(info["score"]) == {-1.0}
    # без каталога — правило «есть кандидаты li» (как has_las у R)
    li = dict(feats[0], class_name="kurgany_tselye", markup_type="li", threshold_group="li", geom=Point(feats[0]["geom"].centroid).buffer(3))
    _, kept2, info2 = orgate.decide(feats + [li], {}, None)
    assert info2["las_source"] == "li_candidates" and kept2 == []
    _, kept3, info3 = orgate.decide(feats, {}, None)
    assert info3["las_source"] == "li_candidates" and len(kept3) > 0


def test_dfine_groups_not_scored():
    fx, feats = _fixture()
    extra = [dict(feats[i], markup_type=g, threshold_group=g) for i, g in ((0, "dfine_or"), (1, "dfine_ae"), (2, "sp_dfine"))]
    rest, kept, info = orgate.decide(feats + extra, {}, fx["catalog"])
    assert len(info["target"]) == 43 and all(i < 43 for i in info["target"])
    assert [f["threshold_group"] for f in rest] == ["dfine_or", "dfine_ae", "sp_dfine"]


def test_unavailable_marker_and_fallback(tmp_path, caplog):
    fx, feats = _fixture()
    cfg = _cfg()
    miss = {**cfg, "orgate": {"enabled": True, "model": str(tmp_path / "no.txt"), "meta": str(tmp_path / "no.json")}}
    with caplog.at_level(logging.ERROR):
        out = watchdog.finish([dict(f) for f in feats], miss)
    assert orgate.MARKER in caplog.text
    assert _key(out) == _key(watchdog.finish([dict(f) for f in feats], cfg))   # откат = R (как при выкл.)
    no_r = {**miss, "rerank_arch": {**cfg["rerank_arch"], "enabled": False}}
    base = {**cfg, "rerank_arch": {**cfg["rerank_arch"], "enabled": False}}
    assert _key(watchdog.finish([dict(f) for f in feats], no_r)) == _key(watchdog.finish([dict(f) for f in feats], base))


def test_enabled_replaces_r_in_finish():
    fx, feats = _fixture()
    cfg = {**_cfg(), "orgate": {**_cfg()["orgate"], "enabled": True}, orgate.CATALOG_KEY: fx["catalog"]}
    out = watchdog.finish([dict(f) for f in feats], cfg)
    _, kept, _ = orgate.decide(feats, {}, fx["catalog"])
    arch = [f for f in out if f["class_name"] == "arkhitektury"]
    assert len(arch) <= len(kept) and len(arch) >= 1
    assert all("conf_stack" in f for f in arch)


def test_catalog_info():
    regs = [SimpleNamespace(name="a", las=[Path("x.las")], utm_json=Path("a/UTM.json")),
            SimpleNamespace(name="b", las=[], utm_json=None)]
    assert orgate.catalog_info(regs) == {"a": {"las": True, "utm_json": "a/UTM.json"}, "b": {"las": False, "utm_json": None}}


def test_pipeline_sets_catalog_only_when_enabled():
    src = (ROOT / "expds" / "pipeline.py").read_text(encoding="utf-8")
    i = src.index("cfg[CATALOG_KEY] = catalog_info(regions)")
    assert 'if (cfg.get("orgate") or {}).get("enabled"):' in src[i - 300:i]


def _bs():
    spec = importlib.util.spec_from_file_location("b6_build_og", ROOT / "scripts" / "build_submission.py")
    bs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bs)
    return bs


def test_not_packaged_when_off(monkeypatch):
    from conftest import allow_missing_submission_models
    allow_missing_submission_models(monkeypatch)
    bs = _bs()
    assert "orgate" in bs.FLAG_MODULES
    rel = {p.relative_to(ROOT).as_posix() for p in bs.collect()}
    assert not {"expds/orgate.py", "models/orgate_g.txt", "models/orgate_g.json"} & rel


def test_packaged_when_on(monkeypatch):
    from conftest import allow_missing_submission_models
    allow_missing_submission_models(monkeypatch)
    bs = _bs()
    text = (ROOT / "configs" / "default.toml").read_text(encoding="utf-8").replace(
        "[orgate]\nenabled = false\n", "[orgate]\nenabled = true\n")
    real = bs.tomllib.loads
    monkeypatch.setattr(bs.tomllib, "loads", lambda s: real(text) if "[orgate]" in s else real(s))
    rel = {p.relative_to(ROOT).as_posix() for p in bs.collect()}
    assert {"expds/orgate.py", "models/orgate_g.txt", "models/orgate_g.json"} <= rel


def _cs():
    spec = importlib.util.spec_from_file_location("b6_cs_og", ROOT / "scripts" / "check_submission.py")
    cs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cs)
    cs.RESULTS.clear()
    return cs


def test_check_submission_marker_and_verdicts(tmp_path):
    import zipfile
    cs = _cs()
    cs.orgate_lines("x\nERROR ORGATE_UNAVAILABLE: нет файлов\n")
    assert [r[0] for r in cs.RESULTS] == ["FAIL"]
    cs.RESULTS.clear()
    cs.orgate_lines("ORGATE G: кандидатов arch 3\n")
    assert cs.RESULTS == []
    assert cs.orgate_verdict({"ok": False, "err": "x"})[0] == "FAIL"
    assert cs.orgate_verdict({"ok": True, "n_arch": 43, "removed": [], "expected_removed": []})[0] == "FAIL"
    assert cs.orgate_verdict({"ok": True, "n_arch": 43, "removed": [1], "expected_removed": [2]})[0] == "FAIL"
    assert cs.orgate_verdict({"ok": True, "n_arch": 43, "removed": [1], "expected_removed": [1]})[0] == "PASS"
    # флаг выкл. — ничего не требуется; вкл. без модели — FAIL
    text = (ROOT / "configs" / "default.toml").read_text(encoding="utf-8")
    zp = tmp_path / "s.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("configs/default.toml", text)
    z = zipfile.ZipFile(zp)
    args = SimpleNamespace(workdir=str(tmp_path), python="python3", cpus="", allow_net=True, timeout=60, threads=2)
    cs.check_orgate(z, tomllib.loads(text), args)
    assert cs.RESULTS == []
    cs.check_config_static(z)
    assert not [r for r in cs.RESULTS if "orgate" in r[2]]
    cs.RESULTS.clear()
    cs.check_orgate(z, {"orgate": {"enabled": True}}, args)
    assert [r[0] for r in cs.RESULTS] == ["FAIL"] and "models/orgate_g.txt" in cs.RESULTS[0][2]


def test_check_submission_smoke_runs_archive_code(tmp_path):
    if not FX.is_file():
        pytest.skip("нет опорного входа смоука ORGATE (вне git)")
    import sys
    import zipfile
    cs = _cs()
    text = (ROOT / "configs" / "default.toml").read_text(encoding="utf-8").replace(
        "[orgate]\nenabled = false\n", "[orgate]\nenabled = true\n")
    zp = tmp_path / "s.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("configs/default.toml", text)
        for p in (ROOT / "expds").glob("*.py"):
            z.write(p, f"expds/{p.name}")
        for n in ("models/orgate_g.txt", "models/orgate_g.json"):
            z.write(ROOT / n, n)
    args = SimpleNamespace(workdir=str(tmp_path), orgate_smoke=str(FX), python=sys.executable, cpus="", allow_net=True, timeout=300, threads=2)
    cs.check_orgate(zipfile.ZipFile(zp), tomllib.loads(text), args)
    assert [r[0] for r in cs.RESULTS] == ["PASS"], cs.RESULTS


def test_check_submission_missing_smoke_input_warns_not_pass(tmp_path):
    """bot-7 13:12: [orgate] enabled, модель в архиве, а опорного входа (--orgate-smoke) на машине нет —
    явный WARN «смоук не выполнен», не PASS и не молчание."""
    import zipfile
    cs = _cs()
    text = (ROOT / "configs" / "default.toml").read_text(encoding="utf-8").replace(
        "[orgate]\nenabled = false\n", "[orgate]\nenabled = true\n")
    zp = tmp_path / "s.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("configs/default.toml", text)
        for n in ("expds/orgate.py", "models/orgate_g.txt", "models/orgate_g.json"):
            z.writestr(n, "x")
    args = SimpleNamespace(workdir=str(tmp_path), orgate_smoke=str(tmp_path / "none.geojson"), python="python3",
                           cpus="", allow_net=True, timeout=60, threads=2)
    cs.check_orgate(zipfile.ZipFile(zp), tomllib.loads(text), args)
    assert [r[0] for r in cs.RESULTS] == ["WARN"]
    assert cs.RESULTS[0][2].startswith("смоук не выполнен") and "none.geojson" in cs.RESULTS[0][2]
