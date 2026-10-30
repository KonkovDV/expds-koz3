"""bot-7 (v3, после сбоя ПК-прогона 29.10): импорт D-FINE строго закреплённого transformers, громкий отказ.

(a) [sp_dfine] включён, классов нет → в логе SP_DFINE_UNAVAILABLE, выдача как v2 (на прогоне сдача не падает);
    check_submission: FAIL по маркеру в логе и по пробе в чистом процессе.
(b) в логе «sp_dfine n_det=… region=…» по участкам и итог «sp_dfine n_det=… total».
Импорт проверяется в отдельных процессах (sys.modules не трогаем у pytest). Настоящий вендор — если есть на box
(EXPDS_TEST_VENDOR или bot-7/work6/vendor_test/lib), иначе эти тесты пропускаются."""
import importlib.util
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from expds import sp_dfine

ROOT = Path(__file__).resolve().parents[1]
VENDOR = Path(os.environ.get("EXPDS_TEST_VENDOR", "/workspace/koz3_shared/bot-7/work6/vendor_test/lib"))


def _pkg(base: Path, version: str, body: str = "") -> Path:
    d = base / "transformers"
    d.mkdir(parents=True)
    (d / "__init__.py").write_text(f"__version__ = {version!r}\n{body}\n", encoding="utf-8")
    return base


def _root(tmp: Path, vendor: Path | None) -> Path:
    r = tmp / "pkg"
    (r / "inference" / "lib").mkdir(parents=True)
    if vendor is not None:
        link = r / "inference" / "lib" / "vendor"
        try:
            link.symlink_to(vendor, target_is_directory=True)
        except OSError:   # Windows без права на символические ссылки
            shutil.copytree(vendor, link)
    return r


def _run(code: str, *paths) -> dict:
    src = textwrap.dedent(f"""
        import json, sys
        sys.path.insert(0, {str(ROOT)!r})
        for p in {[str(p) for p in paths]!r}:
            sys.path.insert(1, p)
        {textwrap.indent(textwrap.dedent(code), '        ').strip()}
    """)
    r = subprocess.run([sys.executable, "-s", "-c", src], capture_output=True, text=True, timeout=600,
                       env={**os.environ, "OMP_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    line = next((x for x in r.stdout.splitlines() if x.startswith("J=")), None)
    assert line, r.stderr[-2000:]
    return json.loads(line[2:])


PROBE = """
from pathlib import Path
from expds import sp_dfine
try:
    C, M, info = sp_dfine.import_dfine(Path(ROOTP))
    print("J=" + json.dumps({"ok": True, "src": info["source"], "ver": info["version"], "C": C.__name__,
                             "pre": info.get("purged_preloaded")}))
except ImportError as e:
    print("J=" + json.dumps({"ok": False, "err": str(e)}))
"""


def _probe(root: Path, *paths, pre: str = "") -> dict:
    return _run(f"ROOTP = {str(root)!r}\n{pre}\n{PROBE}", *paths)


def _broken_vendor(tmp: Path) -> Path:
    # как на Windows/другом python: .so regex/tokenizers из linux-вендора не грузятся при import transformers
    return _pkg(tmp / "bad_vendor", "5.21.0", "raise ImportError('regex._regex: invalid ELF header (имитация)')")


def test_broken_vendor_no_env_is_unavailable(tmp_path):
    out = _probe(_root(tmp_path, _broken_vendor(tmp_path)))
    assert not out["ok"] and sp_dfine.UNAVAILABLE in out["err"] and "invalid ELF" in out["err"]


def test_env_wrong_version_rejected(tmp_path):
    env = _pkg(tmp_path / "site_old", "4.46.0", "class DFineConfig: pass\nclass DFineForObjectDetection: pass")
    out = _probe(_root(tmp_path, _broken_vendor(tmp_path)), env)
    assert not out["ok"] and "4.46.0" in out["err"] and "5.21.0" in out["err"]


def test_env_exact_pin_accepted_when_vendor_broken(tmp_path):
    env = _pkg(tmp_path / "site_pin", "5.21.0", "class DFineConfig: pass\nclass DFineForObjectDetection: pass")
    out = _probe(_root(tmp_path, _broken_vendor(tmp_path)), env)
    assert out["ok"] and out["src"] == "env" and out["ver"] == "5.21.0"


@pytest.mark.skipif(not (VENDOR / "transformers").is_dir(), reason="нет распакованного вендора transformers 5.21.0")
def test_preloaded_old_transformers_purged_vendor_wins(tmp_path):
    """Воспроизведение: в окружении стоит СТАРЫЙ transformers без D-FINE и уже импортирован до пакета."""
    pytest.importorskip("torch")
    old = _pkg(tmp_path / "site_old", "4.40.0", "")          # без DFine*
    out = _probe(_root(tmp_path, VENDOR), old, pre="import transformers; assert transformers.__version__ == '4.40.0'")
    assert out["ok"], out
    assert out["src"] == "vendor" and out["ver"] == sp_dfine.TRANSFORMERS_PIN and out["pre"] == "4.40.0"


def test_region_feats_logs_unavailable_and_returns_v2(monkeypatch, tmp_path, caplog):
    def boom(st, root, dev):
        raise ImportError(f"{sp_dfine.UNAVAILABLE}: D-FINE недоступен — vendor: ImportError: x")
    monkeypatch.setattr(sp_dfine, "load_models", boom)
    cfg = {"sp_dfine": {"enabled": True, "weights": ["w"], "device": "cpu", "strict": False},
           "thresholds_by_markup": {"sp_dfine": {c: 0.5 for c in sp_dfine.CLASSES}}}
    reg = SimpleNamespace(name="r", rasters={"sp_or": [tmp_path / "a.tif"]})
    with caplog.at_level(logging.ERROR, logger=sp_dfine.log.name):
        feats, complete = sp_dfine.region_feats(reg, cfg, None, {}, tmp_path)
    assert feats == [] and complete is False
    errs = [r for r in caplog.records if r.levelno >= logging.ERROR and sp_dfine.UNAVAILABLE in r.getMessage()]
    assert errs, [r.getMessage() for r in caplog.records]


def test_region_feats_no_thresholds_is_loud(tmp_path, caplog):
    cfg = {"sp_dfine": {"enabled": True, "weights": ["w"], "strict": False}}
    reg = SimpleNamespace(name="r", rasters={"sp_or": [tmp_path / "a.tif"]})
    with caplog.at_level(logging.ERROR, logger=sp_dfine.log.name):
        assert sp_dfine.region_feats(reg, cfg, None, {}, tmp_path) == ([], False)
    assert any(sp_dfine.UNAVAILABLE in r.getMessage() for r in caplog.records)


class _St:
    def __init__(self):
        self.feats = []
    def add(self, f):
        self.feats += f
    def snapshot(self):
        return list(self.feats)
    def replace(self, f):
        self.feats = list(f)


class _G:
    soft_ceiling = None
    def ok(self, s=0):
        return True
    def elapsed(self):
        return 1.0


def test_phase_logs_n_det_per_region_and_total(monkeypatch, caplog):
    from expds import pipeline

    def fake(reg, cfg, guard, cache, root, crs_for=None, soft_ok=None):
        cache["models"] = ["m"]
        k = {"a": 3, "b": 0}[reg.name]
        return [{"markup_type": "sp_or", "class_name": "kurgany_tselye", "region_name": reg.name,
                 "threshold_group": "sp_dfine", "confidence": 0.9}] * k, True
    monkeypatch.setattr(sp_dfine, "region_feats", fake)
    regs = [SimpleNamespace(name="a", rasters={"sp_or": ["x"]}), SimpleNamespace(name="b", rasters={"sp_or": ["y"]})]
    with caplog.at_level(logging.INFO, logger=pipeline.log.name):
        assert pipeline._sp_dfine_phase(regs, {"sp_dfine": {"enabled": True}}, _G(), _St(), ROOT) == 3
    msgs = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("sp_dfine n_det=3 region=a") for m in msgs)
    assert any(m.startswith("sp_dfine n_det=0 region=b") for m in msgs)
    assert any(m.startswith("sp_dfine n_det=3 total") for m in msgs)


def test_phase_unavailable_logs_total_zero(monkeypatch, caplog):
    from expds import pipeline
    monkeypatch.setattr(sp_dfine, "region_feats", lambda reg, cfg, guard, cache, root, **k: ([], False))
    regs = [SimpleNamespace(name="a", rasters={"sp_or": ["x"]})]
    with caplog.at_level(logging.INFO, logger=pipeline.log.name):
        pipeline._sp_dfine_phase(regs, {"sp_dfine": {"enabled": True}}, _G(), _St(), ROOT)
    assert any("sp_dfine n_det=0 total" in r.getMessage() and "моделей 0" in r.getMessage() for r in caplog.records)


def _cs():
    spec = importlib.util.spec_from_file_location("check_submission_b7", ROOT / "scripts" / "check_submission.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _zip(files: dict) -> zipfile.ZipFile:
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        for n, t in files.items():
            z.writestr(n, t)
    return zipfile.ZipFile(io.BytesIO(b.getvalue()))


def test_check_submission_fails_on_unavailable_marker():
    cs = _cs()
    z = _zip({"configs/default.toml": "[sp_dfine]\nenabled = true\n"})
    cs.dfine_lines(z, "2026 ERROR expds.sp_dfine: SP_DFINE_UNAVAILABLE: модели не загрузились\n")
    assert any(s == "FAIL" and c == "dfine" for s, c, _ in cs.RESULTS)


def test_check_submission_info_on_n_det():
    cs = _cs()
    z = _zip({"configs/default.toml": "[sp_dfine]\nenabled = true\n"})
    cs.dfine_lines(z, "x INFO expds.pipeline: sp_dfine n_det=12 region=103\nx INFO expds.pipeline: sp_dfine n_det=12 total\n")
    assert not any(s == "FAIL" for s, _, _ in cs.RESULTS)
    assert any(s == "INFO" and c == "dfine" and "n_det=12 total" in m for s, c, m in cs.RESULTS)


def test_check_submission_fails_enabled_without_vendor(tmp_path):
    cs = _cs()
    z = _zip({"configs/default.toml": "[sp_dfine]\nenabled = true\n", "expds/sp_dfine.py": "x = 1\n"})
    args = SimpleNamespace(python=sys.executable, cpus="", allow_net=True, timeout=60, threads=1, workdir=str(tmp_path))
    cs.check_dfine(z, {"sp_dfine": {"enabled": True}}, args)
    assert any(s == "FAIL" and c == "dfine" and "вендора" in m for s, c, m in cs.RESULTS)


def test_check_submission_probe_fails_on_broken_vendor(tmp_path):
    cs = _cs()
    files = {"configs/default.toml": "[sp_dfine]\nenabled = true\n",
             "inference/lib/vendor/transformers/__init__.py": "__version__='5.21.0'\nraise ImportError('ELF (имитация)')\n",
             "inference/lib/vendor/transformers-5.21.0.dist-info/METADATA": "Name: transformers\nVersion: 5.21.0\n"}
    for p in (ROOT / "expds").glob("*.py"):
        files[f"expds/{p.name}"] = p.read_text(encoding="utf-8")
    args = SimpleNamespace(python=sys.executable, cpus="", allow_net=True, timeout=300, threads=1, workdir=str(tmp_path))
    cs.check_dfine(_zip(files), {"sp_dfine": {"enabled": True}}, args)
    fails = [m for s, c, m in cs.RESULTS if s == "FAIL" and c == "dfine"]
    assert fails and "SP_DFINE_UNAVAILABLE" in fails[0], cs.RESULTS


def test_gate_packs_sp_dfine_for_pinned_import():
    """gor_gate.load_backbone берёт import_dfine из sp_dfine — сборщик обязан класть sp_dfine.py и при одном шлюзе."""
    root = Path(__file__).resolve().parents[1]
    assert "import_dfine" in (root / "expds" / "gor_gate.py").read_text(encoding="utf-8")
    assert 'off.discard("sp_dfine.py")' in (root / "scripts" / "build_submission.py").read_text(encoding="utf-8")


def test_check_config_hf_weight_dirs_and_vendored_imports():
    """каталог весов D-FINE (config.json + model.safetensors) — не FAIL «отсутствует»; без файлов — FAIL."""
    cs = _cs()
    for files, want in ((("config.json", "model.safetensors"), "PASS"), (("config.json",), "FAIL")):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("configs/default.toml", '[sp_dfine]\nenabled = true\nweights = ["models/dfine_f0"]\n')
            for f in files:
                z.writestr(f"models/dfine_f0/{f}", b"x")
        got = []
        orig = cs.report
        cs.report = lambda lvl, sec, msg, *a, **k: got.append((lvl, sec, msg))
        try:
            cs.check_config_static(zipfile.ZipFile(buf))
        finally:
            cs.report = orig
        lv = [g[0] for g in got if g[1] == "config" and ("отсутствуют" in g[2] or "путей к весам" in g[2])]
        assert lv == [want], got
        assert not any("models/dfine_f0/model.safetensors" in g[2] for g in got if g[0] == "WARN")


def test_dfine_weights_verdict_fails_on_zero_detections():
    """Веса загрузились, но на опорном участке (f0 даёт детекции) ни одного запроса ≥ 0,3 — FAIL."""
    cs = _cs()
    reg = "103_Гнёздово_Ольшанская"
    assert reg in cs.DFINE_KNOWN and cs.DFINE_KNOWN[reg]["ref_f0"]["n_ge"] > 0
    bad = cs.dfine_verdict([{"weights": "models/dfine_f0", "n_ge": 0, "max": 0.012}], reg)
    assert [s for s, _ in bad] == ["FAIL"]
    ok = cs.dfine_verdict([{"weights": "models/dfine_f0", "n_ge": 14, "max": 0.704}], reg)
    assert [s for s, _ in ok] == ["PASS"]
    assert [s for s, _ in cs.dfine_verdict([], reg)] == ["FAIL"]


def test_dfine_smoke_ndet_zero_on_known_region_fails():
    cs = _cs()
    log0 = ("INFO sp_dfine n_det=0 region=103_Гнёздово_Ольшанская complete=True, прошло 40 с\n"
            "INFO sp_dfine n_det=0 total, участков полностью 1, моделей 1\n")
    assert [s for s, _ in cs.dfine_ndet_verdict(log0)] == ["FAIL"]
    log1 = log0.replace("n_det=0 region", "n_det=154 region").replace("n_det=0 total", "n_det=154 total")
    assert cs.dfine_ndet_verdict(log1) == []
    synth = "sp_dfine n_det=0 region=000_synth complete=True\nsp_dfine n_det=0 total, участков полностью 1, моделей 1\n"
    assert [s for s, _ in cs.dfine_ndet_verdict(synth)] == ["WARN"]


def test_dfine_weights_probe_required_when_enabled(monkeypatch, tmp_path):
    cs = _cs()
    got = []
    monkeypatch.setattr(cs, "report", lambda s, c, m: got.append((s, c, m)))
    args = type("A", (), {"weights": True, "dfine_region": str(tmp_path / "nope"), "workdir": str(tmp_path)})()
    cs.check_dfine_weights(_zip({"configs/default.toml": "[sp_dfine]\nenabled = true\n"}), {"sp_dfine": {"enabled": True}}, args)
    assert got and got[0][0] == "FAIL"
    got.clear()
    cs.check_dfine_weights(_zip({}), {"sp_dfine": {"enabled": False}}, args)
    assert got == []


def test_sp_dfine_fp32_weights_autocast_no_half():
    """T4: .half() без autocast падает (bot-9 §7); веса fp32, fp16 — только autocast на cuda."""
    src = (ROOT / "expds" / "sp_dfine.py").read_text(encoding="utf-8")
    assert ".half()" not in src and "torch.float16" in src and "torch.autocast(" in src
