"""bot-8: scripts/check_submission.py под PIXGBM (ревью бота 4, REVIEW_CHECK_SUBMISSION.md).

1) [geophys.pixgbm] enabled=true — файл модели на каждый класс (object_poly → poly.txt, finds_points → finds.txt
   на maggr), иначе FAIL; лишний файл в папке не засчитывается.
2) psutil/tl2cgen/lleaves без пакета — WARN, только если каждый импорт по AST защищён try/except с фолбэком.
"""
import contextlib
import importlib.util
import io
import re
import textwrap
import tomllib
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("b8_check_submission", ROOT / "scripts" / "check_submission.py")
cs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cs)

PIX_OFF = 'enabled = false\nmodel_dir = "models/pixgbm/all"'
# bot-7 (cherry-pick на bot7-candidate-v3): в v2/v3 PIXGBM включён и у model_dir есть комментарий — тест берёт
# ту же секцию, приведённую к «выкл.», а включает её сам через pix_cfg().
CFG = re.sub(r'(?m)^enabled = (?:true|false)\nmodel_dir = "models/pixgbm/all".*$', PIX_OFF,
             (ROOT / "configs" / "default.toml").read_text(encoding="utf-8"), count=1)
assert PIX_OFF in CFG


def pix_cfg(enabled=True, **kw):
    s = CFG.replace(PIX_OFF, f'enabled = {"true" if enabled else "false"}\nmodel_dir = "models/pixgbm/all"')
    head, sec = s.split("[geophys.pixgbm]\n", 1)
    sec, tail = re.split(r"(?m)^(?=\[)", sec, maxsplit=1)
    for k, v in kw.items():
        sec, n = re.subn(rf"(?m)^{k} = .*$", f"{k} = {v}", sec, count=1)
        assert n == 1, k
    return head + "[geophys.pixgbm]\n" + sec + tail


def mkzip(cfg, extra=()):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr("configs/default.toml", cfg)
        c = tomllib.loads(cfg)
        refs = {v for _, v in cs._strings(c) if v.startswith("models/") and not v.startswith("models/pixgbm")}
        refs |= {"models/rerank_arch_lgbm.txt", "models/rerank_arch_lgbm.json"}
        refs |= {"expds/geo_pixgbm.py"}   # Murder Board 29.10: модуль вне git — его отсутствие отдельный FAIL
        for n in sorted(refs | set(extra)):
            z.writestr(n, b"x")
    return zipfile.ZipFile(io.BytesIO(b.getvalue()))


def run_cfg(cfg, extra=()):
    cs.RESULTS.clear()
    with contextlib.redirect_stdout(io.StringIO()):
        cs.check_config_static(mkzip(cfg, extra))
    return [(s, m) for s, c, m in cs.RESULTS if c == "config"]


def fails(res):
    return [m for s, m in res if s == "FAIL"]


def warns(res):
    return [m for s, m in res if s == "WARN" and "PIXGBM" in m]


POLY, FINDS = "models/pixgbm/all/poly.txt", "models/pixgbm/all/finds.txt"


def test_pixgbm_disabled_needs_no_models():
    assert fails(run_cfg(pix_cfg(enabled=False))) == []


@pytest.mark.parametrize("extra", [(), ("models/pixgbm/all/README.md",), (POLY,), (FINDS,),
                                   ("models/pixgbm/all_old/poly.txt", "models/pixgbm/all_old/finds.txt")])
def test_pixgbm_enabled_requires_every_class_model(extra):
    f = fails(run_cfg(pix_cfg(), extra))
    assert len(f) == 1 and "PIXGBM" in f[0]
    for need in (POLY, FINDS):
        assert (need in f[0]) == (need not in extra)


def test_pixgbm_enabled_with_both_models_passes():
    res = run_cfg(pix_cfg(), (POLY, FINDS))
    assert fails(res) == [] and warns(res) == []
    assert any(s == "PASS" and "PIXGBM" in m for s, m in res)
    assert not any("веса, на которые нет ссылок" in m for _, m in res)   # poly/finds — не «лишний объём»


def test_pixgbm_needed_follows_classes_and_maps():
    assert cs.pixgbm_needed({"classes": ["object_poly"], "maps": ["etgr"]})[1] == [POLY]
    assert cs.pixgbm_needed({"classes": ["finds_points"], "maps": ["maggr"]})[1] == [FINDS]
    assert cs.pixgbm_needed({"classes": ["finds_points"], "maps": ["etgr", "terfl"]})[1] == []   # finds только на maggr
    assert cs.pixgbm_needed({"model_dir": "models/px/"})[1] == ["models/px/finds.txt", "models/px/poly.txt"]
    res = run_cfg(pix_cfg(classes='["object_poly"]', maps='["etgr", "terfl"]'), (POLY,))
    assert fails(res) == []
    res = run_cfg(pix_cfg(classes='["finds_points"]', maps='["etgr"]'))
    assert fails(res) == [] and any("нечего считать" in m for m in warns(res))


def test_pixgbm_bad_classes_or_maps_fail():
    assert any("classes" in m for m in fails(run_cfg(pix_cfg(classes='["kurgany_tselye"]'), (POLY, FINDS))))
    assert any("maps" in m for m in fails(run_cfg(pix_cfg(maps='"maggr"'), (POLY, FINDS))))


def test_pixgbm_backend_so_missing_is_warn():
    res = run_cfg(pix_cfg(backend='"lleaves"'), (POLY, FINDS))
    assert fails(res) == [] and any("lightgbm" in m for m in warns(res))
    assert fails(run_cfg(pix_cfg(backend='"lleaves"'), (POLY,)))   # .txt всё равно обязателен


# ---------- импорты без пакета: WARN только при try/except-фолбэке ----------

def mkroot(tmp_path, src):
    (tmp_path / "inference").mkdir()
    (tmp_path / "expds").mkdir()
    (tmp_path / "inference" / "main.py").write_text("import expds.a\n", encoding="utf-8")
    (tmp_path / "expds" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "expds" / "a.py").write_text(textwrap.dedent(src), encoding="utf-8")
    return cs.guarded_import_sites(tmp_path, {"__main__", "expds", "expds.a"}, {"psutil", "tl2cgen", "lleaves"})


@pytest.mark.parametrize("src,guarded", [
    ("def f():\n    import psutil\n    return psutil\n", False),                       # T8 бота 4: нет try
    ("def f():\n    try:\n        import psutil\n    except Exception:\n        return None\n", True),
    ("try:\n    import psutil\nexcept ImportError:\n    psutil = None\n", True),         # модуль, с фолбэком
    ("try:\n    import psutil\nexcept (ValueError, ModuleNotFoundError):\n    psutil = None\n", True),
    ("try:\n    import psutil\nexcept ValueError:\n    psutil = None\n", False),         # не ловит ImportError
    ("try:\n    import psutil\nexcept ImportError:\n    raise\n", False),                # re-raise — не фолбэк
    ("try:\n    import psutil\nexcept ImportError:\n    pass\nelse:\n    pass\n", True),
    ("import psutil\n", False),
    ("try:\n    pass\nexcept ImportError:\n    import psutil\n", False),                # в обработчике — без защиты
    ("try:\n    def f():\n        import psutil\nexcept Exception:\n    pass\n", False),  # def под try ≠ вызов под try
    # вызов функции/класса под try с фолбэком — защищён (как TreeliteModel в geo_pixgbm.load_models)
    ("class M:\n    def __init__(self):\n        import tl2cgen\n"
     "def load():\n    try:\n        return M()\n    except Exception:\n        return None\n", True),
    ("class M:\n    def __init__(self):\n        import tl2cgen\n"
     "def load():\n    try:\n        return M()\n    except Exception:\n        return None\n"
     "def other():\n    return M()\n", False),                                              # один вызов без защиты
])
def test_guarded_import_sites(tmp_path, src, guarded):
    sites = mkroot(tmp_path, src)
    pkg = "tl2cgen" if "tl2cgen" in src else "psutil"
    assert sites[pkg] and all(g == guarded for _, g in sites[pkg]), sites[pkg]
    assert (pkg in cs.unguarded_imports(tmp_path, {"__main__", "expds", "expds.a"})) == (not guarded)


def test_real_code_lazy_imports_are_guarded():
    """Текущий код: psutil (geo_pixgbm.process_t0) и tl2cgen (TreeliteModel под try в load_models) — с фолбэком."""
    pytest.importorskip("expds.geo_pixgbm", reason="expds/geo_pixgbm.py is held out of main (EhimenNathan-derived lineage review)")
    _, reach = cs.scan_imports(ROOT)
    sites = cs.guarded_import_sites(ROOT, reach, {"psutil", "tl2cgen", "lleaves"})
    assert sites["psutil"] and all(g for _, g in sites["psutil"]), sites["psutil"]
    assert sites["tl2cgen"] and all(g for _, g in sites["tl2cgen"]), sites["tl2cgen"]
    assert sites["lleaves"] == []


def test_pixgbm_enabled_without_module_fails():
    """Murder Board 29.10: expds/geo_pixgbm.py вне git; без него в архиве — FAIL (иначе тихая классика ≠ v2 LB)."""
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr("configs/default.toml", '[geophys.pixgbm]\nenabled = true\nmodel_dir = "models/pixgbm/all"\n')
        for f in ("poly.txt", "finds.txt"):
            z.writestr(f"models/pixgbm/all/{f}", b"x")
    cs.RESULTS.clear()
    cs.check_config_static(zipfile.ZipFile(io.BytesIO(b.getvalue())))
    assert any(s == "FAIL" and "geo_pixgbm.py" in m for s, c, m in cs.RESULTS), cs.RESULTS


def test_sp_dfine_enabled_static_checks():
    """bot-10: [sp_dfine] enabled без весов/порогов/вендорного transformers d_fine — FAIL; полный набор — PASS."""
    cfg = '[sp_dfine]\nenabled = true\nweights = ["models/dfine_final"]\n'
    thr = "\n[thresholds_by_markup.sp_dfine]\n" + "".join(f"{c} = 0.5\n" for c in cs.CLASSES)

    def run(toml, files):
        b = io.BytesIO()
        with zipfile.ZipFile(b, "w") as z:
            z.writestr("configs/default.toml", toml)
            for f in files:
                z.writestr(f, b"x")
        cs.RESULTS.clear()
        cs.check_config_static(zipfile.ZipFile(io.BytesIO(b.getvalue())))
        return [(s, m) for s, c, m in cs.RESULTS if m.startswith("sp_dfine")]

    bad = run(cfg, [])
    assert bad and bad[0][0] == "FAIL" and "d_fine" in bad[0][1] and "model.safetensors" in bad[0][1]
    full = ["models/dfine_final/config.json", "models/dfine_final/model.safetensors",
            "inference/lib/vendor/transformers/models/d_fine/modeling_d_fine.py"]
    assert run(cfg + thr, full)[0][0] == "PASS"
