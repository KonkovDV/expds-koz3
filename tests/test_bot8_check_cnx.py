"""bot-8 bot8-cnx: проверки [li_cnx] в scripts/check_submission.py (check_cnx и соседи), без сборки архива.

Случаи L/N/M/S — из ревью бота 4 (bot-4/REVIEW_CHECK_SUBMISSION_CNX.md)."""
import hashlib
import importlib.util
import io
import json
import re
import types
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CX = ROOT / "inference" / "cnx"
if not any(CX.glob("wheels/rvt_py-*.whl")):
    pytest.skip("rvt_py wheel is not committed to main (archives excluded by the 29.10 merge policy); "
                "restore inference/cnx/wheels/ from origin/bots/bot8-cnx to run the [li_cnx] checker tests",
                allow_module_level=True)
W = "models/cnx/w.pt"
WBYTES = b"fake-cnx-weights"
SMOKE = b"smoke-weights-c5"


@pytest.fixture(scope="module")
def cs():
    spec = importlib.util.spec_from_file_location("cs_cnx", ROOT / "scripts" / "check_submission.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _wman():
    return json.dumps({"weights": [{"path": W, "sha256": _sha(WBYTES), "size": len(WBYTES)}],
                       "never_pack": [{"sha256": _sha(SMOKE), "size": len(SMOKE)}]}).encode()


def _files(mut=lambda n, d: d, weights=WBYTES):
    out = {}
    for p in sorted(CX.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            n = p.relative_to(ROOT).as_posix()
            out[n] = p.read_bytes()
    out["inference/cnx/WEIGHTS_MANIFEST.json"] = _wman()
    if weights is not None:
        out[W] = weights
    return {n: d2 for n, d in out.items() if (d2 := mut(n, d)) is not None}


def _remanifest(f):
    """MANIFEST, пересобранный «как сборка» (самосогласованный) — для изменённого кода."""
    m = json.loads(f["inference/cnx/MANIFEST.json"])
    m["sha256"] = {n[len("inference/cnx/"):]: _sha(d) for n, d in f.items() if n.startswith("inference/cnx/")
                   and n not in ("inference/cnx/MANIFEST.json", "inference/cnx/WEIGHTS_MANIFEST.json")}
    f["inference/cnx/MANIFEST.json"] = json.dumps(m).encode()
    return f


def _prep(f, extra: bytes):
    k = "inference/cnx/cnx_prep.py"
    f[k] = f[k].replace(b"import numpy as np\n", b"import numpy as np\n" + extra + b"\n", 1)
    return _remanifest(f)


def _zip(files):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        for n, d in files.items():
            z.writestr(n, d)
    return zipfile.ZipFile(io.BytesIO(b.getvalue()))


def _cfg(**kw):
    lc = {"enabled": True, "weights": [W], **kw}
    return {"li_cnx": lc, "thresholds_by_markup": {"li_cnx": {"kurgany_tselye": 0.5, "kurgany_povrezhdennye": 0.5}}}


ON = _cfg()


def _run(cs, z, cfg=ON):
    cs.RESULTS.clear()
    third = cs.check_cnx(z, cfg)
    return third, [(s, msg) for s, c, msg in cs.RESULTS if c == "cnx"]


def _has(res, status, needle):
    return any(s == status and needle in msg for s, msg in res)


def test_ok(cs):
    third, res = _run(cs, _zip(_files()))
    assert [s for s, _ in res] == ["PASS"] * 5, res      # config, MANIFEST, prep_code_sha, веса, сеть
    assert {"numpy", "scipy", "laspy"} <= third
    assert not {"matplotlib", "osgeo"} & third           # rvt.blend/tile не достижимы из cnx_prep


def test_disabled(cs):
    third, res = _run(cs, _zip(_files()), {"li_cnx": {"enabled": False}})
    assert third == set() and [s for s, _ in res] == ["WARN"]
    third, res = _run(cs, _zip({}), {})
    assert third == set() and res == []


@pytest.mark.parametrize("mut, cfg, needle", [
    (lambda n, d: d + b"\n" if n.endswith("relief_ext.py") else d, ON, "sha ≠ ['vendor/relief_ext.py']"),
    (lambda n, d: None if n.endswith(".whl") else d, ON, "rvt_py-*.whl"),
    (lambda n, d: None if n.endswith("expds_core.abi3.so") else d, ON, "expds_core.abi3.so"),
    (lambda n, d: d, {"li_cnx": {"enabled": True, "weights": []}}, "weights пуст"),
    (lambda n, d: None if n.endswith("WEIGHTS_MANIFEST.json") else d, ON, "WEIGHTS_MANIFEST"),
])
def test_fail(cs, mut, cfg, needle):
    _, res = _run(cs, _zip(_files(mut)), cfg)
    assert _has(res, "FAIL", needle), res


# ---------- сеть (п. 1 ревью): как основной сканер + динамика
@pytest.mark.parametrize("extra, status, needle", [
    (b"import requests", "FAIL", "import requests"),                                   # N1
    (b"from urllib import request", "FAIL", "urllib.request"),                         # N2
    (b"import importlib; _s = importlib.import_module('soc' + 'ket')", "WARN", "динамический импорт"),  # N3
    (b"_s = __import__('socket')", "FAIL", "import socket"),                           # N4: константа → граф
    (b"_s = __import__('socket')", "WARN", "__import__('socket')"),
    (b"import subprocess; subprocess.run(['curl', 'http://x'])", "WARN", "subprocess.run"),     # N5
    (b"import os; os.system('wget x')", "WARN", "os.system"),
    (b"import torch; torch.hub.load('a/b', 'c')", "FAIL", "torch.hub.load"),           # N6
    (b"from urllib.request import urlopen; urlopen('http://x')", "FAIL", "вызов urlopen"),
    (b"import timm; timm.create_model('x', pretrained=True)", "FAIL", "pretrained=True"),
    # повторное ревью bot-4 5bc6ad1, п. a: алиасы и прочие обходы сканера
    (b"from subprocess import run; run(['curl', 'http://x'])", "WARN", "запуск процесса subprocess.run()"),
    (b"import subprocess as sp; sp.run(['curl', 'http://x'])", "WARN", "запуск процесса subprocess.run()"),
    (b"from subprocess import Popen as P; P(['curl'])", "WARN", "запуск процесса subprocess.Popen()"),
    (b"from os import system; system('wget x')", "WARN", "запуск процесса os.system()"),
    (b"exec('import soc' + 'ket')", "WARN", "exec() — код из строки"),
    (b"_f = eval('1')", "WARN", "eval()"),
    (b"import importlib; getattr(importlib, 'import_' + 'module')('socket')", "WARN", "getattr(importlib, …)"),
    (b"from importlib import import_module as im; im('soc' + 'ket')", "WARN", "динамический импорт import_module"),
    (b"from importlib import import_module as im; im('socket')", "FAIL", "import socket"),
    (b"import ctypes", "WARN", "import ctypes — нативные вызовы"),
    (b"from ctypes import CDLL", "WARN", "from ctypes import"),
    (b"import torch; torch.utils.model_zoo.load_url('http://x')", "FAIL", "вызов load_url"),
    (b"from torch.utils.model_zoo import load_url as lu; lu('http://x')", "FAIL", "вызов load_url"),
])
def test_net_in_prep(cs, extra, status, needle):
    _, res = _run(cs, _zip(_prep(_files(), extra)))
    assert _has(res, status, needle), res


def test_n7_dynamic_load_of_unreachable_net_module(cs):
    f = _files()
    f["inference/cnx/vendor/helper.py"] = b"import socket\n"
    _, res = _run(cs, _zip(_prep(f, b"import importlib; importlib.import_module('hel' + 'per')")))
    assert _has(res, "WARN", "динамический импорт") and _has(res, "WARN", "helper.py:1 import socket"), res
    assert not _has(res, "FAIL", "socket")


def test_n7_constant_dynamic_import_is_reachable(cs):
    f = _files()
    f["inference/cnx/vendor/helper.py"] = b"import socket\n"
    _, res = _run(cs, _zip(_prep(f, b"import importlib; importlib.import_module('helper')")))
    assert _has(res, "FAIL", "helper.py:1 import socket"), res


def test_l2_unreachable_listed_net_file_warns(cs):
    f = _remanifest(dict(_files(), **{"inference/cnx/evil.py": b"import socket\nsocket.create_connection(('x', 80))\n"}))
    _, res = _run(cs, _zip(f))
    assert _has(res, "WARN", "evil.py:1 import socket (модуль недостижим") and not _has(res, "FAIL", "evil"), res


def test_main_scanner_unchanged_calls(cs, tmp_path):
    """Общий net_calls: основной сканер по-прежнему ловит вызовы (WARN в достижимом, как до правки)."""
    (tmp_path / "inference").mkdir()
    (tmp_path / "expds").mkdir()
    (tmp_path / "inference" / "main.py").write_text("import expds.pipeline\n")
    (tmp_path / "expds" / "__init__.py").write_text("")
    (tmp_path / "expds" / "pipeline.py").write_text("import torch\ntorch.hub.load('a', 'b')\n")
    info, reach = cs.scan_imports(tmp_path)
    assert info["expds.pipeline"][2] == [(2, "torch.hub.load()")] and "expds.pipeline" in reach


# ---------- MANIFEST (п. 2)
def test_m1_empty_manifest_fails(cs):
    f = _files()
    f["inference/cnx/MANIFEST.json"] = b'{"sha256": {}}'
    _, res = _run(cs, _zip(f))
    assert _has(res, "FAIL", "пуст"), res


def test_m2_changed_and_unlisted_required_fails(cs):
    f = _files()
    f["inference/cnx/vendor/relief_ext.py"] += b"\n# changed\n"
    m = json.loads(f["inference/cnx/MANIFEST.json"])
    del m["sha256"]["vendor/relief_ext.py"]
    f["inference/cnx/MANIFEST.json"] = json.dumps(m).encode()
    _, res = _run(cs, _zip(f))
    assert _has(res, "FAIL", "обязательные вне MANIFEST ['inference/cnx/vendor/relief_ext.py']"), res
    assert _has(res, "FAIL", "prep_code_sha"), res


def test_m2_unlisted_py_or_so_fails_readme_warns(cs):
    f = _files()
    f["inference/cnx/vendor/extra.py"] = b"x = 1\n"
    f["inference/cnx/vendor/extra.so"] = b"\x7fELF"
    f["inference/cnx/NOTES_README.md"] = b"x"
    f["inference/cnx/_deps/LICENSE.txt"] = b"x"
    _, res = _run(cs, _zip(f))
    assert _has(res, "FAIL", "inference/cnx/vendor/extra.py") and _has(res, "FAIL", "extra.so"), res
    assert _has(res, "WARN", "LICENSE.txt"), res


def test_m3_byte_change_with_stale_manifest(cs):
    _, res = _run(cs, _zip(_files(lambda n, d: d + b"\n" if n.endswith("vendor/expds/lidar.py") else d)))
    assert _has(res, "FAIL", "vendor/expds/lidar.py"), res


def test_prep_code_sha_expected(cs):
    f = _remanifest(_files(lambda n, d: d + b"\n" if n.endswith("vendor/expds/crs.py") else d))   # самосогласованный
    _, res = _run(cs, _zip(f))
    assert _has(res, "FAIL", "prep_code_sha вендора") and not _has(res, "FAIL", "sha ≠ ['"), res
    m = json.loads(f["inference/cnx/MANIFEST.json"])
    del m["prep_code_sha_expected"]
    f["inference/cnx/MANIFEST.json"] = json.dumps(m).encode()
    _, res = _run(cs, _zip(f))
    assert _has(res, "FAIL", "нет prep_code_sha_expected"), res


# ---------- веса по WEIGHTS_MANIFEST
def test_weights_sha_size_and_never_pack(cs):
    _, res = _run(cs, _zip(_files(weights=b"fake-cnx-weightZ")))
    assert _has(res, "FAIL", f"{W}: sha256"), res
    _, res = _run(cs, _zip(_files(weights=SMOKE)))
    assert _has(res, "FAIL", "never_pack"), res
    f = _files()
    f["models/other.pt"] = SMOKE                          # smoke под чужим именем
    _, res = _run(cs, _zip(f))
    assert _has(res, "FAIL", "models/other.pt: smoke-веса"), res
    _, res = _run(cs, _zip(_files()), _cfg(weights=["models/cnx/unlisted.pt"]))
    assert _has(res, "FAIL", "нет в WEIGHTS_MANIFEST"), res


def test_real_weights_manifest_never_packs_c5_smoke(cs):
    m = json.loads((CX / "WEIGHTS_MANIFEST.json").read_text(encoding="utf-8"))
    assert any(e["sha256"].startswith("9b6b4c98f64f19f1") for e in m["never_pack"])


# ---------- пороги, tta, amp
@pytest.mark.parametrize("cfg, needle", [
    ({"li_cnx": {"enabled": True, "weights": [W]}}, "нет [thresholds_by_markup.li_cnx]"),
    (dict(_cfg(), thresholds_by_markup={"li_cnx": {"kurgany_tselye": 0.5}}), "kurgany_povrezhdennye"),
    (dict(_cfg(threshold_group="li"), thresholds_by_markup={"li": {"kurgany_tselye": 0.5,
                                                                   "kurgany_povrezhdennye": 0.5}}), "другой модели"),
    (_cfg(tta=4), "li_cnx.tta=4"),
    (_cfg(tta=True), "li_cnx.tta=True"),
    (_cfg(classes=["gorodishcha"]), "gorodishcha"),
    (_cfg(classes=["nope"]), "nope"),
])
def test_config_fail(cs, cfg, needle):
    _, res = _run(cs, _zip(_files()), cfg)
    assert _has(res, "FAIL", needle), res


def test_config_ok_variants_and_amp_warn(cs):
    for tta in (1, 2, 8):
        _, res = _run(cs, _zip(_files()), _cfg(tta=tta))
        assert not any(s == "FAIL" for s, _ in res), res
    _, res = _run(cs, _zip(_files()), _cfg(amp=False))
    assert _has(res, "WARN", "li_cnx.amp игнорируется"), res


# ---------- layout (п. 5)
def test_layout_whitelist_only_manifest_files(cs):
    cs.RESULTS.clear()
    cs.check_layout(_zip(_files()))
    assert not any("лишние .py" in msg for _, _, msg in cs.RESULTS)
    f = _files()
    f["inference/cnx/evil.py"] = b"import socket\n"
    cs.RESULTS.clear()
    cs.check_layout(_zip(f))
    assert any(s == "WARN" and "inference/cnx/evil.py" in msg for s, _, msg in cs.RESULTS)


# ---------- .so (п. 3)
def test_check_so_covers_vendor_so(cs, tmp_path):
    d = tmp_path / "inference" / "cnx" / "vendor"
    d.mkdir(parents=True)
    (d / "expds_core.abi3.so").write_bytes((CX / "vendor" / "expds_core.abi3.so").read_bytes())
    (tmp_path / "inference" / "lib").mkdir()
    cs.RESULTS.clear()
    cs.check_so(tmp_path)
    res = [(s, m) for s, c, m in cs.RESULTS if c == "so"]
    assert res and res[0][0] == "PASS" and "inference/cnx/vendor/expds_core.abi3.so" in res[0][1], res
    (d / "expds_core.abi3.so").write_bytes(b"not an elf at all....")
    cs.RESULTS.clear()
    cs.check_so(tmp_path)
    assert any(s == "FAIL" and "cnx/vendor" in m for s, c, m in cs.RESULTS), cs.RESULTS


# ---------- проба (п. 6): веса cnx — и при --no-weights
def test_probe_loads_cnx_weights_without_weights_flag(cs, tmp_path, monkeypatch):
    assert "if load_w:\n            import gc\n            from expds import lidar_cnx" not in cs.PROBE
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "default.toml").write_text("")
    out = {"third_fail": {}, "import_fail": {}, "dists": {}, "cfg": [], "cnx_prep": {"ok": True, "err": ""},
           "weights": [{"markup": "li_cnx", "file": W, "ok": False, "err": "CnxError('вендор не совпадает')"}]}
    monkeypatch.setattr(cs.subprocess, "run", lambda *a, **k: types.SimpleNamespace(
        stdout="PROBE_JSON=" + json.dumps(out), stderr="", returncode=0))
    args = types.SimpleNamespace(weights=False, cpus="", allow_net=True, timeout=10, threads=2, lock=None)
    cs.RESULTS.clear()
    cs.run_probe(tmp_path, "python", {"__main__"}, set(), args)
    assert any(s == "FAIL" and c == "weights" and "li_cnx" in m for s, c, m in cs.RESULTS), cs.RESULTS


def test_main_extracts_cnx_weights_for_probe(cs):
    import inspect
    src = inspect.getsource(cs.main)
    assert "n in cnx_w" in src


# ---------- смоук (п. 4)
LOG_OK = "I x: cnx_prep a.las: ch=3 hillshade за 1.0 с (бэкенд rust, factor 1)"


@pytest.mark.parametrize("log, n_las, want", [
    (LOG_OK, 1, "PASS"),
    ("I x: ничего про cnx", 1, "FAIL"),
    ("I x: ничего про cnx", 0, None),
    (LOG_OK + "\nW cnx: веса [li_cnx] не загрузились (boom) — объектов cnx нет", 1, "FAIL"),
    ("W [li_cnx] включён, но нет [thresholds_by_markup.li_cnx]: объекты cnx НЕ выдаются", 0, "FAIL"),
])
def test_smoke_cnx(cs, log, n_las, want):
    cs.RESULTS.clear()
    cs.smoke_cnx(log, True, n_las)
    got = [s for s, c, _ in cs.RESULTS if c == "smoke"]
    assert (want in got) if want else got == [], cs.RESULTS
    cs.RESULTS.clear()
    cs.smoke_cnx(log, False, n_las)
    assert cs.RESULTS == []


def test_smoke_keys_match_pipeline_logs(cs):
    """Ключи смоука — подстроки настоящих строк лога (рефакторинг пайплайна не должен их рассинхронизировать),
    и каждый warning/exception пайплайна про cnx ловится каким-то ключом."""
    src = {p.name: p.read_text(encoding="utf-8") for p in (ROOT / "expds").glob("*.py")}
    allsrc = "\n".join(src.values())
    for k in cs.CNX_SMOKE_FAIL + cs.CNX_SMOKE_WARN:
        assert k.strip() in allsrc, k
    assert re.search(r'log\.info\("cnx_prep %s: ch=', src["lidar_cnx.py"])
    assert cs.CNX_SMOKE_OK.strip() in "cnx_prep %s: ch=" and cs.CNX_SMOKE_OK2 in "cnx_prep %s: ch="
    keys = cs.CNX_SMOKE_FAIL + cs.CNX_SMOKE_WARN
    for ln in src["pipeline.py"].splitlines():
        if re.search(r"(log\.(warning|exception|error)|_warn)\(", ln) and "cnx" in ln:
            assert any(k in ln for k in keys), ln


# ---------- повторное ревью bot-4 5bc6ad1, п. a: без ложных срабатываний
def test_aliases_no_false_positives(cs):
    """Метод .eval() модели, re.compile, getattr на объекте — не динамика."""
    _, res = _run(cs, _zip(_prep(_files(), b"import re; _r = re.compile('x'); _o = object(); getattr(_o, '__class__')\n"
                                          b"class _N:\n    def eval(self): return self\n_N().eval()")))
    assert [s for s, _ in res] == ["PASS"] * 5, res


def test_proc_in_unreachable_module_stays_quiet(cs):
    f = _remanifest(dict(_files(), **{"inference/cnx/evil2.py": b"from subprocess import run\nrun(['curl'])\n"}))
    _, res = _run(cs, _zip(f))
    assert not _has(res, "WARN", "evil2.py"), res      # недостижимое без сети — как и раньше, тихо


def test_import_aliases_helper(cs):
    import ast
    al = cs.import_aliases(ast.parse("import subprocess as sp\nfrom os import system\nfrom a.b import c as d\n"
                                     "import x.y\nfrom . import z\n"))
    assert al == {"sp": "subprocess", "system": "os.system", "d": "a.b.c"}


# ---------- п. b: README/LICENSE-исключение только для текстов
@pytest.mark.parametrize("name, status", [
    ("inference/cnx/README_evil.py", "FAIL"),
    ("inference/cnx/vendor/LICENSE.so", "FAIL"),
    ("inference/cnx/_deps/README.pyc", "FAIL"),
    ("inference/cnx/README2.md", "WARN"),
    ("inference/cnx/_deps/NOTICE", "WARN"),
    ("inference/cnx/COPYING.txt", "WARN"),
])
def test_soft_names_only_for_text(cs, name, status):
    f = _files()
    f[name] = b"import socket\n"
    _, res = _run(cs, _zip(f))
    if status == "FAIL":
        assert _has(res, "FAIL", f"код/данные вне MANIFEST ['{name}']"), res
        assert not _has(res, "WARN", "тексты вне MANIFEST"), res
    else:
        assert _has(res, "WARN", f"тексты вне MANIFEST.json: ['{name}']"), res
        assert not _has(res, "FAIL", "вне MANIFEST ["), res


# ---------- п. c: li_cnx.enabled только bool; иначе ВЫКЛЮЧЕН + WARN (как expds/lidar_cnx.enabled)
@pytest.mark.parametrize("val", ["false", "true", "0", 1, 0, None])
def test_enabled_not_bool_warns_and_is_off(cs, val):
    cfg = _cfg()
    cfg["li_cnx"]["enabled"] = val
    third, res = _run(cs, _zip(_files()), cfg)
    assert third == set() and not any(s in ("FAIL", "PASS") for s, _ in res), res
    assert _has(res, "WARN", f"li_cnx.enabled={val!r} — не bool"), res
    assert _has(res, "WARN", "выключен, а inference/cnx/"), res
    assert cs.cnx_on(cfg) is False


def test_enabled_semantics_match_pipeline_and_build(cs):
    import sys
    sys.path.insert(0, str(ROOT))
    from expds import lidar_cnx
    spec = importlib.util.spec_from_file_location("b8_bsg", ROOT / "scripts" / "build_submission_git.py")
    bsg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bsg)
    for val in (True, False, "false", "true", 1, 0, None):
        cfg = _cfg()
        cfg["li_cnx"]["enabled"] = val
        assert cs.cnx_on(cfg) == lidar_cnx.enabled(cfg) == bsg.cnx_enabled(cfg) == (val is True), val
