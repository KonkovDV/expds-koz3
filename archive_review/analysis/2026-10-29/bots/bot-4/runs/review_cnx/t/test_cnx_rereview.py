"""bot-4: повторное ревью 5bc6ad1 (check_cnx). cs_fix = scripts/check_submission.py@5bc6ad1, cs_rootfix = корневой bot-8."""
import hashlib, importlib.util, io, json, sys, zipfile
from pathlib import Path
HERE = Path(__file__).parent
SRC = HERE.parent / "repo"                      # скретч-клон @5bc6ad1
CX = SRC / "inference" / "cnx"
def load(n):
    s = importlib.util.spec_from_file_location(n, HERE / f"{n}.py"); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
V = {n: load(n) for n in ("cs_fix", "cs_rootfix")}
W = "models/cnx/weights_cnx_c3-sc_f0.pt"
TH = {"li_cnx": {"kurgany_tselye": 0.5, "kurgany_povrezhdennye": 0.5}}
ON = {"li_cnx": {"enabled": True, "weights": [W]}, "thresholds_by_markup": TH}
LAYOUT = {"metadata.json": b"{}", "inference/main.py": b"import expds\n", "configs/default.toml": b"",
          "expds/__init__.py": b"", "expds/pipeline.py": b""}
def base_files():
    return {p.relative_to(CX.parents[1]).as_posix(): p.read_bytes() for p in sorted(CX.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts}
def mkzip(files):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        for n, d in files.items(): z.writestr(n, d)
    return zipfile.ZipFile(io.BytesIO(b.getvalue()))
def remanifest(files):
    m = json.loads(files["inference/cnx/MANIFEST.json"])
    m["sha256"] = {n[len("inference/cnx/"):]: hashlib.sha256(d).hexdigest() for n, d in files.items()
                   if n.startswith("inference/cnx/") and n not in ("inference/cnx/MANIFEST.json", "inference/cnx/WEIGHTS_MANIFEST.json")}
    files["inference/cnx/MANIFEST.json"] = json.dumps(m).encode()
    return files
def patch_prep(files, extra):
    k = "inference/cnx/cnx_prep.py"
    assert b"import numpy as np\n" in files[k]
    files[k] = files[k].replace(b"import numpy as np\n", b"import numpy as np\n" + extra + b"\n", 1)
    return files
def run(mod, fn, *a):
    m = V[mod]; m.RESULTS.clear()
    try: getattr(m, fn)(*a)
    except Exception as e: return [("EXC", repr(e)[:160])]
    return [(s, msg[:170]) for s, c, msg in m.RESULTS if c in ("cnx", "layout", "config", "smoke")]
def show(tag, res):
    worst = "EXC" if any(s == "EXC" for s, _ in res) else "FAIL" if any(s == "FAIL" for s, _ in res) else "WARN" if any(s == "WARN" for s, _ in res) else "PASS"
    print(f"  {tag:10s} {worst:5s} " + " | ".join(f"{s}:{m}" for s, m in res if s in ("FAIL", "WARN", "EXC")))
    return worst
def both(fn, *a):
    for v in V: show(v, run(v, fn, *a))
CASES = {}
def case(f): CASES[f.__name__] = f; return f
def prep(extra, **more): return mkzip(remanifest(patch_prep(dict(base_files(), **more), extra)))

@case
def B0_baseline_on():
    "чистый inference/cnx @5bc6ad1, вкл. — ожидаем PASS"
    both("check_cnx", mkzip(base_files()), ON)
@case
def L1a_only_cnx_zip():
    "как в моём харнессе: zip только из inference/cnx (+evil.py) → FAIL 'нет обязательных файлов' — артефакт"
    f = base_files(); f["inference/cnx/evil.py"] = b"import socket\n"; z = mkzip(f)
    for v in V: show(v, run(v, "check_layout", z) + run(v, "check_cnx", z, {"li_cnx": {"enabled": False}}))
@case
def L1b_with_layout_files():
    "то же + обязательные файлы раскладки → лишний FAIL исчезает"
    f = dict(base_files(), **LAYOUT); f["inference/cnx/evil.py"] = b"import socket\n"; z = mkzip(f)
    for v in V: show(v, run(v, "check_layout", z) + run(v, "check_cnx", z, {"li_cnx": {"enabled": False}}))
@case
def L2_unreachable_net_listed():
    both("check_cnx", mkzip(remanifest(dict(base_files(), **{"inference/cnx/evil.py": b"import socket\nsocket.create_connection(('x',80))\n"}))), ON)
@case
def N1(): both("check_cnx", prep(b"import requests"), ON)
@case
def N2(): both("check_cnx", prep(b"from urllib import request"), ON)
@case
def N3(): both("check_cnx", prep(b"import importlib; _s = importlib.import_module('soc' + 'ket')"), ON)
@case
def N4(): both("check_cnx", prep(b"_s = __import__('socket')"), ON)
@case
def N4b_import_module_const(): both("check_cnx", prep(b"import importlib; _s = importlib.import_module('urllib.request')"), ON)
@case
def N5(): both("check_cnx", prep(b"import subprocess; subprocess.run(['curl', 'http://x'])"), ON)
@case
def N6(): both("check_cnx", prep(b"import torch; torch.hub.load('a/b', 'c')"), ON)
@case
def N7():
    f = base_files(); f["inference/cnx/vendor/helper.py"] = b"import socket\n"
    both("check_cnx", mkzip(remanifest(patch_prep(f, b"import importlib; importlib.import_module('hel' + 'per')"))), ON)
@case
def N8_from_subprocess_import_run():
    "обход: from subprocess import run"
    both("check_cnx", prep(b"from subprocess import run as _r; _r(['curl', 'http://x'])"), ON)
@case
def N9_subprocess_alias():
    "обход: import subprocess as sp; sp.run"
    both("check_cnx", prep(b"import subprocess as _sp; _sp.run(['curl', 'http://x'])"), ON)
@case
def N10_exec():
    "обход: exec('import socket')"
    both("check_cnx", prep(b"exec('import soc' + 'ket')"), ON)
@case
def N11_getattr_importlib():
    "обход: getattr(importlib, 'import_module')('socket')"
    both("check_cnx", prep(b"import importlib; getattr(importlib, 'import_' + 'module')('socket')"), ON)
@case
def N12_ctypes():
    "ctypes (libc connect) — статически не сетевой модуль"
    both("check_cnx", prep(b"import ctypes; ctypes.CDLL(None)"), ON)
@case
def N13_urlopen_call_no_import():
    "вызов urlopen через атрибут без импорта urllib"
    both("check_cnx", prep(b"import torch; torch.utils.model_zoo.load_url('http://x')"), ON)
@case
def M1(): f = base_files(); f["inference/cnx/MANIFEST.json"] = b'{"sha256": {}}'; both("check_cnx", mkzip(f), ON)
@case
def M2():
    f = base_files(); f["inference/cnx/vendor/relief_ext.py"] += b"\n# changed\n"
    m = json.loads(f["inference/cnx/MANIFEST.json"]); del m["sha256"]["vendor/relief_ext.py"]
    f["inference/cnx/MANIFEST.json"] = json.dumps(m).encode(); both("check_cnx", mkzip(f), ON)
@case
def M3(): f = base_files(); f["inference/cnx/vendor/expds/lidar.py"] += b"\n"; both("check_cnx", mkzip(f), ON)
@case
def M3b_vendor_change_with_rebuilt_manifest():
    "вендор изменён, MANIFEST пересобран самосогласованно → prep_code_sha_expected ≠ → FAIL"
    f = base_files(); f["inference/cnx/vendor/expds/lidar.py"] += b"\n"; both("check_cnx", mkzip(remanifest(f)), ON)
@case
def M3c_prep_code_sha_removed():
    f = base_files(); m = json.loads(f["inference/cnx/MANIFEST.json"]); m.pop("prep_code_sha_expected", None)
    f["inference/cnx/MANIFEST.json"] = json.dumps(m).encode(); both("check_cnx", mkzip(f), ON)
@case
def M5_unlisted_so():
    "лишний .so вне MANIFEST"
    f = base_files(); f["inference/cnx/vendor/evil.so"] = b"\x7fELF"; both("check_cnx", mkzip(f), ON)
@case
def M6_unlisted_readme():
    f = base_files(); f["inference/cnx/README2.md"] = b"x"; both("check_cnx", mkzip(f), ON)
@case
def M6b_soft_name_but_code():
    "обход мягких имён: README_evil.py вне MANIFEST (код, но имя начинается с README)"
    f = base_files(); f["inference/cnx/README_evil.py"] = b"import socket\n"; both("check_cnx", mkzip(f), ON)
@case
def M4(): both("check_config_static", mkzip(dict(base_files(), **{"configs/default.toml": f'[li_cnx]\nenabled = true\nweights = ["{W}"]\n[thresholds_by_markup.li_cnx]\nkurgany_tselye = 0.5\nkurgany_povrezhdennye = 0.5\n'.encode()})))
@case
def W1_weights_wrong_sha():
    f = base_files(); f[W] = b"fake"; both("check_cnx", mkzip(f), ON)
@case
def W2_weights_not_in_wmanifest():
    cfg = dict(ON, li_cnx={"enabled": True, "weights": ["models/cnx/other.pt"]}); both("check_cnx", mkzip(base_files()), cfg)
@case
def W3_no_wmanifest():
    f = base_files(); del f["inference/cnx/WEIGHTS_MANIFEST.json"]; both("check_cnx", mkzip(f), ON)
@case
def T1_no_threshold_group():
    both("check_cnx", mkzip(base_files()), {"li_cnx": {"enabled": True, "weights": [W]}})
@case
def T2_forbidden_group_li():
    both("check_cnx", mkzip(base_files()), {"li_cnx": {"enabled": True, "weights": [W], "threshold_group": "li"}, "thresholds_by_markup": {"li": TH["li_cnx"]}})
@case
def T3_tta_bad():
    both("check_cnx", mkzip(base_files()), {"li_cnx": {"enabled": True, "weights": [W], "tta": 4}, "thresholds_by_markup": TH})
def cfgzip(toml): return mkzip({"configs/default.toml": toml.encode()})
@case
def C1():
    both("check_config_static", cfgzip(f'[li_cnx]\nenabled = false\nweights = ["{W}"]\n'))
@case
def C2():
    z = cfgzip(f'[li_cnx]\nenabled = "false"\nweights = ["{W}"]\n')
    for v in V: show(v, run(v, "check_config_static", z) + run(v, "check_cnx", mkzip(base_files()), {"li_cnx": {"enabled": "false", "weights": [W]}, "thresholds_by_markup": TH}))
@case
def C3():
    sys.path.insert(0, str(SRC)); from expds import lidar_cnx
    print("  pipeline enabled: 0→", lidar_cnx.enabled({"li_cnx": {"enabled": 0, "weights": ["m"]}}),
          "'false'→", lidar_cnx.enabled({"li_cnx": {"enabled": "false", "weights": ["m"]}}),
          "true,[]→", lidar_cnx.enabled({"li_cnx": {"enabled": True, "weights": []}}))
    both("check_config_static", cfgzip(f'[li_cnx]\nenabled = 0\nweights = ["{W}"]\n'))
@case
def S1_smoke_cnx():
    ok = "I cnx_prep x.las: ch=5 dtm за 3 с"
    for tag, log, on, n in [("ok", ok, True, 1), ("нет строки", "I всё хорошо", True, 1),
                            ("веса", ok + "\nE cnx: веса [li_cnx] не загрузились (x)", True, 1),
                            ("пороги", "W [li_cnx]: нет порогов — объекты cnx НЕ выдаются", True, 1),
                            ("выкл", "", False, 1), ("0 LAS", "", True, 0)]:
        for v in V: show(f"{v}:{tag}", run(v, "smoke_cnx", log, on, n))
if __name__ == "__main__":
    for n, f in CASES.items():
        print(f"{n}: {(f.__doc__ or '').strip()}"); f()
