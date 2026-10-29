"""bot-4: ревью 54cfab6 (check_cnx). Старая (4fb79c6) / новая (54cfab6) / слитая с 7878087 / корневая bot-8."""
import hashlib, importlib.util, io, json, sys, zipfile
from pathlib import Path
HERE = Path(__file__).parent
SRC = Path("/workspace/koz3_shared/bot-8/wt-cnx")   # только чтение (54cfab6)
CX = SRC / "inference" / "cnx"
def load(n):
    s = importlib.util.spec_from_file_location(n, HERE / f"{n}.py"); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
V = {n: load(n) for n in ("cs_old", "cs_new", "cs_merged", "cs_root")}
ON = {"li_cnx": {"enabled": True, "weights": ["models/w.pt"]}}

def base_files():
    return {p.relative_to(CX.parents[1]).as_posix(): p.read_bytes() for p in sorted(CX.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts}

def mkzip(files):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        for n, d in files.items(): z.writestr(n, d)
    return zipfile.ZipFile(io.BytesIO(b.getvalue()))

def remanifest(files):
    """Пересобрать MANIFEST так, как это сделал бы build (самосогласованно) — для изменённого кода."""
    m = json.loads(files["inference/cnx/MANIFEST.json"])
    m["sha256"] = {n[len("inference/cnx/"):]: hashlib.sha256(d).hexdigest() for n, d in files.items()
                   if n.startswith("inference/cnx/") and n != "inference/cnx/MANIFEST.json"}
    files["inference/cnx/MANIFEST.json"] = json.dumps(m).encode()
    return files

def patch_prep(files, extra):
    k = "inference/cnx/cnx_prep.py"
    files[k] = files[k].replace(b"import numpy as np\n", b"import numpy as np\n" + extra + b"\n", 1)
    return files

def run(mod, fn, *a):
    m = V[mod]; m.RESULTS.clear()
    try: getattr(m, fn)(*a)
    except Exception as e: return [("EXC", repr(e)[:120])]
    return [(s, msg[:150]) for s, c, msg in m.RESULTS if c in ("cnx", "layout", "config")]

def show(tag, res):
    worst = "FAIL" if any(s == "FAIL" for s, _ in res) else "WARN" if any(s == "WARN" for s, _ in res) else "PASS"
    print(f"  {tag:7s} {worst:5s} " + " | ".join(f"{s}:{m}" for s, m in res if s in ("FAIL", "WARN", "EXC")))
    return worst

CASES = {}
def case(f): CASES[f.__name__] = f; return f

@case
def L1_disabled_evil_py_in_cnx():
    """выкл., в inference/cnx/evil.py с import socket: старая — WARN layout, новая — WARN cnx"""
    f = base_files(); f["inference/cnx/evil.py"] = b"import socket\n"
    z = mkzip(f)
    for v in ("cs_old", "cs_new"):
        r = run(v, "check_layout", z)
        if v != "cs_old": r += run(v, "check_cnx", z, {"li_cnx": {"enabled": False}})
        show(v, r)

@case
def L2_enabled_unreachable_net_file_listed_in_manifest():
    """вкл., evil.py c import socket, внесён в MANIFEST, недостижим из cnx_prep → ничего не ловится"""
    f = remanifest(dict(base_files(), **{"inference/cnx/evil.py": b"import socket\nsocket.create_connection(('x',80))\n"}))
    show("cs_new", run("cs_new", "check_cnx", mkzip(f), ON))

@case
def N1_static_import_requests():
    f = remanifest(patch_prep(base_files(), b"import requests"))
    show("cs_new", run("cs_new", "check_cnx", mkzip(f), ON))

@case
def N2_from_urllib_import_request():
    f = remanifest(patch_prep(base_files(), b"from urllib import request"))
    show("cs_new", run("cs_new", "check_cnx", mkzip(f), ON))

@case
def N3_importlib_import_module_socket():
    f = remanifest(patch_prep(base_files(), b"import importlib; _s = importlib.import_module('soc' + 'ket')"))
    show("cs_new", run("cs_new", "check_cnx", mkzip(f), ON))

@case
def N4_dunder_import():
    f = remanifest(patch_prep(base_files(), b"_s = __import__('socket')"))
    show("cs_new", run("cs_new", "check_cnx", mkzip(f), ON))

@case
def N5_subprocess_curl():
    f = remanifest(patch_prep(base_files(), b"import subprocess; subprocess.run(['curl', 'http://x'])"))
    show("cs_new", run("cs_new", "check_cnx", mkzip(f), ON))

@case
def N6_torch_hub_call():
    """NET_CALLS (torch.hub / urlopen / from_pretrained) основной сканер ловит, check_cnx — нет"""
    f = remanifest(patch_prep(base_files(), b"import torch; torch.hub.load('a/b', 'c')"))
    show("cs_new", run("cs_new", "check_cnx", mkzip(f), ON))

@case
def N7_dynamic_load_of_unreachable_module_with_net():
    f = base_files(); f["inference/cnx/vendor/helper.py"] = b"import socket\n"
    f = remanifest(patch_prep(f, b"import importlib; importlib.import_module('hel' + 'per')"))
    show("cs_new", run("cs_new", "check_cnx", mkzip(f), ON))

@case
def M1_empty_manifest():
    """MANIFEST {"sha256": {}} → PASS «0 файлов совпадают» + WARN, а не FAIL"""
    f = base_files(); f["inference/cnx/MANIFEST.json"] = b'{"sha256": {}}'
    show("cs_new", run("cs_new", "check_cnx", mkzip(f), ON))

@case
def M2_manifest_without_required_file_and_file_changed():
    """relief_ext.py изменён и вычеркнут из MANIFEST → только WARN «вне MANIFEST»"""
    f = base_files(); f["inference/cnx/vendor/relief_ext.py"] += b"\n# changed\n"
    m = json.loads(f["inference/cnx/MANIFEST.json"]); del m["sha256"]["vendor/relief_ext.py"]
    f["inference/cnx/MANIFEST.json"] = json.dumps(m).encode()
    show("cs_new", run("cs_new", "check_cnx", mkzip(f), ON))

@case
def M3_byte_change_with_stale_manifest():
    f = base_files(); f["inference/cnx/vendor/expds/lidar.py"] += b"\n"
    show("cs_new", run("cs_new", "check_cnx", mkzip(f), ON))

@case
def M4_weights_missing_from_zip():
    """веса .pt в MANIFEST не входят: их отсутствие ловит check_config_static (li_cnx.weights)"""
    z = mkzip(dict(base_files(), **{"configs/default.toml": b'[li_cnx]\nenabled = true\nweights = ["models/w.pt"]\n'}))
    for v in ("cs_old", "cs_new"): show(v, run(v, "check_config_static", z))

def cfgzip(toml, with_w):
    f = {"configs/default.toml": toml.encode()}
    if with_w: f["models/w.pt"] = b"x"
    return mkzip(f)

@case
def C1_disabled_missing_weights():
    z = cfgzip('[li_cnx]\nenabled = false\nweights = ["models/w.pt"]\n', False)
    for v in ("cs_old", "cs_new", "cs_merged", "cs_root"): show(v, run(v, "check_config_static", z))

@case
def C2_enabled_string_false_spoof():
    """enabled = "false" (строка) — pipeline: bool("false") = True → cnx работает; чекер тоже считает включённым"""
    z = cfgzip('[li_cnx]\nenabled = "false"\nweights = ["models/w.pt"]\n', False)
    for v in ("cs_new", "cs_merged", "cs_root"):
        show(v, run(v, "check_config_static", z) + run(v, "check_cnx", z, {"li_cnx": {"enabled": "false", "weights": ["models/w.pt"]}}))

@case
def C3_enabled_int_0_and_weights_set():
    z = cfgzip('[li_cnx]\nenabled = 0\nweights = ["models/w.pt"]\n', False)
    sys.path.insert(0, str(SRC))
    from expds import lidar_cnx
    print("  pipeline lidar_cnx.enabled:", lidar_cnx.enabled({"li_cnx": {"enabled": 0, "weights": ["models/w.pt"]}}),
          "| enabled='false':", lidar_cnx.enabled({"li_cnx": {"enabled": "false", "weights": ["m"]}}),
          "| enabled=true, weights=[]:", lidar_cnx.enabled({"li_cnx": {"enabled": True, "weights": []}}))
    show("cs_new", run("cs_new", "check_config_static", z))

@case
def S1_smoke_keywords():
    """ключи смоука: какие строки лога pipeline теперь тревожные"""
    import re
    src = (HERE / "cs_new.py").read_text()
    keys = re.search(r'problems = .*?\(\s*(".*?")\)\)\]', src, re.S).group(1)
    lines = ["W x.las: cnx не сработал (boom) — модель li", "W x.las: cnx пропущен мягким потолком — модель li",
             "E cnx: веса [li_cnx] не загрузились — модель li", "I cnx_prep x.las: ch=5 …"]
    ks = eval("(" + keys + ")")
    for ln in lines: print(f"  {any(k in ln for k in ks)!s:5s} {ln}")

if __name__ == "__main__":
    for n, f in CASES.items():
        print(f"{n}: {(f.__doc__ or '').strip()}"); f()
