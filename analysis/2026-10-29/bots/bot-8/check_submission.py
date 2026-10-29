#!/usr/bin/env python3
"""Проверка архива КОЗ №3 перед отправкой на платформу (bot-8).

    python check_submission.py dist/submission.zip                  # всё, включая CPU-смоук
    python check_submission.py dist/submission.zip --no-smoke       # только статика (~секунды)
    python check_submission.py dist/x.zip --python venv/bin/python --timeout 900 --input /path/small_region

Проверки (каждая печатает PASS / WARN / FAIL):
  zip       — архив читается, CRC, нет абсолютных путей и «..», нет дублей, файлы в корне архива;
  layout    — metadata.json, inference/main.py, configs/default.toml, expds/, inference/lib/expds_core*.so;
  forbidden — нет данных выборок (.las .tif .geojson .npy ...), .git, __pycache__, секретов;
  size      — архив ≤ 2 ГиБ (WARN с 1,9 ГиБ);
  metadata  — image не шаблон, entry_point = «python inference/main.py»;
  config    — TOML читается; пути моделей в архиве; классы/группы/режимы из списков кода; бюджет < 30 мин − запас;
  imports   — модули, достижимые из inference/main.py, импортируются в целевом python; нет сетевых модулей;
  weights   — каждый .pt открывается кодом архива (Segmenter / кроп), LightGBM читается;
  so        — ELF x86-64, без AVX-512 (zmm), версия GLIBC;
  smoke     — inference/main.py на синтетическом регионе (LAS + MagGr + SpOr) на CPU: без сети
              (unshare -rn), с таймаутом, 2 ядра (taskset), nice, контроль RSS; затем формат ответа
              по лимитам compute_metrics.py (≤20000 объектов, Polygon, ≤3000 вершин, 7 классов, ≤100 МБ),
              строгий JSON без NaN, пустой HOME/кэш (ничего не скачано), нет временных файлов.
Код возврата: 0 — FAIL нет; 1 — есть FAIL; 2 — ошибка аргументов.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import tomllib
import zipfile
from pathlib import Path, PurePosixPath

LIMIT_ZIP = 2 * 1024 ** 3
WARN_ZIP = int(1.9 * 1024 ** 3)
PLATFORM_S = 1800
# Лимиты materials/compute_metrics.py:200-204
MAX_FEATURES, MAX_POINTS, MAX_FILE_MB = 20000, 3000, 100
CLASSES = ("kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha", "arkhitektury",
           "object_poly", "finds_points", "fortifikatsii")
FORBIDDEN_EXT = {".las", ".laz", ".tif", ".tiff", ".geojson", ".npy", ".npz", ".csv", ".jpg", ".jpeg", ".png",
                 ".jp2", ".shp", ".gpkg", ".zip", ".ipynb", ".pkl", ".parquet"}
FORBIDDEN_PARTS = {".git", ".venv", "venv", "__pycache__", "node_modules", ".ipynb_checkpoints"}
SECRET_NAMES = {".env", "kaggle.json", "id_rsa", "id_ed25519", ".netrc", "credentials", ".git-credentials"}
NET_MODULES = {"socket", "requests", "httpx", "aiohttp", "urllib3", "ftplib", "smtplib", "paramiko", "boto3",
               "huggingface_hub", "wandb", "mlflow", "kaggle", "http.client", "urllib.request", "websocket"}
NET_CALLS = {"load_state_dict_from_url", "hf_hub_download", "from_pretrained", "download_url_to_file", "urlopen"}

RESULTS: list[tuple[str, str, str]] = []


def report(status: str, check: str, msg: str) -> None:
    RESULTS.append((status, check, msg))
    color = {"PASS": "\033[32m", "WARN": "\033[33m", "FAIL": "\033[31m", "INFO": "\033[36m"}.get(status, "")
    end = "\033[0m" if color and sys.stdout.isatty() else ""
    print(f"{color if end else ''}[{status}]{end} {check:9s} {msg}", flush=True)


# ---------------------------------------------------------------- архив
def check_zip(zpath: Path) -> zipfile.ZipFile | None:
    try:
        z = zipfile.ZipFile(zpath)
    except (zipfile.BadZipFile, OSError) as exc:
        report("FAIL", "zip", f"архив не читается: {exc}")
        return None
    bad = z.testzip()
    if bad:
        report("FAIL", "zip", f"CRC не сошёлся: {bad}")
    names = z.namelist()
    unsafe = [n for n in names if n.startswith("/") or "\\" in n or ".." in PurePosixPath(n).parts]
    if unsafe:
        report("FAIL", "zip", f"опасные пути: {unsafe[:5]}")
    dups = {n for n in names if names.count(n) > 1}
    if dups:
        report("FAIL", "zip", f"дубли имён: {sorted(dups)[:5]}")
    links = [i.filename for i in z.infolist() if stat.S_ISLNK(i.external_attr >> 16)]
    if links:
        report("FAIL", "zip", f"симлинки в архиве: {links[:5]}")
    if not (bad or unsafe or dups or links):
        report("PASS", "zip", f"{len(names)} файлов, CRC ок, пути безопасны")
    return z


def check_layout(z: zipfile.ZipFile) -> None:
    names = set(z.namelist())
    need = ["metadata.json", "inference/main.py", "configs/default.toml", "expds/__init__.py", "expds/pipeline.py"]
    miss = [n for n in need if n not in names]
    if miss:
        tops = sorted({n.split("/")[0] for n in names})
        hint = f" (верхний уровень архива: {tops[:5]} — файлы должны лежать в корне, не в подпапке)" if \
            any(f"{t}/inference/main.py" in names for t in tops) else ""
        report("FAIL", "layout", f"нет обязательных файлов: {miss}{hint}")
    else:
        report("PASS", "layout", "metadata.json, inference/main.py, configs/default.toml, expds/ на месте")
    so = [n for n in names if n.startswith("inference/lib/expds_core") and n.endswith(".so")]
    if so:
        report("PASS", "layout", f"ускоритель: {so[0]}")
    else:
        report("WARN", "layout", "нет inference/lib/expds_core*.so — пойдёт запасной путь NumPy (медленнее)")
    extra_entry = [n for n in names if n.startswith("inference/") and n.endswith(".py") and n != "inference/main.py"]
    if extra_entry:
        report("WARN", "layout", f"лишние .py в inference/: {extra_entry[:5]}")


def check_forbidden(z: zipfile.ZipFile) -> None:
    hits, warn = [], []
    for n in z.namelist():
        p = PurePosixPath(n)
        if p.suffix.lower() in FORBIDDEN_EXT:
            hits.append(n)
        elif p.name in SECRET_NAMES or p.suffix in {".pem", ".key"}:
            hits.append(n)
        elif FORBIDDEN_PARTS & set(p.parts[:-1]) or p.suffix == ".pyc":
            warn.append(n)
    if hits:
        report("FAIL", "forbidden", f"{len(hits)} запрещённых файлов (данные/секреты, п. 7.7 ТР): {hits[:8]}")
    else:
        report("PASS", "forbidden", "нет данных выборок и секретов")
    if warn:
        report("WARN", "forbidden", f"{len(warn)} лишних служебных файлов: {warn[:5]}")


def check_size(zpath: Path, z: zipfile.ZipFile) -> None:
    size = zpath.stat().st_size
    raw = sum(i.file_size for i in z.infolist())
    msg = f"архив {size / 2**20:.1f} МиБ ({size} Б), распакованный {raw / 2**20:.1f} МиБ, лимит 2048 МиБ"
    if size > LIMIT_ZIP:
        report("FAIL", "size", msg)
    elif size > WARN_ZIP:
        report("WARN", "size", msg + " — запас < 100 МиБ")
    else:
        report("PASS", "size", msg)
    big = sorted(((i.file_size, i.filename) for i in z.infolist()), reverse=True)[:3]
    report("INFO", "size", "крупнейшие: " + ", ".join(f"{n} {s / 2**20:.0f} МиБ" for s, n in big))


def check_metadata(z: zipfile.ZipFile) -> None:
    try:
        meta = json.loads(z.read("metadata.json"))
    except KeyError:
        return
    except Exception as exc:
        report("FAIL", "metadata", f"metadata.json не JSON: {exc}")
        return
    image, entry = str(meta.get("image") or ""), str(meta.get("entry_point") or "")
    ok = True
    if not image or "<" in image or image.lower() in {"todo", "image", "your/image:tag"} or ":" not in image:
        report("FAIL", "metadata", f"image не задан или шаблон: {image!r}")
        ok = False
    elif image.endswith(":latest"):
        report("WARN", "metadata", f"image с тегом latest — не воспроизводимо: {image}")
    if entry.split() != ["python", "inference/main.py"]:
        report("FAIL" if "inference/main.py" not in entry else "WARN", "metadata",
               f"entry_point {entry!r}, ожидается 'python inference/main.py'")
        ok = False
    if ok:
        report("PASS", "metadata", f"image={image}, entry_point={entry!r}")


# ---------------------------------------------------------------- конфиг (статически)
def _strings(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _strings(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _strings(v, f"{path}[{i}]")
    elif isinstance(obj, str):
        yield path, obj


def check_config_static(z: zipfile.ZipFile) -> dict | None:
    names = set(z.namelist())
    cfg = None
    for n in sorted(x for x in names if x.startswith("configs/") and x.endswith(".toml")):
        try:
            c = tomllib.loads(z.read(n).decode("utf-8"))
        except Exception as exc:
            report("FAIL", "config", f"{n} не читается как TOML: {exc}")
            continue
        if n == "configs/default.toml":
            cfg = c
    if cfg is None:
        return None
    refs = sorted({(k, v) for k, v in _strings(cfg) if v.startswith("models/") or
                   PurePosixPath(v).suffix in {".pt", ".pth", ".onnx", ".ckpt"}})
    verify = cfg.get("verify") or {}
    pixgbm = (cfg.get("geophys") or {}).get("pixgbm") or {}
    missing = [f"{k}={v}" for k, v in refs if v not in names
               and not (k.startswith("verify.") and not verify.get("enabled"))
               and not (k.startswith("geophys.pixgbm.") and not pixgbm.get("enabled"))   # bot-3 PIXGBM выкл.
               and not (k == "geophys.pixgbm.model_dir" and any(n.startswith(v.rstrip("/") + "/") for n in names))]
    if missing:
        report("FAIL", "config", f"файлы из default.toml отсутствуют в архиве (модель молча выключится, "
                                 f"pipeline.py:98-101 на 24bd916): {missing}")
    else:
        report("PASS", "config", f"все {len(refs)} путей к весам из default.toml есть в архиве")
    models_in_zip = {n for n in names if n.startswith("models/") and not n.endswith("/")}
    unused = sorted(models_in_zip - {v for _, v in refs})
    if unused:
        report("WARN", "config", f"в архиве веса, на которые нет ссылок в конфиге (лишний объём): {unused[:6]}")
    t = cfg.get("time") or {}
    budget, margin = float(t.get("budget_s", 1500)), float(t.get("guard_margin_s", 180))
    if budget + 0 >= PLATFORM_S - margin:
        report("FAIL", "config", f"[time] budget_s={budget} ≥ 1800 − guard_margin_s={margin}: сторож сработает раньше бюджета")
    elif not t.get("watchdog", True):
        report("WARN", "config", "[time] watchdog=false — при превышении 30 мин ответа не будет")
    else:
        report("PASS", "config", f"[time] budget_s={budget:g}, guard_margin_s={margin:g}, watchdog включён")
    bad, off = [], []
    tables = [("thresholds", cfg.get("thresholds") or {})] + \
        [(f"thresholds_by_markup.{g}", t or {}) for g, t in (cfg.get("thresholds_by_markup") or {}).items()]
    for name, tbl in tables:
        for k, v in tbl.items():
            if k not in CLASSES:
                bad.append(f"{name}.{k} (нет такого класса — порог молча игнорируется)")
            elif isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                bad.append(f"{name}.{k}={v!r}")
            elif v > 1:
                off.append(f"{name}.{k}={v}")
    dt = cfg.get("default_threshold", 0.15)
    if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not 0 <= dt <= 1:
        bad.append(f"default_threshold={dt!r}")
    if off:
        report("INFO", "config", f"порог > 1 — класс в группе выключен: {off}")
    only = (cfg.get("post") or {}).get("only_classes")
    if only is not None and (not isinstance(only, list) or set(only) - set(CLASSES)):
        bad.append(f"post.only_classes={only}")
    for mt, spec in (cfg.get("models") or {}).items():
        for it in spec if isinstance(spec, list) else [spec]:
            if isinstance(it, dict):
                extra = set(it.get("classes") or []) - set(CLASSES)
                if extra:
                    bad.append(f"models.{mt}.classes {sorted(extra)}")
    if bad:
        report("FAIL", "config", "ошибки значений: " + "; ".join(bad[:10]))
    else:
        report("PASS", "config", "пороги в [0,1], имена классов из 7 допустимых")
    return cfg


# ---------------------------------------------------------------- импорты (статически)
def _module_of(rel: str) -> str | None:
    p = PurePosixPath(rel)
    if p.parts[0] == "expds":
        parts = list(p.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts)
    return None


def scan_imports(root: Path) -> tuple[dict, dict]:
    """{модуль: (третьи пакеты, локальные модули, сетевые находки)}; граф достижимости от main.py."""
    info = {}
    files = {"__main__": root / "inference" / "main.py"}
    for f in (root / "expds").rglob("*.py"):
        files[_module_of(f.relative_to(root).as_posix())] = f
    local = set(files) | {"expds_core"}
    for mod, f in files.items():
        tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        pkg = mod.rsplit(".", 1)[0] if mod != "__main__" and not f.name == "__init__.py" else mod
        third, deps, net = set(), set(), []
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = pkg.split(".")
                    base = base[: len(base) - node.level + 1]
                    head = ".".join(base + ([node.module] if node.module else []))
                    names = [head] + [f"{head}.{a.name}" for a in node.names]
                else:
                    names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            elif isinstance(node, ast.Call):
                fn = node.func
                nm = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
                if nm in NET_CALLS:
                    net.append((node.lineno, f"вызов {nm}()"))
                if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Attribute) and fn.value.attr == "hub":
                    net.append((node.lineno, f"torch.hub.{nm}()"))
                for kw in node.keywords:
                    if kw.arg == "pretrained" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                        net.append((node.lineno, "pretrained=True (скачивание весов)"))
                    if kw.arg in ("encoder_weights", "weights") and isinstance(kw.value, ast.Constant) \
                            and isinstance(kw.value.value, str) and kw.value.value.lower() in ("imagenet", "default"):
                        net.append((node.lineno, f"{kw.arg}={kw.value.value!r} (скачивание весов)"))
            for n in names:
                if not n:
                    continue
                if n in local or n.split(".")[0] in ("expds", "expds_core"):
                    if n in local:
                        deps.add(n)
                    continue
                top = n.split(".")[0]
                if n in NET_MODULES or top in NET_MODULES or ".".join(n.split(".")[:2]) in NET_MODULES:
                    net.append((node.lineno, f"import {n}"))
                if top not in sys.stdlib_module_names:
                    third.add(top)
                elif top in ("urllib", "http") and ".".join(n.split(".")[:2]) in NET_MODULES:
                    pass
        info[mod] = (third, deps, net, f)
    reach, todo = set(), ["__main__", "expds.pipeline"]
    while todo:
        m = todo.pop()
        if m in reach or m not in info:
            continue
        reach.add(m)
        todo += list(info[m][1])
        parts = m.split(".")
        for i in range(1, len(parts)):          # импорт подмодуля исполняет __init__ пакета
            todo.append(".".join(parts[:i]))
    return info, reach


def check_imports_static(root: Path) -> tuple[set, set]:
    info, reach = scan_imports(root)
    third_reach = set().union(*(info[m][0] for m in reach))
    fails, warns = [], []
    for m, (_t, _d, net, f) in sorted(info.items()):
        for line, what in net:
            loc = f"{f.relative_to(root)}:{line} {what}"
            is_import = what.startswith("import ")
            (fails if (m in reach and is_import) else warns).append(loc)
    if fails:
        report("FAIL", "imports", "сетевые модули в коде инференса: " + "; ".join(fails[:8]))
    else:
        report("PASS", "imports", f"в {len(reach)} модулях, достижимых из inference/main.py, нет сетевых импортов")
    if warns:
        report("WARN", "imports", "проверить вручную (вероятно, только обучение): " + "; ".join(warns[:8]))
    report("INFO", "imports", f"сторонние пакеты инференса: {sorted(third_reach)}")
    return reach, third_reach


# ---------------------------------------------------------------- проба кодом архива (подпроцесс)
PROBE = r'''
import importlib, json, sys, traceback, tomllib
from pathlib import Path
root = Path(sys.argv[1]); reach = json.loads(sys.argv[2]); third = json.loads(sys.argv[3]); load_w = sys.argv[4] == "1"
sys.path[:0] = [str(root), str(root / "inference" / "lib")]
out = {"import_fail": {}, "third_fail": {}, "weights": [], "cfg": [], "dists": {}}
import importlib.util, importlib.metadata as md
pd = md.packages_distributions()
for t in third:
    try:
        if importlib.util.find_spec(t) is None:
            out["third_fail"][t] = "не найден"
        else:
            out["dists"][t] = pd.get(t, [t])
    except Exception as e:
        out["third_fail"][t] = repr(e)
for m in reach:
    if m == "__main__":
        continue
    try:
        importlib.import_module(m)
    except BaseException as e:
        out["import_fail"][m] = "".join(traceback.format_exception_only(type(e), e)).strip()
try:
    import expds_core
    out["backend"] = str(getattr(expds_core, "__file__", ""))
except BaseException as e:
    out["backend_err"] = repr(e)
try:
    from expds.constants import CLASSES
    from expds.geophys import MAP_MODES
    from expds.pipeline import model_specs, model_files
    cfg = tomllib.loads((root / "configs" / "default.toml").read_text(encoding="utf-8"))
    for name, sub in (cfg.get("geophys") or {}).items():
        if name == "pixgbm" and isinstance(sub, dict):   # bot-3: maps — список методов, а не режим карт
            bad = [m for m in (sub.get("maps") or []) if m not in ("maggr", "etgr", "terfl")]
            if bad:
                out["cfg"].append(f"geophys.pixgbm.maps: неизвестные методы {bad} (допустимы maggr, etgr, terfl)")
            continue
        if isinstance(sub, dict) and "maps" in sub and sub["maps"] not in MAP_MODES:
            out["cfg"].append(f"geophys.{name}.maps={sub['maps']!r} не из {MAP_MODES} — ValueError, регион без геофизики")
    if "maps" in (cfg.get("geophys") or {}) and cfg["geophys"]["maps"] not in MAP_MODES:
        out["cfg"].append(f"geophys.maps={cfg['geophys']['maps']!r} не из {MAP_MODES}")
    if load_w:
        import gc, torch
        from expds.infer_nn import Segmenter
        models = cfg.get("models") or {}
        for mt in models:
            for path, classes in model_specs(models, mt):
                for f in model_files(path):
                    rec = {"markup": mt, "file": f}
                    try:
                        s = Segmenter(root / f, device="cpu")
                        rec["classes"] = list(s.classes)
                        rec["ok"] = True
                        if classes and set(classes) - set(s.classes):
                            rec["warn"] = f"config classes {sorted(set(classes) - set(s.classes))} нет у модели"
                        del s
                    except Exception as e:
                        rec["ok"] = False; rec["err"] = repr(e)[:300]
                    gc.collect()
                    out["weights"].append(rec)
        lg = (cfg.get("candidates") or {}).get("lgbm_model")
        if lg:
            try:
                import lightgbm as lgb
                b = lgb.Booster(model_file=str(root / lg))
                out["weights"].append({"markup": "lgbm", "file": lg, "ok": True, "classes": [f"{b.num_trees()} деревьев"]})
            except Exception as e:
                out["weights"].append({"markup": "lgbm", "file": lg, "ok": False, "err": repr(e)[:300]})
        v = cfg.get("verify") or {}
        if v.get("enabled") and v.get("weights"):
            try:
                from expds.crop_model import _network
                blob = torch.load(root / v["weights"], map_location="cpu", weights_only=False)
                _network(str(blob.get("encoder") or "resnet18"), False).load_state_dict(blob["state_dict"])
                out["weights"].append({"markup": "verify", "file": v["weights"], "ok": True, "classes": []})
            except Exception as e:
                out["weights"].append({"markup": "verify", "file": v["weights"], "ok": False, "err": repr(e)[:300]})
except BaseException as e:
    out["probe_err"] = traceback.format_exc()[-1500:]
print("PROBE_JSON=" + json.dumps(out, ensure_ascii=False))
'''


def sandbox_cmd(cmd: list[str], cpus: str, net: bool) -> list[str]:
    pre = []
    if net and shutil.which("unshare") and subprocess.run(["unshare", "-rn", "true"], capture_output=True).returncode == 0:
        pre = ["unshare", "-rn"]
    if shutil.which("nice"):
        pre += ["nice", "-n", "10"]
    if cpus and shutil.which("taskset"):
        pre += ["taskset", "-c", cpus]
    return pre + cmd


def child_env(home: Path, threads: int) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("EXPDS_", "PYTHONPATH", "CUDA_VISIBLE"))}
    env.update(HOME=str(home), XDG_CACHE_HOME=str(home / ".cache"), TORCH_HOME=str(home / ".cache" / "torch"),
               HF_HOME=str(home / ".cache" / "hf"), MPLCONFIGDIR=str(home / ".mpl"),
               PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               OPENBLAS_NUM_THREADS=str(threads), RAYON_NUM_THREADS=str(threads), CUDA_VISIBLE_DEVICES="",
               EXPDS_DEVICE="cpu", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", NO_PROXY="*")
    return env


#: Пакеты, которые код импортирует только лениво (внутри функций) и с фолбэком при отсутствии:
#: psutil — geo_pixgbm.process_t0 (иначе момент импорта модуля), tl2cgen/lleaves — opt-in бэкенды PIXGBM
#: (иначе lightgbm). Нет в окружении → WARN, а не FAIL, если на уровне модуля их никто не импортирует.
OPTIONAL_LAZY = {"psutil", "tl2cgen", "lleaves"}


def _module_level_imports(root: Path, reach: set) -> set:
    """Верхние имена третьих пакетов, импортируемых вне тел функций в модулях из reach."""
    info, _ = scan_imports(root)
    top = set()
    for m in reach:
        if m not in info:
            continue
        tree = ast.parse(info[m][3].read_text(encoding="utf-8"))
        todo = list(tree.body)
        while todo:
            n = todo.pop()
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            if isinstance(n, ast.Import):
                top |= {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom) and not n.level and n.module:
                top.add(n.module.split(".")[0])
            todo += list(ast.iter_child_nodes(n))
    return top


def run_probe(root: Path, py: str, reach: set, third: set, args) -> None:
    home = Path(tempfile.mkdtemp(prefix="b8home_"))
    cmd = sandbox_cmd([py, "-c", PROBE, str(root), json.dumps(sorted(reach)), json.dumps(sorted(third)),
                       "1" if args.weights else "0"], args.cpus, not args.allow_net)
    t0 = time.monotonic()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout, env=child_env(home, args.threads),
                           cwd=str(root))
    except subprocess.TimeoutExpired:
        report("FAIL", "imports", f"проба импорта/весов не уложилась в {args.timeout} с")
        return
    line = next((x for x in r.stdout.splitlines() if x.startswith("PROBE_JSON=")), None)
    if line is None:
        report("FAIL", "imports", f"проба упала (код {r.returncode}): {(r.stderr or r.stdout)[-800:]}")
        return
    out = json.loads(line[len("PROBE_JSON="):])
    dt = time.monotonic() - t0
    eager = _module_level_imports(root, reach) if set(out["third_fail"]) & OPTIONAL_LAZY else set()
    opt = {k: v for k, v in out["third_fail"].items() if k in OPTIONAL_LAZY and k not in eager}
    out["third_fail"] = {k: v for k, v in out["third_fail"].items() if k not in opt}
    if opt:
        report("WARN", "imports", f"не установлены в {py} необязательные ленивые пакеты (есть фолбэк): {opt}")
    if out["third_fail"]:
        report("FAIL", "imports", f"не установлены в {py}: {out['third_fail']}")
    if out["import_fail"]:
        report("FAIL", "imports", "модули инференса не импортируются: " +
               "; ".join(f"{m}: {e}" for m, e in out["import_fail"].items())[:1200])
    if not out["third_fail"] and not out["import_fail"]:
        report("PASS", "imports", f"{len(reach) - 1} модулей и {len(third)} сторонних пакетов импортируются без сети ({dt:.1f} с)")
    if args.lock:
        lock = Path(args.lock).read_text(encoding="utf-8").lower().replace("_", "-")
        pinned = {ln.split("==")[0].strip() for ln in lock.splitlines() if "==" in ln and not ln.lstrip().startswith("#")}
        notpinned = sorted({d.lower().replace("_", "-") for t, ds in out["dists"].items() for d in ds} - pinned - {"torch", "torchvision"})
        if notpinned:
            report("FAIL", "imports", f"пакеты не из {Path(args.lock).name} (в образе их может не быть): {notpinned}")
        else:
            report("PASS", "imports", f"все сторонние пакеты закреплены в {Path(args.lock).name}")
    if "backend_err" in out:
        report("WARN", "so", f"expds_core не импортируется: {out['backend_err']}")
    if out.get("probe_err"):
        report("FAIL", "config", f"проба конфига упала: {out['probe_err']}")
    for msg in out["cfg"]:
        report("FAIL", "config", msg)
    if args.weights:
        bad = [w for w in out["weights"] if not w["ok"]]
        warn = [w for w in out["weights"] if w.get("warn")]
        for w in bad:
            report("FAIL", "weights", f"{w['markup']}: {w['file']} не открывается: {w.get('err')}")
        for w in warn:
            report("WARN", "weights", f"{w['markup']}: {w['file']}: {w['warn']}")
        if not bad:
            report("PASS", "weights", f"{len(out['weights'])} файлов весов открываются кодом архива на CPU")
    shutil.rmtree(home, ignore_errors=True)


# ---------------------------------------------------------------- .so
def check_so(root: Path) -> None:
    for so in sorted((root / "inference" / "lib").glob("expds_core*.so")):
        head = so.read_bytes()[:20]
        if head[:4] != b"\x7fELF" or head[4] != 2 or int.from_bytes(head[18:20], "little") != 62:
            report("FAIL", "so", f"{so.name}: не ELF64 x86-64")
            continue
        data = so.read_bytes()
        import re
        glibc = sorted({tuple(map(int, m.decode().split(".")[1:])) for m in re.findall(rb"GLIBC_2\.[0-9]+(?:\.[0-9]+)?", data)})
        gl = "2." + ".".join(map(str, glibc[-1][0:])) if glibc else "?"
        msg = f"{so.name}: ELF x86-64, максимум GLIBC_{gl}"
        if shutil.which("objdump"):
            dis = subprocess.run(["objdump", "-d", "--no-show-raw-insn", str(so)], capture_output=True, text=True).stdout
            zmm = dis.count("%zmm")
            if zmm:
                report("FAIL", "so", f"{msg}; {zmm} инструкций AVX-512 (zmm) — SIGILL на CPU без AVX-512")
                continue
            msg += f"; AVX-512 нет, ymm-инструкций {dis.count('%ymm')}"
        report("PASS", "so", msg)


# ---------------------------------------------------------------- смоук
GEN = r'''
import json, sys
from pathlib import Path
import numpy as np
root = Path(sys.argv[1]); X0, Y0 = 500000.0, 6100000.0; UTM = "EPSG:32636"
MOUNDS = [(30.0, 40.0, 1.2, 5.0), (70.0, 60.0, 0.8, 4.0), (45.0, 85.0, 1.0, 6.0)]
def surf(x, y):
    z = 150.0 + 0.02 * x + 0.01 * y
    for mx, my, h, r in MOUNDS:
        z = z + h * np.exp(-((x - mx) ** 2 + (y - my) ** 2) / (2 * (r / 2) ** 2))
    return z
import laspy, rasterio
from pyproj import CRS
from rasterio.transform import from_origin
r1 = root / "region_0001"
rng = np.random.default_rng(0); n = 120000
x = rng.uniform(0, 100, n); y = rng.uniform(0, 100, n); z = surf(x, y)
cls = np.full(n, 2, np.uint8); veg = rng.random(n) < 0.1; z = np.where(veg, z + rng.uniform(2, 10, n), z); cls[veg] = 5
hdr = laspy.LasHeader(point_format=6, version="1.4"); hdr.scales = np.array([0.001] * 3); hdr.offsets = np.array([X0, Y0, 0.0])
hdr.add_crs(CRS.from_user_input(UTM)); las = laspy.LasData(hdr); las.x, las.y, las.z = x + X0, y + Y0, z; las.classification = cls
p = r1 / "01_region_0001_La_точки" / "01_region_0001_La_точки.las"; p.parent.mkdir(parents=True); las.write(str(p))
(r1 / "UTM.json").write_text(json.dumps({"utm": "36N"}), encoding="utf-8")
# SpOr: RGB 0,5 м с тёмными кругами на местах курганов
res = 0.5; m = int(100 / res); yy, xx = (np.mgrid[0:m, 0:m] + 0.5) * res
img = np.stack([np.full((m, m), v, np.float32) for v in (120, 130, 110)])
for mx, my, h, r in MOUNDS:
    img -= 40 * (((xx - mx) ** 2 + ((100 - yy) - my) ** 2) < r ** 2)
p = r1 / "02_region_0001_SpOr_снимок" / "02_region_0001_SpOr.tif"; p.parent.mkdir(parents=True)
with rasterio.open(p, "w", driver="GTiff", height=m, width=m, count=3, dtype="uint8", transform=from_origin(X0, Y0 + 100, res, res), crs=UTM) as ds:
    ds.write(np.clip(img, 0, 255).astype(np.uint8))
# MagGr: диполи, 0,25 м
r2 = root / "region_0002"; res = 0.25; m = int(40 / res); yy, xx = np.mgrid[0:m, 0:m] * res
a = np.random.default_rng(1).normal(0, 1.0, (m, m)).astype(np.float32)
for px, py in [(8.0, 9.0), (25.0, 30.0), (31.0, 12.0)]:
    a += 25 * np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / 0.18); a -= 12 * np.exp(-((xx - px - 0.8) ** 2 + (yy - py) ** 2) / 0.18)
p = r2 / "07_region_0002_MagGr_карты" / "01_region_0002_MagGr.tif"; p.parent.mkdir(parents=True)
with rasterio.open(p, "w", driver="GTiff", height=m, width=m, count=1, dtype="float32", transform=from_origin(X0, Y0 + 40, res, res), crs=UTM) as ds:
    ds.write(a, 1)
(r2 / "UTM.json").write_text(json.dumps({"utm": "36N"}), encoding="utf-8")
print("ok")
'''


def _tree_rss_kb(pid: int) -> int:
    kids: dict[int, list[int]] = {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                with open(f"/proc/{d}/stat", "rb") as fh:
                    ppid = int(fh.read().rsplit(b")", 1)[1].split()[1])
                kids.setdefault(ppid, []).append(int(d))
            except (OSError, ValueError, IndexError):
                pass
    total, todo = 0, [pid]
    while todo:
        p = todo.pop()
        todo += kids.get(p, [])
        try:
            with open(f"/proc/{p}/status") as fh:
                for ln in fh:
                    if ln.startswith("VmRSS:"):
                        total += int(ln.split()[1])
        except OSError:
            pass
    return total


def validate_output(path: Path) -> None:
    if not path.is_file():
        report("FAIL", "smoke", f"нет файла ответа {path}")
        return
    size_mb = path.stat().st_size / 2 ** 20
    text = path.read_text(encoding="utf-8")

    def no_const(c):
        raise ValueError(f"нестандартная константа JSON {c}")
    try:
        fc = json.loads(text, parse_constant=no_const)
    except ValueError as exc:
        report("FAIL", "format", f"ответ — не строгий JSON: {exc}")
        return
    errs, by_class = [], {}
    if fc.get("type") != "FeatureCollection":
        errs.append("type != FeatureCollection")
    feats = fc.get("features") or []
    if len(feats) > MAX_FEATURES:
        errs.append(f"объектов {len(feats)} > {MAX_FEATURES}")
    if size_mb > MAX_FILE_MB:
        errs.append(f"файл {size_mb:.1f} МБ > {MAX_FILE_MB}")
    crs = (fc.get("crs") or {}).get("properties", {}).get("name")
    for i, f in enumerate(feats):
        pr, g = f.get("properties") or {}, f.get("geometry") or {}
        c = pr.get("class_name")
        if c not in CLASSES:
            errs.append(f"#{i}: class_name {c!r}")
            continue
        by_class[c] = by_class.get(c, 0) + 1
        if g.get("type") != "Polygon":
            errs.append(f"#{i}: geometry {g.get('type')!r}")
            continue
        rings = g.get("coordinates") or []
        npts = sum(len(r) for r in rings)
        if not rings or npts > MAX_POINTS:
            errs.append(f"#{i}: вершин {npts}")
        for r in rings:
            if len(r) < 4 or r[0] != r[-1]:
                errs.append(f"#{i}: кольцо не замкнуто/короче 4 точек")
                break
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for xy in r for v in xy[:2]):
                errs.append(f"#{i}: нечисловые координаты")
                break
            if any(abs(xy[0]) > 20037508.35 or abs(xy[1]) > 20048966.11 for xy in r):
                errs.append(f"#{i}: координаты вне EPSG:3857")
                break
        conf = pr.get("confidence")
        if conf is not None and not (isinstance(conf, (int, float)) and 0 <= conf <= 1):
            errs.append(f"#{i}: confidence {conf}")
    if errs:
        report("FAIL", "format", f"{len(errs)} нарушений: {errs[:6]}")
    else:
        report("PASS", "format", f"FeatureCollection, {len(feats)} Polygon (≤{MAX_FEATURES}), {size_mb:.3f} МБ, "
                                 f"по классам {by_class}{'' if not crs else ', crs ' + crs}")
    if not feats:
        report("WARN" if ALLOW_EMPTY else "FAIL", "format",
               "ответ пуст — на синтетике ожидаются объекты (геофизика/лидар); пустой ответ тратит попытку дня. "
               "Проверьте лог (--allow-empty — понизить до WARN)")


ALLOW_EMPTY = False


def smoke(z: zipfile.ZipFile, args) -> None:
    work = Path(tempfile.mkdtemp(prefix="b8smoke_", dir=args.workdir))
    code, home, out_dir = work / "code", work / "home", work / "out"
    for d in (home, out_dir):
        d.mkdir()
    z.extractall(code)
    # Платформа: --read-only; код пишет только рядом с ответом (main.py:54-70)
    for p in sorted(code.rglob("*"), reverse=True):
        p.chmod(0o555 if p.is_dir() else 0o444)
    code.chmod(0o555)
    if args.input:
        inp = Path(args.input).resolve()
    else:
        inp = work / "input"
        g = subprocess.run([args.python, "-c", GEN, str(inp)], capture_output=True, text=True, timeout=300,
                           env=child_env(home, args.threads))
        if g.returncode:
            report("FAIL", "smoke", f"синтетика не создана (нужны numpy/laspy/rasterio/pyproj): {g.stderr[-600:]}")
            return
    output = out_dir / "result.geojson"
    cmd = sandbox_cmd([args.python, "inference/main.py", str(inp), str(output)], args.cpus, not args.allow_net)
    isolated = cmd[:2] == ["unshare", "-rn"]
    report("INFO", "smoke", f"запуск: {' '.join(cmd[:8])} ... (сеть {'отключена' if isolated else 'НЕ изолирована'}, "
                            f"таймаут {args.timeout} с, лимит RSS {args.max_rss_mb} МиБ)")
    if not isolated and not args.allow_net:
        report("WARN", "smoke", "unshare -rn недоступен — офлайн проверен только статически")
    log_path = work / "run.log"
    t0, peak, killed = time.monotonic(), 0, None
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.Popen(cmd, cwd=str(code), stdout=logf, stderr=subprocess.STDOUT,
                                env=child_env(home, args.threads), start_new_session=True)
        while proc.poll() is None:
            time.sleep(0.25)
            peak = max(peak, _tree_rss_kb(proc.pid))
            if peak > args.max_rss_mb * 1024:
                killed = f"RSS {peak // 1024} МиБ > {args.max_rss_mb}"
            elif time.monotonic() - t0 > args.timeout:
                killed = f"таймаут {args.timeout} с"
            if killed:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                break
    wall = time.monotonic() - t0
    log = log_path.read_text(encoding="utf-8", errors="replace")
    if killed:
        report("FAIL", "smoke", f"прогон снят: {killed}; хвост лога:\n{log[-1500:]}")
    elif proc.returncode != 0:
        report("FAIL", "smoke", f"код возврата {proc.returncode} за {wall:.1f} с; хвост лога:\n{log[-1500:]}")
    else:
        report("PASS", "smoke", f"main.py завершился за {wall:.1f} с, пик RSS {peak / 1024:.0f} МиБ, "
                                f"CPU {args.cpus or 'все'}")
    for ln in log.splitlines():
        if "фазы:" in ln or "общее время" in ln or "записано" in ln or "ускоритель" in ln:
            report("INFO", "smoke", ln.split(": ", 1)[-1][:300] if "INFO" in ln else ln[:300])
    problems = [ln for ln in log.splitlines() if any(k in ln for k in (
        "Traceback", "сбой пайплайна", "не найдена:", "файла весов нет", "не загрузилась", "ошибка модели",
        "упала", "запасной детектор", "MemoryError"))]
    if problems:
        report("WARN" if "сбой пайплайна" not in log else "FAIL", "smoke",
               f"{len(problems)} тревожных строк лога: " + " | ".join(p[-200:] for p in problems[:5]))
    if not killed:
        validate_output(output)
        leftovers = [p.name for p in out_dir.iterdir() if p.name != "result.geojson"]
        if leftovers:
            report("WARN", "smoke", f"рядом с ответом остались файлы: {leftovers}")
        cached = [str(p.relative_to(home)) for p in home.rglob("*") if p.is_file()]
        if cached:
            report("FAIL", "smoke", f"во время прогона что-то записано/скачано в HOME/кэш: {cached[:5]}")
        else:
            report("PASS", "smoke", "HOME и кэши пусты: ничего не скачивалось и не кэшировалось вне вывода")
    if args.keep:
        report("INFO", "smoke", f"рабочая папка сохранена: {work}")
    else:
        for p in code.rglob("*"):
            p.chmod(0o755 if p.is_dir() else 0o644)
        code.chmod(0o755)
        shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Проверка архива КОЗ №3 перед отправкой (PASS/FAIL, код ≠0 при FAIL)")
    ap.add_argument("zip", type=Path, help="путь к submission.zip")
    ap.add_argument("--no-smoke", dest="smoke", action="store_false", help="без прогона main.py")
    ap.add_argument("--no-weights", dest="weights", action="store_false", help="не открывать .pt (быстрее)")
    ap.add_argument("--python", default=sys.executable, help="интерпретатор с пакетами образа (по умолчанию текущий)")
    ap.add_argument("--input", help="своя папка входа вместо синтетики (маленький регион!)")
    ap.add_argument("--timeout", type=int, default=900, help="таймаут смоука и пробы, с (900)")
    ap.add_argument("--max-rss-mb", type=int, default=2000, help="снять прогон, если RSS дерева процессов больше (2000)")
    ap.add_argument("--allow-empty", action="store_true", help="пустой ответ смоука — WARN, а не FAIL")
    ap.add_argument("--cpus", default="0,1", help="taskset -c (0,1); пусто — без привязки")
    ap.add_argument("--threads", type=int, default=2, help="OMP/MKL/RAYON потоки (2); main.py:21 сам ставит OMP=8")
    ap.add_argument("--lock", help="requirements.lock: все сторонние пакеты должны быть в нём")
    ap.add_argument("--allow-net", action="store_true", help="не изолировать сеть")
    ap.add_argument("--workdir", default=None, help="где распаковывать (по умолчанию $TMPDIR)")
    ap.add_argument("--keep", action="store_true", help="не удалять рабочую папку смоука")
    args = ap.parse_args()
    global ALLOW_EMPTY
    ALLOW_EMPTY = args.allow_empty
    import shutil as _sh
    py = args.python if os.sep in args.python else (_sh.which(args.python) or args.python)
    args.python = os.path.abspath(py)  # cwd смоука — папка кода, относительный путь там не найдётся
    if args.workdir:
        Path(args.workdir).mkdir(parents=True, exist_ok=True)
    if not args.zip.is_file():
        print(f"нет файла {args.zip}", file=sys.stderr)
        return 2
    print(f"== check_submission: {args.zip} ({time.strftime('%Y-%m-%d %H:%M:%S %Z')})")
    t0 = time.monotonic()
    z = check_zip(args.zip)
    if z is None:
        return 1
    check_layout(z)
    check_forbidden(z)
    check_size(args.zip, z)
    check_metadata(z)
    cfg = check_config_static(z)
    if "inference/main.py" in z.namelist() and "expds/pipeline.py" in z.namelist():
        tmp = Path(tempfile.mkdtemp(prefix="b8static_", dir=args.workdir))
        try:
            for n in z.namelist():
                if n.endswith(".py") or n.endswith(".so") or n.endswith(".toml"):
                    z.extract(n, tmp)
            for n in z.namelist():                       # веса нужны пробе
                if args.weights and n.startswith("models/"):
                    z.extract(n, tmp)
            reach, third = check_imports_static(tmp)
            check_so(tmp)
            run_probe(tmp, args.python, reach, third, args)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    if args.smoke and not any(s == "FAIL" and c in ("zip", "layout") for s, c, _ in RESULTS):
        smoke(z, args)
    elif not args.smoke:
        report("INFO", "smoke", "пропущен (--no-smoke)")
    n_fail = sum(s == "FAIL" for s, _, _ in RESULTS)
    n_warn = sum(s == "WARN" for s, _, _ in RESULTS)
    verdict = "FAIL" if n_fail else "PASS"
    print(f"== ИТОГ: {verdict} — FAIL {n_fail}, WARN {n_warn}, проверок {len(RESULTS)}, {time.monotonic() - t0:.1f} с")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
