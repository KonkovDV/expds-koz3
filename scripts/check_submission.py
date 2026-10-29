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
    rr = cfg.get("rerank_arch") or {}
    if rr.get("enabled"):   # bot-7: без файлов rerank_arch молча вернёт базу — пути по умолчанию тоже проверяем
        refs = sorted(set(refs) | {("rerank_arch.model", rr.get("model", "models/rerank_arch_lgbm.txt")),
                                   ("rerank_arch.meta", rr.get("meta", "models/rerank_arch_lgbm.json"))})
    # bot-7 v2 (b1a9d28) раскрывал model_dir в poly.txt+finds.txt всегда; при cherry-pick bot-8 7878087 это
    # заменено точной проверкой по классам/картам (check_pixgbm_static) — model_dir как папка не FAIL.
    # bot-7 v3: каталоги HF-весов (sp_dfine.weights[i], gor_gate.backbone) — нужны config.json + model.safetensors
    hf_dirs = {v.rstrip("/") for k, v in refs if k.startswith("sp_dfine.weights") or k == "gor_gate.backbone"
               or k.startswith("dfine_aeor.weights")}   # bot-6: веса D-FINE Ae/Or
    hf_files = {f"{d}/{f}" for d in hf_dirs for f in ("config.json", "model.safetensors")}
    missing = [f"{k}={v}" for k, v in refs if v not in names
               and not (v.rstrip("/") in hf_dirs and {f"{v.rstrip('/')}/config.json", f"{v.rstrip('/')}/model.safetensors"} <= names)
               and not (k.startswith("verify.") and not verify.get("enabled"))
               and not (k.startswith("rerank_arch.") and not rr.get("enabled"))
               and not (k.startswith("dfine_aeor.") and not (cfg.get("dfine_aeor") or {}).get("enabled"))
               and not (k.startswith("geophys.pixgbm.") and not pixgbm.get("enabled"))   # bot-3 PIXGBM выкл.
               and k != "geophys.pixgbm.model_dir"]   # папка: файлы по классам проверяет check_pixgbm_static
    if missing:
        report("FAIL", "config", f"файлы из default.toml отсутствуют в архиве (модель молча выключится, "
                                 f"pipeline.py:98-101 на 24bd916): {missing}")
    else:
        report("PASS", "config", f"все {len(refs)} путей к весам из default.toml есть в архиве")
    pix_files = check_pixgbm_static(pixgbm, names)
    models_in_zip = {n for n in names if n.startswith("models/") and not n.endswith("/")}
    unused = sorted(models_in_zip - {v for _, v in refs} - pix_files - hf_files)
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


#: bot-3 PIXGBM (expds/geo_pixgbm.py): класс → файл модели в model_dir и методы, на которых он считается.
#: load_models молча пропускает отсутствующий <k>.txt (region_feats: «нет poly.txt/finds.txt — классика»),
#: поэтому при enabled=true каждый нужный файл обязан быть в архиве.
PIX_METHODS = ("maggr", "etgr", "terfl")
PIX_CLASS_MODEL = {"object_poly": ("poly", PIX_METHODS), "finds_points": ("finds", ("maggr",))}
PIX_BACKEND_FILES = {"lleaves": (".lleaves.so", ".lleaves.json"), "treelite": (".tl2cgen.so",)}


def pixgbm_needed(pix: dict) -> tuple[str, list[str], list[str]]:
    """(model_dir, нужные .txt, файлы бэкенда) для [geophys.pixgbm] — умолчания как в pipeline.py/geo_pixgbm.py."""
    md = str(pix.get("model_dir", "models/pixgbm/all")).strip("/")
    classes = pix.get("classes", ["finds_points", "object_poly"])
    maps = pix.get("maps", list(PIX_METHODS))
    maps = maps if isinstance(maps, list) else []
    need, extra = [], []
    for c in classes if isinstance(classes, list) else []:
        k, methods = PIX_CLASS_MODEL.get(c, (None, ()))
        if k and set(methods) & set(maps):
            need.append(f"{md}/{k}.txt")
            extra += [f"{md}/{k}{suf}" for suf in PIX_BACKEND_FILES.get(str(pix.get("backend", "lightgbm")), ())]
    return md, sorted(set(need)), sorted(set(extra))


def check_pixgbm_static(pix: dict, names: set) -> set:
    """При [geophys.pixgbm] enabled=true: файл модели на каждый класс (poly.txt / finds.txt) — иначе FAIL.
    Возвращает пути pixgbm, на которые ссылается конфиг (для предупреждения о лишних весах)."""
    if not pix.get("enabled"):
        return set()
    md, need, extra = pixgbm_needed(pix)
    bad = []
    classes, maps = pix.get("classes", ["finds_points", "object_poly"]), pix.get("maps", list(PIX_METHODS))
    if not isinstance(classes, list) or set(classes) - set(PIX_CLASS_MODEL):
        bad.append(f"geophys.pixgbm.classes={classes!r}: pix выдаёт только {sorted(PIX_CLASS_MODEL)}")
    if not isinstance(maps, list) or any(m not in PIX_METHODS for m in maps):
        bad.append(f"geophys.pixgbm.maps={maps!r}: нужен список из {PIX_METHODS}")
    if bad:
        report("FAIL", "config", "PIXGBM: " + "; ".join(bad))
    if not need:
        report("WARN", "config", f"PIXGBM включён, но по classes={classes!r} и maps={maps!r} ему нечего считать")
        return set()
    lost = [f for f in need if f not in names]
    if lost:
        report("FAIL", "config", f"PIXGBM включён, а моделей классов нет в архиве (pix молча уйдёт в классику, "
                                 f"geo_pixgbm.load_models): {lost}")
    else:
        report("PASS", "config", f"PIXGBM: модели классов есть в архиве: {need}")
    lost_x = [f for f in extra if f not in names]
    if lost_x:
        report("WARN", "config", f"PIXGBM backend={pix.get('backend')!r}: нет {lost_x} — фолбэк на lightgbm (медленнее)")
    return set(need) | set(extra)


# ---------------------------------------------------------------- импорты (статически)
def _module_of(rel: str) -> str | None:
    p = PurePosixPath(rel)
    if p.parts[0] == "expds":
        parts = list(p.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts)
    return None


# Необязательные сторонние пакеты: в коде инференса есть фолбэк без них (не FAIL, если не установлены / нет в lock).
OPTIONAL_THIRD = {
    "psutil": "geo_pixgbm.process_t0: ImportError → момент импорта модуля",
    "tl2cgen": "geo_pixgbm backend=treelite (opt-in): ошибка → lightgbm",
    "lleaves": "geo_pixgbm backend=lleaves (opt-in, .so через ctypes): ошибка → lightgbm",
}


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
    third_all = set().union(*(info[m][0] for m in reach))
    third_reach = third_all - set(OPTIONAL_THIRD)
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
    opt = sorted(third_all & set(OPTIONAL_THIRD))
    if opt:
        report("INFO", "imports", "необязательные (есть фолбэк, не проверяются): " + "; ".join(f"{k} — {OPTIONAL_THIRD[k]}" for k in opt))
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
            pm = sub.get("maps", ["maggr", "etgr", "terfl"])
            bad = [m for m in pm if m not in ("maggr", "etgr", "terfl")] if isinstance(pm, list) else [pm]
            if bad:
                out["cfg"].append(f"geophys.pixgbm.maps: неизвестные методы {bad} (нужен список из maggr, etgr, terfl)")
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
        rr = cfg.get("rerank_arch") or {}
        if rr.get("enabled"):
            try:
                import lightgbm as lgb
                rm = rr.get("model", "models/rerank_arch_lgbm.txt")
                b = lgb.Booster(model_file=str(root / rm))
                json.loads((root / rr.get("meta", "models/rerank_arch_lgbm.json")).read_text(encoding="utf-8"))
                out["weights"].append({"markup": "rerank_arch", "file": rm, "ok": True, "classes": [f"{b.num_trees()} деревьев"]})
            except Exception as e:
                out["weights"].append({"markup": "rerank_arch", "file": rr.get("model"), "ok": False, "err": repr(e)[:300]})
        pix = (cfg.get("geophys") or {}).get("pixgbm") or {}
        if pix.get("enabled"):
            for f in json.loads(sys.argv[5]):
                try:
                    import lightgbm as lgb
                    b = lgb.Booster(model_file=str(root / f))
                    out["weights"].append({"markup": "pixgbm", "file": f, "ok": True, "classes": [f"{b.num_trees()} деревьев"]})
                except Exception as e:
                    out["weights"].append({"markup": "pixgbm", "file": f, "ok": False, "err": repr(e)[:300]})
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


#: Пакеты, отсутствие которых в образе допустимо (WARN), если КАЖДЫЙ их импорт в коде инференса защищён
#: try/except с фолбэком: psutil — geo_pixgbm.process_t0 (иначе момент импорта), tl2cgen/lleaves — opt-in бэкенды
#: PIXGBM (иначе lightgbm). Защищённость проверяется статически (guarded_import_sites), иначе FAIL.
OPTIONAL_LAZY = {"psutil", "tl2cgen", "lleaves"}
_FALLBACK_EXC = {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}


def _handler_falls_back(h: ast.ExceptHandler) -> bool:
    """except ловит ImportError (или шире) и не заканчивается raise — значит, дальше идёт фолбэк."""
    if h.type is None:
        caught = {"BaseException"}
    else:
        caught = {e.attr if isinstance(e, ast.Attribute) else getattr(e, "id", "")
                  for e in (h.type.elts if isinstance(h.type, ast.Tuple) else [h.type])}
    return bool(caught & _FALLBACK_EXC) and not isinstance(h.body[-1], ast.Raise)


def _scan_guards(tree: ast.AST) -> tuple[list, list]:
    """(импорты, вызовы): (имя, строка, защищён try/except-фолбэком в своей области, функция, класс).
    Тело функции/лямбды — новая область (защита снаружи def не действует при вызове)."""
    imps, calls = [], []

    def walk(node, g, fn, cls):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for d in node.decorator_list + node.args.defaults + node.args.kw_defaults:
                if d is not None:
                    walk(d, g, fn, cls)
            for st in node.body:
                walk(st, False, node.name, cls)
            return
        if isinstance(node, ast.Lambda):
            walk(node.body, False, "<lambda>", cls)
            return
        if isinstance(node, ast.ClassDef):
            for st in node.body:
                walk(st, g, fn, node.name)
            return
        if isinstance(node, (ast.Try, getattr(ast, "TryStar", ast.Try))):
            ok = any(_handler_falls_back(h) for h in node.handlers)
            for st in node.body:
                walk(st, g or ok, fn, cls)
            for part in (node.handlers, node.orelse, node.finalbody):
                for st in part:
                    walk(st, g, fn, cls)
            return
        if isinstance(node, ast.Import):
            imps.extend([(a.name.split(".")[0], node.lineno, g, fn, cls) for a in node.names])
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            imps.append((node.module.split(".")[0], node.lineno, g, fn, cls))
        elif isinstance(node, ast.Call):
            f = node.func
            nm = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            if nm:
                calls.append((nm, node.lineno, g, fn, cls))
        for ch in ast.iter_child_nodes(node):
            walk(ch, g, fn, cls)

    walk(tree, False, None, None)
    return imps, calls


def guarded_import_sites(root: Path, reach: set, pkgs: set, depth: int = 3) -> dict:
    """{пакет: [(место, защищён)]} по модулям из reach. Импорт защищён, если лежит в try с обработчиком
    ImportError/ModuleNotFoundError/Exception/BaseException/голым except, который не заканчивается raise,
    либо если он внутри функции (метода; для __init__ — класса), КАЖДЫЙ вызов которой в reach защищён так же
    (рекурсивно, до depth уровней). Нет ни одного вызова — не защищён (консервативно)."""
    info, _ = scan_imports(root)
    imps, calls = [], []
    for m in sorted(reach):
        if m in info:
            i, c = _scan_guards(ast.parse(info[m][3].read_text(encoding="utf-8")))
            rel = info[m][3].relative_to(root).as_posix()
            imps += [(rel,) + x for x in i]
            calls += [(rel,) + x for x in c]

    def ok(g, fn, cls, d, seen):
        if g:
            return True
        if fn is None or d <= 0 or (fn, cls) in seen:
            return False
        targets = {cls} if fn in ("__init__", "__new__") and cls else {fn}
        cs = [c for c in calls if c[1] in targets]
        return bool(cs) and all(ok(c[3], c[4], c[5], d - 1, seen | {(fn, cls)}) for c in cs)

    out = {p: [] for p in pkgs}
    for rel, top, line, g, fn, cls in imps:
        if top in out:
            out[top].append((f"{rel}:{line}", ok(g, fn, cls, depth, frozenset())))
    return out


def unguarded_imports(root: Path, reach: set) -> set:
    """Верхние имена пакетов, у которых хотя бы один импорт в reach НЕ защищён фолбэком (их отсутствие — FAIL)."""
    info, _ = scan_imports(root)
    pkgs = set().union(*(info[m][0] for m in reach if m in info))
    return {k for k, v in guarded_import_sites(root, reach, pkgs).items() if not v or not all(g for _, g in v)}


_module_level_imports = unguarded_imports   # совместимость с тестом бота 4 (work/review/cs_test/test_cs.py, T8)


def run_probe(root: Path, py: str, reach: set, third: set, args) -> None:
    home = Path(tempfile.mkdtemp(prefix="b8home_"))
    try:
        pix = (tomllib.loads((root / "configs" / "default.toml").read_text(encoding="utf-8")).get("geophys") or {}).get("pixgbm") or {}
        pix_txt = pixgbm_needed(pix)[1] if isinstance(pix, dict) and pix.get("enabled") else []
    except Exception:
        pix_txt = []
    cmd = sandbox_cmd([py, "-c", PROBE, str(root), json.dumps(sorted(reach)), json.dumps(sorted(third)),
                       "1" if args.weights else "0", json.dumps(pix_txt)], args.cpus, not args.allow_net)
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
    lazy = set(out["third_fail"]) & OPTIONAL_LAZY
    sites = guarded_import_sites(root, reach, lazy) if lazy else {}
    opt = {k: sites[k] for k in lazy if sites[k] and all(g for _, g in sites[k])}
    for k in lazy - set(opt):
        out["third_fail"][k] = (f"{out['third_fail'][k]}; импорт без try/except-фолбэка: "
                                f"{[w for w, g in sites[k] if not g] or 'места импорта не найдены'}")
    out["third_fail"] = {k: v for k, v in out["third_fail"].items() if k not in opt}
    if opt:
        report("WARN", "imports", f"не установлены в {py} необязательные пакеты; каждый импорт в try/except "
                                  f"с фолбэком: {', '.join(f'{k} ({[w for w, _ in v]})' for k, v in sorted(opt.items()))}")
    vend = root / "inference" / "lib" / "vendor"   # bot-7 v3: пакеты из вендора архива (импорт проверяет check_dfine)
    vendored = {k for k in out["third_fail"] if (vend / k).is_dir()}
    if vendored:
        report("INFO", "imports", f"нет в {py}, но есть в inference/lib/vendor (проверка — dfine): {sorted(vendored)}")
        out["third_fail"] = {k: v for k, v in out["third_fail"].items() if k not in vendored}
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
#: bot-7 (v3): проба D-FINE/шлюза в чистом процессе — вендор архива, закреплённый transformers, классы D-FINE.
DFINE_PROBE = r"""
import json, sys
root = sys.argv[1]
sys.path.insert(0, root)
out = {}
try:
    from expds.sp_dfine import import_dfine
    C, M, info = import_dfine(root)
    out = {"ok": True, **{k: info.get(k) for k in ("source", "version", "file", "errors", "purged_preloaded")}}
    if sys.argv[2] == "1":
        from transformers import Dinov2Config, Dinov2Model  # noqa: F401
        out["dinov2"] = True
except Exception as exc:
    out = {"ok": False, "err": f"{type(exc).__name__}: {str(exc)[:1500]}"}
print("DFINE_JSON=" + json.dumps(out, ensure_ascii=False))
"""


#: bot-7 (v3): опорные участки, на которых фолдовая модель f0 bot-9 даёт детекции (bot-7/work6/dfprobe/
#: find_windows.py → windows_103_f0.json: окна 640 px на 0,5 м без отражения, запросы со score ≥ 0,3).
#: n_det = 0 на таком участке значит: импорт прошёл, а веса не работают (не те/битые/не загрузились) — FAIL.
DFINE_KNOWN = {
    "103_Гнёздово_Ольшанская": {
        "raster": "SpOr",
        "windows": [[1440, 0], [2400, 0], [2880, 0], [1920, 480], [2400, 480], [1440, 960], [1920, 960],
                    [1440, 1440], [1920, 1440], [2880, 1440], [3360, 1440], [2880, 1920]],
        "ref_f0": {"n_ge": 14, "max": 0.704, "smoke_n_det": 154},
    },
}
DFINE_MIN_SCORE = 0.3
DFINE_REGION_DEFAULT = "/workspace/data/train-expds-3/103_Гнёздово_Ольшанская"

DFINE_DET_PROBE = r"""
import json, sys
from pathlib import Path
root, tif, wins, thr = Path(sys.argv[1]), Path(sys.argv[2]), json.loads(sys.argv[3]), float(sys.argv[4])
sys.path.insert(0, str(root))
out = {}
try:
    import torch
    torch.set_num_threads(2)
    import tomllib
    from expds import sp_dfine as D
    cfg = tomllib.loads((root / "configs" / "default.toml").read_text(encoding="utf-8"))
    st = dict(D.settings(cfg), batch=1, flip_tta=False)
    models = D.load_models(st, root, torch.device("cpu"))
    sc = D.Scene(tif, D.utm_for(tif, None), st["res_m"])
    per = []
    with torch.no_grad():
        for mi, m in enumerate(models):
            n = 0; mx = 0.0
            for x, y in wins:
                im = sc.read(int(x), int(y), int(st["tile"]))
                o = m(pixel_values=torch.from_numpy(im.transpose(2, 0, 1)[None].astype("float32") / 255))
                p = o.logits.float().sigmoid().flatten()
                n += int((p >= thr).sum()); mx = max(mx, float(p.max()))
            per.append({"weights": st["weights"][mi], "n_ge": n, "max": round(mx, 4)})
    out = {"ok": True, "models": per}
except Exception as exc:
    out = {"ok": False, "err": f"{type(exc).__name__}: {str(exc)[:1500]}"}
print("DFINE_DET_JSON=" + json.dumps(out, ensure_ascii=False))
"""


def dfine_region_raster(region: Path, spec: dict) -> Path | None:
    tifs = sorted(p for p in Path(region).rglob("*") if p.suffix.lower() in (".tif", ".tiff") and spec["raster"] in p.name)
    return tifs[0] if tifs else None


def dfine_verdict(models: list[dict], region: str) -> list[tuple[str, str]]:
    """Вердикт пробы весов: модель без единого запроса score ≥ DFINE_MIN_SCORE на окнах опорного участка — FAIL."""
    ref = DFINE_KNOWN[region]["ref_f0"]
    if not models:
        return [("FAIL", f"{region}: ни одной модели D-FINE не загружено")]
    res = []
    for m in models:
        msg = (f"{region}: {m['weights']} — запросов score ≥ {DFINE_MIN_SCORE} на {len(DFINE_KNOWN[region]['windows'])} "
               f"опорных окнах {m['n_ge']} (f0: {ref['n_ge']}), max {m['max']} (f0: {ref['max']})")
        res.append(("FAIL", msg + " — веса не дают детекций там, где фолдовая модель их даёт") if m["n_ge"] == 0
                   else ("PASS", msg))
    return res


def check_dfine_weights(z: zipfile.ZipFile, cfg: dict, args) -> None:
    """[sp_dfine] enabled: веса из архива на опорном участке с известными детекциями f0 (CPU, batch 1, без
    отражения, ≤2 потока). n = 0 → FAIL. Без опорного участка — FAIL (проба обязательна), с --no-weights — WARN."""
    if not bool((cfg.get("sp_dfine") or {}).get("enabled")):
        return
    if not args.weights:
        report("WARN", "dfine", "--no-weights: проба весов D-FINE на опорном участке пропущена")
        return
    region = Path(args.dfine_region or DFINE_REGION_DEFAULT)
    spec = DFINE_KNOWN.get(region.name)
    tif = dfine_region_raster(region, spec) if spec and region.is_dir() else None
    if spec is None or tif is None:
        report("FAIL", "dfine", f"нет опорного участка для пробы весов D-FINE ({region}); нужен один из "
                                f"{list(DFINE_KNOWN)} (--dfine-region)")
        return
    tmp = Path(tempfile.mkdtemp(prefix="b7dfw_", dir=args.workdir))
    home = Path(tempfile.mkdtemp(prefix="b7dfwhome_", dir=args.workdir))
    try:
        wdirs = [str(w).rstrip("/") + "/" for w in ((cfg.get("sp_dfine") or {}).get("weights") or [])]
        for n in z.namelist():
            if (n.startswith("inference/lib/vendor/") or (n.startswith("expds/") and n.endswith(".py"))
                    or n == "configs/default.toml" or any(n.startswith(w) for w in wdirs)):
                z.extract(n, tmp)
        cmd = sandbox_cmd([args.python, "-s", "-c", DFINE_DET_PROBE, str(tmp), str(tif),
                           json.dumps(spec["windows"]), str(DFINE_MIN_SCORE)], args.cpus, not args.allow_net)
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout,
                               env=child_env(home, args.threads), cwd=str(tmp))
        except subprocess.TimeoutExpired:
            report("FAIL", "dfine", f"проба весов D-FINE не уложилась в {args.timeout} с")
            return
        line = next((x for x in r.stdout.splitlines() if x.startswith("DFINE_DET_JSON=")), None)
        out = json.loads(line.split("=", 1)[1]) if line else {"ok": False, "err": (r.stderr or r.stdout)[-1500:]}
        if not out.get("ok"):
            report("FAIL", "dfine", f"веса D-FINE из архива не загрузились/не отработали: {out.get('err')}")
            return
        for status, msg in dfine_verdict(out["models"], region.name):
            report(status, "dfine", msg)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(home, ignore_errors=True)


def dfine_ndet_verdict(log: str) -> list[tuple[str, str]]:
    """Строки «sp_dfine n_det=N region=R» смоука: N = 0 на опорном участке (DFINE_KNOWN) — FAIL; итог 0 — WARN."""
    import re
    res = []
    for m in re.finditer(r"sp_dfine n_det=(\d+) region=(\S+?)(?:\s|$)", log):
        n, reg = int(m.group(1)), m.group(2)
        if reg in DFINE_KNOWN and n == 0:
            res.append(("FAIL", f"sp_dfine n_det=0 на опорном участке {reg} (f0: {DFINE_KNOWN[reg]['ref_f0']['smoke_n_det']})"
                                " — веса не дают детекций"))
    tot = [int(x) for x in re.findall(r"sp_dfine n_det=(\d+) total", log)]
    if tot and tot[-1] == 0 and not res:
        res.append(("WARN", "sp_dfine n_det=0 total (вход без опорного участка — проверить пробой весов)"))
    return res


def check_dfine(z: zipfile.ZipFile, cfg: dict | None, args) -> None:
    """[sp_dfine]/[gor_gate] enabled: вендор transformers закреплённой версии в архиве и импорт классов
    в чистом процессе целевого python (как на площадке). Иначе FAIL: на прогоне модуль молча дал бы выдачу v2."""
    cfg = cfg or {}
    dfine = bool((cfg.get("sp_dfine") or {}).get("enabled"))
    gate = bool((cfg.get("gor_gate") or {}).get("enabled"))
    aeor = bool((cfg.get("dfine_aeor") or {}).get("enabled"))   # bot-6: тот же вендор и import_dfine
    if not (dfine or gate or aeor):
        return
    what = "+".join(n for n, on in (("sp_dfine", dfine), ("gor_gate", gate), ("dfine_aeor", aeor)) if on)
    names = z.namelist()
    vend = "inference/lib/vendor/"
    meta = [n for n in names if n.startswith(vend + "transformers-") and n.endswith(".dist-info/METADATA")]
    pin = "5.21.0"
    if not any(n.startswith(vend + "transformers/") for n in names):
        report("FAIL", "dfine", f"{what} включён, а вендора transformers ({vend}transformers/) в архиве нет — "
                                f"классы D-FINE зависели бы от образа")
    elif not any(f"transformers-{pin}.dist-info" in n for n in meta):
        report("FAIL", "dfine", f"{what}: версия вендора transformers не {pin}: {meta}")
    mods = [m for m in (("expds/sp_dfine.py", True), ("expds/gor_gate.py", gate), ("expds/dfine_aeor.py", aeor))
            if m[1] and m[0] not in names]
    if mods:
        report("FAIL", "dfine", f"{what} включён, а модуля нет в архиве: {[m[0] for m in mods]}")
        return
    tmp = Path(tempfile.mkdtemp(prefix="b7dfine_", dir=args.workdir))
    home = Path(tempfile.mkdtemp(prefix="b7dfhome_", dir=args.workdir))
    try:
        for n in names:
            if n.startswith(vend) or (n.startswith("expds/") and n.endswith(".py")):
                z.extract(n, tmp)
        cmd = sandbox_cmd([args.python, "-s", "-c", DFINE_PROBE, str(tmp), "1" if gate else "0"], args.cpus,
                          not args.allow_net)
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout, env=child_env(home, args.threads),
                               cwd=str(tmp))
        except subprocess.TimeoutExpired:
            report("FAIL", "dfine", f"проба импорта D-FINE не уложилась в {args.timeout} с")
            return
        line = next((x for x in r.stdout.splitlines() if x.startswith("DFINE_JSON=")), None)
        out = json.loads(line.split("=", 1)[1]) if line else {"ok": False, "err": (r.stderr or r.stdout)[-1500:]}
        if not out.get("ok"):
            report("FAIL", "dfine", f"{what} включён, а классы D-FINE не импортируются в {args.python}: {out.get('err')}")
        elif out.get("source") != "vendor":
            report("WARN", "dfine", f"transformers {out.get('version')} взят из окружения, не из вендора "
                                    f"(вендор: {out.get('errors')})")
        else:
            report("PASS", "dfine", f"{what}: DFineConfig/DFineForObjectDetection из вендора transformers "
                                    f"{out.get('version')}{' + Dinov2' if gate else ''} в чистом процессе")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(home, ignore_errors=True)


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


def dfine_lines(z: zipfile.ZipFile, log: str) -> None:
    """bot-7 (v3): маркеры недоступности D-FINE/шлюза в логе — FAIL; число детекций D-FINE — INFO."""
    bad = [ln for ln in log.splitlines() if "SP_DFINE_UNAVAILABLE" in ln or "GOR_GATE_UNAVAILABLE" in ln
           or "DFINE_AEOR_UNAVAILABLE" in ln]   # bot-6: D-FINE Ae/Or
    if bad:
        report("FAIL", "dfine", f"{len(bad)} строк недоступности D-FINE/шлюза в логе (выдача = v2): "
                                + " | ".join(b[-300:] for b in bad[:3]))
    nd = [ln.split("sp_dfine ", 1)[1][:160] for ln in log.splitlines() if "sp_dfine n_det=" in ln]
    if nd:
        report("INFO", "dfine", " ; ".join(nd[-6:]))
        for status, msg in dfine_ndet_verdict(log):
            report(status, "dfine", msg)
    else:
        try:
            on = bool((tomllib.loads(z.read("configs/default.toml").decode()).get("sp_dfine") or {}).get("enabled"))
        except Exception:
            on = False
        if on:
            report("WARN", "dfine", "[sp_dfine] включён, а строк «sp_dfine n_det=» в логе нет (во входе нет SpOr?)")


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
    dfine_lines(z, log)
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
    ap.add_argument("--dfine-region", help=f"опорный участок пробы весов D-FINE (по умолчанию {DFINE_REGION_DEFAULT})")
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
    check_dfine(z, cfg, args)
    check_dfine_weights(z, cfg or {}, args)
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
