#!/usr/bin/env python3
"""Проверка архива решения перед загрузкой на платформу.

    python scripts/check_submission.py dist/submission.zip [--smoke ПАПКА_ВЫБОРКИ] [--timeout 1800]

Проверяется:
  1. размер ≤ 2 ГБ, пути в архиве относительные;
  2. metadata.json: образ Docker и точка входа, файл точки входа есть в архиве;
  3. configs/default.toml читается; все веса, на которые он ссылается, есть в архиве;
  4. каждый .py компилируется; каждый относительный импорт expds разрешается внутри архива;
  5. в expds/ нет модулей сверх тех, что достижимы по импортам из pipeline (лишний код не попадает);
  6. --smoke: распаковка во временную папку, запуск точки входа на папке выборки, проверка GeoJSON
     валидатором expds.writer.validate (лимиты ТР: 100 МБ, 20 000 объектов, 3000 вершин, Polygon, EPSG:3857).
Код возврата 1 при любой ошибке.
"""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import tempfile
import time
import tomllib
import zipfile
from pathlib import Path

LIMIT_BYTES = 2 * 1024 ** 3


def fail(errors: list[str], msg: str) -> None:
    errors.append(msg)
    print(f"FAIL  {msg}")


def closure(sources: dict[str, str]) -> tuple[set[str], set[str]]:
    """(достижимые модули expds, неразрешённые относительные импорты) по исходникам архива."""
    mods = {Path(n).stem: s for n, s in sources.items() if n.startswith("expds/") and n.count("/") == 1}
    edges: dict[str, set[str]] = {}
    for name, src in mods.items():
        deps: set[str] = set()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.ImportFrom):
                if node.level == 1:
                    deps |= {node.module.split(".")[0]} if node.module else {a.name for a in node.names}
                elif node.module and node.module.split(".")[0] == "expds":
                    parts = node.module.split(".")
                    deps |= {parts[1]} if len(parts) > 1 else {a.name for a in node.names}
        edges[name] = deps
    seen: set[str] = set()
    missing: set[str] = set()
    stack = ["__init__", "pipeline", "watchdog", "writer"]
    while stack:
        m = stack.pop()
        if m in seen:
            continue
        if m not in mods:
            missing.add(m)
            continue
        seen.add(m)
        stack += sorted(edges[m])
    return seen, missing


def model_paths(cfg: dict) -> list[str]:
    out: list[str] = []
    for spec in (cfg.get("models") or {}).values():
        for item in spec if isinstance(spec, list) else [spec]:
            path = item.get("path") if isinstance(item, dict) else item
            out += path if isinstance(path, list) else [path]
    for key, sub in (("candidates", "lgbm_model"),):
        if (cfg.get(key) or {}).get(sub):
            out.append(cfg[key][sub])
    rr = cfg.get("rerank_arch") or {}
    if rr.get("enabled"):
        out += [rr["model"], rr["meta"]]
    ver = cfg.get("verify") or {}
    if ver.get("enabled"):
        out.append(ver["weights"])
    cnx = cfg.get("li_cnx") or {}
    if cnx.get("enabled") is True:
        out += list(cnx.get("weights") or []) + list(cnx.get("weights_b") or [])
    return out


def check_archive(path: Path, errors: list[str]) -> dict:
    size = path.stat().st_size
    print(f"архив {path} — {size} байт")
    if size > LIMIT_BYTES:
        fail(errors, f"размер {size} > 2 ГБ")
    z = zipfile.ZipFile(path)
    names = set(z.namelist())
    bad = [n for n in names if n.startswith("/") or ".." in Path(n).parts]
    if bad:
        fail(errors, f"абсолютные пути или '..': {bad[:5]}")
    meta = json.loads(z.read("metadata.json")) if "metadata.json" in names else {}
    if not meta.get("image") or not meta.get("entry_point"):
        fail(errors, "metadata.json: нужны image и entry_point")
    else:
        entry = meta["entry_point"].split()[-1]
        if entry not in names:
            fail(errors, f"точки входа {entry} нет в архиве")
        print(f"OK    образ {meta['image']}, запуск «{meta['entry_point']}»")
    if "configs/default.toml" not in names:
        fail(errors, "нет configs/default.toml")
        return {}
    cfg = tomllib.loads(z.read("configs/default.toml").decode("utf-8"))
    lost = [p for p in model_paths(cfg) if p not in names]
    pix = (cfg.get("geophys") or {}).get("pixgbm") or {}
    if pix.get("enabled"):
        md = pix.get("model_dir", "models/pix_clean").rstrip("/")
        need = [f"{md}/org/{f}" for f in ("org_etgr_map.txt", "org_etgr_prof.txt", "org_etgr_meta.json")]
        need += [f"{md}/ens6/{m}/finds.txt" for m in ("n40r8", "n40r6", "n40r10", "aug", "gw1d", "gw3d")]
        lost += [n for n in need if n not in names]
        if (pix.get("e1g1") or {}).get("backend") == "lleaves" and not any(n.startswith("vendor/lleaves/") for n in names):
            fail(errors, "backend lleaves, а vendor/lleaves нет в архиве")
    if lost:
        fail(errors, f"нет весов из конфига: {lost}")
    else:
        print(f"OK    веса из конфига на месте ({len(model_paths(cfg))} файлов + геофизика)")
    sources = {n: z.read(n).decode("utf-8") for n in names if n.endswith(".py") and not n.startswith("vendor/")}
    for n, s in sorted(sources.items()):
        try:
            compile(s, n, "exec")
        except SyntaxError as exc:
            fail(errors, f"{n}: не компилируется ({exc})")
    reach, missing = closure(sources)
    if missing:
        fail(errors, f"импорты без модуля в архиве: {sorted(missing)}")
    extra = sorted(n for n in sources if n.startswith("expds/") and n.count("/") == 1 and Path(n).stem not in reach)
    if extra:
        fail(errors, f"лишние модули expds: {extra}")
    if not missing and not extra:
        print(f"OK    модулей expds {len(reach)}, все импорты разрешаются, лишних нет")
    return cfg


def smoke(path: Path, data: Path, timeout: int, errors: list[str]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "solution"
        zipfile.ZipFile(path).extractall(root)
        out = Path(tmp) / "out" / "result.geojson"
        out.parent.mkdir()
        t0 = time.monotonic()
        r = subprocess.run([sys.executable, "inference/main.py", str(data), str(out)], cwd=root,
                           capture_output=True, text=True, timeout=timeout)
        sec = time.monotonic() - t0
        tail = "\n".join(r.stdout.splitlines()[-3:])
        print(f"смоук: код {r.returncode}, {sec:.0f} с\n{tail}")
        if r.returncode != 0 or not out.is_file():
            fail(errors, "смоук: прогон не записал результат")
            return
        sys.path.insert(0, str(root))
        from expds.writer import validate

        rep = validate(out)
        print(f"смоук: {rep['summary']}")
        for e in rep["errors"]:
            fail(errors, f"GeoJSON: {e}")
        if rep["ok"]:
            print("OK    GeoJSON проходит валидатор ТР")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("archive", type=Path)
    ap.add_argument("--smoke", type=Path, default=None, help="папка выборки для пробного запуска")
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()
    errors: list[str] = []
    check_archive(args.archive, errors)
    if args.smoke is not None and not errors:
        smoke(args.archive, args.smoke, args.timeout, errors)
    print("ИТОГ:", "FAIL" if errors else "PASS")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
