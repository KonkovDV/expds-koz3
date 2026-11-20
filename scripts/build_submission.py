#!/usr/bin/env python3
"""Сборка архива решения для платформы.

    python scripts/build_submission.py [--config configs/default.toml] [--out dist/submission.zip]

В архив попадают только файлы, которые исполняет финальная конфигурация:
  * metadata.json и точка входа inference/main.py;
  * модули expds/, достижимые по импортам из expds.pipeline (включая ленивые импорты внутри функций);
  * centerhead/ и inference/cnx/ — если включён лидар ConvNeXt ([li_cnx]);
  * inference/lib/expds_core*.so — Rust-ядро (без него работают NumPy-версии тех же функций);
  * конфиг (кладётся в архив как configs/default.toml) и веса, на которые он ссылается;
  * vendor/ (lleaves, llvmlite) — если пиксельная геофизика идёт бэкендом lleaves.

Архив воспроизводим побайтно: порядок файлов, даты и права фиксированы, поэтому sha256 одной и той же
сборки совпадает на любой машине.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import sys
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZIP_DATE = (2026, 11, 17, 0, 0, 0)
LIMIT_BYTES = 2 * 1024 ** 3


def expds_closure() -> list[Path]:
    """Модули expds/, достижимые по импортам из pipeline, watchdog и writer."""
    mods = {p.stem: p for p in (ROOT / "expds").glob("*.py")}
    edges: dict[str, set[str]] = {m: set() for m in mods}
    for name, path in mods.items():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                if node.level == 1:
                    edges[name] |= {node.module.split(".")[0]} if node.module else {a.name for a in node.names}
                elif node.module and node.module.split(".")[0] == "expds":
                    parts = node.module.split(".")
                    edges[name] |= {parts[1]} if len(parts) > 1 else {a.name for a in node.names}
    seen: set[str] = set()
    stack = ["__init__", "pipeline", "watchdog", "writer"]
    while stack:
        m = stack.pop()
        if m in seen or m not in mods:
            continue
        seen.add(m)
        stack += sorted(edges[m])
    return [mods[m] for m in sorted(seen)]


def tree(path: Path) -> list[Path]:
    """Все файлы каталога без кэшей Python."""
    return sorted(p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts)


def model_paths(cfg: dict) -> list[str]:
    """Пути весов из конфига (относительно корня)."""
    out: list[str] = []
    for spec in (cfg.get("models") or {}).values():
        for item in spec if isinstance(spec, list) else [spec]:
            path = item.get("path") if isinstance(item, dict) else item
            out += path if isinstance(path, list) else [path]
    cand = (cfg.get("candidates") or {}).get("lgbm_model")
    if cand:
        out.append(cand)
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


def collect(cfg: dict, config_path: Path) -> list[tuple[str, Path]]:
    """(путь в архиве, файл на диске) для всех членов архива."""
    files: list[tuple[str, Path]] = []

    def add(p: Path) -> None:
        files.append((p.relative_to(ROOT).as_posix(), p))

    add(ROOT / "metadata.json")
    add(ROOT / "inference" / "main.py")
    for so in sorted((ROOT / "inference" / "lib").glob("expds_core*.so")):
        add(so)
    for p in expds_closure():
        add(p)
    cnx = cfg.get("li_cnx") or {}
    if cnx.get("enabled") is True:
        for p in sorted((ROOT / "centerhead").glob("*.py")):
            add(p)
        for p in tree(ROOT / "inference" / "cnx"):
            add(p)
    pix = (cfg.get("geophys") or {}).get("pixgbm") or {}
    if pix.get("enabled"):
        for p in tree(ROOT / pix.get("model_dir", "models/pix_clean")):
            add(p)
        if (pix.get("e1g1") or {}).get("backend") == "lleaves":
            for p in tree(ROOT / "vendor"):
                add(p)
    for rel in model_paths(cfg):
        add(ROOT / rel)
    files.append(("configs/default.toml", config_path))
    names = [n for n, _ in files]
    dup = {n for n in names if names.count(n) > 1}
    files = [f for i, f in enumerate(files) if f[0] not in dup or names.index(f[0]) == i]
    missing = [n for n, p in files if not p.is_file()]
    if missing:
        sys.exit("нет файлов: " + ", ".join(missing))
    return sorted(files)


def write_zip(files: list[tuple[str, Path]], out: Path) -> str:
    """Детерминированный zip: фиксированные даты, права 0644, порядок по имени. Возвращает sha256."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w") as z:
        for name, path in files:
            info = zipfile.ZipInfo(name, date_time=ZIP_DATE)
            info.external_attr = 0o644 << 16
            compress = zipfile.ZIP_STORED if path.suffix in (".pt", ".so", ".o", ".whl") else zipfile.ZIP_DEFLATED
            info.compress_type = compress
            with path.open("rb") as src, z.open(info, "w", force_zip64=True) as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
    h = hashlib.sha256()
    with out.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "default.toml")
    ap.add_argument("--out", type=Path, default=ROOT / "dist" / "submission.zip")
    args = ap.parse_args()
    cfg = tomllib.loads(args.config.read_text(encoding="utf-8"))
    files = collect(cfg, args.config.resolve())
    sha = write_zip(files, args.out)
    size = args.out.stat().st_size
    if size > LIMIT_BYTES:
        sys.exit(f"архив {size} байт больше лимита платформы 2 ГБ")
    print(f"{args.out}  {len(files)} файлов  {size} байт  sha256 {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
