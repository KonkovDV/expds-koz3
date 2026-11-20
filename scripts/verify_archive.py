#!/usr/bin/env python3
"""Сверка архива решения или рабочей копии с манифестом сданного архива.

    python scripts/verify_archive.py dist/submission.zip      # архив: состав, sha256 членов, метаданные zip
    python scripts/verify_archive.py --local                  # рабочая копия: всё ли на месте перед сборкой

`docs/final_archive.sha256` — sha256 каждого из 178 членов архива финальной версии (попытка 19.11.2026
22:29 МСК). Если sha256 всего архива не совпал, скрипт показывает, какие члены отличаются и откуда они
берутся: из git, из `models/` (веса, `models/MANIFEST.sha256`), из `vendor/` или колеса RVT (публичные
пакеты, `scripts/fetch_thirdparty.py`).

Для архива проверяется и то, от чего зависит побайтная воспроизводимость (`scripts/build_submission.py`):
члены по имени, дата 2026-11-17 00:00, права 0644, без сжатия только `.pt`, `.so`, `.o`, `.whl`.
Код возврата 1 при любом расхождении.
"""
from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "final_archive.sha256"
#: sha256 сданного архива (тот же, что в tests/test_final_config.py и docs/README.md).
FINAL_SHA256 = "171bed3024aa6b55252c0f6b70116952e815a806bb29271d0f26e9737f8b7252"
ZIP_DATE = (2026, 11, 17, 0, 0, 0)
STORED_SUFFIXES = (".pt", ".so", ".o", ".whl")


def load_manifest(path: Path = MANIFEST) -> dict[str, str]:
    """{путь в архиве: sha256} из файла формата sha256sum."""
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            sha, name = line.split(maxsplit=1)
            out[name.strip()] = sha
    return out


def origin(name: str) -> str:
    """Откуда берётся член архива при сборке."""
    if name.startswith("models/"):
        return "веса (models/MANIFEST.sha256)"
    if name.startswith("vendor/"):
        return "vendor/ (lleaves, llvmlite с PyPI)"
    if name.startswith("inference/cnx/wheels/"):
        return "колесо RVT (PyPI)"
    return "git"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def check_zip(path: Path, manifest: dict[str, str]) -> list[str]:
    """Расхождения архива с манифестом и с правилами детерминированной сборки."""
    problems: list[str] = []
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        names = [i.filename for i in infos]
        if names != sorted(names):
            problems.append("члены архива не упорядочены по имени")
        if len(names) != len(set(names)):
            problems.append("в архиве есть повторяющиеся имена")
        for name in sorted(set(manifest) - set(names)):
            problems.append(f"нет члена {name} [{origin(name)}]")
        for info in infos:
            name = info.filename
            if name not in manifest:
                problems.append(f"лишний член {name}")
                continue
            if _sha256(z.read(name)) != manifest[name]:
                problems.append(f"другое содержимое: {name} [{origin(name)}]")
            if info.date_time != ZIP_DATE:
                problems.append(f"дата {info.date_time} вместо {ZIP_DATE}: {name}")
            if info.external_attr >> 16 != 0o644:
                problems.append(f"права {oct(info.external_attr >> 16)} вместо 0o644: {name}")
            want = zipfile.ZIP_STORED if Path(name).suffix in STORED_SUFFIXES else zipfile.ZIP_DEFLATED
            if info.compress_type != want:
                problems.append(f"сжатие {info.compress_type} вместо {want}: {name}")
    return problems


def check_local(root: Path, manifest: dict[str, str]) -> tuple[list[str], list[str]]:
    """(отсутствуют, отличаются) — файлы рабочей копии, из которых собирается архив."""
    missing, different = [], []
    for name, sha in sorted(manifest.items()):
        p = root / name
        if not p.is_file():
            missing.append(name)
        elif _file_sha256(p) != sha:
            different.append(name)
    return missing, different


def _group(names: list[str]) -> str:
    counts: dict[str, int] = {}
    for n in names:
        counts[origin(n)] = counts.get(origin(n), 0) + 1
    return "; ".join(f"{k}: {v}" for k, v in sorted(counts.items()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("archive", type=Path, nargs="?", help="архив решения (.zip)")
    ap.add_argument("--local", action="store_true", help="проверить рабочую копию вместо архива")
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    args = ap.parse_args()
    if args.local == (args.archive is not None):
        ap.error("нужен либо путь к архиву, либо --local")
    manifest = load_manifest(args.manifest)
    if args.local:
        missing, different = check_local(ROOT, manifest)
        for name in different:
            print(f"FAIL  другое содержимое: {name} [{origin(name)}]")
        if missing:
            print(f"нет {len(missing)} файлов ({_group(missing)}); веса — models/README.md, "
                  "vendor/ и колесо RVT — scripts/fetch_thirdparty.py")
        ok = not missing and not different
        print(f"рабочая копия: {len(manifest) - len(missing) - len(different)} из {len(manifest)} файлов совпадают")
        print("ИТОГ:", "PASS — scripts/build_submission.py соберёт сданный архив" if ok else "FAIL")
        return 0 if ok else 1
    sha = _file_sha256(args.archive)
    print(f"архив {args.archive}: sha256 {sha}")
    if sha == FINAL_SHA256:
        print("OK    это архив финальной версии (попытка 19.11.2026 22:29 МСК)")
    problems = check_zip(args.archive, manifest)
    for p in problems:
        print(f"FAIL  {p}")
    if not problems:
        print(f"OK    {len(manifest)} членов совпадают с {args.manifest.name}, метаданные zip детерминированы")
    ok = not problems and sha == FINAL_SHA256
    if not problems and sha != FINAL_SHA256:
        print("FAIL  члены совпадают, а sha256 архива другой: проверьте версию zipfile/zlib (сжатие DEFLATE)")
    print("ИТОГ:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
