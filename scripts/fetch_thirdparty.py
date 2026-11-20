#!/usr/bin/env python3
"""Сторонние файлы архива из PyPI: vendor/ (lleaves, llvmlite) и колесо RVT.

    python scripts/fetch_thirdparty.py               # скачать, проверить sha256 колёс, разложить
    python scripts/fetch_thirdparty.py --check       # только проверить уже разложенные файлы
    python scripts/fetch_thirdparty.py --wheels DIR  # колёса уже скачаны (без сети)

В git этих файлов нет: в архиве финальной версии они лежат как есть. 64 файла `vendor/` — распакованные без
изменений колёса lleaves 1.3.0 и llvmlite 0.44.0 (CPython 3.12, manylinux2014 x86_64), вместе с
`*.dist-info` и текстами лицензий (MIT, BSD-2-Clause). `inference/cnx/wheels/rvt_py-2.2.1-py3-none-any.whl` —
исходное колесо RVT. Скрипт сверяет sha256 каждого колеса и каждого разложенного файла с
`docs/final_archive.sha256`, поэтому результат побайтно равен членам сданного архива.
Не из публичных источников берутся только веса `models/` (models/README.md).
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "final_archive.sha256"

#: имя колеса → (требование pip, sha256 колеса на PyPI, куда класть: "unpack" в vendor/ или путь файла).
WHEELS = {
    "lleaves-1.3.0-py3-none-any.whl":
        ("lleaves==1.3.0", "c25d37b3240f36daa8cf6b48847a63f7b37ec2f57749b3ce2997d2a9d5544eb5", "unpack"),
    "llvmlite-0.44.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl":
        ("llvmlite==0.44.0", "c0143a5ef336da14deaa8ec26c5449ad5b6a2b564df82fcef4be040b9cacfea9", "unpack"),
    "rvt_py-2.2.1-py3-none-any.whl":
        ("rvt-py==2.2.1", "f3b171e056d8a808b7c58d8ed74db6eb1b2088e728008e7a1c9f1e9fb1d2a513",
         "inference/cnx/wheels/rvt_py-2.2.1-py3-none-any.whl"),
}
PREFIXES = ("vendor/", "inference/cnx/wheels/")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def expected() -> dict[str, str]:
    """Члены архива из публичных колёс: {путь: sha256} из docs/final_archive.sha256."""
    out = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.strip():
            sha, name = line.split(maxsplit=1)
            if name.startswith(PREFIXES):
                out[name.strip()] = sha
    return out


def download(dest: Path) -> None:
    """pip download под платформу архива: CPython 3.12, manylinux2014 x86_64, только колёса, без зависимостей."""
    cmd = [sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary", ":all:",
           "--python-version", "3.12", "--implementation", "cp", "--abi", "cp312", "--abi", "none",
           "--platform", "manylinux_2_17_x86_64", "--platform", "manylinux2014_x86_64", "--platform", "any",
           "-d", str(dest)] + [req for req, _, _ in WHEELS.values()]
    subprocess.run(cmd, check=True)


def unpack(wheel: Path, root: Path) -> list[str]:
    """Файлы колеса → root/vendor/<путь в колесе>. Каталоги не создаются отдельно, пути проверяются."""
    written = []
    with zipfile.ZipFile(wheel) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            rel = PurePosixPath(info.filename)
            if rel.is_absolute() or ".." in rel.parts:
                raise SystemExit(f"{wheel.name}: недопустимый путь {info.filename}")
            dest = root / "vendor" / Path(*rel.parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(z.read(info))
            written.append(f"vendor/{rel.as_posix()}")
    return written


def check(root: Path) -> list[str]:
    """Расхождения разложенных файлов с манифестом архива (пустой список — всё совпадает)."""
    bad = []
    for name, sha in sorted(expected().items()):
        p = root / name
        if not p.is_file():
            bad.append(f"нет {name}")
        elif sha256(p) != sha:
            bad.append(f"другое содержимое: {name}")
    return bad


def install(wheels_dir: Path, root: Path) -> None:
    for name, (_req, sha, target) in WHEELS.items():
        wheel = wheels_dir / name
        if not wheel.is_file():
            raise SystemExit(f"нет колеса {wheel}")
        if sha256(wheel) != sha:
            raise SystemExit(f"{name}: sha256 не совпал с колесом PyPI — файл не тот")
        if target == "unpack":
            unpack(wheel, root)
        else:
            dest = root / target
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(wheel.read_bytes())
    want = expected()
    extra = sorted(p.relative_to(root).as_posix() for p in (root / "vendor").rglob("*")
                   if p.is_file() and "__pycache__" not in p.parts
                   and p.relative_to(root).as_posix() not in want)
    if extra:
        raise SystemExit("в vendor/ лишние файлы, архив с ними не совпадёт: " + ", ".join(extra[:5]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="только проверить уже разложенные файлы")
    ap.add_argument("--wheels", type=Path, default=None, help="папка с уже скачанными колёсами")
    args = ap.parse_args()
    if not args.check:
        if args.wheels is not None:
            install(args.wheels, ROOT)
        else:
            with tempfile.TemporaryDirectory() as tmp:
                download(Path(tmp))
                install(Path(tmp), ROOT)
    bad = check(ROOT)
    for b in bad:
        print(f"FAIL  {b}")
    n = len(expected())
    print(f"{n - len(bad)} из {n} файлов vendor/ и колеса RVT совпадают с архивом финальной версии")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
