#!/usr/bin/env python3
"""DOC-1: в архиве решения нет файлов, производных от EN (EhimenNathan), и PIXGBM-legacy не включён.

    python scripts/check_archive_clean.py dist/submission.zip      # или каталог распакованного архива

Код возврата 1 и список нарушений, если в архиве есть:
* expds/geo_pixgbm.py, expds/geo_finds_gbm.py (под любым путём);
* что-либо из models/pixgbm/ (старые модели PIXGBM), analysis/, w4pix/, pc/tools/, scripts/pixgbm*.py;
* configs/*.toml (default.toml и любые другие) с [geophys.pixgbm] enabled = true и impl ≠ "clean"
  (legacy-импорт geo_pixgbm; без модуля — тихий откат на классику).
Вызывается из scripts/build_submission.py после записи архива; содержимое запрещённых файлов не читается.
"""
from __future__ import annotations

import re
import sys
import tomllib
import zipfile
from pathlib import Path

BANNED_NAMES = {"geo_pixgbm.py", "geo_finds_gbm.py"}
BANNED_PREFIXES = ("models/pixgbm/", "analysis/", "w4pix/", "pc/tools/")
BANNED_RE = re.compile(r"(^|/)scripts/pixgbm[^/]*\.py$")


def name_violations(names) -> list[str]:
    bad = []
    for n in names:
        n = n.replace("\\", "/").lstrip("./")
        if n.endswith("/"):
            continue
        if Path(n).name in BANNED_NAMES or n.startswith(BANNED_PREFIXES) or BANNED_RE.search(n) \
                or "/models/pixgbm/" in f"/{n}":
            bad.append(n)
    return bad


def config_violations(name: str, text: str) -> list[str]:
    try:
        cfg = tomllib.loads(text)
    except Exception as exc:
        return [f"{name}: не читается ({exc})"]
    pix = (cfg.get("geophys") or {}).get("pixgbm") or {}
    if pix.get("enabled", False) and str(pix.get("impl", "legacy")) != "clean":
        return [f"{name}: [geophys.pixgbm] enabled = true (impl={pix.get('impl', 'legacy')!r}) — legacy PIXGBM"]
    return []


def check(path: Path) -> list[str]:
    path = Path(path)
    if path.is_dir():
        names = [p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()]
        read = lambda n: (path / n).read_text(encoding="utf-8")  # noqa: E731
    else:
        z = zipfile.ZipFile(path)
        names = z.namelist()
        read = lambda n: z.read(n).decode("utf-8")  # noqa: E731
    bad = [f"файл EN/PIXGBM-legacy: {n}" for n in name_violations(names)]
    for n in names:
        if n.startswith("configs/") and n.endswith(".toml"):
            bad += config_violations(n, read(n))
    return bad


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    bad = check(Path(sys.argv[1]))
    if bad:
        print("DOC-1: архив НЕ чистый:\n  " + "\n  ".join(bad))
        sys.exit(1)
    print(f"DOC-1: OK {sys.argv[1]}")


if __name__ == "__main__":
    main()
