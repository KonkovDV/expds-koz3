#!/usr/bin/env python3
"""DOC-1: в архиве решения нет файлов, производных от EN (EhimenNathan), и PIXGBM-legacy не включён.

    python scripts/check_archive_clean.py dist/submission.zip      # или каталог распакованного архива

Код возврата 1 и список нарушений, если в архиве есть:
* expds/geo_pixgbm.py, expds/geo_finds_gbm.py (под любым путём);
* что-либо из models/pixgbm/ (старые модели PIXGBM), analysis/, w4pix/, pc/tools/, scripts/pixgbm*.py;
* configs/*.toml (default.toml и любые другие) с [geophys.pixgbm] enabled = true и impl ≠ "clean"
  (legacy-импорт geo_pixgbm; без модуля — тихий откат на классику);
* (жёсткий DOC-1, 30.10) любой файл, sha256 которого есть в scripts/en_denylist.sha256 (sha256 файлов,
  производных от EN: все версии geo_pixgbm.py/geo_finds_gbm.py, models/pixgbm/*, analysis/*, scripts/pixgbm*.py
  из всех веток, w4pix/ — посчитаны sha256sum без просмотра содержимого);
* любой файл архива (текст и бинарь), в байтах которого встречается geo_pixgbm | EhimenNathan | pixgbm/
  (без учёта регистра);
* конфиг, ссылающийся на прогоны/модели сравнения на geo_pixgbm (курганы, w4pix, hillshade-parity и т.п.):
  строковые значения configs/*.toml проверяются по CONFIG_BANNED_RE.
Вызывается из scripts/build_submission.py и build_submission_git.py до публикации архива (иначе exit 1);
содержимое запрещённых файлов не читается — только имена, sha256 и поиск шаблона в байтах СОБИРАЕМОГО архива.
"""
from __future__ import annotations

import hashlib
import re
import sys
import tomllib
import zipfile
from pathlib import Path

BANNED_NAMES = {"geo_pixgbm.py", "geo_finds_gbm.py", "pix_legacy.py"}
DENYLIST = Path(__file__).with_name("en_denylist.sha256")
CONTENT_RE = re.compile(rb"geo_pixgbm|EhimenNathan|pixgbm/", re.IGNORECASE)
CONFIG_BANNED_RE = re.compile(r"w4pix|models/pixgbm|pixgbm/|geo_pixgbm|geo_finds_gbm|ehimen|kurgan[_-]?pix|pix[_-]?kurgan"
                              r"|hillshade[_-]?parity|hsparity", re.IGNORECASE)


def load_denylist(path: Path = DENYLIST) -> dict[str, str]:
    """sha256 → метка; файл обязателен (нет файла — сборка не проходит)."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            h, _, label = line.partition("  ")
            out[h.lower()] = label
    return out


def strings(obj, key=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from strings(v, f"{key}.{k}" if key else k)
            yield key, str(k)
    elif isinstance(obj, list):
        for v in obj:
            yield from strings(v, key)
    elif isinstance(obj, str):
        yield key, obj
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
    bad = [f"{name}: {k} = {v!r} — ссылка на прогон/модель сравнения на geo_pixgbm (DOC-1)"
           for k, v in strings(cfg) if CONFIG_BANNED_RE.search(v)]
    pix = (cfg.get("geophys") or {}).get("pixgbm") or {}
    if pix.get("enabled", False) and str(pix.get("impl", "legacy")) != "clean":
        bad.append(f"{name}: [geophys.pixgbm] enabled = true (impl={pix.get('impl', 'legacy')!r}) — legacy PIXGBM")
    return bad


def check(path: Path, denylist: Path = DENYLIST) -> list[str]:
    path = Path(path)
    if path.is_dir():
        names = [p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()]
        raw = lambda n: (path / n).read_bytes()  # noqa: E731
    else:
        z = zipfile.ZipFile(path)
        names = [n for n in z.namelist() if not n.endswith("/")]
        raw = lambda n: z.read(n)  # noqa: E731
    bad = [f"файл EN/PIXGBM-legacy: {n}" for n in name_violations(names)]
    try:
        deny = load_denylist(denylist)
    except OSError as exc:
        return bad + [f"нет списка sha256 EN ({denylist}): {exc}"]
    if not deny:
        bad.append(f"список sha256 EN пуст: {denylist}")
    for n in names:
        if n in bad:
            continue
        b = raw(n)
        h = hashlib.sha256(b).hexdigest()
        if h in deny:
            bad.append(f"sha256 из списка EN: {n} = {deny[h]}")
        m = CONTENT_RE.search(b)
        if m:
            bad.append(f"шаблон EN в содержимом: {n} ({m.group(0).decode('latin-1')!r} @ {m.start()})")
        if n.startswith("configs/") and n.endswith(".toml"):
            bad += config_violations(n, b.decode("utf-8"))
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
