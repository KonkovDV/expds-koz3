#!/usr/bin/env python3
"""Положить в рабочее дерево четыре файла, которых нет в main, но которые были в архиве v2.

Архив v2 (submission_bot7cand_v2_eb268d6.zip, LB 0,41238, sha256 8f3f2671…d589) собран из
коммита eb268d6. В main этих файлов нет (EN-удержание и правило «веса не в git»), а
build_submission.py без них падает («модель из конфига не найдена»). Скрипт берёт их
ПОБАЙТНО из объектов git коммита eb268d6 (или из папки --from), сверяет sha256 и кладёт
неотслеживаемыми (.gitignore). Содержимое не печатается и не коммитится.

    python scripts/v2_release_artifacts.py            # положить и сверить
    python scripts/v2_release_artifacts.py --check    # только сверить (код 1 — нет/не тот)
    python scripts/v2_release_artifacts.py --from DIR # взять из DIR/<путь> вместо git

После него `python scripts/build_submission.py` даёт тот же набор из 50 файлов модулей и весов,
что и v2 (плюс новые модули main, выключенные флагами); golden e2e — scripts/golden_compare.py.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "eb268d6"
MANIFEST = {
    "expds/geo_pixgbm.py": "35e4331bbc690e7eb3b31e07cc290837ce9fd96a05c92f4c999f24eceafda35b",
    "models/pixgbm/all/poly.txt": "d1fe3efafcd4377d41430d8828e881576c9024f9d7a5c89f55e23554e02ec800",
    "models/pixgbm/all/finds.txt": "5fa8c09d77d0914cc925fb6bb925a0ab1b64ccecf3c823d7f3546e71c4ceb04a",
    "models/rerank_arch_lgbm.txt": "0bc2e924b7a7beb79706eae275d516a2668406294e99b6caf44e454030b6d97b",
}


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def from_git(rel: str) -> bytes:
    spec = f"{SOURCE_COMMIT}:{rel}"
    r = subprocess.run(["git", "-C", str(ROOT), "cat-file", "blob", spec], capture_output=True)
    if r.returncode != 0:
        subprocess.run(["git", "-C", str(ROOT), "fetch", "-q", "origin", SOURCE_COMMIT], capture_output=True)
        r = subprocess.run(["git", "-C", str(ROOT), "cat-file", "blob", spec], capture_output=True)
    if r.returncode != 0:
        raise SystemExit(f"нет объекта {spec} (git fetch origin {SOURCE_COMMIT} или --from DIR)")
    return r.stdout


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="только сверить sha256")
    ap.add_argument("--from", dest="src", type=Path, help="папка-источник с теми же путями")
    ap.add_argument("--force", action="store_true", help="перезаписать файл с другим sha256")
    args = ap.parse_args()
    bad = 0
    for rel, want in MANIFEST.items():
        dst = ROOT / rel
        have = sha(dst.read_bytes()) if dst.exists() else None
        if have == want:
            print(f"OK    {rel} {want[:16]}")
            continue
        if args.check:
            print(f"FAIL  {rel}: {'нет файла' if have is None else 'sha256 ' + have[:16] + ' ≠ ' + want[:16]}")
            bad += 1
            continue
        if have is not None and not args.force:
            print(f"FAIL  {rel}: уже есть с sha256 {have[:16]} ≠ {want[:16]} (--force, чтобы заменить)")
            bad += 1
            continue
        data = (args.src / rel).read_bytes() if args.src else from_git(rel)
        if sha(data) != want:
            print(f"FAIL  {rel}: источник sha256 {sha(data)[:16]} ≠ {want[:16]}")
            bad += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        print(f"PUT   {rel} {want[:16]} ({len(data)} Б, из {'--from' if args.src else SOURCE_COMMIT})")
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--", *MANIFEST], capture_output=True, text=True).stdout.split()
    if tracked:
        print(f"FAIL  в индексе git: {tracked} — эти файлы не коммитятся")
        bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
