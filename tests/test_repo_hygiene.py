"""Что может лежать в репозитории: проверка дерева git на данные конкурса, веса, архивы и секреты.

Данные конкурса и всё, что из них получено, публиковать нельзя (ТР, п. 13.10), веса едут в архиве решения, а
не в git (models/README.md). `.gitignore` защищает только от случайного `git add`, а `git add -f` или правка
`.gitignore` проходят без следа, поэтому тест проверяет то, что действительно закоммичено.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Веса, облака, растры, массивы, предсказания и архивы: в git их быть не должно.
FORBIDDEN_SUFFIXES = {
    ".pt", ".pth", ".ckpt", ".onnx", ".safetensors", ".h5",
    ".las", ".laz", ".tif", ".tiff", ".jp2", ".npy", ".npz", ".parquet", ".pkl", ".pickle",
    ".geojson", ".shp", ".gpkg",
    ".zip", ".whl", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar", ".tar",
    ".pdf", ".docx", ".xlsx", ".pptx",
}
# Единственные двоичные файлы дерева — Rust-ядро в архиве решения (сверяется манифестом).
ALLOWED_BINARIES = {"inference/lib/expds_core.abi3.so", "inference/cnx/vendor/expds_core.abi3.so"}
MAX_BYTES = 3_000_000

SECRET_PATTERNS = {
    "ключ доступа AWS": re.compile(rb"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "статический ключ Yandex Cloud": re.compile(rb"\bYCAJ[A-Za-z0-9_\-]{20,}\b"),
    "токен GitHub": re.compile(rb"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}\b|\bgithub_pat_[A-Za-z0-9_]{40,}\b"),
    "токен Kaggle": re.compile(rb"\bKGAT_[A-Za-z0-9]{16,}\b"),
    "kaggle.json": re.compile(rb"\"username\"\s*:\s*\"[^\"]+\"\s*,\s*\"key\"\s*:\s*\"[0-9a-f]{32}\""),
    "закрытый ключ": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "токен Hugging Face": re.compile(rb"\bhf_[A-Za-z0-9]{30,}\b"),
    "пароль в URL": re.compile(rb"[a-z][a-z0-9+.\-]*://[^/\s:@'\"{]+:[^/\s@'\"{}]{6,}@[^\s'\"]+"),
}


def tracked_files() -> list[str]:
    try:
        out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("нет git или это не рабочая копия git (например, распакованный архив решения)")
    files = [f for f in out.decode("utf-8").split("\0") if f]
    if not files:
        pytest.skip("git не видит файлов")
    return files


#: Состав дерева. Новая папка верхнего уровня, документ или модуль expds вне архива добавляются сюда осознанно:
#: так посторонние файлы не попадут в репозиторий вместе с очередным `git add -A`.
TOP_LEVEL = {".dockerignore", ".gitattributes", ".github", ".gitignore", "Dockerfile", "LICENSE",
             "README.md", "README.en.md", "centerhead", "configs", "docs", "expds", "inference", "materials", "metadata.json",
             "models", "pyproject.toml", "requirements-dev.txt", "requirements.txt", "rust", "scripts", "tests",
             "training"}
DOCS = {"docs/README.md", "docs/TECHNICAL_REPORT.md", "docs/ai_declaration.md", "docs/final_archive.sha256"}
EXPDS_OFFLINE = {"expds/gt.py"}  # эталон обучения; в архив решения не входит


def archive_members() -> set[str]:
    lines = (ROOT / "docs" / "final_archive.sha256").read_text(encoding="utf-8").splitlines()
    return {ln.split(None, 1)[1].strip() for ln in lines if ln.strip()}


def test_tree_layout_is_the_published_one():
    files = tracked_files()
    top = {f.split("/", 1)[0] for f in files}
    assert top <= TOP_LEVEL, f"новое на верхнем уровне: {sorted(top - TOP_LEVEL)}"
    docs = {f for f in files if f.startswith("docs/")}
    assert docs <= DOCS, f"новые документы в docs/: {sorted(docs - DOCS)}"
    expds = {f for f in files if f.startswith("expds/")}
    extra = expds - archive_members() - EXPDS_OFFLINE
    assert not extra, f"модули expds/, которых нет в архиве решения: {sorted(extra)}"


def test_no_data_weights_or_archives_are_tracked():
    bad = [f for f in tracked_files()
           if Path(f).suffix.lower() in FORBIDDEN_SUFFIXES or Path(f).name in {"kaggle.json", ".env"}]
    assert not bad, f"в git попали данные, веса или архивы: {bad}"


def test_tracked_files_are_small_and_text_except_the_rust_core():
    big, binary = [], []
    for f in tracked_files():
        p = ROOT / f
        if not p.is_file() or f in ALLOWED_BINARIES:
            continue
        if p.stat().st_size > MAX_BYTES:
            big.append(f)
        if b"\0" in p.read_bytes()[:8192]:
            binary.append(f)
    assert not big, f"файлы больше {MAX_BYTES} байт: {big}"
    assert not binary, f"двоичные файлы вне Rust-ядра: {binary}"


def test_no_secrets_in_tracked_files():
    found = []
    for f in tracked_files():
        p = ROOT / f
        if not p.is_file() or f in ALLOWED_BINARIES:
            continue
        data = p.read_bytes()
        found += [f"{f}: {name}" for name, rx in SECRET_PATTERNS.items() if rx.search(data)]
    assert not found, f"похоже на секрет: {found}"
