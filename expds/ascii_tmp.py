"""Каталог, в пути которого только латиница.

LightGBM и TorchScript на Windows не пишут файл, если в пути есть кириллица.
Домашний каталог пользователя таким путём как раз является, и системный TEMP
тогда тоже. Сдача на площадке идёт по латинскому пути и этот мост не вызывает.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


def _long_name(path: Path) -> str:
    """Windows 8.3 прячет кириллицу: `73B5~1` — это «Пользователь», и LightGBM видит длинное имя."""
    text = os.fspath(path)
    if os.name != "nt":
        return os.path.abspath(text)
    try:
        import ctypes

        buf = ctypes.create_unicode_buffer(4096)
        n = ctypes.windll.kernel32.GetLongPathNameW(text, buf, len(buf))
        if n and n < len(buf):
            return buf.value
    except OSError:
        pass
    return os.path.abspath(text)


def _ascii(path: Path) -> bool:
    try:
        _long_name(path).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def ascii_tmpdir() -> Path:
    """Существующий каталог, куда можно записать файл и чьё имя кодируется ASCII."""
    candidates: list[Path] = []
    env = os.environ.get("EXPDS_ASCII_TMP")
    if env:
        candidates.append(Path(env))
    root = os.environ.get("SystemRoot") or os.environ.get("WINDIR")
    if root:
        candidates.append(Path(root) / "Temp")
    candidates.append(Path("/tmp"))
    here = Path(__file__).resolve().parents[1]
    if _ascii(here):
        candidates.append(here / ".ascii_tmp")
    for folder in candidates:
        if not _ascii(folder):
            continue
        try:
            folder.mkdir(parents=True, exist_ok=True)
            probe = folder / ".expds_ascii_probe"
            probe.write_text("ok", encoding="ascii")
            probe.unlink()
        except OSError:
            continue
        return folder
    raise OSError("нет каталога с латинским путём для LightGBM")


def pin_process_temp() -> Path | None:
    """Перенаправляет TEMP этого процесса, если системный TEMP не латинский."""
    current = Path(tempfile.gettempdir())
    if _ascii(current):
        return current
    try:
        folder = ascii_tmpdir()
    except OSError:
        return None
    os.environ["TEMP"] = os.environ["TMP"] = str(folder)
    tempfile.tempdir = str(folder)
    return folder
