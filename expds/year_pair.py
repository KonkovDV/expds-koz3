"""Парное сравнение двух снимков одного участка.

Имена в выборке — `01_`, `02_`. Года в них нет, поэтому пара 2024/2025
берётся только из даты в имени или в теге снимка. Общий сдвиг яркости
(другой сезон, другая экспозиция) вычитается медианой и пика не даёт.
Остаётся локальная метка: круг, которого нет на втором снимке.

В сдачу не входит. Вызов — `python -m expds.year_pair снимок_2024 снимок_2025`.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_YEAR = re.compile(r"(?<!\d)(20\d{2})(?!\d)")
_TIFF_DT = re.compile(r"(20\d{2})[:\-]")
_TAG_KEYS = ("DateTime", "TIFFTAG_DATETIME", "datetime", "ACQUISITION_DATE", "acquisition_date")


def years_in_text(text: str) -> tuple[int, ...]:
    """Годы 20xx в строке, по порядку, без повторов."""
    found: list[int] = []
    for raw in _YEAR.findall(text or ""):
        year = int(raw)
        if year not in found:
            found.append(year)
    return tuple(found)


def year_from_tags(tags: dict | None) -> int | None:
    """Год из словаря тегов GeoTIFF. Нет даты — None, не угадываем по номеру файла."""
    if not tags:
        return None
    for key in _TAG_KEYS:
        raw = tags.get(key)
        if not raw:
            continue
        match = _TIFF_DT.search(str(raw))
        if match:
            return int(match.group(1))
    return None


def pick_year_pair(names: list[str], early: int = 2024, late: int = 2025) -> tuple[str, str] | None:
    """Один файл раннего года и один позднего. `01_`/`02_` без года — не пара."""
    first = [name for name in names if early in years_in_text(Path(name).name)]
    second = [name for name in names if late in years_in_text(Path(name).name)]
    if len(first) == 1 and len(second) == 1 and first[0] != second[0]:
        return first[0], second[0]
    return None


def _luminance(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=0)
    return arr


def local_change(early: np.ndarray, late: np.ndarray) -> np.ndarray:
    """Модуль разности после вычитания медианного сдвига. Вне кадра — 0.

    Медиана, а не среднее: маленький круг не сдвигает уровень всего снимка.
    """
    a = _luminance(early)
    b = _luminance(late)
    if a.shape != b.shape:
        raise ValueError(f"снимки разного размера: {a.shape} и {b.shape}")
    valid = np.isfinite(a) & np.isfinite(b)
    out = np.zeros(a.shape, np.float32)
    if not bool(valid.any()):
        return out
    shift = float(np.median((b - a)[valid]))
    out[valid] = np.abs(b[valid] - (a[valid] + np.float32(shift)))
    return out


def disk_contrast(change: np.ndarray, radius_px: int) -> np.ndarray:
    """Среднее в круге минус среднее в кольце вокруг него.

    Круглая метка даёт плюс. Прямая граница поля и общий шум — около нуля.
    Окно считает Rust (expds_core.box_mean), если модуль собран.
    """
    from .accel import box_mean

    radius = int(radius_px)
    if radius < 1:
        raise ValueError("радиус в пикселях меньше 1")
    src = np.asarray(change, np.float32)
    inner = box_mean(src, radius)
    outer = box_mean(src, 2 * radius)
    inner_n = float((2 * radius + 1) ** 2)
    outer_n = float((4 * radius + 1) ** 2)
    ring = (outer * outer_n - inner * inner_n) / (outer_n - inner_n)
    return np.nan_to_num(inner - ring, nan=0.0).astype(np.float32, copy=False)


def strongest_mark(change: np.ndarray, radius_px: int) -> tuple[int, int, float]:
    """Строка, столбец и контраст самой сильной круглой метки."""
    score = disk_contrast(change, radius_px)
    flat = int(np.argmax(score))
    row, col = divmod(flat, score.shape[1])
    return int(row), int(col), float(score[row, col])


def _read_pair(early: Path, late: Path, step_m: float, radius_m: float) -> tuple[int, int, float]:
    import rasterio

    def one(path: Path) -> tuple[np.ndarray, int | None]:
        with rasterio.open(path) as src:
            image = src.read().astype(np.float32)
            year = year_from_tags(src.tags())
        return image, year

    a, year_a = one(early)
    b, year_b = one(late)
    if year_a is None:
        year_a = years_in_text(early.name)[:1]
        year_a = year_a[0] if year_a else None
    if year_b is None:
        got = years_in_text(late.name)
        year_b = got[0] if got else None
    if {year_a, year_b} != {2024, 2025}:
        raise SystemExit(
            f"нужны снимки 2024 и 2025, в тегах и именах годы {year_a} и {year_b}. "
            "Номер файла 01/02 годом не считается.")
    radius_px = max(1, round(radius_m / step_m))
    return strongest_mark(local_change(a, b), radius_px)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Локальная разница двух снимков 2024 и 2025")
    parser.add_argument("early", type=Path)
    parser.add_argument("late", type=Path)
    parser.add_argument("--step-m", type=float, default=0.5)
    parser.add_argument("--radius-m", type=float, default=8.0)
    args = parser.parse_args(argv)
    row, col, contrast = _read_pair(args.early, args.late, args.step_m, args.radius_m)
    print(f"peak_row={row} peak_col={col} contrast={contrast:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
