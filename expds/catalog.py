"""Поиск регионов и файлов.

Обучающая выборка: корень/NNN_Название/{01_..._La_точки, 02_..._Li_карты, ...,
06_..._разметка, extra/}. Валидационная: корень/region_XXXX/... (без разметки
и без готовых Li). Функции не полагаются на точные имена папок: файлы
классифицируются по расширению и модальности (см. naming.py).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from .naming import class_of_annotation, markup_type_of, modality_of, normalize_code_text

log = logging.getLogger(__name__)

RASTER_EXT = (".tif", ".tiff")
LAS_EXT = (".las", ".laz")


@dataclass
class Region:
    """Один регион (участок) со всеми найденными файлами."""
    name: str
    path: Path
    las: list[Path] = field(default_factory=list)
    rasters: dict[str, list[Path]] = field(default_factory=dict)        # markup_type → файлы
    li_rasters: list[Path] = field(default_factory=list)                # готовые отмывки (только трейн)
    annotations: list[tuple[Path, str, str]] = field(default_factory=list)  # (файл, класс, markup_type)
    boundaries: list[Path] = field(default_factory=list)                # extra/*_границы.geojson
    noise: list[Path] = field(default_factory=list)                     # extra/*_шум.geojson
    utm_json: Path | None = None
    skipped: list[Path] = field(default_factory=list)

    @property
    def markup_types(self) -> list[str]:
        """Типы данных, по которым у региона есть входные файлы."""
        types = set(self.rasters)
        if self.las:
            types.add("li")
        return sorted(types)


def _choose_las(files: list[Path]) -> list[Path]:
    """По одному LAS на папку; отфильтрованный «ground»-файл предпочтительнее (как бейзлайн)."""
    chosen: dict[Path, Path] = {}
    for p in sorted(files):
        cur = chosen.get(p.parent)
        if cur is None or ("ground" in p.name.lower() and "ground" not in cur.name.lower()):
            chosen[p.parent] = p
    return list(chosen.values())


def scan_region(path: Path) -> Region:
    """Классифицирует все файлы региона."""
    reg = Region(name=path.name, path=path)
    las_files: list[Path] = []
    for f in sorted(path.rglob("*")):
        if not f.is_file():
            continue
        suf = f.suffix.lower()
        low = normalize_code_text(f.name).lower()
        if f.name == "UTM.json":
            reg.utm_json = f
        elif suf in LAS_EXT:
            las_files.append(f)
        elif suf in RASTER_EXT:
            mt = markup_type_of(f)
            if modality_of(f) == "Li":
                reg.li_rasters.append(f)
            elif mt:
                reg.rasters.setdefault(mt, []).append(f)
            else:
                reg.skipped.append(f)
        elif suf in (".geojson", ".json"):
            if "границ" in low:
                reg.boundaries.append(f)
            elif "шум" in low:
                reg.noise.append(f)
            else:
                cls, mt = class_of_annotation(f), markup_type_of(f)
                if cls and mt:
                    reg.annotations.append((f, cls, mt))
                else:
                    reg.skipped.append(f)
        else:
            reg.skipped.append(f)
    reg.las = _choose_las(las_files)
    return reg


def scan_root(root: str | Path) -> list[Region]:
    """Все регионы в корне выборки (папки первого уровня с файлами внутри)."""
    root = Path(root)
    regions = [scan_region(d) for d in sorted(root.iterdir()) if d.is_dir() and not d.name.startswith(".")]
    regions = [r for r in regions if r.las or r.rasters or r.annotations or r.li_rasters]
    log.info("найдено регионов: %d в %s", len(regions), root)
    return regions


def read_utm_json(path: Path | None) -> str | None:
    """EPSG из UTM.json региона: поле crs/CRS («EPSG:32636») или utm/UTM («36N»)."""
    if path is None or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # битый файл не должен ронять пайплайн
        log.warning("не прочитан %s: %s", path, exc)
        return None
    crs = data.get("crs") or data.get("CRS")
    if crs:
        text = str(crs).strip()
        if text.lower().startswith("urn:ogc:def:crs:epsg::"):
            return f"EPSG:{text.split('::')[-1]}"
        if text.lower().startswith("epsg") and ":" not in text:
            return f"EPSG:{text[4:].strip(': ')}"
        return text
    utm = data.get("utm") or data.get("UTM")
    if not utm:
        return None
    text = str(utm).strip().upper()
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits or not 1 <= int(digits) <= 60:
        return None
    # Южное полушарие — только явная «S» после номера зоны. Буквы широтных поясов
    # MGRS (C…X, например «36U») и слова «UTM ZONE» полушарие не меняют
    # (у бейзлайна «36U» превращалось в южную зону 32736).
    tail = text.replace("UTM", "").replace("ZONE", "").strip()
    south = tail.endswith("S") and not tail.endswith("NS")
    return f"EPSG:{327 if south else 326}{int(digits):02d}"
