"""Поиск регионов и файлов.

Обучающая выборка: корень/NNN_Название/{01_..._La_точки, 02_..._Li_карты, ...,
06_..._разметка, extra/}. Валидационная: корень/region_XXXX/... (без разметки
и без готовых Li). Функции не полагаются на точные имена папок: файлы
классифицируются по расширению и модальности (см. naming.py).
"""
from __future__ import annotations

import json
import logging
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .naming import class_of_annotation, markup_type_of, modality_of

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


def _bare_stem(path: Path) -> str:
    name = path.stem.lower()
    for suffix in ("_ground", "-ground"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _overlap_ratio(a: tuple, b: tuple) -> float:
    """Доля площади меньшей рамки, покрытая пересечением."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    iw, ih = min(ax1, bx1) - max(ax0, bx0), min(ay1, by1) - max(ay0, by0)
    if iw <= 0 or ih <= 0:
        return 0.0
    smaller = min(max(ax1 - ax0, 0.0) * max(ay1 - ay0, 0.0), max(bx1 - bx0, 0.0) * max(by1 - by0, 0.0))
    return (iw * ih) / smaller if smaller else 0.0


def _same_cloud(a: Path, box_a: tuple | None, b: Path, box_b: tuple | None) -> bool:
    """Сырой и ground одного охвата — одно облако. Разные тайлы — нет.

    Непрочитанная рамка склеивается только с таким же именем или с другой
    непрочитанной рамкой. Иначе один битый заголовок забрал бы все тайлы папки.
    """
    if _bare_stem(a) == _bare_stem(b):
        return True
    if box_a is None and box_b is None:
        return True
    if box_a is None or box_b is None:
        return False
    return _overlap_ratio(box_a, box_b) >= 0.8


def choose_las_boxes(records: list[tuple[Path, tuple | None]]) -> list[Path]:
    """По одному файлу на облако. Тайлы одной папки с разным охватом сохраняются.

    Из пары берётся имя с «ground». Пересечение считается по цепочке: если
    средний файл накрывает оба края, остаётся один проход.
    """
    by_parent: dict[Path, list[tuple[Path, tuple | None]]] = defaultdict(list)
    for path, box in records:
        by_parent[path.parent].append((path, box))
    chosen: list[Path] = []
    for group in by_parent.values():
        parent = list(range(len(group)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i, (path_i, box_i) in enumerate(group):
            for j in range(i):
                path_j, box_j = group[j]
                if _same_cloud(path_i, box_i, path_j, box_j):
                    parent[find(i)] = find(j)
        clusters: dict[int, list[Path]] = defaultdict(list)
        for i, (path, _box) in enumerate(group):
            clusters[find(i)].append(path)
        for paths in clusters.values():
            best = paths[0]
            for path in paths[1:]:
                if "ground" in path.name.lower() and "ground" not in best.name.lower():
                    best = path
            chosen.append(best)
    return chosen


def _las_box(path: Path) -> tuple | None:
    try:
        import laspy
        with laspy.open(path) as src:
            h = src.header
            return (float(h.mins[0]), float(h.mins[1]), float(h.maxs[0]), float(h.maxs[1]))
    except Exception:
        log.warning("рамка LAS не прочитана, файл не считается отдельным тайлом: %s", path.name)
        return None


def _choose_las(files: list[Path]) -> list[Path]:
    """Облака папки: ground вместо сырого того же охвата, остальные тайлы все."""
    return choose_las_boxes([(path, _las_box(path)) for path in files])


def scan_region(path: Path) -> Region:
    """Классифицирует все файлы региона."""
    reg = Region(name=path.name, path=path)
    las_files: list[Path] = []
    for f in sorted(path.rglob("*")):
        if not f.is_file():
            continue
        suf = f.suffix.lower()
        low = unicodedata.normalize("NFC", f.name).lower()   # без замены двойников: иначе «границы» → «гpaницы»
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
            # По последнему слову имени: «Шумгора_Li_курганы_целые» — разметка, а не маска шума.
            last = low.rsplit(".", 1)[0].split("_")[-1]
            if last.startswith("границ"):
                reg.boundaries.append(f)
            elif last == "шум":
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
