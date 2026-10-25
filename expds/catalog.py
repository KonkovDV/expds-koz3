"""Поиск регионов и файлов.

Обучающая выборка: корень/NNN_Название/{01_..._La_точки, 02_..._Li_карты, ...,
06_..._разметка, extra/}. Валидационная: корень/region_XXXX/... (без разметки
и без готовых Li). Функции не полагаются на точные имена папок: файлы
классифицируются по расширению и модальности (см. naming.py).
"""
from __future__ import annotations

import json
import logging
import re
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
    """Сырой и ground одного охвата — одно облако.

    В одной папке одинаковое имя решает само: так сырой файл и ground не
    расходятся из-за кривого заголовка. В разных папках одного участка то же
    имя склеивается только при пересечении меньшей рамки не меньше 80 % или
    если рамка не прочиталась. Два тайла с одним именем и разными охватами
    остаются оба. Две непрочитанные рамки склеиваются только в одной папке.
    """
    same_name = _bare_stem(a) == _bare_stem(b)
    same_dir = a.parent == b.parent
    if same_name and same_dir:
        return True
    if box_a is None and box_b is None:
        return same_dir
    if box_a is None or box_b is None:
        return False
    return _overlap_ratio(box_a, box_b) >= 0.8


def choose_las_boxes(records: list[tuple[Path, tuple | None]]) -> list[Path]:
    """По одному файлу на облако. Тайлы с разным охватом сохраняются.

    Из пары берётся имя с «ground». Пересечение считается по цепочке и по всему
    участку: копия того же облака в другой папке не даёт второй проход.
    """
    parent = list(range(len(records)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, (path_i, box_i) in enumerate(records):
        for j in range(i):
            path_j, box_j = records[j]
            if _same_cloud(path_i, box_i, path_j, box_j):
                parent[find(i)] = find(j)
    clusters: dict[int, list[Path]] = defaultdict(list)
    for i, (path, _box) in enumerate(records):
        clusters[find(i)].append(path)
    chosen: list[Path] = []
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
                # Модальности вне ТР (георадар и любое неузнанное имя) не открываются.
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
                    # structure.poly метрика не считает; файл без класса в эталон не входит.
                    reg.skipped.append(f)
        else:
            reg.skipped.append(f)
    reg.las = _choose_las(las_files)
    return reg


def _has_data(reg: Region) -> bool:
    return bool(reg.las or reg.rasters or reg.annotations or reg.li_rasters)


def _looks_like_region(name: str) -> bool:
    """Папка участка, а не модальности: region_*, NNN_Название или *_FINAL."""
    return name.lower().startswith("region") or name.endswith("_FINAL") or bool(re.match(r"^\d{3}_", name))


def scan_root(root: str | Path) -> list[Region]:
    """Регионы в корне. Если папок участков нет, корень сам считается одним участком.

    Обычная раскладка (папки первого уровня с данными) не меняется. Запасные:
    участки на уровень глубже одной обёртки и плоская папка с файлами.
    """
    root = Path(root)
    dirs = [d for d in sorted(root.iterdir()) if d.is_dir() and not d.name.startswith(".")]
    regions = [r for r in (scan_region(d) for d in dirs) if _has_data(r)]
    if regions and len(dirs) == 1 and not _looks_like_region(dirs[0].name):
        inner_dirs = [d for d in sorted(dirs[0].iterdir()) if d.is_dir() and _looks_like_region(d.name)]
        inner = [r for r in (scan_region(d) for d in inner_dirs) if _has_data(r)]
        if len(inner) > 1:
            log.warning("участки лежат на уровень глубже: %s", dirs[0].name)
            regions = inner
    modality_dirs = [r for r in regions if re.match(r"^\d{2}_", r.path.name)]
    if regions and len(modality_dirs) == len(regions):
        log.warning("во входной папке папки модальностей — это один участок: %s", root.name)
        regions = []
    if not regions:
        flat = scan_region(root)
        if _has_data(flat):
            log.warning("во входной папке нет подпапок участков — обрабатываю её как один участок")
            regions = [flat]
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
        low = text.lower()
        # «EPSG» без кода не должно становиться строкой «EPSG:»: такой CRS ломает участок.
        if low not in ("epsg", "epsg:"):
            if low.startswith("urn:ogc:def:crs:epsg::"):
                code = text.split("::")[-1].strip()
                if code.isdigit():
                    return f"EPSG:{code}"
            elif low.startswith("epsg") and ":" not in text:
                code = text[4:].strip(": ")
                if code.isdigit():
                    return f"EPSG:{code}"
            else:
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
