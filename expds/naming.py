"""Разбор имён файлов и папок датасета.

Особенности, которые здесь учтены:
* в 7 файлах разметки трейна код модальности набран кириллицей («Ае»
  вместо «Ae») — без нормализации эти файлы теряются;
* в валидационной выборке регионы называются region_XXXX, а имена файлов
  могут отличаться от трейна (например, «Lidar», «немецкая аэрофотосъёмка»),
  поэтому кроме кодов проверяются и ключевые слова.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .constants import CLASS_ALIASES, MODALITY_TO_MARKUP

# Кириллические буквы, похожие на латинские (в кодах модальностей).
_LOOKALIKES = str.maketrans({
    "А": "A", "а": "a", "В": "B", "Е": "E", "е": "e", "К": "K", "М": "M",
    "Н": "H", "О": "O", "о": "o", "Р": "P", "р": "p", "С": "C", "с": "c",
    "Т": "T", "Х": "X", "х": "x", "у": "y",
})

_CODES = ("MagGr", "EtGr", "TerFl", "SpOr", "La", "Li", "Ae", "Or")
_CODE_RE = re.compile(r"(?:^|_)(" + "|".join(_CODES) + r")(?=_|\.|$)")

# Ключевые слова для имён, где кода нет (валидация и подписи не на русском).
# Код MagGr/EtGr/TerFl/SpOr/Ae/Or в имени проверяется раньше и побеждает.
# Слова ниже — только если кода нет: спутник без лидара и планшеты геофизики
# могут быть подписаны по-английски. Более длинное слово стоит выше короткого.
_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("магнит", "MagGr"), ("magnetometry", "MagGr"), ("magnetometer", "MagGr"),
    ("magnet", "MagGr"), ("maggr", "MagGr"),
    ("электро", "EtGr"), ("resistivity", "EtGr"), ("etgr", "EtGr"),
    ("термо", "TerFl"), ("thermography", "TerFl"), ("thermal", "TerFl"), ("terfl", "TerFl"),
    ("спутник", "SpOr"), ("satellite", "SpOr"), ("spor", "SpOr"), ("sp_or", "SpOr"),
    ("ортофото", "Or"), ("orthophoto", "Or"),
    ("немецк", "Ae"), ("luft", "Ae"), ("аэрофото", "Ae"), ("aerial", "Ae"),
    ("lidar", "Li"), ("hillshade", "Li"), ("отмывк", "Li"),
)


def normalize_code_text(text: str) -> str:
    """NFC + замена кириллических двойников латиницей (только для поиска кодов)."""
    return unicodedata.normalize("NFC", text).translate(_LOOKALIKES)


def modality_of(path: str | Path) -> str | None:
    """Код модальности (La, Li, Ae, SpOr, Or, MagGr, EtGr, TerFl) по пути файла."""
    p = Path(path)
    if p.suffix.lower() in (".las", ".laz"):
        return "La"
    for part in (p.name, p.parent.name):
        m = _CODE_RE.search(normalize_code_text(part))
        if m:
            return m.group(1)
    low = unicodedata.normalize("NFC", str(p)).lower()
    for key, code in _KEYWORDS:
        if key in low:
            return code
    return None


def code_is_cyrillic(path: str | Path) -> bool:
    """True, если код модальности в имени набран кириллическими двойниками («Ае»)."""
    raw = unicodedata.normalize("NFC", Path(path).name)
    m = _CODE_RE.search(raw.translate(_LOOKALIKES))
    if not m:
        return False
    return any("CYRILLIC" in unicodedata.name(ch, "") for ch in raw[m.start(1):m.end(1)])


def markup_type_of(path: str | Path) -> str | None:
    """Значение properties.markup_type для файла (li, ae, sp_or, or, maggr, etgr, terfl)."""
    code = modality_of(path)
    return MODALITY_TO_MARKUP.get(code) if code else None


def class_of_annotation(path: str | Path) -> str | None:
    """Канонический класс по имени файла разметки или None (игнорируемый/неизвестный).

    Примеры: «Юшково_Ае_курганы_поврежденные.geojson» → kurgany_povrezhdennye,
    «Гнёздово_Усадьба_EtGr_профиль_14_ObjectPoly.geojson» → object_poly.
    """
    stem = unicodedata.normalize("NFC", Path(path).stem)
    low = stem.lower()
    # Сначала самые длинные названия, чтобы «курганы_целые» не стало «курганы».
    for alias in sorted(CLASS_ALIASES, key=len, reverse=True):
        if low.endswith(alias.lower()):
            return CLASS_ALIASES[alias]
    return None


@dataclass(frozen=True)
class SiteName:
    """Имя участка: номер (если есть) и название."""
    raw: str
    number: int | None
    name: str


def parse_site(dirname: str) -> SiteName:
    """«001_Гнёздово_Нивленское» → (1, «Гнёздово_Нивленское»); «region_0042» → (None, ...)."""
    m = re.match(r"^(\d{3})_(.+)$", dirname)
    if m:
        return SiteName(dirname, int(m.group(1)), m.group(2))
    return SiteName(dirname, None, dirname)


def site_family(dirname: str) -> str:
    """Группа участков для честной проверки (например, «Гнёздово», «Семибратнее»).

    Берётся первое слово названия: участки одной «семьи» (Гнёздово_Терраса_1,
    Гнёздово_Пойма) нельзя разносить по обучению и проверке.
    """
    name = parse_site(dirname).name
    return re.split(r"[_\s]", name, maxsplit=1)[0]
