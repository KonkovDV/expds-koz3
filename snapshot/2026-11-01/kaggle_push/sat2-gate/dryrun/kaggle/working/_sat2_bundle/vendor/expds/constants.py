"""Константы задачи: классы, веса метрики, типы данных и лимиты выходного файла.

Источник — проект ТР КОЗ №3 (Приложения 1 и 4) и официальный compute_metrics.py.
Если организаторы обновят метрику, правьте значения здесь и сверяйте с их файлом.
"""
from __future__ import annotations

#: Веса классов в итоговой метрике (проценты).
CLASS_WEIGHTS: dict[str, float] = {
    "kurgany_povrezhdennye": 30.0,
    "kurgany_tselye": 20.0,
    "gorodishcha": 15.0,
    "arkhitektury": 10.0,
    "object_poly": 10.0,
    "finds_points": 10.0,
    "fortifikatsii": 5.0,
}
CLASSES: tuple[str, ...] = tuple(CLASS_WEIGHTS)

#: Классы ДЗЗ (лидар, аэрофото, спутник, ортофото) и геофизики.
RS_CLASSES = ("kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha",
              "arkhitektury", "fortifikatsii")
GEO_CLASSES = ("object_poly", "finds_points")

#: Названия классов в именах файлов разметки → канонический класс.
#: None — класс, который метрика игнорирует (structure.poly и т. п.).
CLASS_ALIASES: dict[str, str | None] = {
    "курганы_поврежденные": "kurgany_povrezhdennye",
    "курганы_повреждённые": "kurgany_povrezhdennye",
    "курганы_разрушенные": "kurgany_povrezhdennye",
    "поврежденные": "kurgany_povrezhdennye",
    "курганы_целые": "kurgany_tselye",
    "курганы": "kurgany_tselye",
    "целые": "kurgany_tselye",
    "городища": "gorodishcha",
    "городище": "gorodishcha",
    "фортификации": "fortifikatsii",
    "фортификация": "fortifikatsii",
    "архитектуры": "arkhitektury",
    "архитектура": "arkhitektury",
    "findspoints": "finds_points",
    "findspoint": "finds_points",
    "objectpoly": "object_poly",
    "objectspoly": "object_poly",
    "structurepoly": None,
    "structpoly": None,
    "structrpoly": None,
}
for _c in CLASSES:
    CLASS_ALIASES[_c] = _c

#: Коды модальностей в именах файлов → значение properties.markup_type.
MODALITY_TO_MARKUP: dict[str, str] = {
    "La": "li",      # облака точек: результат по ним относится к типу li
    "Li": "li",
    "Ae": "ae",
    "SpOr": "sp_or",
    "Or": "or",
    "MagGr": "maggr",
    "EtGr": "etgr",
    "TerFl": "terfl",
}
MARKUP_TYPES = ("li", "ae", "sp_or", "or", "maggr", "etgr", "terfl")
GEO_MARKUP = ("maggr", "etgr", "terfl")

#: Ограничения выходного файла (официальный compute_metrics.py, koz3-1.5.0).
MAX_FEATURES = 20_000
MAX_VERTICES = 3_000
MAX_FILE_MB = 100.0
POINT_TOLERANCE_M = 2.0

#: Лимит времени на платформе и запас (секунды).
PLATFORM_LIMIT_S = 30 * 60
DEFAULT_BUDGET_S = 25 * 60


def optimal_threshold(f1: float) -> float:
    """Порог выдачи кандидата для достижимого F1 класса: p* = F / (4 − F).

    Следует из Precision = TP / (TP + 0.5·FP). Это стартовая точка подбора,
    окончательные пороги подбираются официальной метрикой (expds.metric).
    """
    f1 = min(max(f1, 0.0), 1.0)
    return f1 / (4.0 - f1)
