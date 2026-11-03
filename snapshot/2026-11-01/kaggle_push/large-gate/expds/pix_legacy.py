"""Загрузчик прежней (legacy) реализации PIXGBM bot-3 для [geophys.pixgbm] impl != "clean".

DOC-1 (W4): этот модуль и то, что он грузит, в архив решения НЕ входят (scripts/check_archive_clean.py);
в архиве impl="clean" (expds/pix_clean.py). Нужен только для исследований/сравнения в репозитории.
"""
from __future__ import annotations

import importlib

MODEL_DIR = "models/pixgbm/all"


def load():
    return importlib.import_module(".geo_pixgbm", __package__)
