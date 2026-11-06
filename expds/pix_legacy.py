"""Необязательный загрузчик. В архив решения не входит.
"""
from __future__ import annotations

import importlib

MODEL_DIR = "models/pixgbm/all"


def load():
    return importlib.import_module(".geo_pixgbm", __package__)
