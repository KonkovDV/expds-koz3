"""centerhead — декодирование точечной (CenterNet) головы курганов лидарной сети ConvNeXt-T.

Модули: config (константы и параметры декодирования), d8 (обратные преобразования D8 для
усреднения по поворотам и отражениям), targets (кодирование размера), decode (пики карты центров,
NMS и круги курганов).
"""
from .config import CLASSES5, HM_CLASSES, MASK_CLASSES, LI_TAGS12, DEFAULTS, cfg_with  # noqa: F401

__version__ = "0.1.0"
