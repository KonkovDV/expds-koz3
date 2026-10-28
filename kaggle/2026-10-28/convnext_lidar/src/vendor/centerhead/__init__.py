"""centerhead — точечная (CenterNet) голова курганов для лидарной U-Net (bot-9, ConvNeXt-T).

Модули без torch: config, d8, targets, geo, decode, features, rerank, export.
Модули с torch (импорт лениво): losses, head.
См. INTERFACE.md.
"""
from .config import CLASSES5, HM_CLASSES, MASK_CLASSES, LI_TAGS12, DEFAULTS, cfg_with  # noqa: F401

__version__ = "0.1.0"
