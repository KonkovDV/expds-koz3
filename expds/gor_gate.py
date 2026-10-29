"""bot-7 (кандидат v3): заготовка шлюза городищ DINOv2 bot-5 (ветка bot5-gor-patch-kaggle abb89d0). Выключен.

Идея bot-5 (вариант 'gate'): уверенность городищ SpOr (markup sp_or, без threshold_group) умножается на
max p(городище) по виньеткам 384 м (+ контекст 768 м) сцены; p — PCA-64 + логистическая регрессия
на признаках DINOv2 [CLS, mean(patch)] × 2 масштаба. Замер bot-5: +0,005057 к A (ниже MDE, только в пакете).

Артефактов для сдачи нет (на 29.10): головы (PCA mu/sd/W, коэффициенты LR), обученной на всех данных,
и порога для умноженной уверенности. Поэтому модуль только проверяет наличие файлов и при их отсутствии
пропускает объекты без изменений с одной строкой в журнале. Сама математика шлюза (gate_confidences)
готова и покрыта тестом; извлечение признаков DINOv2 не подключено, пока нет головы и порога.
Умноженные объекты получают threshold_group = ``sp_or_gated``: старый порог 0,994085 к ним неприменим.
"""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

GATED_GROUP = "sp_or_gated"
DEFAULTS = {
    "enabled": False,
    "backbone": "models/dinov2-small",       # snapshot facebook/dinov2-small ed25f3a31f01632728cabb09d1542f84ab7b0056
    "head": "models/gor_gate/head.npz",      # pca_mu, pca_sd, pca_w, lr_coef, lr_intercept
    "meta": "models/gor_gate/meta.json",     # размеры виньеток, растяжка, версия признаков
}


def settings(cfg: dict) -> dict:
    return dict(DEFAULTS, **(cfg.get("gor_gate") or {}))


def enabled(cfg: dict) -> bool:
    return bool((cfg.get("gor_gate") or {}).get("enabled", False))


def missing_artifacts(cfg: dict, root: Path) -> list[str]:
    st = settings(cfg)
    miss = []
    for key in ("backbone", "head", "meta"):
        p = Path(st[key])
        p = p if p.is_absolute() else Path(root) / p
        if not p.exists():
            miss.append(f"{key}={p}")
    if "gorodishcha" not in ((cfg.get("thresholds_by_markup") or {}).get(GATED_GROUP) or {}):
        miss.append(f"[thresholds_by_markup.{GATED_GROUP}] gorodishcha")
    return miss


def is_gated_target(f: dict) -> bool:
    return (f.get("markup_type") == "sp_or" and not f.get("threshold_group")
            and f.get("class_name") == "gorodishcha")


def gate_confidences(feats: list[dict], region_p: dict[str, float]) -> list[dict]:
    """conf городищ SpOr × p участка (по сценам/участку — max p виньеток); участок без p не трогается."""
    out = []
    for f in feats:
        reg = f.get("region_name")
        if is_gated_target(f) and reg in region_p:
            f = dict(f, confidence=float(f.get("confidence", 1.0)) * float(region_p[reg]),
                     threshold_group=GATED_GROUP)
        out.append(f)
    return out


def apply(feats: list[dict], regions: dict, cfg: dict, root: Path) -> list[dict]:
    """При enabled: без артефактов — объекты без изменений и строка в журнале."""
    if not enabled(cfg):
        return feats
    miss = missing_artifacts(cfg, root)
    if miss:
        log.warning("gor_gate: нет артефактов (%s) — шлюз городищ не применён", "; ".join(miss))
        return feats
    log.warning("gor_gate: артефакты есть, но извлечение признаков DINOv2 в кандидате v3 не подключено "
                "(ждём формат головы от bot-5) — шлюз не применён")
    return feats
