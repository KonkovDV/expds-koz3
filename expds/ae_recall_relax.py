"""Relax ae / li_ae / fusion thresholds for aerial recall (bot-3).

Root cause on LB ae R~0.013 (eb268d6): ae network is fort-only at tau~0.998;
li_to_ae copies cut by li_ae ~0.993/1.000001 and fusion.povr_min=0.99;
no-LAS ae sites have nothing to copy — need [dfine_aeor] for kurgans.
Section [ae_recall_relax] enabled=false by default; enabled=true remaps
thresholds_by_markup and fusion before apply_thresholds / li_to_ae.
D-FINE Ae ([dfine_aeor]) is a separate flag.
"""
from __future__ import annotations

import copy
import logging
from typing import Any

log = logging.getLogger(__name__)

#: Fix B defaults (ae_ortho_diagnosis.md): ae_fort ~0.995; optional li_ae/fusion.
DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "thresholds_by_markup": {
        "ae": {
            "fortifikatsii": 0.995,
            "kurgany_povrezhdennye": 0.99053,
            "kurgany_tselye": 0.802255,
        },
        "li_ae": {
            "kurgany_povrezhdennye": 0.95,
            "kurgany_tselye": 0.85,
        },
    },
    "fusion": {
        "povr_min": 0.9,
        "tselye_min": 0.7,
    },
}


def settings(cfg: dict | None) -> dict:
    """Сливает [ae_recall_relax] с DEFAULTS (ключи TOML главнее)."""
    out = copy.deepcopy(DEFAULTS)
    block = (cfg or {}).get("ae_recall_relax") or {}
    if not isinstance(block, dict):
        return out
    for k, v in block.items():
        if k in ("thresholds_by_markup", "fusion") and isinstance(v, dict):
            base = out.setdefault(k, {})
            for gk, gv in v.items():
                if isinstance(gv, dict) and isinstance(base.get(gk), dict):
                    base[gk] = {**base[gk], **gv}
                else:
                    base[gk] = gv
        else:
            out[k] = v
    return out


def enabled(cfg: dict | None) -> bool:
    return bool(settings(cfg).get("enabled"))


def apply(cfg: dict) -> dict:
    """Возвращает cfg; при enabled=true глубоко обновляет thresholds_by_markup и fusion.

    Не мутирует входной словарь (копия верхнего уровня + вложенных таблиц порогов).
    """
    st = settings(cfg)
    if not st.get("enabled"):
        return cfg
    out = dict(cfg)
    thr = copy.deepcopy(out.get("thresholds_by_markup") or {})
    for group, classes in (st.get("thresholds_by_markup") or {}).items():
        if not isinstance(classes, dict):
            continue
        slot = dict(thr.get(group) or {})
        slot.update({k: float(v) for k, v in classes.items()})
        thr[group] = slot
    out["thresholds_by_markup"] = thr
    fus = dict(out.get("fusion") or {})
    for k, v in (st.get("fusion") or {}).items():
        fus[k] = v
    out["fusion"] = fus
    log.info(
        "ae_recall_relax: ON — ae fort->%.4f li_ae povr->%.4f / tsel->%.4f fusion.povr_min->%.3f",
        thr.get("ae", {}).get("fortifikatsii", float("nan")),
        thr.get("li_ae", {}).get("kurgany_povrezhdennye", float("nan")),
        thr.get("li_ae", {}).get("kurgany_tselye", float("nan")),
        float(fus.get("povr_min", float("nan"))),
    )
    return out
