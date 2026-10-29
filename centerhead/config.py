"""Константы и дефолты. Все числа ниже — нетюнингованные стартовые значения, кроме помеченных «из bot-9»
(взяты из bot-9/kaggle/convnext_lidar/src/convnext_main.py CFG) и «из метрики» (compute_metrics.py)."""
from __future__ import annotations

# Порядок классов выхода сети bot-9 (convnext_main.py CLASSES): out[:, :2] — теплокарты, out[:, 2:] — маски.
CLASSES5 = ("kurgany_tselye", "kurgany_povrezhdennye", "gorodishcha", "fortifikatsii", "arkhitektury")
HM_CLASSES = CLASSES5[:2]       # канал 0 — целые, 1 — повреждённые
MASK_CLASSES = CLASSES5[2:]     # отдельная seg-голова (BCE+Dice у bot-9)

# Порядок 12 каналов рельефа (bot-2/RELIEF.md, li_tags); первые 5 = TAGS5 bot-9.
LI_TAGS12 = ("lrm_s", "lrm_l", "slope", "svf_r", "opp_r", "opn_r", "tpi5_r", "tri3_r", "curv_r", "slrm8_r",
             "dens_r", "int_r")

DEFAULTS: dict = {
    # геометрия растра
    "res": 0.5,                 # м/пикс (из bot-9)
    "crop": 512,                # (из bot-9)
    # цели
    "sigma_mode": "centernet",  # "centernet" (gaussian_radius, min_overlap) | "linear" (sigma_k*D, как у bot-9)
    "min_overlap": 0.7,         # CenterNet
    "sigma_k": 0.15,            # (из bot-9) для sigma_mode="linear"
    "sigma_min_px": 1.0,        # (из bot-9)
    "default_diam_m": 8.0,      # (из bot-9) для GT-точек без размера
    "size_mode": "log_r",       # цель размера: log(радиус в пикс) | "r" (радиус в пикс)
    # лосс
    "focal_alpha": 2.0, "focal_beta": 4.0,
    "w_hm": 1.0, "w_wh": 0.1, "w_off": 1.0,     # веса CenterNet (wh 0.1, off 1)
    # голова
    "head_ch": 64, "hm_bias": -2.19,
    "stride": 4,                # шаг выхода головы относительно входа (предложение bot-1: stride-4 декодера, 128×128 при 512)
    # декодирование
    "peak_pool": 3,             # окно локального максимума, пикс
    "min_score": 0.05,          # (из bot-9)
    "topk": 20000,              # на класс на участок
    "nms_min_m": 1.0,           # подавление, если d < max(nms_min_m, nms_frac*min(r_i, r_j))
    "nms_frac": 0.8,
    "nms_cross_class": False,   # False: пики tselye и povr в одной точке сохраняются оба (задача 7: Δ −0,0067 при удалении)
    # выход
    "circle_r_m": 0.5,          # радиус круга-выхода, м (обоснование: work/centerhead/radius_sim.json)
    "circle_points": 32,        # как CIRCLE_POINTS метрики
    "markup_type": "li",
    # признаки реранкера
    "nb_radii_m": (10.0, 25.0, 50.0, 200.0),
    "feat_win_px": 10,          # полуокно патча признаков (21×21)
    "gor_thr": 0.5, "fort_thr": 0.5,
}


def cfg_with(**over) -> dict:
    c = dict(DEFAULTS)
    unknown = set(over) - set(c)
    if unknown:
        raise KeyError(f"неизвестные ключи конфига: {sorted(unknown)}")
    c.update(over)
    return c
