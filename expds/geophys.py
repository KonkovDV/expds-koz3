"""Классические детекторы геофизики (MagGr, EtGr, TerFl) — без видеокарты.

finds_points: точечные аномалии магниторазведки — экстремумы высокочастотной
составляющей |z − фон| выше k·MAD. Выход — маленький круг (центроид ≤ 2 м от
эталонной точки засчитывается).

object_poly: площадные аномалии — связные области, где сглаженная
нормированная аномалия выше порога, с фильтром по площади.

Обучающих участков геофизики мало (три «семьи»), поэтому начинать стоит с
классики; нейросеть — только если выиграет на проверке по «семьям».
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import accel
from .raster import circle_3857, mask_to_polygons, pixel_to_xy

log = logging.getLogger(__name__)


@dataclass
class GeoParams:
    """Параметры классики; подбираются на трейне официальной метрикой."""
    background_m: float = 4.0       # радиус фона для высоких частот
    peak_radius_m: float = 1.0      # подавление соседних пиков
    peak_k: float = 4.0             # порог пика, в MAD
    point_circle_m: float = 0.3
    poly_k: float = 2.5             # порог области, в MAD
    poly_k_low: float | None = None  # гистерезис: граница области до этого порога, если внутри есть poly_k
    smooth_m: float = 0.5
    min_area_m2: float = 0.3
    max_area_m2: float = 60.0
    band_mode: str = "first"        # как RGB-карта сводится к числу (scalar_from_bands)
    signal: str = "abs"             # finds_points: пики |hp| или аналитического сигнала ("asa")
    polarity: str = "abs"           # object_poly: |hp|, только положительные ("pos") или отрицательные ("neg")


#: Как цветная или «серая» RGB-карта сводится к одному числу на пиксель. Сданный балл снят с "first".
BAND_MODES = ("first", "mean", "luma", "pca", "auto")


def scalar_from_bands(bands: np.ndarray, mode: str = "first") -> np.ndarray:
    """C×H×W (NaN — нет данных) → H×W.

    first — первый канал, как в сданном детекторе. mean и luma — среднее и яркость
    BT.601. pca — проекция на главную ось цвета: у двухцветной шкалы это и есть
    величина поля. Знак у детекторов не важен, они берут модуль высоких частот;
    ось всё равно ориентируется по яркости, чтобы результат был детерминирован.
    auto — pca, если каналы различаются хотя бы в 1 % пикселей, иначе first.
    """
    if mode not in BAND_MODES:
        raise ValueError(f"неизвестный режим каналов: {mode}")
    a = np.asarray(bands, dtype=np.float32)
    if a.ndim == 2:
        return a
    if a.shape[0] == 1 or mode == "first":
        return a[0]
    rgb = a[:3]
    valid = np.all(np.isfinite(rgb), axis=0)
    if mode == "auto":
        if not valid.any():
            return a[0]
        x = rgb[:, valid]
        differ = np.mean(np.any(x != x[:1], axis=0))
        if differ < 0.01:
            return a[0]
        mode = "pca"
    if mode == "mean":
        out = rgb.mean(axis=0)
    elif mode == "luma":
        w = np.array([0.299, 0.587, 0.114], np.float32)[: rgb.shape[0]]
        out = np.tensordot(w / w.sum(), rgb, axes=1)
    else:
        x = rgb[:, valid].astype(np.float64)
        if x.shape[1] < 2:
            return a[0]
        step = max(1, x.shape[1] // 2_000_000)
        sample = x[:, ::step]
        mean = sample.mean(axis=1, keepdims=True)
        cov = np.cov(sample - mean)
        vals, vecs = np.linalg.eigh(np.atleast_2d(cov))
        axis = vecs[:, int(np.argmax(vals))]
        luma = np.array([0.299, 0.587, 0.114])[: rgb.shape[0]]
        if float(axis @ luma) < 0:
            axis = -axis
        out = np.tensordot(axis.astype(np.float32), rgb - mean.astype(np.float32)[:, :, None], axes=1)
    return np.where(valid, out, np.nan).astype(np.float32)


def read_band(path: str | Path, mode: str = "first") -> tuple[np.ndarray, object, str | None, float]:
    """Одно число на пиксель (scalar_from_bands) как float32 с NaN вместо nodata.

    Альфа-канал, если он есть, только маскирует пиксели: прозрачное — нет данных.
    """
    from .raster import open_metric

    with open_metric(path) as ds:
        if mode == "first" or ds.count == 1:
            a = ds.read(1, masked=True).astype(np.float32).filled(np.nan)
        else:
            n = min(ds.count, 3)
            bands = ds.read(list(range(1, n + 1)), masked=True).astype(np.float32).filled(np.nan)
            alpha = [i + 1 for i, ci in enumerate(ds.colorinterp) if getattr(ci, "name", "") == "alpha"]
            if alpha:
                bands[:, ds.read(alpha[0]) == 0] = np.nan
            a = scalar_from_bands(bands, mode)
        crs = ds.crs.to_string() if ds.crs else None
        return a, ds.transform, crs, float(abs(ds.res[0]))


def robust_z(a: np.ndarray) -> np.ndarray:
    """(a − медиана) / (1,4826·MAD) по конечным значениям."""
    return accel.robust_z(a)


def strength_confidence(v: np.ndarray | float) -> np.ndarray | float:
    """Уверенность из силы аномалии (в MAD): монотонно и без насыщения, чтобы порог различал сильные пики."""
    return v / (v + 10.0)


def rank_within_source(feats: list[dict]) -> list[dict]:
    """Уверенность = доля объектов того же растра со силой не выше (0…1].

    Карты MagGr — визуализации с разным растяжением контраста: сила аномалии
    в MAD у эталонных точек отличается между местами в десятки раз, а её ранг
    среди пиков своей карты сопоставим.
    """
    groups: dict[tuple, list[int]] = {}
    for i, f in enumerate(feats):
        groups.setdefault((f["region_name"], f["source"]), []).append(i)
    out = [dict(f) for f in feats]
    for idx in groups.values():
        v = np.array([feats[i]["confidence"] for i in idx])
        ranks = np.searchsorted(np.sort(v), v, side="right") / len(v)
        for i, r in zip(idx, ranks):
            out[i]["confidence"] = float(r)
    return out


def highpass(z: np.ndarray, res: float, background_m: float) -> np.ndarray:
    """Высокочастотная составляющая: z минус среднее окна радиусом background_m."""
    return (z - accel.box_mean(z, max(1, round(background_m / res)))).astype(np.float32)


SIGNALS = ("abs", "asa")
POLARITIES = ("abs", "pos", "neg")


def analytic_signal(hp: np.ndarray, res: float, upward_m: float = 0.3) -> np.ndarray:
    """Амплитуда аналитического сигнала √(gx² + gy² + gz²) в единицах MAD самой амплитуды.

    Все производные — в спектре: gx = F⁻¹(i·kx·F), gy = F⁻¹(i·ky·F), gz = F⁻¹(|k|·F)
    (Nabighian, 1972; Roest et al., 1992). Перед ними поле продолжается вверх на
    upward_m: множитель e^(−|k|·h) — физический фильтр потенциального поля, без него
    производные раздувают шум пикселя. Максимум амплитуды стоит над компактным
    источником при любом наклонении. Пустые пиксели на время преобразования — ноль,
    на выходе снова NaN.
    """
    finite = np.isfinite(hp)
    f = np.where(finite, hp, 0.0).astype(np.float64)
    ny, nx = f.shape
    py, px = ny + ny // 4, nx + nx // 4            # запас от заворота краёв
    spec = np.fft.rfft2(f, s=(py, px))
    ky = 2 * np.pi * np.fft.fftfreq(py, d=res)[:, None]
    kx = 2 * np.pi * np.fft.rfftfreq(px, d=res)[None, :]
    k = np.sqrt(kx * kx + ky * ky)
    spec = spec * np.exp(-k * max(float(upward_m), 0.0))
    def back(m):
        return np.fft.irfft2(spec * m, s=(py, px))[:ny, :nx]

    amp = np.sqrt(back(1j * kx) ** 2 + back(1j * ky) ** 2 + back(k) ** 2)
    v = amp[finite]
    if v.size:
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med))) * 1.4826 or 1.0
        amp = (amp - med) / mad
    return np.where(finite, amp, np.nan).astype(np.float32)


def detection_image(hp: np.ndarray, res: float, signal: str = "abs") -> np.ndarray:
    """Неотрицательная карта для поиска пиков находок: |hp| (сдача) или аналитический сигнал."""
    if signal not in SIGNALS:
        raise ValueError(f"неизвестный сигнал: {signal}")
    if signal == "asa":
        return analytic_signal(hp, res)
    return np.where(np.isfinite(hp), np.abs(hp), np.nan).astype(np.float32)


def polarity_image(hp: np.ndarray, polarity: str = "abs") -> np.ndarray:
    """|hp| (сдача), только положительная или только отрицательная часть аномалии."""
    if polarity not in POLARITIES:
        raise ValueError(f"неизвестная полярность: {polarity}")
    if polarity == "pos":
        return np.where(np.isfinite(hp), np.maximum(hp, 0), np.nan).astype(np.float32)
    if polarity == "neg":
        return np.where(np.isfinite(hp), np.maximum(-hp, 0), np.nan).astype(np.float32)
    return np.abs(hp).astype(np.float32)


def peaks_from(hp: np.ndarray, tr, crs: str | None, res: float, region: str, markup_type: str, source: str,
               p: GeoParams, k_min: float | None = None) -> list[dict]:
    """Пики карты обнаружения (detection_image) не ниже k_min (по умолчанию peak_k) → круги finds_points."""
    mag = detection_image(hp, res, p.signal)
    rows, cols, vals = accel.find_peaks(mag, max(1, round(p.peak_radius_m / res)),
                                        p.peak_k if k_min is None else k_min)
    xs, ys = pixel_to_xy(tr, rows, cols)
    return [{"geom": circle_3857(float(x), float(y), p.point_circle_m, crs), "class_name": "finds_points",
             "confidence": float(strength_confidence(v)), "region_name": region, "markup_type": markup_type,
             "source": source, "original_crs": crs} for x, y, v in zip(xs, ys, vals)]


def polys_from(hp: np.ndarray, tr, crs: str | None, res: float, region: str, markup_type: str, source: str,
               p: GeoParams) -> list[dict]:
    """Связные области сглаженной |hp| выше poly_k с допустимой площадью → полигоны object_poly."""
    from scipy.ndimage import label as cc_label

    sm = accel.box_mean(polarity_image(hp, p.polarity), max(1, round(p.smooth_m / res)))
    finite = np.isfinite(sm)
    if p.poly_k_low is not None and p.poly_k_low < p.poly_k:
        lab, n = cc_label(finite & (sm >= p.poly_k_low))
        if n == 0:
            return []
        strong = np.zeros(n + 1, bool)
        strong[np.unique(lab[finite & (sm >= p.poly_k)])] = True
        strong[0] = False
        lab, n = cc_label(strong[lab])
    else:
        lab, n = cc_label(finite & (sm >= p.poly_k))
    if n == 0:
        return []
    sizes = np.bincount(lab.ravel()) * res * res
    keep = (sizes >= p.min_area_m2) & (sizes <= p.max_area_m2)
    keep[0] = False
    lab = np.where(keep[lab], lab, 0)
    peak = accel.label_max(lab, np.nan_to_num(sm, nan=0), n)
    return [{"geom": poly, "class_name": "object_poly", "confidence": float(strength_confidence(peak[k])),
             "region_name": region, "markup_type": markup_type, "source": source, "original_crs": crs}
            for poly, k in mask_to_polygons(lab, tr, crs)]


def finds_points(path: str | Path, region: str, markup_type: str, p: GeoParams | None = None,
                 k_min: float | None = None) -> list[dict]:
    """Точечные аномалии → круги класса finds_points."""
    p = p or GeoParams()
    a, tr, crs, res = read_band(path, p.band_mode)
    return peaks_from(highpass(robust_z(a), res, p.background_m), tr, crs, res, region, markup_type,
                      Path(path).name, p, k_min)


def object_polys(path: str | Path, region: str, markup_type: str, p: GeoParams | None = None) -> list[dict]:
    """Площадные аномалии → полигоны класса object_poly."""
    p = p or GeoParams()
    a, tr, crs, res = read_band(path, p.band_mode)
    return polys_from(highpass(robust_z(a), res, p.background_m), tr, crs, res, region, markup_type,
                      Path(path).name, p)


#: Пределы площади object_poly по описанию классов: MagGr — диаметр от 5 м, EtGr — 0,3–12 м, TerFl — 0,3–30 м.
AREA_M2 = {"maggr": (5.0, 3000.0), "etgr": (0.05, 150.0), "terfl": (0.05, 1000.0)}
#: Аномалии EtGr глубже 2 м — геология, а не археология (описание классов).
ETGR_MAX_DEPTH_M = 2.0


def select_rasters(files: list[Path], markup_type: str) -> list[Path]:
    """Растры для детекторов: без цветных копий серых карт и без срезов EtGr глубже 2 м."""
    import re

    gray = [f for f in files if "цвет" not in f.name.lower()]
    files = gray or files
    if markup_type != "etgr":
        return files
    out = []
    for f in files:
        m = re.search(r"карта_\d+_(\d+(?:[.,]\d+)?)", f.stem)
        if m is None or float(m.group(1).replace(",", ".")) <= ETGR_MAX_DEPTH_M:
            out.append(f)
    return out


def load_mask(files: list[Path], markup_type: str, fallback_crs: str | None):
    """Объединение полигонов масок (extra/*_границы, *_шум) нужной модальности в EPSG:3857; None — нет масок."""
    import json

    from shapely.geometry import shape
    from shapely.ops import unary_union
    from shapely.validation import make_valid

    from .crs import WEB_MERCATOR, guess_geojson_crs, to_crs
    from .naming import markup_type_of

    parts = []
    for f in files:
        if markup_type_of(f) != markup_type:
            continue
        try:
            d = json.loads(Path(f).read_text(encoding="utf-8-sig"))
            src = guess_geojson_crs(d, fallback_crs)
            if src is None:
                continue
            gs = [make_valid(shape(x["geometry"])) for x in d.get("features", []) if x.get("geometry")]
            if gs:
                parts.append(to_crs(unary_union(gs), src, WEB_MERCATOR))
        except Exception as exc:
            log.warning("маска %s не прочитана: %s", Path(f).name, exc)
    return unary_union(parts).buffer(0) if parts else None


def params_for(cfg: dict, markup_type: str, class_name: str) -> tuple[GeoParams, dict]:
    """Параметры детектора из [geophys] и [geophys.<модальность>.<класс>] конфига.

    Возвращает GeoParams и опции после детектора: use_noise (маска шума),
    min_confidence (порог этой модальности; [thresholds] задаёт порог только по классу),
    rank (confidence = "rank": уверенность — ранг силы среди объектов своего растра).
    """
    base = {k: v for k, v in (cfg or {}).items() if not isinstance(v, dict)}
    over = ((cfg or {}).get(markup_type) or {}).get(class_name) or {}
    merged = {**base, **over}
    p = GeoParams(**{k: v for k, v in merged.items() if k in GeoParams.__dataclass_fields__})
    lo, hi = AREA_M2.get(markup_type, (p.min_area_m2, p.max_area_m2))
    p.min_area_m2, p.max_area_m2 = over.get("min_area_m2", lo), over.get("max_area_m2", hi)
    return p, {"use_noise": bool(merged.get("use_noise", False)),
               "min_confidence": float(merged.get("min_confidence", 0.0)),
               "rank": merged.get("confidence", "strength") == "rank"}


def source_crs(path: str | Path) -> str | None:
    """CRS растра строкой или None (нет привязки или файл не читается)."""
    import rasterio

    try:
        with rasterio.open(path) as ds:
            return ds.crs.to_string() if ds.crs else None
    except Exception:
        return None


def layer_copies(feats: list[dict], sources: dict[str, str | None], decay: float = 1.0,
                 group: str = "etgr_copy") -> list[dict]:
    """Копии объектов в остальные слои того же участка (срезы глубины и профили EtGr).

    Каждый слой EtGr — отдельный эталон (письмо организаторов 20.10), а яма, очаг или
    фундамент видны на соседних глубинах. Объект, найденный в одном слое, выдаётся и в
    остальных: копия засчитывается против эталона своего слоя. Копия не ставится в слой,
    где уже есть объект того же класса, накрывающий её внутреннюю точку; из нескольких
    копий в одно место остаётся самая уверенная. Копируются только слои с той же CRS:
    профиль без привязки в плане с картой не совмещается.
    Уверенность копии — исходная × decay, группа порога — group (своя, подбирается табло).
    sources — {имя файла слоя: CRS}; слой без CRS не получает и не отдаёт копий.
    """
    from shapely.strtree import STRtree

    if not feats or len(sources) < 2:
        return []
    by_layer: dict[tuple, list[dict]] = {}
    for f in feats:
        by_layer.setdefault((f.get("region_name"), f.get("source"), f["class_name"]), []).append(f)
    trees = {k: (STRtree([f["geom"] for f in v]), [f["geom"] for f in v]) for k, v in by_layer.items()}
    best: dict[tuple, dict] = {}
    for f in sorted(feats, key=lambda f: -float(f.get("confidence", 0.0))):
        region, src, cls = f.get("region_name"), f.get("source"), f["class_name"]
        pt = f["geom"].representative_point()
        src_crs = sources.get(src)
        for dst, dst_crs in sources.items():
            if dst == src or src_crs is None or dst_crs != src_crs:
                continue
            hit = trees.get((region, dst, cls))
            if hit is not None and any(hit[1][int(i)].contains(pt) for i in hit[0].query(pt)):
                continue
            key = (region, dst, cls, round(pt.x, 1), round(pt.y, 1))
            if key in best:
                continue
            best[key] = dict(f, source=dst, confidence=float(f.get("confidence", 1.0)) * decay,
                             threshold_group=group, copied_from=src)
    return list(best.values())


def apply_masks(feats: list[dict], boundary=None, noise=None) -> list[dict]:
    """Оставляет объекты внутри границ съёмки и вне шума (по внутренней точке)."""
    from shapely.prepared import prep

    b = prep(boundary) if boundary is not None else None
    n = prep(noise) if noise is not None else None
    out = []
    for f in feats:
        pt = f["geom"].representative_point()
        if (b is None or b.contains(pt)) and (n is None or not n.contains(pt)):
            out.append(f)
    return out
