"""Пакетное построение геометрий метрики: то же, что официальная `_extract_geometry`, но быстрее.

Профиль `scripts/scoreboard.py --honest` (prof/cprofile_build_curves.txt): ≈ 90 %
времени build_curves уходит в `_extract_geometry` → конструктор `shapely.Polygon`
(≈ 40 мкс на объект: обход вложенных списков, проверки типов, LinearRing).

Здесь простые полигоны (тип "Polygon", кольца из пар чисел, ≥ 4 точек после
замыкания) разбираются Rust-функцией `expds_core.flatten_polygons` в плоские
массивы и создаются одним вызовом `shapely.from_ragged_array`; валидность
проверяется векторно. Невалидные, MultiPolygon, Point и всё необычное идут
через официальную `_extract_geometry` по одному — результат совпадает
побайтно (WKB), это проверяет тест быстрой метрики.

Без собранного Rust-модуля (или EXPDS_BACKEND=numpy / EXPDS_FASTGEOM=0) работает
прежний путь: официальная функция на каждый объект.
"""
from __future__ import annotations

import os

import numpy as np

_rs = None
if os.environ.get("EXPDS_BACKEND", "auto").lower() != "numpy":
    try:
        import expds_core as _rs  # type: ignore
        if not hasattr(_rs, "flatten_polygons"):
            _rs = None
    except ImportError:
        _rs = None


def enabled() -> bool:
    return _rs is not None and os.environ.get("EXPDS_FASTGEOM", "1") != "0"


def flatten(gdicts: list):
    """expds_core.flatten_polygons(gdicts) или None без Rust-модуля."""
    if not enabled() or not gdicts:
        return None
    return _rs.flatten_polygons(gdicts)


def extract_many(mod, features: list[dict], flat=None) -> list:
    """[официальная _extract_geometry(f) for f in features], пакетно.
    flat — уже посчитанный flatten([f.get("geometry") for f in features]) (чтобы не разбирать дважды)."""
    if not enabled() or not features:
        return [mod._extract_geometry(f) for f in features]
    import shapely

    if flat is None:
        flat = _rs.flatten_polygons([f.get("geometry") for f in features])
    coords, ring_off, geom_off, taken = flat
    out: list = [None] * len(features)
    done = np.zeros(len(features), dtype=bool)
    if taken.size:
        polys = shapely.from_ragged_array(shapely.GeometryType.POLYGON, coords, (ring_off, geom_off))
        valid = par_is_valid(polys)
        for i, g, ok in zip(taken.tolist(), polys, valid.tolist()):
            if ok:
                out[i] = g
                done[i] = True
    for i in np.flatnonzero(~done).tolist():
        out[i] = mod._extract_geometry(features[i])
    return out


def max_abs_x(geoms: list) -> np.ndarray:
    """Векторная `_geom_max_abs_x`: max(|minx|, |maxx|) с той же семантикой max() у NaN."""
    import shapely

    b = shapely.bounds(np.asarray(geoms, dtype=object))
    a, c = np.abs(b[:, 0]), np.abs(b[:, 2])
    return np.where(c > a, c, a)


# ───────────── параллельный пересчёт координат в UTM (тот же PROJ, по кускам) ─────────────

import threading as _threading

_TLS = _threading.local()
_POOL = None  # общий пул потоков метрики


def metric_threads() -> int:
    """Потоки метрики: EXPDS_METRIC_THREADS, иначе EXPDS_THREADS (как у expds_core), иначе min(CPU, 8)."""
    for var in ("EXPDS_METRIC_THREADS", "EXPDS_THREADS"):
        try:
            n = int(os.environ.get(var, "0"))
        except ValueError:
            n = 0
        if n > 0:
            return n
    return max(1, min(os.cpu_count() or 1, 8))


def _pool(n_thr: int):
    global _POOL
    if _POOL is None or _POOL._max_workers != n_thr:
        from concurrent.futures import ThreadPoolExecutor
        _POOL = ThreadPoolExecutor(max_workers=n_thr, thread_name_prefix="expds-metric")
    return _POOL


def par_is_valid(polys: np.ndarray, min_chunk: int = 20_000) -> np.ndarray:
    """shapely.is_valid кусками в потоках (GEOS отпускает GIL; каждый объект независим)."""
    import shapely

    n, n_thr = len(polys), metric_threads()
    if n_thr <= 1 or n < 2 * min_chunk:
        return shapely.is_valid(polys)
    bounds = np.linspace(0, n, min(n_thr, n // min_chunk) + 1).astype(int)
    parts = list(_pool(n_thr).map(lambda ab: shapely.is_valid(polys[ab[0]:ab[1]]), zip(bounds[:-1], bounds[1:])))
    return np.concatenate(parts)


def _local_transformer(src: int, dst: int):
    from pyproj import Transformer

    cache = getattr(_TLS, "tr", None)
    if cache is None:
        cache = _TLS.tr = {}
    tr = cache.get((src, dst))
    if tr is None:   # объекты PROJ не потокобезопасны — у каждого потока свой, с теми же параметрами
        tr = cache[(src, dst)] = Transformer.from_crs(src, dst, always_xy=True)
    return tr


def utm_coords_fn(base_tr, dst_epsg: int, src_epsg: int = 3857, min_chunk: int = 50_000):
    """Функция для shapely.transform: как `np.column_stack(tr.transform(x, y))`, но кусками в потоках.

    PROJ считает каждую точку независимо, поэтому результат побайтно тот же
    (проверяется тестом); pyproj отпускает GIL на время расчёта.
    """
    n_thr = metric_threads()

    def fn(c):
        n = c.shape[0]
        if n_thr <= 1 or n < 2 * min_chunk or not enabled():
            return np.column_stack(base_tr.transform(c[:, 0], c[:, 1]))
        bounds = np.linspace(0, n, min(n_thr, n // min_chunk) + 1).astype(int)
        out = np.empty((n, 2), dtype=np.float64)

        def work(a, b):
            x, y = _local_transformer(src_epsg, dst_epsg).transform(c[a:b, 0], c[a:b, 1])
            out[a:b, 0] = x
            out[a:b, 1] = y

        list(_pool(n_thr).map(lambda ab: work(*ab), zip(bounds[:-1], bounds[1:])))
        return out

    return fn
