"""Оценка официальной метрикой и точный подбор порогов по классам.

Официальный compute_metrics.py в архив не кладётся: путь к вашей копии с
платформы передаётся явно (--metric) или через EXPDS_METRIC. Свою метрику
не пишем: геометрические проверки берутся из официального кода.

Почему подбор точный и быстрый. Метрика сопоставляет предсказания жадно, по
убыванию confidence, и каждый класс считает отдельно. Отбросить предсказания
ниже порога τ — значит отрезать «хвост» отсортированного списка; решения для
более уверенных предсказаний от хвоста не зависят. Поэтому достаточно ОДНОГО
прохода сопоставления на всех кандидатах: запоминаем, какие попали (hit), и для
любого τ получаем TP = hit с conf ≥ τ, FP = (всего с conf ≥ τ) − TP, FN = GT − TP.
Перебираются все различимые пороги, а не сетка; совпадение с полным официальным
расчётом проверяет verify() (и тест test_metric_exact.py).

    python -m expds.metric score --pred out.geojson --gt work/gt_val.geojson --metric materials/compute_metrics.py
    python -m expds.metric sweep --pred cand.geojson --gt work/gt_train.geojson --metric ... \
        --markup li --regions holdout.txt --bootstrap 300 --out configs/thresholds.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import fastgeom as _fastgeom
from . import geomcache as _geomcache
from .constants import CLASS_WEIGHTS, CLASSES

EPS = 1e-6


@lru_cache(maxsize=2)
def load_official(path: str | None = None):
    """Импортирует официальный compute_metrics.py как модуль (рядом может лежать его config.py)."""
    path = path or os.environ.get("EXPDS_METRIC")
    if not path or not Path(path).exists():
        raise FileNotFoundError("укажите путь к официальному compute_metrics.py (--metric или EXPDS_METRIC)")
    parent = str(Path(path).resolve().parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location("koz3_compute_metrics", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def score(pred: dict, gt: dict, metric_path: str | None = None) -> dict:
    """Полный результат официальной метрики (final_score, per_class, auxiliary)."""
    mod = load_official(metric_path)
    return mod.F1Calculator().calculate_f1(pred, gt)


def score_by_site(pred: dict, gt: dict, metric_path: str | None = None) -> dict:
    """Эталон для режима scope="site": официальный calculate_f1 отдельно на каждом участке
    (разбиение — официальной split_merged_by_source_and_class), TP/FP/FN классов складываются,
    F1 и взвешивание — формулами calculate_f1 (те же epsilon и нормировка на классы с эталоном)."""
    mod = load_official(metric_path)
    calc = mod.F1Calculator()
    ps, gs = mod.split_merged_by_source_and_class(pred), mod.split_merged_by_source_and_class(gt)
    tot = {c: [0, 0, 0] for c in calc.target_classes}
    for site in list(ps) + [k for k in gs if k not in ps]:
        pf = [f for fc in ps.get(site, {}).values() for f in fc["features"]]
        gf = [f for fc in gs.get(site, {}).values() for f in fc["features"]]
        r = calc.calculate_f1({"type": "FeatureCollection", "features": pf},
                              {"type": "FeatureCollection", "features": gf})["per_class"]
        for c in calc.target_classes:
            for k, key in enumerate(("tp", "fp", "fn")):
                tot[c][k] += r[c][key]
    per, num, den = {}, 0.0, 0.0
    for c, (tp, fp, fn) in tot.items():
        precision = tp / (tp + 0.5 * fp + calc.epsilon)
        recall = tp / (tp + fn + calc.epsilon)
        f1 = 2 * precision * recall / (precision + recall + calc.epsilon)
        per[c] = {"f1": f1, "tp": tp, "fp": fp, "fn": fn, "num_ground_truth": tp + fn}
        if tp + fn > 0:
            num += f1 * mod.CLASS_WEIGHTS.get(c, 0.0)
            den += mod.CLASS_WEIGHTS.get(c, 0.0)
    return {"final_score": num / den if den > 0 else 0.0, "per_class": per}


# ─────────────────────────── один проход сопоставления ───────────────────────

@dataclass
class ClassCurve:
    """Итог одного прохода для класса: уверенности по убыванию и попадания."""
    conf: np.ndarray
    hit: np.ndarray
    pred_region: list[str]
    n_gt: int
    gt_regions: Counter = field(default_factory=Counter)
    pred_group: list[str] = field(default_factory=list)


def threshold_group(props: dict) -> str:
    """Группа порога объекта, как в post.apply_thresholds: threshold_group, иначе markup_type."""
    return str(props.get("threshold_group") or props.get("markup_type") or "")


_UTM_GEOM_CACHE: dict[tuple, dict[int, tuple[object, object]]] = {}
_SHM_CACHE_DIR = _geomcache.default_dir()   # каталог файлового кэша по умолчанию; EXPDS_GEOM_CACHE важнее


def _todo_batch_hash(todo_feats: list, todo_meta: list, mode: str, is_gt: bool, r_tol: float, c_pts: int) -> str:
    import hashlib

    h = hashlib.sha1()
    h.update(f"{mode}:{is_gt}:{r_tol}:{c_pts}:{len(todo_feats)}:".encode())
    confs = np.array([m[1] for m in todo_meta], dtype=np.float64)
    h.update(confs.tobytes())
    step = max(1, len(todo_feats) // 512)
    for i in range(0, len(todo_feats), step):
        _, gdict, _ = todo_feats[i]
        cls, _, reg, grp, _ = todo_meta[i]
        h.update(f"{cls}:{reg}:{grp}:{gdict.get('type')}:{gdict.get('coordinates')!r};".encode())
    return h.hexdigest()


def site_key(props: dict):
    """Ключ участка, как в официальной split_merged_by_source_and_class:
    region_name (по умолчанию "default_source"), а при непустом sub_region_name — "region__sub"."""
    region_name = props.get("region_name", "default_source")
    sub_region_name = props.get("sub_region_name", "")
    if sub_region_name and sub_region_name.strip():
        return f"{region_name}__{sub_region_name}"
    return region_name


# Синонимы markup_type → канонический слой (как в materials/compute_metrics.DATA_TYPE_GROUPS,
# но maggr/etgr/terfl/geogr не склеиваем: правила КОЗ — отдельные слои/карты).
_LAYER_ALIASES = {
    "la": "li",
    "li": "li",
    "ae": "ae",
    "german_ae": "ae",
    "sp_or": "sp_or",
    "spor": "sp_or",
    "or": "or",
    "maggr": "maggr",
    "etgr": "etgr",
    "geogr": "geogr",
    "terfl": "terfl",
}


def layer_key(props: dict) -> str:
    """Ключ слоя данных: нормализованный markup_type (la→li, spor→sp_or, german_ae→ae).

    Пустой/отсутствующий markup_type → "?": такие объекты матчятся только между собой.
    Неизвестный ключ оставляем как есть (lowercase), чтобы он не сливался с известными слоями.
    """
    raw = props.get("markup_type")
    if raw is None:
        return "?"
    key = str(raw).strip().lower()
    if not key:
        return "?"
    return _LAYER_ALIASES.get(key, key)


def match_scope(scope: str | None = None) -> str:
    """Область сопоставления: "global" — один пул на весь файл, как F1Calculator.calculate_f1
    (по умолчанию); "site" — эталон только своего участка (site_key); "layer" — эталон только
    того же слоя данных (layer_key / markup_type). site и layer — диагностика; официальная
    метрика — global. Переключатель: аргумент scope или переменная EXPDS_MATCH_SCOPE."""
    s = (scope or os.environ.get("EXPDS_MATCH_SCOPE") or "global").strip().lower()
    if s not in ("global", "site", "layer"):
        raise ValueError(f"EXPDS_MATCH_SCOPE/scope: ожидается global, site или layer, получено {s!r}")
    if s != "global" and not _SCOPE_WARNED:
        # non-global — только диагностика; официальная метрика сопоставляет глобально. Не молча.
        _SCOPE_WARNED.append(1)
        print(f"expds.metric: ВНИМАНИЕ — scope={s} (диагностика), не как официальная метрика (global)",
              file=sys.stderr, flush=True)
    return s


_SCOPE_WARNED: list = []


def _collect(mod, calc, fc: dict, is_gt: bool, sites_out: dict | None = None,
             key_fn=site_key) -> dict[str, list[tuple]]:
    """Повторяет F1Calculator._collect_by_class, сохраняя регион объекта (с кэшем в RAM + /dev/shm и векторизацией UTM).

    sites_out + key_fn: параллельный список ключей области (site_key или layer_key) по классам.
    """
    import gc

    gc_was = gc.isenabled()
    gc.disable()   # циклов здесь не создаётся; сборщик на миллионах списков GeoJSON съедал ~30 % времени
    try:
        return _collect_impl(mod, calc, fc, is_gt, sites_out, key_fn=key_fn)
    finally:
        if gc_was:
            gc.enable()


def _collect_impl(mod, calc, fc: dict, is_gt: bool, sites_out: dict | None = None,
                  key_fn=site_key) -> dict[str, list[tuple]]:
    import shapely

    out: dict[str, list[tuple]] = {}
    mode, r_tol, c_pts = calc.input_crs_mode, calc.r_tolerance, calc.circle_points
    sub_cache = _UTM_GEOM_CACHE.setdefault((mode, is_gt, r_tol, c_pts), {})
    if len(sub_cache) > 1_200_000:
        sub_cache.clear()
    target_set = set(calc.target_classes)
    todo_feats: list[tuple[dict, object, int]] = []
    todo_meta: list[tuple[str, float, str, str, int]] = []
    ordered_slots: list[tuple[str, object, float, str, str]] = []

    norm_cache: dict = {}
    slot_sites: list = []
    get_conf = mod._get_confidence
    sub_get = sub_cache.get
    for feature in fc.get("features", []):
        props = feature.get("properties", {}) or {}
        raw_cls = props.get("class_name", "unknown")
        if type(raw_cls) is str:   # кэш только для str: у 1/True/1.0 общий хэш, но разный str()
            cls = norm_cache.get(raw_cls, norm_cache)
            if cls is norm_cache:
                cls = norm_cache[raw_cls] = mod.normalize_class_name(raw_cls)
        else:
            cls = mod.normalize_class_name(raw_cls)
        if cls is None or cls not in target_set:
            continue
        gdict = feature.get("geometry")
        if not gdict:
            continue
        conf = props.get("confidence", 1.0)
        if type(conf) is not float:      # для float официальная _get_confidence вернула бы его же
            conf = get_conf(feature)
        reg = props.get("region_name", "default_source")
        if type(reg) is not str:
            reg = str(reg)
        grp = props.get("threshold_group") or props.get("markup_type") or ""   # = threshold_group(props)
        if type(grp) is not str:
            grp = str(grp)
        if sites_out is not None:
            slot_sites.append(key_fn(props))
        gid = id(gdict)
        hit = sub_get(gid)
        if hit is not None and hit[0] is gdict:
            ordered_slots.append((cls, hit[1], conf, reg, grp))
        else:
            slot_idx = len(ordered_slots)
            ordered_slots.append((cls, None, conf, reg, grp))
            todo_feats.append((feature, gdict, gid))
            todo_meta.append((cls, conf, reg, grp, slot_idx))

    if todo_feats:
        shm_loaded = False
        cache_dir = _geomcache.resolve_dir(_SHM_CACHE_DIR) if len(todo_feats) >= _geomcache.min_batch() else None
        flat = None
        bkey = None
        if cache_dir is not None or _fastgeom.enabled():
            flat = _fastgeom.flatten([t[1] for t in todo_feats])
        if cache_dir is not None:
            try:
                bkey = _geomcache.batch_key(mod, mode, is_gt, r_tol, c_pts, [t[1] for t in todo_feats], flat)
                got = _geomcache.load(cache_dir, bkey, len(todo_feats))
                if got is not None:
                    valid_idx, geoms = got
                    for g, idx in zip(geoms, valid_idx):
                        _, gdict, gid = todo_feats[idx]
                        sub_cache[gid] = (gdict, g)
                        cls, conf, reg, grp, slot_idx = todo_meta[idx]
                        ordered_slots[slot_idx] = (cls, g, conf, reg, grp)
                    shm_loaded = True
            except Exception:
                shm_loaded = False
                bkey = None

        if not shm_loaded:
            raw_geoms = []
            valid_idx = []
            extracted = _fastgeom.extract_many(mod, [t[0] for t in todo_feats], flat)
            for idx, g in enumerate(extracted):
                if g is not None:
                    raw_geoms.append(g)
                    valid_idx.append(idx)
            if raw_geoms:
                if mode == "utm":
                    proj_geoms = raw_geoms
                else:
                    proj_geoms = list(raw_geoms)
                    if mode == "webmercator":
                        need_idx = list(range(len(raw_geoms)))
                    elif _fastgeom.enabled():
                        need_idx = np.flatnonzero(
                            _fastgeom.max_abs_x(raw_geoms) > mod._UTM_ABS_X_THRESHOLD).tolist()
                    else:
                        need_idx = [k_i for k_i, g in enumerate(raw_geoms)
                                    if mod._geom_max_abs_x(g) > mod._UTM_ABS_X_THRESHOLD]
                    if need_idx:
                        try:
                            sub_raw = [raw_geoms[k_i] for k_i in need_idx]
                            cents = shapely.centroid(sub_raw)
                            xs = shapely.get_x(cents)
                            ys = shapely.get_y(cents)
                            lons, lats = mod._get_transformer_3857_to_wgs84().transform(xs, ys)
                            zones = np.clip(((np.asarray(lons) + 180.0) // 6).astype(np.int32) + 1, 1, 60)
                            epsgs = np.where(np.asarray(lats) >= 0, 32600, 32700) + zones
                            for epsg in np.unique(epsgs):
                                z_mask = np.nonzero(epsgs == epsg)[0]
                                tr = mod._get_transformer_3857_to_utm(int(epsg))
                                z_geoms = [sub_raw[z_i] for z_i in z_mask]
                                z_proj = shapely.transform(z_geoms, _fastgeom.utm_coords_fn(tr, int(epsg)))
                                for z_i, pg in zip(z_mask, z_proj):
                                    proj_geoms[need_idx[z_i]] = pg
                        except Exception:
                            for k_i in need_idx:
                                proj_geoms[k_i] = mod._to_utm(raw_geoms[k_i], mode)
                final_geoms = []
                is_point = (shapely.get_type_id(np.asarray(proj_geoms, dtype=object)) == 0).tolist()
                for g, idx, pt in zip(proj_geoms, valid_idx, is_point):
                    if pt and not is_gt:
                        g = mod._point_to_circle(g, r_tol, c_pts)
                    final_geoms.append(g)
                    _, gdict, gid = todo_feats[idx]
                    sub_cache[gid] = (gdict, g)
                    cls, conf, reg, grp, slot_idx = todo_meta[idx]
                    ordered_slots[slot_idx] = (cls, g, conf, reg, grp)
                if bkey is not None and final_geoms:
                    _geomcache.save(cache_dir, bkey, len(todo_feats), valid_idx, final_geoms)

    if sites_out is not None:
        for (cls, g, conf, reg, grp), site in zip(ordered_slots, slot_sites):
            if g is not None:
                out.setdefault(cls, []).append((g, conf, reg, grp))
                sites_out.setdefault(cls, []).append(site)
        return out
    for cls, g, conf, reg, grp in ordered_slots:
        if g is not None:
            out.setdefault(cls, []).append((g, conf, reg, grp))
    return out


def _collect_reference(mod, calc, fc: dict, is_gt: bool, sites_out: dict | None = None,
                        key_fn=site_key) -> dict[str, list[tuple]]:
    """Прежняя реализация _collect (эталон для тестов bot-8). Повторяет F1Calculator._collect_by_class, сохраняя регион объекта (с кэшем в RAM + /dev/shm и векторизацией UTM)."""
    import pickle
    import shapely

    out: dict[str, list[tuple]] = {}
    mode, r_tol, c_pts = calc.input_crs_mode, calc.r_tolerance, calc.circle_points
    sub_cache = _UTM_GEOM_CACHE.setdefault((mode, is_gt, r_tol, c_pts), {})
    if len(sub_cache) > 1_200_000:
        sub_cache.clear()
    target_set = set(calc.target_classes)
    todo_feats: list[tuple[dict, object, int]] = []
    todo_meta: list[tuple[str, float, str, str, int]] = []
    ordered_slots: list[tuple[str, object, float, str, str]] = []
    slot_sites: list[str] = []

    for feature in fc.get("features", []):
        props = feature.get("properties", {}) or {}
        cls = mod.normalize_class_name(props.get("class_name", "unknown"))
        if cls is None or cls not in target_set:
            continue
        gdict = feature.get("geometry")
        if not gdict:
            continue
        conf = mod._get_confidence(feature)
        reg = str(props.get("region_name", "default_source"))
        grp = threshold_group(props)
        if sites_out is not None:
            slot_sites.append(key_fn(props))
        gid = id(gdict)
        hit = sub_cache.get(gid)
        if hit is not None and hit[0] is gdict:
            ordered_slots.append((cls, hit[1], conf, reg, grp))
        else:
            slot_idx = len(ordered_slots)
            ordered_slots.append((cls, None, conf, reg, grp))
            todo_feats.append((feature, gdict, gid))
            todo_meta.append((cls, conf, reg, grp, slot_idx))

    if todo_feats:
        shm_loaded = False
        shm_file = None
        ref_dir = _geomcache.resolve_dir(_SHM_CACHE_DIR)   # прежний pickle-кэш; каталог — как у нового (EXPDS_GEOM_CACHE)
        if len(todo_feats) >= 200 and ref_dir is not None:
            try:
                bhash = _todo_batch_hash(todo_feats, todo_meta, mode, is_gt, r_tol, c_pts)
                shm_file = ref_dir / f"{bhash}.wkb.pkl"
                if shm_file.exists():
                    valid_idx, wkbs = pickle.loads(shm_file.read_bytes())
                    geoms = shapely.from_wkb(wkbs)
                    for g, idx in zip(geoms, valid_idx):
                        _, gdict, gid = todo_feats[idx]
                        sub_cache[gid] = (gdict, g)
                        cls, conf, reg, grp, slot_idx = todo_meta[idx]
                        ordered_slots[slot_idx] = (cls, g, conf, reg, grp)
                    shm_loaded = True
            except Exception:
                shm_loaded = False

        if not shm_loaded:
            raw_geoms = []
            valid_idx = []
            # w0: пакетный разбор (Rust flatten_polygons + shapely.from_ragged_array), побайтно = официальному;
            # выключается EXPDS_FASTGEOM=0 / EXPDS_BACKEND=numpy / без собранного expds_core
            from . import fastgeom as _fastgeom
            extracted = _fastgeom.extract_many(mod, [t[0] for t in todo_feats])
            for idx, g in enumerate(extracted):
                if g is not None:
                    raw_geoms.append(g)
                    valid_idx.append(idx)
            if raw_geoms:
                if mode == "utm":
                    proj_geoms = raw_geoms
                else:
                    proj_geoms = list(raw_geoms)
                    if mode == "webmercator":
                        need_idx = list(range(len(raw_geoms)))
                    elif _fastgeom.enabled():
                        need_idx = np.flatnonzero(
                            _fastgeom.max_abs_x(raw_geoms) > mod._UTM_ABS_X_THRESHOLD).tolist()
                    else:
                        need_idx = [
                            k_i for k_i, g in enumerate(raw_geoms)
                            if mod._geom_max_abs_x(g) > mod._UTM_ABS_X_THRESHOLD
                        ]
                    if need_idx:
                        try:
                            sub_raw = [raw_geoms[k_i] for k_i in need_idx]
                            cents = shapely.centroid(sub_raw)
                            xs = shapely.get_x(cents)
                            ys = shapely.get_y(cents)
                            lons, lats = mod._get_transformer_3857_to_wgs84().transform(xs, ys)
                            zones = np.clip(((np.asarray(lons) + 180.0) // 6).astype(np.int32) + 1, 1, 60)
                            epsgs = np.where(np.asarray(lats) >= 0, 32600, 32700) + zones
                            for epsg in np.unique(epsgs):
                                z_mask = np.nonzero(epsgs == epsg)[0]
                                tr = mod._get_transformer_3857_to_utm(int(epsg))
                                z_geoms = [sub_raw[z_i] for z_i in z_mask]
                                if _fastgeom.enabled():
                                    z_proj = shapely.transform(z_geoms, _fastgeom.utm_coords_fn(tr, int(epsg)))
                                else:
                                    z_proj = shapely.transform(
                                        z_geoms, lambda c, _tr=tr: np.column_stack(_tr.transform(c[:, 0], c[:, 1]))
                                    )
                                for z_i, pg in zip(z_mask, z_proj):
                                    proj_geoms[need_idx[z_i]] = pg
                        except Exception:
                            for k_i in need_idx:
                                proj_geoms[k_i] = mod._to_utm(raw_geoms[k_i], mode)
                final_geoms = []
                for g, idx in zip(proj_geoms, valid_idx):
                    if g.geom_type == "Point" and not is_gt:
                        g = mod._point_to_circle(g, r_tol, c_pts)
                    final_geoms.append(g)
                    _, gdict, gid = todo_feats[idx]
                    sub_cache[gid] = (gdict, g)
                    cls, conf, reg, grp, slot_idx = todo_meta[idx]
                    ordered_slots[slot_idx] = (cls, g, conf, reg, grp)
                if shm_file is not None and final_geoms:
                    try:
                        ref_dir.mkdir(parents=True, exist_ok=True)
                        wkbs = shapely.to_wkb(final_geoms)
                        tmp = shm_file.with_suffix(f".tmp.{os.getpid()}")
                        tmp.write_bytes(pickle.dumps((valid_idx, wkbs), protocol=pickle.HIGHEST_PROTOCOL))
                        tmp.replace(shm_file)
                    except Exception:
                        pass

    for k, (cls, g, conf, reg, grp) in enumerate(ordered_slots):
        if g is not None:
            out.setdefault(cls, []).append((g, conf, reg, grp))
            if sites_out is not None:
                sites_out.setdefault(cls, []).append(slot_sites[k])
    return out


# ───────────── быстрое сопоставление: векторная проверка пар + память между вызовами ─────────────
# sha256 исходников официальных _match_quality/_match_polygon_gt/_match_point_gt/_polygon_coverage/
# _inner_point_in_mask (compute_metrics.py sha256 299792cc…0081). Если официальный код другой —
# векторный путь выключается и каждая пара идёт через calc._match_quality, как раньше.
_MATCH_SRC_SHA = "b9fce323c44c26837476d88b767d56cf02b9334e2773c438e15f613219d7bf8c"
_MATCH_OK: dict = {}
_PAIR_MEMO: dict = {}      # (id(mod), cov_thr, r_tol) -> {(id(pred), id(gt)): (pred, gt, ok, s)}
_GT_TREES: dict = {}       # (r_tol, ids эталонов) -> (эталоны, STRtree)


def _fast_match_enabled(mod, calc) -> bool:
    if os.environ.get("EXPDS_FAST_MATCH", "1") == "0":
        return False
    if "_match_quality" in vars(calc) or type(calc) is not mod.F1Calculator:
        return False
    hit = _MATCH_OK.get(id(mod))
    if hit is not None and hit[0] is mod:
        return hit[1]
    ok = False
    try:
        import hashlib
        import inspect
        import shapely.geometry

        C = mod.F1Calculator
        src = "".join(inspect.getsource(f) for f in (C._match_quality, C._match_polygon_gt, C._match_point_gt,
                                                     mod._polygon_coverage, mod._inner_point_in_mask))
        ok = (hashlib.sha256(src.encode()).hexdigest() == _MATCH_SRC_SHA
              and mod.Point is shapely.geometry.Point and mod.math.hypot is __import__("math").hypot)
    except Exception:
        ok = False
    _MATCH_OK[id(mod)] = (mod, ok)
    return ok


def _pair_quality_vec(calc, P: np.ndarray, G: np.ndarray) -> tuple[list, list]:
    """[calc._match_quality(P[k], G[k]) for k], векторно (те же вызовы GEOS, та же арифметика float64)."""
    import math

    import shapely

    n = len(P)
    ok = np.zeros(n, dtype=bool)
    sc = np.zeros(n, dtype=np.float64)
    gpt = shapely.get_type_id(G) == 0          # isinstance(gt, Point)
    ip = np.flatnonzero(gpt)
    if ip.size:                                  # _match_point_gt
        r = calc.r_tolerance
        cen = shapely.centroid(P[ip])
        cx, cy = shapely.get_x(cen).tolist(), shapely.get_y(cen).tolist()
        gx, gy = shapely.get_x(G[ip]).tolist(), shapely.get_y(G[ip]).tolist()
        for t, k in enumerate(ip.tolist()):
            d = math.hypot(cx[t] - gx[t], cy[t] - gy[t])
            if d <= r:
                ok[k] = True
                sc[k] = -d
    iq = np.flatnonzero(~gpt)
    if iq.size:                                  # _match_polygon_gt
        A, B = P[iq], G[iq]
        aa, ab = shapely.area(A), shapely.area(B)
        mn = np.where(ab < aa, ab, aa)           # = min(a.area, b.area), включая NaN
        need = shapely.is_valid(A) & shapely.is_valid(B) & ~(mn <= 0)
        cov = np.zeros(iq.size, dtype=np.float64)
        w = np.flatnonzero(need)
        if w.size:
            inter = shapely.area(shapely.intersection(A[w], B[w]))
            with np.errstate(all="ignore"):
                cov[w] = inter / mn[w]
        h = np.flatnonzero(cov >= calc.coverage_threshold)
        hit = np.zeros(iq.size, dtype=bool)
        if h.size:
            pts = shapely.point_on_surface(A[h])
            inside = shapely.contains(B[h], pts)
            rest = np.flatnonzero(~inside)
            if rest.size:
                inside[rest] = shapely.distance(shapely.boundary(B[h][rest]), pts[rest]) < 1e-10
            hit[h] = inside
        ok[iq] = hit
        sc[iq] = cov
    return ok.tolist(), sc.tolist()


def _pair_quality(mod, calc, pg_ord: list, gg: list, qp: list, qg: list) -> tuple[list, list]:
    """(ok, s) для пар (pg_ord[qp[k]], gg[qg[k]]) — с памятью между вызовами build_curves."""
    n = len(qp)
    if not _fast_match_enabled(mod, calc):
        res = [calc._match_quality(pg_ord[r], gg[j]) for r, j in zip(qp, qg)]
        return [x[0] for x in res], [x[1] for x in res]
    try:
        cap = int(os.environ.get("EXPDS_PAIR_MEMO", "2000000"))
    except ValueError:
        cap = 2_000_000
    memo = _PAIR_MEMO.setdefault((id(mod), calc.coverage_threshold, calc.r_tolerance), {}) if cap > 0 else {}
    oks: list = [False] * n
    ss: list = [0.0] * n
    miss: list[int] = []
    get = memo.get
    for k in range(n):
        a = pg_ord[qp[k]]
        b = gg[qg[k]]
        v = get((id(a), id(b)))
        if v is not None and v[0] is a and v[1] is b:
            oks[k] = v[2]
            ss[k] = v[3]
        else:
            miss.append(k)
    if miss:
        P = np.empty(len(miss), dtype=object)
        G = np.empty(len(miss), dtype=object)
        P[:] = [pg_ord[qp[k]] for k in miss]
        G[:] = [gg[qg[k]] for k in miss]
        try:
            o2, s2 = _pair_quality_vec(calc, P, G)
        except Exception:   # исключение GEOS в пачке — по одной паре официальной функцией (там свои except)
            res = [calc._match_quality(a, b) for a, b in zip(P, G)]
            o2, s2 = [x[0] for x in res], [x[1] for x in res]
        if cap > 0 and len(memo) + len(miss) > cap:
            memo.clear()
        for t, k in enumerate(miss):
            oks[k] = o2[t]
            ss[k] = s2[t]
            if cap > 0:
                a, b = P[t], G[t]
                memo[(id(a), id(b))] = (a, b, o2[t], s2[t])
    return oks, ss


def _gt_tree(calc, gg: list):
    """STRtree эталонов класса (точки — буфер r_tolerance), с кэшем на тот же список объектов."""
    import shapely
    from shapely.strtree import STRtree

    key = (calc.r_tolerance, tuple(map(id, gg)))
    hit = _GT_TREES.get(key)
    if hit is not None and len(hit[0]) == len(gg) and all(a is b for a, b in zip(hit[0], gg)):
        return hit[1]
    arr = np.empty(len(gg), dtype=object)
    arr[:] = gg
    is_pt = shapely.get_type_id(arr) == 0
    geoms = arr.copy()
    if is_pt.any():
        geoms[is_pt] = shapely.buffer(arr[is_pt], calc.r_tolerance)   # = x.buffer(r) для каждой точки
    tree = STRtree(geoms)
    if len(_GT_TREES) >= 32:
        _GT_TREES.clear()
    _GT_TREES[key] = (list(gg), tree)
    return tree


def build_curves(pred: dict, gt: dict, metric_path: str | None = None,
                 scope: str | None = None) -> dict[str, ClassCurve]:
    """Официальное жадное сопоставление по каждому классу, один раз на всех кандидатах.

    Кандидаты на пару ищутся пространственным индексом; порядок перебора эталонов
    (по возрастанию индекса) и правило выбора лучшего — как в официальном коде.
    scope (или EXPDS_MATCH_SCOPE): "global" — весь файл одним пулом, как calculate_f1;
    "site" — пара допускается только внутри одного участка (site_key);
    "layer" — пара допускается только внутри одного слоя данных (layer_key / markup_type).
    Счётчики TP/FP/FN по слоям складываются (микро-F1), как у scope=site по участкам.
    EXPDS_FAST_METRIC=0 — прежний путь целиком (build_curves_reference + _collect_reference).
    """
    import shapely

    if not _fast_on():
        return build_curves_reference(pred, gt, metric_path, scope)
    mod = load_official(metric_path)
    calc = mod.F1Calculator()
    scope_s = match_scope(scope)
    keyed = scope_s in ("site", "layer")
    key_fn = layer_key if scope_s == "layer" else site_key
    p_sites: dict | None = {} if keyed else None
    g_sites: dict | None = {} if keyed else None
    preds = _collect(mod, calc, pred, False, p_sites, key_fn=key_fn)
    gts = _collect(mod, calc, gt, True, g_sites, key_fn=key_fn)
    return match_collected(calc, preds, gts, p_sites, g_sites, mod=mod)


def build_curves_with_pairs(pred: dict, gt: dict, metric_path: str | None = None,
                            scope: str | None = None) -> tuple[dict[str, ClassCurve], list[dict]]:
    """build_curves + список TP-пар с pred_layer/gt_layer (layer_key).

    Сбор геометрий один раз с layer_key в аннотацию. Фильтр пар: global (нет) или
    layer (same layer_key). scope=site здесь не поддерживается (см. build_curves).
    Нужен для диагностики Δ global↔layer: какие TP — cross-layer.
    """
    mod = load_official(metric_path)
    calc = mod.F1Calculator()
    scope_s = match_scope(scope)
    if scope_s == "site":
        raise ValueError("build_curves_with_pairs: scope=site не поддерживается; используйте build_curves")
    p_layers: dict = {}
    g_layers: dict = {}
    preds = _collect(mod, calc, pred, False, p_layers, key_fn=layer_key)
    gts = _collect(mod, calc, gt, True, g_layers, key_fn=layer_key)
    if scope_s == "layer":
        p_filt, g_filt = p_layers, g_layers
    else:
        p_filt = g_filt = None
    pairs: list[dict] = []
    curves = match_collected(calc, preds, gts, p_filt, g_filt, mod=mod,
                             pairs_out=pairs, p_ann=p_layers, g_ann=g_layers)
    return curves, pairs


def match_collected(calc, preds: dict, gts: dict, p_sites: dict | None = None,
                    g_sites: dict | None = None, mod=None,
                    pairs_out: list | None = None,
                    p_ann: dict | None = None,
                    g_ann: dict | None = None) -> dict[str, ClassCurve]:
    """Сопоставление по уже собранным (геометрия UTM, conf, участок, группа) спискам.

    Вынесено из build_curves, чтобы большой стек можно было собирать кусками
    (по семьям, пачками) и не держать весь GeoJSON в памяти (scripts/family_curves.py).
    p_sites/g_sites (ключи области по классам, из _collect: site_key или layer_key) —
    для scope="site"|"layer"; None = global.
    pairs_out: если список — дописываем TP-пары {class, conf, quality, pred_layer, gt_layer,
    pred_region, gt_region, cross_layer}. p_ann/g_ann — ключи слоя для аннотации
    (обычно layer_key); если None, берём p_sites/g_sites.
    mod — официальный модуль метрики; по умолчанию модуль, из которого создан calc.
    """
    import shapely

    if mod is None:   # модуль load_official не регистрируется в sys.modules: пространство имён класса calc
        import types
        mod = types.SimpleNamespace(**type(calc).__init__.__globals__)
    per_site = p_sites is not None and g_sites is not None
    ann_p = p_ann if p_ann is not None else p_sites
    ann_g = g_ann if g_ann is not None else g_sites
    curves: dict[str, ClassCurve] = {}
    for cls in calc.target_classes:
        p, g = preds.get(cls, []), gts.get(cls, [])
        order = sorted(range(len(p)), key=lambda i: p[i][1], reverse=True)   # стабильно, как sorted()
        hits = np.zeros(len(p), dtype=bool)
        if p and g:
            gg = [x[0] for x in g]
            tree = _gt_tree(calc, gg)
            pg_ord = [p[i][0] for i in order]
            q_pred, q_gt = tree.query(pg_ord)
            if per_site and len(q_pred):
                site_ids: dict = {}
                ps_ord = np.array([site_ids.setdefault(p_sites[cls][i], len(site_ids)) for i in order], dtype=np.int64)
                gs = np.array([site_ids.setdefault(x, len(site_ids)) for x in g_sites[cls]], dtype=np.int64)
                same = ps_ord[q_pred] == gs[q_gt]
                q_pred, q_gt = q_pred[same], q_gt[same]
            if len(q_pred):
                # официально: предсказания не-Polygon пропускаются; пары — в порядке (ранг, индекс эталона)
                arr = np.empty(len(pg_ord), dtype=object)
                arr[:] = pg_ord
                poly = shapely.get_type_id(arr) == 3
                keep = poly[q_pred]
                q_pred, q_gt = q_pred[keep], q_gt[keep]
                o = np.lexsort((q_gt, q_pred))
                qp, qg = q_pred[o].tolist(), q_gt[o].tolist()
                oks, ss = _pair_quality(mod, calc, pg_ord, gg, qp, qg)
                # жадный выбор, как в официальном цикле: пары с ok=False лучшего не меняют
                matched: set[int] = set()
                k, n_pairs = 0, len(qp)
                while k < n_pairs:
                    r_int = qp[k]
                    best, best_j = float("-inf"), -1
                    while k < n_pairs and qp[k] == r_int:
                        if oks[k]:
                            j = qg[k]
                            if j not in matched:
                                s_k = ss[k]
                                if s_k > best + 1e-9:
                                    best, best_j = s_k, j
                        k += 1
                    if best_j >= 0:
                        matched.add(best_j)
                        hits[r_int] = True
                        if pairs_out is not None:
                            pi = order[r_int]
                            pl = ann_p[cls][pi] if ann_p is not None and cls in ann_p else "?"
                            gl = ann_g[cls][best_j] if ann_g is not None and cls in ann_g else "?"
                            pairs_out.append({
                                "class": cls,
                                "conf": float(p[pi][1]),
                                "quality": float(best),
                                "pred_layer": pl,
                                "gt_layer": gl,
                                "pred_region": p[pi][2],
                                "gt_region": g[best_j][2],
                                "cross_layer": pl != gl,
                            })
        curves[cls] = ClassCurve(conf=np.array([p[i][1] for i in order], dtype=np.float64), hit=hits,
                                 pred_region=[p[i][2] for i in order], n_gt=len(g),
                                 gt_regions=Counter(x[2] for x in g), pred_group=[p[i][3] for i in order])
    return curves


def build_curves_reference(pred: dict, gt: dict, metric_path: str | None = None,
                 scope: str | None = None) -> dict[str, ClassCurve]:
    """Прежняя (до ускорения bot-8) реализация build_curves — эталон для тестов: каждая пара через
    calc._match_quality, дерево строится заново. Официальное жадное сопоставление по каждому классу, один раз на всех кандидатах.

    Кандидаты на пару ищутся пространственным индексом; порядок перебора эталонов
    (по возрастанию индекса) и правило выбора лучшего — как в официальном коде.
    scope (или EXPDS_MATCH_SCOPE): "global" — весь файл одним пулом, как calculate_f1;
    "site" — пара допускается только внутри одного участка (site_key);
    "layer" — пара допускается только внутри одного слоя данных (layer_key / markup_type).
    """
    from shapely.strtree import STRtree

    mod = load_official(metric_path)
    calc = mod.F1Calculator()
    scope_s = match_scope(scope)
    keyed = scope_s in ("site", "layer")
    key_fn = layer_key if scope_s == "layer" else site_key
    p_sites: dict | None = {} if keyed else None
    g_sites: dict | None = {} if keyed else None
    preds = _collect_reference(mod, calc, pred, False, p_sites, key_fn=key_fn)
    gts = _collect_reference(mod, calc, gt, True, g_sites, key_fn=key_fn)
    curves: dict[str, ClassCurve] = {}
    for cls in calc.target_classes:
        p, g = preds.get(cls, []), gts.get(cls, [])
        order = sorted(range(len(p)), key=lambda i: p[i][1], reverse=True)   # стабильно, как sorted()
        hits = np.zeros(len(p), dtype=bool)
        if p and g:
            gg = [x[0] for x in g]
            tree = STRtree([x.buffer(calc.r_tolerance) if x.geom_type == "Point" else x for x in gg])
            pg_ord = [p[i][0] for i in order]
            q_pred, q_gt = tree.query(pg_ord)
            if keyed and len(q_pred):
                site_ids: dict = {}
                ps_ord = np.array([site_ids.setdefault(p_sites[cls][i], len(site_ids)) for i in order], dtype=np.int64)
                gs = np.array([site_ids.setdefault(x, len(site_ids)) for x in g_sites[cls]], dtype=np.int64)
                same = ps_ord[q_pred] == gs[q_gt]
                q_pred, q_gt = q_pred[same], q_gt[same]
            if len(q_pred):
                splits = np.flatnonzero(np.diff(q_pred)) + 1
                rank_groups = np.split(q_gt, splits)
                unique_ranks = q_pred[np.r_[0, splits]]
                matched: set[int] = set()
                for rank, cand_js in zip(unique_ranks, rank_groups):
                    r_int = int(rank)
                    pg = pg_ord[r_int]
                    if pg.geom_type != "Polygon":
                        continue
                    best, best_j = float("-inf"), -1
                    for j in sorted(int(k) for k in cand_js):
                        if j in matched:
                            continue
                        ok, s = calc._match_quality(pg, gg[j])
                        if ok and s > best + 1e-9:
                            best, best_j = s, j
                    if best_j >= 0:
                        matched.add(best_j)
                        hits[r_int] = True
        curves[cls] = ClassCurve(conf=np.array([p[i][1] for i in order], dtype=np.float64), hit=hits,
                                 pred_region=[p[i][2] for i in order], n_gt=len(g),
                                 gt_regions=Counter(x[2] for x in g), pred_group=[p[i][3] for i in order])
    return curves



def _group_cuts(conf: np.ndarray, hit: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Допустимые срезы группы (уверенности по убыванию): сколько оставить, TP и порог на каждом."""
    if conf.size == 0:
        return np.array([0]), np.array([0.0]), np.array([np.inf])
    last = np.r_[np.flatnonzero(conf[1:] != conf[:-1]), conf.size - 1]
    keep = np.r_[0, last + 1]
    tp = np.r_[0.0, np.cumsum(hit)[last].astype(float)]
    lower = np.r_[conf[last[:-1] + 1], -np.inf] if last.size else np.array([-np.inf])
    taus = np.r_[conf[0] + 1e-6, np.where(np.isfinite(lower), (conf[last] + lower) / 2, 0.0)]
    return keep, tp, taus


def _f1_along(groups: list[str], cuts: dict, k: dict, g: str, n_gt: int) -> np.ndarray:
    """F1 класса, когда у группы g меняется срез, а остальные группы зафиксированы."""
    tp_o = sum(cuts[h][1][k[h]] for h in groups if h != g)
    n_o = sum(cuts[h][0][k[h]] for h in groups if h != g)
    keep, tp, _ = cuts[g]
    t, n = tp_o + tp, n_o + keep
    return np.where(n > 0, _f1(t, n - t, n_gt - t), 0.0)


def _plateau_indices(groups: list[str], cuts: dict, k: dict, n_gt: int) -> dict:
    """Один проход: каждая группа встаёт на середину своего плато при уже найденных остальных.

    Повторный подъём здесь не нужен. Середина чуть ниже острого максимума, и следующий
    проход снова уехал бы к краю.
    """
    chosen = dict(k)
    for g in groups:
        chosen[g] = plateau_index(_f1_along(groups, cuts, chosen, g, n_gt))
    return chosen


def sweep_groups_reference(curves: dict[str, ClassCurve], rounds: int = 20, pick: str = "max") -> dict:
    """Пороги по (группа, класс), максимизирующие общий F1 класса по всем группам.

    Метрика сводит класс по всем модальностям и участкам: F1 ≈ 2TP / (TP + 0,5·FP + GT).
    Покоординатный подъём: порог одной группы — лучший при остальных. Два старта
    (все группы выключены, все включены), берётся лучший. Попадания — из общего
    прохода, поэтому группы, борющиеся за один эталон, оцениваются приближённо;
    точный балл — verify_groups.

    pick="max" оставляет острый максимум. pick="plateau" после него сдвигает каждую
    группу на середину плато F1 (допуск 0,002). В конфиг этот сдвиг не записывается,
    пока официальный балл на новых участках его не подтвердит.
    """
    out: dict[str, dict] = {}
    for cls, c in curves.items():
        groups = sorted(set(c.pred_group)) if c.pred_group else []
        if c.n_gt == 0:
            continue
        if not groups:
            out[cls] = {"f1": 0.0, "n_gt": c.n_gt, "groups": {}}
            continue
        g_arr = np.array(c.pred_group)
        cuts = {g: _group_cuts(c.conf[g_arr == g], c.hit[g_arr == g]) for g in groups}

        def f1_of(k: dict) -> float:
            tp = sum(cuts[g][1][k[g]] for g in groups)
            n = sum(cuts[g][0][k[g]] for g in groups)
            return float(_f1(tp, n - tp, c.n_gt - tp)) if n else 0.0

        best_k, best_f = None, -1.0
        for start in ("none", "all"):
            k = {g: 0 if start == "none" else len(cuts[g][0]) - 1 for g in groups}
            for _ in range(rounds):
                changed = False
                for g in groups:
                    f = _f1_along(groups, cuts, k, g, c.n_gt)
                    j = int(np.argmax(f))
                    if j != k[g] and f[j] > f[k[g]] + 1e-12:
                        k[g], changed = j, True
                if not changed:
                    break
            f = f1_of(k)
            if f > best_f:
                best_f, best_k = f, dict(k)
        if pick == "plateau" and best_k is not None:
            best_k = _plateau_indices(groups, cuts, best_k, c.n_gt)
            best_f = f1_of(best_k)
        elif pick != "max":
            raise ValueError(f"неизвестный выбор порога: {pick}")
        out[cls] = {"f1": best_f, "n_gt": c.n_gt,
                    "groups": {g: {"threshold": round(float(cuts[g][2][best_k[g]]), 6),
                                   "kept": int(cuts[g][0][best_k[g]]), "tp": int(cuts[g][1][best_k[g]])}
                               for g in groups}}
    return out


def official_weighted(per_class: dict[str, dict]) -> float:
    """Взвешенный балл по F1 классов с эталоном (как в метрике)."""
    num = sum(CLASS_WEIGHTS.get(c, 0.0) * v["f1"] for c, v in per_class.items() if v.get("n_gt", 0) > 0)
    den = sum(CLASS_WEIGHTS.get(c, 0.0) for c, v in per_class.items() if v.get("n_gt", 0) > 0)
    return num / den if den else 0.0


def apply_group_thresholds(fc: dict, taus: dict[str, dict[str, float]], metric_path: str | None = None) -> dict:
    """Оставляет объекты с confidence ≥ taus[группа][класс]; нет порога — объект остаётся."""
    norm = load_official(metric_path).normalize_class_name
    feats = []
    for f in fc.get("features", []):
        pr = f.get("properties") or {}
        tau = taus.get(threshold_group(pr), {}).get(norm(pr.get("class_name")), 0.0)
        if float(pr.get("confidence", 1.0)) >= tau:
            feats.append(f)
    return {**{k: v for k, v in fc.items() if k != "features"}, "features": feats}


def _f1(tp, fp, fn):
    """F1 ровно как в метрике: FP с весом 0,5 в точности, эпсилон 1e-6."""
    precision = tp / (tp + 0.5 * fp + EPS)
    recall = tp / (tp + fn + EPS)
    return 2 * precision * recall / (precision + recall + EPS)


def curve_table(c: ClassCurve) -> tuple[np.ndarray, np.ndarray]:
    """Все различимые пороги класса (оставляем conf ≥ τ) и F1 на каждом; первый — «ничего не выдавать»."""
    if c.conf.size == 0:
        return np.array([np.inf]), np.array([0.0])
    last = np.r_[np.flatnonzero(c.conf[1:] != c.conf[:-1]), c.conf.size - 1]
    tp = np.cumsum(c.hit)[last].astype(float)
    n = (last + 1).astype(float)
    return np.r_[np.inf, c.conf[last]], np.r_[0.0, _f1(tp, n - tp, c.n_gt - tp)]


def plateau_index(f1: np.ndarray, tol: float = 0.002) -> int:
    """Середина плато: все пороги, где F1 не ниже максимума больше чем на tol.

    Острый оптимум на одном значении confidence плохо переносится на новые участки.
    При ровной кривой берётся середина, а не левый или правый край.
    """
    scores = np.asarray(f1, dtype=float)
    if scores.size == 0:
        return 0
    ok = np.flatnonzero(scores >= float(scores.max()) - tol)
    return int(ok[len(ok) // 2])


def score_at(curves: dict[str, ClassCurve], thresholds: dict[str, float]) -> tuple[float, dict]:
    """Итоговый балл при порогах (классы без эталона из среднего исключаются, как в метрике)."""
    num = den = 0.0
    per = {}
    for cls, c in curves.items():
        tau = thresholds.get(cls) or 0.0
        kept = c.conf >= tau
        tp = int(c.hit[kept].sum())
        fp = int(kept.sum()) - tp
        f1 = float(_f1(tp, fp, c.n_gt - tp))
        per[cls] = {"f1": f1, "tp": tp, "fp": fp, "fn": c.n_gt - tp, "n_gt": c.n_gt}
        if c.n_gt > 0:
            num += CLASS_WEIGHTS.get(cls, 0.0) * f1
            den += CLASS_WEIGHTS.get(cls, 0.0)
    return (num / den if den else 0.0), per


def curve_where(c: ClassCurve, regions: set[str]) -> ClassCurve:
    """Та же кривая, только объекты и эталон выбранных участков. Порядок уверенности сохраняется."""
    if not c.pred_region:
        mask = np.zeros(0, dtype=bool)
    else:
        mask = np.fromiter((r in regions for r in c.pred_region), dtype=bool, count=len(c.pred_region))
    gt = Counter({r: n for r, n in c.gt_regions.items() if r in regions})
    groups = [g for g, keep in zip(c.pred_group, mask) if keep] if c.pred_group else []
    return ClassCurve(c.conf[mask], c.hit[mask], [r for r, keep in zip(c.pred_region, mask) if keep],
                      int(sum(gt.values())), gt, groups)


def score_group_thresholds(curves: dict[str, ClassCurve], taus: dict[str, dict[str, float]]) -> tuple[float, dict]:
    """Балл по кривым при пороге (группа, класс). Класс без эталона из среднего выпадает."""
    num = den = 0.0
    per = {}
    for cls, c in curves.items():
        if c.conf.size == 0:
            kept = np.zeros(0, dtype=bool)
        else:
            tau = np.fromiter((taus.get(g, {}).get(cls, 0.0) for g in c.pred_group),
                              dtype=np.float64, count=len(c.pred_group))
            kept = c.conf >= tau
        tp = int(c.hit[kept].sum()) if kept.size else 0
        fp = int(kept.sum()) - tp
        f1 = float(_f1(tp, fp, c.n_gt - tp))
        per[cls] = {"f1": f1, "tp": tp, "fp": fp, "fn": c.n_gt - tp, "n_gt": c.n_gt}
        if c.n_gt > 0:
            num += CLASS_WEIGHTS.get(cls, 0.0) * f1
            den += CLASS_WEIGHTS.get(cls, 0.0)
    return (num / den if den else 0.0), per


def bootstrap_reference(curves: dict[str, ClassCurve], n: int = 200, seed: int = 0) -> dict[str, dict]:
    """Устойчивость порогов: регионы выбираются с возвращением, оптимум ищется заново.

    Регионы в пространстве не пересекаются, поэтому попадания из общего прохода
    переиспользуются с весами (сколько раз регион попал в выборку).
    """
    regions = sorted({r for c in curves.values() for r in c.pred_region} |
                     {r for c in curves.values() for r in c.gt_regions})
    if not regions or n <= 0:
        return {}
    idx = {r: i for i, r in enumerate(regions)}
    rng = np.random.default_rng(seed)
    taus: dict[str, list[float]] = {c: [] for c in curves}
    for _ in range(n):
        w = np.bincount(rng.integers(0, len(regions), len(regions)), minlength=len(regions)).astype(float)
        for cls, c in curves.items():
            n_gt = sum(w[idx[r]] * k for r, k in c.gt_regions.items())
            if n_gt == 0 or c.conf.size == 0:
                continue
            pw = np.array([w[idx[r]] for r in c.pred_region])
            last = np.r_[np.flatnonzero(c.conf[1:] != c.conf[:-1]), c.conf.size - 1]
            tp, nn = np.cumsum(c.hit * pw)[last], np.cumsum(pw)[last]
            taus[cls].append(float(c.conf[last[int(np.argmax(_f1(tp, nn - tp, n_gt - tp)))]]))
    return {cls: {"p10": float(np.percentile(v, 10)), "p50": float(np.median(v)),
                  "p90": float(np.percentile(v, 90)), "samples": len(v)}
            for cls, v in taus.items() if v}


def paired_bootstrap_reference(a: dict[str, ClassCurve], b: dict[str, ClassCurve], tau_a: dict[str, float],
                     tau_b: dict[str, float], n: int = 1000, seed: int = 0) -> dict:
    """Разница баллов A − B на одних и тех же выборках регионов с возвращением.

    Пороги фиксированы (как в сабмите). Одни веса регионов на обе модели убирают
    из разницы разброс «какие участки попались» — сравнение парное.
    """
    regions = sorted({r for cs in (a, b) for c in cs.values() for r in c.pred_region} |
                     {r for cs in (a, b) for c in cs.values() for r in c.gt_regions})
    if not regions or n <= 0:
        return {}
    idx = {r: i for i, r in enumerate(regions)}
    rng = np.random.default_rng(seed)

    def prep(curves, taus):
        out = {}
        for cls, c in curves.items():
            kept = c.conf >= (taus.get(cls) or 0.0)
            out[cls] = (np.array([idx[r] for r in c.pred_region], np.int64)[kept], c.hit[kept],
                        {idx[r]: k for r, k in c.gt_regions.items()})
        return out

    pa, pb = prep(a, tau_a), prep(b, tau_b)
    classes = sorted(set(pa) | set(pb))
    diffs, per = [], {c: [] for c in classes}
    for _ in range(n):
        w = np.bincount(rng.integers(0, len(regions), len(regions)), minlength=len(regions)).astype(float)
        s = []
        for prepared in (pa, pb):
            num = den = 0.0
            f1s = {}
            for cls in classes:
                if cls not in prepared:
                    continue
                reg, hit, gtr = prepared[cls]
                n_gt = sum(w[r] * k for r, k in gtr.items())
                if n_gt == 0:
                    continue
                pw = w[reg]
                tp = float((pw * hit).sum())
                f1 = float(_f1(tp, float(pw.sum()) - tp, n_gt - tp))
                f1s[cls] = f1
                num += CLASS_WEIGHTS.get(cls, 0.0) * f1
                den += CLASS_WEIGHTS.get(cls, 0.0)
            s.append((num / den if den else 0.0, f1s))
        diffs.append(s[0][0] - s[1][0])
        for cls in classes:
            if cls in s[0][1] and cls in s[1][1]:
                per[cls].append(s[0][1][cls] - s[1][1][cls])

    def summary(v):
        v = np.asarray(v)
        return {"mean": float(v.mean()), "p5": float(np.percentile(v, 5)), "p50": float(np.median(v)),
                "p95": float(np.percentile(v, 95)), "p_gt0": float((v > 0).mean())}

    return {"score": summary(diffs), "per_class": {c: summary(v) for c, v in per.items() if v}, "samples": n}


# ─────────────────────────── честный балл: порог с других семей ─────────────────

def family_folds(regions, family_of, k: int | None = None) -> list[set[str]]:
    """Фолды участков по семьям. k=None или k ≥ числа семей — одна семья на фолд.

    Семьи раскладываются жадно по размеру в k корзин, порядок детерминирован.
    Одна семья никогда не делится между фолдами.
    """
    fams: dict[str, set[str]] = {}
    for r in regions:
        fams.setdefault(family_of(r), set()).add(r)
    names = sorted(fams)
    if not k or k >= len(names):
        return [fams[f] for f in names]
    bins: list[set[str]] = [set() for _ in range(k)]
    sizes = [0] * k
    for f in sorted(names, key=lambda f: (-len(fams[f]), f)):
        i = int(np.argmin(sizes))
        bins[i] |= fams[f]
        sizes[i] += len(fams[f])
    return [b for b in bins if b]


def honest_counts_reference(curves: dict[str, ClassCurve], folds: list[set[str]], pick: str = "max",
                  unseen: str = "drop") -> tuple[dict[str, dict[str, list[int]]], dict]:
    """TP, FP и эталоны каждого участка при порогах, подобранных без его фолда.

    Для фолда пороги (группа, класс) ищет sweep_groups на остальных участках и
    применяет к участкам фолда. Так устроен «честный балл» 0,329489: порог семьи
    не видит её эталона. unseen="drop" выбрасывает объекты группы, которой на
    остальных участках не было (порог не из чего подобрать); "keep" их оставляет.

    Возвращает ({класс: {участок: [tp, fp, n_gt]}}, сводку по фолдам).
    """
    every = set().union(*folds) if folds else set()
    counts: dict[str, dict[str, list[int]]] = {cls: {} for cls in curves}
    info = {"folds": len(folds), "unseen_dropped": 0, "unseen_kept": 0}
    for held in folds:
        train = every - held
        res = sweep_groups_reference({cls: curve_where(c, train) for cls, c in curves.items()}, pick=pick)
        taus: dict[str, dict[str, float]] = {}
        for cls, v in res.items():
            for g, gv in v["groups"].items():
                taus.setdefault(g, {})[cls] = gv["threshold"]
        for cls, c in curves.items():
            hc = curve_where(c, held)
            per = counts[cls]
            for r, n in hc.gt_regions.items():
                per.setdefault(r, [0, 0, 0])[2] += int(n)
            if hc.conf.size == 0:
                continue
            raw = [taus.get(g, {}).get(cls) for g in hc.pred_group]
            missing = np.fromiter((t is None for t in raw), dtype=bool, count=len(raw))
            fill = np.inf if unseen == "drop" else -np.inf
            tau = np.fromiter((fill if t is None else t for t in raw), dtype=np.float64, count=len(raw))
            info["unseen_dropped" if unseen == "drop" else "unseen_kept"] += int(missing.sum())
            kept = hc.conf >= tau
            for r, hit in zip(np.asarray(hc.pred_region, dtype=object)[kept], hc.hit[kept]):
                per.setdefault(r, [0, 0, 0])[0 if hit else 1] += 1
    return counts, info


def score_counts(counts: dict[str, dict[str, list[int]]], regions: set[str] | None = None,
                 weights: dict[str, float] | None = None) -> tuple[float, dict]:
    """Итоговый балл по счётчикам участков (как в метрике: класс без эталона выпадает).

    weights — кратность участка в выборке бутстрепа; без неё каждый участок один раз.
    """
    num = den = 0.0
    per = {}
    for cls, by_region in counts.items():
        tp = fp = n_gt = 0.0
        for r, (a, b, g) in by_region.items():
            if regions is not None and r not in regions:
                continue
            w = 1.0 if weights is None else weights.get(r, 0.0)
            tp, fp, n_gt = tp + w * a, fp + w * b, n_gt + w * g
        f1 = float(_f1(tp, fp, n_gt - tp))
        per[cls] = {"f1": f1, "tp": tp, "fp": fp, "fn": n_gt - tp, "n_gt": n_gt}
        if n_gt > 0:
            num += CLASS_WEIGHTS.get(cls, 0.0) * f1
            den += CLASS_WEIGHTS.get(cls, 0.0)
    return (num / den if den else 0.0), per


def paired_bootstrap_counts_reference(a: dict, b: dict, family_of, n: int = 1000, seed: int = 0,
                            regions: set[str] | None = None) -> dict:
    """A − B по честным счётчикам. Выборка с возвращением — семьи, не участки.

    Участки одной семьи соседствуют и делят разметчика: если тянуть участки
    по одному, разброс занижается и доля A > B выглядит увереннее, чем есть.
    """
    regs = {r for cs in (a, b) for by in cs.values() for r in by}
    if regions is not None:
        regs &= regions
    fams: dict[str, list[str]] = {}
    for r in sorted(regs):
        fams.setdefault(family_of(r), []).append(r)
    names = sorted(fams)
    if not names or n <= 0:
        return {}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        pick = np.bincount(rng.integers(0, len(names), len(names)), minlength=len(names))
        w = {r: float(k) for f, k in zip(names, pick) if k for r in fams[f]}
        diffs.append(score_counts(a, None, w)[0] - score_counts(b, None, w)[0])
    v = np.asarray(diffs)
    return {"mean": float(v.mean()), "p5": float(np.percentile(v, 5)), "p50": float(np.median(v)),
            "p95": float(np.percentile(v, 95)), "p_gt0": float((v > 0).mean()), "samples": n,
            "families": len(names)}


# ─────────────── быстрые версии (bot-8): те же результаты, без циклов Python по объектам ───────────────
#
# Прежние реализации сохранены с суффиксом _reference; tests/test_bot8_fast_metric.py
# сверяет их побайтно. EXPDS_FAST_METRIC=0 возвращает прежний путь целиком.
#
# Почему результат совпадает точно. Счётчики TP/FP/GT — целые числа, а их взвешенные
# суммы (веса бутстрепа тоже целые) в float64 точны в любом порядке сложения, пока
# меньше 2^53. Поэтому суммы по объектам можно заменить суммами по участкам, а циклы —
# матричным умножением. F1 считается той же формулой и в том же порядке операций
# (IEEE-754 одинаков для скаляров Python и массивов NumPy); взвешенное среднее
# накапливается по классам в прежнем порядке. Генератор случайных чисел вызывается
# теми же вызовами в том же порядке.

def _fast_on() -> bool:
    return os.environ.get("EXPDS_FAST_METRIC", "1") != "0"


def _curve_codes(c: ClassCurve):
    """Коды участков (по первому появлению) и групп (по алфавиту) кривой; кэшируется на объекте."""
    cache = c.__dict__.get("_bot8_codes")
    if (cache is not None and cache[0] is c.pred_region and cache[1] is c.pred_group
            and cache[2] == (len(c.pred_region), len(c.pred_group))):
        return cache[3]
    rmap: dict[str, int] = {}
    rcodes = np.fromiter((rmap.setdefault(r, len(rmap)) for r in c.pred_region), dtype=np.int64,
                         count=len(c.pred_region))
    gnames = sorted(set(c.pred_group)) if c.pred_group else []
    gmap = {g: i for i, g in enumerate(gnames)}
    gcodes = np.fromiter((gmap[g] for g in c.pred_group), dtype=np.int64, count=len(c.pred_group))
    val = (list(rmap), rcodes, gnames, gcodes)
    c.__dict__["_bot8_codes"] = (c.pred_region, c.pred_group, (len(c.pred_region), len(c.pred_group)), val)
    return val


def _sweep_one(conf: np.ndarray, hit: np.ndarray, gcodes: np.ndarray, gnames: list[str], n_gt: int,
               rounds: int, pick: str):
    """Тело sweep_groups_reference для одного класса; группы заданы кодами. None — класс пропускается."""
    if n_gt == 0:
        return None
    if gcodes.size == 0:
        return {"f1": 0.0, "n_gt": n_gt, "groups": {}}
    order = np.argsort(gcodes, kind="stable")      # внутри группы порядок (по убыванию conf) сохраняется
    gs = gcodes[order]
    present, starts = np.unique(gs, return_index=True)
    ends = np.r_[starts[1:], gs.size]
    cs, hs = conf[order], hit[order]
    groups = [gnames[int(k)] for k in present]
    cuts = {g: _group_cuts(cs[a:b], hs[a:b]) for g, a, b in zip(groups, starts, ends)}

    def f1_of(k: dict) -> float:
        tp = sum(cuts[g][1][k[g]] for g in groups)
        n = sum(cuts[g][0][k[g]] for g in groups)
        return float(_f1(tp, n - tp, n_gt - tp)) if n else 0.0

    best_k, best_f = None, -1.0
    for start in ("none", "all"):
        k = {g: 0 if start == "none" else len(cuts[g][0]) - 1 for g in groups}
        for _ in range(rounds):
            changed = False
            for g in groups:
                f = _f1_along(groups, cuts, k, g, n_gt)
                j = int(np.argmax(f))
                if j != k[g] and f[j] > f[k[g]] + 1e-12:
                    k[g], changed = j, True
            if not changed:
                break
        f = f1_of(k)
        if f > best_f:
            best_f, best_k = f, dict(k)
    if pick == "plateau" and best_k is not None:
        best_k = _plateau_indices(groups, cuts, best_k, n_gt)
        best_f = f1_of(best_k)
    elif pick != "max":
        raise ValueError(f"неизвестный выбор порога: {pick}")
    return {"f1": best_f, "n_gt": n_gt,
            "groups": {g: {"threshold": round(float(cuts[g][2][best_k[g]]), 6),
                           "kept": int(cuts[g][0][best_k[g]]), "tp": int(cuts[g][1][best_k[g]])}
                       for g in groups}}


def _groups_ok(c: ClassCurve) -> bool:
    return (not c.pred_group and c.conf.size == 0) or len(c.pred_group) == c.conf.size


def sweep_groups(curves: dict[str, ClassCurve], rounds: int = 20, pick: str = "max") -> dict:
    """Как sweep_groups_reference (см. её описание), но группы — целые коды, а не массив строк."""
    if not _fast_on() or not all(_groups_ok(c) or not c.pred_group for c in curves.values()):
        return sweep_groups_reference(curves, rounds, pick)
    out: dict[str, dict] = {}
    for cls, c in curves.items():
        _, _, gnames, gcodes = _curve_codes(c)
        r = _sweep_one(c.conf, c.hit, gcodes, gnames, c.n_gt, rounds, pick)
        if r is not None:
            out[cls] = r
    return out


def bootstrap(curves: dict[str, ClassCurve], n: int = 200, seed: int = 0) -> dict[str, dict]:
    """Как bootstrap_reference: веса участков берутся индексом, а не списком Python на каждой выборке."""
    if not _fast_on():
        return bootstrap_reference(curves, n, seed)
    regions = sorted({r for c in curves.values() for r in c.pred_region} |
                     {r for c in curves.values() for r in c.gt_regions})
    if not regions or n <= 0:
        return {}
    idx = {r: i for i, r in enumerate(regions)}
    prep = {}
    for cls, c in curves.items():
        rnames, rcodes, _, _ = _curve_codes(c)
        glob = np.array([idx[r] for r in rnames], dtype=np.int64)[rcodes] if rcodes.size else rcodes
        last = np.r_[np.flatnonzero(c.conf[1:] != c.conf[:-1]), c.conf.size - 1] if c.conf.size else None
        prep[cls] = (glob, last)
    rng = np.random.default_rng(seed)
    taus: dict[str, list[float]] = {c: [] for c in curves}
    for _ in range(n):
        w = np.bincount(rng.integers(0, len(regions), len(regions)), minlength=len(regions)).astype(float)
        for cls, c in curves.items():
            n_gt = sum(w[idx[r]] * k for r, k in c.gt_regions.items())
            if n_gt == 0 or c.conf.size == 0:
                continue
            glob, last = prep[cls]
            pw = w[glob]
            tp, nn = np.cumsum(c.hit * pw)[last], np.cumsum(pw)[last]
            taus[cls].append(float(c.conf[last[int(np.argmax(_f1(tp, nn - tp, n_gt - tp)))]]))
    return {cls: {"p10": float(np.percentile(v, 10)), "p50": float(np.median(v)),
                  "p90": float(np.percentile(v, 90)), "samples": len(v)}
            for cls, v in taus.items() if v}


def _scores_matrix(per_class: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]):
    """Балл по выборкам: per_class = [(класс, tp, fp, n_gt)] — векторы по выборкам. Порядок классов как в цикле."""
    num = den = None
    f1s = {}
    for cls, tp, fp, n_gt in per_class:
        f1 = _f1(tp, fp, n_gt - tp)
        ok = n_gt > 0
        f1s[cls] = (f1, ok)
        wc = CLASS_WEIGHTS.get(cls, 0.0)
        if num is None:
            num = np.zeros(tp.shape)
            den = np.zeros(tp.shape)
        num = np.where(ok, num + wc * f1, num)
        den = np.where(ok, den + wc, den)
    return num, den, f1s


def paired_bootstrap(a: dict[str, ClassCurve], b: dict[str, ClassCurve], tau_a: dict[str, float],
                     tau_b: dict[str, float], n: int = 1000, seed: int = 0) -> dict:
    """Как paired_bootstrap_reference: суммы по объектам заменены суммами по участкам × веса (матрично)."""
    if not _fast_on():
        return paired_bootstrap_reference(a, b, tau_a, tau_b, n, seed)
    regions = sorted({r for cs in (a, b) for c in cs.values() for r in c.pred_region} |
                     {r for cs in (a, b) for c in cs.values() for r in c.gt_regions})
    if not regions or n <= 0:
        return {}
    R = len(regions)
    idx = {r: i for i, r in enumerate(regions)}
    rng = np.random.default_rng(seed)
    W = np.empty((n, R), dtype=np.float64)
    for it in range(n):
        W[it] = np.bincount(rng.integers(0, R, R), minlength=R)

    def prep(curves, taus):
        out = {}
        for cls, c in curves.items():
            kept = c.conf >= (taus.get(cls) or 0.0)
            rnames, rcodes, _, _ = _curve_codes(c)
            glob = (np.array([idx[r] for r in rnames], dtype=np.int64)[rcodes] if rcodes.size
                    else np.zeros(0, np.int64))[kept]
            hit = c.hit[kept].astype(bool)
            tp_r = np.bincount(glob[hit], minlength=R).astype(np.float64)
            n_r = np.bincount(glob, minlength=R).astype(np.float64)
            g_r = np.zeros(R)
            for r, k in c.gt_regions.items():
                g_r[idx[r]] += k
            out[cls] = (tp_r, n_r, g_r)
        return out

    pa, pb = prep(a, tau_a), prep(b, tau_b)
    classes = sorted(set(pa) | set(pb))
    res = []
    for prepared in (pa, pb):
        rows = []
        for cls in classes:
            if cls not in prepared:
                continue
            tp_r, n_r, g_r = prepared[cls]
            tp, nn, n_gt = W @ tp_r, W @ n_r, W @ g_r
            rows.append((cls, tp, nn - tp, n_gt))
        num, den, f1s = _scores_matrix(rows)
        if num is None:
            num, den = np.zeros(n), np.zeros(n)
        res.append((np.where(den != 0, num / np.where(den != 0, den, 1.0), 0.0), f1s))
    diffs = res[0][0] - res[1][0]
    per = {}
    for cls in classes:
        if cls in res[0][1] and cls in res[1][1]:
            (fa, oka), (fb, okb) = res[0][1][cls], res[1][1][cls]
            both = oka & okb
            per[cls] = (fa - fb)[both]
        else:
            per[cls] = np.zeros(0)

    def summary(v):
        v = np.asarray(v)
        return {"mean": float(v.mean()), "p5": float(np.percentile(v, 5)), "p50": float(np.median(v)),
                "p95": float(np.percentile(v, 95)), "p_gt0": float((v > 0).mean())}

    return {"score": summary(diffs), "per_class": {c: summary(v) for c, v in per.items() if len(v)}, "samples": n}


def honest_counts(curves: dict[str, ClassCurve], folds: list[set[str]], pick: str = "max",
                  unseen: str = "drop") -> tuple[dict[str, dict[str, list[int]]], dict]:
    """Как honest_counts_reference (см. её описание): маски фолдов и счётчики участков — векторно.

    Порядок ключей в словарях счётчиков тот же (участки эталона, затем участки
    оставленных объектов в порядке убывания уверенности).
    """
    if not _fast_on() or not all(_groups_ok(c) for c in curves.values()):
        return honest_counts_reference(curves, folds, pick, unseen)
    every = set().union(*folds) if folds else set()
    counts: dict[str, dict[str, list[int]]] = {cls: {} for cls in curves}
    info = {"folds": len(folds), "unseen_dropped": 0, "unseen_kept": 0}
    codes = {cls: _curve_codes(c) for cls, c in curves.items()}
    fill = np.inf if unseen == "drop" else -np.inf
    for held in folds:
        train = every - held
        res: dict[str, dict] = {}
        for cls, c in curves.items():
            rnames, rcodes, gnames, gcodes = codes[cls]
            lut = np.fromiter((r in train for r in rnames), dtype=bool, count=len(rnames))
            m = lut[rcodes] if rcodes.size else np.zeros(0, dtype=bool)
            n_gt = int(sum({r: k for r, k in c.gt_regions.items() if r in train}.values()))
            r = _sweep_one(c.conf[m], c.hit[m], gcodes[m], gnames, n_gt, 20, pick)
            if r is not None:
                res[cls] = r
        taus: dict[str, dict[str, float]] = {}
        for cls, v in res.items():
            for g, gv in v["groups"].items():
                taus.setdefault(g, {})[cls] = gv["threshold"]
        for cls, c in curves.items():
            rnames, rcodes, gnames, gcodes = codes[cls]
            per = counts[cls]
            for r, k in c.gt_regions.items():
                if r in held:
                    per.setdefault(r, [0, 0, 0])[2] += int(k)
            lut = np.fromiter((r in held for r in rnames), dtype=bool, count=len(rnames))
            m = lut[rcodes] if rcodes.size else np.zeros(0, dtype=bool)
            if not m.any():
                continue
            has = np.array([cls in taus.get(g, {}) for g in gnames], dtype=bool)
            tval = np.array([taus.get(g, {}).get(cls, 0.0) for g in gnames], dtype=np.float64)
            gh = gcodes[m]
            missing = ~has[gh]
            tau = np.where(missing, fill, tval[gh])
            info["unseen_dropped" if unseen == "drop" else "unseen_kept"] += int(missing.sum())
            kept = c.conf[m] >= tau
            rk = rcodes[m][kept]
            if rk.size == 0:
                continue
            hk = c.hit[m][kept].astype(bool)
            uniq, first = np.unique(rk, return_index=True)
            tpc = np.bincount(rk[hk], minlength=len(rnames))
            fpc = np.bincount(rk[~hk], minlength=len(rnames))
            for code in uniq[np.argsort(first)].tolist():
                lst = per.setdefault(rnames[code], [0, 0, 0])
                lst[0] += int(tpc[code])
                lst[1] += int(fpc[code])
    return counts, info


def paired_bootstrap_counts(a: dict, b: dict, family_of, n: int = 1000, seed: int = 0,
                            regions: set[str] | None = None) -> dict:
    """Как paired_bootstrap_counts_reference: веса семей → веса участков, суммы — матрично."""
    if not _fast_on():
        return paired_bootstrap_counts_reference(a, b, family_of, n, seed, regions)
    regs = {r for cs in (a, b) for by in cs.values() for r in by}
    if regions is not None:
        regs &= regions
    fams: dict[str, list[str]] = {}
    for r in sorted(regs):
        fams.setdefault(family_of(r), []).append(r)
    names = sorted(fams)
    if not names or n <= 0:
        return {}
    rng = np.random.default_rng(seed)
    picks = np.empty((n, len(names)), dtype=np.float64)
    for it in range(n):
        picks[it] = np.bincount(rng.integers(0, len(names), len(names)), minlength=len(names))
    # участки вне regs получают вес 0 (как weights.get(r, 0.0) в score_counts)
    every = sorted({r for cs in (a, b) for by in cs.values() for r in by})
    ridx = {r: i for i, r in enumerate(every)}
    fam_of_r = np.full(len(every), -1, dtype=np.int64)
    for fi, f in enumerate(names):
        for r in fams[f]:
            fam_of_r[ridx[r]] = fi
    W = np.where(fam_of_r >= 0, picks[:, np.clip(fam_of_r, 0, None)], 0.0) if every else np.zeros((n, 0))

    def scores(cs):
        rows = []
        for cls, by_region in cs.items():
            A = np.zeros(len(every)); B = np.zeros(len(every)); G = np.zeros(len(every))
            for r, (x, y, g) in by_region.items():
                i = ridx[r]
                A[i] += x; B[i] += y; G[i] += g
            rows.append((cls, W @ A, W @ B, W @ G))
        num, den, _ = _scores_matrix(rows)
        if num is None:
            return np.zeros(n)
        return np.where(den != 0, num / np.where(den != 0, den, 1.0), 0.0)

    v = scores(a) - scores(b)
    return {"mean": float(v.mean()), "p5": float(np.percentile(v, 5)), "p50": float(np.median(v)),
            "p95": float(np.percentile(v, 95)), "p_gt0": float((v > 0).mean()), "samples": n,
            "families": len(names)}


def apply_thresholds(fc: dict, thresholds: dict[str, float], metric_path: str | None = None) -> dict:
    """Оставляет предсказания с confidence ≥ порога своего класса."""
    norm = load_official(metric_path).normalize_class_name
    feats = [f for f in fc.get("features", [])
             if float((f.get("properties") or {}).get("confidence", 1.0))
             >= (thresholds.get(norm((f.get("properties") or {}).get("class_name"))) or 0.0)]
    return {**{k: v for k, v in fc.items() if k != "features"}, "features": feats}


def verify(pred: dict, gt: dict, thresholds: dict[str, float], metric_path: str | None = None,
           curves: dict[str, ClassCurve] | None = None) -> dict:
    """Сверка: быстрый балл против полного официального расчёта на отфильтрованных предсказаниях."""
    curves = curves or build_curves(pred, gt, metric_path)
    fast, _ = score_at(curves, thresholds)
    official = score(apply_thresholds(pred, thresholds, metric_path), gt, metric_path)["final_score"]
    return {"fast": fast, "official": official, "abs_diff": abs(fast - official)}


def sweep(pred: dict, gt: dict, metric_path: str | None = None, n_boot: int = 0) -> dict:
    """Лучший порог и F1 по каждому классу + итоговый балл с этими порогами.

    Порог ставится посередине между последним оставленным и первым отброшенным
    значением confidence — так он устойчивее переносится на новые данные.
    Классы без эталона пропускаются: по этим данным их порог не определить.
    """
    curves = build_curves(pred, gt, metric_path)
    best: dict[str, dict] = {}
    for cls in CLASSES:
        c = curves.get(cls)
        if c is None or c.n_gt == 0:
            continue
        taus, f1 = curve_table(c)
        k = int(np.argmax(f1))
        if np.isinf(taus[k]):
            tau = float(c.conf.max() + 1e-6) if c.conf.size else 1.0
        else:
            lower = c.conf[c.conf < taus[k]]
            tau = float((taus[k] + lower.max()) / 2) if lower.size else float(taus[k])
        best[cls] = {"threshold": round(tau, 6), "f1": round(float(f1[k]), 4),
                     "f1_all": round(float(f1[-1]), 4), "n_gt": c.n_gt, "n_pred": int(c.conf.size),
                     "curve": [(float(t), round(float(v), 5)) for t, v in zip(taus[1:], f1[1:])]}
    thresholds = {c: v["threshold"] for c, v in best.items()}
    est, _ = score_at(curves, thresholds)
    return {"thresholds": thresholds, "per_class": best, "estimated_score": round(est, 6),
            "bootstrap": bootstrap(curves, n_boot) if n_boot else {}, "_curves": curves}


def _filter(fc: dict, markup: set[str] | None, regions: set[str] | None) -> dict:
    feats = [f for f in fc["features"]
             if (markup is None or (f.get("properties") or {}).get("markup_type") in markup)
             and (regions is None or (f.get("properties") or {}).get("region_name") in regions)]
    return {**fc, "features": feats}


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Официальная метрика и точный подбор порогов")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("score", "sweep"):
        p = sub.add_parser(name)
        p.add_argument("--pred", required=True)
        p.add_argument("--gt", required=True)
        p.add_argument("--metric", default=None)
        p.add_argument("--markup", default=None, help="только эти markup_type, через запятую (например, li)")
        p.add_argument("--regions", default=None, help="файл со списком регионов (по одному в строке)")
        if name == "sweep":
            p.add_argument("--out", default=None, help="куда сохранить пороги (JSON)")
            p.add_argument("--bootstrap", type=int, default=0, help="повторов бутстрепа по регионам")
    p = sub.add_parser("sweep-groups", help="совместные пороги по (группа порога, класс) на общий F1 класса")
    p.add_argument("--pred", required=True)
    p.add_argument("--gt", required=True)
    p.add_argument("--metric", default=None)
    p.add_argument("--regions", default=None)
    p.add_argument("--out", default=None, help="JSON {группа: {класс: порог}} для [thresholds_by_markup]")
    p = sub.add_parser("compare", help="A против B: пороги — оптимум каждой, разница — парный бутстреп по регионам")
    p.add_argument("--pred", required=True, help="A")
    p.add_argument("--pred-b", required=True, help="B")
    p.add_argument("--gt", required=True)
    p.add_argument("--metric", default=None)
    p.add_argument("--markup", default=None)
    p.add_argument("--regions", default=None)
    p.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    pred = json.loads(Path(args.pred).read_text(encoding="utf-8-sig"))
    gt = json.loads(Path(args.gt).read_text(encoding="utf-8-sig"))
    markup = set(args.markup.split(",")) if getattr(args, "markup", None) else None
    regions = ({ln.strip() for ln in Path(args.regions).read_text(encoding="utf-8").splitlines() if ln.strip()}
               if args.regions else None)
    pred, gt = _filter(pred, markup, regions), _filter(gt, markup, regions)
    if args.cmd == "sweep-groups":
        curves = build_curves(pred, gt, args.metric)
        res = sweep_groups(curves)
        taus: dict[str, dict[str, float]] = {}
        for cls, v in res.items():
            print(f"{cls:24s} GT {v['n_gt']:5d}  F1 {v['f1']:.3f}")
            for g, gv in v["groups"].items():
                taus.setdefault(g, {})[cls] = gv["threshold"]
                print(f"    {g:8s} порог {gv['threshold']:.6f}  оставлено {gv['kept']:6d}  TP {gv['tp']:5d}")
        est = official_weighted(res)
        official = score(apply_group_thresholds(pred, taus, args.metric), gt, args.metric)["final_score"]
        print(f"балл по кривым {est:.6f}; официальный расчёт {official:.6f}")
        if args.out:
            Path(args.out).write_text(json.dumps(taus, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    if args.cmd == "compare":
        pred_b = _filter(json.loads(Path(args.pred_b).read_text(encoding="utf-8-sig")), markup, regions)
        ra, rb = sweep(pred, gt, args.metric), sweep(pred_b, gt, args.metric)
        res = paired_bootstrap(ra["_curves"], rb["_curves"], ra["thresholds"], rb["thresholds"], args.boot)
        print(f"балл A {ra['estimated_score']:.4f}, B {rb['estimated_score']:.4f}")
        s = res["score"]
        print(f"A − B: среднее {s['mean']:+.4f}, 90% интервал [{s['p5']:+.4f}; {s['p95']:+.4f}], "
              f"доля выборок с A > B {s['p_gt0']:.2f}")
        for cls, v in res["per_class"].items():
            fa, fb = ra["per_class"][cls]["f1"], rb["per_class"][cls]["f1"]
            print(f"  {cls:24s} F1 {fa:.3f} − {fb:.3f}: [{v['p5']:+.3f}; {v['p95']:+.3f}], A > B в {v['p_gt0']:.2f}")
        return
    if args.cmd == "score":
        res = score(pred, gt, args.metric)
        print(json.dumps({"final_score": res["final_score"],
                          "per_class": {c: {k: v[k] for k in ("f1", "tp", "fp", "fn")}
                                        for c, v in res["per_class"].items()}},
                         ensure_ascii=False, indent=2))
        return
    res = sweep(pred, gt, args.metric, args.bootstrap)
    check = verify(pred, gt, res["thresholds"], args.metric, res["_curves"])
    all_in = score_at(res["_curves"], {c: 0.0 for c in res["_curves"]})[0]
    print(f"{'класс':24s} {'GT':>6s} {'pred':>6s} {'порог':>8s} {'F1':>6s} {'F1 всё':>7s}  бутстреп p10–p50–p90")
    for cls, v in res["per_class"].items():
        b = res["bootstrap"].get(cls)
        bt = f"{b['p10']:.3f}–{b['p50']:.3f}–{b['p90']:.3f}" if b else "—"
        print(f"{cls:24s} {v['n_gt']:6d} {v['n_pred']:6d} {v['threshold']:8.4f} {v['f1']:6.3f} {v['f1_all']:7.3f}  {bt}")
    print(f"балл при порогах {res['estimated_score']:.6f}; официальный расчёт {check['official']:.6f} "
          f"(расхождение {check['abs_diff']:.1e}); без порогов {all_in:.6f}")
    if args.out:
        Path(args.out).write_text(json.dumps(res["thresholds"], indent=2), encoding="utf-8")
        print(f"пороги → {args.out} (перенесите в configs/default.toml, секция [thresholds])")


if __name__ == "__main__":
    _cli()
