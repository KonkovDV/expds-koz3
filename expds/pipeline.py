"""Сквозной инференс: папка валидационной выборки → result.geojson.

Порядок (приоритет — по весу классов и надёжности):
1. лидар: LAS → DEM → (модель li, если есть) иначе CPU-кандидаты курганов;
2. геофизика: классика (finds_points, object_poly) + модель, если есть;
3. оптика (ae, sp_or, or): только при наличии модели.
Бюджет времени контролируется между шагами: при нехватке тяжёлые шаги
пропускаются, но файл с найденным к этому моменту записывается всегда.
"""
from __future__ import annotations

import logging
import time
import tomllib
from dataclasses import fields
from pathlib import Path

import numpy as np

from . import accel, candidates, geophys, post, writer
from .catalog import Region, read_utm_json, scan_root
from .constants import DEFAULT_BUDGET_S

log = logging.getLogger(__name__)


class TimeGuard:
    """Учёт времени: сколько прошло и хватает ли запаса на следующий шаг."""

    def __init__(self, budget_s: float):
        self.t0 = time.monotonic()
        self.budget = budget_s

    def elapsed(self) -> float:
        return time.monotonic() - self.t0

    def left(self) -> float:
        return self.budget - self.elapsed()

    def ok(self, need_s: float = 0.0) -> bool:
        return self.left() > need_s


def load_config(path: str | Path | None) -> dict:
    """Конфиг TOML (configs/default.toml); отсутствующий файл → пустой словарь."""
    if path and Path(path).exists():
        with open(path, "rb") as f:
            return tomllib.load(f)
    return {}


def _params(cls, cfg: dict):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in (cfg or {}).items() if k in names})


def _segmenter(models: dict, markup: str, root: Path):
    path = models.get(markup)
    if not path:
        return None
    p = Path(path)
    p = p if p.is_absolute() else root / p
    if not p.exists():
        log.warning("модель %s не найдена: %s", markup, p)
        return None
    from .infer_nn import Segmenter

    return Segmenter(p)


def run_region(reg: Region, cfg: dict, guard: TimeGuard, seg_cache: dict, root: Path) -> list[dict]:
    """Все предсказания одного региона."""
    from .lidar import build_dem, hillshade_set

    feats: list[dict] = []
    crs_region = read_utm_json(reg.utm_json)
    cand_p = _params(candidates.CandidateParams, cfg.get("candidates"))
    geo_p = _params(geophys.GeoParams, cfg.get("geophys"))
    models = cfg.get("models", {})
    kurgan_model = cfg.get("candidates", {}).get("lgbm_model")

    # 1. Лидар
    for las in reg.las:
        if not guard.ok(cfg.get("time", {}).get("reserve_s", 120)):
            log.warning("время на исходе: пропускаю %s", las.name)
            break
        try:
            d = build_dem(las, crs=crs_region)
            if d.crs is None:
                log.warning("%s: нет CRS (ни UTM.json, ни WKT в LAS) — пропуск", las.name)
                continue
            seg = seg_cache.setdefault("li", _segmenter(models, "li", root))
            if seg is not None:
                feats += _li_with_model(seg, d, reg.name, las.name, hillshade_set)
            else:
                c = candidates.detect(d, cand_p)
                s = candidates.score(c["features"], root / kurgan_model if kurgan_model else None)
                feats += candidates.to_features(c, s, d.crs, reg.name, cand_p, source=las.name)
        except Exception as exc:
            log.exception("%s: ошибка лидара (%s)", las.name, exc)

    # 2. Геофизика
    for mt in ("maggr", "etgr", "terfl"):
        for path in reg.rasters.get(mt, []):
            if not guard.ok(30):
                break
            try:
                if mt == "maggr":
                    feats += geophys.finds_points(path, reg.name, mt, geo_p)
                feats += geophys.object_polys(path, reg.name, mt, geo_p)
            except Exception as exc:
                log.exception("%s: ошибка геофизики (%s)", path.name, exc)

    # 3. Оптика — только обученными моделями
    for mt in ("ae", "sp_or", "or"):
        seg = seg_cache.setdefault(mt, _segmenter(models, mt, root))
        if seg is None:
            continue
        for path in reg.rasters.get(mt, []):
            if not guard.ok(60):
                break
            try:
                feats += _raster_with_model(seg, path, reg.name, mt)
            except Exception as exc:
                log.exception("%s: ошибка модели %s (%s)", path.name, mt, exc)
    return feats


def _li_with_model(seg, d, region: str, source: str, hillshade_set) -> list[dict]:
    """Отмывки из DEM → рабочий шаг модели → вероятности → объекты."""
    from rasterio.transform import from_origin

    from .candidates import downsample_mean
    from .infer_nn import probs_to_features

    tags = seg.meta.get("li_tags") or ["g"]
    images = hillshade_set(d, tags=tuple(tags))
    k = max(1, int(round(seg.res / d.res)))
    chans = []
    for t in tags:
        a = images[t] if images[t].ndim == 3 else images[t][None]
        chans += [downsample_mean(a[i].astype(np.float32), k) for i in range(a.shape[0])]
    img = np.clip(np.nan_to_num(np.stack(chans), nan=0), 0, 255).astype(np.uint8)
    probs = seg.predict(img)
    tr = from_origin(d.left, d.top, d.res * k, d.res * k)
    return probs_to_features(probs, seg.classes, tr, d.crs, d.res * k, region, "li", source)


def _raster_with_model(seg, path: Path, region: str, markup: str) -> list[dict]:
    """Растр целиком на рабочем шаге модели → объекты."""
    import rasterio
    from rasterio.enums import Resampling

    from .infer_nn import probs_to_features
    from .tiles import fix_channels, to_uint8

    with rasterio.open(path) as ds:
        scale = seg.res / abs(ds.res[0])
        h, w = max(1, int(ds.height / scale)), max(1, int(ds.width / scale))
        n = min(ds.count, 3)
        a = ds.read(list(range(1, n + 1)), out_shape=(n, h, w),
                    resampling=Resampling.average if scale > 1 else Resampling.bilinear)
        tr = ds.transform * rasterio.Affine.scale(ds.width / w, ds.height / h)
        crs = ds.crs.to_string() if ds.crs else None
    img = fix_channels(to_uint8(a, markup), int(seg.meta["in_channels"]))
    probs = seg.predict(img)
    return probs_to_features(probs, seg.classes, tr, crs, seg.res, region, markup, path.name)


def run(input_dir: str | Path, output_path: str | Path, config_path: str | Path | None = None,
        root: str | Path | None = None) -> dict:
    """Полный прогон; файл результата записывается в любом случае."""
    root = Path(root or Path(__file__).resolve().parents[1])
    cfg = load_config(config_path or root / "configs" / "default.toml")
    guard = TimeGuard(cfg.get("time", {}).get("budget_s", DEFAULT_BUDGET_S))
    log.info("ускоритель: %s", accel.backend())
    feats: list[dict] = []
    seg_cache: dict = {}
    summary = {"regions": 0, "raw": 0}
    try:
        regions = scan_root(input_dir)
        summary["regions"] = len(regions)
        # Сначала регионы с LAS: лидар даёт основную часть балла.
        regions.sort(key=lambda r: (not r.las, r.name))
        for reg in regions:
            if not guard.ok(cfg.get("time", {}).get("reserve_s", 120)):
                log.warning("бюджет времени исчерпан, остальные регионы пропущены")
                break
            t = time.monotonic()
            feats += run_region(reg, cfg, guard, seg_cache, root)
            log.info("%s: объектов %d, %.1f с, прошло %.0f с", reg.name, len(feats),
                     time.monotonic() - t, guard.elapsed())
    finally:
        summary["raw"] = len(feats)
        out = post.only_known_classes(feats)
        out = post.apply_thresholds(out, cfg.get("thresholds", {}), cfg.get("default_threshold", 0.15))
        out = post.merge_within_markup(out)
        out = post.finalize(out)
        out = post.cap_features(out)
        writer.write_geojson(out, output_path)
        summary.update(written=len(out), seconds=round(guard.elapsed(), 1))
        log.info("записано %d объектов в %s за %.1f с", len(out), output_path, guard.elapsed())
    return summary
