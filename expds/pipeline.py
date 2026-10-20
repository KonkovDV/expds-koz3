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

from . import accel, candidates, fusion, geophys, post, writer
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


def model_specs(models: dict, markup: str) -> list[tuple[str, list[str] | None]]:
    """Модели модальности из [models]: путь, таблица {path, classes} или их список.

    classes ограничивает выход модели, чтобы две модели одной модальности
    (например, мелкая и крупная для городищ) не выдавали один класс дважды.
    """
    v = models.get(markup)
    out = []
    for it in v if isinstance(v, list) else ([v] if v else []):
        if isinstance(it, str):
            out.append((it, None))
        elif isinstance(it, dict) and it.get("path"):
            out.append((it["path"], list(it.get("classes") or []) or None))
    return out


def _segmenters(models: dict, markup: str, root: Path, tta: int = 1) -> list:
    from .infer_nn import Segmenter

    segs = []
    for path, classes in model_specs(models, markup):
        p = Path(path)
        p = p if p.is_absolute() else root / p
        if not p.exists():
            log.warning("модель %s не найдена: %s", markup, p)
            continue
        seg = Segmenter(p)
        seg.keep = set(classes) if classes else None
        seg.tta = tta
        segs.append(seg)
    return segs


def _cached_segmenters(seg_cache: dict, cfg: dict, markup: str, root: Path) -> list:
    """Модели модальности загружаются один раз за прогон."""
    if markup not in seg_cache:
        seg_cache[markup] = _segmenters(cfg.get("models", {}), markup, root,
                                        int(cfg.get("tta", {}).get(markup, 1)))
    return seg_cache[markup]


def _kept(seg, feats: list[dict]) -> list[dict]:
    keep = getattr(seg, "keep", None)
    return feats if keep is None else [f for f in feats if f["class_name"] in keep]


def run_region(reg: Region, cfg: dict, guard: TimeGuard, seg_cache: dict, root: Path) -> list[dict]:
    """Все предсказания одного региона."""
    from .lidar import build_dem, dem_params, hillshade_set

    feats: list[dict] = []
    crs_region = read_utm_json(reg.utm_json)
    cand_p = _params(candidates.CandidateParams, cfg.get("candidates"))
    kurgan_model = cfg.get("candidates", {}).get("lgbm_model")

    # 1. Лидар
    for las in reg.las:
        if not guard.ok(cfg.get("time", {}).get("reserve_s", 120)):
            log.warning("время на исходе: пропускаю %s", las.name)
            break
        try:
            segs = _cached_segmenters(seg_cache, cfg, "li", root)
            dems: dict[tuple, object] = {}

            def dem_for(meta):
                dp = dem_params(meta, cfg)
                key = (dp["res"], dp["fill"])
                if key not in dems:
                    dems[key] = build_dem(las, res=dp["res"], fill=dp["fill"], crs=crs_region)
                return dems[key]

            d = dem_for(segs[0].meta if segs else None)
            if d.crs is None:
                log.warning("%s: нет CRS (ни UTM.json, ни WKT в LAS) — пропуск", las.name)
                continue
            if segs:
                rescore = cfg.get("li", {}).get("mode", "rescore") == "rescore"
                for seg in segs:
                    feats += _kept(seg, _li_with_model(seg, dem_for(seg.meta), reg.name, las.name, hillshade_set,
                                                       cand_p if rescore else None))
            else:
                c = candidates.detect(d, cand_p)
                s = candidates.score(c["features"], root / kurgan_model if kurgan_model else None)
                feats += [dict(f, threshold_group="li_cpu")
                          for f in candidates.to_features(c, s, d.crs, reg.name, cand_p, source=las.name)]
        except Exception as exc:
            log.exception("%s: ошибка лидара (%s)", las.name, exc)

    fp = _params(fusion.FusionParams, cfg.get("fusion"))
    if fp.li_to_ae and reg.rasters.get("ae"):
        try:
            feats += fusion.li_to_ae(feats, reg.rasters["ae"], fp)
        except Exception as exc:
            log.exception("%s: ошибка слияния li→ae (%s)", reg.name, exc)

    # 2. Геофизика: параметры и порог — по модальности; маски extra/ — после детекторов
    geo_cfg = cfg.get("geophys", {})
    for mt in ("maggr", "etgr", "terfl"):
        files = geophys.select_rasters(reg.rasters.get(mt, []), mt)
        if not files:
            continue
        classes = ("finds_points", "object_poly") if mt == "maggr" else ("object_poly",)
        opts = {c: geophys.params_for(geo_cfg, mt, c) for c in classes}
        boundary = geophys.load_mask(reg.boundaries, mt, crs_region) if mt != "etgr" else None
        noise = geophys.load_mask(reg.noise, mt, crs_region)
        for path in files:
            if not guard.ok(30):
                break
            try:
                for cls in classes:
                    p, o = opts[cls]
                    found = (geophys.finds_points(path, reg.name, mt, p) if cls == "finds_points"
                             else geophys.object_polys(path, reg.name, mt, p))
                    if o["rank"]:
                        found = geophys.rank_within_source(found)
                    found = geophys.apply_masks(found, boundary, noise if o["use_noise"] else None)
                    feats += [f for f in found if f["confidence"] >= o["min_confidence"]]
            except Exception as exc:
                log.exception("%s: ошибка геофизики (%s)", path.name, exc)

    # 3. Оптика — только обученными моделями
    for mt in ("ae", "sp_or", "or"):
        if not reg.rasters.get(mt):
            continue
        for seg in _cached_segmenters(seg_cache, cfg, mt, root):
            for path in reg.rasters.get(mt, []):
                if not guard.ok(60):
                    break
                try:
                    feats += _kept(seg, _raster_with_model(seg, path, reg.name, mt))
                except Exception as exc:
                    log.exception("%s: ошибка модели %s (%s)", path.name, mt, exc)
    return feats


def _li_with_model(seg, d, region: str, source: str, hillshade_set, cand_p=None) -> list[dict]:
    """Каналы из DEM на рабочем шаге модели → вероятности → объекты.

    С cand_p курганы выдаются по кандидатам-пикам рельефа с уверенностью модели
    в их центрах: пики разделяют соседние насыпи, которые маска сливает в один
    объект (а слитый полигон метрика не засчитывает). Остальные классы — масками.
    """
    from rasterio.transform import from_origin

    from .candidates import downsample_mean
    from .infer_nn import probs_to_features

    if seg.meta.get("li_source") == "las":
        from .lidar import relief_channels

        img, tr = relief_channels(d, seg.res, tuple(seg.meta["li_tags"]))
        probs = seg.predict(img)
        keep = getattr(seg, "keep", None)
        out_classes = [c for c in seg.classes if keep is None or c in keep]
        kur = [c for c in ("kurgany_povrezhdennye", "kurgany_tselye") if c in out_classes] if cand_p else []
        rest = [c for c in out_classes if c not in kur]
        feats = probs_to_features(probs[[seg.classes.index(c) for c in rest]], rest, tr, d.crs, tr.a,
                                  region, "li", source) if rest else []
        if kur:
            c = candidates.detect(d, cand_p)
            s = candidates.sample_probs(c, probs[[seg.classes.index(k) for k in kur]], tr)
            scores = np.column_stack([np.zeros(len(s), np.float32), s])
            feats += candidates.to_features(c, scores, d.crs, region, cand_p, classes=tuple(kur), source=source)
        return feats
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


def _raster_with_model(seg, path: Path, region: str, markup: str, block: int = 4096,
                       overlap: int = 256) -> list[dict]:
    """Растр на рабочем шаге модели окнами block×block с перекрытием → объекты (память — одно окно)."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import Window
    from shapely.geometry import box

    from .crs import WEB_MERCATOR, to_crs
    from .infer_nn import probs_to_features
    from .tiles import MAX_UPSAMPLE, band_stats, fix_channels, to_uint8, working_res

    stats = band_stats(str(path), markup)
    half = overlap // 2
    feats: list[dict] = []
    with rasterio.open(path) as ds:
        r = working_res(abs(ds.res[0]), seg.res, seg.meta.get("max_upsample") or MAX_UPSAMPLE)
        scale = r / abs(ds.res[0])
        w, h = max(1, int(ds.width / scale)), max(1, int(ds.height / scale))
        n = min(ds.count, 3)
        crs = ds.crs.to_string() if ds.crs else None
        step = block - overlap
        xs = list(range(0, max(w - overlap, 1), step))
        ys = list(range(0, max(h - overlap, 1), step))
        for iy, y0 in enumerate(ys):
            for ix, x0 in enumerate(xs):
                bw, bh = min(block, w - x0), min(block, h - y0)
                win = Window(x0 * scale, y0 * scale, bw * scale, bh * scale)
                a = ds.read(list(range(1, n + 1)), window=win, out_shape=(n, bh, bw),
                            resampling=Resampling.average if scale > 1 else Resampling.bilinear)
                if not a.any():
                    continue
                tr = ds.window_transform(win) @ rasterio.Affine.scale(win.width / bw, win.height / bh)
                img = fix_channels(to_uint8(a, markup, stats), int(seg.meta["in_channels"]))
                found = probs_to_features(seg.predict(img), seg.classes, tr, crs, r, region, markup, path.name)
                # середина окна: соседние окна делят перекрытие пополам, объект попадает ровно в одно
                c0 = (half if ix > 0 else 0, half if iy > 0 else 0)
                c1 = (bw - (half if ix < len(xs) - 1 else 0), bh - (half if iy < len(ys) - 1 else 0))
                (ax, ay), (bx, by) = tr @ c0, tr @ c1
                core = box(min(ax, bx), min(ay, by), max(ax, bx), max(ay, by))
                core = to_crs(core, crs, WEB_MERCATOR) if crs else core
                feats += [f for f in found if core.contains(f["geom"].representative_point())]
    return feats


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
        out = post.apply_thresholds(out, cfg.get("thresholds", {}), cfg.get("default_threshold", 0.15),
                                    cfg.get("thresholds_by_markup", {}))
        out = post.merge_within_markup(out)
        out = post.finalize(out)
        out = post.cap_features(out)
        writer.write_geojson(out, output_path)
        summary.update(written=len(out), seconds=round(guard.elapsed(), 1))
        log.info("записано %d объектов в %s за %.1f с", len(out), output_path, guard.elapsed())
    return summary
