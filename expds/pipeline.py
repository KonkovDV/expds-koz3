"""Сквозной инференс: папка валидационной выборки → result.geojson.

Порядок по всему набору, а не внутри участка. Иначе облака LAS съедают
30 минут, и геофизика (вес 20) с оптикой не запускаются:

1. геофизика по всем участкам — растры маленькие;
2. лидар: LAS → DEM → модель, иначе CPU-кандидаты; участки с LAS первыми;
3. оптика (ae, sp_or, or) на оставшемся времени, только при наличии модели.

Файл с тем, что уже найдено, записывается всегда.
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


def model_specs(models: dict, markup: str) -> list[tuple[str | list[str], list[str] | None]]:
    """Модели модальности из [models]: путь, таблица {path, classes} или их список.

    classes ограничивает выход модели, чтобы две модели одной модальности
    (например, мелкая и крупная для городищ) не выдавали один класс дважды.
    path может быть списком: это ансамбль, вероятности моделей усредняются.
    """
    v = models.get(markup)
    out = []
    for it in v if isinstance(v, list) else ([v] if v else []):
        if isinstance(it, str):
            out.append((it, None))
        elif isinstance(it, dict) and it.get("path"):
            out.append((it["path"], list(it.get("classes") or []) or None))
    return out


def model_files(path: str | list[str]) -> list[str]:
    """Файлы весов одной записи model_specs."""
    return list(path) if isinstance(path, list) else [path]


def _segmenters(models: dict, markup: str, root: Path, tta: int = 1) -> list:
    from .infer_nn import Ensemble, Segmenter

    segs = []
    for path, classes in model_specs(models, markup):
        files = [Path(f) if Path(f).is_absolute() else root / f for f in model_files(path)]
        missing = [f for f in files if not f.exists()]
        if missing:
            log.warning("модель %s не найдена: %s", markup, ", ".join(map(str, missing)))
            continue
        members = [Segmenter(f) for f in files]
        seg = members[0] if len(members) == 1 else Ensemble(members)
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


def geophys_feats(reg: Region, cfg: dict, guard: TimeGuard) -> list[dict]:
    """Классические finds_points и object_poly. Маски extra/ применяются после детектора."""
    feats: list[dict] = []
    crs_region = read_utm_json(reg.utm_json)
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
            if not guard.ok(5):
                log.warning("время на исходе: геофизика %s пропущена", path.name)
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
    return feats


def cpu_kurgans(d, region: str, source: str, cfg: dict, root: Path, cand_p) -> list[dict]:
    """Пики рельефа и LightGBM. Включается, когда модели нет или она упала на этом LAS."""
    kurgan_model = cfg.get("candidates", {}).get("lgbm_model")
    found = candidates.detect(d, cand_p)
    scores = candidates.score(found["features"], root / kurgan_model if kurgan_model else None)
    return [dict(f, threshold_group="li_cpu")
            for f in candidates.to_features(found, scores, d.crs, region, cand_p, source=source)]


def lidar_feats(reg: Region, cfg: dict, guard: TimeGuard, seg_cache: dict, root: Path,
                prefetch=None) -> list[dict]:
    """LAS → объекты лидара и, где есть аэрофото, копии курганов в слой ae.

    Сбой загрузки или инференса модели не оставляет участок без курганов:
    для этого LAS считается запасной детектор.
    """
    from .lidar import build_dem, dem_params, hillshade_set

    feats: list[dict] = []
    mosaic_kurgans: list[dict] = []
    crs_region = read_utm_json(reg.utm_json)
    cand_p = _params(candidates.CandidateParams, cfg.get("candidates"))
    for las in reg.las:
        if not guard.ok(cfg.get("time", {}).get("reserve_s", 120)):
            log.warning("время на исходе: пропускаю %s", las.name)
            break
        try:
            segs = _cached_segmenters(seg_cache, cfg, "li", root)
        except Exception:
            log.exception("модель li не загрузилась, дальше запасной детектор")
            seg_cache["li"] = []
            segs = []
        dems: dict[tuple, object] = {}

        def dem_for(meta, _las=las):
            dp = dem_params(meta, cfg)
            key = (dp["res"], dp["fill"])
            if key not in dems:
                dems[key] = (prefetch.get(_las, dp["res"], dp["fill"], crs_region) if prefetch is not None
                             else build_dem(_las, res=dp["res"], fill=dp["fill"], crs=crs_region))
            return dems[key]

        produced = False
        if segs:
            try:
                from .lidar import DEFAULTS, build_dem_mosaic, would_coarsen

                dp0 = dem_params(segs[0].meta, cfg)
                split = (bool(cfg.get("lidar", {}).get("mosaic", False))
                         and would_coarsen(las, dp0["res"], DEFAULTS["max_pixels"]))
                if split:
                    d = (prefetch.get(las, dp0["res"], dp0["fill"], crs_region) if prefetch is not None
                         else build_dem(las, res=dp0["res"], fill=dp0["fill"], crs=crs_region))
                else:
                    d = dem_for(segs[0].meta)
                if d.crs is None:
                    log.warning("%s: нет CRS (ни UTM.json, ни WKT в LAS) — пропуск", las.name)
                    continue
                rescore = cfg.get("li", {}).get("mode", "rescore") == "rescore"
                group = lidar_threshold_group(d.info)
                floors = output_floors(cfg, group or "li")
                kur = {"kurgany_povrezhdennye", "kurgany_tselye"}
                for seg in segs:
                    found = _kept(seg, _li_with_model(seg, d, reg.name, las.name, hillshade_set,
                                                      cand_p if rescore else None, cfg.get("li"), floors))
                    if split:
                        found = [f for f in found if f.get("class_name") not in kur]
                    if group:
                        for f in found:
                            if not f.get("threshold_group"):
                                f["threshold_group"] = group
                    feats += found
                if split:
                    try:
                        mosaic = build_dem_mosaic(las, res=dp0["res"], fill=dp0["fill"], crs=crs_region)
                    except Exception:
                        log.exception("%s: мозаика не собралась, курганы с грубой сетки", las.name)
                        mosaic = None
                    fresh: list[dict] = []
                    if mosaic is not None and mosaic.crs is not None:
                        for seg in segs:
                            found = _li_with_model(seg, mosaic, reg.name, las.name, hillshade_set,
                                                   cand_p if rescore else None, cfg.get("li"), None)
                            fresh += [f for f in found if f.get("class_name") in kur]
                    else:
                        for seg in segs:
                            found = _kept(seg, _li_with_model(seg, d, reg.name, las.name, hillshade_set,
                                                              cand_p if rescore else None, cfg.get("li"), floors))
                            found = [f for f in found if f.get("class_name") in kur]
                            if group:
                                for f in found:
                                    if not f.get("threshold_group"):
                                        f["threshold_group"] = group
                            fresh = found
                    mosaic_kurgans += [f for f in fresh if not f.get("threshold_group")]
                    feats += fresh
                produced = True
            except Exception:
                log.exception("%s: инференс li упал, запасной детектор", las.name)
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
        if produced:
            continue
        try:
            d = dem_for(None)
        except Exception:
            log.exception("%s: DEM для запасного детектора не построен", las.name)
            continue
        if d.crs is None:
            log.warning("%s: нет CRS — пропуск", las.name)
            continue
        try:
            feats += cpu_kurgans(d, reg.name, las.name, cfg, root, cand_p)
        except Exception:
            log.exception("%s: запасной детектор упал", las.name)

    fp = _params(fusion.FusionParams, cfg.get("fusion"))
    if fp.li_to_ae and reg.rasters.get("ae"):
        try:
            feats += fusion.li_to_ae(feats, reg.rasters["ae"], fp)
        except Exception as exc:
            log.exception("%s: ошибка слияния li→ae (%s)", reg.name, exc)
    for f in mosaic_kurgans:
        if f.get("markup_type") == "li" and not f.get("threshold_group"):
            f["threshold_group"] = "li_mosaic"
    return feats


def optical_feats(reg: Region, cfg: dict, guard: TimeGuard, seg_cache: dict, root: Path) -> list[dict]:
    """Обученные модели ae, sp_or и or. Без файла весов модальность пропускается."""
    feats: list[dict] = []
    for mt in ("ae", "sp_or", "or"):
        if not reg.rasters.get(mt):
            continue
        try:
            segs = _cached_segmenters(seg_cache, cfg, mt, root)
        except Exception:
            log.exception("модель %s не загрузилась, модальность пропущена", mt)
            seg_cache[mt] = []
            continue
        for seg in segs:
            for path in reg.rasters.get(mt, []):
                if not guard.ok(60):
                    break
                try:
                    feats += _kept(seg, _raster_with_model(seg, path, reg.name, mt,
                                                           min_conf=output_floors(cfg, mt)))
                except Exception as exc:
                    log.exception("%s: ошибка модели %s (%s)", path.name, mt, exc)
    return feats


def run_region(reg: Region, cfg: dict, guard: TimeGuard, seg_cache: dict, root: Path,
               prefetch=None) -> list[dict]:
    """Один участок целиком: лидар, затем геофизика и оптика. Для прогона набора — три прохода в run()."""
    return (lidar_feats(reg, cfg, guard, seg_cache, root, prefetch)
            + geophys_feats(reg, cfg, guard) + optical_feats(reg, cfg, guard, seg_cache, root))


def lidar_threshold_group(info: dict | None) -> str | None:
    """Группа порога этого LAS: li_coarse, если шаг DEM увеличен (облако не влезло в 60 млн ячеек)."""
    try:
        factor = int((info or {}).get("factor") or 1)
    except (TypeError, ValueError):
        return None
    return "li_coarse" if factor > 1 else None


def output_floors(cfg: dict, markup: str) -> dict[str, float]:
    """Пороги выдачи классов для объектов этого markup_type — те же, что потом применит post.apply_thresholds."""
    from .constants import CLASSES

    thr = cfg.get("thresholds", {})
    own = cfg.get("thresholds_by_markup", {}).get(markup, {})
    default = cfg.get("default_threshold", 0.15)
    return {c: float(own.get(c, thr.get(c, default))) for c in CLASSES}


def _li_with_model(seg, d, region: str, source: str, hillshade_set, cand_p=None,
                   li_cfg: dict | None = None, min_conf: dict[str, float] | None = None) -> list[dict]:
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
                                  region, "li", source, min_conf=min_conf) if rest else []
        if kur:
            c = candidates.detect(d, cand_p)
            kp = probs[[seg.classes.index(k) for k in kur]]
            lc = li_cfg or {}
            if lc.get("nn_peaks", False):
                c = candidates.add_prob_peaks(c, kp.max(axis=0), tr, lc.get("nn_peak_min", 0.3),
                                              lc.get("nn_peak_radius_m", 2.5), lc.get("nn_peak_dedupe_m", 2.0))
            s = candidates.sample_probs(c, kp, tr)
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
    return probs_to_features(probs, seg.classes, tr, d.crs, d.res * k, region, "li", source, min_conf=min_conf)


def _raster_with_model(seg, path: Path, region: str, markup: str, block: int = 4096,
                       overlap: int = 256, min_conf: dict[str, float] | None = None) -> list[dict]:
    """Растр на рабочем шаге модели окнами block×block с перекрытием → объекты (память — одно окно)."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import Window
    from shapely.geometry import box

    from .crs import WEB_MERCATOR, to_crs
    from .infer_nn import probs_to_features
    from .tiles import MAX_UPSAMPLE, band_stats, fix_channels, to_uint8, working_res

    as_centers = seg.meta.get("target") == "centers"

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
                probs = seg.predict(img)
                if as_centers:
                    from .centers import centers_to_features

                    found = centers_to_features(
                        probs, seg.classes, tr, crs, r, region, markup, path.name,
                        radius_m=float(seg.meta.get("center_radius_m") or 6.0),
                        min_dist_m=float(12.0 if seg.meta.get("min_dist_m") is None else seg.meta["min_dist_m"]),
                        min_conf=min_conf)
                else:
                    found = probs_to_features(probs, seg.classes, tr, crs, r, region, markup, path.name,
                                              min_conf=min_conf)
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
    """Полный прогон. Контрольные записи — после геофизики, после лидара и по сторожу времени."""
    from .watchdog import RunState, arm_platform, checkpoint

    root = Path(root or Path(__file__).resolve().parents[1])
    cfg = load_config(config_path or root / "configs" / "default.toml")
    infer = cfg.get("infer", {})
    try:
        from .infer_nn import configure_runtime
        configure_runtime(bool(infer.get("amp", False)), bool(infer.get("tf32", False)),
                          bool(infer.get("deterministic", True)))
    except Exception:
        log.exception("точность видеокарты не настроена, дальше без моделей torch")
    guard = TimeGuard(cfg.get("time", {}).get("budget_s", DEFAULT_BUDGET_S))
    log.info("ускоритель: %s", accel.backend())
    state = RunState(Path(output_path), cfg)
    if cfg.get("time", {}).get("watchdog", True):
        margin = cfg.get("time", {}).get("guard_margin_s", 180)
        arm_platform(state, margin_s=float(margin))
    seg_cache: dict = {}
    summary = {"regions": 0, "raw": 0}
    prefetch = None
    try:
        from .lidar import DemPrefetch

        regions = scan_root(input_dir)
        summary["regions"] = len(regions)
        regions.sort(key=lambda r: (not r.las, r.name))
        depth = int(cfg.get("lidar", {}).get("prefetch", 2))
        if depth > 0:
            prefetch = DemPrefetch([(las, read_utm_json(r.utm_json)) for r in regions for las in r.las], depth)
        for reg in regions:
            found = geophys_feats(reg, cfg, guard)
            state.add(found)
            if found:
                log.info("%s: геофизика %+d, прошло %.0f с", reg.name, len(found), guard.elapsed())
        checkpoint(state, "геофизика")
        reserve = cfg.get("time", {}).get("reserve_s", 120)
        for reg in regions:
            if not guard.ok(reserve):
                log.warning("бюджет времени исчерпан, оставшийся лидар пропущен")
                break
            t = time.monotonic()
            found = lidar_feats(reg, cfg, guard, seg_cache, root, prefetch)
            state.add(found)
            log.info("%s: лидар %+d, %.1f с, прошло %.0f с", reg.name, len(found),
                     time.monotonic() - t, guard.elapsed())
        checkpoint(state, "лидар")
        for reg in regions:
            if not guard.ok(60):
                log.warning("на оптику времени не осталось")
                break
            found = optical_feats(reg, cfg, guard, seg_cache, root)
            state.add(found)
            if found:
                log.info("%s: оптика %+d, прошло %.0f с", reg.name, len(found), guard.elapsed())
    finally:
        if prefetch is not None:
            prefetch.close()
        summary["raw"] = len(state.snapshot())
        summary["written"] = checkpoint(state, "конец")
        summary["seconds"] = round(guard.elapsed(), 1)
        log.info("записано %d объектов в %s за %.1f с", summary["written"], output_path, guard.elapsed())
    return summary
