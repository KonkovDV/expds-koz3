"""Сквозной инференс: папка валидационной выборки → result.geojson.

Порядок по всему набору, а не внутри участка. Иначе облака LAS съедают
30 минут, и геофизика (вес 20) с оптикой не запускаются:

1. геофизика по всем участкам — растры маленькие;
2. лидар: LAS → DEM → модель, иначе CPU-кандидаты; участки с LAS первыми;
3. оптика (ae, sp_or, or) на оставшемся времени, только при наличии модели.

Файл результата записывается в конце и сторожем времени.
"""
from __future__ import annotations

import hashlib
import logging
import signal
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


def model_ensemble_only(models: dict, markup: str) -> list[list[str] | None]:
    """Классы, по которым усредняется ансамбль. Нет ключа — среднее по всем классам."""
    v = models.get(markup)
    out = []
    for it in v if isinstance(v, list) else ([v] if v else []):
        out.append(list(it["ensemble_classes"]) if isinstance(it, dict) and it.get("ensemble_classes") else None)
    return out


def model_files(path: str | list[str]) -> list[str]:
    """Файлы весов одной записи model_specs."""
    return list(path) if isinstance(path, list) else [path]


def _segmenters(models: dict, markup: str, root: Path, tta: int = 1) -> list:
    from .infer_nn import Ensemble, Segmenter

    segs = []
    for (path, classes), only in zip(model_specs(models, markup), model_ensemble_only(models, markup)):
        files = [Path(f) if Path(f).is_absolute() else root / f for f in model_files(path)]
        missing = [f for f in files if not f.exists()]
        if missing:
            log.warning("модель %s не найдена: %s", markup, ", ".join(map(str, missing)))
            continue
        members = [Segmenter(f) for f in files]
        seg = members[0] if len(members) == 1 else Ensemble(members, only=only)
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


def region_crs(reg: Region, cfg: dict | None = None) -> str | None:
    """CRS участка. Без [crs] policy это UTM.json, как в замере 0,353793."""
    from .crs import resolve_region_crs

    policy = ((cfg or {}).get("crs") or {}).get("policy", "utm_json")
    utm = read_utm_json(getattr(reg, "utm_json", None))
    if policy == "utm_json":
        return utm
    rasters = [p for files in reg.rasters.values() for p in files]
    return resolve_region_crs(utm, list(reg.las), rasters, policy)


def _with_copies(feats: list[dict], reg: Region, cfg: dict) -> list[dict]:
    rules = (cfg.get("fusion") or {}).get("copy") or []
    if not rules:
        return feats
    try:
        return feats + fusion.copy_layers(feats, reg.rasters, rules, fallback_crs=region_crs(reg, cfg))
    except Exception:
        log.exception("%s: копии слоёв пропущены", reg.name)
        return feats


def geophys_feats(reg: Region, cfg: dict, guard: TimeGuard) -> list[dict]:
    """Классические finds_points и object_poly. Маски extra/ применяются после детектора."""
    feats: list[dict] = []
    crs_region = region_crs(reg, cfg)
    geo_cfg = cfg.get("geophys", {})
    for mt in ("maggr", "etgr", "terfl"):
        maps = (geo_cfg.get(mt) or {}).get("maps", "gray")
        files = geophys.select_rasters(reg.rasters.get(mt, []), mt, maps)
        if not files:
            continue
        if (geo_cfg.get(mt) or {}).get("classic", True) is False:
            # Классика выключена конфигом: эту модальность считает сеть (geo_model_feats).
            continue
        classes = ("finds_points", "object_poly") if mt == "maggr" else ("object_poly",)
        opts = {c: geophys.params_for(geo_cfg, mt, c) for c in classes}
        boundary = geophys.survey_boundary(reg.boundaries, mt, crs_region, geo_cfg)
        noise = geophys.load_mask(reg.noise, mt, crs_region)
        mine: list[dict] = []
        for path in files:
            if not guard.ok(5):
                log.warning("время на исходе: геофизика %s пропущена", path.name)
                break
            try:
                b_map, n_map = geophys.sane_masks(path, boundary, noise, geo_cfg, crs_region)
                for cls in classes:
                    p, o = opts[cls]
                    found = (geophys.finds_points(path, reg.name, mt, p, fallback_crs=crs_region)
                             if cls == "finds_points"
                             else geophys.object_polys(path, reg.name, mt, p, fallback_crs=crs_region))
                    if o["rank"]:
                        found = geophys.rank_within_source(found)
                    found = geophys.apply_masks(found, b_map, n_map if o["use_noise"] else None)
                    mine += [f for f in found if f["confidence"] >= o["min_confidence"]]
            except Exception as exc:
                log.exception("%s: ошибка геофизики (%s)", path.name, exc)
        if maps == "all" and "finds_points" in classes and len(files) > 1:
            # Одна находка на серой и цветной карте — один объект ([geophys.<модальность>] maps = "all").
            mine = geophys.dedupe_points(mine, 2.0, crs_region)
        mt_cfg = geo_cfg.get(mt) or {}
        if mt_cfg.get("layer_copy", False):
            try:
                layers = {f.name: geophys.source_crs(f, crs_region) for f in files}
                mine += geophys.layer_copies(mine, layers, float(mt_cfg.get("layer_copy_decay", 1.0)), f"{mt}_copy")
            except Exception:
                log.exception("%s: копии по слоям %s пропущены", reg.name, mt)
        feats += mine
    return feats


def cpu_kurgans(d, region: str, source: str, cfg: dict, root: Path, cand_p) -> list[dict]:
    """Пики рельефа и LightGBM. Включается, когда модели нет или она упала на этом LAS."""
    kurgan_model = cfg.get("candidates", {}).get("lgbm_model")
    found = candidates.detect(d, cand_p)
    scores = candidates.score(found["features"], root / kurgan_model if kurgan_model else None)
    return [dict(f, threshold_group="li_cpu")
            for f in candidates.to_features(found, scores, d.crs, region, cand_p, source=source)]


def li_mask_feats(reg: Region, las: Path, cfg: dict, seg_cache: dict, root: Path, dem_for) -> list[dict]:
    """Городища отдельной модели на своём DEM. Группа порога li_mask, не li и не li_coarse.

    Модель видит шаг и каналы из своего чекпойнта. Курганы она не выдаёт.
    Нет файла весов — пустой список, основной лидар не трогается.
    """
    from .lidar import hillshade_set

    segs = _cached_segmenters(seg_cache, cfg, "li_mask", root)
    out: list[dict] = []
    for seg in segs:
        d = dem_for(seg.meta)
        if d.crs is None:
            continue
        found = _kept(seg, _li_with_model(seg, d, reg.name, las.name, hillshade_set, None, None,
                                          output_floors(cfg, "li_mask")))
        for feat in found:
            if feat.get("class_name") != "gorodishcha":
                continue
            feat["threshold_group"] = "li_mask"
            out.append(feat)
    return out


def li_fort_feats(reg: Region, las: Path, cfg: dict, seg_cache: dict, root: Path, dem_for) -> list[dict]:
    """Фортификации отдельной маской на DEM 1 м. Группа порога li_fort, не li и не li_coarse.

    Модель видит шаг и каналы из своего чекпойнта. Курганы она не выдаёт.
    Нет файла весов — пустой список, основной лидар не трогается.
    """
    from .lidar import hillshade_set

    segs = _cached_segmenters(seg_cache, cfg, "li_fort", root)
    out: list[dict] = []
    for seg in segs:
        d = dem_for(seg.meta)
        if d.crs is None:
            continue
        found = _kept(seg, _li_with_model(seg, d, reg.name, las.name, hillshade_set, None, None,
                                          output_floors(cfg, "li_fort")))
        for feat in found:
            if feat.get("class_name") != "fortifikatsii":
                continue
            feat["threshold_group"] = "li_fort"
            out.append(feat)
    return out


def lidar_feats(reg: Region, cfg: dict, guard: TimeGuard, seg_cache: dict, root: Path,
                prefetch=None) -> list[dict]:
    """LAS → объекты лидара и, где есть аэрофото, копии курганов в слой ae.

    Сбой загрузки или инференса модели не оставляет участок без курганов:
    для этого LAS считается запасной детектор.
    """
    from .lidar import build_dem, dem_params, hillshade_set

    feats: list[dict] = []
    mosaic_kurgans: list[dict] = []
    crs_region = region_crs(reg, cfg)
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
                hf = cfg.get("hillfort") or {}
                if hf.get("enabled", False):
                    # Своя ЦМР, как в scripts/hillfort_oof.py. Параметры, включая
                    # relief_radius_l_m, берутся из конфига: принятое табло — один проход.
                    try:
                        from . import hillfort
                        hp = hillfort.params_from(hf)
                        feats += hillfort.candidates(hillfort.build_dem(las, crs_region, hp), reg.name, las.name, hp)
                    except Exception:
                        log.exception("%s: кандидаты городищ по рельефу пропущены", las.name)
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
            if guard.ok(cfg.get("time", {}).get("reserve_s", 120)):
                try:
                    feats += li_mask_feats(reg, las, cfg, seg_cache, root, dem_for)
                except Exception:
                    log.exception("%s: маска городищ пропущена", las.name)
                try:
                    feats += li_fort_feats(reg, las, cfg, seg_cache, root, dem_for)
                except Exception:
                    log.exception("%s: маска фортификаций пропущена", las.name)
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

    # Группа мозаики ставится до копий: копия мозаики — li_ae, копия мелкой сетки — ae, как в табло.
    for f in mosaic_kurgans:
        if f.get("markup_type") == "li" and not f.get("threshold_group"):
            f["threshold_group"] = "li_mosaic"
    fp = _params(fusion.FusionParams, cfg.get("fusion"))
    if fp.li_to_ae and reg.rasters.get("ae"):
        try:
            feats += fusion.li_to_ae(feats, reg.rasters["ae"], fp, fallback_crs=crs_region)
        except Exception as exc:
            log.exception("%s: ошибка слияния li→ae (%s)", reg.name, exc)
    return _with_copies(feats, reg, cfg)


# Круги, которые принятый пакет уже оставляет. Новый круг ближе 12 м к такому
# кругу того же участка и класса выбрасывается. Числа — совместный оптимум
# замера 0,353793 (notes/pack_wide_dev_2026-10-23.json). sp_or и sp_focal —
# одна карта центров: в сдаче группа читается из markup_type, в табло она
# подписана sp_focal. В конфиге порог лидара 0,06415 / 0,194220.
# 0,05333 — более ранняя точка, в конфиг её не возвращать.
KEPT_KURGAN_CONF = {
    "ae": {"kurgany_povrezhdennye": 0.99053, "kurgany_tselye": 0.802255},
    "li": {"kurgany_povrezhdennye": 0.06415, "kurgany_tselye": 0.19422},
    "li_ae": {"kurgany_povrezhdennye": 0.993061, "kurgany_tselye": 1.000001},
    "li_mosaic": {"kurgany_povrezhdennye": 0.959005, "kurgany_tselye": 0.17042},
    "sp_focal": {"kurgany_povrezhdennye": 0.364475, "kurgany_tselye": 0.372525},
    "sp_or": {"kurgany_povrezhdennye": 0.364475, "kurgany_tselye": 0.372525},
}


def ground_point(geom):
    """Внутренняя точка в метрах.

    Мелкие координаты уже метры. Крупные — веб-меркатор: вся фигура переводится
    в зону UTM по центроиду, и только потом берётся внутренняя точка. Так же
    устроен отбор 12 м в принятом пакете.
    """
    from shapely import get_coordinates

    xs = get_coordinates(geom)[:, 0]
    if xs.size == 0 or float(np.max(np.abs(xs))) <= 1_000_000:
        return geom.representative_point()
    from .crs import WEB_MERCATOR, to_crs, transformer, utm_for_lonlat

    lon, lat = transformer(WEB_MERCATOR, "EPSG:4326").transform(geom.centroid.x, geom.centroid.y)
    return to_crs(geom, WEB_MERCATOR, utm_for_lonlat(float(lon), float(lat))).representative_point()


def kept_kurgan_anchors(feats: list[dict]) -> list[dict]:
    """Круги курганов, которые сданный порог своей группы уже оставляет."""
    kept = []
    for feat in feats:
        cls = feat.get("class_name")
        group = feat.get("threshold_group") or feat.get("markup_type")
        thr = KEPT_KURGAN_CONF.get(group, {}).get(cls)
        if thr is None or float(feat.get("confidence") or 0) < thr:
            continue
        kept.append(feat)
    return kept


def drop_near_kurgans(cands: list[dict], anchors: list[dict], dist_m: float = 12.0) -> list[dict]:
    """Убирает круг, если ближе dist_m к уже оставленному кругу того же участка и класса."""
    from shapely.strtree import STRtree

    buckets: dict[tuple, list] = {}
    for feat in anchors:
        cls = feat.get("class_name")
        if cls not in ("kurgany_povrezhdennye", "kurgany_tselye") or feat.get("geom") is None:
            continue
        buckets.setdefault((feat.get("region_name"), cls), []).append(ground_point(feat["geom"]))
    trees = {key: STRtree(pts) for key, pts in buckets.items() if pts}
    kept = []
    for feat in cands:
        pts = buckets.get((feat.get("region_name"), feat.get("class_name")))
        if not pts or feat.get("geom") is None:
            kept.append(feat)
            continue
        point = ground_point(feat["geom"])
        key = (feat.get("region_name"), feat.get("class_name"))
        near = any(pts[int(i)].distance(point) <= dist_m for i in trees[key].query(point.buffer(dist_m)))
        if not near:
            kept.append(feat)
    return kept


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
                                                           min_conf=output_floors(cfg, mt),
                                                           fallback_crs=region_crs(reg, cfg)))
                except Exception as exc:
                    log.exception("%s: ошибка модели %s (%s)", path.name, mt, exc)
    feats = _wide_without_lidar(reg, cfg, guard, seg_cache, root, feats)
    return _with_copies(feats, reg, cfg) + geo_model_feats(reg, cfg, guard, seg_cache, root)


#: Модальности геофизики, для которых [models] может назвать сеть. Группа порога — «<модальность>_nn».
GEO_NN = ("maggr", "etgr", "terfl")


def geo_model_feats(reg: Region, cfg: dict, guard: TimeGuard, seg_cache: dict, root: Path) -> list[dict]:
    """Сеть на картах геофизики: [models] maggr / etgr / terfl. Без строки в конфиге — пусто.

    Растры те же, что у классики (select_rasters: без цветных копий и срезов EtGr глубже 2 м).
    Объекты идут в группу «<модальность>_nn», порог — [thresholds_by_markup.<модальность>_nn].
    Без этой секции действует [thresholds] класса (у геофизики 0,0), поэтому сеть
    включается только вместе с порогами из табло. Классику той же модальности выключает
    [geophys.<модальность>] classic = false, иначе выдаются обе ветки.
    Маски границ и шума здесь не применяются: так же предсказывает scripts/nn_oof.py.
    """
    out: list[dict] = []
    models = cfg.get("models", {})
    for mt in GEO_NN:
        if not models.get(mt):
            continue
        maps = ((cfg.get("geophys") or {}).get(mt) or {}).get("maps", "gray")
        files = geophys.select_rasters(reg.rasters.get(mt, []), mt, maps)
        if not files:
            continue
        try:
            segs = _cached_segmenters(seg_cache, cfg, mt, root)
        except Exception:
            log.exception("модель %s не загрузилась, сеть геофизики пропущена", mt)
            seg_cache[mt] = []
            continue
        group = f"{mt}_nn"
        floors = output_floors(cfg, group)
        added: list[dict] = []
        for seg in segs:
            for path in files:
                if not guard.ok(30):
                    log.warning("время на исходе: сеть %s пропущена на %s", mt, path.name)
                    break
                try:
                    found = _kept(seg, _raster_with_model(seg, path, reg.name, mt, min_conf=floors,
                                                          fallback_crs=region_crs(reg, cfg)))
                except Exception as exc:
                    log.exception("%s: ошибка сети %s (%s)", path.name, mt, exc)
                    continue
                for feat in found:
                    feat["threshold_group"] = group
                added += found
        if maps == "all" and len(files) > 1:
            # Только находки этой модальности: чужие классы в том же списке не склеиваются.
            added = geophys.dedupe_points(added, 2.0, region_crs(reg, cfg))
        out += added
    return out


def _wide_without_lidar(reg: Region, cfg: dict, guard: TimeGuard, seg_cache: dict, root: Path,
                        feats: list[dict]) -> list[dict]:
    """Окно 256 м только там, где нет LAS.

    Круг ближе 12 м к кругу карты центров, который сданный порог уже оставляет,
    выбрасывается. Иначе совместный перебор поднимает порог сданной карты.
    """
    if reg.las or not cfg.get("models", {}).get("sp_focal2") or not reg.rasters.get("sp_or"):
        return feats
    try:
        segs = _cached_segmenters(seg_cache, cfg, "sp_focal2", root)
    except Exception:
        log.exception("модель sp_focal2 не загрузилась")
        seg_cache["sp_focal2"] = []
        return feats
    floors = output_floors(cfg, "sp_focal2")
    added: list[dict] = []
    for seg in segs:
        for path in reg.rasters.get("sp_or", []):
            if not guard.ok(60):
                break
            try:
                found = _kept(seg, _raster_with_model(seg, path, reg.name, "sp_or", min_conf=floors,
                                                      fallback_crs=region_crs(reg, cfg)))
            except Exception as exc:
                log.exception("%s: ошибка окна 256 м (%s)", path.name, exc)
                continue
            for feat in found:
                feat["threshold_group"] = "sp_focal2"
            added += found
    return feats + drop_near_kurgans(added, kept_kurgan_anchors(feats), 12.0)


def li_dev_feats(reg: Region, cfg: dict, guard: TimeGuard, seg_cache: dict, root: Path,
                 prefetch=None) -> list[dict]:
    """Круги канала σ на пиках рельефа. Группа li_dev.

    Отбор 12 м делает run: сюда попадают все круги выше пола выдачи.
    Нет весов — пустой список.
    """
    from .lidar import build_dem, dem_params, hillshade_set

    if not reg.las or not cfg.get("models", {}).get("li_dev"):
        return []
    try:
        segs = _cached_segmenters(seg_cache, cfg, "li_dev", root)
    except Exception:
        log.exception("модель li_dev не загрузилась")
        seg_cache["li_dev"] = []
        return []
    crs_region = region_crs(reg, cfg)
    cand_p = _params(candidates.CandidateParams, cfg.get("candidates"))
    floors = output_floors(cfg, "li_dev")
    out: list[dict] = []
    for las in reg.las:
        if not guard.ok(60):
            break
        for seg in segs:
            dp = dem_params(seg.meta, cfg)
            d = (prefetch.get(las, dp["res"], dp["fill"], crs_region) if prefetch is not None
                 else build_dem(las, res=dp["res"], fill=dp["fill"], crs=crs_region))
            if d.crs is None:
                continue
            found = _kept(seg, _li_with_model(seg, d, reg.name, las.name, hillshade_set,
                                              cand_p, cfg.get("li"), floors))
            for feat in found:
                if feat.get("class_name") not in ("kurgany_povrezhdennye", "kurgany_tselye"):
                    continue
                feat["threshold_group"] = "li_dev"
                out.append(feat)
    return out


def run_region(reg: Region, cfg: dict, guard: TimeGuard, seg_cache: dict, root: Path,
               prefetch=None) -> list[dict]:
    """Один участок: лидар, геофизика, оптика. Канала σ здесь нет. Платформа идёт через run(): четыре прохода."""
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
    Чекпойнт с target="centers" отдаёт круги вокруг пиков, как оптика. У сданных
    весов лидара этого поля нет.
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
        if seg.meta.get("target") == "centers":
            from .centers import centers_to_features

            idx = [seg.classes.index(c) for c in out_classes]
            return centers_to_features(
                probs[idx], out_classes, tr, d.crs, float(tr.a), region, "li", source,
                radius_m=float(seg.meta.get("center_radius_m") or 6.0),
                min_dist_m=float(12.0 if seg.meta.get("min_dist_m") is None else seg.meta["min_dist_m"]),
                min_conf=min_conf)
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


def _read_ahead(jobs: list, read, depth: int | None = None):
    """(job, read(job)) по порядку; следующие depth окон читаются в фоновом потоке.

    EXPDS_READ_AHEAD=0 — читать в основном потоке, как раньше.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor

    depth = int(os.environ.get("EXPDS_READ_AHEAD", "1")) if depth is None else depth
    if depth <= 0 or len(jobs) < 2:
        for job in jobs:
            yield job, read(job)
        return
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="expds-read")
    try:
        futs = [pool.submit(read, job) for job in jobs[:depth]]
        for i, job in enumerate(jobs):
            if i + depth < len(jobs):
                futs.append(pool.submit(read, jobs[i + depth]))
            yield job, futs[i].result()
            futs[i] = None
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def _raster_with_model(seg, path: Path, region: str, markup: str, block: int = 4096,
                       overlap: int = 256, min_conf: dict[str, float] | None = None,
                       fallback_crs: str | None = None) -> list[dict]:
    """Растр на рабочем шаге модели окнами block×block с перекрытием → объекты (память — одно окно).

    CRS — через crs.usable_raster_crs: безымянная LOCAL_CS с координатами UTM получает fallback_crs.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import Window
    from shapely.geometry import box

    from .crs import WEB_MERCATOR, to_crs, usable_raster_crs
    from .infer_nn import probs_to_features
    from .tiles import MAX_UPSAMPLE, band_stats, fix_channels, to_uint8, working_res

    as_centers = seg.meta.get("target") == "centers"

    stats = band_stats(str(path), markup)
    half = overlap // 2
    feats: list[dict] = []
    from .raster import open_metric

    with open_metric(path) as ds:
        native = abs(ds.res[0])
        if seg.meta.get("pixel_units") == "ground":
            from .raster import ground_step, raster_kind_and_latitude

            kind, lat = raster_kind_and_latitude(ds.crs, ds.transform, ds.width, ds.height)
            r = ground_step(native, kind, lat, seg.res, seg.meta.get("max_upsample") or MAX_UPSAMPLE)
        else:
            r = working_res(native, seg.res, seg.meta.get("max_upsample") or MAX_UPSAMPLE)
        scale = r / native
        w, h = max(1, int(ds.width / scale)), max(1, int(ds.height / scale))
        n = min(ds.count, 3)
        crs = usable_raster_crs(ds.crs.to_string() if ds.crs else None, ds.bounds, fallback_crs)
        step = block - overlap
        xs = list(range(0, max(w - overlap, 1), step))
        ys = list(range(0, max(h - overlap, 1), step))
        def read_block(job):
            _, _, y0, x0 = job
            bw, bh = min(block, w - x0), min(block, h - y0)
            win = Window(x0 * scale, y0 * scale, bw * scale, bh * scale)
            return bw, bh, win, ds.read(list(range(1, n + 1)), window=win, out_shape=(n, bh, bw),
                                        resampling=Resampling.average if scale > 1 else Resampling.bilinear)

        jobs = [(iy, ix, y0, x0) for iy, y0 in enumerate(ys) for ix, x0 in enumerate(xs)]
        # Следующее окно читается (GDAL отпускает GIL), пока текущее идёт через сеть и полигоны.
        # Читает только поток чтения, по одному окну за раз: данные те же, порядок тот же.
        blocks = _read_ahead(jobs, read_block)
        try:
            for (iy, ix, y0, x0), (bw, bh, win, a) in blocks:
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
        finally:
            blocks.close()   # дождаться фонового чтения до закрытия ds
    return feats


def run(input_dir: str | Path, output_path: str | Path, config_path: str | Path | None = None,
        root: str | Path | None = None) -> dict:
    """Полный прогон: геофизика, лидар, оптика, затем канал σ.

    Круг канала σ остаётся, только если он дальше 12 м от кургана, который
    сданный порог уже оставляет. Окно 256 м отбирается так же в optical_feats.
    Файл результата пишется в конце. Сторож пишет его же незадолго до снятия
    контейнера, не после геофизики.
    """
    from .watchdog import RunState, arm_platform, arm_signal, checkpoint

    root = Path(root or Path(__file__).resolve().parents[1])
    cfg_path = Path(config_path or root / "configs" / "default.toml")
    cfg = load_config(cfg_path)
    infer = cfg.get("infer", {})
    try:
        from .infer_nn import configure_runtime
        configure_runtime(bool(infer.get("amp", False)), bool(infer.get("tf32", False)),
                          bool(infer.get("deterministic", True)))
    except Exception:
        log.exception("точность видеокарты не настроена, дальше без моделей torch")
    guard = TimeGuard(cfg.get("time", {}).get("budget_s", DEFAULT_BUDGET_S))
    log.info("ускоритель: %s", accel.backend())
    torch_version, cuda_version = "нет", "нет"
    try:
        import torch
        torch_version = torch.__version__
        cuda_version = str(torch.version.cuda or "нет")
        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "torch не загрузился"
    log.info("устройство: %s", device)
    state = RunState(Path(output_path), cfg)
    state.note.update({
        "device": device,
        "torch": torch_version,
        "cuda": cuda_version,
        "backend": accel.backend(),
        "config_sha256": hashlib.sha256(cfg_path.read_bytes()).hexdigest() if cfg_path.is_file() else "нет файла",
        "budget_s": cfg.get("time", {}).get("budget_s", DEFAULT_BUDGET_S),
        "reserve_s": cfg.get("time", {}).get("reserve_s", 120),
    })
    if cfg.get("time", {}).get("watchdog", True):
        margin = cfg.get("time", {}).get("guard_margin_s", 180)
        arm_platform(state, margin_s=float(margin))
    seg_cache: dict = {}
    summary = {"regions": 0, "raw": 0}
    prefetch = None
    phases: list[tuple[str, int, float]] = []
    regions = []
    previous_term = arm_signal(state)
    try:
        from .lidar import DemPrefetch

        regions = scan_root(input_dir)
        summary["regions"] = len(regions)
        n_las = sum(len(r.las) for r in regions)
        by_raster: dict[str, int] = {}
        n_skipped = 0
        for reg in regions:
            n_skipped += len(getattr(reg, "skipped", ()) or ())
            for markup, files in getattr(reg, "rasters", {}).items():
                by_raster[markup] = by_raster.get(markup, 0) + len(files)
        n_rasters = sum(by_raster.values())
        raster_text = ", ".join(f"{name} {by_raster[name]}" for name in sorted(by_raster)) or "нет"
        log.info("к разбору: регионов %d, LAS %d, растров %d (%s), пропущено файлов %d",
                 len(regions), n_las, n_rasters, raster_text, n_skipped)
        try:
            from .health import log_inventory, runtime_report

            log.info("среда: %s", runtime_report(Path(output_path)))
            log_inventory(regions, cfg, root)
        except Exception:
            log.exception("отчёт о среде пропущен")
        state.note.update({
            "regions": len(regions),
            "regions_with_las": sum(1 for reg in regions if reg.las),
            "las_files": n_las,
            "rasters": raster_text,
            "skipped": n_skipped,
        })
        regions.sort(key=lambda r: (not r.las, r.name))
        depth = int(cfg.get("lidar", {}).get("prefetch", 2))
        if depth > 0:
            prefetch = DemPrefetch([(las, region_crs(r, cfg)) for r in regions for las in r.las], depth)
        t_phase = time.monotonic()
        n_phase = 0
        for reg in regions:
            found = geophys_feats(reg, cfg, guard)
            state.add(found)
            n_phase += len(found)
            if found:
                log.info("%s: геофизика %+d, прошло %.0f с", reg.name, len(found), guard.elapsed())
        phases.append(("геофизика", n_phase, time.monotonic() - t_phase))
        reserve = cfg.get("time", {}).get("reserve_s", 120)
        t_phase = time.monotonic()
        n_phase = 0
        for reg in regions:
            if not guard.ok(reserve):
                log.warning("бюджет времени исчерпан, оставшийся лидар пропущен")
                break
            t = time.monotonic()
            try:
                found = lidar_feats(reg, cfg, guard, seg_cache, root, prefetch)
            except Exception:
                log.exception("%s: лидар пропущен", reg.name)
                found = []
            state.add(found)
            n_phase += len(found)
            log.info("%s: лидар %+d, %.1f с, прошло %.0f с", reg.name, len(found),
                     time.monotonic() - t, guard.elapsed())
        phases.append(("лидар", n_phase, time.monotonic() - t_phase))
        t_phase = time.monotonic()
        n_phase = 0
        for reg in regions:
            if not guard.ok(60):
                log.warning("на оптику времени не осталось")
                break
            try:
                found = optical_feats(reg, cfg, guard, seg_cache, root)
            except Exception:
                log.exception("%s: оптика пропущена", reg.name)
                found = []
            state.add(found)
            n_phase += len(found)
            if found:
                log.info("%s: оптика %+d, прошло %.0f с", reg.name, len(found), guard.elapsed())
        phases.append(("оптика", n_phase, time.monotonic() - t_phase))
        t_phase = time.monotonic()
        n_phase = 0
        for reg in regions:
            if not reg.las:
                continue
            if not guard.ok(60):
                log.warning("на канал σ времени не осталось")
                break
            try:
                fresh = li_dev_feats(reg, cfg, guard, seg_cache, root, prefetch)
                pool = [f for f in state.snapshot()
                        if f.get("region_name") == reg.name and f.get("threshold_group") != "li_dev"]
                kept = drop_near_kurgans(fresh, kept_kurgan_anchors(pool), 12.0)
            except Exception:
                log.exception("%s: канал σ пропущен", reg.name)
                kept = []
            state.add(kept)
            n_phase += len(kept)
            if kept:
                log.info("%s: канал σ %+d, прошло %.0f с", reg.name, len(kept), guard.elapsed())
        phases.append(("канал σ", n_phase, time.monotonic() - t_phase))
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        if prefetch is not None:
            prefetch.close()
        summary["raw"] = len(state.snapshot())
        if (cfg.get("verify") or {}).get("enabled") and guard.ok(120):
            # Ошибка проверки кропа не должна оставить площадку без файла ответа.
            try:
                from .crop_model import maybe_verify

                checked = maybe_verify(state.snapshot(), {r.name: r for r in regions}, cfg)
                if checked is not None:
                    state.replace(checked)
            except Exception:
                log.exception("проверка кропа пропущена")
        summary["phases"] = {name: {"raw": n, "seconds": round(sec, 1)} for name, n, sec in phases}
        state.note["phases"] = summary["phases"]
        summary["written"] = checkpoint(state, "конец")
        summary["seconds"] = round(guard.elapsed(), 1)
        for handler in logging.getLogger().handlers:
            handler.flush()
    return summary
