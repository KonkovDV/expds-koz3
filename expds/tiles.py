"""Кэш тайлов для обучения нейросети на слабой видеокарте.

Тайлы нарезаются один раз на CPU и сохраняются в .npy: обучение потом не
ждёт чтения гигантских растров. Позитивы — окна вокруг эталонных объектов
(со сдвигом), негативы — случайные окна и окна вокруг «шума» (extra/*_шум).

Маски хранятся битами: бит k = класс meta["classes"][k] (экономия ×5 по диску).

    python -m expds.tiles --root D:/expds3 --gt work/gt_train.geojson --markup li \
        --out work/cache/li --res 0.5 --tile 256

Для li каналы по умолчанию строятся из LAS (lidar.relief_channels) — той же
функцией, что на инференсе. --li-source tif берёт готовые карты Li трейна;
их нельзя воспроизвести из LAS, поэтому модель на них не переносится на валидацию.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.windows import Window
from shapely.geometry import box, shape

from .catalog import Region, read_utm_json, scan_root
from .constants import GEO_CLASSES, GEO_MARKUP, RS_CLASSES
from .crs import WEB_MERCATOR, raster_crs, to_crs
from .naming import site_family

log = logging.getLogger(__name__)


def classes_for(markup: str) -> list[str]:
    """Классы, которые учит модель данной модальности."""
    return list(GEO_CLASSES) if markup in GEO_MARKUP else list(RS_CLASSES)


def sources_for(reg: Region, markup: str, li_tags: list[str]) -> list[list[Path]]:
    """Наборы файлов-каналов для модальности: для li — по тегам (g, c, ch), иначе по файлу."""
    if markup != "li":
        return [[p] for p in reg.rasters.get(markup, [])]
    sets: dict[str, dict[str, Path]] = {}
    for p in reg.li_rasters:
        stem, _, tag = p.stem.rpartition("_")
        sets.setdefault(stem, {})[tag] = p
    return [[s[t] for t in li_tags] for s in sets.values() if all(t in s for t in li_tags)]


#: Растр увеличивается до рабочего шага не больше чем в MAX_UPSAMPLE раз: аэрофото 1,6 м
#: на шаге 0,5 м — это 2,4 Гпикс и 44,7 ГБ на одну карту вероятностей.
MAX_UPSAMPLE = 2.0


def working_res(native_res: float, res: float, max_upsample: float = MAX_UPSAMPLE) -> float:
    """Рабочий шаг растра: res, но не мельче native_res / max_upsample."""
    return max(res, native_res / max_upsample)


@lru_cache(maxsize=256)
def band_stats(path: str, markup: str, max_side: int = 2048) -> tuple | None:
    """Перевод каналов растра в uint8, общий для всего снимка: ((a, b), …), y = (x − a)·b; None — уже uint8.

    Считается по уменьшенному обзору снимка, поэтому тайлы обучения и окна
    инференса растягиваются одинаково (по своим перцентилям они растягивались бы по-разному).
    """
    import math

    with rasterio.open(path) as ds:
        if ds.dtypes[0] == "uint8":
            return None
        n = min(ds.count, 3)
        f = max(1, math.ceil(max(ds.width, ds.height) / max_side))
        a = ds.read(list(range(1, n + 1)), out_shape=(n, max(1, ds.height // f), max(1, ds.width // f)),
                    resampling=Resampling.nearest, masked=True).astype(np.float32).filled(np.nan)
    out = []
    for x in a:
        fin = x[np.isfinite(x)]
        if fin.size == 0:
            out.append((0.0, 1.0))
        elif markup in GEO_MARKUP:
            med = float(np.median(fin))
            mad = float(np.median(np.abs(fin - med))) * 1.4826 or 1.0
            out.append((med - 4 * mad, 31.75 / mad))
        else:
            lo, hi = np.percentile(fin, [1, 99])
            out.append((float(lo), 255.0 / max(float(hi - lo), 1e-6)))
    return tuple(out)


def to_uint8(a: np.ndarray, markup: str, stats: tuple | None = None) -> np.ndarray:
    """Каналы → uint8: геофизика — робастная нормировка, остальное — как есть или растяжка.

    stats (band_stats) задаёт растяжку всего снимка; без них — по самому массиву.
    """
    if a.dtype == np.uint8:
        return a
    a = a.astype(np.float32)
    out = np.zeros(a.shape, np.uint8)
    if stats is not None:
        for c in range(a.shape[0]):
            lo, k = stats[min(c, len(stats) - 1)]
            out[c] = np.clip(np.nan_to_num((a[c] - lo) * k, nan=0), 0, 255).astype(np.uint8)
        return out
    for c in range(a.shape[0]):
        x = a[c]
        fin = np.isfinite(x)
        if not fin.any():
            continue
        if markup in GEO_MARKUP:
            med = np.median(x[fin])
            mad = np.median(np.abs(x[fin] - med)) * 1.4826 or 1.0
            y = np.clip((x - med) / mad, -4, 4) * 31.75 + 127.5
        else:
            lo, hi = np.percentile(x[fin], [1, 99])
            y = (x - lo) / max(hi - lo, 1e-6) * 255
        out[c] = np.clip(np.nan_to_num(y, nan=0), 0, 255).astype(np.uint8)
    return out


def n_channels(markup: str, li_tags: list[str]) -> int:
    """Число каналов модели: li — по тегам (c, ch — 3, остальные — 1), геофизика — 1, оптика — 3."""
    if markup == "li":
        return sum(3 if t in ("c", "ch") else 1 for t in li_tags)
    return 1 if markup in GEO_MARKUP else 3


def cut_window(arr: np.ndarray, transform, cx: float, cy: float, tile: int):
    """Окно tile×tile из C×H×W, выровненное по сетке массива; снаружи — 0. Возвращает (окно, привязка)."""
    step = transform.a
    col0 = round((cx - transform.c) / step - tile / 2)
    row0 = round((transform.f - cy) / step - tile / 2)
    c, h, w = arr.shape
    out = np.zeros((c, tile, tile), arr.dtype)
    r0, r1 = max(0, row0), min(h, row0 + tile)
    c0, c1 = max(0, col0), min(w, col0 + tile)
    if r1 > r0 and c1 > c0:
        out[:, r0 - row0:r1 - row0, c0 - col0:c1 - col0] = arr[:, r0:r1, c0:c1]
    return out, from_origin(transform.c + col0 * step, transform.f - row0 * step, step, step)


#: Классы, маски которых сжимаются к центру при shrink > 0 (плотные курганные группы).
SHRINK_CLASSES = ("kurgany_povrezhdennye", "kurgany_tselye")


def shrink_polygon(g, shrink: float):
    """Полигон, сжатый к центру: радиус эквивалентного круга умножается на (1 − shrink).

    Соседние насыпи на маске перестают касаться, и связные области модели
    разделяют экземпляры. Метрика засчитывает такой объект: покрытие меньшей
    фигуры = 1, внутренняя точка — внутри эталона.
    """
    import math

    if shrink <= 0 or g.geom_type == "Point" or g.area <= 0:
        return g
    r = math.sqrt(g.area / math.pi)
    s = g.buffer(-shrink * r)
    if s.is_empty or s.area < 0.05 * g.area:
        s = g.representative_point().buffer(max(r * (1 - shrink), 0.5))
    return s


def iter_sources(reg: Region, markup: str, li_tags: list[str], li_source: str, res: float, tile: int,
                 dem: dict | None = None):
    """Источники тайлов региона: (имя, CRS, рамка, read(cx, cy) → (окно, привязка))."""
    if markup == "li" and li_source == "las":
        from .lidar import MODEL_DEM, build_dem, relief_channels

        dem = dem or MODEL_DEM
        for las in reg.las:
            d = build_dem(las, res=dem["res"], fill=dem["fill"], crs=read_utm_json(reg.utm_json))
            if d.crs is None:
                log.warning("%s: нет CRS у %s — пропуск", reg.name, las.name)
                continue
            arr, tr = relief_channels(d, res, tuple(li_tags))
            h, w = arr.shape[1:]
            bnd = box(tr.c, tr.f - h * tr.a, tr.c + w * tr.a, tr.f)
            yield las.name, d.crs, bnd, lambda cx, cy, a=arr, t=tr: cut_window(a, t, cx, cy, tile)
        return
    for files in sources_for(reg, markup, li_tags):
        crs = raster_crs(files[0]) or read_utm_json(reg.utm_json)
        if crs is None:
            log.warning("%s: нет CRS у %s — пропуск", reg.name, files[0].name)
            continue
        with rasterio.open(files[0]) as ds:
            bnd = box(*ds.bounds)
            r = working_res(abs(ds.res[0]), res)
        stats = band_stats(str(files[0]), markup) if len(files) == 1 else None
        yield files[0].name, crs, bnd, lambda cx, cy, f=files, r=r, st=stats: (
            to_uint8(read_window(f, cx, cy, r, tile), markup, st),
            from_origin(cx - tile * r / 2, cy + tile * r / 2, r, r))


def fix_channels(img: np.ndarray, c: int) -> np.ndarray:
    """Приводит число каналов к c: лишние отбрасываются, недостающие повторяют последний."""
    if img.shape[0] >= c:
        return img[:c]
    return np.concatenate([img, np.repeat(img[-1:], c - img.shape[0], axis=0)], axis=0)


def read_window(files: list[Path], cx: float, cy: float, res: float, tile: int) -> np.ndarray:
    """Окно tile×tile с центром (cx, cy) в CRS растра и шагом res; все каналы файлов подряд."""
    chans = []
    half = tile * res / 2
    for f in files:
        with rasterio.open(f) as ds:
            inv = ~ds.transform
            x0, y0, x1, y1 = cx - half, cy + half, cx + half, cy - half
            col0, row0 = inv.a * x0 + inv.b * y0 + inv.c, inv.d * x0 + inv.e * y0 + inv.f
            col1, row1 = inv.a * x1 + inv.b * y1 + inv.c, inv.d * x1 + inv.e * y1 + inv.f
            win = Window(col0, row0, col1 - col0, row1 - row0)
            n = min(ds.count, 3)
            scale = res / abs(ds.res[0])
            a = ds.read(list(range(1, n + 1)), window=win, out_shape=(n, tile, tile), boundless=True,
                        fill_value=0, resampling=Resampling.average if scale > 1 else Resampling.bilinear)
            chans.append(a)
    return np.concatenate(chans, axis=0)


def build_cache(root: str, gt_path: str, markup: str, out: str, res: float = 0.5, tile: int = 256,
                neg_per_pos: float = 1.0, jitter: float = 0.25, li_tags: list[str] | None = None,
                max_per_region: int = 3000, seed: int = 0, shard_size: int = 2000,
                li_source: str = "las", shrink: float = 0.0, dem: dict | None = None,
                only_classes: list[str] | None = None, pos_repeat: int = 1,
                only_regions: list[str] | None = None) -> dict:
    """Строит кэш тайлов и возвращает сводку.

    dem — {res, fill} DEM для li из LAS (по умолчанию MODEL_DEM); only_classes — учить
    только эти классы (например, городища на грубом шаге); pos_repeat — тайлов на
    объект (каждый со своим случайным сдвигом), для редких классов.
    """
    from .lidar import MODEL_DEM, RELIEF_TAGS

    dem = dict(dem or MODEL_DEM)
    rng = random.Random(seed)
    li_tags = list(li_tags or (RELIEF_TAGS if li_source == "las" else ["g"]))
    n_ch = n_channels(markup, li_tags)
    classes = [c for c in classes_for(markup) if not only_classes or c in only_classes]
    if not classes:
        raise ValueError(f"нет классов {only_classes} в модальности {markup}")
    gt = json.loads(Path(gt_path).read_text(encoding="utf-8"))["features"]
    by_region: dict[str, list[tuple]] = {}
    for f in gt:
        pr = f["properties"]
        if pr.get("markup_type") == markup and pr["class_name"] in classes:
            by_region.setdefault(pr["region_name"], []).append((shape(f["geometry"]), pr["class_name"]))
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs, masks, rows = [], [], []
    shard = 0
    total = {"tiles": 0, "positive": 0}

    def flush():
        nonlocal shard, imgs, masks
        if not imgs:
            return
        np.save(out_dir / f"images_{shard:04d}.npy", np.stack(imgs))
        np.save(out_dir / f"masks_{shard:04d}.npy", np.stack(masks))
        shard += 1
        imgs, masks = [], []

    for reg in scan_root(root):
        if only_regions is not None and reg.name not in only_regions:
            continue
        objs = by_region.get(reg.name, [])
        for src_name, crs, bnd, read in iter_sources(reg, markup, li_tags, li_source, res, tile, dem):
            local = [(to_crs(g, WEB_MERCATOR, crs), c) for g, c in objs]
            local = [(g, c) for g, c in local if g.intersects(bnd)]
            centers = [(g.representative_point().x, g.representative_point().y, 1) for g, _ in local] * pos_repeat
            rng.shuffle(centers)
            centers = centers[:max_per_region]
            n_neg = max(5, int(len(centers) * neg_per_pos))
            minx, miny, maxx, maxy = bnd.bounds
            centers += [(rng.uniform(minx, maxx), rng.uniform(miny, maxy), 0) for _ in range(n_neg)]
            for noise in reg.noise:
                try:
                    data = json.loads(noise.read_text(encoding="utf-8"))
                    for nf in data.get("features", [])[:200]:
                        p = to_crs(shape(nf["geometry"]), WEB_MERCATOR, crs).representative_point()
                        if bnd.contains(p):
                            centers.append((p.x, p.y, 0))
                except Exception:
                    pass
            for cx, cy, pos in centers:
                j = tile * res * jitter
                cx += rng.uniform(-j, j) if pos else 0
                cy += rng.uniform(-j, j) if pos else 0
                img, tr = read(cx, cy)
                if not img.any():
                    continue
                m = np.zeros((tile, tile), np.uint8)
                for k, cls in enumerate(classes):
                    shp = [(g.buffer(1.0) if g.geom_type == "Point"
                            else shrink_polygon(g, shrink) if cls in SHRINK_CLASSES else g)
                           for g, c in local if c == cls]
                    if shp:
                        r = rasterize(((g, 1) for g in shp), out_shape=(tile, tile), transform=tr,
                                      all_touched=True, dtype="uint8")
                        m |= (r.astype(np.uint8) << k)
                imgs.append(fix_channels(to_uint8(img, markup), n_ch))
                masks.append(m)
                rows.append([shard, len(imgs) - 1, reg.name, site_family(reg.name), src_name,
                             round(cx, 2), round(cy, 2), int(m.any())])
                total["tiles"] += 1
                total["positive"] += int(m.any())
                if len(imgs) >= shard_size:
                    flush()
        log.info("%s: тайлов всего %d", reg.name, total["tiles"])
    flush()
    with (out_dir / "index.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["shard", "idx", "region", "family", "file", "cx", "cy", "positive"])
        w.writerows(rows)
    li_src = li_source if markup == "li" else None
    meta = {"markup": markup, "classes": classes, "res": res, "tile": tile,
            "li_tags": li_tags if markup == "li" else None,
            "dem_res": dem["res"] if li_src == "las" else None, "dem_fill": dem["fill"] if li_src == "las" else None,
            "li_source": li_src, "flips": markup != "li" or (li_src == "las" and "g" not in li_tags),
            "channels": n_ch, "shrink": shrink, "max_upsample": MAX_UPSAMPLE, **total}
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Кэш тайлов для обучения")
    ap.add_argument("--root", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--markup", required=True, choices=["li", "ae", "sp_or", "or", "maggr", "etgr", "terfl"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=float, default=0.5)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--neg-per-pos", type=float, default=1.0)
    ap.add_argument("--li-source", choices=["las", "tif"], default="las",
                    help="li: каналы из LAS (как на инференсе) или готовые карты Li трейна")
    ap.add_argument("--li-tags", nargs="*", default=None,
                    help="las: lrm_s lrm_l lrm_xl slope g; tif: g i c ch")
    ap.add_argument("--max-per-region", type=int, default=3000)
    ap.add_argument("--shrink", type=float, default=0.0,
                    help="сжать маски курганов к центру на эту долю радиуса (разделение плотных групп)")
    ap.add_argument("--dem-res", type=float, default=None, help="шаг DEM для li из LAS (по умолчанию 0,5 м)")
    ap.add_argument("--dem-fill", choices=["pyramid", "delaunay"], default=None)
    ap.add_argument("--classes", nargs="*", default=None, help="учить только эти классы")
    ap.add_argument("--pos-repeat", type=int, default=1, help="тайлов на объект (разные сдвиги)")
    ap.add_argument("--regions", nargs="*", default=None, help="только эти участки")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from .lidar import MODEL_DEM

    dem = {"res": args.dem_res or MODEL_DEM["res"], "fill": args.dem_fill or MODEL_DEM["fill"]}
    meta = build_cache(args.root, args.gt, args.markup, args.out, args.res, args.tile, args.neg_per_pos,
                       li_tags=args.li_tags, max_per_region=args.max_per_region, li_source=args.li_source,
                       shrink=args.shrink, dem=dem, only_classes=args.classes, pos_repeat=args.pos_repeat,
                       only_regions=args.regions)
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    _cli()
