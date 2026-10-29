"""bot-7 (кандидат v3): детектор D-FINE bot-9 на растрах SpOr, группа порога ``sp_dfine``. По умолчанию выключен.

Порт инференса ``bot-9/kaggle/dfine_spor/src/dfine_main.py`` (Scene + infer_region) без загрузки данных
и без сетевого кода: окно 640 px на 0,5 м в UTM участка, перекрытие 160, flip-TTA, top-k 100, min_score 0,05,
NMS по классам на файл (IoU 0,5). Перевод в объекты пайплайна — вариант C из bot-9 DFINE.md / prep5.py:
по (участок, класс) жадные кластеры (центры ≤ merge_m м на земле или покрытие bbox ≥ 0,5) → круг radius_m м
в центре самого уверенного, уверенность — его. Вариант E (``convert = "ellipse"``) — эллипсы как есть.

Что зафиксировано (не подбиралось здесь):
* markup_type ``sp_or``, threshold_group ``sp_dfine`` (как в OOF bot-9; указание bot 1);
* без ``[thresholds_by_markup.sp_dfine]`` для всех классов модуль ничего не выдаёт (``require_thresholds``):
  общий порог 0,15 к D-FINE не применим, порог нужен из OOF-подбора bot-9/bot-10;
* ``drop_groups`` (вариант R) — только явным списком; по умолчанию пусто (вариант C), см. REPORT_v3.md;
* ансамбль фолдов: ``ensemble = "max"`` (кластер C берёт лучший из всех моделей, как NMS) или ``"mean"``
  (среднее по моделям, отсутствующая модель = 0, как score-averaging в WBF). Ни один не измерен честно —
  у фолдовых моделей нет общего OOF; по умолчанию одна модель.

torch / transformers импортируются лениво, только при enabled = true.
"""
from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

CLASSES = ("kurgany_tselye", "kurgany_povrezhdennye", "gorodishcha", "fortifikatsii", "arkhitektury")
GROUP = "sp_dfine"
MARKUP = "sp_or"
R_EARTH = 6378137.0

DEFAULTS = {
    "enabled": False,
    "weights": [],            # каталоги HF (config.json + model.safetensors) от корня решения
    "res_m": 0.5,
    "tile": 640,
    "overlap": 160,
    "batch": 8,
    "flip_tta": True,
    "fp16": True,
    "min_score": 0.05,
    "max_det": 100,
    "nms_iou": 0.5,
    "skip_empty": 0.95,       # окно пропускается, если доля пустых пикселей больше
    "convert": "circle",      # circle = вариант C, ellipse = вариант E
    "radius_m": 4.0,
    "merge_m": 4.0,
    "ensemble": "max",
    "drop_groups": [],        # вариант R: ["sp_or", "sp_focal", "sp_arch", "sp_focal2"]
    "require_thresholds": True,
    "soft_need_s": 150.0,     # запас до мягкого потолка на один растр
    "device": "auto",
}


def settings(cfg: dict) -> dict:
    own = dict(cfg.get("sp_dfine") or {})
    return dict(DEFAULTS, **own)


def enabled(cfg: dict) -> bool:
    return bool((cfg.get("sp_dfine") or {}).get("enabled", False))


def thresholds_ready(cfg: dict) -> tuple[bool, list[str]]:
    """Есть ли порог группы sp_dfine для каждого класса D-FINE. Возвращает (готово, недостающие классы)."""
    own = (cfg.get("thresholds_by_markup") or {}).get(GROUP) or {}
    missing = [c for c in CLASSES if c not in own]
    return not missing, missing


# ── геометрия ────────────────────────────────────────────────────────────────────────────────────────────
def ground_scale(y_3857: float) -> float:
    """Метры на земле на единицу EPSG:3857 (cos φ), как post._ground_scale."""
    return max(math.cos(math.atan(math.sinh(y_3857 / R_EARTH))), 1e-6)


def _cover(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    sm = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return w * h / sm if sm > 0 else 0.0


def cluster_circles(dets: list[dict], radius_m: float = 4.0, merge_m: float = 4.0,
                    ensemble: str = "max", n_models: int = 1) -> list[dict]:
    """Вариант C: dets — эллипсы EPSG:3857 (ключи geom, class_name, confidence, region_name, model).

    Жадно по убыванию уверенности: объект входит в первый кластер, где есть член с расстоянием центров
    ≤ merge_m (на земле) или покрытием bbox ≥ 0,5. Круг radius_m в центре лучшего; уверенность —
    лучшая (``max``) или среднее лучших по моделям (``mean``; модель без члена кластера даёт 0).
    """
    from shapely.geometry import Point

    by_key: dict[tuple, list] = {}
    for d in dets:
        x0, y0, x1, y1 = d["geom"].bounds
        by_key.setdefault((d.get("region_name"), d["class_name"]), []).append(
            ((x0 + x1) / 2, (y0 + y1) / 2, float(d["confidence"]), (x0, y0, x1, y1), d.get("model", 0), d))
    cell = 100.0
    out = []
    for (reg, cls), items in sorted(by_key.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        items.sort(key=lambda c: -c[2])
        grid: dict[tuple, list] = {}
        clusters: list[list] = []
        for c in items:
            x, y, _, bx = c[0], c[1], c[2], c[3]
            pad = merge_m / ground_scale(y)
            hit = None
            seen = set()
            for i in range(int(math.floor((bx[0] - pad) / cell)), int(math.floor((bx[2] + pad) / cell)) + 1):
                for j in range(int(math.floor((bx[1] - pad) / cell)), int(math.floor((bx[3] + pad) / cell)) + 1):
                    for k in grid.get((i, j), ()):
                        if k in seen:
                            continue
                        seen.add(k)
                        for m in clusters[k]:
                            if (math.hypot(x - m[0], y - m[1]) * ground_scale((y + m[1]) / 2) <= merge_m
                                    or _cover(bx, m[3]) >= 0.5):
                                hit = k if hit is None else min(hit, k)
                                break
            if hit is None:
                clusters.append([c])
                hit = len(clusters) - 1
            else:
                clusters[hit].append(c)
            for i in range(int(math.floor(bx[0] / cell)), int(math.floor(bx[2] / cell)) + 1):
                for j in range(int(math.floor(bx[1] / cell)), int(math.floor(bx[3] / cell)) + 1):
                    lst = grid.setdefault((i, j), [])
                    if hit not in lst:
                        lst.append(hit)
        for cl in clusters:
            b = cl[0]
            if ensemble == "mean" and n_models > 1:
                best: dict = {}
                for m in cl:
                    best[m[4]] = max(best.get(m[4], 0.0), m[2])
                conf = sum(best.values()) / n_models
            else:
                conf = b[2]
            src = b[5]
            out.append({"geom": Point(b[0], b[1]).buffer(radius_m / ground_scale(b[1]), quad_segs=4),
                        "class_name": cls, "confidence": float(conf), "region_name": reg,
                        "markup_type": MARKUP, "threshold_group": GROUP, "views": len(cl),
                        "source": src.get("source"), "original_crs": src.get("original_crs")})
    return out


def as_features(dets: list[dict], st: dict, n_models: int) -> list[dict]:
    if st["convert"] == "ellipse":
        return [{k: v for k, v in dict(d, markup_type=MARKUP, threshold_group=GROUP).items() if k != "model"}
                for d in dets]
    return cluster_circles(dets, float(st["radius_m"]), float(st["merge_m"]), str(st["ensemble"]), n_models)


def drop_groups(feats: list[dict], groups, regions) -> list[dict]:
    """Вариант R: убрать объекты групп ``groups`` (кроме sp_dfine) на участках ``regions``.

    Группа — как у scoreboard bot-10: threshold_group, иначе для markup sp_or по классу:
    курганы → sp_focal, архитектура → sp_arch, городища и прочее → sp_or. Участки без D-FINE не трогаются.
    """
    groups = set(groups or ())
    regions = set(regions or ())
    if not groups or not regions:
        return feats

    def grp(f: dict) -> str | None:
        g = f.get("threshold_group")
        if g:
            return g
        if f.get("markup_type") != MARKUP:
            return f.get("markup_type")
        cls = f.get("class_name")
        if cls in ("kurgany_tselye", "kurgany_povrezhdennye"):
            return "sp_focal"
        if cls == "arkhitektury":
            return "sp_arch"
        return "sp_or"

    return [f for f in feats if not (f.get("region_name") in regions and grp(f) in groups and grp(f) != GROUP)]


# ── растр ────────────────────────────────────────────────────────────────────────────────────────────────
def utm_for(path: Path, fallback_crs: str | None) -> str:
    import rasterio
    from pyproj import Transformer

    if fallback_crs:
        return fallback_crs
    with rasterio.open(path) as ds:
        lon, lat = Transformer.from_crs(ds.crs, "EPSG:4326", always_xy=True).transform(
            *ds.xy(ds.height // 2, ds.width // 2))
    return f"EPSG:{(32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1}"


class Scene:
    """Растр SpOr в UTM на res_m (WarpedVRT; уменьшение — average, иначе bilinear), окна RGB uint8 с нулями за краем."""

    def __init__(self, path: Path, crs: str, res_m: float = 0.5):
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.vrt import WarpedVRT
        from rasterio.warp import calculate_default_transform

        self.src = rasterio.open(path)
        self.crs = crs
        tr, w, h = calculate_default_transform(self.src.crs, crs, self.src.width, self.src.height,
                                               *self.src.bounds, resolution=res_m)
        try:
            src_m = abs(calculate_default_transform(self.src.crs, crs, self.src.width, self.src.height,
                                                    *self.src.bounds)[0].a)
        except Exception:
            src_m = res_m
        rs = Resampling.average if src_m < res_m else Resampling.bilinear
        self.vrt = WarpedVRT(self.src, crs=crs, transform=tr, width=w, height=h, resampling=rs, nodata=0)
        self.W, self.H, self.T = w, h, tr
        self.bands = list(range(1, min(3, self.src.count) + 1))
        self.stretch = None
        if self.src.dtypes[0] != "uint8":
            ov = self.src.read(1, out_shape=(min(1024, self.src.height), min(1024, self.src.width)))
            v = ov[ov > 0]
            self.stretch = tuple(np.percentile(v, [1, 99])) if v.size else (0, 1)

    def read(self, x0: int, y0: int, n: int) -> np.ndarray:
        from rasterio.windows import Window

        a = np.zeros((len(self.bands), n, n), dtype=self.src.dtypes[0])
        cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(self.W, x0 + n), min(self.H, y0 + n)
        if cx1 > cx0 and cy1 > cy0:
            a[:, cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0] = self.vrt.read(
                self.bands, window=Window(cx0, cy0, cx1 - cx0, cy1 - cy0))
        if a.shape[0] < 3:
            a = np.repeat(a[:1], 3, 0)
        if self.stretch:
            lo, hi = self.stretch
            a = np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1e-6) * 255, 0, 255)
        return np.ascontiguousarray(a.astype(np.uint8).transpose(1, 2, 0))

    def close(self) -> None:
        self.vrt.close()
        self.src.close()


def windows(W: int, H: int, tile: int, overlap: int) -> list[tuple[int, int]]:
    step = tile - overlap
    xs = list(range(0, max(1, W - overlap), step)) or [0]
    ys = list(range(0, max(1, H - overlap), step)) or [0]
    return [(x, y) for y in ys for x in xs]


# ── модель ───────────────────────────────────────────────────────────────────────────────────────────────
#: Переименование ключей чекпойнта D-FINE → модули transformers 5.x. Копия правил "d_fine" из
#: transformers/conversion_mapping.py (5.21.0), которые from_pretrained применяет сам. Сверено побитово с
#: from_pretrained на dfine_f1 bot-9 (bot-7/work6/load_smoke.log).
_KEY_RULES = (
    ("out_proj", "o_proj"),
    (r"layers\.(\d+)\.fc1", r"layers.\1.mlp.layers.0"),
    (r"layers\.(\d+)\.fc2", r"layers.\1.mlp.layers.1"),
    (r"encoder\.encoder\.(\d+)\.layers", r"encoder.aifi.\1.layers"),
)


def _rename_keys(sd: dict) -> dict:
    import re

    out = {}
    for k, v in sd.items():
        for pat, rep in _KEY_RULES:
            k = re.sub(pat, rep, k)
        out[k] = v
    return out


def load_models(st: dict, root: Path, device) -> list:
    """DFineForObjectDetection из каталогов весов (config.json + model.safetensors), только локально:
    конфиг → пустая модель → load_state_dict. Без from_pretrained/хаба (check_submission: нет сетевых
    вызовов). Проверки как check_model у bot-9: 5 классов, нет meta-тензоров, нет неожиданных ключей,
    пропущенные ключи допустимы только связанные (tied) копии уже загруженных тензоров."""
    import torch
    from safetensors.torch import load_file
    from transformers import DFineConfig, DFineForObjectDetection

    models = []
    for rel in st["weights"]:
        p = Path(rel)
        p = p if p.is_absolute() else Path(root) / p
        cfg = DFineConfig.from_json_file(str(p / "config.json"))
        if int(cfg.num_labels) != len(CLASSES):
            raise RuntimeError(f"D-FINE {p}: num_labels {cfg.num_labels} != {len(CLASSES)}")
        with torch.device("cpu"):
            m = DFineForObjectDetection(cfg)
        sd = _rename_keys(load_file(str(p / "model.safetensors"), device="cpu"))
        res = m.load_state_dict(sd, strict=False)
        if res.unexpected_keys:
            raise RuntimeError(f"D-FINE {p}: лишние ключи {res.unexpected_keys[:3]}")
        if res.missing_keys:
            m.tie_weights()
            own = dict(list(m.named_parameters(remove_duplicate=False)) + list(m.named_buffers(remove_duplicate=False)))
            ptr = {own[k].data_ptr() for k in sd if k in own}
            loose = [k for k in res.missing_keys if k not in own or own[k].data_ptr() not in ptr]
            if loose:
                raise RuntimeError(f"D-FINE {p}: нет весов для {len(loose)} тензоров ({', '.join(loose[:3])})")
        bad = [n for n, t in list(m.named_parameters()) + list(m.named_buffers()) if t.is_meta]
        if bad:
            raise RuntimeError(f"D-FINE {p}: {len(bad)} тензоров на meta ({', '.join(bad[:3])})")
        models.append(m.eval().to(device))
    return models


def detect_scene(models: list, scene, st: dict, device, keep_going=lambda: True) -> tuple[list, int, int]:
    """Окна сцены → по каждой модели (xyxy px, score, class) после NMS по классам. Окно читается один раз
    на все модели. Возвращает (по моделям список (B, L, C) numpy, окон сделано, окон всего)."""
    import torch
    from torchvision.ops import batched_nms

    T, bs = int(st["tile"]), int(st["batch"])
    wins = windows(scene.W, scene.H, T, int(st["overlap"]))
    acc = [([], [], []) for _ in models]
    done = 0
    use_amp = bool(st["fp16"]) and getattr(device, "type", str(device)) == "cuda"
    with torch.no_grad():
        for b in range(0, len(wins), bs):
            if not keep_going():
                break
            chunk, imgs = [], []
            for x, y in wins[b:b + bs]:
                im = scene.read(x, y, T)
                if (im.max(2) == 0).mean() > float(st["skip_empty"]):
                    continue
                chunk.append((x, y))
                imgs.append(im)
            done = min(b + bs, len(wins))
            if not imgs:
                continue
            px = torch.from_numpy(np.stack(imgs).transpose(0, 3, 1, 2).astype(np.float32) / 255).to(device)
            for mi, model in enumerate(models):
                for flip in ((False, True) if st["flip_tta"] else (False,)):
                    inp = px.flip(3) if flip else px
                    with torch.autocast(getattr(device, "type", "cpu"), dtype=torch.float16, enabled=use_amp):
                        o = model(pixel_values=inp)
                    prob = o.logits.float().sigmoid()
                    bx = o.pred_boxes.float()
                    if flip:
                        bx = bx.clone()
                        bx[..., 0] = 1 - bx[..., 0]
                    Q, C = prob.shape[1:]
                    sc_, idx = prob.flatten(1).topk(min(int(st["max_det"]), Q * C), 1)
                    for i, (x, y) in enumerate(chunk):
                        keep = sc_[i] >= float(st["min_score"])
                        q, c = idx[i][keep] // C, idx[i][keep] % C
                        bb = bx[i][q] * T
                        xyxy = torch.stack([bb[:, 0] - bb[:, 2] / 2 + x, bb[:, 1] - bb[:, 3] / 2 + y,
                                            bb[:, 0] + bb[:, 2] / 2 + x, bb[:, 1] + bb[:, 3] / 2 + y], 1)
                        acc[mi][0].append(xyxy.cpu())
                        acc[mi][1].append(sc_[i][keep].cpu())
                        acc[mi][2].append(c.cpu())
    res = []
    for B, L, Cc in acc:
        if not B:
            res.append((np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)))
            continue
        B, L, Cc = torch.cat(B), torch.cat(L), torch.cat(Cc)
        k = batched_nms(B, L, Cc, float(st["nms_iou"]))
        res.append((B[k].numpy(), L[k].numpy(), Cc[k].numpy().astype(int)))
    return res, done, len(wins)


def boxes_to_ellipses(B, L, Cc, transform, crs: str, region: str, source: str, model_i: int) -> list[dict]:
    """Боксы в пикселях сцены → 32-угольные эллипсы в EPSG:3857, как у bot-9."""
    from pyproj import Transformer
    from shapely.geometry import Polygon

    if len(L) == 0:
        return []
    to3857 = Transformer.from_crs(crs, "EPSG:3857", always_xy=True).transform
    t = np.linspace(0, 2 * np.pi, 33)[:-1]
    a, b, c0, d, e, f0 = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
    out = []
    for (x0, y0, x1, y1), s, c in zip(B, L, Cc):
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        rx, ry = max((x1 - x0) / 2, 0.5), max((y1 - y0) / 2, 0.5)
        px, py = cx + rx * np.cos(t), cy + ry * np.sin(t)
        gx, gy = to3857(a * px + b * py + c0, d * px + e * py + f0)
        out.append({"geom": Polygon(np.column_stack([gx, gy])), "class_name": CLASSES[int(c)],
                    "confidence": round(float(s), 5), "region_name": region, "source": source,
                    "original_crs": crs, "model": model_i})
    return out


def _device(st: dict):
    import torch

    if st["device"] != "auto":
        return torch.device(st["device"])
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def region_feats(reg, cfg: dict, guard, cache: dict, root: Path, crs_for=None, soft_ok=None) -> tuple[list[dict], bool]:
    """Объекты D-FINE участка и флаг «все растры SpOr пройдены полностью» (нужен для drop_groups).

    cache — общий словарь прогона (модели грузятся один раз). Ошибка или нехватка времени → что успели.
    """
    st = settings(cfg)
    paths = list((getattr(reg, "rasters", {}) or {}).get(MARKUP, []))
    if not paths:
        return [], False
    if "models" not in cache:
        cache["models"], cache["device"] = [], None
        ready, missing = thresholds_ready(cfg)
        if st["require_thresholds"] and not ready:
            log.error("sp_dfine: нет порогов [thresholds_by_markup.sp_dfine] для %s — D-FINE не запускается",
                      ", ".join(missing))
        elif not st["weights"]:
            log.error("sp_dfine: weights пуст — D-FINE не запускается")
        else:
            try:
                dev = _device(st)
                t0 = time.monotonic()
                cache["models"], cache["device"] = load_models(st, root, dev), dev
                log.info("sp_dfine: моделей %d на %s за %.1f с", len(cache["models"]), dev, time.monotonic() - t0)
            except Exception:
                log.exception("sp_dfine: модели не загрузились, D-FINE пропущен")
    models = cache["models"]
    if not models:
        return [], False
    dets: list[dict] = []
    complete = True
    for path in paths:
        if guard is not None and not guard.ok(60):
            complete = False
            break
        if soft_ok is not None and not soft_ok(f"D-FINE {Path(path).name}", float(st["soft_need_s"])):
            complete = False
            break
        sc = None
        try:
            crs = utm_for(Path(path), crs_for(reg) if crs_for else None)
            sc = Scene(Path(path), crs, float(st["res_m"]))
            ok = (lambda: guard.window_ok()) if guard is not None and hasattr(guard, "window_ok") else (lambda: True)
            t0 = time.monotonic()
            res, done, total = detect_scene(models, sc, st, cache["device"], ok)
            complete &= done >= total
            for mi, (B, L, Cc) in enumerate(res):
                dets += boxes_to_ellipses(B, L, Cc, sc.T, crs, reg.name, Path(path).name, mi)
            log.info("sp_dfine: %s %dx%d px, окон %d/%d, боксов %d, %.1f с", Path(path).name, sc.W, sc.H, done,
                     total, sum(len(r[1]) for r in res), time.monotonic() - t0)
        except Exception:
            complete = False
            log.exception("sp_dfine: %s пропущен", Path(path).name)
        finally:
            if sc is not None:
                sc.close()
    return as_features(dets, st, len(models)), complete
