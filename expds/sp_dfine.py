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
* ``drop_min_det`` (bot-2) — R применяется только к участкам с ≥N объектов ``sp_dfine`` (по умолчанию 1; багфикс: при 0 детекций v2 SpOr сохраняются);
* ``agreement_m`` (bot-2 A3/A4) — если >0, оставлять D-FINE только если центр ≤ agreement_m м от уже имеющегося sp_or не-sp_dfine; 0 = OFF;
* ``EXPDS_DFINE_DUMP`` (env, bot-2) — каталог дампа сырых боксов (pixel + EPSG:3857), GT и
  ``stages.json`` (счётчики raw→ellipse→features→would_pass_thr); по умолчанию ВЫКЛ.;
* ``EXPDS_STRICT=1`` (bot-2 A3 / bot-5 v4) — при ``enabled=true`` и сбое импорта/загрузки D-FINE —
  громкий отказ (ERROR + WARNING-файл + raise), а не тихий fallback «как v2»; по умолчанию ВЫКЛ.
  Вендор ``inference/lib/vendor`` — manylinux/cp312 (площадка/Kaggle/docker); на Windows-host без docker
  import падает (``regex``/``tokenizers`` .so) → docker ``pantum/n7q`` или env transformers==5.21.0;
* ансамбль фолдов: ``ensemble = "max"`` (кластер C берёт лучший из всех моделей, как NMS) или ``"mean"``
  (среднее по моделям, отсутствующая модель = 0, как score-averaging в WBF). Ни один не измерен честно —
  у фолдовых моделей нет общего OOF; по умолчанию одна модель.

torch / transformers импортируются лениво, только при enabled = true.
"""
from __future__ import annotations

import os
import json

import logging
import math
import os
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
    # bot-2: R применяется только к участкам с ≥ drop_min_det объектов threshold_group=sp_dfine (багфикс, ВКЛ.).
    "drop_min_det": 1,
    # bot-2 A3: держать D-FINE только рядом с уже найденным sp_or (U-Net и др.); 0 = OFF.
    # На A3 Docker все 13 pass_thr ≥70 м от U-Net → agreement_m=50 отсекает FP без CRS-фикса.
    "agreement_m": 0.0,
    "require_thresholds": True,
    "soft_need_s": 150.0,     # запас до мягкого потолка на один растр
    "device": "auto",
    # bot-10 (Murder Board 29.10): при enabled = true отсутствие DFine* в transformers, весов или порогов —
    # ЖЁСТКАЯ ошибка прогона (иначе v3 молча превращается в v2). strict = false — старое мягкое поведение.
    "strict": True,
    "transformers_version": "5.21.0",   # пин: из колёс bot-9 submission_dfine (requirements-dfine.txt)
}


def settings(cfg: dict) -> dict:
    own = dict(cfg.get("sp_dfine") or {})
    return dict(DEFAULTS, **own)


def enabled(cfg: dict) -> bool:
    return bool((cfg.get("sp_dfine") or {}).get("enabled", False))


class DFineUnavailable(RuntimeError):
    """D-FINE включён, но не может работать (нет классов transformers, весов, порогов или версия не та)."""


def preflight(cfg: dict, root: Path) -> dict:
    """Проверка до начала прогона при [sp_dfine] enabled = true (bot-10, Murder Board 29.10).

    Возвращает сведения для протокола; при strict = true на любой проблеме бросает DFineUnavailable,
    чтобы v3 не превратился молча в v2 (контейнер без DFine*, пустые weights, нет порогов, чужая версия).
    """
    st = settings(cfg)
    info: dict = {"enabled": True, "strict": bool(st["strict"]), "transformers": "нет", "problems": []}
    problems = info["problems"]
    ready, missing = thresholds_ready(cfg)
    if st["require_thresholds"] and not ready:
        problems.append("нет порогов [thresholds_by_markup.sp_dfine] для " + ", ".join(missing))
    if not st["weights"]:
        problems.append("weights пуст")
    for rel in st["weights"]:
        p = Path(rel)
        p = p if p.is_absolute() else Path(root) / p
        for name in ("config.json", "model.safetensors"):
            if not (p / name).is_file():
                problems.append(f"нет {p / name}")
    try:
        _vendor_path(root)
        import transformers
        info["transformers"] = str(getattr(transformers, "__version__", "?"))
        from transformers import DFineConfig, DFineForObjectDetection  # noqa: F401
    except Exception as e:  # ImportError или сломанный transformers
        problems.append(f"transformers без DFineConfig/DFineForObjectDetection ({type(e).__name__}: {e})")
    pin = st.get("transformers_version")
    if pin and info["transformers"] not in ("нет", pin):
        problems.append(f"transformers {info['transformers']} != пин {pin}")
    if problems:
        msg = "sp_dfine включён, но не готов: " + "; ".join(problems)
        if st["strict"]:
            raise DFineUnavailable(msg)
        log.error("%s (strict = false: D-FINE будет пропущен, прогон = v2)", msg)
    return info


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


def _feat_xy(f: dict) -> tuple[float, float] | None:
    g = f.get("geom")
    if g is None:
        return None
    try:
        c = g.centroid
        return float(c.x), float(c.y)
    except Exception:
        return None


def filter_agreement(feats: list[dict], anchors: list[dict], agreement_m: float) -> list[dict]:
    """Оставить D-FINE, чей центр ≤ ``agreement_m`` м (EPSG:3857 ground) от якоря sp_or.

    Якорь — объект с ``markup_type=sp_or`` и ``threshold_group != sp_dfine`` (U-Net / прочие).
    ``agreement_m <= 0`` → без фильтра (OFF). Пустые якоря → все D-FINE отбрасываются при >0.
    """
    m = float(agreement_m or 0.0)
    if m <= 0:
        return feats
    pts: list[tuple[float, float]] = []
    for a in anchors or ():
        if a.get("markup_type") != MARKUP:
            continue
        if a.get("threshold_group") == GROUP:
            continue
        xy = _feat_xy(a)
        if xy is not None:
            pts.append(xy)
    if not pts:
        return []
    kept = []
    for f in feats:
        xy = _feat_xy(f)
        if xy is None:
            continue
        x, y = xy
        # ground metres ≈ Euclidean * ground_scale(y) in WebMercator
        scale = ground_scale(y)
        if any(math.hypot(x - ax, y - ay) * scale <= m for ax, ay in pts):
            kept.append(f)
    return kept


# ── растр ────────────────────────────────────────────────────────────────────────────────────────────────
def _crs_is_geographic(crs: str) -> bool:
    """True, если CRS географический (градусы). WarpedVRT с res_m=0.5° уничтожил бы метраж."""
    try:
        from rasterio.crs import CRS
        return bool(CRS.from_user_input(crs).is_geographic)
    except Exception:
        s = (crs or "").upper()
        return "4326" in s or s in ("OGC:CRS84", "CRS84", "WGS84")


def utm_for(path: Path, fallback_crs: str | None) -> str:
    """Метрическая CRS сцены: UTM.json участка, если она проецированная; иначе зона UTM по центру растра.
    bot-6 crsfix (находка bot-9): у 008_Герите в UTM.json «urn:ogc:def:crs:EPSG::4326» (один из 210 участков) —
    сцена строилась с шагом 0,5° вместо 0,5 м (1×1 px, 0 предсказаний). Географическая CRS не берётся."""
    import rasterio
    from pyproj import Transformer

    from .crs import is_geographic

    if fallback_crs and not is_geographic(fallback_crs):
        return fallback_crs
    if fallback_crs:   # страховка (bot-7 v3fix + bot-6 crsfix): градусы мимо read_utm_json — зона UTM по центру растра
        log.warning("[crs] %s: CRS участка %s географическая — сцена в зоне UTM по центру растра", Path(path).name, fallback_crs)
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
        if _crs_is_geographic(crs):
            self.src.close()
            raise ValueError(f"sp_dfine.Scene: CRS {crs!r} географический — нужен проекционный UTM (res_m в метрах)")
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


VENDOR = "inference/lib/vendor"   # распакованные колёса bot-9 submission_dfine/wheels (transformers 5.21.0 и зависимости, cp312)
#: закреплённая версия transformers (вендор); другая версия — только если совпадает ровно (напр. venv ПК с колёсами win)
TRANSFORMERS_PIN = "5.21.0"
#: маркер в логе: D-FINE включён, но недоступен → выдача как v2; check_submission по нему даёт FAIL
UNAVAILABLE = "SP_DFINE_UNAVAILABLE"
#: пакеты вендора: при сбое импорта из вендора их полу-импортированные модули вычищаются из sys.modules
_VENDOR_PKGS = ("transformers", "tokenizers", "regex", "typer", "rich", "markdown_it", "mdurl", "pygments",
                "shellingham", "annotated_doc")


#: bot-7 (29.10, PC e2e бота 1 в образе pantum/n7q:20261024): у tqdm в образе нет версии в метаданных
#: (importlib.metadata.version → None), и transformers.dependency_versions_check падал при импорте
#: («Unable to compare versions for tqdm>=4.60: need=4.60 found=None»). Пакеты, версии которых transformers
#: 5.21.0 читает из метаданных при импорте (dependency_versions_check, is_*_available): имя дистрибутива
#: (нормализованное) → имя модуля. Для них при отсутствии/пустой версии метаданных берётся module.__version__.
_META_DEPS = {"tqdm": "tqdm", "regex": "regex", "packaging": "packaging", "filelock": "filelock", "numpy": "numpy",
              "tokenizers": "tokenizers", "huggingface-hub": "huggingface_hub", "safetensors": "safetensors",
              "pyyaml": "yaml", "torch": "torch", "torchvision": "torchvision", "typer": "typer",
              "pillow": "PIL", "accelerate": "accelerate", "rich": "rich"}
#: маркер WARNING в логе: версия пакета взята из module.__version__, а не из метаданных
DEPMETA = "SP_DFINE_DEPMETA"
#: WARNING-файл / строка STATUS: D-FINE включён, но не загрузился (не тихий fallback) — port from 829ea7d
WARN_MARKER = "WARNING SP_DFINE_UNAVAILABLE"
_META_USED: dict = {}


def strict_mode() -> bool:
    """``EXPDS_STRICT=1`` — при сбое загрузки D-FINE (и enabled) поднимать исключение, не глотать."""
    return (os.environ.get("EXPDS_STRICT") or "").strip() in ("1", "true", "TRUE", "yes", "YES")


def status_dir() -> Path | None:
    """Куда писать WARNING-файл: ``EXPDS_STATUS_DIR`` или ``EXPDS_DFINE_DUMP`` (если задан)."""
    raw = (os.environ.get("EXPDS_STATUS_DIR") or os.environ.get("EXPDS_DFINE_DUMP") or "").strip()
    if not raw:
        return None
    p = Path(raw)
    p.mkdir(parents=True, exist_ok=True)
    return p


def vendor_platform_hint(vendor: Path) -> str:
    """Подсказка при сбое импорта: Linux-.so на Windows / нет transformers в вендоре."""
    import sys
    if not vendor.is_dir():
        return (f"vendor отсутствует: ожидался {vendor.as_posix()} "
                f"(упакуйте inference/lib/vendor; на Kaggle/docker Linux OK)")
    has_tr = (vendor / "transformers").is_dir()
    so = list(vendor.rglob("*.so"))[:3]
    pyd = list(vendor.rglob("*.pyd"))[:1]
    bits = []
    if not has_tr:
        bits.append("в вендоре нет каталога transformers/")
    if sys.platform.startswith("win") and so and not pyd:
        bits.append(
            f"вендор Linux (.so, напр. {so[0].name}): на Windows-host не грузится — "
            f"запускайте в docker pantum/n7q (как площадка/Kaggle) или поставьте "
            f"transformers=={TRANSFORMERS_PIN} в env"
        )
    return "; ".join(bits)


def write_unavailable_warning(reason: str, *, detail: str = "") -> Path | None:
    """ERROR в лог уже вызван вызывающим; здесь — WARNING-файл для STATUS.txt раннера."""
    log.error("%s: %s", WARN_MARKER, reason)
    d = status_dir()
    if d is None:
        return None
    path = d / "SP_DFINE_WARNING.txt"
    body = f"{WARN_MARKER}: {reason}\n"
    if detail:
        body += detail.rstrip() + "\n"
    path.write_text(body, encoding="utf-8")
    (d / "STATUS_WARNING.txt").write_text(f"{WARN_MARKER}: {reason}\n", encoding="utf-8")
    return path


def _norm_dist(name: str) -> str:
    import re
    return re.sub(r"[-_.]+", "-", str(name)).lower()


def _mod_version(mod_name: str | None):
    """module.__version__, если модуль импортируется и версия похожа на версию (tqdm без метаданных даёт None/UNKNOWN)."""
    import importlib
    if mod_name is None:
        return None
    try:
        mv = getattr(importlib.import_module(mod_name), "__version__", None)
    except Exception:  # noqa: BLE001 — модуля нет
        return None
    mv = str(mv) if mv is not None else ""
    return mv if mv[:1].isdigit() else None


def _dir_version(distribution_name: str):
    """Версия из имени каталога метаданных (<имя>-<версия>.dist-info / .egg-info), если METADATA пуст или нет."""
    import importlib.metadata as md
    import re
    try:
        path = getattr(md.distribution(distribution_name), "_path", None)
    except Exception:  # noqa: BLE001
        return None
    m = re.match(r"^[^-]+-([0-9][^-]*?)(?:-py[0-9.]+)?\.(?:dist|egg)-info$", Path(str(path or "")).name)
    return m.group(1) if m else None


def _depmeta_warn(name: str, how: str, value) -> None:
    if name not in _META_USED:
        _META_USED[name] = value
        log.warning("%s: %s %s", DEPMETA, name, how)


def _metadata_shim() -> None:
    """Защита импорта transformers от неполных метаданных пакетов образа (только в этом процессе, один раз).

    1) importlib.metadata.version для _META_DEPS: если метаданных нет (PackageNotFoundError) или в них нет
       версии (None), версия берётся из module.__version__ (модуль импортируется), затем из имени каталога
       <имя>-<версия>.dist-info; WARNING SP_DFINE_DEPMETA с именем пакета. Иначе — как без шима.
    2) крючок импорта transformers.utils.versions: require_version (dependency_versions_check при импорте) при
       «Unable to compare versions … found=None» или «distribution was not found» для пакета из _META_DEPS,
       модуль которого импортируется, — WARNING SP_DFINE_DEPMETA вместо исключения. Слишком старая версия
       (ImportError) и неимпортируемый модуль по-прежнему роняют импорт → SP_DFINE_UNAVAILABLE."""
    import importlib.abc
    import importlib.metadata as md
    import sys
    if not getattr(md.version, "_b7_shim", False):
        orig = md.version

        def version(distribution_name):
            missing = False
            try:
                got = orig(distribution_name)
            except md.PackageNotFoundError:
                got, missing = None, True
            if got:
                return got
            key = _norm_dist(distribution_name)
            if key not in _META_DEPS:
                if missing:
                    raise md.PackageNotFoundError(distribution_name)
                return got
            why = "нет dist-info" if missing else "version=None в метаданных"
            mv = _mod_version(_META_DEPS[key])
            if mv:
                _depmeta_warn(distribution_name, f"{why} — взята {_META_DEPS[key]}.__version__ = {mv}", mv)
                return mv
            dv = None if missing else _dir_version(distribution_name)
            if dv:
                _depmeta_warn(distribution_name, f"{why} — взята версия из имени каталога метаданных = {dv}", dv)
                return dv
            if missing:
                raise md.PackageNotFoundError(distribution_name)
            return got

        version._b7_shim = True
        version._b7_orig = orig
        md.version = version

    if "transformers.utils.versions" in sys.modules:
        _patch_versions(sys.modules["transformers.utils.versions"])
    if not any(isinstance(f, _VersionsHook) for f in sys.meta_path):
        sys.meta_path.insert(0, _VersionsHook())


def _patch_versions(mod) -> None:
    """transformers.utils.versions.require_version: см. _metadata_shim, п. 2."""
    import importlib.metadata as md
    import re
    orig = getattr(mod, "require_version", None)
    if orig is None or getattr(orig, "_b7_shim", False):
        return

    def require_version(requirement, hint=None):
        try:
            return orig(requirement, hint)
        except (ValueError, md.PackageNotFoundError) as exc:
            msg = str(exc)
            if not ("found=None" in msg or isinstance(exc, md.PackageNotFoundError)):
                raise
            m = re.match(r"^\s*([A-Za-z0-9_.\-]+)", str(requirement))
            key = _norm_dist(m.group(1)) if m else ""
            if key not in _META_DEPS or not _importable(_META_DEPS[key]):
                raise
            _depmeta_warn(f"require:{key}", f"«{requirement}»: версия не определяется ({type(exc).__name__}), модуль "
                          f"{_META_DEPS[key]} импортируется — проверка версии transformers пропущена", None)
            return None

    require_version._b7_shim = True
    mod.require_version = require_version


def _importable(mod_name: str) -> bool:
    import importlib
    try:
        importlib.import_module(mod_name)
        return True
    except Exception:  # noqa: BLE001
        return False


try:
    import importlib.abc as _iabc
    _MetaBase = _iabc.MetaPathFinder
except Exception:  # pragma: no cover
    _MetaBase = object


class _VersionsHook(_MetaBase):
    """После исполнения transformers.utils.versions подменяет require_version (_patch_versions) — до того, как
    dependency_versions_check возьмёт эти имена."""

    def find_spec(self, name, path=None, target=None):
        if name != "transformers.utils.versions":
            return None
        import sys
        spec = None
        for f in sys.meta_path:
            if f is self or not hasattr(f, "find_spec"):
                continue
            spec = f.find_spec(name, path, target)
            if spec is not None:
                break
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return spec
        inner = spec.loader.exec_module

        def exec_module(module, _inner=inner):
            _inner(module)
            _patch_versions(module)

        try:
            spec.loader.exec_module = exec_module
        except Exception:  # noqa: BLE001 — не вышло: как без крючка
            pass
        return spec


def _vendor_path(root: Path) -> None:
    """transformers нет в образе: при наличии VENDOR в архиве ставим его первым в sys.path (pip на площадке нет)."""
    import os
    import sys
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    v = Path(root) / VENDOR
    if v.is_dir():
        if str(v) in sys.path:
            sys.path.remove(str(v))
        sys.path.insert(0, str(v))


def _purge(pkgs, keep_under: Path | None = None) -> list[str]:
    """Выгрузить модули пакетов pkgs из sys.modules (кроме загруженных из keep_under). Возвращает выгруженные."""
    import sys
    gone = []
    for name in list(sys.modules):
        top = name.split(".", 1)[0]
        if top not in pkgs:
            continue
        mod = sys.modules.get(name)
        f = getattr(mod, "__file__", None) or ""
        if keep_under is not None and f and Path(f).resolve().is_relative_to(keep_under.resolve()):
            continue
        sys.modules.pop(name, None)
        gone.append(name)
    return gone


def import_dfine(root: Path) -> tuple:
    """Классы D-FINE строго закреплённой версии. Порядок:
    1) уже загруженный НЕ из вендора transformers (старый из site-packages / чужой импорт) выгружается;
    0) шим importlib.metadata.version (_metadata_shim): у зависимостей без версии в метаданных (tqdm в образе
       площадки) версия берётся из module.__version__, WARNING SP_DFINE_DEPMETA; модуля нет → громкий отказ;
    2) вендор архива первым в sys.path → import (в образе python:3.12 linux так и должно быть; в вендоре и tqdm
       с dist-info, его метаданные находятся раньше метаданных образа);
    3) сбой (на Windows/другом python не грузятся .so regex/tokenizers из вендора) → вендор вычищается из sys.path
       и sys.modules, пробуем transformers окружения, но только версии ровно TRANSFORMERS_PIN;
    иначе ImportError с причинами обеих попыток. Возвращает (DFineConfig, DFineForObjectDetection, info)."""
    import importlib
    import sys
    v = Path(root) / VENDOR
    info: dict = {"pin": TRANSFORMERS_PIN, "vendor": str(v), "vendor_present": v.is_dir(), "errors": []}
    tr = sys.modules.get("transformers")
    if tr is not None and not (v.is_dir() and Path(getattr(tr, "__file__", "") or "/").resolve().is_relative_to(v.resolve())):
        info["purged_preloaded"] = getattr(tr, "__version__", "?")
        _purge(("transformers",))
    if v.is_dir():   # и чужие tokenizers/regex, загруженные до нас, — иначе вендорный transformers возьмёт их
        _purge(("tokenizers", "regex"), keep_under=v)
    _metadata_shim()   # до импорта transformers: dependency_versions_check читает версии из метаданных
    attempts = (["vendor"] if v.is_dir() else []) + ["env"]
    for src in attempts:
        if src == "vendor":
            _vendor_path(root)
        else:
            if v.is_dir():
                while str(v) in sys.path:
                    sys.path.remove(str(v))
                _purge(_VENDOR_PKGS)
            import os
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        importlib.invalidate_caches()
        try:
            import transformers
            ver = getattr(transformers, "__version__", "?")
            if ver != TRANSFORMERS_PIN:
                raise ImportError(f"transformers {ver} != закреплённой {TRANSFORMERS_PIN} ({transformers.__file__})")
            from transformers import DFineConfig, DFineForObjectDetection
            info.update(source=src, version=ver, file=str(transformers.__file__), meta_fallback=dict(_META_USED))
            return DFineConfig, DFineForObjectDetection, info
        except Exception as exc:  # noqa: BLE001 — причина пишется в лог/FAIL
            info["errors"].append(f"{src}: {type(exc).__name__}: {str(exc)[:300]}")
            _purge(("transformers",))
    hint = vendor_platform_hint(v)
    if hint:
        info["errors"].append(f"hint: {hint}")
    info["platform"] = __import__("sys").platform
    raise ImportError(f"{UNAVAILABLE}: D-FINE недоступен — " + " | ".join(info["errors"]))


def load_models(st: dict, root: Path, device) -> list:
    """DFineForObjectDetection из каталогов весов (config.json + model.safetensors), только локально:
    конфиг → пустая модель → load_state_dict. Без from_pretrained/хаба (check_submission: нет сетевых
    вызовов). Проверки как check_model у bot-9: 5 классов, нет meta-тензоров, нет неожиданных ключей,
    пропущенные ключи допустимы только связанные (tied) копии уже загруженных тензоров."""
    import torch
    from safetensors.torch import load_file
    DFineConfig, DFineForObjectDetection, info = import_dfine(root)
    log.info("sp_dfine: transformers %s из %s (%s)", info["version"], info["source"], info["file"])

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



def dump_root() -> Path | None:
    """Каталог дампа из ``EXPDS_DFINE_DUMP`` (пусто / не задан → ``None``, фича ВЫКЛ.)."""
    raw = (os.environ.get("EXPDS_DFINE_DUMP") or "").strip()
    if not raw:
        return None
    p = Path(raw)
    p.mkdir(parents=True, exist_ok=True)
    return p


def count_would_pass_thr(feats: list[dict], cfg: dict) -> int:
    """Сколько объектов sp_dfine прошли бы ``post.apply_thresholds`` с текущим ``[thresholds_by_markup.sp_dfine]``.

    Нужен dump-диагностике A≡B: если raw>0, а would_pass_thr=0 — порог/группа срезает всё до finish.
    """
    from . import post

    by = cfg.get("thresholds_by_markup") or {}
    tau = post.threshold_fn(cfg.get("thresholds", {}), cfg.get("default_threshold", 0.15), by)
    return sum(1 for f in feats if f.get("confidence", 1.0) >= tau(f))


def append_stage(out_dir: Path, row: dict) -> Path:
    """Дописывает одну запись в ``stages.jsonl`` (одна строка = один участок / фаза)."""
    path = out_dir / "stages.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False))
        fh.write("\n")
    return path


def write_stages_summary(out_dir: Path, rows: list[dict], **extra) -> Path:
    """Итоговый ``stages.json``: сумма счётчиков по участкам + extra (n_models, thresholds_ready…)."""
    keys = ("n_raw_boxes", "n_ellipses", "n_features", "n_would_pass_thr", "n_paths", "n_paths_done")
    totals = {k: sum(int(r.get(k, 0) or 0) for r in rows) for k in keys}
    payload = {"regions": rows, "totals": totals, **extra}
    path = out_dir / "stages.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_boxes_dump(out_dir: Path, *, region: str, source: str, crs: str, transform,
                     model_i: int, B, L, Cc, ellipses: list[dict]) -> Path:
    """Пишет JSON: пиксельные xyxy + преобразованные центры/bbox в EPSG:3857 (+ crs сцены).

    Нужен PC-прогону для проверки CRS-гипотезы (pixel → scene CRS → 3857 vs GT 3857).
    """
    from pyproj import Transformer

    a, b, c0, d, e, f0 = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
    to3857 = Transformer.from_crs(crs, "EPSG:3857", always_xy=True).transform
    rows = []
    for i, ((x0, y0, x1, y1), s, c) in enumerate(zip(B, L, Cc)):
        cx, cy = (float(x0) + float(x1)) / 2, (float(y0) + float(y1)) / 2
        # углы bbox в CRS сцены → 3857
        corners_src = [
            (a * float(x0) + b * float(y0) + c0, d * float(x0) + e * float(y0) + f0),
            (a * float(x1) + b * float(y0) + c0, d * float(x1) + e * float(y0) + f0),
            (a * float(x1) + b * float(y1) + c0, d * float(x1) + e * float(y1) + f0),
            (a * float(x0) + b * float(y1) + c0, d * float(x0) + e * float(y1) + f0),
        ]
        corners_3857 = [to3857(x, y) for x, y in corners_src]
        cx_src, cy_src = a * cx + b * cy + c0, d * cx + e * cy + f0
        cx_3857, cy_3857 = to3857(cx_src, cy_src)
        ell = ellipses[i] if i < len(ellipses) else None
        eb = list(ell["geom"].bounds) if ell is not None else None
        rows.append({
            "i": i,
            "class_name": CLASSES[int(c)] if int(c) < len(CLASSES) else str(int(c)),
            "score": float(s),
            "xyxy_px": [float(x0), float(y0), float(x1), float(y1)],
            "center_px": [cx, cy],
            "center_scene_crs": [cx_src, cy_src],
            "center_3857": [cx_3857, cy_3857],
            "bbox_corners_3857": [[float(x), float(y)] for x, y in corners_3857],
            "ellipse_bounds_3857": eb,
        })
    payload = {
        "region": region,
        "source": source,
        "scene_crs": crs,
        "target_crs": "EPSG:3857",
        "model_i": model_i,
        "n": len(rows),
        "affine_a_b_c_d_e_f": [a, b, c0, d, e, f0],
        "boxes": rows,
    }
    safe_src = source.replace("/", "_").replace("\\", "_")
    path = out_dir / f"{region}__{safe_src}__m{model_i}_boxes.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_gt_dump(out_dir: Path, reg) -> Path | None:
    """GT участка (SpOr и пр.) в EPSG:3857 через ``expds.gt.read_annotation``; нет аннотаций → None."""
    anns = list(getattr(reg, "annotations", None) or [])
    if not anns:
        return None
    # importlib — чтобы AST-граф инференса не тянул train-only gt.py (test_train_only_modules…).
    import importlib
    from shapely.geometry import mapping
    read_annotation = importlib.import_module(f"{__package__}.gt").read_annotation

    feats = []
    report: list[dict] = []
    for path, cls, mt in anns:
        for it in read_annotation(Path(path), cls, mt, reg, report=report):
            g = it["geom"]
            feats.append({
                "type": "Feature",
                "properties": {
                    "class_name": it["class_name"],
                    "markup_type": it["markup_type"],
                    "region_name": it["region_name"],
                    "source": it.get("source"),
                    "src_crs": it.get("src_crs"),
                },
                "geometry": mapping(g),
            })
    payload = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
        "region": getattr(reg, "name", None),
        "n": len(feats),
        "files": report,
        "features": feats,
    }
    path = out_dir / f"{getattr(reg, 'name', 'region')}__gt_3857.geojson"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


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
    При ``EXPDS_DFINE_DUMP`` пишет stages.jsonl: raw_boxes → ellipses → features → would_pass_thr.
    """
    st = settings(cfg)
    paths = list((getattr(reg, "rasters", {}) or {}).get(MARKUP, []))
    stage = {
        "region": getattr(reg, "name", None),
        "n_paths": len(paths),
        "n_paths_done": 0,
        "n_raw_boxes": 0,
        "n_ellipses": 0,
        "n_features": 0,
        "n_would_pass_thr": 0,
        "n_models": None,
        "thresholds_ready": None,
        "skip_reason": None,
    }
    if not paths:
        stage["skip_reason"] = "no_sp_or_rasters"
        _emit_stage(stage)
        return [], False
    if "models" not in cache:
        cache["models"], cache["device"] = [], None
        ready, missing = thresholds_ready(cfg)
        cache["thresholds_ready"] = ready
        cache["thresholds_missing"] = missing
        if st["require_thresholds"] and not ready:
            msg = f"нет порогов [thresholds_by_markup.sp_dfine] для {', '.join(missing)}"
            log.error("%s: %s — D-FINE не запускается, выдача как v2", UNAVAILABLE, msg)
            cache["load_error"] = msg
            stage["skip_reason"] = "thresholds_missing:" + ",".join(missing)
            write_unavailable_warning(msg)
            if strict_mode():
                raise RuntimeError(f"{UNAVAILABLE}: {msg}")
        elif not st["weights"]:
            msg = "weights пуст — D-FINE не запускается"
            log.error("%s: %s, выдача как v2", UNAVAILABLE, msg)
            cache["load_error"] = msg
            stage["skip_reason"] = "weights_empty"
            write_unavailable_warning(msg)
            if strict_mode():
                raise RuntimeError(f"{UNAVAILABLE}: {msg}")
        else:
            try:
                dev = _device(st)
                t0 = time.monotonic()
                cache["models"], cache["device"] = load_models(st, root, dev), dev
                log.info("sp_dfine: моделей %d на %s за %.1f с", len(cache["models"]), dev, time.monotonic() - t0)
            except Exception as exc:  # noqa: BLE001 — без strict сдача не падает: выдача как v2
                detail = f"{type(exc).__name__}: {str(exc)[:800]}"
                log.exception("%s: модели не загрузились (%s) — D-FINE пропущен, выдача как v2",
                              UNAVAILABLE, detail[:500])
                cache["load_error"] = detail
                stage["skip_reason"] = f"load_failed:{type(exc).__name__}"
                write_unavailable_warning(f"enabled=true, но модели не загрузились ({type(exc).__name__})",
                                          detail=detail)
                if strict_mode():   # EXPDS_STRICT=1: исходная ошибка как есть
                    raise
                if st["strict"]:
                    raise DFineUnavailable(f"sp_dfine: модели не загрузились ({detail[:500]})") from exc
        if st["strict"] and ((not ready and st["require_thresholds"]) or not st["weights"]):
            raise DFineUnavailable("sp_dfine: нет порогов или весов при strict = true")
    models = cache["models"]
    stage["n_models"] = len(models)
    stage["thresholds_ready"] = cache.get("thresholds_ready")
    if not models:
        if stage["skip_reason"] is None:
            stage["skip_reason"] = "no_models"
        _emit_stage(stage)
        return [], False
    dets: list[dict] = []
    complete = True
    for path in paths:
        if guard is not None and not guard.ok(60):
            complete = False
            stage["skip_reason"] = (stage["skip_reason"] or "") + ";guard_time"
            break
        if soft_ok is not None and not soft_ok(f"D-FINE {Path(path).name}", float(st["soft_need_s"])):
            complete = False
            stage["skip_reason"] = (stage["skip_reason"] or "") + ";soft_ceiling"
            break
        sc = None
        try:
            crs = utm_for(Path(path), crs_for(reg) if crs_for else None)
            sc = Scene(Path(path), crs, float(st["res_m"]))
            ok = (lambda: guard.window_ok()) if guard is not None and hasattr(guard, "window_ok") else (lambda: True)
            t0 = time.monotonic()
            res, done, total = detect_scene(models, sc, st, cache["device"], ok)
            complete &= done >= total
            dump = dump_root()
            n_raw = 0
            for mi, (B, L, Cc) in enumerate(res):
                n_raw += int(len(L))
                ell = boxes_to_ellipses(B, L, Cc, sc.T, crs, reg.name, Path(path).name, mi)
                if dump is not None and len(L):
                    try:
                        write_boxes_dump(dump, region=reg.name, source=Path(path).name, crs=crs,
                                         transform=sc.T, model_i=mi, B=B, L=L, Cc=Cc, ellipses=ell)
                    except Exception:
                        log.exception("sp_dfine: dump боксов не записан (%s m%d)", Path(path).name, mi)
                dets += ell
            stage["n_raw_boxes"] += n_raw
            stage["n_paths_done"] += 1
            log.info("sp_dfine: %s %dx%d px, окон %d/%d, боксов %d, %.1f с", Path(path).name, sc.W, sc.H, done,
                     total, n_raw, time.monotonic() - t0)
        except Exception:
            complete = False
            log.exception("sp_dfine: %s пропущен", Path(path).name)
            stage["skip_reason"] = (stage["skip_reason"] or "") + f";path_err:{Path(path).name}"
        finally:
            if sc is not None:
                sc.close()
    # n_ellipses should be total dets, not wrongly accumulated mid-loop
    stage["n_ellipses"] = len(dets)
    feats = as_features(dets, st, len(models))
    stage["n_features"] = len(feats)
    stage["n_would_pass_thr"] = count_would_pass_thr(feats, cfg) if feats else 0
    dump = dump_root()
    if dump is not None:
        try:
            gp = write_gt_dump(dump, reg)
            if gp is not None:
                log.info("sp_dfine: GT dump %s", gp)
        except Exception:
            log.exception("sp_dfine: GT dump не записан (%s)", getattr(reg, "name", "?"))
        try:
            append_stage(dump, stage)
            log.info("sp_dfine stages %s: raw=%d ell=%d feat=%d pass_thr=%d skip=%s",
                     stage["region"], stage["n_raw_boxes"], stage["n_ellipses"], stage["n_features"],
                     stage["n_would_pass_thr"], stage["skip_reason"])
        except Exception:
            log.exception("sp_dfine: stages dump не записан (%s)", getattr(reg, "name", "?"))
    return feats, complete


def _emit_stage(stage: dict) -> None:
    """Пишет stage-строку, если dump включён (ранний выход region_feats)."""
    dump = dump_root()
    if dump is None:
        return
    try:
        append_stage(dump, stage)
        log.info("sp_dfine stages %s: skip=%s models=%s", stage.get("region"), stage.get("skip_reason"),
                 stage.get("n_models"))
    except Exception:
        log.exception("sp_dfine: stages dump не записан (%s)", stage.get("region"))
