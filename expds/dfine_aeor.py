"""bot-6 (на bot7-candidate-v3 @ 360c870; каркас bot-1 6c2511c): D-FINE всех классов на Ae (аэрофото 1940-х)
и Or (ортофото), фаза ``dfine_aeor``. ПО УМОЛЧАНИЮ ВЫКЛЮЧЕН ([dfine_aeor] enabled = false).

Соглашения (bot 7, 29.10 12:14):
* markup_type родной — ``ae`` / ``or``; threshold_group ``dfine_ae`` / ``dfine_or``, пороги —
  [thresholds_by_markup.dfine_ae / dfine_or] или JSON-файл ``thresholds_file`` ({группа: {класс: порог}},
  формат ``python -m expds.metric --out``), который заполняется ТОЛЬКО из вложенного OOF ядер bot-9. Своих чисел нет.
* D-FINE импортируется только через ``sp_dfine.import_dfine`` (вендор transformers 5.21.0). Любая неготовность
  (нет весов, порогов, импорта) → ERROR с маркером ``DFINE_AEOR_UNAVAILABLE`` (check_submission → FAIL) и выдача
  без фазы. Кэш моделей общий с sp_dfine (по пути каталога весов).
* Вход модели как в обучении bot-9 (bot-9/DFINE_FINAL_ORAE.md, kaggle/dfine_spor/src/dfine_main.py:53,262-291,479):
  UTM участка, 0,5 м (WarpedVRT; уменьшение average), RGB uint8 (1 канал → 3 копии, не-uint8 → растяжка 1–99 %),
  pixel_values = x/255 без mean/std. Окно/перекрытие — как при обучении: Ae 640/160, Or 1024/288 (§9).
* Слияние видов TTA (flip × масштабы × модели) — WBF или NMS строго внутри ОДНОГО растра одной модальности;
  круги варианта C — тоже внутри растра. Межрастровое слияние — post.merge_within_markup с ключом
  (регион, markup, класс, threshold_group): dfine_ae склеивается только с dfine_ae, не с прочими ae
  (watchdog._aeor_split: своя короткая цепочка; merge_union=true — общая, по умолчанию выкл.).
* Память: растр никогда не читается целиком — окна WarpedVRT (Scene.read), батч окон в один заранее
  выделенный float32-буфер (без np.stack/astype-копий), кэш GDAL ограничен ``gdal_cache_mb``.
* Сторож времени: оценка с/окно-вид (замер на первом растре или ``tile_s``), запас ``safety`` (8×) к остатку
  бюджета фазы (``budget_s``) и TimeGuard; деградация full → без flip → один масштаб → пропуск, с журналом.
"""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import numpy as np

from . import sp_dfine as S

log = logging.getLogger(__name__)

UNAVAILABLE = "DFINE_AEOR_UNAVAILABLE"
#: Классы метрики (materials/compute_metrics.py CLASS_WEIGHTS; тест сверяет).
COMPETITION_CLASSES = ("kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha", "arkhitektury", "object_poly",
                       "fortifikatsii", "finds_points")
#: Метки моделей bot-9 по умолчанию (dfine_main.py:66), если в config.json нет id2label.
MODEL_CLASSES = S.CLASSES
GROUPS = {"ae": "dfine_ae", "or": "dfine_or"}
LEVELS = ("full", "no_flip", "one_scale", "skip")

DEFAULTS = {
    "enabled": False,
    "markups": ["ae", "or"],
    "weights": {},                       # {"ae": ["models/dfine_ae"], "or": ["models/dfine_or"]} — каталоги HF
    "thresholds_file": "",               # JSON из OOF bot-9 (вложенный подбор); пусто — только TOML
    "class_map": {},                     # {метка модели: класс метрики}; пусто — тождественно по id2label
    "res_m": {"ae": 0.5, "or": 0.5},
    "tile": {"ae": 640, "or": 1024},
    "overlap": {"ae": 160, "or": 288},
    "batch": {"ae": 8, "or": 4},
    "flip_tta": True,
    "scales": [1.0],                     # множители res_m; [1.0, 1.5] — второй масштаб
    "fuse": "wbf",                       # wbf | nms — только внутри одного растра
    "fuse_iou": 0.55,
    "nms_iou": 0.5,
    "fp16": True,
    "min_score": 0.05,                   # пол генерации (ниже порогов OOF)
    "max_det": 100,
    "skip_empty": 0.95,
    "radius_m": 4.0,
    "merge_m": 4.0,
    "max_objects": 20000,
    "require_thresholds": True,
    "soft_need_s": 60.0,                 # мягкий потолок: запас на участок
    "budget_s": 120.0,                   # потолок фазы на весь набор (bot 7: ≤2 мин)
    "safety": 8.0,                       # оценка × safety ≤ остаток
    "tile_s": None,                      # с на окно-вид на целевой машине; None — замер на первом растре
    "device": "auto",
    "gdal_cache_mb": 256,                # GDAL_CACHEMAX на время чтения растра (по умолчанию GDAL — 5 % ОЗУ)
    "merge_union": False,                # True — dfine_* в общей цепочке и склейке с прочими ae/or (не включать)
}
_PER_MK = ("res_m", "tile", "overlap", "batch")


def settings(cfg: dict) -> dict:
    own = dict(cfg.get("dfine_aeor") or {})
    st = dict(DEFAULTS, **own)
    for k in _PER_MK:
        base = dict(DEFAULTS[k])
        v = own.get(k)
        if isinstance(v, dict):
            base.update(v)
        elif v is not None:
            base = {m: v for m in base}
        st[k] = base
    w = own.get("weights") or {}
    st["weights"] = {m: list(w) for m in st["markups"]} if isinstance(w, list) else \
        {m: list(w.get(m, [])) for m in st["markups"]}
    st["scales"] = [float(s) for s in (st["scales"] or [1.0])]
    return st


def enabled(cfg: dict) -> bool:
    return bool((cfg.get("dfine_aeor") or {}).get("enabled", False))


class AeOrUnavailable(RuntimeError):
    """dfine_aeor включён, но не готов (веса, пороги, импорт D-FINE)."""


def _abs(root: Path, rel) -> Path:
    p = Path(str(rel))
    return p if p.is_absolute() else Path(root) / p


#: Имена групп в OOF bot-9 → наши (bot-9 переименует; до тех пор загрузчик принимает оба).
GROUP_ALIASES = {"ae_dfine": "dfine_ae", "or_dfine": "dfine_or"}


def load_thresholds_file(cfg: dict, root: Path) -> dict:
    """Пороги из thresholds_file → cfg[thresholds_by_markup] (только группы dfine_*; ключи TOML главнее).
    Нет файла при заданном пути — AeOrUnavailable с путём."""
    st = settings(cfg)
    rel = st.get("thresholds_file")
    if not rel:
        return {}
    p = _abs(root, rel)
    if not p.is_file():
        raise AeOrUnavailable(f"thresholds_file не найден: {p} (заполняется из OOF bot-9, см. AEOR_DETECT.md)")
    data = json.loads(p.read_text(encoding="utf-8"))
    tbm = cfg.setdefault("thresholds_by_markup", {})
    used: dict = {}
    for g, thr in data.items():
        g = GROUP_ALIASES.get(g, g)   # OOF bot-9 называет группы ae_dfine/or_dfine
        if g not in GROUPS.values() or not isinstance(thr, dict):
            continue
        own = tbm.setdefault(g, {})
        for c, v in thr.items():
            if c not in own:
                own[c] = float(v)
                used.setdefault(g, {})[c] = float(v)
    return used


def thresholds_ready(cfg: dict, markup: str) -> tuple[bool, list[str]]:
    own = (cfg.get("thresholds_by_markup") or {}).get(GROUPS[markup]) or {}
    missing = [c for c in MODEL_CLASSES if c not in own]
    return not missing, missing


def model_labels(wdir: Path, class_map: dict) -> list[str]:
    """Метки выходов модели → классы метрики (id2label из config.json, затем class_map)."""
    labels = list(MODEL_CLASSES)
    cj = Path(wdir) / "config.json"
    if cj.is_file():
        i2l = json.loads(cj.read_text(encoding="utf-8")).get("id2label") or {}
        if i2l:
            labels = [str(i2l.get(str(i), i2l.get(i, f"LABEL_{i}"))) for i in range(len(i2l))]
    out = [str(class_map.get(lb, lb)) for lb in labels]
    bad = [c for c in out if c not in COMPETITION_CLASSES]
    if bad:
        raise AeOrUnavailable(f"{wdir}: метки вне классов метрики {bad} (задайте class_map)")
    return out


def preflight(cfg: dict, root: Path) -> dict:
    """Проверки до работы. Неготовность → ERROR с маркером и {'ok': False}: фаза не запускается (выдача как v3)."""
    st = settings(cfg)
    info: dict = {"enabled": True, "ok": False, "markups": list(st["markups"]), "problems": []}
    pr = info["problems"]
    try:
        info["thresholds_file"] = load_thresholds_file(cfg, root)
    except Exception as e:  # noqa: BLE001
        pr.append(str(e))
    for mk in st["markups"]:
        if mk not in GROUPS:
            pr.append(f"неизвестная модальность {mk}")
            continue
        ok, miss = thresholds_ready(cfg, mk)
        if st["require_thresholds"] and not ok:
            pr.append(f"нет порогов {GROUPS[mk]} для " + ", ".join(miss))
        if not st["weights"].get(mk):
            pr.append(f"weights.{mk} пуст (ждём чекпойнт bot-9)")
        for rel in st["weights"].get(mk, []):
            p = _abs(root, rel)
            for name in ("config.json", "model.safetensors"):
                if not (p / name).is_file():
                    pr.append(f"нет весов {p / name}")
    if pr:
        log.error("%s: [dfine_aeor] включён, но не готов — фаза пропущена, выдача как v3: %s", UNAVAILABLE, "; ".join(pr))
        return info
    info["ok"] = True
    return info


# ── нарезка ──────────────────────────────────────────────────────────────────────────────────────────────
def windows(W: int, H: int, tile: int, overlap: int) -> list[tuple[int, int]]:
    """Сетка окон как в обучении (sp_dfine.windows), но последний ряд/столбец прижат к краю: покрытие полное."""
    step = int(tile) - int(overlap)
    assert step > 0, "overlap >= tile"

    def axis(n: int) -> list[int]:
        if n <= tile:
            return [0]
        xs = list(range(0, n - tile + 1, step))
        if xs[-1] + tile < n:
            xs.append(n - tile)
        return xs
    return [(x, y) for y in axis(int(H)) for x in axis(int(W))]


def n_windows(W: int, H: int, tile: int, overlap: int) -> int:
    return len(windows(W, H, tile, overlap))


# ── слияние внутри одного растра ─────────────────────────────────────────────────────────────────────────
def _iou(b: np.ndarray, B: np.ndarray) -> np.ndarray:
    w = np.clip(np.minimum(b[2], B[:, 2]) - np.maximum(b[0], B[:, 0]), 0, None)
    h = np.clip(np.minimum(b[3], B[:, 3]) - np.maximum(b[1], B[:, 1]), 0, None)
    inter = w * h
    ua = (b[2] - b[0]) * (b[3] - b[1]) + (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) - inter
    return np.where(ua > 0, inter / np.maximum(ua, 1e-9), 0.0)


def wbf(boxes, scores, labels, n_views: int, iou: float = 0.55):
    """Weighted boxes fusion (своя реализация): по классу, по убыванию уверенности; слитый бокс — среднее
    с весами-уверенностями; уверенность — сумма / max(n_views, членов)."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    oB, oS, oL = [], [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        idx = idx[np.argsort(-scores[idx], kind="stable")]
        fused: list[np.ndarray] = []
        members: list[list[int]] = []
        for i in idx:
            j = -1
            if fused:
                ious = _iou(boxes[i], np.asarray(fused))
                k = int(np.argmax(ious))
                if ious[k] >= iou:
                    j = k
            if j < 0:
                fused.append(boxes[i].copy())
                members.append([i])
            else:
                members[j].append(i)
                m = np.asarray(members[j])
                w = scores[m]
                fused[j] = (boxes[m] * w[:, None]).sum(0) / max(w.sum(), 1e-12)
        for fb, m in zip(fused, members):
            oB.append(fb)
            oS.append(float(scores[m].sum() / max(n_views, len(m))))
            oL.append(int(c))
    if not oB:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    return np.asarray(oB), np.clip(np.asarray(oS), 0, 1), np.asarray(oL, dtype=int)


def nms(boxes, scores, labels, iou: float = 0.5):
    """NMS по классам на numpy (без torchvision — для тестов и CPU)."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    keep: list[int] = []
    for c in np.unique(labels):
        idx = list(np.where(labels == c)[0][np.argsort(-scores[labels == c], kind="stable")])
        while idx:
            i = idx.pop(0)
            keep.append(i)
            if idx:
                ious = _iou(boxes[i], boxes[idx])
                idx = [j for j, v in zip(idx, ious) if v < iou]
    keep = sorted(keep)
    return boxes[keep], scores[keep], labels[keep]


def fuse_views(views: list, st: dict):
    """Слияние видов ОДНОГО растра одной модальности (вызывается только из detect_image)."""
    n = len(views)
    if not n:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    B = np.concatenate([np.asarray(v[0], dtype=np.float64).reshape(-1, 4) for v in views])
    L = np.concatenate([np.asarray(v[1], dtype=np.float64).reshape(-1) for v in views])
    C = np.concatenate([np.asarray(v[2], dtype=int).reshape(-1) for v in views])
    if n == 1 or not len(L):
        return B, L, C
    if st["fuse"] == "wbf":
        return wbf(B, L, C, n, float(st["fuse_iou"]))
    return nms(B, L, C, float(st["nms_iou"]))


# ── сторож времени ───────────────────────────────────────────────────────────────────────────────────────
def views_at(level: str, st: dict) -> tuple[tuple, list]:
    flips = (False, True) if st["flip_tta"] else (False,)
    if level == "full":
        return flips, list(st["scales"])
    if level == "no_flip":
        return (False,), list(st["scales"])
    if level == "one_scale":
        return (False,), [st["scales"][0]]
    return (), []


def plan_level(tiles_at_scale: dict, t_view: float | None, n_models: int, st: dict, left_s: float):
    """Самый богатый уровень, у которого оценка × safety ≤ left_s. t_view неизвестно → 'one_scale' (замер),
    если хоть что-то осталось. Возвращает (уровень, оценка с)."""
    if left_s <= 0:
        return "skip", 0.0
    if t_view is None:
        return "one_scale", float("nan")
    est = 0.0
    for level in LEVELS[:-1]:
        flips, scales = views_at(level, st)
        est = sum(tiles_at_scale[s] for s in scales) * len(flips) * max(n_models, 1) * float(t_view)
        if est * float(st["safety"]) <= left_s:
            return level, est
    return "skip", est


# ── геопривязка ──────────────────────────────────────────────────────────────────────────────────────────
def boxes_to_ellipses(B, L, C, transform, crs: str, region: str, source: str, labels: list[str]) -> list[dict]:
    """Боксы в пикселях сцены (UTM, transform) → 32-угольные эллипсы EPSG:3857 (как sp_dfine, свои метки)."""
    from pyproj import Transformer
    from shapely.geometry import Polygon

    if len(L) == 0:
        return []
    to3857 = Transformer.from_crs(crs, "EPSG:3857", always_xy=True).transform
    t = np.linspace(0, 2 * np.pi, 33)[:-1]
    a, b, c0, d, e, f0 = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
    out = []
    # Площадь пикселя в м² сцены (UTM и т.п.): |det(affine)|.
    pix_m2 = abs(a * e - b * d)
    for (x0, y0, x1, y1), s, c in zip(B, L, C):
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        rx, ry = max((x1 - x0) / 2, 0.5), max((y1 - y0) / 2, 0.5)
        px, py = cx + rx * np.cos(t), cy + ry * np.sin(t)
        gx, gy = to3857(a * px + b * py + c0, d * px + e * py + f0)
        # box_area_m2 — площадь бокса ДО cluster_circles (круг r=4 ≈ 50 м² иначе).
        box_area_m2 = float(abs(x1 - x0) * abs(y1 - y0) * pix_m2)
        out.append({"geom": Polygon(np.column_stack([gx, gy])), "class_name": labels[int(c)],
                    "confidence": round(float(s), 5), "region_name": region, "source": source,
                    "original_crs": crs, "model": 0, "box_area_m2": box_area_m2})
    return out


def image_feats(B, L, C, transform, crs: str, region: str, source: str, markup: str, st: dict,
                labels: list[str] | None = None) -> list[dict]:
    """Боксы ОДНОГО растра → круги варианта C (склейка только внутри растра и класса)."""
    el = boxes_to_ellipses(B, L, C, transform, crs, region, source, labels or list(MODEL_CLASSES))
    out = S.cluster_circles(el, float(st["radius_m"]), float(st["merge_m"]))
    for f in out:
        f["markup_type"] = markup
        f["threshold_group"] = GROUPS[markup]
        f["source"] = source
    return out


# ── инференс ─────────────────────────────────────────────────────────────────────────────────────────────
def forward_boxes(model, px, flip: bool, tile: int, st: dict):
    """Один вид батча: (список по окнам (xyxy px окна, score, cls)). Модель — DFineForObjectDetection или
    заглушка с тем же выходом (logits [N,Q,C], pred_boxes cxcywh 0..1)."""
    import torch

    inp = px.flip(3) if flip else px
    o = model(pixel_values=inp)
    prob = o.logits.float().sigmoid()
    bx = o.pred_boxes.float()
    if flip:
        bx = bx.clone()
        bx[..., 0] = 1 - bx[..., 0]
    Q, Cn = prob.shape[1:]
    sc_, idx = prob.flatten(1).topk(min(int(st["max_det"]), Q * Cn), 1)
    out = []
    for i in range(px.shape[0]):
        keep = sc_[i] >= float(st["min_score"])
        q, c = idx[i][keep] // Cn, idx[i][keep] % Cn
        bb = bx[i][q] * tile
        xyxy = torch.stack([bb[:, 0] - bb[:, 2] / 2, bb[:, 1] - bb[:, 3] / 2,
                            bb[:, 0] + bb[:, 2] / 2, bb[:, 1] + bb[:, 3] / 2], 1)
        out.append((xyxy.cpu().numpy().astype(np.float64), sc_[i][keep].cpu().numpy().astype(np.float64),
                    c.cpu().numpy().astype(int)))
    return out


def detect_scene(models: list, scene, st: dict, mk: str, flips, device, keep_going=lambda: True, left=None):
    """Окна сцены → виды (модель × flip), у каждого NMS по классам (склейка перекрытий окон).
    Возвращает ([(B, L, C)], окон сделано, окон всего).

    Срок (bot-7 13:12): keep_going() и left() (секунд до конца фазы/окна сторожа) проверяются перед КАЖДЫМ
    чтением окна, а не только между батчами: на Or без обзоров одно окно читается ~7–23 с (056/033), и батч
    из 4 окон после последней проверки переехал бы потолок фазы на ~90 с. Следующее окно читается, только если
    left() > max(время чтения окна) + max(время прямого прохода батча) по замерам этой сцены; прерванный батч
    не прогоняется, если на прямой проход времени уже нет. Перерасход ограничен одним чтением окна (GDAL не
    прерывается) — до первого замера на сцене."""
    import torch

    T, ov, bs = int(st["tile"][mk]), int(st["overlap"][mk]), int(st["batch"][mk])
    wins = windows(scene.W, scene.H, T, ov)
    acc = [([], [], []) for _ in range(len(models) * len(flips))]
    buf = np.empty((bs, 3, T, T), dtype=np.float32)   # один буфер на сцену: bs×3×T×T×4 Б (Or: 4×3×1024²×4 = 48 МБ)
    done = 0
    use_amp = bool(st["fp16"]) and getattr(device, "type", str(device)) == "cuda"
    t_read = t_fwd = 0.0                              # макс. замеры этой сцены, с
    left = left or (lambda: float("inf"))
    stopped = False
    with torch.no_grad():
        for b in range(0, len(wins), bs):
            if not keep_going() or left() <= t_read + t_fwd:
                break
            chunk = []
            for i, (x, y) in enumerate(wins[b:b + bs]):
                if i and (not keep_going() or left() <= t_read + t_fwd):
                    stopped = True
                    break
                t0 = time.monotonic()
                im = scene.read(x, y, T)          # одно окно (WarpedVRT), не весь растр
                t_read = max(t_read, time.monotonic() - t0)
                if (im.max(2) == 0).mean() > float(st["skip_empty"]):
                    continue
                buf[len(chunk)] = im.transpose(2, 0, 1)   # uint8 → float32 прямо в буфер батча
                chunk.append((x, y))
                del im
            done = min(b + bs, len(wins)) if not stopped else b + i
            if not chunk:
                if stopped:
                    break
                continue
            if stopped and left() <= t_fwd:
                done = b
                break
            t0 = time.monotonic()
            arr = buf[:len(chunk)]
            arr *= np.float32(1.0 / 255)
            px = torch.from_numpy(arr).to(device)   # на CPU — тот же буфер, без копии
            v = 0
            for model in models:
                for flip in flips:
                    with torch.autocast(getattr(device, "type", "cpu"), dtype=torch.float16, enabled=use_amp):
                        res = forward_boxes(model, px, flip, T, st)
                    for (x, y), (bb, s, c) in zip(chunk, res):
                        acc[v][0].append(bb + np.array([x, y, x, y], dtype=np.float64))
                        acc[v][1].append(s)
                        acc[v][2].append(c)
                    v += 1
            t_fwd = max(t_fwd, time.monotonic() - t0)
            if stopped:
                break
    out = []
    for B, L, C in acc:
        if not B:
            out.append((np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)))
            continue
        out.append(nms(np.concatenate(B), np.concatenate(L), np.concatenate(C), float(st["nms_iou"])))
    return out, done, len(wins)


class Scene(S.Scene):
    """Как sp_dfine.Scene (UTM на res_m, WarpedVRT, average при уменьшении, окна RGB uint8), но для сильно
    уменьшаемых растров (Or 1–3 см → 0,5 м) источник открывается на обзоре GDAL: самом грубом с пикселем
    ≤ res_m × ovr_max_frac (по умолчанию ½ — усреднение до res_m остаётся). Без этого окно 1024 px на 0,5 м
    читает ~40 000² пикселей полного разрешения (на 008_Герите Or GDAL просит 718 МБ на один кусок
    варпа и падает при 2 ГБ). Пока пиксель источника ≥ res_m × frac (Ae, SpOr) — ровно sp_dfine.Scene."""

    def __init__(self, path: Path, crs: str, res_m: float = 0.5, ovr_max_frac: float = 0.5,
                 warp_mem_mb: int = 64):
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.vrt import WarpedVRT
        from rasterio.warp import calculate_default_transform

        src = rasterio.open(path)
        try:
            src_m = abs(calculate_default_transform(src.crs, crs, src.width, src.height, *src.bounds)[0].a)
        except Exception:
            src_m = res_m
        ovrs = src.overviews(1) if src.count else []
        lvl = None
        for i, f in enumerate(ovrs):
            if src_m * f <= res_m * float(ovr_max_frac):
                lvl = i
        self.overview_level, self.src_m = lvl, src_m
        if lvl is None:
            src.close()
            super().__init__(path, crs, res_m)
            return
        bounds = src.bounds
        src.close()
        self.src = rasterio.open(path, overview_level=lvl)
        self.crs = crs
        tr, w, h = calculate_default_transform(self.src.crs, crs, self.src.width, self.src.height, *bounds,
                                               resolution=res_m)
        self.vrt = WarpedVRT(self.src, crs=crs, transform=tr, width=w, height=h, resampling=Resampling.average,
                             nodata=0, warp_mem_limit=int(warp_mem_mb))
        self.W, self.H, self.T = w, h, tr
        self.bands = list(range(1, min(3, self.src.count) + 1))
        self.stretch = None
        if self.src.dtypes[0] != "uint8":
            ov = self.src.read(1, out_shape=(min(1024, self.src.height), min(1024, self.src.width)))
            v = ov[ov > 0]
            self.stretch = tuple(np.percentile(v, [1, 99])) if v.size else (0, 1)


def detect_image(models: list, path: Path, crs: str, st: dict, mk: str, level: str, device,
                 keep_going=lambda: True, scene_cls=None, left=None):
    """Виды выбранного уровня одного растра → слитые боксы в пикселях базовой сцены.
    Возвращает (B, L, C, transform базовой сцены, окон сделано, окон всего)."""
    if scene_cls is None:   # настоящий растр: кэш GDAL ограничен (окна WarpedVRT, растр целиком не читается)
        import rasterio

        with rasterio.Env(GDAL_CACHEMAX=int(st.get("gdal_cache_mb") or 256) * 1024 * 1024):
            return detect_image(models, path, crs, st, mk, level, device, keep_going, Scene, left)
    flips, scales = views_at(level, st)
    base_res = float(st["res_m"][mk])
    views: list = []
    T0 = None
    done_all = total_all = 0
    for s in scales:
        if views and not keep_going():
            break
        sc = scene_cls(path, crs, base_res * float(s))
        try:
            if T0 is None:
                T0 = sc.T
            res, done, total = detect_scene(models, sc, st, mk, flips, device, keep_going, left)
            done_all += done
            total_all += total
            k = abs(sc.T.a) / abs(T0.a)
            for B, L, C in res:
                views.append((np.asarray(B) * k, L, C))
        finally:
            sc.close()
    if T0 is None:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int), None, 0, 0
    B, L, C = fuse_views(views, st)
    return B, L, C, T0, done_all, total_all


def _load(st: dict, mk: str, root: Path, cache: dict) -> tuple[list, list[str]]:
    """Модели модальности: общий кэш с sp_dfine по пути каталога (cache['by_path'])."""
    by_path = cache.setdefault("by_path", {})
    sp_w = [str(_abs(root, w)) for w in (((cache.get("cfg") or {}).get("sp_dfine") or {}).get("weights") or [])]
    for w, m in zip(sp_w, cache.get("models") or []):   # модели, уже загруженные фазой sp_dfine
        by_path.setdefault(w, m)
    todo = [w for w in st["weights"][mk] if str(_abs(root, w)) not in by_path]
    if todo:
        if cache.get("device") is None:
            cache["device"] = S._device(st)
        ms = S.load_models({"weights": [str(_abs(root, w)) for w in todo]}, root, cache["device"])  # import_dfine внутри
        for w, m in zip(todo, ms):
            by_path[str(_abs(root, w))] = m
    models = [by_path[str(_abs(root, w))] for w in st["weights"][mk]]
    labels = model_labels(_abs(root, st["weights"][mk][0]), dict(st.get("class_map") or {}))
    return models, labels


def raster_bytes(p) -> int:
    try:
        return Path(p).stat().st_size
    except OSError:
        return 0


def order_regions(regions, markups) -> list:
    """Дешёвое → дорогое: по сумме байт растров Ae/Or участка (устойчиво, пустые отброшены)."""
    def cost(reg):
        return sum(raster_bytes(p) for m in markups for p in (getattr(reg, "rasters", {}) or {}).get(m, []))
    return sorted([r for r in regions if cost(r) > 0 or any((r.rasters or {}).get(m) for m in markups)], key=cost)


def region_feats(reg, cfg: dict, guard, cache: dict, root: Path, crs_for=None, soft_ok=None,
                 phase_left=lambda: float("inf"), scene_cls=None, loader=None) -> list[dict]:
    """Объекты одного участка. cache — общий словарь прогона (модели, замер с/окно-вид)."""
    st = settings(cfg)
    real = scene_cls is None
    scene_cls = scene_cls or Scene
    loader = loader or _load
    items = sorted(((mk, p) for mk in st["markups"] if mk in GROUPS
                    for p in (getattr(reg, "rasters", {}) or {}).get(mk, [])), key=lambda t: raster_bytes(t[1]))
    out: list[dict] = []
    for mk, path in items:
        key = f"models_{mk}"
        if key not in cache:
            try:
                cache[key] = loader(st, mk, root, cache)
            except Exception as exc:  # noqa: BLE001 — громко, без падения сдачи
                log.error("%s: модели %s не загрузились (%s: %s) — модальность пропущена", UNAVAILABLE, mk,
                          type(exc).__name__, str(exc)[:500])
                cache[key] = ([], [])
        models, labels = cache[key]
        if not models:
            continue
        if soft_ok is not None and not soft_ok(f"D-FINE {mk} {Path(path).name}", float(st["soft_need_s"])):
            break
        sc = None
        try:
            crs = S.utm_for(Path(path), crs_for(reg) if crs_for else None)
            sc = scene_cls(Path(path), crs, float(st["res_m"][mk]))
            W, H = sc.W, sc.H
            sc.close()
            sc = None
            T, ov = int(st["tile"][mk]), int(st["overlap"][mk])
            tiles = {s: n_windows(math.ceil(W / s), math.ceil(H / s), T, ov) for s in st["scales"]}
            left = phase_left()
            if guard is not None and hasattr(guard, "window_left"):
                left = min(left, guard.window_left() - float(getattr(guard, "window_reserve_s", 0) or 0))
            t_view = st["tile_s"] if st["tile_s"] is not None else cache.get("t_view", {}).get(mk)
            level, est = plan_level(tiles, t_view, len(models), st, left)
            if level != "full":
                log.warning("dfine_aeor: %s %s уровень %s (оценка %.1f с × %.0f vs остаток %.1f с)", mk,
                            Path(path).name, level, est, float(st["safety"]), left)
            if level == "skip":
                continue

            def ok():
                g = guard.window_ok() if guard is not None and hasattr(guard, "window_ok") else True
                return g and phase_left() > 0

            def left_s():
                lf = phase_left()
                if guard is not None and hasattr(guard, "window_left"):
                    lf = min(lf, guard.window_left() - float(getattr(guard, "window_reserve_s", 0) or 0))
                return lf
            t0 = time.monotonic()
            B, L, C, T0, done, total = detect_image(models, Path(path), crs, st, mk, level, cache.get("device"), ok,
                                                    None if real else scene_cls, left_s)
            dt = time.monotonic() - t0
            flips, _ = views_at(level, st)
            if done:
                tv = dt / (done * len(flips) * max(len(models), 1))
                prev = cache.setdefault("t_view", {}).get(mk)
                cache["t_view"][mk] = tv if prev is None else max(prev, tv)
            if T0 is not None:
                fs = image_feats(B, L, C, T0, crs, reg.name, Path(path).name, mk, st, labels)
                out += fs
                log.info("dfine_aeor: %s %s уровень %s, окон %d/%d, боксов %d → кругов %d, %.1f с", mk,
                         Path(path).name, level, done, total, len(L), len(fs), dt)
        except Exception:
            log.exception("dfine_aeor: %s пропущен", Path(path).name)
        finally:
            if sc is not None:
                sc.close()
    return out


def cap(feats: list[dict], max_objects: int) -> list[dict]:
    """Не больше max_objects объектов на группу dfine_ae/dfine_or (самые уверенные)."""
    keep: list[dict] = []
    for g in GROUPS.values():
        mine = sorted((f for f in feats if f.get("threshold_group") == g), key=lambda f: -float(f["confidence"]))
        keep += mine[:max_objects]
    return [f for f in feats if f.get("threshold_group") not in GROUPS.values()] + keep
