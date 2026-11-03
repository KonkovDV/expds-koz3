"""bot-1 (кандидат v4): D-FINE на растрах Ae (аэрофото) и Or (ортофото), группы ``ae_dfine`` / ``or_dfine``.

ПО УМОЛЧАНИЮ ВЫКЛЮЧЕН ([aeor_dfine] enabled = false). Включать только после honest-10 приёмки ботом 10
(bot-1/AEOR_DETECT_2026-10-29.md). Без порогов [thresholds_by_markup.ae_dfine] / [...or_dfine] для всех
классов модуль ничего не выдаёт (``require_thresholds``), а при strict = true прогон падает на preflight.

Отличия от ``sp_dfine`` (тот же загрузчик весов и та же сцена WarpedVRT):
* модальности ``ae`` и ``or``; свои веса и разрешение на модальность (Or по умолчанию 0,25 м — объекты мелкие);
* TTA: flip × масштабы ``scales`` (множители к res_m, например [1.0, 1.5]); слияние видов и моделей внутри
  ОДНОГО снимка — WBF (``fuse = "wbf"``, собственная реализация) или NMS (``fuse = "nms"``);
* CLAHE по яркости окна (``clahe = true``, нужен cv2) — аэрофото архивные, контраст плавает;
* круги варианта C склеиваются только внутри одного снимка и одного класса (dedupe не идёт между снимками
  и типами данных — правило соревнования).

torch / transformers / cv2 импортируются лениво, только при enabled = true.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from . import sp_dfine as S

log = logging.getLogger(__name__)

CLASSES = S.CLASSES
GROUPS = {"ae": "ae_dfine", "or": "or_dfine"}

DEFAULTS = {
    "enabled": False,
    "markups": ["ae", "or"],
    "weights": {},            # {"ae": ["models/dfine_aeor"], "or": ["models/dfine_aeor"]}; каталоги HF от корня
    "res_m": {"ae": 0.5, "or": 0.25},
    "tile": 640,
    "overlap": 160,
    "batch": 8,
    "flip_tta": True,
    "scales": [1.0],          # множители res_m; [1.0, 1.5] = второй масштаб крупнее по земле
    "clahe": False,
    "clahe_clip": 2.0,
    "fuse": "wbf",            # wbf | nms — слияние видов TTA и моделей внутри одного снимка
    "fuse_iou": 0.55,
    "fp16": True,
    "min_score": 0.05,
    "max_det": 100,
    "nms_iou": 0.5,
    "skip_empty": 0.95,
    "radius_m": 4.0,
    "merge_m": 4.0,
    "max_objects": 20000,     # предел на модальность за прогон (лимит выхода 20 000 — общий, режем заранее)
    "require_thresholds": True,
    "soft_need_s": 150.0,
    "device": "auto",
    "strict": True,
    "transformers_version": "5.21.0",
}


def settings(cfg: dict) -> dict:
    own = dict(cfg.get("aeor_dfine") or {})
    st = dict(DEFAULTS, **own)
    res = dict(DEFAULTS["res_m"])
    if isinstance(own.get("res_m"), dict):
        res.update(own["res_m"])
    elif own.get("res_m") is not None:
        res = {m: float(own["res_m"]) for m in res}
    st["res_m"] = res
    w = own.get("weights") or {}
    st["weights"] = {m: list(w) for m in st["markups"]} if isinstance(w, list) else {m: list(w.get(m, [])) for m in st["markups"]}
    return st


def enabled(cfg: dict) -> bool:
    return bool((cfg.get("aeor_dfine") or {}).get("enabled", False))


class AeOrUnavailable(S.DFineUnavailable):
    """aeor_dfine включён, но не готов (веса, пороги, transformers, cv2 при clahe)."""


def thresholds_ready(cfg: dict, markup: str) -> tuple[bool, list[str]]:
    own = (cfg.get("thresholds_by_markup") or {}).get(GROUPS[markup]) or {}
    missing = [c for c in CLASSES if c not in own]
    return not missing, missing


def preflight(cfg: dict, root: Path) -> dict:
    st = settings(cfg)
    info: dict = {"enabled": True, "strict": bool(st["strict"]), "markups": list(st["markups"]), "problems": []}
    pr = info["problems"]
    for mk in st["markups"]:
        if mk not in GROUPS:
            pr.append(f"неизвестная модальность {mk}")
            continue
        ok, miss = thresholds_ready(cfg, mk)
        if st["require_thresholds"] and not ok:
            pr.append(f"нет порогов [thresholds_by_markup.{GROUPS[mk]}] для " + ", ".join(miss))
        if not st["weights"].get(mk):
            pr.append(f"weights.{mk} пуст")
        for rel in st["weights"].get(mk, []):
            p = Path(rel) if Path(rel).is_absolute() else Path(root) / rel
            for name in ("config.json", "model.safetensors"):
                if not (p / name).is_file():
                    pr.append(f"нет {p / name}")
    try:
        S._vendor_path(root)
        import transformers
        info["transformers"] = str(getattr(transformers, "__version__", "?"))
        from transformers import DFineConfig, DFineForObjectDetection  # noqa: F401
        pin = st.get("transformers_version")
        if pin and info["transformers"] != pin:
            pr.append(f"transformers {info['transformers']} != пин {pin}")
    except Exception as e:
        pr.append(f"transformers без DFine ({type(e).__name__}: {e})")
    if st["clahe"]:
        try:
            import cv2  # noqa: F401
        except Exception as e:
            pr.append(f"clahe = true, но cv2 нет ({type(e).__name__})")
    if pr:
        msg = "aeor_dfine включён, но не готов: " + "; ".join(pr)
        if st["strict"]:
            raise AeOrUnavailable(msg)
        log.error("%s (strict = false: ветка пропущена)", msg)
    return info


# ── слияние внутри одного снимка ─────────────────────────────────────────────────────────────────────────
def _iou(b: np.ndarray, B: np.ndarray) -> np.ndarray:
    w = np.clip(np.minimum(b[2], B[:, 2]) - np.maximum(b[0], B[:, 0]), 0, None)
    h = np.clip(np.minimum(b[3], B[:, 3]) - np.maximum(b[1], B[:, 1]), 0, None)
    inter = w * h
    ua = (b[2] - b[0]) * (b[3] - b[1]) + (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) - inter
    return np.where(ua > 0, inter / np.maximum(ua, 1e-9), 0.0)


def wbf(boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, n_views: int, iou: float = 0.55):
    """Weighted boxes fusion (своя реализация): по классу, по убыванию уверенности; бокс входит в кластер,
    если IoU с его текущим слитым боксом ≥ iou. Слитый бокс — среднее с весами-уверенностями; уверенность —
    сумма уверенностей / max(n_views, членов) (вид, не давший бокса, тянет вниз, как в WBF)."""
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


def _clahe_reader(scene, clip: float):
    import cv2

    cl = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(8, 8))
    raw = scene.read

    def read(x0: int, y0: int, n: int) -> np.ndarray:
        im = raw(x0, y0, n)
        if im.max() == 0:
            return im
        lab = cv2.cvtColor(im, cv2.COLOR_RGB2LAB)
        lab[..., 0] = cl.apply(lab[..., 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        out[im.max(2) == 0] = 0
        return out

    scene.read = read
    return scene


def detect_views(models: list, scene, st: dict, device, keep_going=lambda: True):
    """Окна сцены → виды (модель × flip), у каждого свой NMS по классам (склейка перекрытий окон).
    Возвращает ([(B, L, C) по видам], окон сделано, окон всего). Окно читается один раз."""
    import torch
    from torchvision.ops import batched_nms

    T, bs = int(st["tile"]), int(st["batch"])
    wins = S.windows(scene.W, scene.H, T, int(st["overlap"]))
    flips = (False, True) if st["flip_tta"] else (False,)
    acc = [([], [], []) for _ in range(len(models) * len(flips))]
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
            v = 0
            for model in models:
                for flip in flips:
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
                        acc[v][0].append(torch.stack([bb[:, 0] - bb[:, 2] / 2 + x, bb[:, 1] - bb[:, 3] / 2 + y,
                                                      bb[:, 0] + bb[:, 2] / 2 + x, bb[:, 1] + bb[:, 3] / 2 + y], 1).cpu())
                        acc[v][1].append(sc_[i][keep].cpu())
                        acc[v][2].append(c.cpu())
                    v += 1
    out = []
    for B, L, Cc in acc:
        if not B:
            out.append((np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)))
            continue
        B, L, Cc = torch.cat(B), torch.cat(L), torch.cat(Cc)
        k = batched_nms(B, L, Cc, float(st["nms_iou"]))
        out.append((B[k].numpy().astype(np.float64), L[k].numpy().astype(np.float64), Cc[k].numpy().astype(int)))
    return out, done, len(wins)


def fuse_views(views: list, st: dict):
    """Слияние видов одного снимка: WBF (по умолчанию) или общий NMS."""
    views = [v for v in views]
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
    import torch
    from torchvision.ops import batched_nms

    k = batched_nms(torch.as_tensor(B, dtype=torch.float32), torch.as_tensor(L, dtype=torch.float32),
                    torch.as_tensor(C), float(st["nms_iou"])).numpy()
    return B[k], L[k], C[k]


def detect_image(models: list, path: Path, crs: str, st: dict, markup: str, device, keep_going=lambda: True):
    """Все виды TTA (масштабы × модели × flip) одного снимка → слитые боксы в пикселях базовой сцены.
    Возвращает (B, L, C, transform базовой сцены, полностью?)."""
    base_res = float(st["res_m"][markup])
    views: list = []
    T0 = None
    complete = True
    for s in st["scales"]:
        sc = S.Scene(path, crs, base_res * float(s))
        try:
            if T0 is None:
                T0 = sc.T
            if st["clahe"]:
                _clahe_reader(sc, st["clahe_clip"])
            res, done, total = detect_views(models, sc, st, device, keep_going)
            complete &= done >= total
            k = abs(sc.T.a) / abs(T0.a)   # 1.0 для первого масштаба
            for B, L, C in res:
                views.append((np.asarray(B) * k, L, C))   # общий верхний левый угол сцен → только масштаб
        finally:
            sc.close()
    if T0 is None:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int), None, complete
    B, L, C = fuse_views(views, st)
    return B, L, C, T0, complete


def image_feats(B, L, C, transform, crs: str, region: str, source: str, markup: str, st: dict) -> list[dict]:
    """Боксы одного снимка → круги варианта C (склейка только внутри снимка и класса)."""
    el = S.boxes_to_ellipses(B, L, C, transform, crs, region, source, 0)
    out = S.cluster_circles(el, float(st["radius_m"]), float(st["merge_m"]))
    for f in out:
        f["markup_type"] = markup
        f["threshold_group"] = GROUPS[markup]
        f["source"] = source
    return out


def region_feats(reg, cfg: dict, guard, cache: dict, root: Path, crs_for=None, soft_ok=None) -> list[dict]:
    st = settings(cfg)
    out: list[dict] = []
    for mk in st["markups"]:
        paths = list((getattr(reg, "rasters", {}) or {}).get(mk, []))
        if not paths or mk not in GROUPS:
            continue
        key = f"models_{mk}"
        if key not in cache:
            ok, _ = thresholds_ready(cfg, mk)
            if (st["require_thresholds"] and not ok) or not st["weights"].get(mk):
                if st["strict"]:
                    raise AeOrUnavailable(f"aeor_dfine: нет порогов или весов для {mk}")
                cache[key] = []
            else:
                dev = S._device(st)
                cache[key] = S.load_models(dict(st, weights=st["weights"][mk]), root, dev)
                cache["device"] = dev
        models = cache[key]
        if not models:
            continue
        for path in paths:
            if guard is not None and not guard.ok(60):
                break
            if soft_ok is not None and not soft_ok(f"D-FINE {mk} {Path(path).name}", float(st["soft_need_s"])):
                break
            try:
                crs = S.utm_for(Path(path), crs_for(reg) if crs_for else None)
                ok = (lambda: guard.window_ok()) if guard is not None and hasattr(guard, "window_ok") else (lambda: True)
                t0 = time.monotonic()
                B, L, C, T0, _ = detect_image(models, Path(path), crs, st, mk, cache["device"], ok)
                if T0 is not None:
                    fs = image_feats(B, L, C, T0, crs, reg.name, Path(path).name, mk, st)
                    out += fs
                    log.info("aeor_dfine: %s %s боксов %d → кругов %d, %.1f с", mk, Path(path).name, len(L), len(fs),
                             time.monotonic() - t0)
            except S.DFineUnavailable:
                raise
            except Exception:
                log.exception("aeor_dfine: %s пропущен", Path(path).name)
    return out


def cap(feats: list[dict], max_objects: int) -> list[dict]:
    """Не больше max_objects объектов ae_dfine/or_dfine на модальность (самые уверенные)."""
    keep: list[dict] = []
    for g in GROUPS.values():
        mine = sorted((f for f in feats if f.get("threshold_group") == g), key=lambda f: -float(f["confidence"]))
        keep += mine[:max_objects]
    return [f for f in feats if f.get("threshold_group") not in GROUPS.values()] + keep
