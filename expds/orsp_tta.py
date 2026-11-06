"""TTA и мультимасштаб для U-Net спутника и D-FINE SpOr. По умолчанию выключено.

Включение — секция ``[orsp_tta] enabled = true``. Пока флаг выключен, ни U-Net, ни D-FINE
не меняют путь: спутник остаётся на ``[tta]`` / ``[ms]`` (нет секции → tta 1, один масштаб),
D-FINE остаётся на своём ``flip_tta``.

U-Net спутника (markup из ``unet_markups``, по умолчанию ``sp_or`` и ``sp_focal2``):
отражения — уже ``Segmenter._forward`` (биты tta 1/2/4/8), маски усредняются;
мультимасштаб — ``Segmenter.predict`` (зум кадра, затем среднее карт вероятностей).
Аэрофото и ортофото эта секция не трогает.

D-FINE SpOr: виды d4 (повороты на 90°) или d8 (повороты и отражение) и несколько
шагов земли (множители ``res_m``). Боксы одного растра сливаются WBF (по умолчанию)
или NMS. Между снимками и типами данных слияния нет.

Оценка D-FINE этим модулем не снималась: на пробной сборке группа sp_or
после D-FINE дала F1 0. Пока система координат и порог не проверены заново,
цифры D-FINE помечены «не оценено» и в честный замер не входят.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

_LOGGED: set[str] = set()

DEFAULTS = {
    "enabled": False,
    # U-Net спутника. 0 — не подменять [tta]; 1|2|4|8 — как Segmenter.tta.
    "unet_tta": 0,
    # Зум кадра, как [ms]: 1.25 — картинка крупнее (мельче метр/пиксель). [1.0] — выкл.
    "unet_scales": [1.0],
    "unet_markups": ["sp_or", "sp_focal2"],
    # D-FINE. off — штатный flip_tta внутри detect_scene. d4 — 4 поворота, d8 — ещё отражение.
    "dfine_views": "off",
    # Множители res_m. 0.75 — мельче 0,5 м, 1.25 — грубее. [1.0] — один масштаб.
    "dfine_gsd": [1.0],
    # Слияние видов одного растра до cluster_circles. wbf | nms.
    "dfine_fuse": "wbf",
    "dfine_fuse_iou": 0.55,
}

# Честный замер D-FINE с этими флагами не делался.
DFINE_EVAL = "не оценено"


def settings(cfg: dict) -> dict:
    own = dict(cfg.get("orsp_tta") or {})
    st = dict(DEFAULTS, **own)
    marks = st.get("unet_markups") or []
    st["unet_markups"] = [str(m) for m in marks]
    scales = st.get("unet_scales")
    if isinstance(scales, (int, float)):
        scales = [scales]
    st["unet_scales"] = [float(s) for s in (scales or [1.0])] or [1.0]
    gsd = st.get("dfine_gsd")
    if isinstance(gsd, (int, float)):
        gsd = [gsd]
    st["dfine_gsd"] = [float(s) for s in (gsd or [1.0])] or [1.0]
    try:
        st["unet_tta"] = int(st["unet_tta"])
    except (TypeError, ValueError):
        st["unet_tta"] = 0
    st["dfine_views"] = str(st.get("dfine_views") or "off")
    st["dfine_fuse"] = str(st.get("dfine_fuse") or "wbf")
    return st


def enabled(cfg: dict) -> bool:
    return bool(settings(cfg)["enabled"])


def _scales_active(scales: list[float]) -> bool:
    return not (len(scales) == 1 and abs(float(scales[0]) - 1.0) < 1e-9)


def apply_unet(cfg: dict, markup: str, tta: int, scales: list[float]) -> tuple[int, list[float]]:
    """Подмена tta и масштабов U-Net. Флаг выключен или чужой markup — аргументы как пришли."""
    st = settings(cfg)
    if not st["enabled"] or markup not in st["unet_markups"]:
        return tta, scales
    out_tta, out_scales = int(tta), [float(s) for s in scales]
    if st["unet_tta"] in (1, 2, 4, 8):
        out_tta = int(st["unet_tta"])
    elif st["unet_tta"] not in (0,):
        log.warning("orsp_tta: unet_tta %s не из {0,1,2,4,8} — tta спутника не меняю", st["unet_tta"])
    if _scales_active(st["unet_scales"]):
        out_scales = list(st["unet_scales"])
    if out_tta != int(tta) or out_scales != [float(s) for s in scales]:
        key = f"unet:{markup}"
        if key not in _LOGGED:
            _LOGGED.add(key)
            log.info("orsp_tta: U-Net %s tta %s→%s, масштабы %s→%s (среднее масок)",
                     markup, tta, out_tta, [float(s) for s in scales], out_scales)
    return out_tta, out_scales


def _view_tuples(name: str) -> list[tuple[int, bool]]:
    if name == "d4":
        return [(k, False) for k in range(4)]
    if name == "d8":
        return [(k, flip) for k in range(4) for flip in (False, True)]
    raise ValueError(name)


def dfine_plan(cfg: dict) -> dict | None:
    """None — D-FINE идёт штатным detect_scene. Иначе словарь видов и шагов земли.

    Включённый флаг только с U-Net (dfine_views=off и dfine_gsd=[1.0]) тоже даёт None:
    сломанный выход D-FINE не затрагивается замерем спутника.
    """
    st = settings(cfg)
    if not st["enabled"]:
        return None
    views = st["dfine_views"]
    gsd = list(st["dfine_gsd"])
    gsd_on = _scales_active(gsd)
    if views not in ("off", "d4", "d8"):
        log.error("orsp_tta: dfine_views %r неизвестен — виды D-FINE не включаю", views)
        views = "off"
    if views == "off" and not gsd_on:
        return None
    if st["dfine_fuse"] not in ("wbf", "nms"):
        log.error("orsp_tta: dfine_fuse %r неизвестен — беру wbf", st["dfine_fuse"])
        st["dfine_fuse"] = "wbf"
    key = "dfine"
    if key not in _LOGGED:
        _LOGGED.add(key)
        log.warning(
            "orsp_tta: D-FINE views=%s gsd=%s fuse=%s — %s "
            "(на пробной сборке F1 группы sp_or был 0; этот путь ещё не замерен)",
            views, gsd, st["dfine_fuse"], DFINE_EVAL,
        )
    return {
        "views": None if views == "off" else _view_tuples(views),
        "gsd": gsd,
        "fuse": st["dfine_fuse"],
        "iou": float(st["dfine_fuse_iou"]),
    }


def cost(cfg: dict) -> dict:
    """Множители прямых проходов относительно выключенного флага. Это не замер времени."""
    st = settings(cfg)
    unet_tta = st["unet_tta"] if st["enabled"] and st["unet_tta"] in (1, 2, 4, 8) else 1
    unet_scales = st["unet_scales"] if st["enabled"] and _scales_active(st["unet_scales"]) else [1.0]
    n_views = {"off": 1, "d4": 4, "d8": 8}.get(st["dfine_views"], 1)
    if not st["enabled"] or st["dfine_views"] == "off":
        n_views = 1
    n_gsd = len(st["dfine_gsd"]) if st["enabled"] and _scales_active(st["dfine_gsd"]) else 1
    # Штатный D-FINE уже делает горизонтальное отражение (flip_tta), это 2 прохода.
    # d4/d8 заменяют его, а не добавляются сверху.
    dfine_vs_flip = (n_views * n_gsd) / 2.0 if (st["enabled"] and (st["dfine_views"] != "off" or n_gsd > 1)) else 1.0
    if st["enabled"] and st["dfine_views"] == "off" and n_gsd > 1:
        dfine_vs_flip = float(n_gsd)  # каждый масштаб сам делает flip_tta
    return {
        "enabled": st["enabled"],
        "unet_factor": int(unet_tta) * len(unet_scales),
        "unet_tta": unet_tta,
        "unet_scales": unet_scales,
        "dfine_factor_vs_flip_tta": dfine_vs_flip,
        "dfine_views": st["dfine_views"] if st["enabled"] else "off",
        "dfine_gsd": st["dfine_gsd"] if st["enabled"] else [1.0],
        "dfine_eval": DFINE_EVAL,
    }


# ── боксы: обратное преобразование вида и перевод между масштабами ──────────
def _inv_point(x: float, y: float, rot: int, hflip: bool, size: float) -> tuple[float, float]:
    """Точка из координат повёрнутого кадра в исходный. size — сторона квадратного тайла.

    Прямое преобразование кадра: rot90 против часовой (k раз), затем отражение по ширине.
    Непрерывные координаты на отрезке [0, size], не индексы пикселей.
    """
    if hflip:
        x = size - x
    rot %= 4
    if rot == 0:
        return x, y
    if rot == 1:  # был CCW 90: (x, y) -> (y, size - x)
        return size - y, x
    if rot == 2:
        return size - x, size - y
    return y, size - x  # был CCW 270


def invert_xyxy(xyxy: np.ndarray, rot: int, hflip: bool, size: float) -> np.ndarray:
    """xyxy в пикселях повёрнутого тайла → xyxy исходного тайла."""
    src = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    out = np.empty_like(src)
    for i, (x0, y0, x1, y1) in enumerate(src):
        pts = (_inv_point(x0, y0, rot, hflip, size), _inv_point(x1, y0, rot, hflip, size),
               _inv_point(x1, y1, rot, hflip, size), _inv_point(x0, y1, rot, hflip, size))
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        out[i] = (min(xs), min(ys), max(xs), max(ys))
    return out


def boxes_to_base(xyxy, src_T, base_T) -> np.ndarray:
    """xyxy пикселей сцены src → пиксели базовой сцены (то же начало, другой шаг)."""
    src = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    if src.size == 0:
        return src.reshape(0, 4)

    def world(T, col, row):
        return T.a * col + T.b * row + T.c, T.d * col + T.e * row + T.f

    det = base_T.a * base_T.e - base_T.b * base_T.d
    if abs(det) < 1e-18:
        raise ValueError("базовый трансформ сцены вырожден")

    def pix(x, y):
        x = x - base_T.c
        y = y - base_T.f
        col = (base_T.e * x - base_T.b * y) / det
        row = (-base_T.d * x + base_T.a * y) / det
        return col, row

    out = np.empty_like(src)
    for i, (x0, y0, x1, y1) in enumerate(src):
        cols, rows = [], []
        for col, row in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
            c, r = pix(*world(src_T, col, row))
            cols.append(c)
            rows.append(r)
        out[i] = (min(cols), min(rows), max(cols), max(rows))
    return out


# ── слияние боксов одного растра ─────────────────────────────────────────────
def _iou(b: np.ndarray, other: np.ndarray) -> np.ndarray:
    w = np.clip(np.minimum(b[2], other[:, 2]) - np.maximum(b[0], other[:, 0]), 0, None)
    h = np.clip(np.minimum(b[3], other[:, 3]) - np.maximum(b[1], other[:, 1]), 0, None)
    inter = w * h
    ua = (b[2] - b[0]) * (b[3] - b[1]) + (other[:, 2] - other[:, 0]) * (other[:, 3] - other[:, 1]) - inter
    return np.where(ua > 0, inter / np.maximum(ua, 1e-9), 0.0)


def wbf(boxes, scores, labels, n_views: int, iou: float = 0.55):
    """Weighted boxes fusion по классу. Уверенность — сумма / max(n_views, число членов), обрезка до 1.

    Объект, видимый на всех видах, не штрафуется. Одиночный вид делится на n_views — меньше ложных.
    Координаты — среднее с весами-уверенностями. Разные классы не сливаются.
    """
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
                members.append([int(i)])
            else:
                members[j].append(int(i))
                m = np.asarray(members[j])
                w = scores[m]
                fused[j] = (boxes[m] * w[:, None]).sum(0) / max(float(w.sum()), 1e-12)
        for fb, m in zip(fused, members):
            oB.append(fb)
            oS.append(float(scores[m].sum() / max(n_views, len(m))))
            oL.append(int(c))
    if not oB:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    return np.asarray(oB), np.clip(np.asarray(oS), 0, 1), np.asarray(oL, dtype=int)


def nms(boxes, scores, labels, iou: float = 0.5):
    """NMS по классам. Остаётся бокс с большей уверенностью, пересечения подавляются."""
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
                ious = _iou(boxes[i], boxes[np.asarray(idx)])
                idx = [j for j, v in zip(idx, ious) if v < iou]
    keep = sorted(keep)
    if not keep:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    return boxes[keep], scores[keep], labels[keep]


def fuse_views(parts: list, how: str, iou: float, nms_iou: float):
    """Слияние видов одного растра. Пустой вид входит в n_views, но не добавляет боксов.

    Один вид возвращается как есть (без деления уверенности). Разные растры сюда не передавать.
    """
    n = len(parts)
    boxes, scores, labels = [], [], []
    for B, L, C in parts:
        if len(np.asarray(L).reshape(-1)) == 0:
            continue
        boxes.append(np.asarray(B, dtype=np.float64).reshape(-1, 4))
        scores.append(np.asarray(L, dtype=np.float64).reshape(-1))
        labels.append(np.asarray(C, dtype=int).reshape(-1))
    if not boxes:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    B = np.concatenate(boxes)
    L = np.concatenate(scores)
    C = np.concatenate(labels)
    if n <= 1:
        return B, L, C
    if how == "nms":
        return nms(B, L, C, nms_iou)
    return wbf(B, L, C, n, iou)


def _decode_view(model, px, rot: int, hflip: bool, tile: int, st: dict, device, use_amp: bool):
    """Один вид батча → список по окнам (xyxy тайла, score, cls) уже в координатах исходного тайла."""
    import torch

    t = px
    if rot:
        t = torch.rot90(t, int(rot) % 4, dims=(2, 3))
    if hflip:
        t = t.flip(3)
    with torch.autocast(getattr(device, "type", "cpu"), dtype=torch.float16, enabled=use_amp):
        o = model(pixel_values=t)
    prob = o.logits.float().sigmoid()
    bx = o.pred_boxes.float()
    Q, Cn = prob.shape[1:]
    sc_, idx = prob.flatten(1).topk(min(int(st["max_det"]), Q * Cn), 1)
    out = []
    size = float(tile)
    for i in range(px.shape[0]):
        keep = sc_[i] >= float(st["min_score"])
        q, c = idx[i][keep] // Cn, idx[i][keep] % Cn
        bb = bx[i][q] * tile
        xyxy = torch.stack([bb[:, 0] - bb[:, 2] / 2, bb[:, 1] - bb[:, 3] / 2,
                            bb[:, 0] + bb[:, 2] / 2, bb[:, 1] + bb[:, 3] / 2], 1)
        inv = invert_xyxy(xyxy.cpu().numpy(), rot, hflip, size)
        out.append((inv, sc_[i][keep].cpu().numpy().astype(np.float64), c.cpu().numpy().astype(int)))
    return out


def _nms_per_model(acc, nms_iou: float):
    import torch
    from torchvision.ops import batched_nms

    res = []
    for B, L, Cc in acc:
        if not B or all(t.numel() == 0 for t in B):
            res.append((np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)))
            continue
        Bt, Lt, Ct = torch.cat(B), torch.cat(L), torch.cat(Cc)
        k = batched_nms(Bt, Lt, Ct, float(nms_iou))
        res.append((Bt[k].numpy(), Lt[k].numpy(), Ct[k].numpy().astype(int)))
    return res


def detect_views(models, scene, st: dict, device, views: list[tuple[int, bool]], keep_going=lambda: True):
    """Окна сцены × виды. На каждый вид — NMS по окнам. Возвращает по моделям список видов (B, L, C)."""
    import torch

    from .sp_dfine import windows

    T, bs = int(st["tile"]), int(st["batch"])
    wins = windows(scene.W, scene.H, T, int(st["overlap"]))
    n_v = len(views)
    acc = [[[[], [], []] for _ in range(n_v)] for _ in models]
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
                for vi, (rot, hflip) in enumerate(views):
                    decoded = _decode_view(model, px, rot, hflip, T, st, device, use_amp)
                    for (x, y), (bb, s, c) in zip(chunk, decoded):
                        acc[mi][vi][0].append(torch.from_numpy(bb + np.array([x, y, x, y], dtype=np.float64)))
                        acc[mi][vi][1].append(torch.from_numpy(s))
                        acc[mi][vi][2].append(torch.from_numpy(c.astype(np.int64)))
    per_model = []
    for mi in range(len(models)):
        per_model.append(_nms_per_model(acc[mi], float(st["nms_iou"])))
    return per_model, done, len(wins)


def detect_dfine(models, base_scene, path, crs: str, st: dict, device, keep_going, plan: dict):
    """Виды и масштабы одного растра → боксы в пикселях базовой сцены (тот же transform, что у boxes_to_ellipses).

    gsd = 1 и views is None сюда не попадает (dfine_plan даёт None): штатный detect_scene не заменяется.
    """
    from .sp_dfine import Scene, detect_scene

    base_res = float(st["res_m"])
    per_model: list[list] = [[] for _ in models]
    done = total = 0
    for gsd in plan["gsd"]:
        own = None
        scene = base_scene
        if abs(float(gsd) - 1.0) > 1e-9:
            own = Scene(path, crs, base_res * float(gsd))
            scene = own
        try:
            if plan["views"] is None:
                res, d, t = detect_scene(models, scene, st, device, keep_going)
                batches = [[triple] for triple in res]
            else:
                batches, d, t = detect_views(models, scene, st, device, plan["views"], keep_going)
            done += d
            total += t
            for mi, parts in enumerate(batches):
                for B, L, C in parts:
                    per_model[mi].append((boxes_to_base(B, scene.T, base_scene.T), L, C))
        finally:
            if own is not None:
                own.close()
        if not keep_going():
            break
    nms_iou = float(st["nms_iou"])
    fused = [fuse_views(parts, plan["fuse"], float(plan["iou"]), nms_iou) for parts in per_model]
    return fused, done, total
