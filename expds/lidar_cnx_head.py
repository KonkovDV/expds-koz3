"""Курганы cnx-лидара декодированием головы «как в ядре» (`[li_cnx] decode = "head"`).

Зачем (поправка 1 PREREG c5-in, `kaggle/2026-11-05/li_cnx_in/PREREG.md`): гейт меряет OOF ядра bot-9, а прод
по умолчанию (`decode = "rescore"`) берёт кандидатов из пиков ЦМР и лишь снимает с карты сети вероятность.
Здесь курганы получаются тем же путём, что в ядре (`snapshot/2026-11-01/kaggle_push/koz3-cnx-li-c3sc-f4-v2`,
функции `predict_region` и `decode`):

1. окна `infer.tile` (512) с шагом `infer.stride` (384), только там, где есть съёмка (проверка каждого 8-го пикселя);
2. D8: rot90(k) + отражение по W; среднее логитов `hm`/`mask` и сырых `wh`/`off` после обратного поворота
   (`off` — поворот компонент, `centerhead.d8.torch_untta_offsets`); при нескольких весах — среднее и по весам;
3. сшивка средним по перекрытиям; sigmoid на `hm` и масках, обнуление вне съёмки;
4. `centerhead.decode`: пики `peak_pool`, `min_score`, NMS по расстоянию в классе; клетки головы, где маска
   городищ в среднем > 0,5, подавлены; круги `circle_r_m` в родной СК → EPSG:3857.

Параметры берутся из контракта весов (`infer`, `thresholds`), а не из конфига: на них получен OOF гейта.
Отличие от ядра — точность: ядро на GPU считало в fp16 (autocast), прод — как `[infer] amp` (по умолчанию FP32).
"""
from __future__ import annotations

import logging
import time

import numpy as np

log = logging.getLogger(__name__)

HM_CLASSES = ("kurgany_tselye", "kurgany_povrezhdennye")   # канал 0, 1 ключа hm (centerhead.config.HM_CLASSES)


class HeadError(RuntimeError):
    """Отказ декодирования головы: пайплайн выдаёт участок без курганов cnx."""


def params(contract: dict) -> dict:
    """Параметры инференса и декодирования из контракта весов (ровно те, которыми получен OOF)."""
    th = contract.get("thresholds") or {}
    inf = contract.get("infer") or {}
    need_th = ("mask_thr", "circle_r_m", "head_min_score", "head_peak_pool", "head_topk", "head_nms_min_m",
               "head_nms_frac", "head_nms_cross_class", "head_size_mode")
    lost = [k for k in need_th if k not in th] + [k for k in ("tile", "stride", "tta") if k not in inf]
    if lost:
        raise HeadError(f"в контракте весов нет параметров декодирования головы: {lost}")
    stride = int(contract["out_stride"][0])
    return dict(tile=int(inf["tile"]), step=int(inf["stride"]), tta=int(inf["tta"]), stride=stride,
                mask_thr=float(th["mask_thr"]), circle_r_m=float(th["circle_r_m"]),
                max_det=int(th.get("max_det") or th["head_topk"]),
                cfg=dict(min_score=float(th["head_min_score"]), peak_pool=int(th["head_peak_pool"]),
                         topk=int(th["head_topk"]), nms_min_m=float(th["head_nms_min_m"]),
                         nms_frac=float(th["head_nms_frac"]), nms_cross_class=bool(th["head_nms_cross_class"]),
                         size_mode=str(th["head_size_mode"]), stride=stride))


def tile_origins(H: int, W: int, S: int, step: int, stride: int) -> list[tuple[int, int]]:
    """= cnx_head.tile_origins ядра (и centerhead.decode.tile_origins): шаг step, последний тайл выровнен на stride."""
    if step % stride or S % stride:
        raise HeadError("step и S должны делиться на stride")

    def ax(n):
        last = max(0, (n - S) // stride * stride)
        return sorted(set(list(range(0, max(1, n - S), step)) + [last]))
    return [(y, x) for y in ax(H) for x in ax(W)]


def _blocks(a: np.ndarray, s: int) -> np.ndarray:
    H, W = a.shape
    h, w = -(-H // s), -(-W // s)
    p = np.zeros((h * s, w * s), np.float32)
    p[:H, :W] = a
    return p.reshape(h, s, w, s)


def block_any(a: np.ndarray, s: int) -> np.ndarray:
    return a.astype(bool) if s == 1 else _blocks(a, s).max(axis=(1, 3)) > 0


def block_mean(a: np.ndarray, s: int) -> np.ndarray:
    return a.astype(np.float32) if s == 1 else _blocks(a, s).mean(axis=(1, 3))


def _untta_map(t, k: int, fl: bool):
    import torch
    t = t.flip(3) if fl else t
    return torch.rot90(t, -k, (2, 3))


def predict_maps(seg, x: np.ndarray, valid: np.ndarray, p: dict, deadline=None, batch: int = 8, extra=None) -> dict:
    """x: float32 (C,H,W) в нормировке контракта; valid: (H,W) bool (u8 > 0). → карты как у predict_region ядра:
    {"hm": (2,h,w) prob, "wh": (1,h,w), "off": (2,h,w), "mask": (3,H,W) prob, "_valid": (H,W), "_valid_h": (h,w)}.
    extra — [(seg_b, x_b), …]: ансамбль (PREREG li_ens_in) — логиты всех сетей всех наборов и всех видов D8 усредняются
    вместе; x_b той же формы H×W (свои каналы). Без extra вычисления и порядок сложения — прежние."""
    import torch

    from centerhead.d8 import torch_untta_offsets

    from .lidar_cnx import PAD_VALUE

    C, H, W = x.shape
    members = [([flat.net for flat in seg.nets], x)] + [([flat.net for flat in sb.nets], xb_) for sb, xb_ in (extra or [])]
    if any(xm.shape[1:] != (H, W) for _, xm in members):
        raise HeadError(f"входы ансамбля разной формы: {[xm.shape for _, xm in members]}")
    S, st, s = p["tile"], p["step"], p["stride"]
    h4, w4, n4 = -(-H // s), -(-W // s), S // s
    tiles = [t for t in tile_origins(H, W, S, st, s) if valid[t[0]:t[0] + S:8, t[1]:t[1] + S:8].any()]
    tf = [(k, f) for f in (False, True) for k in range(4)][:max(1, int(p["tta"]))]
    acc: dict = {}
    cnt = np.zeros((H, W), np.float32)
    cnt4 = np.zeros((h4, w4), np.float32)
    dev = seg.device
    amp = bool(seg.amp)

    def window(xm, yy, xx):
        out = np.full((xm.shape[0], S, S), PAD_VALUE, np.float32)
        hh, ww = min(S, H - yy), min(S, W - xx)
        out[:, :hh, :ww] = xm[:, yy:yy + hh, xx:xx + ww]
        return out

    with torch.inference_mode():
        for b in range(0, len(tiles), batch):
            if deadline is not None and deadline():
                raise HeadError(f"время вышло на окне {b}/{len(tiles)}")
            bt = tiles[b:b + batch]
            o: dict = {}
            n = 0
            for nets, xm in members:
                xb = torch.from_numpy(np.stack([window(xm, yy, xx) for yy, xx in bt])).to(dev)
                for net in nets:
                    for k, fl in tf:
                        xi = torch.rot90(xb, k, (2, 3))
                        xi = xi.flip(3) if fl else xi
                        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                            yo = net(xi.contiguous())
                        yo = {kk: v.float() for kk, v in yo.items() if v is not None}
                        r = {"mask": _untta_map(yo["mask"], k, fl), "hm": _untta_map(yo["hm"], k, fl)}
                        if yo.get("wh") is not None:
                            r["wh"] = _untta_map(yo["wh"], k, fl)
                        if yo.get("off") is not None:
                            r["off"] = torch_untta_offsets(yo["off"], k, fl)
                        for kk, v in r.items():
                            o[kk] = o[kk] + v if kk in o else v
                        n += 1
            for kk, v in o.items():
                v = (v / n).cpu().numpy()
                full = kk == "mask"
                Hh, Ww, nn_, d = (H, W, S, 1) if full else (h4, w4, n4, s)
                if kk not in acc:
                    acc[kk] = np.zeros((v.shape[1], Hh, Ww), np.float32)
                for (yy, xx), oi in zip(bt, v):
                    y0, x0 = yy // d, xx // d
                    hh, ww = min(nn_, Hh - y0), min(nn_, Ww - x0)
                    acc[kk][:, y0:y0 + hh, x0:x0 + ww] += oi[:, :hh, :ww]
            for yy, xx in bt:
                hh, ww = min(S, H - yy), min(S, W - xx)
                cnt[yy:yy + hh, xx:xx + ww] += 1
                y0, x0 = yy // s, xx // s
                hh, ww = min(n4, h4 - y0), min(n4, w4 - x0)
                cnt4[y0:y0 + hh, x0:x0 + ww] += 1
    va = valid & (cnt > 0)
    va4 = block_any(va, s) & (cnt4 > 0)
    for kk in acc:
        full = kk == "mask"
        c_, v_ = (cnt, va) if full else (cnt4, va4)
        acc[kk] /= np.maximum(c_, 1)[None]
        if kk in ("mask", "hm"):
            acc[kk] = (1.0 / (1.0 + np.exp(-acc[kk]))) * v_[None]
    if "hm" not in acc:
        acc = {"hm": np.zeros((2, h4, w4), np.float32), "mask": np.zeros((3, H, W), np.float32)}
    acc["_valid"], acc["_valid_h"] = va, va4
    return acc


def decode_kurgans(maps: dict, geo: dict, p: dict) -> list[tuple[float, float, int, float]]:
    """Курганы ядра: (X, Y в родной СК, класс 0 = целый / 1 = повреждённый, score). = decode() ядра, ветка головы."""
    from centerhead.config import cfg_with
    from centerhead.decode import decode

    s = p["stride"]
    st = float(geo["step"])
    gor = maps["mask"][0] > p["mask_thr"]
    sup = block_mean(gor, s) > 0.5
    t = decode(maps["hm"], float(geo["left"]), float(geo["top"]), st * s, wh=maps.get("wh"), off=maps.get("off"),
               valid=maps["_valid_h"] & ~sup, cfg=cfg_with(**p["cfg"]))
    n = p.get("max_det")
    if n and len(t["score"]) > n:
        o = np.argsort(-np.asarray(t["score"]), kind="stable")[:n]
        t = {k: np.asarray(v)[o] for k, v in t.items()}
    return [(float(X), float(Y), int(c), float(sc)) for X, Y, c, sc in zip(t["X"], t["Y"], t["cls"], t["score"])]


def to_features(dets, geo: dict, p: dict, region: str, source: str, classes=HM_CLASSES) -> list[dict]:
    """Круги circle_r_m (quad_segs=4, как в ядре) в родной СК → EPSG:3857; markup_type li."""
    from pyproj import Transformer
    from shapely.geometry import Point, mapping
    from shapely.ops import transform as stf

    keep = set(classes)
    to3857 = Transformer.from_crs(geo["crs"], "EPSG:3857", always_xy=True).transform
    out = []
    for X, Y, c, sc in dets:
        name = HM_CLASSES[c]
        if name not in keep:
            continue
        g = stf(to3857, Point(X, Y).buffer(p["circle_r_m"], quad_segs=4))
        out.append({"type": "Feature", "geometry": mapping(g), "geom": g, "class_name": name,
                    "confidence": round(min(1.0, max(0.0, sc)), 5), "region_name": region, "markup_type": "li",
                    "source": source, "original_crs": str(geo["crs"])})
    return out


def site_feats(seg, las, crs, opts: dict, region: str, deadline=None, time_left=None, prep_dir=None,
               seg_b=None) -> list[dict]:
    """Подготовка участка (как bind_las) → карты головы → курганы. Любой отказ — исключение (вызов ловит).
    seg_b — второй набор весов (ансамбль c5 + c12, PREREG li_ens_in): одна подготовка на max(каналов), каждому набору —
    свои первые каналы (теги контрактов — префиксы тегов подготовки), логиты усредняются (predict_maps extra)."""
    import shutil
    import tempfile
    from pathlib import Path

    from . import lidar_cnx as LC

    p = params(seg.contract)
    if seg_b is not None and params(seg_b.contract) != p:
        raise HeadError("у наборов весов ансамбля разные параметры инференса/декодирования в контрактах")
    ch = max(seg.in_channels, seg_b.in_channels) if seg_b is not None else seg.in_channels
    timeout = float(opts.get("prep_timeout_s") or LC.DEFAULTS["prep_timeout_s"])
    if time_left is not None:
        timeout = min(timeout, float(time_left()))
    files = [Path(f) for f in (las if isinstance(las, (list, tuple)) else [las])]
    if not files:
        raise HeadError("нет LAS")
    t0 = time.monotonic()
    tmp = None
    try:
        if len(files) > 1:
            tmp = Path(tempfile.mkdtemp(prefix="cnx_merge_"))
            src = LC.merge_las(files, tmp / "site.las")
        else:
            src = files[0]
        x, geo = LC.run_prep(src, ch, crs, timeout, prep_dir)
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
    tags = list(geo.get("tags") or [])
    for sg in [seg] + ([seg_b] if seg_b is not None else []):
        lt = list(sg.meta["li_tags"])
        if tags[:len(lt)] != lt or (seg_b is None and tags != lt):
            raise HeadError(f"каналы подготовки {tags} ≠ in_tags модели {lt}")
    valid = x[0] > LC.PAD_VALUE + 1e-6
    t1 = time.monotonic()
    if seg_b is None:
        maps = predict_maps(seg, x, valid, p, deadline=deadline)
    else:
        maps = predict_maps(seg, x[:seg.in_channels], valid, p, deadline=deadline, extra=[(seg_b, x[:seg_b.in_channels])])
    del x
    dets = decode_kurgans(maps, geo, p)
    feats = to_features(dets, geo, p, region, files[0].name, classes=opts.get("classes") or HM_CLASSES)
    log.info("cnx head %s: %d курганов (окно %d/%d, tta %d, %d весов, %s), подготовка %.1f с, сеть+декод %.1f с",
             region, len(feats), p["tile"], p["step"], p["tta"], len(seg.nets) + (len(seg_b.nets) if seg_b is not None else 0),
             "fp16" if seg.amp else "fp32",
             t1 - t0, time.monotonic() - t1)
    return feats
