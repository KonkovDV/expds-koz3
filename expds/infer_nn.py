"""Инференс U-Net скользящим окном и превращение вероятностей в объекты.

Вероятности по классам → связные области выше нижнего порога → разделение
слипшихся курганов по пикам вероятности → полигоны с уверенностью = максимум
вероятности в объекте. Окончательный отбор — порогами по классам (post).
"""
from __future__ import annotations

import logging
from pathlib import Path

import os

import numpy as np

from . import accel
from .raster import mask_to_polygons

log = logging.getLogger(__name__)

#: Как считать на платформе. Пороги подобраны в FP32; половинная точность и TF32
#: их сдвигают. Включается из конфига [infer] через configure_runtime.
_RUNTIME = {"amp": False, "tf32": False, "deterministic": True}

#: Радиус разделения экземпляров по пикам (м); 0 — не делить (крупные объекты).
SPLIT_RADIUS_M = {"kurgany_povrezhdennye": 4.0, "kurgany_tselye": 4.0, "arkhitektury": 0.0,
                  "gorodishcha": 0.0, "fortifikatsii": 0.0, "object_poly": 0.0, "finds_points": 1.0}


def configure_runtime(amp: bool = False, tf32: bool = False, deterministic: bool = True) -> None:
    """Фиксирует точность инференса. По умолчанию FP32 без TF32: так подобраны пороги."""
    import torch

    _RUNTIME["amp"] = bool(amp)
    _RUNTIME["tf32"] = bool(tf32)
    _RUNTIME["deterministic"] = bool(deterministic)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    # torch ≥ 2.9: после allow_tf32 = False свёртки остаются в "none" (наследование), не в "ieee"
    conv = getattr(torch.backends.cudnn, "conv", None)
    if conv is not None and hasattr(conv, "fp32_precision"):
        conv.fp32_precision = "tf32" if tf32 else "ieee"
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = bool(deterministic)
    log.info("amp=%s tf32=%s deterministic=%s", _RUNTIME["amp"], _RUNTIME["tf32"], _RUNTIME["deterministic"])


def fp16_ok(device) -> bool:
    """FP16 (AMP) только на CUDA и не на GTX 16xx: у этих карт половинная точность в cuDNN даёт NaN."""
    import torch

    return device.type == "cuda" and "GTX 16" not in torch.cuda.get_device_name(device)


def inference_amp(device) -> bool:
    """AMP инференса: и карта умеет FP16, и конфиг его не запретил."""
    return bool(_RUNTIME["amp"]) and fp16_ok(device)


class Segmenter:
    """Загруженная модель из чекпойнта expds.train."""

    def __init__(self, path: str | Path, device: str | None = None):
        import torch
        import segmentation_models_pytorch as smp

        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.meta = {k: v for k, v in ck.items() if k != "state_dict"}
        self.classes: list[str] = ck["classes"]
        self.res: float = ck["res"]
        self.tile: int = ck["tile"]
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        arch = ck.get("arch") or "Unet"
        arch_kwargs = ck.get("arch_kwargs") or {}
        # Сданные веса без поля arch — тот же U-Net, что и раньше. Другой декодер только из чекпойнта.
        def build():
            if arch == "Unet" and not arch_kwargs:
                return smp.Unet(encoder_name=ck["encoder"], encoder_weights=None,
                                in_channels=ck["in_channels"], classes=len(self.classes))
            return smp.create_model(arch, encoder_name=ck["encoder"], encoder_weights=None,
                                    in_channels=ck["in_channels"], classes=len(self.classes),
                                    **arch_kwargs)

        # Сеть строится на устройстве meta (без случайной инициализации весов, ~0,2 с на модель),
        # веса чекпойнта подставляются как есть (assign). Если что-то осталось без значения
        # (буфер вне state_dict) — обычный путь.
        model = None
        if os.environ.get("EXPDS_META_INIT", "1") == "1":
            with torch.device("meta"):
                model = build()
            model.load_state_dict(ck["state_dict"], assign=True)
            if any(t.is_meta for t in (*model.parameters(), *model.buffers())):
                model = None
        if model is None:
            model = build()
            model.load_state_dict(ck["state_dict"])
        self.model = model
        self.model.to(self.device).eval()
        self.amp = inference_amp(self.device)
        self.tta = 1        # 1, 2, 4 или 8 вариантов из поворотов/отражений (модели учились с ними)

    def _auto_batch(self) -> int:
        """Батч окон по свободной памяти.

        Замер 21.10, GTX 1650, FP32, тот же U-Net, тайл 128, батч 64:
        прямой проход 75 мс; карта 1400×1600 — 0,32 с, из них почти всё этот проход.
        channels_last на том же батче 128 мс и сдвигает логиты до 4e-4: не используется.
        Сбор окон одним index вместо цикла — те же числа и не быстрее 75 мс ядер.
        torch.compile, TF32 и граф CUDA сюда не входят. TF32 и компилятор меняют
        ядра свёртки, а пороги сидят у 0,99. Граф на батче 64 замедлил тот же
        проход до 0,55 с: запись графа на этой карте дороже запуска ядер.
        """
        if self.device.type != "cuda":
            return 16
        import torch

        free, _ = torch.cuda.mem_get_info(self.device)
        channels = int(self.meta.get("in_channels") or 3)
        # Вход и активации U-Net: грубая оценка 48 float32 на пиксель канала.
        per = 4 * channels * int(self.tile) * int(self.tile) * 48
        return int(min(64, max(8, free * 0.25 / max(per, 1))))

    @staticmethod
    def _oom(exc: BaseException) -> bool:
        return "out of memory" in str(exc).lower()

    def _forward(self, xb):
        """Сигмоида модели, усреднённая по self.tta отражениям/транспонированиям окна."""
        import torch

        outs = []
        for k in range(max(1, min(int(self.tta), 8))):
            t = xb.flip(-1) if k & 1 else xb
            t = t.flip(-2) if k & 2 else t
            t = t.transpose(-1, -2) if k & 4 else t
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
                p = self.model(t).float().sigmoid()
            p = p.transpose(-1, -2) if k & 4 else p
            p = p.flip(-2) if k & 2 else p
            outs.append(p.flip(-1) if k & 1 else p)
        return outs[0] if len(outs) == 1 else torch.stack(outs).mean(0)

    def predict(self, img: np.ndarray, overlap: float = 0.25, batch: int | None = None) -> np.ndarray:
        """img: C×H×W uint8 на рабочем шаге модели → вероятности K×H×W float32."""
        import torch

        c, h, w = img.shape
        t = self.tile
        if batch is None:
            batch = self._auto_batch()
        batch = max(1, int(batch))
        step = max(1, int(t * (1 - overlap)))
        ys = list(range(0, max(h - t, 0) + 1, step)) or [0]
        xs = list(range(0, max(w - t, 0) + 1, step)) or [0]
        if ys[-1] + t < h:
            ys.append(h - t)
        if xs[-1] + t < w:
            xs.append(w - t)
        pad_h, pad_w = max(0, t - h), max(0, t - w)
        # Без паддинга np.pad всё равно копирует весь снимок: берём как есть.
        x_all = np.pad(img, ((0, 0), (0, pad_h), (0, pad_w))) if pad_h or pad_w else np.ascontiguousarray(img)
        ramp = np.minimum(np.arange(t) + 1, np.arange(t)[::-1] + 1).astype(np.float32)
        weight = np.minimum.outer(ramp, ramp)
        weight /= weight.max()
        coords = [(y, x) for y in ys for x in xs]
        need = x_all.nbytes + (len(self.classes) + 1) * x_all.shape[1] * x_all.shape[2] * 4
        if self.device.type == "cuda":
            free = torch.cuda.mem_get_info(self.device)[0]
            # Слоты — по желанию (EXPDS_SLOT_ACCUM=1): побитно то же, но на CPU медленнее цикла,
            # а на CUDA мелкие ядра цикла в основном прячутся за асинхронным проходом сети.
            if os.environ.get("EXPDS_SLOT_ACCUM") == "1" and _slots_ok(ys, xs, t, step) and need + slot_bytes(len(self.classes), x_all.shape[1:]) < 0.6 * free:
                return self._predict_on_device_slots(x_all, ys, xs, weight, batch, step)[:, :h, :w]
            if need < 0.6 * free:
                return self._predict_on_device(x_all, coords, weight, batch)[:, :h, :w]
        acc = np.zeros((len(self.classes), x_all.shape[1], x_all.shape[2]), np.float32)
        wsum = np.zeros(x_all.shape[1:], np.float32)
        with torch.inference_mode():
            s = 0
            while s < len(coords):
                chunk = coords[s:s + batch]
                xb = np.stack([x_all[:, y:y + t, x:x + t] for y, x in chunk]).astype(np.float32)
                xb = torch.from_numpy((xb / 255.0 - 0.5) / 0.25).to(self.device)
                try:
                    pr = self._forward(xb).cpu().numpy()
                except RuntimeError as exc:
                    del xb
                    if not self._oom(exc) or batch == 1:
                        raise
                    torch.cuda.empty_cache()
                    batch = max(1, batch // 2)
                    log.warning("нехватка памяти, батч окон уменьшен до %s", batch)
                    continue
                for (y, x), p in zip(chunk, pr):
                    acc[:, y:y + t, x:x + t] += p * weight
                    wsum[y:y + t, x:x + t] += weight
                s += len(chunk)
        return (acc / np.maximum(wsum, 1e-6))[:, :h, :w]

    def _predict_on_device(self, x_all: np.ndarray, coords: list[tuple[int, int]], weight: np.ndarray,
                           batch: int) -> np.ndarray:
        """Как predict, но снимок, окна и накопление — в памяти устройства; на CPU — только итог."""
        import torch

        t = self.tile
        dev = self.device
        x = torch.from_numpy(x_all).to(dev)
        acc = torch.zeros((len(self.classes), *x.shape[1:]), device=dev)
        wsum = torch.zeros(x.shape[1:], device=dev)
        wt = torch.from_numpy(weight).to(dev)
        with torch.inference_mode():
            s = 0
            while s < len(coords):
                chunk = coords[s:s + batch]
                xb = torch.stack([x[:, y:y + t, x0:x0 + t] for y, x0 in chunk]).float()
                xb = xb.div_(255.0).sub_(0.5).div_(0.25)
                try:
                    pr = self._forward(xb)
                except RuntimeError as exc:
                    del xb
                    if not self._oom(exc) or batch == 1:
                        raise
                    torch.cuda.empty_cache()
                    batch = max(1, batch // 2)
                    log.warning("нехватка памяти, батч окон уменьшен до %s", batch)
                    continue
                for (y, x0), p in zip(chunk, pr):
                    acc[:, y:y + t, x0:x0 + t] += p * wt
                    wsum[y:y + t, x0:x0 + t] += wt
                s += len(chunk)
        return (acc / wsum.clamp_min(1e-6)).cpu().numpy()


    def _predict_on_device_slots(self, x_all: np.ndarray, ys: list[int], xs: list[int], weight: np.ndarray,
                                 batch: int, step: int) -> np.ndarray:
        """То же, что _predict_on_device, бит в бит, но без питоновского цикла по окнам.

        Окна строки/столбца с одной чётностью номера не пересекаются (шаг ≥ t/2), последнее
        добавочное окно у края — свой слот. Поэтому окно батча не складывается, а
        записывается (index_copy_) в плоскость своего слота: в плоскости у пикселя не
        больше одного вклада. В конце вклады пикселя суммируются в том же порядке, что
        и в цикле (окна по строкам, затем по столбцам, начиная с нуля), — сложения
        float32 те же, результат совпадает побитно. Вместо 3–4 ядер на окно — ≤ 9
        записей на батч и 9 сборок на снимок.
        """
        import torch

        t = self.tile
        dev = self.device
        k_cls = len(self.classes)
        c = k_cls + 1                       # последний канал — сумма весов
        _, hh, ww = x_all.shape
        hw = hh * ww
        sy, cov_y = _axis_slots(ys, t, hh, step)  # слот окна по номеру; слоты пикселя по порядку
        sx, cov_x = _axis_slots(xs, t, ww, step)
        x = torch.from_numpy(x_all).to(dev)
        planes = torch.zeros((10, c, hw), device=dev)   # 3×3 слота + нулевая плоскость
        wt = torch.from_numpy(weight).to(dev)
        base = (torch.arange(t, device=dev)[:, None] * ww + torch.arange(t, device=dev)[None, :]).reshape(-1)
        coords = [(iy, ix) for iy in range(len(ys)) for ix in range(len(xs))]
        with torch.inference_mode():
            s = 0
            while s < len(coords):
                chunk = coords[s:s + batch]
                xb = torch.stack([x[:, ys[iy]:ys[iy] + t, xs[ix]:xs[ix] + t] for iy, ix in chunk]).float()
                xb = xb.div_(255.0).sub_(0.5).div_(0.25)
                try:
                    pr = self._forward(xb)
                except RuntimeError as exc:
                    del xb
                    if not self._oom(exc) or batch == 1:
                        raise
                    torch.cuda.empty_cache()
                    batch = max(1, batch // 2)
                    log.warning("нехватка памяти, батч окон уменьшен до %s", batch)
                    continue
                vals = torch.cat([pr * wt, wt.expand(len(chunk), 1, t, t)], dim=1)
                groups: dict[int, list[int]] = {}
                for j, (iy, ix) in enumerate(chunk):
                    groups.setdefault(sy[iy] * 3 + sx[ix], []).append(j)
                for slot, js in groups.items():
                    off = torch.tensor([ys[chunk[j][0]] * ww + xs[chunk[j][1]] for j in js], device=dev)
                    idx = (off[:, None] + base[None, :]).reshape(-1)
                    sel = vals if len(js) == len(chunk) else vals[torch.tensor(js, device=dev)]
                    planes[slot].index_copy_(1, idx, sel.permute(1, 0, 2, 3).reshape(c, -1))
                s += len(chunk)
            acc = torch.zeros((c, hw), device=dev)
            for a in range(cov_y.shape[0]):
                for b in range(cov_x.shape[0]):
                    ry, rx = cov_y[a], cov_x[b]
                    n = np.where((ry[:, None] >= 0) & (rx[None, :] >= 0), ry[:, None] * 3 + rx[None, :], 9)
                    n = torch.from_numpy(n.reshape(1, 1, -1)).to(dev).expand(1, c, hw)
                    acc += torch.gather(planes, 0, n)[0]
            del planes
            acc = acc.view(c, hh, ww)
        return (acc[:k_cls] / acc[k_cls].clamp_min(1e-6)).cpu().numpy()


def _axis_slots(starts: list[int], t: int, n: int, step: int) -> tuple[list[int], np.ndarray]:
    """Слоты окон одной оси и для каждого пикселя оси — слоты накрывающих его окон по порядку.

    Слот окна — чётность номера, у последнего добавочного окна (не на шаге) — 2.
    Возвращает (слот окна, массив A×n: слот a-го по порядку окна над пикселем, −1 — нет).
    """
    slot = [2 if s0 != i * step else i % 2 for i, s0 in enumerate(starts)]
    cnt = np.zeros(n, np.int64)
    cov = np.full((4, n), -1, np.int64)
    for i, s0 in enumerate(starts):
        seg = slice(s0, min(s0 + t, n))
        cov[cnt[seg], np.arange(seg.start, seg.stop)] = slot[i]
        cnt[seg] += 1
    a = int(cnt.max()) if n else 1
    return slot, cov[:max(a, 1)]


def _slots_ok(ys: list[int], xs: list[int], t: int, step: int) -> bool:
    """Слоты применимы, если окна одной чётности не пересекаются: шаг ≥ t/2, все окна на сетке
    шага, кроме, может быть, последнего (добавочного у края)."""
    if 2 * step < t:
        return False
    return all(s0 == i * step for st in (ys, xs) for i, s0 in enumerate(st[:-1]))


def slot_bytes(k_cls: int, shape: tuple[int, int]) -> int:
    """Память плоскостей слотов: 10 плоскостей и сумма, по (классы + вес) float32."""
    return 11 * (k_cls + 1) * int(shape[0]) * int(shape[1]) * 4


class Ensemble:
    """Несколько моделей одной схемы: вероятность — среднее их карт.

    Модели должны совпадать классами, шагом, тайлом и входом (каналы li, DEM),
    иначе карты нельзя усреднять. meta и прочие поля — от первой модели.
    """

    _SAME = ("classes", "res", "tile", "in_channels", "li_tags", "li_source", "dem_res", "dem_fill", "markup")

    def __init__(self, members: list[Segmenter], only: list[str] | None = None):
        """only — классы, по которым карты усредняются. Остальные берутся у первой модели."""
        if not members:
            raise ValueError("пустой ансамбль")
        first = members[0].meta
        for m in members[1:]:
            diff = [k for k in self._SAME if m.meta.get(k) != first.get(k)]
            if diff:
                raise ValueError(f"модели ансамбля различаются: {diff}")
        self.members = members
        self.meta = first
        self.classes = members[0].classes
        self.res = members[0].res
        self.tile = members[0].tile
        self.keep = None
        unknown = [c for c in (only or []) if c not in self.classes]
        if unknown:
            raise ValueError(f"классы ансамбля не из модели: {unknown}")
        self.only = [self.classes.index(c) for c in only] if only else None

    @property
    def tta(self) -> int:
        return self.members[0].tta

    @tta.setter
    def tta(self, value: int) -> None:
        for m in self.members:
            m.tta = value

    def predict(self, img: np.ndarray, overlap: float = 0.25, batch: int | None = None) -> np.ndarray:
        acc = self.members[0].predict(img, overlap, batch)
        if self.only is not None:
            idx = self.only
            part = acc[idx].copy()
            for member in self.members[1:]:
                part += member.predict(img, overlap, batch)[idx]
            acc[idx] = part / np.float32(len(self.members))
            return acc
        for member in self.members[1:]:
            acc += member.predict(img, overlap, batch)
        return acc / np.float32(len(self.members))


def split_by_peaks(prob: np.ndarray, mask: np.ndarray, radius_px: int, min_peak: float) -> np.ndarray:
    """Экземпляры: связные области маски, разделённые по ближайшему пику вероятности."""
    from scipy.ndimage import distance_transform_edt, find_objects, label

    lab, n = label(mask)
    if n == 0 or radius_px <= 0:
        return lab
    rows, cols, _ = accel.find_peaks(np.where(mask, prob, np.nan).astype(np.float32), radius_px, min_peak)
    if len(rows) < 2:
        return lab
    peaks_of: dict[int, list[tuple[int, int]]] = {}
    for r, c in zip(rows, cols):
        peaks_of.setdefault(int(lab[r, c]), []).append((int(r), int(c)))
    inst = lab.copy()
    next_id = n + 1
    for region, sl in enumerate(find_objects(lab), start=1):
        pk = peaks_of.get(region, [])
        if sl is None or len(pk) < 2:
            continue  # область с одним пиком (или без пиков) остаётся целой
        sub = lab[sl] == region
        markers = np.zeros(sub.shape, np.int32)
        for k, (r, c) in enumerate(pk, start=1):
            markers[r - sl[0].start, c - sl[1].start] = k
        _, (ir, ic) = distance_transform_edt(markers == 0, return_indices=True)
        nearest = markers[ir, ic]
        block = inst[sl]
        block[sub] = nearest[sub] + next_id - 1
        next_id += len(pk)
    return inst


def probs_to_features(probs: np.ndarray, classes: list[str], transform, crs: str | None, res: float,
                      region: str, markup: str, source: str, low: float = 0.05,
                      min_area_px: int = 3, min_conf: dict[str, float] | None = None) -> list[dict]:
    """Вероятности K×H×W → объекты пайплайна (полигоны EPSG:3857 с уверенностью).

    min_conf — пороги выдачи по классам: экземпляры ниже них отбрасываются до
    полигонизации. Итог после post.apply_thresholds тот же, но GDAL и shapely не
    строят тысячи полигонов, которые всё равно отсекаются. Для OOF не задавать.
    """
    out = []
    for k, cls in enumerate(classes):
        p = probs[k]
        mask = p >= low
        if not mask.any():
            continue
        floor = (min_conf or {}).get(cls)
        if floor is not None and float(p.max()) < floor:
            continue
        r = SPLIT_RADIUS_M.get(cls, 0.0)
        inst = split_by_peaks(p, mask, int(round(r / res)) if r else 0, low * 2)
        n = int(inst.max())
        if n == 0:
            continue
        sizes = np.bincount(inst.ravel(), minlength=n + 1)
        peak = accel.label_max(inst, p, n)
        keep = sizes >= min_area_px
        if floor is not None:
            keep &= peak.astype(np.float64) >= floor
        inst = np.where(keep[inst], inst, 0)
        if not inst.any():
            continue
        for poly, lab in mask_to_polygons(inst, transform, crs, simplify=res * 0.5):
            out.append({"geom": poly, "class_name": cls, "confidence": float(peak[lab]), "region_name": region,
                        "markup_type": markup, "source": source, "original_crs": crs})
    return out
