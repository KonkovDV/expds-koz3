"""bot-8: лидарные модели ConvNeXt bot-9 (in_kind = "cnx3" / "cnx5").

Контракт: bot-9/CNX_CONTRACT_FOR_BOT8.md §6 и bot-9/for_bot8/.

* Подготовка входа — ``inference/cnx/cnx_prep.py`` (побайтная копия bot-9) отдельным подпроцессом
  на весь LAS: у вендора bot-9 те же имена модулей (``expds``, ``expds_core``), в нашем процессе их
  импортировать нельзя. CRS — та же, что у пайплайна (``region_crs``: UTM.json).
* Сеть — ``expds.cnx_contract`` (копия bot-9): CnxUnetCenter + CnxFlat → 5 логитов
  [hm_t, hm_p, gor, fort, arch] на полном разрешении (hm шага 4 повторён nearest).
* Веса — ``weights_cnx_{variant}_f{fold}.pt`` через ``torch.load(weights_only=True)``; несколько
  файлов — ансамбль (среднее логитов). TTA — тоже среднее логитов, как при обучении.
* Точность — как ``[infer] amp`` (infer_nn.inference_amp): amp=true на CUDA — autocast float16 (никогда
  bf16), окно с NaN/Inf пересчитывается в FP32; amp=false и CPU — FP32. Ключ ``li_cnx.amp`` не читается.
* TTA — ``li_cnx.tta`` ∈ {1, 2, 8}: 1 — без отражений, 2 — + отражение по W, 8 — группа D8 (как OOF и
  будущие пороги bot-9, среднее логитов); по умолчанию 8. Время инференса ∝ tta. Если до мягкого потолка
  (или конца бюджета) прогноз не укладывается, TTA понижается 8→2→1 с предупреждением в журнале.
* Участок из нескольких LAS (контракт §6.1): все LAS участка сливаются в один (сетка по min/max всех
  точек) и готовятся целиком; LAS не режется (это сломало бы нормировку svf/opp у cnx5). Паритет с
  обучением на таких участках не гарантирован (на обучении слияние делал свой код bot-9).
* Окна: H, W дополняются до кратных 32 значением входа u8 0 (вне съёмки, −1,0079); окно 1024 с
  перекрытием 192 (CPU — 512/128), начало окна кратно 32 (и 4 — шагу hm).

Результат — объект с тем же интерфейсом, что Segmenter в ``pipeline._li_with_model``
(``meta``, ``classes``, ``res``, ``keep``) и уже посчитанными ``probs``/``transform`` для участка:
курганы идут тем же rescore (кандидаты-пики ЦМР каждого LAS → вероятность модели в центре),
остальные классы (только если явно в ``li_cnx.classes``) — масками ``probs_to_features``.

Встраивание (pipeline.cnx_phase) — ДОБАВОЧНОЕ: модель li и всё остальное работают как без cnx, cnx
отдельной фазой после оптики добавляет объекты, а дубли уже найденных (курганы ближе 12 м к оставленным
кругам, маски — пересечение с полигоном того же класса) удаляются. По умолчанию classes = курганы.

Пороги: шкала уверенности cnx своя, пороги li к ней НЕ применяются. Нужна своя группа
``thresholds_by_markup.<threshold_group>`` (по умолчанию ``li_cnx``) со значением для каждого класса
выхода. Её нет (или это группа li*) — cnx объектов НЕ выдаёт (громкое предупреждение), выдача li не
меняется; check_submission такой архив отклоняет (FAIL). OOF-порогов cnx пока нет — cnx выключен.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

KINDS = {"cnx3": 3, "cnx5": 5, "cnx12": 12}   # cnx12 — 05.11, kaggle/2026-11-05/li_c12_in
#: Каталог вендора подготовки внутри решения (inference/cnx).
PREP_DIR = Path(__file__).resolve().parent.parent / "inference" / "cnx"
#: Значение входа для u8 0 (вне съёмки) — ровно как cnx_prep.u8_to_input.
PAD_VALUE = float((np.zeros(1, np.uint8).astype(np.float32) - 128.0)[0] / np.float32(127.0))
HW_MULT = 32

DEFAULTS = {
    "enabled": False,
    "weights": [],
    "tile": 1024, "overlap": 192,          # CUDA
    "tile_cpu": 512, "overlap_cpu": 128,   # CPU: память ≤ 2 ГБ
    "tta": 8,                              # {1, 2, 8}; 8 = D8, как OOF bot-9
    "scales": [1.0],                       # пирамида зума кадра; [1.0] — один проход, как без ключа
    "batch": 0,                            # 0 — авто (CUDA 2, CPU 1)
    "prep_timeout_s": 600,
    "soft_need_s": 120,
    "classes": ["kurgany_povrezhdennye", "kurgany_tselye"],   # gor/fort/arch — только явно, после замера
    "threshold_group": "li_cnx",           # своя группа порогов; группы li* запрещены (другая шкала)
    "dedupe_m": 12.0,                      # курган cnx ближе — дубль оставленного круга
    # 05.11 (поправка 1 PREREG c5-in): "rescore" — курганы по пикам ЦМР с вероятностью сети (как было);
    # "head" — курганы декодированием головы, как OOF ядра bot-9 (expds/lidar_cnx_head.py), по параметрам контракта.
    "decode": "rescore",
    # вариант R гейта: курганы групп li / li_coarse / li_mosaic участка снимаются, курганы cnx идут без отбора дублей
    "replace_li_kurgans": False,
    # ансамбль (PREREG li_ens_in): второй набор весов другого входа (c12 при c5 в weights), только decode = "head";
    # одна подготовка на max(каналов), логиты всех сетей усредняются
    "weights_b": [],
}
DECODE_ALLOWED = ("rescore", "head")
#: Группы курганов лидарного слоя, которые снимает вариант R (как LI_KUR_GROUPS гейта c5-in).
LI_KUR_GROUPS = ("li", "li_coarse", "li_mosaic")


def is_li_kurgan(f: dict, region: str) -> bool:
    """Курган лидарного слоя участка (группа — threshold_group, иначе markup_type ∈ LI_KUR_GROUPS): его снимает R."""
    return (f.get("region_name") == region and f.get("class_name") in ("kurgany_povrezhdennye", "kurgany_tselye")
            and (f.get("threshold_group") or f.get("markup_type")) in LI_KUR_GROUPS)
TTA_ALLOWED = (1, 2, 8)
#: Группы порогов других лидарных моделей: к шкале cnx не применимы.
FORBIDDEN_GROUPS = frozenset({"li", "li_coarse", "li_mosaic", "li_ae", "li_cpu", "li_dev", "li_mask", "li_fort"})
KURGANS = ("kurgany_povrezhdennye", "kurgany_tselye")


class CnxError(RuntimeError):
    """Отказ cnx (подготовка, веса, инференс): пайплайн откатывается на модель li."""


def cnx_cfg(cfg: dict | None) -> dict:
    out = dict(DEFAULTS)
    out.update((cfg or {}).get("li_cnx") or {})
    return out


def threshold_problem(cfg: dict | None, classes=None) -> str | None:
    """None — у cnx есть свои пороги для всех классов выхода; иначе текст причины (cnx тогда молчит)."""
    o = cnx_cfg(cfg)
    group = o.get("threshold_group")
    if not group or not isinstance(group, str):
        return "li_cnx.threshold_group не задан"
    if group in FORBIDDEN_GROUPS:
        return f"li_cnx.threshold_group={group!r} — пороги другой модели (шкала cnx своя)"
    own = ((cfg or {}).get("thresholds_by_markup") or {}).get(group)
    if not isinstance(own, dict) or not own:
        return f"нет [thresholds_by_markup.{group}] — OOF-порогов cnx нет"
    need = list(classes if classes is not None else (o.get("classes") or []))
    lost = [c for c in need if c not in own]
    if lost:
        return f"в [thresholds_by_markup.{group}] нет порогов для {lost}"
    return None


def _scale_list(raw) -> list[float]:
    """Множители зума кадра. Один множитель 1 — отдельный проход не запускается."""
    if isinstance(raw, (int, float)):
        raw = [raw]
    if not raw:
        return [1.0]
    out = []
    for item in raw:
        value = float(item)
        if value <= 0:
            raise CnxError(f"li_cnx.scales: множитель {item!r} должен быть > 0")
        out.append(value)
    return out


def _one_scale(scales: list[float]) -> bool:
    return len(scales) == 1 and abs(scales[0] - 1.0) < 1e-9


def tta_value(raw) -> int:
    try:
        t = int(raw)
    except (TypeError, ValueError):
        raise CnxError(f"li_cnx.tta={raw!r}: допустимо {TTA_ALLOWED}") from None
    if t not in TTA_ALLOWED or isinstance(raw, bool):
        raise CnxError(f"li_cnx.tta={raw!r}: допустимо {TTA_ALLOWED}")
    return t


_ENABLED_WARNED: set = set()


def enabled(cfg: dict | None) -> bool:
    """cnx включён, только если li_cnx.enabled — TOML-bool true и weights заданы. Не bool (строка "false"/"true",
    0/1) — ВЫКЛЮЧЕН с WARN: безопаснее остаться на проверенной базе, чем молча включить cnx по enabled = "false"
    (bool("false") is True). check_submission считает так же (cnx_on) и даёт WARN."""
    c = (cfg or {}).get("li_cnx") or {}
    en = c.get("enabled", False)
    if not isinstance(en, bool):
        if repr(en) not in _ENABLED_WARNED:
            _ENABLED_WARNED.add(repr(en))
            log.warning("li_cnx.enabled=%r — не bool (в TOML true/false без кавычек): cnx ВЫКЛЮЧЕН", en)
        return False
    return en and bool(c.get("weights"))


def weight_files(cfg: dict | None, key: str = "weights") -> list[str]:
    w = ((cfg or {}).get("li_cnx") or {}).get(key) or []
    return [w] if isinstance(w, str) else [str(x) for x in w]


def _out_classes(contract: dict) -> list[str]:
    """Имена 5 выходов CnxFlat: hm_* (шаг out_stride) и mask_* — по порядку out_names."""
    names = list(contract["out_names"])
    hm = [n[3:] for n in names if n.startswith("hm_")]
    mask = [n[5:] for n in names if n.startswith("mask_")]
    return hm + mask


def _ramp(n: int, overlap: int) -> np.ndarray:
    i = np.arange(n, dtype=np.float32)
    r = np.minimum(i + 1, n - i)
    return np.minimum(r, max(1, overlap // 2)).astype(np.float32)


def _starts(n: int, t: int, step: int) -> list[int]:
    s = list(range(0, max(1, n - t + 1), step))
    if s[-1] != n - t:
        s.append(n - t)
    return s


class CnxSegmenter:
    """Ансамбль весов контракта bot-9 одного in_kind. predict: float32 C×H×W → вероятности K×H×W."""

    def __init__(self, paths, device: str | None = None, opts: dict | None = None, amp: bool | None = None):
        import torch

        from .cnx_contract import CnxFlat, build_from_arch, load_contract

        paths = [Path(p) for p in ([paths] if isinstance(paths, (str, Path)) else paths)]
        if not paths:
            raise CnxError("нет файлов весов cnx")
        o = dict(DEFAULTS)
        o.update(opts or {})
        self.opts = o
        self.device = torch.device(device or os.environ.get("EXPDS_DEVICE")
                                   or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.nets, ref = [], None
        for p in paths:
            c = load_contract(p)
            kind = c.get("in_kind")
            if kind not in KINDS or int(c["in_channels"]) != KINDS[kind]:
                raise CnxError(f"{p.name}: in_kind={kind!r}, in_channels={c.get('in_channels')}")
            sig = (kind, int(c["in_channels"]), tuple(c["out_names"]), tuple(c["out_stride"]),
                   float(c.get("cell_m", 0.5)), c.get("prep_code_sha"))
            if ref is None:
                ref, self.contract = sig, {k: v for k, v in c.items() if k != "state_dict"}
            elif sig != ref:
                raise CnxError(f"{p.name}: контракт не совпадает с {paths[0].name}: {sig} ≠ {ref}")
            net = build_from_arch(c["arch"])
            net.load_state_dict({k: v.float() for k, v in c["state_dict"].items()})
            self.nets.append(CnxFlat(net, c).to(self.device).eval())
            del c
        self.kind, self.in_channels = ref[0], ref[1]
        self.stride = max(int(s) for s in ref[3])
        self.classes = _out_classes(self.contract)
        self.res = ref[4]
        self.keep = set(o["classes"]) if o.get("classes") else None
        if self.keep is not None and self.keep - set(self.classes):
            raise CnxError(f"li_cnx.classes {sorted(self.keep - set(self.classes))} нет у модели {self.classes}")
        from .infer_nn import fp16_ok, inference_amp
        if "amp" in (opts or {}):
            log.warning("li_cnx.amp не читается: точность cnx — как [infer] amp")
        # [infer] amp (configure_runtime); amp= — только для тестов. fp16 лишь там, где fp16_ok (CUDA).
        self.amp = (inference_amp(self.device) if amp is None else bool(amp)) and fp16_ok(self.device)
        cuda = self.device.type == "cuda"
        self.tile = int(o["tile"] if cuda else o["tile_cpu"])
        self.overlap = int(o["overlap"] if cuda else o["overlap_cpu"])
        self.tta = tta_value(o.get("tta", DEFAULTS["tta"]))
        self.scales = _scale_list(o.get("scales", DEFAULTS["scales"]))
        self.batch = int(o.get("batch") or (2 if cuda else 1))
        if self.tile % HW_MULT or self.tile < HW_MULT or not 0 <= self.overlap < self.tile:
            raise CnxError(f"окно {self.tile}/{self.overlap}: окно кратно {HW_MULT}, перекрытие < окна")
        self.meta = {
            "in_kind": self.kind, "in_channels": self.in_channels, "li_source": "cnx",
            "li_tags": list(self.contract.get("in_tags") or []), "classes": self.classes, "res": self.res,
            # ЦМР пайплайна для кандидатов курганов — та же, что у модели li (0,1 м, Delaunay):
            "dem_res": 0.1, "dem_fill": "delaunay", "max_upsample": 2.0,
            "files": [p.name for p in paths], "prep_code_sha": ref[5], "tile": self.tile,
        }
        self.nan_fallbacks = 0
        self.tta_used = self.tta

    # ------------------------------------------------------------------ сеть
    def _logits(self, xb, amp: bool, tta: int | None = None, nets=None):
        """Среднее логитов по весам и TTA (k: бит0 flip W, бит1 flip H, бит2 транспонирование).
        tta 1 — k=0; 2 — k∈{0,1} (отражение по W); 8 — все 8 элементов D8."""
        import torch

        acc = None
        n = 0
        for net in (self.nets if nets is None else nets):
            for k in range(self.tta if tta is None else tta):
                t = xb.flip(-1) if k & 1 else xb
                t = t.flip(-2) if k & 2 else t
                t = t.transpose(-1, -2) if k & 4 else t
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=amp):
                    z = net(t).float()
                z = z.transpose(-1, -2) if k & 4 else z
                z = z.flip(-2) if k & 2 else z
                z = z.flip(-1) if k & 1 else z
                acc = z if acc is None else acc + z
                n += 1
        return acc / n

    def _forward(self, xb, tta: int | None = None):
        import torch

        z = self._logits(xb, self.amp, tta)
        if self.amp and not bool(torch.isfinite(z).all()):
            self.nan_fallbacks += 1
            if self.nan_fallbacks == 1:
                log.warning("cnx: NaN/Inf в fp16 — окно пересчитано в FP32")
            z = self._logits(xb, False, tta)
        if not bool(torch.isfinite(z).all()):
            raise CnxError("cnx: NaN/Inf в логитах и в FP32")
        return z

    def choose_tta(self, n_batches: int, t_one: float, budget_s: float | None) -> int:
        """TTA для прогона: наибольшее из {self.tta, 2, 1} (не больше self.tta), чей прогноз времени
        n_batches × весов × tta × t_one укладывается в budget_s. Понижение — предупреждение в журнале."""
        if budget_s is None or self.tta == 1:
            return self.tta
        for t in [v for v in (8, 2, 1) if v <= self.tta]:
            est = n_batches * len(self.nets) * t * t_one
            if est <= budget_s:
                if t != self.tta:
                    log.warning("cnx: TTA %d → %d: прогноз %.0f с при tta %d > запаса %.0f с (пороги — под tta %d)",
                                self.tta, t, n_batches * len(self.nets) * self.tta * t_one, self.tta,
                                budget_s, self.tta)
                return t
        log.warning("cnx: даже tta 1 (прогноз %.0f с) не укладывается в %.0f с — tta 1, дальше дедлайн",
                    n_batches * len(self.nets) * t_one, budget_s)
        return 1

    def predict(self, x: np.ndarray, deadline=None, budget_s: float | None = None) -> np.ndarray:
        """Вероятности K×H×W. scales = [1.0] — ровно один проход _predict_native.

        Несколько множителей (0,8 / 1 / 1,25) — среднее вероятностей после возврата
        на исходную сетку. Время растёт пропорционально числу множителей.
        """
        if _one_scale(self.scales):
            return self._predict_native(x, deadline, budget_s)
        from scipy import ndimage as ndi

        base = np.asarray(x, np.float32)
        _, h, w = base.shape
        acc = None
        for scale in self.scales:
            src = base if abs(scale - 1.0) < 1e-9 else np.stack(
                [ndi.zoom(base[i], scale, order=1) for i in range(base.shape[0])]).astype(np.float32)
            prob = self._predict_native(src, deadline, budget_s)
            if prob.shape[1] != h or prob.shape[2] != w:
                fy, fx = h / prob.shape[1], w / prob.shape[2]
                prob = np.stack([ndi.zoom(prob[i], (fy, fx), order=1) for i in range(prob.shape[0])])
                prob = prob[:, :h, :w]
                if prob.shape[1] < h or prob.shape[2] < w:
                    pad = np.zeros((prob.shape[0], h, w), np.float32)
                    pad[:, :prob.shape[1], :prob.shape[2]] = prob
                    prob = pad
            acc = prob.astype(np.float32) if acc is None else acc + prob.astype(np.float32)
        return (acc / np.float32(len(self.scales))).astype(np.float32)

    def _predict_native(self, x: np.ndarray, deadline=None, budget_s: float | None = None) -> np.ndarray:
        """x: float32 C×H×W (нормировка контракта) → вероятности K×H×W float32 (sigmoid среднего логитов).

        deadline — вызываемый без аргументов, True = время вышло (между окнами): CnxError.
        budget_s — секунд на инференс (мягкий потолок / бюджет): по замеру одного прохода первого окна
        TTA может быть понижен 8→2→1 (choose_tta); фактический — в self.tta_used.
        """
        import torch

        x = np.asarray(x, np.float32)
        if x.ndim != 3 or x.shape[0] != self.in_channels:
            raise CnxError(f"вход {x.shape}, ожидается {self.in_channels}×H×W")
        c, h, w = x.shape
        hp, wp = (-(-max(h, 1) // HW_MULT) * HW_MULT, -(-max(w, 1) // HW_MULT) * HW_MULT)
        if (hp, wp) != (h, w):
            x = np.pad(x, ((0, 0), (0, hp - h), (0, wp - w)), constant_values=PAD_VALUE)
        th, tw = min(self.tile, hp), min(self.tile, wp)
        step_h = max(HW_MULT, ((th - self.overlap) // HW_MULT) * HW_MULT)
        step_w = max(HW_MULT, ((tw - self.overlap) // HW_MULT) * HW_MULT)
        k = len(self.classes)
        acc = np.zeros((k, hp, wp), np.float32)
        wsum = np.zeros((hp, wp), np.float32)
        wt = np.outer(_ramp(th, self.overlap), _ramp(tw, self.overlap)).astype(np.float32)
        pos = [(r, q) for r in _starts(hp, th, step_h) for q in _starts(wp, tw, step_w)]
        batch = max(1, self.batch)
        i = 0
        tta = self.tta
        with torch.inference_mode():
            if budget_s is not None and self.tta > 1:
                r, q = pos[0]
                xb = torch.from_numpy(np.ascontiguousarray(x[None, :, r:r + th, q:q + tw])).to(self.device)
                t_one = float("inf")
                for _ in range(2):                            # 1-й проход — прогрев (cudnn, кэши)
                    t0 = time.perf_counter()               # monotonic на Windows шагает ~15 мс
                    self._logits(xb, self.amp, 1, self.nets[:1])
                    if self.device.type == "cuda":
                        torch.cuda.synchronize()
                    t_one = min(t_one, time.perf_counter() - t0)
                t_one *= batch                                # грубо: пакет ~ batch окон
                tta = self.choose_tta(-(-len(pos) // batch), t_one, budget_s)
            self.tta_used = tta
            while i < len(pos):
                if deadline is not None and deadline():
                    raise CnxError(f"cnx: время вышло на окне {i}/{len(pos)}")
                chunk = pos[i:i + batch]
                xb = torch.from_numpy(np.stack([x[:, r:r + th, q:q + tw] for r, q in chunk])).to(self.device)
                try:
                    z = self._forward(xb, tta).cpu().numpy()
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower() and batch > 1:
                        batch //= 2
                        torch.cuda.empty_cache()
                        continue
                    raise
                for (r, q), zz in zip(chunk, z):
                    acc[:, r:r + th, q:q + tw] += zz * wt
                    wsum[r:r + th, q:q + tw] += wt
                i += len(chunk)
        acc /= wsum
        acc = acc[:, :h, :w]
        return (1.0 / (1.0 + np.exp(-acc))).astype(np.float32)


class CnxBound:
    """CnxSegmenter + посчитанные для одного LAS вероятности: «сегментер» для _li_with_model."""

    def __init__(self, seg: CnxSegmenter, probs: np.ndarray, transform, geo: dict):
        self.seg, self.probs, self.transform, self.geo = seg, probs, transform, geo
        self.meta, self.classes, self.res, self.keep = seg.meta, seg.classes, seg.res, seg.keep

    def predict(self, img):            # не вызывается: вход cnx не из relief_channels
        raise CnxError("CnxBound.predict: вероятности уже посчитаны")

    def view(self, keep) -> "CnxBound":
        """Тот же результат с другим набором выходных классов (курганы по кандидатам / маски отдельно)."""
        b = CnxBound(self.seg, self.probs, self.transform, self.geo)
        b.keep = set(keep)
        return b


# ---------------------------------------------------------------------- подготовка
def prep_files() -> list[Path]:
    """Файлы вендора подготовки, нужные в архиве при включённом cnx (без __pycache__)."""
    return sorted(p for p in PREP_DIR.rglob("*") if p.is_file() and "__pycache__" not in p.parts)


def run_prep(las, ch: int, crs: str | None, timeout_s: float, prep_dir: Path | None = None,
             tmp_root: str | None = None) -> tuple[np.ndarray, dict]:
    """cnx_prep.py подпроцессом на весь LAS → (x float32 [ch,H,W], geo). Временный npz удаляется."""
    pd = Path(prep_dir or PREP_DIR)
    script, vendor, deps = pd / "cnx_prep.py", pd / "vendor", pd / "_deps"
    if not script.is_file() or not vendor.is_dir():
        raise CnxError(f"нет вендора подготовки cnx в {pd}")
    if ch in (5, 12) and not (deps / "rvt").is_dir():
        try:
            import importlib.util
            if importlib.util.find_spec("rvt") is None:
                raise CnxError("нет rvt (rvt-py 2.2.1) для каналов svf_r/opp_r")
        except ValueError:
            raise CnxError("нет rvt (rvt-py 2.2.1)")
    if timeout_s <= 0:
        raise CnxError("нет времени на подготовку cnx")
    tmp_root = tmp_root or os.environ.get("EXPDS_TMP") or None
    if tmp_root:
        Path(tmp_root).mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="cnx_prep_", dir=tmp_root))
    try:
        out = tmp / "x.npz"
        env = {k: v for k, v in os.environ.items() if k not in ("CNX_VENDOR",)}
        env["PYTHONPATH"] = os.pathsep.join([str(deps)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        cmd = [sys.executable, str(script), str(las), "--ch", str(int(ch)), "--out", str(out),
               "--vendor", str(vendor)] + (["--crs", str(crs)] if crs else [])
        t0 = time.monotonic()
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, env=env, cwd=str(tmp))
        except subprocess.TimeoutExpired:
            raise CnxError(f"cnx_prep не уложился в {timeout_s:.0f} с") from None
        if r.returncode != 0 or not out.is_file():
            raise CnxError(f"cnx_prep код {r.returncode}: {(r.stderr or r.stdout)[-600:]}")
        with np.load(out, allow_pickle=False) as z:
            u8 = z["u8"]
            geo = json.loads(str(z["geo"]))
        log.info("cnx_prep %s: ch=%d %s за %.1f с (бэкенд %s, factor %s)", Path(las).name, ch,
                 "×".join(map(str, u8.shape)), time.monotonic() - t0, geo.get("backend"), geo.get("dem_factor"))
        x = (u8.astype(np.float32) - 128.0) / 127.0     # = cnx_prep.u8_to_input
        return x, geo
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def merge_las(paths, out: Path, chunk: int = 1_000_000) -> Path:
    """Слияние LAS/LAZ участка в один LAS (контракт §6.1): точки всех файлов подряд, сетка подготовки —
    по min/max всех точек (заголовок пишет laspy). Память — один кусок chunk точек.

    Формат точек и CRS (VLR LASF_Projection) — первого файла; общие поля копируются (x, y, z,
    classification, …), шаг квантования — наименьший из файлов, смещение — минимум по файлам.
    """
    import laspy

    paths = [Path(p) for p in paths]
    hs = []
    for p in paths:
        with laspy.open(str(p)) as f:
            hs.append(f.header)
    h0 = hs[0]
    hdr = laspy.LasHeader(point_format=h0.point_format.id, version=h0.version)
    hdr.scales = np.min(np.array([h.scales for h in hs], np.float64), axis=0)
    mins = np.min(np.array([h.mins for h in hs], np.float64), axis=0)
    maxs = np.max(np.array([h.maxs for h in hs], np.float64), axis=0)
    hdr.offsets = np.floor(mins)
    span = np.ceil((maxs - hdr.offsets) / hdr.scales)
    if np.any(span >= 2 ** 31 - 1):
        raise CnxError(f"слияние LAS: охват {maxs - mins} не влезает в int32 при шаге {hdr.scales}")
    for v in h0.vlrs:
        if getattr(v, "user_id", "") == "LASF_Projection":
            hdr.vlrs.append(v)
    out_dims = set(hdr.point_format.dimension_names) - {"X", "Y", "Z"}
    n = 0
    with laspy.open(str(out), mode="w", header=hdr) as w:
        for p in paths:
            with laspy.open(str(p)) as f:
                common = sorted(out_dims & set(f.header.point_format.dimension_names))
                for pts in f.chunk_iterator(chunk):
                    rec = laspy.ScaleAwarePointRecord.zeros(len(pts), header=hdr)
                    rec.x, rec.y, rec.z = pts.x, pts.y, pts.z
                    for name in common:
                        rec[name] = pts[name]
                    w.write_points(rec)
                    n += len(pts)
    log.info("cnx: слито %d LAS участка → %s (%d точек)", len(paths), out.name, n)
    return out


def geo_transform(geo: dict):
    from rasterio.transform import Affine

    a, b, c, d, e, f = (float(v) for v in geo["transform"])
    return Affine(a, b, c, d, e, f)


def _crs_equal(a, b) -> bool:
    if a is None or b is None:
        return True
    if str(a) == str(b):
        return True
    try:
        from pyproj import CRS
        return CRS.from_user_input(a) == CRS.from_user_input(b)
    except Exception:
        return False


def bind_las(seg: CnxSegmenter, las, crs: str | None, opts: dict, dem_crs=None, deadline=None,
             time_left=None, prep_dir: Path | None = None, infer_budget=None) -> CnxBound:
    """Подготовка участка и инференс → CnxBound. Любой отказ — CnxError (или исключение torch).

    las — один путь или список LAS участка: несколько — сначала merge_las во временный LAS (удаляется).
    infer_budget — вызываемый → секунд на инференс (для понижения TTA) или None.
    """
    timeout = float(opts.get("prep_timeout_s") or DEFAULTS["prep_timeout_s"])
    if time_left is not None:
        timeout = min(timeout, float(time_left()))
    t0 = time.monotonic()
    files = [Path(p) for p in (las if isinstance(las, (list, tuple)) else [las])]
    if not files:
        raise CnxError("нет LAS")
    tmp = None
    try:
        if len(files) > 1:
            root = os.environ.get("EXPDS_TMP") or None
            if root:
                Path(root).mkdir(parents=True, exist_ok=True)
            tmp = Path(tempfile.mkdtemp(prefix="cnx_merge_", dir=root))
            src = merge_las(files, tmp / "site.las")
            if time_left is not None:
                timeout = min(timeout, float(time_left()))
        else:
            src = files[0]
        x, geo = run_prep(src, seg.in_channels, crs, timeout, prep_dir)
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
    las = files[0] if len(files) == 1 else Path(f"{files[0].stem}+{len(files) - 1}")
    if list(geo.get("tags") or []) != list(seg.meta["li_tags"]):
        raise CnxError(f"каналы подготовки {geo.get('tags')} ≠ in_tags модели {seg.meta['li_tags']}")
    if not _crs_equal(geo.get("crs"), dem_crs):
        raise CnxError(f"CRS подготовки {geo.get('crs')} ≠ CRS ЦМР пайплайна {dem_crs}")
    t1 = time.monotonic()
    probs = seg.predict(x, deadline=deadline, budget_s=None if infer_budget is None else float(infer_budget()))
    del x
    log.info("cnx %s %s: %s, окно %d/%d, tta %d, %s, %.1f с инференс (подготовка %.1f с)", seg.kind,
             Path(las).name, "×".join(map(str, probs.shape)), seg.tile, seg.overlap, seg.tta_used,
             "fp16" if seg.amp else "fp32", time.monotonic() - t1, t1 - t0)
    return CnxBound(seg, probs, geo_transform(geo), geo)


def check_prep_code(contract: dict, prep_dir: Path | None = None) -> None:
    """sha256 файлов вендора == contract["prep_code_files"] (на них учились веса); иначе CnxError."""
    import hashlib

    want = contract.get("prep_code_files") or {}
    if not want:
        raise CnxError("в весах нет prep_code_files — вход не проверить")
    vendor = Path(prep_dir or PREP_DIR) / "vendor"
    bad = []
    for rel, sha in sorted(want.items()):
        f = vendor / rel
        got = hashlib.sha256(f.read_bytes()).hexdigest() if f.is_file() else None
        if got != sha:
            bad.append(f"{rel}: {got and got[:12]} ≠ {sha[:12]}")
    if bad:
        raise CnxError(f"вендор подготовки не совпадает с весами: {bad}")


def load(cfg: dict, root: Path, device: str | None = None, prep_dir: Path | None = None,
         amp: bool | None = None, key: str = "weights") -> CnxSegmenter:
    files = weight_files(cfg, key)
    paths = [Path(root) / f for f in files]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise CnxError(f"нет весов cnx: {missing}")
    seg = CnxSegmenter(paths, device=device, opts=cnx_cfg(cfg), amp=amp)
    check_prep_code(seg.contract, prep_dir)
    log.info("cnx: %s, %d весов (%s), классы %s, окно %d/%d, tta %d, %s", seg.kind, len(seg.nets),
             ", ".join(seg.meta["files"]), seg.classes, seg.tile, seg.overlap, seg.tta,
             "fp16" if seg.amp else "fp32")
    return seg
