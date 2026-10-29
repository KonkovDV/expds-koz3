"""bot-8: лидарные модели ConvNeXt bot-9 (in_kind = "cnx3" / "cnx5").

Контракт: bot-9/CNX_CONTRACT_FOR_BOT8.md §6 и bot-9/for_bot8/.

* Подготовка входа — ``inference/cnx/cnx_prep.py`` (побайтная копия bot-9) отдельным подпроцессом
  на весь LAS: у вендора bot-9 те же имена модулей (``expds``, ``expds_core``), в нашем процессе их
  импортировать нельзя. CRS — та же, что у пайплайна (``region_crs``: UTM.json).
* Сеть — ``expds.cnx_contract`` (копия bot-9): CnxUnetCenter + CnxFlat → 5 логитов
  [hm_t, hm_p, gor, fort, arch] на полном разрешении (hm шага 4 повторён nearest).
* Веса — ``weights_cnx_{variant}_f{fold}.pt`` через ``torch.load(weights_only=True)``; несколько
  файлов — ансамбль (среднее логитов). TTA — тоже среднее логитов, как при обучении.
* Точность: CUDA — autocast float16 (не bf16), окно с NaN/Inf пересчитывается в FP32; CPU — FP32.
* Окна: H, W дополняются до кратных 32 значением входа u8 0 (вне съёмки, −1,0079); окно 1024 с
  перекрытием 192 (CPU — 512/128), начало окна кратно 32 (и 4 — шагу hm).

Результат — объект с тем же интерфейсом, что Segmenter в ``pipeline._li_with_model``
(``meta``, ``classes``, ``res``, ``keep``) и уже посчитанными ``probs``/``transform`` для одного LAS:
дальше курганы идут тем же rescore (кандидаты-пики ЦМР → вероятность модели в центре),
остальные классы — масками ``probs_to_features``, пороги — группы li (li_coarse/li_mosaic).
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

KINDS = {"cnx3": 3, "cnx5": 5}
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
    "tta": 1,                              # 1…8 (обучение/OOF bot-9: 8)
    "amp": True,                           # fp16 autocast на CUDA
    "batch": 0,                            # 0 — авто (CUDA 2, CPU 1)
    "prep_timeout_s": 600,
    "soft_need_s": 120,
    "classes": None,                       # None — все 5 выходов; иначе подмножество
    "threshold_group": None,               # None — группы li / li_coarse / li_mosaic, как у li
}


class CnxError(RuntimeError):
    """Отказ cnx (подготовка, веса, инференс): пайплайн откатывается на модель li."""


def cnx_cfg(cfg: dict | None) -> dict:
    out = dict(DEFAULTS)
    out.update((cfg or {}).get("li_cnx") or {})
    return out


def enabled(cfg: dict | None) -> bool:
    c = (cfg or {}).get("li_cnx") or {}
    return bool(c.get("enabled", False)) and bool(c.get("weights"))


def weight_files(cfg: dict | None) -> list[str]:
    w = ((cfg or {}).get("li_cnx") or {}).get("weights") or []
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

    def __init__(self, paths, device: str | None = None, opts: dict | None = None):
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
        from .infer_nn import fp16_ok
        self.amp = bool(o.get("amp", True)) and fp16_ok(self.device)
        cuda = self.device.type == "cuda"
        self.tile = int(o["tile"] if cuda else o["tile_cpu"])
        self.overlap = int(o["overlap"] if cuda else o["overlap_cpu"])
        self.tta = max(1, min(int(o.get("tta") or 1), 8))
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

    # ------------------------------------------------------------------ сеть
    def _logits(self, xb, amp: bool):
        """Среднее логитов по весам и TTA (k: бит0 flip W, бит1 flip H, бит2 транспонирование)."""
        import torch

        acc = None
        n = 0
        for net in self.nets:
            for k in range(self.tta):
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

    def _forward(self, xb):
        import torch

        z = self._logits(xb, self.amp)
        if self.amp and not bool(torch.isfinite(z).all()):
            self.nan_fallbacks += 1
            if self.nan_fallbacks == 1:
                log.warning("cnx: NaN/Inf в fp16 — окно пересчитано в FP32")
            z = self._logits(xb, False)
        if not bool(torch.isfinite(z).all()):
            raise CnxError("cnx: NaN/Inf в логитах и в FP32")
        return z

    def predict(self, x: np.ndarray, deadline=None) -> np.ndarray:
        """x: float32 C×H×W (нормировка контракта) → вероятности K×H×W float32 (sigmoid среднего логитов).

        deadline — вызываемый без аргументов, True = время вышло (между окнами): CnxError.
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
        with torch.inference_mode():
            while i < len(pos):
                if deadline is not None and deadline():
                    raise CnxError(f"cnx: время вышло на окне {i}/{len(pos)}")
                chunk = pos[i:i + batch]
                xb = torch.from_numpy(np.stack([x[:, r:r + th, q:q + tw] for r, q in chunk])).to(self.device)
                try:
                    z = self._forward(xb).cpu().numpy()
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
    if ch == 5 and not (deps / "rvt").is_dir():
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
             time_left=None, prep_dir: Path | None = None) -> CnxBound:
    """Подготовка LAS и инференс → CnxBound. Любой отказ — CnxError (или исключение torch)."""
    timeout = float(opts.get("prep_timeout_s") or DEFAULTS["prep_timeout_s"])
    if time_left is not None:
        timeout = min(timeout, float(time_left()))
    t0 = time.monotonic()
    x, geo = run_prep(las, seg.in_channels, crs, timeout, prep_dir)
    if list(geo.get("tags") or []) != list(seg.meta["li_tags"]):
        raise CnxError(f"каналы подготовки {geo.get('tags')} ≠ in_tags модели {seg.meta['li_tags']}")
    if not _crs_equal(geo.get("crs"), dem_crs):
        raise CnxError(f"CRS подготовки {geo.get('crs')} ≠ CRS ЦМР пайплайна {dem_crs}")
    t1 = time.monotonic()
    probs = seg.predict(x, deadline=deadline)
    del x
    log.info("cnx %s %s: %s, окно %d/%d, tta %d, %s, %.1f с инференс (подготовка %.1f с)", seg.kind,
             Path(las).name, "×".join(map(str, probs.shape)), seg.tile, seg.overlap, seg.tta,
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


def load(cfg: dict, root: Path, device: str | None = None, prep_dir: Path | None = None) -> CnxSegmenter:
    files = weight_files(cfg)
    paths = [Path(root) / f for f in files]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise CnxError(f"нет весов cnx: {missing}")
    seg = CnxSegmenter(paths, device=device, opts=cnx_cfg(cfg))
    check_prep_code(seg.contract, prep_dir)
    log.info("cnx: %s, %d весов (%s), классы %s, окно %d/%d, tta %d, %s", seg.kind, len(seg.nets),
             ", ".join(seg.meta["files"]), seg.classes, seg.tile, seg.overlap, seg.tta,
             "fp16" if seg.amp else "fp32")
    return seg
