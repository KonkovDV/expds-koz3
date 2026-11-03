"""bot-7 (кандидат v3): шлюз городищ DINOv2-S bot-5 (FOR_BOT7_GOR_V3.md; ветка bot5-gor-patch-kaggle abb89d0 + 82058b8). Выключен.

Вариант bot-5 'gate': уверенность городищ SpOr (markup sp_or, без threshold_group) умножается на p_scene участка —
max p(городище) по всем виньеткам 384 м (шаг 192 м, контекст 768 м, 224 px) всех растров SpOr участка;
признаки — замороженный facebook/dinov2-small (Apache-2.0, ed25f3a3…) [CLS, mean(patch)] × 2 масштаба = 1536-d,
голова gor_v3_head.npz (z-норм. → PCA-64 → /qsd → логрегрессия). Порт extract_feats.py и gor_v3.py MODE=apply без
изменения арифметики. Замер bot-5 (honest-10 к A): +0,005057, LOFO +0,002423, 5/5 — ниже MDE, только в пакете.
При варианте R D-FINE (drop sp_or на пройденных участках) шлюз бессмыслен: целей нет — признаки не считаются.
Умноженные объекты получают threshold_group = ``sp_or_gated`` (порог bot-5 gor_v3_taus.json gate.sp_or 0,992648).
Участок, у которого пройдены не все растры (время), не трогается.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

GATED_GROUP = "sp_or_gated"
MARKUP = "sp_or"
DEFAULTS = {
    "enabled": False,
    "backbone": "models/dinov2-small",                 # config.json + model.safetensors, facebook/dinov2-small@ed25f3a31f01632728cabb09d1542f84ab7b0056
    "backbone_md5": "ea251be6e151ba8609ad0bc9b2e13a78",  # md5 model.safetensors (bot-5)
    "head": "models/gor_gate/gor_v3_head.npz",
    "head_md5": "acb95c52956d712a16c3dd389237a691",
    "device": "auto",
    "batch": 16,
    "soft_need_s": 20.0,     # мягкий потолок: запас на один растр
    "sec_per_patch": 0.45,   # справочно: CPU 2 потока, замер bot-7 с чтением растра 0,41–0,47 с/виньетку (bot-5: 0,13–0,16)
}
PS, ST, CTX, PX = 384.0, 192.0, 768.0, 224
MEAN = np.array([0.485, 0.456, 0.406])[:, None, None]
STD = np.array([0.229, 0.224, 0.225])[:, None, None]


def settings(cfg: dict) -> dict:
    return dict(DEFAULTS, **(cfg.get("gor_gate") or {}))


def enabled(cfg: dict) -> bool:
    return bool((cfg.get("gor_gate") or {}).get("enabled", False))


def _p(root: Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else Path(root) / p


def missing_artifacts(cfg: dict, root: Path) -> list[str]:
    st = settings(cfg)
    miss = []
    for key, rel in (("backbone", f"{st['backbone']}/config.json"), ("backbone", f"{st['backbone']}/model.safetensors"),
                     ("head", st["head"])):
        if not _p(root, rel).exists():
            miss.append(f"{key}={_p(root, rel)}")
    if "gorodishcha" not in ((cfg.get("thresholds_by_markup") or {}).get(GATED_GROUP) or {}):
        miss.append(f"[thresholds_by_markup.{GATED_GROUP}] gorodishcha")
    return miss


def is_gated_target(f: dict) -> bool:
    return (f.get("markup_type") == MARKUP and not f.get("threshold_group")
            and f.get("class_name") == "gorodishcha")


def gate_confidences(feats: list[dict], region_p: dict[str, float]) -> list[dict]:
    """conf городищ SpOr × p участка (max p виньеток); участок без p не трогается."""
    out = []
    for f in feats:
        reg = f.get("region_name")
        if is_gated_target(f) and reg in region_p:
            f = dict(f, confidence=float(f.get("confidence", 1.0)) * float(region_p[reg]),
                     threshold_group=GATED_GROUP)
        out.append(f)
    return out


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_head(path: Path, md5: str | None) -> dict:
    if md5 and _md5(path) != md5:
        raise RuntimeError(f"gor_gate: md5 головы {path} не {md5}")
    H = dict(np.load(path))
    for k in ("mu", "sd", "W", "qsd", "w", "feat_dim"):
        if k not in H:
            raise RuntimeError(f"gor_gate: в голове нет {k}")
    return H


def head_p(H: dict, F: np.ndarray) -> np.ndarray:
    """gor_v3.py MODE=apply: p = σ([1, ((F−mu)/sd)·W/qsd]·w)."""
    F = np.asarray(F, np.float32)
    assert F.shape[1] == int(H["feat_dim"]), "признаки не той модели (DINOv2-S, 1536)"
    Q = np.c_[np.ones(len(F)), (((F - H["mu"]) / H["sd"]) @ H["W"]) / H["qsd"]]
    return 1 / (1 + np.exp(-Q @ H["w"]))


#: transformers 5.x: имена слоёв внимания снапшота 4.x → новые (как conversion mapping from_pretrained; без хаба)
_KEY_RULES = ((".attention.attention.query.", ".attention.q_proj."), (".attention.attention.key.", ".attention.k_proj."),
              (".attention.attention.value.", ".attention.v_proj."), (".attention.output.dense.", ".attention.o_proj."))


def _rename(k: str) -> str:
    for a, b in _KEY_RULES:
        k = k.replace(a, b)
    return k


def load_backbone(st: dict, root: Path, device):
    """Dinov2Model из config.json + model.safetensors, без хаба (как load_models D-FINE)."""
    import torch
    from safetensors.torch import load_file

    from .sp_dfine import import_dfine

    import_dfine(root)   # тот же закреплённый transformers 5.21.0 (вендор → env той же версии), иначе ImportError
    from transformers import Dinov2Config, Dinov2Model

    d = _p(root, st["backbone"])
    w = d / "model.safetensors"
    if st.get("backbone_md5") and _md5(w) != st["backbone_md5"]:
        raise RuntimeError(f"gor_gate: md5 {w} не {st['backbone_md5']}")
    m = Dinov2Model(Dinov2Config.from_json_file(str(d / "config.json")))
    sd = {_rename(k): v for k, v in load_file(str(w)).items()}
    res = m.load_state_dict(sd, strict=False)
    if res.unexpected_keys or res.missing_keys:
        raise RuntimeError(f"gor_gate: ключи бэкбона: missing {res.missing_keys[:5]} unexpected {res.unexpected_keys[:5]}")
    if any(p.is_meta for p in m.parameters()):
        raise RuntimeError("gor_gate: meta-тензоры в бэкбоне")
    return m.eval().to(device)


def _device(st: dict):
    import torch

    if st["device"] != "auto":
        return torch.device(st["device"])
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tif_feats(model, path: Path, utm_epsg: int, device, batch: int = 16, keep_going=lambda: True):
    """Порт extract_feats.py bot-5 (та же растяжка, усреднение, сетка, отбор valid ≥ 0,5): (F [n,1536] float16, n_total, done)."""
    import rasterio
    import torch
    from pyproj import Transformer
    from rasterio.enums import Resampling

    with rasterio.open(path) as d:
        tr = Transformer.from_crs(d.crs, utm_epsg, always_xy=True)
        cx0, cy0 = (d.bounds.left + d.bounds.right) / 2, (d.bounds.bottom + d.bounds.top) / 2
        xa, ya = tr.transform([cx0, cx0 + d.transform.a * 100], [cy0, cy0])
        gsd = math.hypot(xa[1] - xa[0], ya[1] - ya[0]) / 100
        alpha = d.count >= 4 and d.colorinterp[3].name == "alpha"
        sc = max(1, int(max(d.width, d.height) / 1024))
        small = d.read(out_shape=(d.count, d.height // sc, d.width // sc), resampling=Resampling.average)
        vs = small[3] > 0 if alpha else np.all(small[:3] != (d.nodata if d.nodata is not None else -1e9), 0)
        st = []
        for b in range(3):
            v = small[b][vs].astype(np.float32)
            st.append(np.percentile(v, [2, 98]) if v.size > 100 else (0, 255))
        del small
        pps, pst = PS / gsd, ST / gsd
        dec = {}
        for size in (PS, CTX):
            f_ = (size / PX) / gsd
            Hh, Ww = max(1, int(round(d.height / f_))), max(1, int(round(d.width / f_)))
            arr = d.read(out_shape=(d.count, Hh, Ww), resampling=Resampling.average)
            val = arr[3] > 127 if alpha else (np.all(arr[:3] != (d.nodata if d.nodata is not None else -1e9), 0)
                                               & np.any(arr[:3] != 0, 0))
            x = np.zeros((3, Hh, Ww), np.float32)
            for b_ in range(3):
                x[b_] = np.clip((arr[b_] - st[b_][0]) / max(st[b_][1] - st[b_][0], 1e-6), 0, 1)
            x = ((x * val) - MEAN) / STD
            pad = PX
            xp = np.zeros((3, Hh + 2 * pad, Ww + 2 * pad), np.float32) + ((0 - MEAN) / STD).astype(np.float32)
            xp[:, pad:pad + Hh, pad:pad + Ww] = x
            vp = np.zeros((Hh + 2 * pad, Ww + 2 * pad), bool)
            vp[pad:pad + Hh, pad:pad + Ww] = val
            dec[size] = (xp, vp, f_, pad)
            del arr
        nx = max(1, int(math.ceil((d.width - pps) / pst)) + 1)
        ny = max(1, int(math.ceil((d.height - pps) / pst)) + 1)
    crops = []
    for iy in range(ny):
        for ix in range(nx):
            cpx, cpy = ix * pst + pps / 2, iy * pst + pps / 2
            one, ok = [], True
            for size in (PS, CTX):
                xp, vp, f_, pad = dec[size]
                c0, r0 = int(round(cpx / f_ - PX / 2)) + pad, int(round(cpy / f_ - PX / 2)) + pad
                if size == PS and vp[r0:r0 + PX, c0:c0 + PX].mean() < 0.5:
                    ok = False
                    break
                one.append(np.ascontiguousarray(xp[:, r0:r0 + PX, c0:c0 + PX]))
            if ok:
                crops.append(one)
    out = []
    with torch.no_grad():
        for i in range(0, len(crops), max(1, int(batch))):
            if not keep_going():
                return np.zeros((0, 1536), np.float16), len(crops), False
            ch = crops[i:i + batch]
            parts = []
            for s in range(2):
                o = model(pixel_values=torch.from_numpy(np.stack([c[s] for c in ch])).to(device)).last_hidden_state.float().cpu()
                parts.append(torch.cat([o[:, 0], o[:, 1:].mean(1)], 1).numpy())
            out.append(np.concatenate(parts, 1).astype(np.float16))
    F = np.concatenate(out) if out else np.zeros((0, 1536), np.float16)
    return F, len(crops), True


def _epsg(reg) -> int | None:
    from .catalog import read_utm_json

    c = read_utm_json(getattr(reg, "utm_json", None))
    paths = list((getattr(reg, "rasters", {}) or {}).get(MARKUP, []))
    if c and paths:   # bot-6 crsfix: географическая CRS UTM.json (008_Герите, 4326) → зона UTM по центру растра
        from .sp_dfine import utm_for
        c = utm_for(Path(paths[0]), c)
    try:
        return int(str(c).split(":")[-1]) if c else None
    except ValueError:
        return None


def apply(feats: list[dict], regions: dict, cfg: dict, root: Path, guard=None, soft_ok=None) -> list[dict]:
    """При enabled и полном наборе артефактов: p_scene по участкам с целями шлюза, затем gate_confidences.
    Нет артефактов / ошибка / нехватка времени — объекты (участка) без изменений и строка в журнале."""
    if not enabled(cfg):
        return feats
    miss = missing_artifacts(cfg, root)
    if miss:
        log.warning("gor_gate: нет артефактов (%s) — шлюз городищ не применён", "; ".join(miss))
        from . import degraded
        degraded.mark("gor_gate", "нет артефактов: " + "; ".join(miss))
        return feats
    st = settings(cfg)
    targets = sorted({f.get("region_name") for f in feats if is_gated_target(f)} & set(regions))
    if not targets:
        log.info("gor_gate: целей (городища sp_or без threshold_group) нет — признаки не считаются")
        return feats
    t0 = time.monotonic()
    try:
        H = load_head(_p(root, st["head"]), st.get("head_md5"))
        dev = _device(st)
        model = load_backbone(st, root, dev)
    except Exception:
        log.exception("GOR_GATE_UNAVAILABLE: голова/бэкбон не загружены — шлюз не применён, выдача как v2")
        from . import degraded
        degraded.mark("gor_gate", "GOR_GATE_UNAVAILABLE: голова/бэкбон не загружены")
        return feats
    keep = (lambda: guard.ok(30)) if guard is not None else (lambda: True)
    region_p: dict[str, float] = {}
    for name in targets:
        reg = regions[name]
        paths = list((getattr(reg, "rasters", {}) or {}).get(MARKUP, []))
        epsg = _epsg(reg)
        if not paths or epsg is None:
            log.warning("%s: gor_gate — нет SpOr или UTM.json, участок не трогается", name)
            continue
        ps, done = [], True
        for p in paths:
            if not keep() or (soft_ok is not None and not soft_ok(f"gor_gate {name}", float(st["soft_need_s"]))):
                done = False
                break
            try:
                F, n, ok = tif_feats(model, Path(p), epsg, dev, int(st["batch"]), keep)
            except Exception:
                log.exception("%s: gor_gate — растр %s не прочитан", name, p)
                done = False
                break
            if not ok:
                done = False
                break
            if len(F):
                ps.append(float(head_p(H, F).max()))
        if done and ps:
            region_p[name] = max(ps)
            log.info("%s: gor_gate p_scene %.4f (%d растров), прошло %.0f с", name, region_p[name], len(paths),
                     time.monotonic() - t0)
        else:
            log.warning("%s: gor_gate — участок пройден не полностью, не трогается", name)
    out = gate_confidences(feats, region_p)
    log.info("gor_gate: %d/%d участков с p_scene, %.0f с", len(region_p), len(targets), time.monotonic() - t0)
    return out
