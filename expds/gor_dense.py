"""Плотный локализатор городищ DINOv2-S на спутниковых снимках. По умолчанию выключен.

Тот же стандартный выпуск facebook/dinov2-small, что у шлюза: готовые веса Apache-2.0,
голова своя и учится на снимках конкурса. Изображения LVD-142M в обучение не входят.

Вариант PFnoD (REPORT14_gor_dense.md, work14/pkg_gor2.py; PACKAGE-ONLY +0,015406, P 1,000):
F|all+cp:add — точки-кандидаты городищ из плотных патч-токенов замороженного facebook/dinov2-small, плюс
выключенный класс gorodishcha в выдаче sp_dfine («noD»). Гейт-умножение sp_or выключено (при R целей нет);
p участка головы гейта gor_v3 входит только в уверенность точки: conf = 0,5·(p_scene + пик).

Порт без изменения арифметики (scripts/gor_dense/dense_feats.py, dense_probe.Probe.peaks, gor_dense_eval rows):
* сцена целиком → шаги GSDS (2 и 4 м/пкс, усреднение), растяжка 2–98 % по обзору ≤1024 пкс, ImageNet-норм.;
* тайлы 518 пкс шаг 392, токены 14 пкс усредняются по перекрытиям; вторая шкала — билинейно в центрах токенов первой;
  признак = [токен 2 м, токен 4 м] = 768 (float16, как в обучении);
* голова gor_dense_head.npz: p = σ([1, ((F−mu)/sd)·W/qsd]·w) (PCA-64 → L2-логрег, вариант 'all', трудные негативы);
* гаусс σ = 1 токен, нормированный по валидным токенам (vf ≥ 0,5); приор центра кадра sm·exp(−½(d/s)²),
  d — до середины охвата токенов файла (UTM участка); top-200 на файл, K = 2 на участок с подавлением 150 м;
* точка → круг 10 м (как circle_pairs оценки, quad_segs 32) строится в UTM участка (UTM.json) и повершинно
  переводится в EPSG:3857 (все источники пайплайна в 3857; метрика переводит 3857 → UTM обратно).
Объекты: class gorodishcha, markup_type ``gor_dense`` (своя группа merge_within_markup), threshold_group ``gor_dense``
(порог [thresholds_by_markup.gor_dense] gorodishcha, применяется в watchdog.finish до merge), в файл пишется
markup_type ``sp_or`` (out_markup_type: в списке ТР; writer.to_feature_collection).
noD: на участках, где точки посчитаны, объекты sp_dfine класса из drop_dfine_classes удаляются (= τ sp_dfine
gorodishcha = inf). Участок, не пройденный целиком (время/ошибка), не трогается вовсе (выдача как v3).
"""
from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import numpy as np

from . import gor_gate

log = logging.getLogger(__name__)

GROUP = "gor_dense"
MARKUP = "gor_dense"
OUT_MARKUP = "sp_or"
CLASS = "gorodishcha"
SRC_MARKUP = "sp_or"            # растры SpOr
DFINE_GROUP = "sp_dfine"
UNAVAILABLE = "GOR_DENSE_UNAVAILABLE"
DEFAULTS = {
    "enabled": False,
    "backbone": "models/dinov2-small",                  # тот же снапшот, что у gor_gate (ed25f3a3…)
    "backbone_md5": "ea251be6e151ba8609ad0bc9b2e13a78",
    "gate_head": "models/gor_gate/gor_v3_head.npz",     # p_scene: голова гейта gor_v3, без умножения sp_or
    "gate_head_md5": "acb95c52956d712a16c3dd389237a691",
    "head": "models/gor_dense/gor_dense_head.npz",      # плотная голова 'all' (scripts/gor_dense/export_head.py)
    "head_md5": "",
    "centre_s": None,        # s приора центра, м — ОБЯЗАТЕЛЕН (у одного значения нет; см. открытые вопросы)
    "device": "auto",
    "batch": 16,             # виньетки p_scene
    "gsds": [2.0, 4.0],
    "tile": 518,
    "stride": 392,
    "sigma": 1.0,
    "vf_min": 0.5,
    "top_per_file": 200,
    "k": 2,
    "suppress_m": 150.0,
    "circle_r_m": 10.0,
    "circle_quad_segs": 32,
    "drop_dfine_classes": ["gorodishcha"],   # noD
    "soft_need_s": 30.0,
}
MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]


def settings(cfg: dict) -> dict:
    return dict(DEFAULTS, **(cfg.get("gor_dense") or {}))


def enabled(cfg: dict) -> bool:
    return bool((cfg.get("gor_dense") or {}).get("enabled", False))


def missing_artifacts(cfg: dict, root: Path) -> list[str]:
    st = settings(cfg)
    p = gor_gate._p
    miss = []
    for key, rel in (("backbone", f"{st['backbone']}/config.json"), ("backbone", f"{st['backbone']}/model.safetensors"),
                     ("gate_head", st["gate_head"]), ("head", st["head"])):
        if not p(root, rel).exists():
            miss.append(f"{key}={p(root, rel)}")
    s = st.get("centre_s")
    if s is None or not (float(s) > 0):
        miss.append("[gor_dense] centre_s")
    if CLASS not in ((cfg.get("thresholds_by_markup") or {}).get(GROUP) or {}):
        miss.append(f"[thresholds_by_markup.{GROUP}] {CLASS}")
    return miss


# ---------------------------------------------------------------- плотные токены (порт dense_feats.py)
def _run_tiles(model, x, C, tile, stride, device, half, keep_going):
    import torch

    H, W = x.shape[1:]
    Ht, Wt = H // 14, W // 14
    acc = np.zeros((Ht, Wt, C), np.float32)
    cnt = np.zeros((Ht, Wt), np.float32)
    th, tw = min(tile, Ht * 14), min(tile, Wt * 14)
    ys = list(range(0, max(Ht * 14 - th, 0) + 1, stride))
    xs = list(range(0, max(Wt * 14 - tw, 0) + 1, stride))
    if ys[-1] != Ht * 14 - th:
        ys.append(Ht * 14 - th)
    if xs[-1] != Wt * 14 - tw:
        xs.append(Wt * 14 - tw)
    for y0 in ys:
        for x0 in xs:
            if not keep_going():
                return None, 0
            t = torch.from_numpy(np.ascontiguousarray(x[:, y0:y0 + th, x0:x0 + tw]))[None].to(device)
            if half:
                t = t.half()
            with torch.no_grad():
                o = model(pixel_values=t).last_hidden_state[0, 1:].float().cpu().numpy().reshape(th // 14, tw // 14, C)
            acc[y0 // 14:(y0 + th) // 14, x0 // 14:(x0 + tw) // 14] += o
            cnt[y0 // 14:(y0 + th) // 14, x0 // 14:(x0 + tw) // 14] += 1
    return acc / np.maximum(cnt, 1)[..., None], len(ys) * len(xs)


def _bilinear(Fm, r, c):
    h, w = Fm.shape[:2]
    r = np.clip(r, 0, h - 1)
    c = np.clip(c, 0, w - 1)
    r0 = np.floor(r).astype(int)
    c0 = np.floor(c).astype(int)
    r1 = np.minimum(r0 + 1, h - 1)
    c1 = np.minimum(c0 + 1, w - 1)
    a = (r - r0)[:, None]
    b = (c - c0)[:, None]
    return Fm[r0, c0] * (1 - a) * (1 - b) + Fm[r0, c1] * (1 - a) * b + Fm[r1, c0] * a * (1 - b) + Fm[r1, c1] * a * b


def dense_tokens(model, path: Path, utm_epsg: int, device, gsds=(2.0, 4.0), tile=518, stride=392,
                 keep_going=lambda: True) -> dict | None:
    """Токены файла на сетке первой шкалы: F (n, C·len(gsds)) float16, row, col, ux, uy (UTM участка), vf, H, W.
    None — прервано (время)."""
    import rasterio
    from pyproj import Transformer
    from rasterio.enums import Resampling

    C = int(model.config.hidden_size)
    half = getattr(device, "type", str(device)) == "cuda"
    ntiles = 0
    with rasterio.open(path) as d:
        tr = Transformer.from_crs(d.crs, utm_epsg, always_xy=True)
        cx0, cy0 = (d.bounds.left + d.bounds.right) / 2, (d.bounds.bottom + d.bounds.top) / 2
        xa, ya = tr.transform([cx0, cx0 + d.transform.a * 100], [cy0, cy0])
        gsd = math.hypot(xa[1] - xa[0], ya[1] - ya[0]) / 100
        sc = max(1, int(max(d.width, d.height) / 1024))
        small = d.read(out_shape=(d.count, max(1, d.height // sc), max(1, d.width // sc)), resampling=Resampling.average)
        alpha = d.count >= 4 and d.colorinterp[3].name == "alpha"
        nod = d.nodata if d.nodata is not None else -1e9
        vs = small[3] > 0 if alpha else np.all(small[:3] != nod, 0)
        st = []
        for b in range(3):
            v = small[b][vs].astype(np.float32)
            st.append(np.percentile(v, [2, 98]) if v.size > 100 else (0, 255))
        del small
        maps = []
        for g in gsds:
            fct = g / gsd
            Hh, Ww = max(14, int(round(d.height / fct))), max(14, int(round(d.width / fct)))
            Hh, Ww = Hh // 14 * 14, Ww // 14 * 14
            arr = d.read(out_shape=(d.count, Hh, Ww), resampling=Resampling.average)
            val = arr[3] > 127 if alpha else (np.all(arr[:3] != nod, 0) & np.any(arr[:3] != 0, 0))
            x = np.zeros((3, Hh, Ww), np.float32)
            for b_ in range(3):
                x[b_] = np.clip((arr[b_] - st[b_][0]) / max(st[b_][1] - st[b_][0], 1e-6), 0, 1)
            x = ((x * val) - MEAN) / STD
            del arr
            Fm, nt = _run_tiles(model, x, C, int(tile), int(stride), device, half, keep_going)
            if Fm is None:
                return None
            ntiles += nt
            vfrac = val[:Hh // 14 * 14, :Ww // 14 * 14].reshape(Hh // 14, 14, Ww // 14, 14).mean((1, 3))
            maps.append((Fm, vfrac, d.height / Hh, d.width / Ww))
            del x, val
        F0, v0, sy0, sx0 = maps[0]
        h0, w0 = F0.shape[:2]
        rr, cc = np.meshgrid(np.arange(h0), np.arange(w0), indexing="ij")
        rr, cc = rr.ravel(), cc.ravel()
        py = (rr + 0.5) * 14 * sy0
        px = (cc + 0.5) * 14 * sx0
        parts = [F0[rr, cc]]
        for Fm, _vf, syk, sxk in maps[1:]:
            parts.append(_bilinear(Fm, py / (14 * syk) - 0.5, px / (14 * sxk) - 0.5))
        X0, Y0 = rasterio.transform.xy(d.transform, py, px, offset="ul")
        ux, uy = tr.transform(np.asarray(X0), np.asarray(Y0))
    return dict(F=np.concatenate(parts, 1).astype(np.float16), row=rr, col=cc, ux=np.asarray(ux, np.float64),
                uy=np.asarray(uy, np.float64), vf=v0[rr, cc].astype(np.float16).astype(np.float32), H=int(h0), W=int(w0),
                gsd=float(gsd), tiles=int(ntiles))


# ---------------------------------------------------------------- голова, сглаживание, пики (порт dense_probe)
def head_p(H: dict, F: np.ndarray) -> np.ndarray:
    """Как Probe.pred после set_proj: Q = ((F−mu)/sd)·W / qsd (float32), p = σ([1, Q]·w) (float64)."""
    F = np.asarray(F).astype(np.float32)
    assert F.shape[1] == int(H["feat_dim"]), f"признаки {F.shape[1]} ≠ голове {int(H['feat_dim'])}"
    Q = ((F - H["mu"].astype(np.float32)) / H["sd"].astype(np.float32)) @ H["W"].astype(np.float32)
    Q = Q / H["qsd"].astype(np.float32)
    Q = np.c_[np.ones(len(Q), np.float32), Q].astype(np.float64)
    return 1 / (1 + np.exp(-Q @ np.asarray(H["w"], np.float64)))


def file_candidates(p_tok, tok: dict, cp: float, sigma=1.0, vf_min=0.5, top=200) -> list[tuple]:
    """Probe.peaks для одного файла: [(score, ux, uy)] — top по сглаженной карте с приором центра."""
    from scipy.ndimage import gaussian_filter

    H, W = tok["H"], tok["W"]
    row, col, ux, uy = tok["row"], tok["col"], tok["ux"], tok["uy"]
    n = len(row)
    if not n:
        return []
    cx = (ux.min() + ux.max()) / 2
    cy = (uy.min() + uy.max()) / 2
    dcen = np.hypot(ux - cx, uy - cy)
    pm = np.zeros((H, W))
    vm = np.zeros((H, W))
    km = np.full((H, W), -1)
    pm[row, col] = p_tok
    vm[row, col] = tok["vf"] >= vf_min
    km[row, col] = np.arange(n)
    num = gaussian_filter(pm * vm, sigma, mode="constant")
    den = gaussian_filter(vm, sigma, mode="constant")
    sm = np.where(vm > 0, num / np.maximum(den, 1e-6), -1.0)
    if np.isfinite(cp):
        dm = np.zeros((H, W))
        dm[row, col] = dcen
        sm = np.where(sm >= 0, sm * np.exp(-0.5 * (dm / cp) ** 2), sm)
    out = []
    for q in np.argsort(-sm.ravel())[:top]:
        a, b = divmod(int(q), W)
        k = km[a, b]
        if k >= 0 and sm[a, b] >= 0:
            out.append((float(sm[a, b]), float(ux[k]), float(uy[k])))
    return out


def region_peaks(cands: list[tuple], k=2, suppress_m=150.0) -> list[tuple]:
    cand = sorted(cands, key=lambda t: -t[0])
    out: list[tuple] = []
    for c in cand:
        if all(math.hypot(c[1] - o[1], c[2] - o[2]) > suppress_m for o in out):
            out.append(c)
        if len(out) == k:
            break
    return out


def circle_3857(x_utm: float, y_utm: float, utm_epsg: int, r_m: float = 10.0, quad_segs: int = 32):
    """Круг r_m (м) вокруг точки UTM участка → Polygon EPSG:3857 (повершинно, без приближения cos φ)."""
    from pyproj import Transformer
    from shapely.geometry import Point, Polygon

    ring = np.asarray(Point(x_utm, y_utm).buffer(r_m, quad_segs=quad_segs).exterior.coords)
    tr = Transformer.from_crs(int(utm_epsg), 3857, always_xy=True)
    gx, gy = tr.transform(ring[:, 0], ring[:, 1])
    return Polygon(np.column_stack([gx, gy]))


def point_features(region: str, peaks: list[tuple], p_scene: float, utm_epsg: int, st: dict, source: str = "") -> list[dict]:
    out = []
    for k, (ps, x, y) in enumerate(peaks):
        out.append({"geom": circle_3857(x, y, utm_epsg, float(st["circle_r_m"]), int(st["circle_quad_segs"])),
                    "class_name": CLASS, "confidence": float(0.5 * (float(p_scene) + float(ps))),
                    "region_name": region, "markup_type": MARKUP, "threshold_group": GROUP,
                    "out_markup_type": OUT_MARKUP, "source": f"{source}#gor_dense{k}",
                    "original_crs": f"EPSG:{int(utm_epsg)}",
                    "gor_dense_peak": float(ps), "gor_dense_p_scene": float(p_scene)})
    return out


def is_dfine_dropped(f: dict, classes) -> bool:
    return (f.get("threshold_group") == DFINE_GROUP or str(f.get("threshold_group") or "").startswith(DFINE_GROUP + "_")) \
        and f.get("class_name") in classes


def apply_points(feats: list[dict], region_points: dict[str, list[dict]], drop_classes) -> list[dict]:
    """Точки участков + noD: на участках из region_points объекты sp_dfine классов drop_classes удаляются."""
    drop = set(drop_classes or ())
    regs = set(region_points)
    out = [f for f in feats if not (f.get("region_name") in regs and is_dfine_dropped(f, drop))]
    for r in sorted(region_points):
        out.extend(region_points[r])
    return out


# ---------------------------------------------------------------- фаза пайплайна
def load_dense_head(path: Path, md5: str | None) -> dict:
    if md5 and gor_gate._md5(path) != md5:
        raise RuntimeError(f"gor_dense: md5 головы {path} не {md5}")
    H = dict(np.load(path))
    for k in ("mu", "sd", "W", "qsd", "w", "feat_dim"):
        if k not in H:
            raise RuntimeError(f"gor_dense: в голове нет {k}")
    return H


def region_points(model, H, G, reg, st, dev, keep, soft_ok=None, timing: dict | None = None) -> tuple[list[dict], bool]:
    """(точки участка, пройден целиком). timing[file] = секунды (плотные токены + p_scene)."""
    name = reg.name
    paths = list((getattr(reg, "rasters", {}) or {}).get(SRC_MARKUP, []))
    epsg = gor_gate._epsg(reg)
    if not paths or epsg is None:
        return [], False
    cands, ps = [], []
    for p in paths:
        if not keep() or (soft_ok is not None and not soft_ok(f"gor_dense {name}", float(st["soft_need_s"]))):
            return [], False
        t0 = time.monotonic()
        Fg, _n, ok = gor_gate.tif_feats(model, Path(p), epsg, dev, int(st["batch"]), keep)
        if not ok:
            return [], False
        if len(Fg):
            ps.append(float(gor_gate.head_p(G, Fg).max()))
        tok = dense_tokens(model, Path(p), epsg, dev, st["gsds"], st["tile"], st["stride"], keep)
        if tok is None:
            return [], False
        pt = head_p(H, tok["F"])
        cands += file_candidates(pt, tok, float(st["centre_s"]), float(st["sigma"]), float(st["vf_min"]),
                                 int(st["top_per_file"]))
        if timing is not None:
            timing[str(p)] = dict(s=time.monotonic() - t0, tiles=tok["tiles"], vignettes=int(_n), tokens=len(tok["row"]))
    pk = region_peaks(cands, int(st["k"]), float(st["suppress_m"]))
    p_scene = max(ps) if ps else 0.0       # как gp_.get(r, 0.0)
    return point_features(name, pk, p_scene, epsg, st, Path(paths[0]).name), True


def apply(feats: list[dict], regions: dict, cfg: dict, root: Path, guard=None, soft_ok=None) -> tuple[list[dict], int]:
    """При enabled и полном наборе артефактов: точки gor_dense по участкам с SpOr + noD на этих участках.
    Нет артефактов / ошибка / нехватка времени — объекты (участка) без изменений и строка в журнале."""
    if not enabled(cfg):
        return feats, 0
    miss = missing_artifacts(cfg, root)
    if miss:
        log.error("%s: нет артефактов/параметров (%s) — точки городищ не добавлены, noD не применён", UNAVAILABLE,
                  "; ".join(miss))
        return feats, 0
    st = settings(cfg)
    targets = sorted(n for n, r in regions.items() if (getattr(r, "rasters", {}) or {}).get(SRC_MARKUP))
    if not targets:
        log.info("gor_dense: участков с SpOr нет")
        return feats, 0
    t0 = time.monotonic()
    try:
        G = gor_gate.load_head(gor_gate._p(root, st["gate_head"]), st.get("gate_head_md5"))
        H = load_dense_head(gor_gate._p(root, st["head"]), st.get("head_md5"))
        dev = gor_gate._device(st)
        model = gor_gate.load_backbone(st, root, dev)
        if getattr(dev, "type", "") == "cuda":
            model = model.half()
    except Exception:
        log.exception("%s: голова/бэкбон не загружены — точки не добавлены, noD не применён, выдача как v3", UNAVAILABLE)
        return feats, 0
    keep = (lambda: guard.ok(30)) if guard is not None else (lambda: True)
    done: dict[str, list[dict]] = {}
    for name in targets:
        try:
            pts, ok = region_points(model, H, G, regions[name], st, dev, keep, soft_ok)
        except Exception:
            log.exception("%s: %s: gor_dense — ошибка, участок не трогается", UNAVAILABLE, name)
            continue
        if ok:
            done[name] = pts
            log.info("%s: gor_dense %d точек (conf %s), прошло %.0f с", name, len(pts),
                     ", ".join(f"{f['confidence']:.3f}" for f in pts), time.monotonic() - t0)
        else:
            log.warning("%s: gor_dense — участок пройден не полностью (время/нет UTM.json), не трогается", name)
    out = apply_points(feats, done, st.get("drop_dfine_classes"))
    n_pts = sum(len(v) for v in done.values())
    log.info("gor_dense: %d/%d участков, %d точек, noD удалил %d объектов sp_dfine, %.0f с", len(done), len(targets),
             n_pts, len(feats) + n_pts - len(out), time.monotonic() - t0)
    return out, n_pts
