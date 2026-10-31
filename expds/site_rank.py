"""Доучивание карты центров с порядком пиков на участке (PROMPT_WITHIN_SITE_RANK_20261031, 5fd33ce).

Не входит в сдачу и в default.toml: вызывается только ядром kaggle/2026-10-31/within_site_rank/.
Старт — готовый чекпойнт (models/spor_focal_r18.pt), та же сеть и классы. Потеря на шаге:
    focal_center_loss(тайлы кэша обучающих семей) + site_rank_loss(один участок обучающей семьи), коэффициент 1.
Пары для site_rank_loss: центры эталонов участка против ≤8 самых громких пиков участка, которые не попадают ни
в один эталон своего класса. Пики ищутся тем же путём, что у инференса (растр на рабочем шаге, скользящее окно
Segmenter, accel.find_peaks, 12 м, пол 0,05), заново в начале каждого раунда. Пилотная семья не входит ни в тайлы,
ни в пары.
"""
from __future__ import annotations

import json
import logging
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
KURGANS = ("kurgany_povrezhdennye", "kurgany_tselye")


def load_site_rasters(root, gt_path, seg, families_out: set[str], log_every: int = 20):
    """Растры sp_or обучающих участков на рабочем шаге модели (uint8 C×H×W) и эталоны курганов в пикселях."""
    import rasterio
    import shapely
    from rasterio.enums import Resampling
    from shapely.geometry import shape

    from .catalog import read_utm_json, scan_root
    from .crs import WEB_MERCATOR, to_crs, usable_raster_crs
    from .naming import site_family
    from .raster import open_metric
    from .tiles import MAX_UPSAMPLE, band_stats, fix_channels, to_uint8, working_res

    gt = defaultdict(list)
    for f in json.load(open(gt_path, encoding="utf-8"))["features"]:
        p = f["properties"]
        if p.get("markup_type") == "sp_or" and p.get("class_name") in seg.classes:
            gt[p["region_name"]].append((seg.classes.index(p["class_name"]), shape(f["geometry"])))
    sites = []
    for reg in scan_root(root):
        fam = site_family(reg.name)
        files = reg.rasters.get("sp_or") or []
        if not files or fam in families_out or not gt.get(reg.name):
            continue
        site = {"region": reg.name, "family": fam, "rasters": []}
        for path in files:
            stats = band_stats(str(path), "sp_or")
            with open_metric(path) as ds:
                native = abs(ds.res[0])
                r = working_res(native, seg.res, seg.meta.get("max_upsample") or MAX_UPSAMPLE)
                scale = r / native
                w, h = max(1, int(ds.width / scale)), max(1, int(ds.height / scale))
                n = min(ds.count, 3)
                crs = usable_raster_crs(ds.crs.to_string() if ds.crs else None, ds.bounds, read_utm_json(reg.utm_json))
                a = ds.read(list(range(1, n + 1)), out_shape=(n, h, w),
                            resampling=Resampling.average if scale > 1 else Resampling.bilinear)
                tr = ds.transform * rasterio.Affine.scale(ds.width / w, ds.height / h)
            img = fix_channels(to_uint8(a, "sp_or", stats), int(seg.meta["in_channels"]))
            inv = ~tr
            polys = []
            for c, g in gt[reg.name]:
                gg = to_crs(g, WEB_MERCATOR, crs) if crs else g
                gp = shapely.transform(gg, lambda xy: np.stack(inv * (xy[:, 0], xy[:, 1]), axis=1))  # (col,row)
                if not gp.intersects(shapely.box(0, 0, w, h)):
                    continue
                ctr = gp.centroid if gp.contains(gp.centroid) else gp.representative_point()
                polys.append((c, gp, (int(ctr.y), int(ctr.x))))
            if polys:
                site["rasters"].append({"file": Path(path).name, "img": np.ascontiguousarray(img), "polys": polys})
        if site["rasters"]:
            sites.append(site)
            if len(sites) % log_every == 0:
                log.info("rasters loaded: %d sites", len(sites))
    return sites


def mine_site(seg, site, min_dist_px: int = 12, floor: float = 0.05, top_fp: int = 8):
    """Пики участка текущей сетью: эталонные центры, лучшие пики внутри эталонов и ≤top_fp громких ложных."""
    import shapely

    from . import accel

    peaks = []  # (raster, row, col, cls, conf, inside_same_cls_gt, gt_key or None)
    gts = []
    for ri, ras in enumerate(site["rasters"]):
        probs = seg.predict(ras["img"])
        for c in range(probs.shape[0]):
            rows, cols, vals = accel.find_peaks(np.asarray(probs[c], np.float32), min_dist_px, floor)
            polys = [(gi, gp) for gi, (cc, gp, _) in enumerate(ras["polys"]) if cc == c]
            inside_of = np.full(len(rows), -1)
            for gi, gp in polys:
                hit = shapely.contains_xy(gp, cols + 0.5, rows + 0.5)
                inside_of[hit & (inside_of < 0)] = gi
            for r_, c_, v, gi in zip(rows, cols, vals, inside_of):
                peaks.append((ri, int(r_), int(c_), c, float(v), gi >= 0))
        for gi, (c, gp, ctr) in enumerate(ras["polys"]):
            gts.append((ri, ctr[0], ctr[1], c))
    fps = sorted([p for p in peaks if not p[5]], key=lambda p: -p[4])[:top_fp]
    return {"gts": gts, "fps": fps, "n_peaks": len(peaks)}


def _crop(img, r, c, t, rng, jitter):
    """Окно t×t с точкой (r, c) внутри (случайный сдвиг ≤ jitter), нули за краем. -> (окно, r_in, c_in)."""
    oy = int(rng.integers(-jitter, jitter + 1)) if jitter else 0
    ox = int(rng.integers(-jitter, jitter + 1)) if jitter else 0
    y0, x0 = r - t // 2 + oy, c - t // 2 + ox
    C, H, W = img.shape
    out = np.zeros((C, t, t), np.uint8)
    ys, xs = max(0, y0), max(0, x0)
    ye, xe = min(H, y0 + t), min(W, x0 + t)
    if ye > ys and xe > xs:
        out[:, ys - y0:ye - y0, xs - x0:xe - x0] = img[:, ys:ye, xs:xe]
    return out, r - y0, c - x0


def finetune(init: str, cache: str, root: str, gt_path: str, out: str, pilot_families: list[str],
             rounds: int = 6, steps_per_round: int = 200, batch: int = 32, lr: float = 1e-4,
             max_gt_per_site: int = 24, top_fp: int = 8, deadline_s: float = 5.0 * 3600, seed: int = 0,
             report_path: str | None = None) -> dict:
    import torch

    from .centers import focal_center_loss, gaussians_from_binary, site_rank_loss
    from .infer_nn import Segmenter
    from .train import GpuTiles, load_index

    t_start = time.time()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg = Segmenter(Path(init), device=str(dev))
    ck = torch.load(init, map_location="cpu", weights_only=False)
    model = seg.model
    k = len(seg.classes)
    cache_p = Path(cache)
    meta = json.loads((cache_p / "meta.json").read_text(encoding="utf-8"))
    assert list(meta["classes"])[:k] == list(seg.classes), (meta["classes"], seg.classes)
    rows = [r for r in load_index(cache_p) if r["family"] not in set(pilot_families)]
    sigma_m = float(ck.get("center_sigma_m") or 0.0)

    def pre(ym):
        return gaussians_from_binary(ym[:, :k], sigma_m / float(meta["res"]), by_radius=sigma_m <= 0, spike=True)

    tiles = GpuTiles(cache_p, rows, k, bool(meta.get("flips", False)), dev, precompute_target=None)
    log.info("tiles (pilot out): %d; pilot families %s", len(tiles), pilot_families)
    sites = load_site_rasters(root, gt_path, seg, set(pilot_families))
    log.info("train sites with GT: %d, rasters %d (%.0f s)", len(sites), sum(len(s["rasters"]) for s in sites),
             time.time() - t_start)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total = rounds * steps_per_round
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    t = int(seg.tile)
    hist = []
    step = 0
    for rd in range(rounds):
        t0 = time.time()
        model.eval()
        mined = [mine_site(seg, s, top_fp=top_fp) for s in sites]
        # Диагностика обучающих семей: медиана громких ложных над лучшим пиком внутри эталона (тихие эталоны).
        rec = {"round": rd, "mine_s": round(time.time() - t0, 1),
               "sites_with_pairs": sum(1 for m in mined if m["fps"] and m["gts"]),
               "fp_conf_median": float(np.median([p[4] for m in mined for p in m["fps"]])) if any(m["fps"] for m in mined) else None}
        pair_sites = [i for i, m in enumerate(mined) if m["gts"] and any(f[3] == g[3] for f in m["fps"] for g in m["gts"])]
        if not pair_sites:
            raise RuntimeError("no train site with GT/false-peak pairs of the same class")
        if (time.time() - t_start) > deadline_s:
            log.warning("deadline before round %d training", rd); hist.append(rec); break
        model.train()
        lf_sum = lr_sum = 0.0
        n_st = 0
        for _ in range(steps_per_round):
            idx = torch.randint(0, len(tiles), (batch,), device=dev)
            x, ym = tiles.get(idx, augment=True)
            y = pre(ym)
            si = pair_sites[int(rng.integers(len(pair_sites)))]
            m, s = mined[si], sites[si]
            fps = m["fps"]
            fp_cls = {f[3] for f in fps}
            gts = [g for g in m["gts"] if g[3] in fp_cls]
            if len(gts) > max_gt_per_site:
                gts = [gts[j] for j in rng.choice(len(gts), max_gt_per_site, replace=False)]
            pts = [(g[0], g[1], g[2], g[3]) for g in gts] + [(f[0], f[1], f[2], f[3]) for f in fps]
            crops, where = [], []
            for ri, r_, c_, cl in pts:
                cr, ry, cx = _crop(s["rasters"][ri]["img"], r_, c_, t, rng, t // 4)
                crops.append(cr); where.append((ry, cx, cl))
            xc = torch.from_numpy(np.stack(crops)).to(dev).float().div_(255.0)
            # одно и то же отражение для всей пачки участка (D4), координаты пересчитываются
            ops = rng.integers(0, 2, 3)
            wh = np.array(where)
            if ops[0]: xc = xc.flip(-1); wh[:, 1] = t - 1 - wh[:, 1]
            if ops[1]: xc = xc.flip(-2); wh[:, 0] = t - 1 - wh[:, 0]
            if ops[2]: xc = xc.transpose(-1, -2); wh[:, [0, 1]] = wh[:, [1, 0]]
            xc = xc.sub(0.5).div(0.25)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                logits = model(torch.cat([x, xc], 0))
            lt, lc = logits[:batch], logits[batch:].float()
            lf = focal_center_loss(lt, y)
            ii = torch.arange(len(pts), device=dev)
            vals = lc[ii, torch.as_tensor(wh[:, 2], device=dev), torch.as_tensor(wh[:, 0], device=dev),
                      torch.as_tensor(wh[:, 1], device=dev)]
            ng = len(gts)
            cls_t = torch.as_tensor(wh[:, 2], device=dev)
            lr_ = site_rank_loss(vals[:ng], cls_t[:ng], vals[ng:], cls_t[ng:])
            loss = lf + lr_
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            lf_sum += float(lf.detach()); lr_sum += float(lr_.detach()); n_st += 1
            if (time.time() - t_start) > deadline_s:
                log.warning("deadline inside round %d", rd); break
        rec.update({"steps": n_st, "focal": round(lf_sum / max(n_st, 1), 4), "rank": round(lr_sum / max(n_st, 1), 4),
                    "round_s": round(time.time() - t0, 1), "lr": float(sched.get_last_lr()[0])})
        hist.append(rec)
        log.info(json.dumps(rec, ensure_ascii=False))
        model.eval()
        _save(ck, model, out, {"round": rd + 1, "steps": step, "pilot_families": pilot_families, "history": hist})
        if report_path:
            Path(report_path).write_text(json.dumps(hist, ensure_ascii=False, indent=1), encoding="utf-8")
        if (time.time() - t_start) > deadline_s:
            break
    return {"history": hist, "steps": step, "sites": len(sites), "tiles": len(tiles), "out": out,
            "seconds": round(time.time() - t_start)}


def _save(ck: dict, model, out: str, info: dict) -> None:
    import torch

    d = {kk: v for kk, v in ck.items() if kk != "state_dict"}
    d["state_dict"] = {kk: v.detach().float().cpu() if v.is_floating_point() else v.detach().cpu()
                       for kk, v in model.state_dict().items()}
    d["site_rank"] = info
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(d, out)
