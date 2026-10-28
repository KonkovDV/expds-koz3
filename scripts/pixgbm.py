#!/usr/bin/env python3
"""Пиксельный LightGBM геофизики (expds/geo_pixgbm.py): выборка, обучение leave-family-out, прогноз.

    python scripts/pixgbm.py extract --root DATA --out WORK/feats
    python scripts/pixgbm.py train   --feats WORK/feats --models WORK/models      # по модели на каждую отложенную семью + all
    python scripts/pixgbm.py predict --root DATA --models WORK/models --out WORK/pix_lofo.geojson   # участок ← модель без его семьи
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from expds import geo_pixgbm as G, geophys, writer          # noqa: E402
from expds.catalog import scan_root                        # noqa: E402
from expds.gt import read_annotation                       # noqa: E402
from expds.naming import site_family                       # noqa: E402
from expds.pipeline import load_config, region_crs         # noqa: E402

GEO_REGIONS = tuple(str(n) for n in range(200, 218))


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def regions(root, only=None):
    for reg in scan_root(root):
        if reg.name[:3] in GEO_REGIONS and (not only or reg.name in only):
            yield reg


def sample(y_list, use, rng, n_pos=3000, n_hard=2000, n_rand=6000, gsd=0.1):
    idx = [np.flatnonzero(use.ravel())]
    pick = set()
    flat = use.ravel()
    chosen = []
    for y in y_list:
        if y is None:
            continue
        yf = y.ravel()
        pos = np.flatnonzero((yf == 1) & flat)
        if pos.size:
            chosen.append(rng.choice(pos, min(n_pos, pos.size), replace=False))
            near = __import__("scipy.ndimage", fromlist=["x"]).binary_dilation(y == 1, iterations=max(1, int(round(3.0 / gsd)))).ravel()
            hard = np.flatnonzero(near & (yf == 0) & flat)
            if hard.size:
                chosen.append(rng.choice(hard, min(n_hard, hard.size), replace=False))
    allv = idx[0]
    if allv.size:
        chosen.append(rng.choice(allv, min(n_rand, allv.size), replace=False))
    return np.unique(np.concatenate(chosen)) if chosen else np.zeros(0, int)


def cmd_extract(a):
    cfg = load_config(a.config)
    geo_cfg = cfg.get("geophys", {})
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    log = []
    for reg in regions(a.root, a.regions):
        crs_region = region_crs(reg, cfg)
        fam = site_family(reg.name)
        anns = {}
        for f, cls, mt in reg.annotations:
            anns.setdefault((mt, cls, G.label_key(f)), []).append(f)
        for mt in ("maggr", "etgr", "terfl"):
            files = G.pick_rasters(reg.rasters.get(mt, []))
            if not files:
                continue
            boundary = geophys.survey_boundary(reg.boundaries, mt, crs_region, geo_cfg)
            noise = geophys.load_mask(reg.noise, mt, crs_region)
            for f in files:
                t0 = time.time()
                key = G.raster_key(f)
                gt = {}
                for cls in ("object_poly", "finds_points"):
                    fs = anns.get((mt, cls, key))
                    if not fs and len(files) == 1:
                        # имя карты не совпало с разметкой (207: «уч7» vs «7»); одна карта метода —
                        # берём единственную разметку этого метода
                        cand = [v for (m2, c2, _k), v in anns.items() if m2 == mt and c2 == cls]
                        fs = cand[0] if len(cand) == 1 else None
                    gt[cls] = None if not fs else [o["geom"] for p in fs for o in read_annotation(p, cls, mt, reg)]
                if gt["object_poly"] is None and gt["finds_points"] is None:
                    log.append(dict(region=reg.name, raster=f.name, skip="нет разметки"))
                    print(reg.name, f.name, "нет разметки — пропуск", flush=True)
                    continue
                d = G.load_map(f, G.GSD[mt], crs_region, boundary, noise)
                if not d["crs"]:
                    log.append(dict(region=reg.name, raster=f.name, skip="нет CRS")); continue
                yp, yf = G.label_masks(d, gt["object_poly"], gt["finds_points"] if mt == "maggr" else None)
                use = d["valid"] & ~d["noise"]
                ids = sample([yp, yf], use, rng, gsd=d["gsd"])
                W = d["z"].shape[1]
                rows, cols = ids // W, ids % W
                names, X = G.features_for(d, mt, rows, cols)
                Yp = yp.ravel()[ids] if yp is not None else np.full(len(ids), -2, np.int8)
                Yf = yf.ravel()[ids] if yf is not None else np.full(len(ids), -2, np.int8)
                np.savez(out / f"{reg.name}__{f.stem}.npz", X=X, yp=Yp, yf=Yf, names=np.array(names),
                         family=fam, region=reg.name, mt=mt)
                rec = dict(region=reg.name, raster=f.name, mt=mt, gsd=d["gsd"], shape=list(d["z"].shape),
                           n=int(len(ids)), poly_pos=int((Yp == 1).sum()), finds_pos=int((Yf == 1).sum()),
                           n_gt_poly=None if gt["object_poly"] is None else len(gt["object_poly"]),
                           n_gt_finds=None if gt["finds_points"] is None else len(gt["finds_points"]),
                           sec=round(time.time() - t0, 1), maxrss_mb=round(rss_mb()))
                log.append(rec); print(json.dumps(rec, ensure_ascii=False), flush=True)
                del d, X, yp, yf, use
    (out / "extract_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1))


def load_feats(feats):
    parts = []
    for p in sorted(Path(feats).glob("*.npz")):
        z = np.load(p)
        parts.append(dict(X=z["X"], yp=z["yp"], yf=z["yf"], names=list(z["names"]), family=str(z["family"]),
                          region=str(z["region"]), mt=str(z["mt"])))
    return parts


def fit(parts, target, rounds):
    import lightgbm as lgb
    names = parts[0]["names"]
    X, y, w, fam = [], [], [], []
    for p in parts:
        assert p["names"] == names
        yy = p[target]
        m = yy >= 0
        if not m.any():
            continue
        X.append(p["X"][m]); y.append(yy[m]); fam += [p["family"]] * int(m.sum())
    X = np.concatenate(X); y = np.concatenate(y).astype(np.float32); fam = np.array(fam)
    w = np.ones(len(y), np.float32)
    fams = sorted(set(fam))
    for f in fams:                      # семьи поровну, внутри семьи классы поровну
        for c in (0, 1):
            m = (fam == f) & (y == c)
            if m.any():
                w[m] = 1.0 / (len(fams) * 2 * m.sum())
    w *= len(w) / w.sum()
    b = lgb.train(dict(G.LGBM, seed=0), lgb.Dataset(X, label=y, weight=w, feature_name=names), num_boost_round=rounds)
    return b, dict(n=int(len(y)), pos=int(y.sum()), families=fams)


def cmd_train(a):
    parts = load_feats(a.feats)
    fams = sorted({p["family"] for p in parts})
    out = Path(a.models)
    info = {}
    for held in fams + ["all"]:
        tr = [p for p in parts if p["family"] != held]
        (out / held).mkdir(parents=True, exist_ok=True)
        for target, name in (("yp", "poly"), ("yf", "finds")):
            sub = [p for p in tr if (p[target] >= 0).any()]
            if not sub:
                continue
            t0 = time.time()
            b, inf = fit(sub, target, a.rounds)
            b.save_model(str(out / held / f"{name}.txt"))
            inf.update(sec=round(time.time() - t0, 1), maxrss_mb=round(rss_mb()))
            imp = sorted(zip(b.feature_name(), b.feature_importance("gain")), key=lambda t: -t[1])[:12]
            inf["top_gain"] = [(k, round(float(v))) for k, v in imp]
            info[f"{held}/{name}"] = inf
            print(held, name, json.dumps(inf, ensure_ascii=False), flush=True)
    (out / "train_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))


def cmd_predict(a):
    import lightgbm as lgb
    cfg = load_config(a.config)
    geo_cfg = cfg.get("geophys", {})
    feats, stats = [], []
    for reg in regions(a.root, a.regions):
        held = "all" if a.all else site_family(reg.name)
        md = Path(a.models) / held
        bst = {k: lgb.Booster(model_file=str(md / f"{k}.txt")) for k in ("poly", "finds") if (md / f"{k}.txt").exists()}
        crs_region = region_crs(reg, cfg)
        for mt in ("maggr", "etgr", "terfl"):
            files = G.pick_rasters(reg.rasters.get(mt, []))
            if not files:
                continue
            boundary = geophys.survey_boundary(reg.boundaries, mt, crs_region, geo_cfg)
            noise = geophys.load_mask(reg.noise, mt, crs_region)
            use = {k: v for k, v in bst.items() if k == "poly" or mt == "maggr"}
            for f in files:
                t0 = time.time()
                d = G.load_map(f, G.GSD[mt], crs_region, boundary, noise)
                if not d["crs"]:
                    continue
                Pm = G.prob_map(d, mt, use)
                fp = G.poly_candidates(Pm["poly"], d, reg.name, mt, f.name, a.t_pix) if "poly" in Pm else []
                ff = G.finds_candidates(Pm["finds"], d, reg.name, mt, f.name) if "finds" in Pm else []
                feats += fp + ff
                rec = dict(region=reg.name, raster=f.name, mt=mt, model=held, poly=len(fp), finds=len(ff),
                           sec=round(time.time() - t0, 1), maxrss_mb=round(rss_mb()))
                stats.append(rec); print(json.dumps(rec, ensure_ascii=False), flush=True)
                del d, Pm
    fc = writer.to_feature_collection(feats)
    for feat, src in zip(fc["features"], feats):
        for key in ("threshold_group", "source"):
            feat["properties"][key] = src[key]
    Path(a.out).write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")
    Path(a.out).with_suffix(".stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for n in ("extract", "train", "predict"):
        p = sp.add_parser(n)
        p.add_argument("--config", default=str(ROOT / "configs" / "default.toml"))
        p.add_argument("--regions", nargs="*", default=None)
    sp.choices["extract"].add_argument("--root", required=True); sp.choices["extract"].add_argument("--out", required=True)
    sp.choices["train"].add_argument("--feats", required=True); sp.choices["train"].add_argument("--models", required=True)
    sp.choices["train"].add_argument("--rounds", type=int, default=G.ROUNDS)
    pr = sp.choices["predict"]
    pr.add_argument("--root", required=True); pr.add_argument("--models", required=True); pr.add_argument("--out", required=True)
    pr.add_argument("--t-pix", type=float, default=G.T_PIX); pr.add_argument("--all", action="store_true",
                    help="модель all (на всех семьях) — только для сдачи, не для оценки")
    a = ap.parse_args()
    {"extract": cmd_extract, "train": cmd_train, "predict": cmd_predict}[a.cmd](a)


if __name__ == "__main__":
    main()
