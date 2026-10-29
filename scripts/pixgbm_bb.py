#!/usr/bin/env python3
"""PIXGBM + замороженный бэкбон (expds/geo_backbone.py), bot-3. Те же пиксели выборки, что у scripts/pixgbm.py extract
(тот же rng), к 72 признакам добавляются PCA-32 признаков бэкбона (PCA внутри обучающих семей фолда).

    python scripts/pixgbm_bb.py extract --root DATA --feats PIXGBM/feats --out BB/bbfeats --variants B1,B2
    python scripts/pixgbm_bb.py train   --feats PIXGBM/feats --bbfeats BB/bbfeats --models BB/models --variant B1   (B0 — контроль без бэкбона)
    python scripts/pixgbm_bb.py predict --root DATA --models BB/models --variants B1,B2 --out BB/pix_lofo   (→ pix_lofo_B1.geojson …)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import pixgbm as P                                         # noqa: E402
from expds import geo_backbone as BB, geo_pixgbm as G, geophys, writer   # noqa: E402
from expds.gt import read_annotation                       # noqa: E402
from expds.naming import site_family                       # noqa: E402
from expds.pipeline import load_config, region_crs         # noqa: E402

K_PCA = 32
NT = int(os.environ.get("OMP_NUM_THREADS", "2") or 2)


def cmd_extract(a):
    cfg = load_config(a.config); geo_cfg = cfg.get("geophys", {})
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)                         # дословно как pixgbm.py extract
    vs = a.variants.split(",")
    bks = {} if a.dry else {v: BB.Backbone(v, NT) for v in vs}
    log, k = [], 0
    fresh = tuple(x for x in a.fresh_rng.split(",") if x)
    for reg in P.regions(a.root, a.regions):
        # pixgbm/feats: 207 в общем прогоне пропущен (rng не тронут), потом извлечён отдельно `--regions 207…` со своим rng(0)
        rg = np.random.default_rng(0) if reg.name.startswith(fresh) else rng
        crs_region = region_crs(reg, cfg)
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
                        cand = [v for (m2, c2, _k), v in anns.items() if m2 == mt and c2 == cls]
                        fs = cand[0] if len(cand) == 1 else None
                    gt[cls] = None if not fs else [o["geom"] for p in fs for o in read_annotation(p, cls, mt, reg)]
                if gt["object_poly"] is None and gt["finds_points"] is None:
                    continue
                d = G.load_map(f, G.GSD[mt], crs_region, boundary, noise)
                if not d["crs"]:
                    continue
                yp, yf = G.label_masks(d, gt["object_poly"], gt["finds_points"] if mt == "maggr" else None)
                use = d["valid"] & ~d["noise"]
                ids = P.sample([yp, yf], use, rg, gsd=d["gsd"])
                W = d["z"].shape[1]
                rows, cols = ids // W, ids % W
                name = f"{reg.name}__{f.stem}.npz"
                ref = np.load(Path(a.feats) / name)
                Yp = yp.ravel()[ids] if yp is not None else np.full(len(ids), -2, np.int8)
                Yf = yf.ravel()[ids] if yf is not None else np.full(len(ids), -2, np.int8)
                ok = len(ids) == len(ref["yp"]) and (Yp == ref["yp"]).all() and (Yf == ref["yf"]).all()
                xok = None
                if k % a.verify_every == 0:
                    _, X = G.features_for(d, mt, rows, cols)
                    xok = bool(np.array_equal(X, ref["X"]))
                if not ok or xok is False:
                    raise SystemExit(f"{name}: выборка не совпала с pixgbm/feats (ok={ok}, X={xok})")
                if a.dry:
                    print(name, len(ids), "ok", xok, flush=True); k += 1; continue
                x, vb, fct = BB.bb_input(d)
                rec = dict(region=reg.name, raster=f.name, mt=mt, n=int(len(ids)), f=fct, bb_shape=list(x.shape[1:]),
                           x_verified=xok, sec_load=round(time.time() - t0, 1))
                arrs = {}
                for v, bk in bks.items():
                    t1 = time.time()
                    Fm = BB.bb_map(x, vb, bk)
                    arrs[v] = BB.sample(Fm, rows, cols, fct, bk.spec["stride"]).astype(np.float16)
                    rec[f"sec_{v}"] = round(time.time() - t1, 1)
                    del Fm
                np.savez(out / name, **arrs)
                rec["maxrss_mb"] = round(P.rss_mb()); log.append(rec); k += 1
                print(json.dumps(rec, ensure_ascii=False), flush=True)
                del d, x, vb, yp, yf, use
    (out / "extract_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1))


def bbload(p):
    fn, v = p["B"]
    return np.load(fn)[v]


def fit_pca(parts, v, k=K_PCA):
    s = ss = None; n = 0
    for p in parts:
        B = bbload(p).astype(np.float64)
        if s is None:
            s = np.zeros(B.shape[1]); ss = np.zeros((B.shape[1], B.shape[1]))
        s += B.sum(0); ss += B.T @ B; n += len(B)
    mean = s / n
    cov = ss / n - np.outer(mean, mean)
    w, U = np.linalg.eigh(cov)
    o = np.argsort(-w)[:k]
    return mean.astype(np.float32), U[:, o].astype(np.float32), (w[o].sum() / max(w.sum(), 1e-12))


def fit_bb(parts, target, rounds, mean, comps):
    """= pixgbm.fit (тот же порядок строк, веса, параметры), но X собирается в заранее выделенный массив
    [72 признака | PCA бэкбона] без промежуточных копий (RAM ≤ 2 ГБ)."""
    import lightgbm as lgb
    names = parts[0]["names"] + [f"bb{i:02d}" for i in range(comps.shape[1])]
    k0 = len(parts[0]["names"])
    n = sum(int((p[target] >= 0).sum()) for p in parts)
    X = np.empty((n, len(names)), np.float32); y = np.empty(n, np.float32); fam = []
    i = 0
    for p in parts:
        assert p["names"] == parts[0]["names"]
        m = p[target] >= 0
        k = int(m.sum())
        if not k:
            continue
        X[i:i + k, :k0] = p["X"][m]
        X[i:i + k, k0:] = (bbload(p)[m].astype(np.float32) - mean) @ comps
        y[i:i + k] = p[target][m]; fam += [p["family"]] * k; i += k
    fam = np.array(fam)
    w = np.ones(len(y), np.float32)
    fams = sorted(set(fam))
    for f in fams:
        for c in (0, 1):
            mm = (fam == f) & (y == c)
            if mm.any():
                w[mm] = 1.0 / (len(fams) * 2 * mm.sum())
    w *= len(w) / w.sum()
    ds = lgb.Dataset(X, label=y, weight=w, feature_name=names, free_raw_data=True)
    b = lgb.train(dict(G.LGBM, seed=0), ds, num_boost_round=rounds)
    return b, dict(n=int(len(y)), pos=int(y.sum()), families=fams)


def cmd_train(a):
    parts = P.load_feats(a.feats)
    v = a.variant
    if v != "B0":
        for p, fn in zip(parts, sorted(Path(a.feats).glob("*.npz"))):
            p["B"] = (Path(a.bbfeats) / fn.name, v)          # читается с диска по фолдам (RAM ≤ 2 ГБ)
    fams = sorted({p["family"] for p in parts})
    out = Path(a.models) / v
    info = {}
    for held in fams + ["all"]:
        tr = [p for p in parts if p["family"] != held]
        (out / held).mkdir(parents=True, exist_ok=True)
        pca = None
        if v != "B0":
            mean, comps, ev = fit_pca(tr, v)
            np.savez(out / held / "pca.npz", mean=mean, comps=comps)
            pca = (mean, comps)
            info[f"{held}/pca_explained"] = float(ev)
        for target, name in (("yp", "poly"), ("yf", "finds")):
            sub = [p for p in tr if (p[target] >= 0).any()]
            if not sub:
                continue
            t0 = time.time()
            b, inf = P.fit(sub, target, a.rounds) if pca is None else fit_bb(sub, target, a.rounds, *pca)
            b.save_model(str(out / held / f"{name}.txt"))
            inf.update(sec=round(time.time() - t0, 1), maxrss_mb=round(P.rss_mb()))
            gi = dict(zip(b.feature_name(), b.feature_importance("gain")))
            tot = sum(gi.values())
            inf["bb_gain_share"] = round(sum(x for kk, x in gi.items() if kk.startswith("bb")) / tot, 4) if tot else 0
            inf["top_gain"] = [(kk, round(float(x))) for kk, x in sorted(gi.items(), key=lambda t: -t[1])[:12]]
            info[f"{held}/{name}"] = inf
            print(held, name, json.dumps(inf, ensure_ascii=False), flush=True)
    (out / "train_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))


def prob_maps(d, mt, models, bbp, fct):
    """Признаки окон считаются один раз; models {вариант: {poly|finds: Booster}}, bbp {вариант: (PCA-карта, шаг)}."""
    c, b = G.trends(d["z"], d["valid"], d["gsd"])
    H, W = d["z"].shape
    use = d["valid"] & ~d["noise"]
    out = {v: {k: np.zeros((H, W), np.float32) for k in ms} for v, ms in models.items()}
    tm = {"feat": 0.0, "bbsample": 0.0, "pred": 0.0}
    for ty, tx, (y0, x0, y1, x1) in G.windows(H, W, d["gsd"]):
        cy1, cx1 = min(ty + G.CORE, H), min(tx + G.CORE, W)
        rr, cc = np.nonzero(use[ty:cy1, tx:cx1])
        if not rr.size:
            continue
        rr, cc = rr + ty - y0, cc + tx - x0
        h, ww = y1 - y0, x1 - x0
        t = time.perf_counter()
        F = G.features_at(d["z"][y0:y1, x0:x1], d["zd"][y0:y1, x0:x1], d["valid"][y0:y1, x0:x1],
                          G._up(c[0], b, y0, x0, h, ww), G._up(c[1], b, y0, x0, h, ww), d["gsd"], G.MT_ID[mt], rr, cc)
        tm["feat"] += time.perf_counter() - t
        for v, ms in models.items():
            for k, m in ms.items():
                names = m.feature_name()
                base = np.column_stack([F[n] for n in names if not n.startswith("bb")])
                if v in bbp:
                    t = time.perf_counter()
                    Fp, s = bbp[v]
                    Xb = BB.sample(Fp, rr + y0, cc + x0, fct, s)
                    tm["bbsample"] += time.perf_counter() - t
                    X = np.hstack([base, Xb])
                else:
                    X = base
                t = time.perf_counter()
                out[v][k][rr + y0, cc + x0] = m.predict(X, num_threads=NT).astype(np.float32)
                tm["pred"] += time.perf_counter() - t
        del F
    return out, tm


def cmd_predict(a):
    import lightgbm as lgb
    cfg = load_config(a.config); geo_cfg = cfg.get("geophys", {})
    vs = a.variants.split(",")
    bks = {v: BB.Backbone(v, NT) for v in vs if v not in ("B0", "S")}
    feats = {v: [] for v in vs}; stats = []
    for reg in P.regions(a.root, a.regions):
        held = "all" if a.all else site_family(reg.name)
        mods, pcas = {}, {}
        for v in vs:
            md = Path(a.models) / v / held if v != "S" else Path(a.s_models) / held
            mods[v] = {k: lgb.Booster(model_file=str(md / f"{k}.txt")) for k in ("poly", "finds") if (md / f"{k}.txt").exists()}
            if v in bks:
                z = np.load(md / "pca.npz"); pcas[v] = (z["mean"], z["comps"])
        crs_region = region_crs(reg, cfg)
        for mt in ("maggr", "etgr", "terfl"):
            files = G.pick_rasters(reg.rasters.get(mt, []))
            if not files:
                continue
            boundary = geophys.survey_boundary(reg.boundaries, mt, crs_region, geo_cfg)
            noise = geophys.load_mask(reg.noise, mt, crs_region)
            use = {v: {k: m for k, m in ms.items() if k == "poly" or mt == "maggr"} for v, ms in mods.items()}
            for f in files:
                t0 = time.time()
                d = G.load_map(f, G.GSD[mt], crs_region, boundary, noise)
                if not d["crs"]:
                    continue
                rec = dict(region=reg.name, raster=f.name, mt=mt, model=held, mpx=round(d["z"].size / 1e6, 3),
                           mpx_valid=round(float((d["valid"] & ~d["noise"]).sum()) / 1e6, 3), sec_load=round(time.time() - t0, 2))
                bbp, fct = {}, 1
                if bks:
                    t1 = time.time()
                    x, vb, fct = BB.bb_input(d)
                    rec["sec_bbinput"] = round(time.time() - t1, 2)
                    for v, bk in bks.items():
                        t1 = time.time()
                        Fm = BB.bb_map(x, vb, bk)
                        rec[f"sec_fwd_{v}"] = round(time.time() - t1, 2); t1 = time.time()
                        bbp[v] = (BB.project(Fm, *pcas[v]), bk.spec["stride"])
                        rec[f"sec_pca_{v}"] = round(time.time() - t1, 2)
                        del Fm
                    del x, vb
                Pm, tm = prob_maps(d, mt, use, bbp, fct)
                rec.update({f"sec_{k}": round(x, 2) for k, x in tm.items()})
                for v in vs:
                    fp = G.poly_candidates(Pm[v]["poly"], d, reg.name, mt, f.name, a.t_pix) if "poly" in Pm[v] else []
                    ff = G.finds_candidates(Pm[v]["finds"], d, reg.name, mt, f.name) if "finds" in Pm[v] else []
                    feats[v] += fp + ff
                    rec[f"n_{v}"] = [len(fp), len(ff)]
                rec.update(sec=round(time.time() - t0, 1), maxrss_mb=round(P.rss_mb()))
                stats.append(rec); print(json.dumps(rec, ensure_ascii=False), flush=True)
                del d, Pm, bbp
    for v in vs:
        fc = writer.to_feature_collection(feats[v])
        for feat, src in zip(fc["features"], feats[v]):
            for key in ("threshold_group", "source"):
                feat["properties"][key] = src[key]
        Path(f"{a.out}_{v}.geojson").write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")
    Path(f"{a.out}.stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for n in ("extract", "train", "predict"):
        p = sp.add_parser(n)
        p.add_argument("--config", default=str(ROOT / "configs" / "default.toml"))
        p.add_argument("--regions", nargs="*", default=None)
    e = sp.choices["extract"]
    e.add_argument("--root", required=True); e.add_argument("--feats", required=True); e.add_argument("--out", required=True)
    e.add_argument("--variants", default="B1,B2"); e.add_argument("--verify-every", type=int, default=10)
    e.add_argument("--fresh-rng", default="207_", help="участки со своим rng(0), как в pixgbm/extract207.log")
    e.add_argument("--dry", action="store_true", help="только проверка выборки, без бэкбона")
    t = sp.choices["train"]
    t.add_argument("--feats", required=True); t.add_argument("--bbfeats", default=""); t.add_argument("--models", required=True)
    t.add_argument("--variant", required=True); t.add_argument("--rounds", type=int, default=G.ROUNDS)
    pr = sp.choices["predict"]
    pr.add_argument("--root", required=True); pr.add_argument("--models", required=True); pr.add_argument("--out", required=True)
    pr.add_argument("--variants", default="B1,B2"); pr.add_argument("--s-models", default="/workspace/koz3_shared/bot-3/pixgbm/models")
    pr.add_argument("--t-pix", type=float, default=G.T_PIX); pr.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if a.cmd == "extract" and a.regions:
        raise SystemExit("extract: rng общий на все участки — --regions нельзя (выборка разойдётся)")
    {"extract": cmd_extract, "train": cmd_train, "predict": cmd_predict}[a.cmd](a)


if __name__ == "__main__":
    main()
