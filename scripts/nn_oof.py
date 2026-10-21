#!/usr/bin/env python3
"""Предсказания нейросети «вне обучения» по «семьям» участков — для честной оценки и порогов.

Для каждого фолда модель учится на кэше тайлов без своих семей, затем
предсказывает их данные тем же путём, что на инференсе: li — из LAS
(pipeline._li_with_model, режим из конфига), остальные модальности — по
растрам (pipeline._raster_with_model). Результат проходит ту же постобработку.

    python scripts/nn_oof.py --markup li --root D:/expds3 --cache work/cache/li --work work/li_oof \
        --out work/li_oof.geojson --folds 5 --epochs 30
    python -m expds.metric sweep --pred work/li_oof.geojson --gt work/gt_train.geojson --markup li --bootstrap 300
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from expds import candidates, post, writer
from expds.catalog import read_utm_json, scan_root
from expds.lidar import DemPrefetch, build_dem, dem_params, hillshade_set
from expds.naming import site_family
from expds.pipeline import _li_with_model, _params, _raster_with_model, load_config

log = logging.getLogger("nn_oof")


def fold_cache(src: Path, dst: Path, rows: list[dict], fields: list[str]) -> None:
    """Кэш фолда: шарды — жёсткими ссылками (копией, если ссылки недоступны), индекс — свой."""
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.glob("*.npy"):
        target = dst / f.name
        if not target.exists():
            try:
                os.link(f, target)
            except OSError:
                shutil.copy2(f, target)
    shutil.copy2(src / "meta.json", dst / "meta.json")
    with (dst / "index.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def predict_region(seg, reg, markup: str, cand_p, li_cfg: dict | None = None, prefetch=None) -> list[dict]:
    """Предсказания модели по данным региона — тем же путём, что в pipeline.run_region."""
    feats: list[dict] = []
    if markup == "li":
        dp = dem_params(seg.meta)
        crs = read_utm_json(reg.utm_json)
        for las in reg.las:
            d = (prefetch.get(las, dp["res"], dp["fill"], crs) if prefetch is not None
                 else build_dem(las, res=dp["res"], fill=dp["fill"], crs=crs))
            if d.crs is not None:
                feats += _li_with_model(seg, d, reg.name, las.name, hillshade_set, cand_p, li_cfg)
        return feats
    for path in reg.rasters.get(markup, []):
        feats += _raster_with_model(seg, path, reg.name, markup)
    return feats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--markup", default="li")
    ap.add_argument("--root", required=True, help="корень обучающей выборки")
    ap.add_argument("--cache", required=True, help="кэш тайлов модальности по всем участкам (expds.tiles)")
    ap.add_argument("--work", required=True, help="папка для кэшей и моделей фолдов")
    ap.add_argument("--out", required=True, help="GeoJSON предсказаний вне обучения")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--encoder", default="resnet18")
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.toml"))
    ap.add_argument("--resume", action="store_true", help="не переобучать фолды с отметкой done.json")
    ap.add_argument("--tta", type=int, default=None, help="вариантов отражений на инференсе (иначе [tta] конфига)")
    ap.add_argument("--copy-paste", type=float, default=0.0,
                    help="вероятность Simple Copy-Paste кургана на другой тайл при обучении")
    ap.add_argument("--seeds", type=int, default=1,
                    help="моделей на фолд с разными зёрнами; предсказание — их среднее (ансамбль)")
    ap.add_argument("--target", choices=("mask", "centers"), default="mask",
                    help="centers — гауссиана в центре насыпи, на выходе круг")
    ap.add_argument("--center-sigma", type=float, default=4.0, help="ширина гауссианы, м")
    ap.add_argument("--center-radius", type=float, default=6.0, help="радиус круга на выходе, м")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from expds.infer_nn import Ensemble, Segmenter
    from expds.train import train

    cache, work = Path(args.cache), Path(args.work)
    with (cache / "index.csv").open(encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        fields, rows = list(rd.fieldnames or []), list(rd)
    tiles = Counter(r["family"] for r in rows)
    pos = Counter(r["family"] for r in rows if r["positive"] == "1")
    fams = sorted(tiles)
    fold = candidates.family_folds(fams, [pos[f] for f in fams], args.folds)
    cfg = load_config(args.config)
    rescore = cfg.get("li", {}).get("mode", "rescore") == "rescore"
    cand_p = _params(candidates.CandidateParams, cfg.get("candidates")) if rescore else None
    regions = [r for r in scan_root(args.root) if (r.las if args.markup == "li" else r.rasters.get(args.markup))]
    feats: list[dict] = []
    report = []
    for k in sorted(set(fold.values())):
        test = {f for f in fams if fold[f] == k}
        train_fams = [f for f in fams if f not in test]
        n_train = sum(tiles[f] for f in train_fams)
        # Семья проверки должна содержать объекты: иначе пиксельный F1 не из чего считать.
        pool = [f for f in train_fams if pos[f] > 0] or train_fams
        val = min(pool, key=lambda f: (abs(tiles[f] - 0.1 * n_train), f))
        t0 = time.perf_counter()
        fc = work / f"fold{k}"
        models = []
        for s in range(max(1, args.seeds)):
            tag = "" if s == 0 else f"_s{s}"
            model, done = fc / f"{args.markup}{tag}.pt", fc / f"done{tag}.json"
            if args.resume and done.exists() and model.exists():
                res = json.loads(done.read_text(encoding="utf-8"))
            else:
                fold_cache(cache, fc / "cache", [r for r in rows if r["family"] not in test], fields)
                res = train(str(fc / "cache"), str(model), encoder=args.encoder, epochs=args.epochs,
                            batch=args.batch, val_families=[val], workers=0, copy_paste=args.copy_paste,
                            seed=s if s else None, target=args.target,
                            center_sigma_m=args.center_sigma, center_radius_m=args.center_radius)
                res = {"train_tiles": res["train_tiles"], "best_val_loss": res["best_val_loss"]}
                done.write_text(json.dumps(res), encoding="utf-8")
            models.append(model)
        members = [Segmenter(m) for m in models]
        seg = members[0] if len(members) == 1 else Ensemble(members)
        seg.tta = args.tta if args.tta is not None else int(cfg.get("tta", {}).get(args.markup, 1))
        n_reg = 0
        fold_regions = [reg for reg in regions if site_family(reg.name) in test]
        prefetch = (DemPrefetch([(las, read_utm_json(r.utm_json)) for r in fold_regions for las in r.las],
                                int(cfg.get("lidar", {}).get("prefetch", 2)))
                    if args.markup == "li" and int(cfg.get("lidar", {}).get("prefetch", 2)) > 0 else None)
        try:
            for reg in fold_regions:
                try:
                    got = predict_region(seg, reg, args.markup, cand_p, cfg.get("li"), prefetch)
                except Exception:
                    log.exception("%s: предсказание пропущено", reg.name)
                    continue
                feats += got
                n_reg += 1
                log.info("предсказание %s: %+d", reg.name, len(got))
                del got
                gc.collect()
        finally:
            if prefetch is not None:
                prefetch.close()
        rec = {"fold": k, "test": sorted(test), "val": val, **res, "test_regions": n_reg,
               "minutes": round((time.perf_counter() - t0) / 60, 1)}
        report.append(rec)
        log.info(json.dumps(rec, ensure_ascii=False))
        writer.write_geojson(post.finalize(post.merge_within_markup(feats)), work / "partial.geojson")
    out = post.finalize(post.merge_within_markup(feats))
    writer.write_geojson(out, args.out)
    (work / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"written": len(out), "folds": report}, ensure_ascii=False))


if __name__ == "__main__":
    main()
