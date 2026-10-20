#!/usr/bin/env python3
"""Предсказания модели li «вне обучения» по «семьям» участков — для честной оценки и порогов.

Для каждого фолда модель учится на кэше тайлов без своих семей, затем
предсказывает их LAS тем же путём, что на инференсе (pipeline._li_with_model,
режим из configs/default.toml). Результат проходит ту же постобработку.

    python scripts/li_oof.py --root D:/expds3 --cache work/cache/li --work work/li_oof \
        --out work/li_oof.geojson --folds 5 --epochs 30
    python -m expds.metric sweep --pred work/li_oof.geojson --gt work/gt_train.geojson --markup li --bootstrap 300
"""
from __future__ import annotations

import argparse
import csv
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
from expds.lidar import build_dem, hillshade_set
from expds.naming import site_family
from expds.pipeline import _li_with_model, _params, load_config

log = logging.getLogger("li_oof")


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="корень обучающей выборки")
    ap.add_argument("--cache", required=True, help="кэш тайлов li по всем участкам (expds.tiles)")
    ap.add_argument("--work", required=True, help="папка для кэшей и моделей фолдов")
    ap.add_argument("--out", required=True, help="GeoJSON предсказаний вне обучения")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--encoder", default="resnet18")
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.toml"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from expds.infer_nn import Segmenter
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
    cand_p = _params(candidates.CandidateParams, cfg.get("candidates"))
    rescore = cfg.get("li", {}).get("mode", "rescore") == "rescore"
    regions = [r for r in scan_root(args.root) if r.las]
    feats: list[dict] = []
    report = []
    for k in sorted(set(fold.values())):
        test = {f for f in fams if fold[f] == k}
        train_fams = [f for f in fams if f not in test]
        n_train = sum(tiles[f] for f in train_fams)
        val = min(train_fams, key=lambda f: (abs(tiles[f] - 0.1 * n_train), f))
        t0 = time.perf_counter()
        fc = work / f"fold{k}"
        fold_cache(cache, fc / "cache", [r for r in rows if r["family"] not in test], fields)
        model = fc / "li.pt"
        res = train(str(fc / "cache"), str(model), encoder=args.encoder, epochs=args.epochs, batch=args.batch,
                    val_families=[val], workers=0)
        seg = Segmenter(model)
        n_las = 0
        for reg in regions:
            if site_family(reg.name) not in test:
                continue
            for las in reg.las:
                d = build_dem(las, crs=read_utm_json(reg.utm_json))
                if d.crs is None:
                    continue
                feats += _li_with_model(seg, d, reg.name, las.name, hillshade_set, cand_p if rescore else None)
                n_las += 1
        rec = {"fold": k, "test": sorted(test), "val": val, "train_tiles": res["train_tiles"],
               "best_val_loss": round(res["best_val_loss"], 4), "test_las": n_las,
               "minutes": round((time.perf_counter() - t0) / 60, 1)}
        report.append(rec)
        log.info(json.dumps(rec, ensure_ascii=False))
    out = post.finalize(post.merge_within_markup(feats))
    writer.write_geojson(out, args.out)
    (work / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"written": len(out), "folds": report}, ensure_ascii=False))


if __name__ == "__main__":
    main()
