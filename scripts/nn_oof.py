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
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from expds import candidates, li_half, post, writer
from expds.catalog import read_utm_json, scan_root
from expds.lidar import DemPrefetch, build_dem, dem_params, hillshade_set
from expds.naming import site_family
from expds.pipeline import _li_with_model, _params, _raster_with_model, load_config

log = logging.getLogger("nn_oof")


def repeat_rows(rows: list[dict], regions: set[str], times: int, skip_families: set[str] | None = None) -> list[dict]:
    """Повторяет тайлы выбранных участков. Семьи из skip_families остаются в одном экземпляре.

    Повтор нужен, когда одна съёмка редка в кэше: эпоха видит её несколько раз,
    а проверка по семье не раздувается. times=1 возвращает те же строки.
    """
    if times < 1:
        raise ValueError(f"times={times}: нужно хотя бы один раз")
    if times == 1 or not regions:
        return list(rows)
    skip = skip_families or set()
    out = []
    for row in rows:
        copies = times if row.get("region") in regions and row.get("family") not in skip else 1
        out.extend(dict(row) for _ in range(copies))
    return out


def fold_cache(src: Path, dst: Path, rows: list[dict], fields: list[str]) -> None:
    """Кэш фолда: шарды — жёсткими ссылками (копией, если ссылки недоступны), индекс — свой."""
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.glob("*.npy"):
        target = dst / f.name
        if not target.exists():
            try:
                os.link(f, target)
            except OSError:
                try:
                    os.symlink(f.resolve(), target)
                except OSError:
                    shutil.copy2(f, target)
    shutil.copy2(src / "meta.json", dst / "meta.json")
    with (dst / "index.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def predict_region(seg, reg, markup: str, cand_p, li_cfg: dict | None = None, prefetch=None,
                   geo_maps: str = "gray", half_cfg: dict | None = None) -> list[dict]:
    """Предсказания модели по данным региона — тем же путём, что в pipeline.run_region.

    geo_maps="all" — карты геофизики все, и серые, и цветные; находки в одном месте
    разных карт сливаются в один объект (как [geophys.<модальность>] maps = "all" сдачи).
    """
    feats: list[dict] = []
    if markup == "li":
        dp = dem_params(seg.meta)
        crs = read_utm_json(reg.utm_json)
        for las in reg.las:
            d = (prefetch.get(las, dp["res"], dp["fill"], crs) if prefetch is not None
                 else build_dem(las, res=dp["res"], fill=dp["fill"], crs=crs))
            if d.crs is not None:
                feats += _li_with_model(seg, d, reg.name, las.name, hillshade_set, cand_p, li_cfg)
                if half_cfg is not None and li_half.dump_dir() is not None:
                    # EXPDS_LI_HALF_DUMP: второй проход 1 м только в выгрузку, --out не меняется.
                    # DEM здесь без мозаики: у крупных LAS он огрублён (0,2–0,3 м); на шаге 1 м
                    # это лишь другое усреднение блоков, потому skip_coarse тут не действует.
                    try:
                        li_half.half_pass(seg, d, reg.name, las.name, cand_p, li_half.params(half_cfg),
                                          li_half.dump_dir(), {"oof": True})
                    except Exception:
                        log.exception("%s/%s: li_half пропущен", reg.name, las.name)
        return feats
    files = reg.rasters.get(markup, [])
    if markup in ("maggr", "etgr", "terfl"):
        from expds import geophys

        files = geophys.select_rasters(files, markup, geo_maps)    # те же карты, что у сдачи
    for path in files:
        feats += _raster_with_model(seg, path, reg.name, markup, fallback_crs=read_utm_json(reg.utm_json))
    if markup in ("maggr", "etgr", "terfl") and geo_maps == "all" and len(files) > 1:
        from expds import geophys

        feats = geophys.dedupe_points(feats, 2.0, read_utm_json(reg.utm_json))
    return feats


def predict_with_weights(args, predict_mts: list[str]) -> None:
    """Режим --weights: одна модель на все участки, без фолдов и обучения."""
    from expds.infer_nn import Ensemble, Segmenter

    cfg = load_config(args.config)
    members = [Segmenter(Path(w)) for w in args.weights]
    seg = members[0] if len(members) == 1 else Ensemble(members)
    seg.tta = args.tta if args.tta is not None else int(cfg.get("tta", {}).get(args.markup, 1))
    rescore = cfg.get("li", {}).get("mode", "rescore") == "rescore"
    cand_p = _params(candidates.CandidateParams, cfg.get("candidates")) if rescore else None
    feats: list[dict] = []
    n_reg = 0
    for reg in scan_root(args.root):
        if not (reg.las if args.markup == "li" else any(reg.rasters.get(mt) for mt in predict_mts)):
            continue
        try:
            for mt in predict_mts:
                feats += predict_region(seg, reg, mt, cand_p, cfg.get("li"), geo_maps=args.geo_maps, half_cfg=cfg)
            n_reg += 1
        except Exception:
            log.exception("%s: предсказание пропущено", reg.name)
    out = post.finalize(post.merge_within_markup(feats))
    writer.write_geojson(out, args.out)
    print(json.dumps({"written": len(out), "regions": n_reg, "weights": args.weights, "markups": predict_mts},
                     ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--markup", default="li")
    ap.add_argument("--root", required=True, help="корень обучающей выборки")
    ap.add_argument("--cache", default=None, help="кэш тайлов модальности по всем участкам (expds.tiles)")
    ap.add_argument("--work", default=None, help="папка для кэшей и моделей фолдов")
    ap.add_argument("--out", required=True, help="GeoJSON предсказаний вне обучения")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold-indices", nargs="*", type=int, default=None,
                    help="обучать/предсказывать только указанные номера фолдов (распределённая ферма GPU)")
    ap.add_argument("--train-only", action="store_true",
                    help="только обучить веса выбранных фолдов без прогона инференса по растрам/LAS")
    ap.add_argument("--merge-only", action="store_true",
                    help="собрать итоговый GeoJSON из уже посчитанных work/fold*/pred.geojson без обучения")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--encoder", default="resnet18")
    ap.add_argument("--device", default=None, help="устройство PyTorch, например cuda:0 или cuda:1")
    ap.add_argument("--parallel-gpus", action="store_true",
                    help="запустить несколько фолдов параллельно на cuda:0 и cuda:1 (2×T4 Kaggle)")
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.toml"))
    ap.add_argument("--resume", action="store_true", help="не переобучать фолды с отметкой done.json")
    ap.add_argument("--predict-only", action="store_true",
                    help="только предсказание готовыми весами фолдов (с --resume); нет весов фолда — ошибка, без обучения")
    ap.add_argument("--tta", type=int, default=None, help="вариантов отражений на инференсе (иначе [tta] конфига)")
    ap.add_argument("--copy-paste", type=float, default=0.0,
                    help="вероятность Simple Copy-Paste кургана на другой тайл при обучении")
    ap.add_argument("--photo", action="store_true",
                    help="на обучении яркость ±0,08 и контраст ±0,15; проверка без этого")
    ap.add_argument("--scale", type=float, default=0.0,
                    help="случайный масштаб тайла, доля; 0,15 даёт 0,85–1,15. Отдельный опыт, вместе с --photo не включать")
    ap.add_argument("--seeds", type=int, default=1,
                    help="моделей на фолд с разными зёрнами; предсказание — их среднее (ансамбль)")
    ap.add_argument("--target", choices=("mask", "centers"), default="mask",
                    help="centers — гауссиана в центре насыпи, на выходе круг")
    ap.add_argument("--center-sigma", type=float, default=4.0, help="ширина гауссианы, м")
    ap.add_argument("--center-radius", type=float, default=6.0, help="радиус круга на выходе, м")
    ap.add_argument("--center-min-dist", type=float, default=12.0,
                    help="минимальное расстояние между центрами, м")
    ap.add_argument("--center-loss", choices=("bce", "focal"), default="bce")
    ap.add_argument("--center-unbiased", action="store_true",
                    help="гауссиана центра между клетками, без шипа в 1 (отдельный опыт, не рецепт сдачи)")
    ap.add_argument("--center-edge-ignore", action="store_true",
                    help="насыпь, обрезанную краем тайла, не учить: без пика, область −1 (отдельный опыт)")
    ap.add_argument("--use-ignore", action="store_true",
                    help="читать бит «не учить» кэша (--ignore-other-layers): объекты других слоёв — не фон")
    ap.add_argument("--classes", nargs="*", default=None,
                    help="префикс классов кэша: биты маски идут в этом порядке")
    ap.add_argument("--large-kurgan-weight", type=float, default=1.0,
                    help="sp_large: вес пика кургана от --large-area-m2; 1 — выключено")
    ap.add_argument("--large-damaged-weight", type=float, default=1.0,
                    help="sp_large: вес пика повреждённого кургана от --large-area-m2; 1 — выключено")
    ap.add_argument("--large-area-m2", type=float, default=1000.0)
    ap.add_argument("--repeat-regions", default=None,
                    help="файл участков, чьи тайлы повторяются в обучении (семья проверки — нет)")
    ap.add_argument("--repeat", type=int, default=1, help="сколько раз такой тайл входит в эпоху")
    ap.add_argument("--arch", default="Unet", help="декодер; Unet — рецепт сданных весов")
    ap.add_argument("--ema", type=float, default=0.0, help="экспоненциальное среднее весов, 0 — выключено")
    ap.add_argument("--flips", choices=("auto", "none", "all"), default="auto",
                    help="повороты и отражения тайлов на обучении; none — направление солнца на снимке сохраняется")
    ap.add_argument("--predict-markups", nargs="+", default=None,
                    help="типы снимков, которые предсказывает модель фолда (общий кэш scripts/cache_merge.py); "
                         "по умолчанию — только --markup")
    ap.add_argument("--geo-maps", choices=("gray", "all"), default="gray",
                    help="геофизика: только серая карта участка (как в сдаче) или все карты с дедупликацией находок")
    ap.add_argument("--weights", nargs="+", default=None,
                    help="готовые веса без обучения: все участки предсказывает эта модель (несколько — ансамбль). "
                         "Не вне обучения, если модель видела эти семьи; для переноса на другой тип снимков")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    predict_mts = args.predict_markups or [args.markup]
    if args.markup == "li" and predict_mts != ["li"]:
        raise SystemExit("--predict-markups — только для снимков и геофизики, не для li")
    if args.weights:
        predict_with_weights(args, predict_mts)
        return
    if not args.cache or not args.work:
        raise SystemExit("нужны --cache и --work (или --weights без обучения)")

    from expds.infer_nn import Ensemble, Segmenter
    from expds.train import train

    cache, work = Path(args.cache), Path(args.work)
    repeat_regions = ({ln.strip() for ln in Path(args.repeat_regions).read_text(encoding="utf-8").splitlines() if ln.strip()}
                      if args.repeat_regions else set())
    with (cache / "index.csv").open(encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        fields, rows = list(rd.fieldnames or []), list(rd)
    if args.classes:
        meta0 = json.loads((cache / "meta.json").read_text(encoding="utf-8"))
        missing = [c for c in args.classes if c not in meta0["classes"]]
        if missing:
            raise SystemExit(f"в кэше нет классов {missing}")
        bits = [meta0["classes"].index(c) for c in args.classes]
        by_shard: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for n, row in enumerate(rows):
            by_shard[int(row["shard"])].append((n, int(row["idx"])))
        hit = [False] * len(rows)
        for shard, items in by_shard.items():
            masks = np.load(cache / f"masks_{shard:04d}.npy", mmap_mode="r")
            for n, idx in items:
                mask = np.asarray(masks[idx])
                hit[n] = any(int(((mask >> b) & 1).any()) for b in bits)
        for n, row in enumerate(rows):
            row["positive"] = "1" if hit[n] else "0"
        print(f"тайлов выбранных классов {sum(hit)} из {len(rows)}", flush=True)
    tiles = Counter(r["family"] for r in rows)
    pos = Counter(r["family"] for r in rows if r["positive"] == "1")
    fams = sorted(tiles)
    fold = candidates.family_folds(fams, [pos[f] for f in fams], args.folds)
    cfg = load_config(args.config)
    rescore = cfg.get("li", {}).get("mode", "rescore") == "rescore"
    cand_p = _params(candidates.CandidateParams, cfg.get("candidates")) if rescore else None
    regions = ([] if (args.train_only or args.merge_only) else
               [r for r in scan_root(args.root)
                if (r.las if args.markup == "li" else any(r.rasters.get(mt) for mt in predict_mts))])
    feats: list[dict] = []
    report = []
    all_folds = sorted(set(fold.values()))
    if args.merge_only:
        for k in all_folds:
            fc = work / f"fold{k}"
            pf = fc / "pred.geojson"
            if not pf.exists():
                raise SystemExit(f"нет предсказаний фолда {k}: {pf}")
            feats += json.loads(pf.read_text(encoding="utf-8")).get("features", [])
            rf = fc / "fold_report.json"
            if rf.exists():
                report.append(json.loads(rf.read_text(encoding="utf-8")))
        out = post.finalize(post.merge_within_markup(feats))
        writer.write_geojson(out, args.out)
        (work / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"written": len(out), "folds": report}, ensure_ascii=False))
        return
    want_folds = set(args.fold_indices) if args.fold_indices is not None else set(all_folds)
    selected_folds = [f for f in all_folds if f in want_folds]
    if args.parallel_gpus and args.train_only and args.device is None and len(selected_folds) > 1:
        import subprocess
        import torch
        n_gpus = max(2, torch.cuda.device_count()) if torch.cuda.is_available() else 2
        base_argv = [a for a in sys.argv[1:] if a != "--parallel-gpus"]
        # Удаляем старый --fold-indices и его аргументы
        clean_argv = []
        skip_fold_vals = False
        for tok in base_argv:
            if tok == "--fold-indices":
                skip_fold_vals = True
                continue
            if skip_fold_vals:
                if tok.startswith("-"):
                    skip_fold_vals = False
                    clean_argv.append(tok)
                continue
            clean_argv.append(tok)
        procs = []
        for idx_f, fk in enumerate(selected_folds):
            dev_str = f"cuda:{idx_f % n_gpus}"
            cmd = [sys.executable, str(Path(__file__).resolve()), *clean_argv,
                   "--fold-indices", str(fk), "--device", dev_str]
            log.info("параллельный запуск фолда %d на %s", fk, dev_str)
            procs.append((fk, dev_str, subprocess.Popen(cmd)))
        failed = []
        for fk, dev_str, p_sub in procs:
            rc = p_sub.wait()
            if rc != 0:
                failed.append((fk, dev_str, rc))
        if failed:
            raise SystemExit(f"ошибка параллельного обучения фолдов: {failed}")
        for fk in selected_folds:
            rf = work / f"fold{fk}" / "fold_report.json"
            if rf.exists():
                report.append(json.loads(rf.read_text(encoding="utf-8")))
        (work / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"trained_folds": len(report), "folds": report, "parallel_gpus": True}, ensure_ascii=False))
        return
    for k in selected_folds:
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
            elif args.predict_only:
                raise SystemExit(f"--predict-only: нет готовых весов {model} / {done}")
            else:
                train_rows = [r for r in rows if r["family"] not in test]
                if repeat_regions and args.repeat > 1:
                    before = len(train_rows)
                    train_rows = repeat_rows(train_rows, repeat_regions, args.repeat, skip_families={val})
                    log.info("повтор участков: тайлов фолда %d → %d", before, len(train_rows))
                fold_cache(cache, fc / "cache", train_rows, fields)
                if args.classes:
                    meta_path = fc / "cache" / "meta.json"
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    want = list(args.classes)
                    missing = [c for c in want if c not in meta["classes"]]
                    if missing:
                        raise SystemExit(f"в кэше нет классов {missing}")
                    meta["class_bits"] = [meta["classes"].index(c) for c in want]
                    meta["classes"] = want
                    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
                res = train(str(fc / "cache"), str(model), encoder=args.encoder, epochs=args.epochs,
                            batch=args.batch, val_families=[val], workers=0, copy_paste=args.copy_paste,
                            seed=s if s else None, target=args.target,
                            center_sigma_m=args.center_sigma, center_radius_m=args.center_radius,
                            center_loss_name=args.center_loss, min_dist_m=args.center_min_dist,
                            center_spike=not args.center_unbiased, center_edge_ignore=args.center_edge_ignore, use_ignore=args.use_ignore,
                            photo=args.photo, scale=args.scale, arch=args.arch, ema=args.ema,
                            flips_mode=args.flips, device=args.device,
                            large_kurgan_weight=args.large_kurgan_weight,
                            large_damaged_weight=args.large_damaged_weight, large_area_m2=args.large_area_m2)
                res = {"train_tiles": res["train_tiles"], "best_val_loss": res["best_val_loss"]}
                done.write_text(json.dumps(res), encoding="utf-8")
            models.append(model)
        if args.train_only:
            rec = {"fold": k, "test": sorted(test), "val": val, **res, "train_only": True,
                   "minutes": round((time.perf_counter() - t0) / 60, 1)}
            report.append(rec)
            (fc / "fold_report.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
            log.info(json.dumps(rec, ensure_ascii=False))
            continue
        members = [Segmenter(m) for m in models]
        seg = members[0] if len(members) == 1 else Ensemble(members)
        seg.tta = args.tta if args.tta is not None else int(cfg.get("tta", {}).get(args.markup, 1))
        n_reg = 0
        fold_start = len(feats)
        fold_regions = [reg for reg in regions if site_family(reg.name) in test]
        prefetch = (DemPrefetch([(las, read_utm_json(r.utm_json)) for r in fold_regions for las in r.las],
                                int(cfg.get("lidar", {}).get("prefetch", 2)))
                    if args.markup == "li" and int(cfg.get("lidar", {}).get("prefetch", 2)) > 0 else None)
        try:
            for reg in fold_regions:
                try:
                    got = []
                    for mt in predict_mts:
                        got += predict_region(seg, reg, mt, cand_p, cfg.get("li"), prefetch, args.geo_maps, half_cfg=cfg)
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
        fold_feats = feats[fold_start:]
        writer.write_geojson(post.finalize(post.merge_within_markup(list(fold_feats))), fc / "pred.geojson")
        rec = {"fold": k, "test": sorted(test), "val": val, **res, "test_regions": n_reg,
               "minutes": round((time.perf_counter() - t0) / 60, 1)}
        (fc / "fold_report.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        report.append(rec)
        log.info(json.dumps(rec, ensure_ascii=False))
        writer.write_geojson(post.finalize(post.merge_within_markup(feats)), work / "partial.geojson")
    if args.train_only:
        (work / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"trained_folds": len(report), "folds": report}, ensure_ascii=False))
        return
    out = post.finalize(post.merge_within_markup(feats))
    writer.write_geojson(out, args.out)
    (work / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"written": len(out), "folds": report}, ensure_ascii=False))


if __name__ == "__main__":
    main()
