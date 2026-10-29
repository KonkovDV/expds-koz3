#!/usr/bin/env python3
"""Переранжирование кандидатов курганов стека весами crop-verify из ядра Kaggle koz3-gpu-s4, честно по семьям.

s4 (`cropverify/<variant>/weights/fam10_foldKK.pt`) учил ResNet-18 «есть курган в центре кропа 64 px»
на кэше li_tin_59_t128 по 10 блокам семей (`expds.metric.family_folds`). Каждый чекпойнт хранит
`heldout_regions`: кандидат участка R оценивается ТОЛЬКО моделью, где R отложен. Так вероятность
кропа для участка не видела его разметку, и табло `--honest 10` остаётся честным.
Кропы режет `crop_verify.extract_crops` (то же чтение LAS и TIN 0,1 м, что кэш обучения).
Дальше — как в `crop_verify.run`: медиана группы для кандидатов без кропа, смешение --blend.

    python scripts/scoreboard.py <флаги базы> --dump-stack F:/expds3/work/stack_base.geojson
    python scripts/s4_rerank.py --root F:/2/train-expds-3 --stack F:/expds3/work/stack_base.geojson \
        --gt F:/expds3/work/gt_train_v2.geojson --weights <s4_out>/cropverify/r18_imnet/weights \
        --groups li li_dev li_mosaic --out F:/expds3/work/stack_s4_imnet.geojson --report notes/s4_imnet_rerank.json
    python scripts/scoreboard.py <те же флаги> --rerank F:/expds3/work/stack_s4_imnet.geojson --honest 10
"""
from __future__ import annotations

import argparse, importlib.util, json, logging, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("crop_verify", ROOT / "scripts" / "crop_verify.py")
cv = importlib.util.module_from_spec(spec); spec.loader.exec_module(cv)
log = logging.getLogger("s4_rerank")


def load_fold_models(wdir: Path, device: str):
    import torch
    models = []
    for pt in sorted(wdir.glob("fam10_fold*.pt")):
        blob = torch.load(pt, map_location="cpu", weights_only=False)
        net = cv.make_model(blob.get("encoder", "resnet18"), pretrained=False)
        net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in blob["state_dict"].items()})
        models.append((pt.name, set(blob["heldout_regions"]), net.to(device).eval(), blob))
    if len(models) != 10:
        raise SystemExit(f"ожидалось 10 чекпойнтов fam10_fold*.pt в {wdir}, найдено {len(models)}")
    seen = [r for _, h, _, _ in models for r in h]
    if len(seen) != len(set(seen)):
        raise SystemExit("один участок отложен в двух блоках — чекпойнты не из одного прогона")
    return models


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--stack", required=True); ap.add_argument("--gt", required=True)
    ap.add_argument("--metric", default=str(ROOT / "materials" / "compute_metrics.py"))
    ap.add_argument("--weights", required=True, help="<s4>/cropverify/<variant>/weights")
    ap.add_argument("--classes", nargs="+", default=["kurgany_povrezhdennye", "kurgany_tselye"])
    ap.add_argument("--groups", nargs="*", default=None, help="группы порога; без них — все кандидаты li")
    ap.add_argument("--min-conf", type=float, default=0.05)
    ap.add_argument("--blend", choices=["geo", "p", "none"], default="geo")
    ap.add_argument("--device", default="cpu"); ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", required=True); ap.add_argument("--report", default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    models = load_fold_models(Path(a.weights), a.device)
    b0 = models[0][3]; tile, res, tags = int(b0["tile"]), float(b0["res"]), tuple(b0.get("li_tags") or ())
    from expds.lidar import RELIEF_TAGS
    tags = tags or tuple(RELIEF_TAGS)
    stack = json.loads(Path(a.stack).read_text(encoding="utf-8"))
    gt = json.loads(Path(a.gt).read_text(encoding="utf-8"))
    picked = cv.select(stack, set(a.classes), "li", set(a.groups) if a.groups else None, a.min_conf)
    if not picked:
        raise SystemExit("нет кандидатов под фильтр")
    feats = [dict(f, properties=dict(f.get("properties") or {})) for f in stack["features"]]
    regions = np.array([str(feats[i]["properties"].get("region_name")) for i in picked])
    X, ok = cv.extract_crops(a.root, stack, picked, "li", res, tile, tags)
    p = np.full(len(picked), np.nan); used = np.full(len(picked), -1)
    for k, (name, held, net, _) in enumerate(models):
        idx = np.flatnonzero(ok & np.isin(regions, list(held)))
        if idx.size:
            p[idx] = cv.predict(net, X[idx], batch=a.batch); used[idx] = k
        log.info("%s: %d кандидатов", name, idx.size)
    no_model = sorted(set(regions[ok & (used < 0)].tolist()))
    if no_model:
        log.warning("участки без отложенной модели (остаются без кропа): %s", no_model)
    rr = cv.load_rerank()
    y_any, y_first = cv.labels_of(rr, stack, gt, a.metric, picked)
    conf = np.array([float(feats[i]["properties"].get("confidence", 0.0)) for i in picked])
    keys = [(feats[i]["properties"].get("threshold_group") or feats[i]["properties"].get("markup_type"),
             feats[i]["properties"].get("class_name")) for i in picked]
    med = cv.group_median(p, keys)
    rep = {"weights": a.weights, "candidates": len(picked), "with_crop": int(ok.sum()), "verified": int(np.isfinite(p).sum()),
           "regions_without_heldout_model": no_model, "hits_first": int(y_first.sum()), "blend": a.blend, "groups": {}}
    m = np.isfinite(p)
    if m.any() and 0 < y_first[m].sum() < m.sum():
        rep["auc_stack_first"] = round(rr.auc(conf[m], y_first[m]), 4)
        rep["auc_crop_first"] = round(rr.auc(p[m], y_first[m]), 4)
        rep["auc_crop_any"] = round(rr.auc(p[m], y_any[m]), 4)
        rep["auc_blend_first"] = round(rr.auc(cv.blend(conf[m], p[m], "geo"), y_first[m]), 4)
    for f in feats:
        f["properties"]["confidence_base"] = f["properties"].get("confidence")
    for k, i in enumerate(picked):
        pk = float(p[k]) if np.isfinite(p[k]) else med.get(keys[k])
        if pk is None:
            continue
        pr = feats[i]["properties"]; pr["verify_p"] = round(pk, 6)
        pr["confidence"] = round(float(cv.blend(np.array([conf[k]]), np.array([pk]), a.blend)[0]), 6)
    for key in sorted(med, key=str):
        rep["groups"][f"{key[0]}:{key[1]}"] = {"median_p": round(med[key], 4)}
    Path(a.out).write_text(json.dumps({"type": "FeatureCollection", "features": feats}, ensure_ascii=False), encoding="utf-8")
    if a.report:
        Path(a.report).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
