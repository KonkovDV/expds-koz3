#!/usr/bin/env python3
"""Честный honest-10 для правок полноты аэрофото. Только CPU и уже посчитанные OOF.

Не ходит в сеть, не читает Яндекс/Google/Bing/Esri и не сдаёт на лидерборд.
Если папки стека нет, выходит с кодом 2 и ничего не считает.

Сравнивает с базой A (те же --base-add, что у для 0,314615):

* baseline — стек как есть;
* mosaic_intact — целые копии группы li_ae переименованы в ae
  (то же, что [ae_recall] mosaic_intact = true на инференсе);
* dfine_oof — к стеку добавлен OOF D-FINE Ae (сырые уверенности NMS, как в ядре);
* both — оба сразу.

Два числа на вариант:

* honest-10 — взвешенный балл scripts/scoreboard.py (порог группы с других семей);
* аэрофото — справочный блок официальной метрики (precision с штрафом 0,5·FP,
  recall, F1 = 2PR/(P+R)) на объектах, прошедших эти же честные пороги.

Пример на ПК (без видеокарты; время — как один scoreboard --honest 10,
умноженный на число вариантов, обычно 2 или 4):

  python -X utf8 scripts/eval_ae_recall.py ^
    --work F:\\expds3\\work --train F:\\2\\train-expds-3 ^
    --expect 0.314615 --honest 10 ^
    --dfine-oof F:\\expds3\\work\\oof_dfine_ae.geojson ^
    --out F:\\expds3\\work\\ae_recall_honest10.json

Успех: код 0, в JSON есть baseline.honest10 и delta по каждому варианту,
а baseline.honest10 близок к 0,314615, если передан --expect.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import scoreboard as sb  # noqa: E402

# Файлы лежат в --work. Список — docs/TEAM_REPORTS_2026-10-28.md, база A.
DEFAULT_BASE = (
    "ae_fort_mask_oof.geojson:ae_fort:fortifikatsii:ae",
    "li_fort_mask_oof.geojson:li_fort:fortifikatsii:li",
    "spor_focal_wide_opt.geojson:sp_focal2",
    "gor_phys.geojson:gor_phys",
    "surf_arch_sp.geojson:sp_arch:arkhitektury:sp_or",
    "surf_arch_or.geojson:or:arkhitektury:or",
)


def _resolve(spec: str, work: Path) -> str:
    path, group, *rest = sb._split_windows(spec) if len(spec) > 2 and spec[1] == ":" else spec.split(":")
    p = Path(path)
    if not p.is_absolute():
        p = work / p
    return ":".join([str(p), group, *rest])


def _need(work: Path, train: Path) -> list[str]:
    missing = []
    for rel in ("fusion/li_hard2_fused.geojson", "mosaic_li", "spor_gor_oof.geojson",
                "geo_cur.geojson", "spor_focal_oof.geojson", "li_gor_mask_oof.geojson",
                "gt_train_v2.geojson"):
        if not (work / rel).exists():
            missing.append(str(work / rel))
    if not train.exists():
        missing.append(str(train))
    return missing


def _retag_intact(feats: list[dict]) -> list[dict]:
    out = []
    for feat in feats:
        p = dict(feat.get("properties") or {})
        if p.get("threshold_group") == "li_ae" and p.get("class_name") == "kurgany_tselye":
            p["threshold_group"] = "ae"
            feat = {**feat, "properties": p}
        out.append(feat)
    return out


def _with_dfine(feats: list[dict], path: Path) -> list[dict]:
    extra = []
    for feat in sb.load(path)["features"]:
        p = dict(feat.get("properties") or {})
        group = p.get("threshold_group") or p.get("group") or "dfine_ae"
        if group == "ae_dfine":
            group = "dfine_ae"
        p["threshold_group"] = group
        p["markup_type"] = p.get("markup_type") or "ae"
        extra.append({**feat, "properties": p})
    return list(feats) + extra


def _slice_curves(curves: dict, train: set[str]):
    from collections import Counter

    import numpy as np

    from expds.metric import ClassCurve

    out = {}
    for cls, c in curves.items():
        mask = np.fromiter((r in train for r in c.pred_region), dtype=bool, count=len(c.pred_region))
        gt_regions = Counter({r: k for r, k in c.gt_regions.items() if r in train})
        groups = [g for g, keep in zip(c.pred_group, mask) if keep] if c.pred_group else []
        out[cls] = ClassCurve(
            conf=c.conf[mask], hit=c.hit[mask],
            pred_region=[r for r, keep in zip(c.pred_region, mask) if keep],
            n_gt=int(sum(gt_regions.values())), gt_regions=gt_regions, pred_group=groups)
    return out


def _honest_kept(feats: list[dict], curves: dict, folds: list[set[str]]) -> list[dict]:
    """Объекты, прошедшие порог, снятый на остальных фолдах (как scoreboard --honest)."""
    from expds.metric import sweep_groups

    every = set().union(*folds) if folds else set()
    taus_of: dict[str, dict] = {}
    for held in folds:
        res = sweep_groups(_slice_curves(curves, every - held))
        taus: dict[str, dict[str, float]] = {}
        for cls, v in res.items():
            for g, gv in v["groups"].items():
                taus.setdefault(g, {})[cls] = gv["threshold"]
        for region in held:
            taus_of[region] = taus
    kept = []
    for feat in feats:
        p = feat.get("properties") or {}
        region = p.get("region_name") or ""
        taus = taus_of.get(region)
        if not taus:
            continue
        cls = p.get("class_name")
        group = p.get("threshold_group") or p.get("markup_type") or ""
        thr = taus.get(group, {}).get(cls)
        if thr is None:
            continue
        if float(p.get("confidence") or 0.0) >= thr:
            kept.append(feat)
    return kept


def _aerial(kept: list[dict], gt: dict, metric: str) -> dict:
    from expds.metric import score

    result = score({"type": "FeatureCollection", "features": kept}, gt, metric)
    row = ((result.get("auxiliary") or {}).get("by_data_type") or {}).get("аэрофото") or {}
    p = float(row.get("precision") or 0.0)
    r = float(row.get("recall") or 0.0)
    f1 = (2 * p * r / (p + r)) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": row.get("tp"), "fp": row.get("fp"),
            "fn": row.get("fn"), "final_score": result.get("final_score")}


def _delta(base: dict, other: dict) -> dict:
    out = {}
    for key in ("honest10",):
        if base.get(key) is not None and other.get(key) is not None:
            out[key] = other[key] - base[key]
    for key in ("precision", "recall", "f1"):
        b, a = (base.get("aerial") or {}).get(key), (other.get("aerial") or {}).get(key)
        if b is not None and a is not None:
            out[f"aerial_{key}"] = a - b
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--gt", default=None)
    ap.add_argument("--geo", default=None)
    ap.add_argument("--metric", default=str(ROOT / "materials" / "compute_metrics.py"))
    ap.add_argument("--expect", type=float, default=None, help="честная база A, обычно 0.314615")
    ap.add_argument("--honest", type=int, default=10)
    ap.add_argument("--bootstrap", type=int, default=300)
    ap.add_argument("--base-add", action="append", default=[], help="пусто — шесть файлов базы A в --work")
    ap.add_argument("--dfine-oof", default=None, help="OOF D-FINE Ae (GeoJSON). Без файла вариант не считается")
    ap.add_argument("--skip-aerial", action="store_true",
                    help="не звать официальную метрику повторно; в JSON будет только honest-10")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    work, train = Path(args.work), Path(args.train)
    missing = _need(work, train)
    if missing:
        print("Нет данных стека. Числа не считались. Не хватает:", file=sys.stderr)
        for item in missing:
            print(" ", item, file=sys.stderr)
        print("Команда, когда папка на месте, — в шапке scripts/eval_ae_recall.py", file=sys.stderr)
        return 2

    base_add = args.base_add or list(DEFAULT_BASE)
    gt_path = Path(args.gt) if args.gt else work / "gt_train_v2.geojson"
    geo = Path(args.geo) if args.geo else work / "geo_cur.geojson"
    try:
        feats, gt, new, optical = sb.build_stack(work, train, geo, gt_path)
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 2
    for spec in base_add:
        feats += sb.added(_resolve(spec, work))

    variants: dict[str, list[dict]] = {"baseline": feats, "mosaic_intact": _retag_intact(feats)}
    if args.dfine_oof:
        oof = Path(args.dfine_oof)
        if not oof.is_file():
            print("нет OOF:", oof, file=sys.stderr)
            return 2
        both = _with_dfine(feats, oof)
        variants["dfine_oof"] = both
        variants["dfine_oof+mosaic_intact"] = _retag_intact(both)

    from expds.metric import family_folds, paired_bootstrap_counts
    from expds.naming import site_family

    rows = {}
    regions = {((f.get("properties") or {}).get("region_name")) for f in gt["features"]}
    regions.discard(None)
    folds = family_folds(regions, site_family, None if args.honest < 0 else args.honest)
    for name, data in variants.items():
        print(f"=== {name} ===", flush=True)
        _scores, _taus, kept_in, _per = sb.evaluate(name, data, gt, new, optical, args.metric)
        honest_scores, honest_counts = sb.honest(name, new, optical, args.honest, "family", "drop")
        row = {"honest10": float(honest_scores[0]), "honest10_new": float(honest_scores[1]),
               "honest10_optical": float(honest_scores[2]), "counts": honest_counts,
               "insample_all": float(_scores[0])}
        if not args.skip_aerial:
            print(f"{name}: официальная метрика по честным порогам…", flush=True)
            kept = _honest_kept(data, sb.LAST_CURVES[name], folds)
            row["aerial"] = _aerial(kept, gt, args.metric)
            row["n_honest_kept"] = len(kept)
            row["n_insample_kept"] = len(kept_in["features"])
            ae = row["aerial"]
            print(f"{name}: аэрофото P {ae['precision']:.4f} R {ae['recall']:.4f} F1 {ae['f1']:.4f}", flush=True)
        rows[name] = row

    base = rows["baseline"]
    if args.expect is not None and abs(base["honest10"] - args.expect) > 0.0005:
        print(f"честная база {base['honest10']:.6f} не совпала с {args.expect:.6f}", file=sys.stderr)
        return 1
    report = {"expect": args.expect, "variants": {}, "note":
              "honest10 — взвешенный балл scoreboard --honest. aerial — официальный блок «аэрофото» "
              "после порогов, снятых на других семьях. Не лидерборд."}
    for name, row in rows.items():
        pub = {k: v for k, v in row.items() if k != "counts"}
        if name != "baseline":
            pub["delta"] = _delta(base, row)
            boot = paired_bootstrap_counts(row["counts"], base["counts"], site_family, args.bootstrap)
            pub["bootstrap_families"] = {k: float(boot[k]) for k in ("mean", "p5", "p95", "p_gt0") if k in boot}
        report["variants"][name] = pub
        if name != "baseline":
            d = pub["delta"]
            print(f"{name}: honest10 {d.get('honest10', float('nan')):+.6f}  "
                  f"aerial R {d.get('aerial_recall', float('nan')):+.4f}  "
                  f"P {d.get('aerial_precision', float('nan')):+.4f}  "
                  f"F1 {d.get('aerial_f1', float('nan')):+.4f}", flush=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("JSON:", args.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
