#!/usr/bin/env python3
"""Табло сдачи вне обучения: сборка стека, официальный балл, три числа и бутстреп.

Стек — то, что сейчас в configs/default.toml:
  • лидар `li_r18_hard2` вне обучения со слиянием в аэрофото (`fusion/li_hard2_fused.geojson`);
    облака с грубой сеткой помечаются `li_coarse` по заголовкам LAS;
  • курганы мозаики 0,1→0,5 м для 24 грубых облаков, группа `li_mosaic` (`mosaic_li/*.geojson`);
  • городища спутника (`spor_gor_oof.geojson`), карта центров курганов спутника (`sp_focal`);
  • геофизика (`--geo`);
  • маска городищ лидара, окно 384 м, группа `li_mask` (`li_gor_mask_oof.geojson`).

Пороги каждой группы подбираются перебором на всех участках (`sweep_groups`). Три числа:
все участки, 34 новых LAS (`notes/regions_las_new.txt`), участки без лидара.

`--honest K` добавляет честный балл: порог группы подбирается на других семьях и применяется
к своей (K фолдов семей, -1 — одна семья на фолд). На стеке 0,350588 так получалось 0,329489,
а площадка дала 0,32850. Новая группа порогов поднимает балл «на всех» и тогда, когда переноса
нет, поэтому решение о конфиге — по строке «честная приёмка».

    python scripts/scoreboard.py --work F:/expds3/work --train F:/expds3/train-expds-3 \
        --metric materials/compute_metrics.py --expect 0.338586

Опыт поверх стека (база печатается первой и сверяется с --expect):

    ... --add F:/expds3/work/ae_fort_mask_oof.geojson:ae_fort:fortifikatsii:ae --out notes/x.json
    ... --replace sp_focal=F:/expds3/work/spor_hardneg_oof.geojson

Файлы предсказаний и выборка в git не лежат, только этот код и числа в notes/.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

KUR = {"kurgany_povrezhdennye", "kurgany_tselye"}


def load(path: Path) -> dict:
    import gc

    p = Path(path)
    print(f"читаю {p.name}", flush=True)
    gc.disable()
    return json.loads(p.read_bytes())


def props(feat: dict) -> dict:
    return feat.setdefault("properties", {})


def subset(feats: list[dict], regions: set[str]) -> list[dict]:
    return [f for f in feats if (f.get("properties") or {}).get("region_name") in regions]


def tagged(feats: list[dict], group: str, class_name: str | None = None, markup: str | None = None) -> list[dict]:
    for feat in feats:
        p = props(feat)
        p["threshold_group"] = group
        if class_name:
            p["class_name"] = class_name
        if markup:
            p["markup_type"] = markup
    return feats


def drop_coarse_kurgans(feats: list[dict], names: set[str]) -> list[dict]:
    """Курганы лидара и их копии в аэрофото на грубых облаках заменяются мозаикой."""
    out = []
    for f in feats:
        p = f.get("properties") or {}
        if p.get("region_name") in names and p.get("class_name") in KUR and (
            p.get("markup_type") in ("li", "ae") or p.get("threshold_group") in ("li", "li_coarse", "li_ae")
        ):
            continue
        out.append(f)
    return out


def mosaic_kurgans(folder: Path) -> tuple[set[str], list[dict]]:
    names = {p.stem for p in folder.glob("*.geojson")}
    kept = []
    for name in sorted(names):
        for f in load(folder / f"{name}.geojson")["features"]:
            p = props(f)
            if p.get("class_name") not in KUR:
                continue
            p["threshold_group"] = "li_ae" if p.get("markup_type") == "ae" or p.get("threshold_group") == "li_ae" \
                else "li_mosaic"
            kept.append(f)
    return names, kept


GEO_MARKUPS = {"maggr", "etgr", "terfl"}


def is_geophysics(feat: dict) -> bool:
    """Объект файла геофизики: модальность или её группа порога (в том числе etgr_copy)."""
    p = feat.get("properties") or {}
    mark = str(p.get("markup_type") or "")
    group = str(p.get("threshold_group") or "")
    if mark in GEO_MARKUPS or group in GEO_MARKUPS:
        return True
    head = group.split("_", 1)[0]
    return head in GEO_MARKUPS and group != head


def swap_geophysics(feats: list[dict], path: Path) -> list[dict]:
    """База остаётся на своём файле геофизики; в опыте он заменён целиком."""
    kept = [f for f in feats if not is_geophysics(f)]
    kept.extend(load(path)["features"])
    return kept


def satellite_centers(fc: dict) -> list[dict]:
    out = []
    for f in fc["features"]:
        p = dict(f.get("properties") or {})
        if p.get("class_name") not in KUR:
            continue
        p["threshold_group"] = "sp_focal"
        p["markup_type"] = p.get("markup_type") or "sp_or"
        out.append({**f, "properties": p})
    return out


def build_stack(work: Path, train: Path, geo: Path, gt_path: Path | None = None,
                fused_path: Path | None = None, mosaic_dir: Path | None = None):
    """fused_path и mosaic_dir заменяют лидар целиком: другая модель li сравнивается заменой, а не второй группой."""
    from expds.coarse import las_boxes, tag

    gt = load(gt_path or work / "gt_train.geojson")
    boxes = las_boxes(train)
    fused = load(fused_path or work / "fusion" / "li_hard2_fused.geojson")
    tag(fused, boxes)
    names, mosaic = mosaic_kurgans(mosaic_dir or work / "mosaic_li")
    if len(names) < 24:
        raise SystemExit(f"мозаика не на всех 24 грубых облаках: {len(names)}")
    extra = (load(work / "spor_gor_oof.geojson")["features"] + load(geo)["features"]
             + satellite_centers(load(work / "spor_focal_oof.geojson")))
    feats = drop_coarse_kurgans(fused["features"], names) + mosaic + extra
    tag({"type": "FeatureCollection", "features": feats}, boxes)
    feats += tagged(load(work / "li_gor_mask_oof.geojson")["features"], "li_mask", "gorodishcha", "li")
    regions = {(f.get("properties") or {}).get("region_name") for f in gt["features"]}
    optical = {r for r in regions if r} - set(boxes)
    new = {ln.strip() for ln in (ROOT / "notes" / "regions_las_new.txt").read_text(encoding="utf-8").splitlines()
           if ln.strip()}
    return feats, gt, new, optical


OFFICIAL = False


def final_score(kept: dict, gt: dict, metric: str) -> float:
    """Итоговый балл набора после порогов.

    Официальный F1Calculator сопоставляет каждое предсказание со всеми эталонами
    класса без индекса и ещё раз так же считает справочную мягкую полноту. Здесь —
    то же жадное сопоставление (build_curves: тот же порядок, та же проверка пары),
    но только по парам с пересекающимися рамками. Остальные пары попасть не могут.
    Совпадение с официальным расчётом проверяется флагом --official.
    """
    from expds.metric import build_curves, score, score_at

    if OFFICIAL:
        return score(kept, gt, metric)["final_score"]
    return score_at(build_curves(kept, gt, metric), {})[0]


def evaluate(name: str, feats: list[dict], gt: dict, new: set[str], optical: set[str], metric: str):
    from expds.metric import apply_group_thresholds, build_curves, sweep_groups

    fc = {"type": "FeatureCollection", "features": feats}
    print(f"{name}: объектов {len(feats)}", flush=True)
    curves = build_curves(fc, gt, metric)
    res = sweep_groups(curves)
    taus: dict = {}
    per = {}
    for cls, v in res.items():
        if v["f1"] <= 0 and not any(g["kept"] for g in v["groups"].values()):
            continue
        per[cls] = {"f1": v["f1"], "n_gt": v["n_gt"],
                    "groups": {g: {"threshold": gv["threshold"], "kept": gv["kept"], "tp": gv["tp"]}
                               for g, gv in v["groups"].items()}}
        print(f"  {cls:24s} F1 {v['f1']:.4f} GT {v['n_gt']}", flush=True)
        for g, gv in v["groups"].items():
            taus.setdefault(g, {})[cls] = gv["threshold"]
            if gv["kept"]:
                print(f"    {g:11s} порог {gv['threshold']:.6f}  оставлено {gv['kept']:5d}  TP {gv['tp']}", flush=True)
    kept = apply_group_thresholds(fc, taus, metric)
    scores = [final_score(kept, gt, metric)]
    for regions in (new, optical):
        scores.append(final_score({"type": "FeatureCollection", "features": subset(kept["features"], regions)},
                                  {"type": "FeatureCollection", "features": subset(gt["features"], regions)},
                                  metric))
    print(f"{name}: все {scores[0]:.6f}; 34 новых {scores[1]:.6f}; без лидара {scores[2]:.6f}", flush=True)
    LAST_CURVES[name] = curves
    return scores, taus, kept, per


#: Кривые последнего evaluate по имени прогона: честный балл берёт их, а не строит заново.
LAST_CURVES: dict = {}


def unit_of(by: str):
    """Единица честной проверки: семья (новые памятники) или участок (соседний планшет той же семьи)."""
    from expds.naming import site_family

    return site_family if by == "family" else (lambda r: r)


def honest(name: str, new: set[str], optical: set[str], folds_k: int, by: str = "family", unseen: str = "drop"):
    """Порог группы подбирается на других фолдах и применяется к своему (как cross_family.json).

    by="family" — фолд из целых семей: оценка для новых памятников. by="region" — фолд из
    участков: оценка для соседних планшетов уже знакомых памятников. Какая из них ближе
    к площадке, показывает scripts/site_overlap.py.
    """
    from expds.metric import family_folds, honest_counts, score_counts

    curves = LAST_CURVES[name]
    regions = {r for c in curves.values() for r in c.pred_region} | {r for c in curves.values() for r in c.gt_regions}
    folds = family_folds(regions, unit_of(by), None if folds_k < 0 else folds_k)
    counts, info = honest_counts(curves, folds, unseen=unseen)
    scores = [score_counts(counts)[0], score_counts(counts, new)[0], score_counts(counts, optical)[0]]
    unit = "семей" if by == "family" else "участков"
    print(f"{name}, честно ({info['folds']} фолдов {unit}): все {scores[0]:.6f}; 34 новых {scores[1]:.6f}; "
          f"без лидара {scores[2]:.6f}; без порога выброшено {info['unseen_dropped']}, оставлено {info['unseen_kept']}", flush=True)
    return scores, counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, help="папка с предсказаниями вне обучения")
    ap.add_argument("--train", required=True, help="корень трейна: рамки LAS для li_coarse")
    ap.add_argument("--metric", default=str(ROOT / "materials" / "compute_metrics.py"))
    ap.add_argument("--geo", default=None, help="геофизика базы; по умолчанию <work>/geo_cur.geojson")
    ap.add_argument("--geo-experiment", default=None,
                    help="файл геофизики только для опыта; база остаётся на --geo")
    ap.add_argument("--gt", default=None,
                    help="эталон; по умолчанию <work>/gt_train_v2.geojson (слои EtGr не склеены)")
    ap.add_argument("--expect", type=float, default=None, help="база должна совпасть с точностью 0,0005")
    ap.add_argument("--base-add", action="append", default=[],
                    help="путь:группа[:класс[:слой]] — уже принятая ветка, входит в базу")
    ap.add_argument("--add", action="append", default=[], help="путь:группа[:класс[:слой]]")
    ap.add_argument("--replace", action="append", default=[],
                    help="группа=путь: убрать объекты группы и добавить файл; своя группа объекта сохраняется")
    ap.add_argument("--fused", default=None,
                    help="другой лидар со слиянием в аэрофото вместо fusion/li_hard2_fused.geojson (опыт, не база)")
    ap.add_argument("--mosaic-dir", default=None, help="курганы мозаики этого лидара вместо <work>/mosaic_li")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--honest", type=int, default=0,
                    help="честный балл: K фолдов семей (порог группы — с других семей); -1 — одна семья на фолд; "
                         "0 — выключено. Приёмка по нему печатается отдельной строкой")
    ap.add_argument("--honest-by", choices=("family", "region"), default="family",
                    help="фолды из семей (новые памятники, приёмка) или из участков (соседние планшеты). "
                         "region — только после site_overlap.py на настоящих входах выборки, не на mock_val")
    ap.add_argument("--honest-unseen", choices=("drop", "keep"), default="drop",
                    help="группа без порога на обучающих фолдах: drop (как было) или keep (как сдача, уровень ближе к LB). "
                         "Δ двух вариантов сравнимы, только если у них одинаковое число таких объектов")
    ap.add_argument("--official", action="store_true",
                    help="итоговые баллы полным официальным расчётом (медленно; для сверки)")
    ap.add_argument("--out", default=None, help="JSON с тремя числами, порогами и бутстрепом")
    ap.add_argument("--dump-kept", default=None,
                    help="база после подобранных порогов — вход scripts/error_types.py (в рабочую папку)")
    ap.add_argument("--dump-stack", default=None,
                    help="стек базы до порогов — вход scripts/rerank_oof.py (в рабочую папку)")
    ap.add_argument("--rerank", default=None,
                    help="опыт — весь стек из файла scripts/rerank_oof.py вместо стека базы")
    ap.add_argument("--sp-fusion", action="store_true",
                    help="опыт — слияние уверенностей sp_focal/sp_focal2 для курганов (post.sp_score_fusion, bot-6 ORSP)")
    args = ap.parse_args()
    global OFFICIAL
    OFFICIAL = args.official
    work = Path(args.work)
    geo = Path(args.geo) if args.geo else work / "geo_cur.geojson"
    gt_path = Path(args.gt) if args.gt else work / "gt_train_v2.geojson"
    feats, gt, new, optical = build_stack(work, Path(args.train), geo, gt_path)
    for spec in args.base_add:
        feats += added(spec)
    base, _, kept_b, per_b = evaluate("база", feats, gt, new, optical, args.metric)
    if args.dump_kept:
        Path(args.dump_kept).write_text(json.dumps(kept_b, ensure_ascii=False), encoding="utf-8")
    if args.dump_stack:
        Path(args.dump_stack).write_text(json.dumps({"type": "FeatureCollection", "features": feats},
                                                    ensure_ascii=False), encoding="utf-8")
    if args.expect is not None and abs(base[0] - args.expect) > 0.0005:
        raise SystemExit(f"база {base[0]:.6f} не совпала с {args.expect:.6f}: стек собран не так")
    honest_b = honest("база", new, optical, args.honest, args.honest_by, args.honest_unseen) if args.honest else None
    swap_li = bool(args.fused or args.mosaic_dir)
    if not args.add and not args.replace and not swap_li and not args.rerank and not args.geo_experiment \
            and not args.sp_fusion:
        if args.out:
            Path(args.out).write_text(json.dumps({"base": base, "per_class": per_b,
                                                  "honest_base": honest_b[0] if honest_b else None},
                                                 ensure_ascii=False, indent=2), encoding="utf-8")
        return
    mixed = list(feats)
    if swap_li:
        # Опыт с другим лидаром: стек собирается заново из его файлов, база остаётся прежней.
        mixed, _, _, _ = build_stack(work, Path(args.train), geo, gt_path,
                                     Path(args.fused) if args.fused else None,
                                     Path(args.mosaic_dir) if args.mosaic_dir else None)
        for spec in args.base_add:
            mixed += added(spec)
    for spec in args.replace:
        group, path = spec.split("=", 1)
        mixed = [f for f in mixed if (f.get("properties") or {}).get("threshold_group") != group]
        fresh = load(Path(path))["features"]
        for f in fresh:
            props(f).setdefault("threshold_group", group)
        mixed += fresh
    for spec in args.add:
        mixed += added(spec)
    if args.geo_experiment:
        mixed = swap_geophysics(mixed, Path(args.geo_experiment))
    if args.rerank:
        # Переоценённый стек заменяет базу целиком; --add и --replace к нему не применяются.
        mixed = load(Path(args.rerank))["features"]
    if args.sp_fusion:
        from expds.post import sp_score_fusion
        mixed = sp_score_fusion(mixed)
    from expds.metric import build_curves, paired_bootstrap

    after, taus, kept_a, per_a = evaluate("опыт", mixed, gt, new, optical, args.metric)
    boot = paired_bootstrap(build_curves(kept_a, gt, args.metric), build_curves(kept_b, gt, args.metric),
                            {}, {}, args.bootstrap)["score"]
    labels = ("все", "34 новых", "без лидара")
    for label, b, a in zip(labels, base, after):
        print(f"{label}: {b:.6f} → {a:.6f} ({a - b:+.6f})", flush=True)
    print(f"A − B: среднее {boot['mean']:+.4f}, 90% [{boot['p5']:+.4f}; {boot['p95']:+.4f}], "
          f"A > B в {boot['p_gt0']:.3f}", flush=True)
    accept = after[0] - base[0] >= 0.003 and boot["p_gt0"] >= 0.9 and after[1] >= base[1] and after[2] >= base[2]
    print("приёмка:", "проходит" if accept else "не проходит", flush=True)
    honest_res = None
    if honest_b is not None:
        from expds.metric import paired_bootstrap_counts
        from expds.naming import site_family

        honest_a = honest("опыт", new, optical, args.honest, args.honest_by, args.honest_unseen)
        # Бутстреп всегда по семьям: участки одной семьи не независимы при любой единице фолда.
        hboot = paired_bootstrap_counts(honest_a[1], honest_b[1], site_family, args.bootstrap)
        for label, b, a in zip(labels, honest_b[0], honest_a[0]):
            print(f"честно, {label}: {b:.6f} → {a:.6f} ({a - b:+.6f})", flush=True)
        print(f"честно, A − B по семьям: среднее {hboot['mean']:+.4f}, 90% [{hboot['p5']:+.4f}; {hboot['p95']:+.4f}], "
              f"A > B в {hboot['p_gt0']:.3f}", flush=True)
        honest_accept = (honest_a[0][0] - honest_b[0][0] >= 0.002 and hboot["p_gt0"] >= 0.8
                         and honest_a[0][1] >= honest_b[0][1] - 0.001 and honest_a[0][2] >= honest_b[0][2] - 0.001)
        print("честная приёмка:", "проходит" if honest_accept else "не проходит", flush=True)
        honest_res = {"base": honest_b[0], "after": honest_a[0], "bootstrap_families": hboot,
                      "accept": honest_accept, "folds": args.honest, "by": args.honest_by}
    if args.out:
        Path(args.out).write_text(json.dumps({
            "base_all": base[0], "all": after[0], "delta_all": after[0] - base[0],
            "base_new": base[1], "new": after[1], "delta_new": after[1] - base[1],
            "base_optical": base[2], "optical": after[2], "delta_optical": after[2] - base[2],
            "bootstrap": boot, "accept": accept, "per_class": per_a,
            "thresholds": {g: {c: round(t, 6) for c, t in cls.items()} for g, cls in taus.items()},
            "add": args.add, "replace": args.replace, "geo": geo.name, "gt": gt_path.name,
            "honest": honest_res,
        }, ensure_ascii=False, indent=2), encoding="utf-8")


def added(spec: str) -> list[dict]:
    """Объекты из «путь:группа[:класс[:слой]]» с проставленной группой."""
    path, group, *rest = _split_windows(spec) if re.match(r"^[A-Za-z]:[\\/]", spec) else spec.split(":")
    return tagged(load(Path(path))["features"], group, rest[0] if rest else None, rest[1] if len(rest) > 1 else None)


def _split_windows(spec: str) -> list[str]:
    """«C:\\путь\\файл.geojson:группа:класс» — двоеточие диска не разделитель."""
    drive, rest = spec[:2], spec[2:]
    parts = rest.split(":")
    return [drive + parts[0]] + parts[1:]


if __name__ == "__main__":
    import os
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
