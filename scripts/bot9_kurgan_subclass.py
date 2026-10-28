#!/usr/bin/env python3
"""bot-9: подкласс кургана (повреждённый / целый) по окну рельефа вокруг кандидата.

Зачем: 489 ложных «повреждённых» и 591 ложный «целый» — настоящие курганы эталона с другим
подклассом (docs/REASONING_AND_DISCOVERY_LOG_2026-10-28.md:80). Детектор находит насыпь, но
путает подкласс; веса метрики 30 и 20 (materials/compute_metrics.py:83–91).

Шаги (CPU, на ПК; обучение — ноутбук notebooks/01_kurgan_subclass.ipynb на Kaggle GPU):

    python scripts/bot9_kurgan_subclass.py crops --root F:/2/train-expds-3 \
        --stack F:/expds3/work/stack_base.geojson --gt F:/expds3/work/gt_train_v2.geojson \
        --markup li --res 0.5 --tile 64 --li-tags lrm_s lrm_l slope --out F:/expds3/work/kurgan_subclass_crops.npz
    # → загрузить .npz как датасет Kaggle koz3-kurgan-subclass-crops, прогнать ноутбук,
    #   скачать kurgan_subclass_oof.csv
    python scripts/bot9_kurgan_subclass.py apply --stack F:/expds3/work/stack_base.geojson \
        --oof kurgan_subclass_oof.csv --thr 0.5 --out F:/expds3/work/stack_subclass.geojson
    python scripts/scoreboard.py <флаги раздела 0> --rerank F:/expds3/work/stack_subclass.geojson --honest 10

Метка окна кандидата — подкласс ближайшего кургана эталона в радиусе 2 м (EPSG:3857 × cos φ), как радиус
метрики; кандидат без кургана эталона в обучение не идёт (подкласс у фона не определён), но
вероятность для него ноутбук тоже считает. Потеря взвешена 30 (повреждённый) : 20 (целый) —
веса метрики; это то же, что порог 0,4 = 20 / (30 + 20) на невзвешенной модели, поэтому
порог применения 0,5 задан заранее и на табло не подбирается.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

POVR, TSEL = "kurgany_povrezhdennye", "kurgany_tselye"


def _xy3857(geoms):
    from shapely.geometry import shape

    return [(shape(g).centroid.x, shape(g).centroid.y) for g in geoms]


def _merc_scale(y: float) -> float:
    """cos(широты) точки EPSG:3857: метры на местности = метры Меркатора × cos φ."""
    import math

    return math.cos(2 * math.atan(math.exp(y / 6378137.0)) - math.pi / 2)


def cmd_crops(a) -> None:
    from scipy.spatial import cKDTree

    import crop_verify as cv
    from expds.naming import site_family

    stack = json.loads(Path(a.stack).read_text(encoding="utf-8"))
    gt = json.loads(Path(a.gt).read_text(encoding="utf-8"))
    gfe = [f for f in gt["features"] if (f.get("properties") or {}).get("class_name") in (POVR, TSEL)]
    picked = cv.select(stack, {POVR, TSEL}, a.markup, set(a.groups) if a.groups else None, a.min_conf)
    # окна эталона — «кандидаты» с центром в кургане эталона
    gstack = {"features": [{"geometry": f["geometry"], "properties": {"region_name": f["properties"]["region_name"]}}
                           for f in gfe]}
    tags = tuple(a.li_tags or ())
    Xg, okg = cv.extract_crops(a.root, gstack, list(range(len(gfe))), a.markup, a.res, a.tile, tags)
    Xc, okc = cv.extract_crops(a.root, stack, picked, a.markup, a.res, a.tile, tags)
    gxy = _xy3857([f["geometry"] for f in gfe])
    cxy = _xy3857([stack["features"][i]["geometry"] for i in picked])
    greg = [f["properties"]["region_name"] for f in gfe]
    creg = [str(stack["features"][i]["properties"].get("region_name")) for i in picked]
    ycls = np.array([1 if f["properties"]["class_name"] == POVR else 0 for f in gfe], np.int8)
    yc = np.full(len(picked), -1, np.int8)
    for reg in set(creg):
        gi = [k for k, r in enumerate(greg) if r == reg]
        ci = [k for k, r in enumerate(creg) if r == reg]
        if not gi or not ci:
            continue
        tree = cKDTree(np.array([gxy[k] for k in gi]))
        d, j = tree.query(np.array([cxy[k] for k in ci]), k=1)
        lim = a.match_m / _merc_scale(gxy[gi[0]][1])  # один множитель на участок
        for k, dd, jj in zip(ci, d, j):
            if dd <= lim:
                yc[k] = ycls[gi[jj]]
    X = np.concatenate([Xg[okg], Xc[okc]])
    y = np.concatenate([ycls[okg], yc[okc]])
    region = np.array([r for r, o in zip(greg, okg) if o] + [r for r, o in zip(creg, okc) if o])
    kind = np.array(["gt"] * int(okg.sum()) + ["cand"] * int(okc.sum()))
    sidx = np.concatenate([np.full(int(okg.sum()), -1), np.asarray(picked, np.int64)[okc]])
    cls = np.array([""] * int(okg.sum()) +
                   [stack["features"][i]["properties"]["class_name"] for i, o in zip(picked, okc) if o])
    fam = np.array([site_family(r) for r in region])
    np.savez_compressed(a.out, X=X, y=y, region=region, family=fam, kind=kind, stack_idx=sidx, cand_class=cls,
                        res=a.res, tile=a.tile, li_tags=np.array(tags), markup=a.markup)
    print(json.dumps({"out": a.out, "gt_crops": int(okg.sum()), "cand_crops": int(okc.sum()),
                      "cand_labeled": int((yc[okc] >= 0).sum()), "povr_share_gt": float(ycls.mean()) if len(ycls) else None,
                      "families": int(len(set(fam)))}, ensure_ascii=False))


def cmd_apply(a) -> None:
    stack = json.loads(Path(a.stack).read_text(encoding="utf-8"))
    n = 0
    with open(a.oof, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            i = int(r["stack_idx"])
            p = r.get("p_povr")
            if i < 0 or p in (None, "", "nan"):
                continue
            pr = stack["features"][i]["properties"]
            new = POVR if float(p) >= a.thr else TSEL
            pr["subclass_p_povr"] = round(float(p), 6)
            if pr.get("class_name") != new:
                pr["class_name_base"] = pr.get("class_name")
                pr["class_name"] = new
                n += 1
    Path(a.out).write_text(json.dumps(stack, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"out": a.out, "relabeled": n, "thr": a.thr}, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("crops")
    c.add_argument("--root", required=True)
    c.add_argument("--stack", required=True)
    c.add_argument("--gt", required=True)
    c.add_argument("--markup", default="li")
    c.add_argument("--groups", nargs="*", default=None)
    c.add_argument("--min-conf", type=float, default=0.05)
    c.add_argument("--res", type=float, default=0.5)
    c.add_argument("--tile", type=int, default=64)
    c.add_argument("--li-tags", nargs="*", default=["lrm_s", "lrm_l", "slope"])
    c.add_argument("--match-m", type=float, default=2.0)
    c.add_argument("--out", required=True)
    p = sub.add_parser("apply")
    p.add_argument("--stack", required=True)
    p.add_argument("--oof", required=True)
    p.add_argument("--thr", type=float, default=0.5, help="0,5 при потере 30:20 (≡ 0,4 без весов)")
    p.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.dry_run:
        print(json.dumps(vars(a), ensure_ascii=False))
        return
    {"crops": cmd_crops, "apply": cmd_apply}[a.cmd](a)


if __name__ == "__main__":
    main()
