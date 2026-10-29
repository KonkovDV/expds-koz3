#!/usr/bin/env python3
"""Задача 11: выгрузка li_half (EXPDS_LI_HALF_DUMP, *.npz) → стек семей + новые курганы группы li_half.

Стек читается потоком из FeatureCollection (по умолчанию /workspace/rerank/stack_arch.geojson) и
пишется в JSONL по семьям без изменений. Из него же собираются якоря: курганы markup li групп li,
li_mosaic и li_dev (σ) с уверенностью не ниже принятого порога (KEPT_KURGAN_CONF main 24bd916) — ровно как
pipeline.lidar_feats → li_half.new_only. Поправка 29.10 09:35 MSK (смоук бота 8): отбор ПЕРЕКРЁСТНЫЙ по классу —
круг li_half любого класса выбрасывается, если ближе 12 м есть принятый курган ЛЮБОГО класса (целый или
повреждённый) того же участка: метрика сопоставляет по классу, тот же холм в другом классе — ложное
срабатывание. Якоря также — курганы канала σ (группа li_dev) с уверенностью ≥ порога main 24bd916
(0,99131 / 0,99957): стек A их содержит и не меняется. Остальные дописываются с threshold_group=li_half. Порог li_half подбирает
family_curves/honest_counts вложенно (на остальных семьях) — без ручного порога.

    python li_half_eval.py --dump DIR --out FAMDIR [--stack stack.geojson] [--min-conf 0.0]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

EXPDS = Path(os.environ.get("EXPDS_REPO", "/workspace/koz3_shared/bot-10/repo"))   # только чтение
sys.path.insert(0, str(EXPDS)); sys.path.insert(0, str(EXPDS / "scripts"))

from expds.naming import site_family  # noqa: E402

KUR = ("kurgany_povrezhdennye", "kurgany_tselye")
KEPT = {"li": {"kurgany_povrezhdennye": 0.06415, "kurgany_tselye": 0.19422},
        "li_mosaic": {"kurgany_povrezhdennye": 0.959005, "kurgany_tselye": 0.17042},
        "li_dev": {"kurgany_povrezhdennye": 0.99131, "kurgany_tselye": 0.99957}}   # configs/default.toml main 24bd916


def is_anchor(pr: dict, regions: set) -> bool:
    """Принятый курган стека (li / li_mosaic / li_dev, conf ≥ порога группы) на участке из выгрузки."""
    g = str(pr.get("threshold_group") or pr.get("markup_type") or "")
    cls = pr.get("class_name")
    return (str(pr.get("region_name", "default_source")) in regions and pr.get("markup_type") == "li"
            and cls in KUR and g in KEPT and float(pr.get("confidence") or 0) >= KEPT[g][cls])


def drop_near_any(cands: list[dict], anchor_pts: list, dist_m: float) -> list[dict]:
    """Перекрёстный по классу отбор: круг вон, если ближе dist_m (по точкам ground_point, метры) есть
    якорь любого класса. anchor_pts — уже точки ground_point якорей этого участка."""
    from shapely.strtree import STRtree
    from expds.pipeline import ground_point

    if not anchor_pts:
        return list(cands)
    tree = STRtree(anchor_pts)
    kept = []
    for f in cands:
        if f.get("geom") is None:
            kept.append(f); continue
        p = ground_point(f["geom"])
        if not any(anchor_pts[int(i)].distance(p) <= dist_m for i in tree.query(p.buffer(dist_m))):
            kept.append(f)
    return kept


def half_feats(npz: Path, circle_m: float | None, min_conf: float, group: str):
    from shapely import buffer, points
    from expds.crs import WEB_MERCATOR, to_crs

    d = np.load(npz, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    r = float(circle_m if circle_m is not None else meta["cand"]["circle_m"])
    crs = str(d["crs"]) or None
    x, y, s = np.asarray(d["x"]), np.asarray(d["y"]), np.asarray(d["scores"], np.float32)
    classes = [str(c) for c in d["classes"]]
    if len(x) == 0:
        return [], meta
    geoms = buffer(points(x, y), r, quad_segs=4)
    if crs and crs != WEB_MERCATOR:
        geoms = [to_crs(g, crs, WEB_MERCATOR) for g in geoms]
    out = []
    for i, g in enumerate(geoms):
        for k, cls in enumerate(classes):
            if cls in KUR and float(s[i, k]) >= min_conf:
                out.append({"geom": g, "class_name": cls, "confidence": float(s[i, k]), "region_name": str(d["region"]),
                            "markup_type": "li", "threshold_group": group, "source": str(d["source"]),
                            "original_crs": crs})
    return out, meta


def main() -> None:
    from shapely.geometry import mapping, shape
    from split_stack_by_family import iter_features

    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stack", default="/workspace/rerank/stack_arch.geojson")
    ap.add_argument("--group", default="li_half")
    ap.add_argument("--dedupe-m", type=float, default=12.0)
    ap.add_argument("--min-conf", type=float, default=0.0, help="пол уверенности до сопоставления (диск)")
    ap.add_argument("--circle-m", type=float, default=None, help="радиус круга; по умолчанию из meta выгрузки")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    handles: dict[str, object] = {}

    def fh(fam):
        if fam not in handles:
            handles[fam] = open(out / f"{fam}.jsonl", "w", encoding="utf-8")
        return handles[fam]

    from expds.pipeline import ground_point

    npzs = sorted(Path(a.dump).glob("*.npz"))
    dump_regs = {str(np.load(z, allow_pickle=False)["region"]) for z in npzs}
    anchors, n_stack, groups = [], 0, Counter()
    for f, raw in iter_features(Path(a.stack)):
        pr = f.get("properties") or {}
        reg = str(pr.get("region_name", "default_source"))
        fh(site_family(reg)).write(raw + "\n")          # как split_stack_by_family: строка без изменений
        n_stack += 1
        g = str(pr.get("threshold_group") or pr.get("markup_type") or "")
        groups[g] += 1
        if g == a.group:
            raise SystemExit(f"в стеке уже есть группа {a.group}")
        if is_anchor(pr, dump_regs):
            # якорь хранится сразу точкой в метрах (ground_point); класс для отбора не важен
            anchors.append({"geom": ground_point(shape(f["geometry"])), "class_name": pr.get("class_name"),
                            "region_name": reg, "group": g})
    stats = {"stack": n_stack, "anchors": len(anchors), "anchors_by_group": dict(Counter(x["group"] for x in anchors)),
             "dedupe": "cross_class", "maps": 0, "regions": {}}
    new_n = 0
    for z in npzs:
        hf, meta = half_feats(z, a.circle_m, a.min_conf, a.group)
        reg = hf[0]["region_name"] if hf else z.stem.split("__")[0]
        kept = drop_near_any(hf, [x["geom"] for x in anchors if x["region_name"] == reg], a.dedupe_m)
        # справка: сколько осталось бы без якорей σ (в pipeline канал σ идёт ПОСЛЕ li_half и его не отсекает)
        n_wo_dev = len(drop_near_any(hf, [x["geom"] for x in anchors if x["region_name"] == reg and x["group"] != "li_dev"],
                                     a.dedupe_m))
        for f in kept:
            pr = {k: f[k] for k in ("class_name", "region_name", "markup_type", "threshold_group", "source", "original_crs")}
            pr.update(confidence=round(f["confidence"], 6), crs="EPSG:3857")
            fh(site_family(reg)).write(json.dumps({"type": "Feature", "geometry": mapping(f["geom"]), "properties": pr},
                                                  ensure_ascii=False) + "\n")
        st = stats["regions"].setdefault(reg, {"maps": 0, "half": 0, "new": 0, "new_wo_li_dev": 0, "mosaic": False,
                                               "dem_factor": []})
        st["maps"] += 1; st["half"] += len(hf); st["new"] += len(kept); st["new_wo_li_dev"] += n_wo_dev
        st["mosaic"] |= bool(meta.get("mosaic")); st["dem_factor"].append(float(meta.get("dem_info_factor", 1)))
        stats["maps"] += 1; new_n += len(kept)
    for h in handles.values():
        h.close()
    stats["new"] = new_n
    stats["stack_groups"] = dict(groups.most_common())
    (out / "_li_half_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: stats[k] for k in ("stack", "anchors", "maps", "new")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
