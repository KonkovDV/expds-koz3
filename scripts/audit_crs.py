#!/usr/bin/env python3
"""Аудит CRS участков трейна: UTM.json против заголовка LAS и голоса растров.

Для каждого участка с LAS печатает обе CRS, их разбор, правдоподобие, голос
растров и — главное — сколько эталонов li попадает в рамку облака при каждой
CRS. Правильная та, при которой рамка облака накрывает разметку лидара. Итог
решает [crs] policy в конфиге: если в трейне UTM.json хотя бы раз неправ, а
голос растров это видит, — "vote"; если голос молчит — смотреть по таблице.

    python scripts/audit_crs.py --train F:/expds3/train-expds-3 \
        --gt F:/expds3/work/gt_train_v2.geojson --out notes/crs_audit_2026-10-24.json

Имена участков трейна в notes/ допустимы (как в остальных логах), выборки — нет.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def gt_points(path: Path) -> dict[str, list[tuple[float, float]]]:
    """Центры эталонов li по участкам (EPSG:3857)."""
    from shapely.geometry import shape

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, list] = defaultdict(list)
    for f in data.get("features", []):
        p = f.get("properties") or {}
        if p.get("markup_type") not in ("li", None):
            continue
        try:
            c = shape(f["geometry"]).representative_point()
        except Exception:
            continue
        out[p.get("region_name")].append((c.x, c.y))
    return out


def inside(box3857: tuple | None, pts: list) -> int:
    if not box3857:
        return 0
    x0, y0, x1, y1 = box3857
    return sum(1 for x, y in pts if x0 <= x <= x1 and y0 <= y <= y1)


def main() -> None:
    from expds.catalog import read_utm_json, scan_root
    from expds.crs import _box_in, _overlap_votes, _same, las_box, las_crs, plausible, resolve_region_crs, valid_crs

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--gt", default=None, help="эталон в EPSG:3857 (gt_train_v2.geojson)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    gts = gt_points(Path(a.gt)) if a.gt else {}
    rows = []
    for reg in scan_root(a.train):
        if not reg.las:
            continue
        raw_utm = read_utm_json(reg.utm_json)
        utm = valid_crs(raw_utm)
        las = valid_crs(las_crs(reg.las[0]))
        box = las_box(reg.las[0])
        rasters = [r for files in reg.rasters.values() for r in files]
        row = {"region": reg.name, "utm_json": raw_utm, "las": las, "same": _same(utm, las),
               "las_files": len(reg.las), "rasters": len(rasters)}
        for name, crs in (("utm_json", utm), ("las", las)):
            if not crs:
                continue
            b3857 = _box_in(crs, "EPSG:3857", box) if box else None
            row[f"{name}_plausible"] = plausible(crs, box)
            row[f"{name}_votes"] = _overlap_votes(crs, box, rasters) if box else None
            row[f"{name}_gt_inside"] = inside(b3857, gts.get(reg.name, []))
        row["gt_li"] = len(gts.get(reg.name, []))
        for policy in ("utm_json", "las", "vote"):
            row[f"pick_{policy}"] = resolve_region_crs(raw_utm, reg.las, rasters, policy)
        rows.append(row)
        flag = "" if row["same"] or not (utm and las) else "  ← РАСХОЖДЕНИЕ"
        print(f"{reg.name:40s} UTM.json {utm}  LAS {las}  эталонов li {row['gt_li']}{flag}", flush=True)
    bad = [r for r in rows if not r["same"] and r.get("utm_json_gt_inside") is not None
           and r.get("las_gt_inside") is not None]
    summary = {
        "regions_with_las": len(rows),
        "no_crs": sum(1 for r in rows if not r["las"] and not valid_crs(r["utm_json"])),
        "utm_json_invalid": sum(1 for r in rows if r["utm_json"] and not valid_crs(r["utm_json"])),
        "conflicts": len(bad),
        "conflicts_where_las_is_right": sum(1 for r in bad if r["las_gt_inside"] > r["utm_json_gt_inside"]),
        "conflicts_where_vote_is_right": sum(
            1 for r in bad if (r["pick_vote"] == r["las"]) == (r["las_gt_inside"] > r["utm_json_gt_inside"])),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps({"summary": summary, "regions": rows}, ensure_ascii=False, indent=2),
                               encoding="utf-8")


if __name__ == "__main__":
    main()
