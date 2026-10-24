#!/usr/bin/env python3
"""Опыты без GPU: преобразование готовых предсказаний вне обучения для табло.

Табло (`scripts/scoreboard.py`) принимает файлы через `--add путь:группа` и
`--replace группа=путь`. Здесь из уже посчитанных OOF-файлов собираются варианты
постобработки, которые идут в конфиг без переобучения:

  copy           копии объектов одного слоя в другой в покрытии снимков участка
                 (то же, что [[fusion.copy]] на инференсе);
  core           замена полигонов «ядром» — кругом в полюсе недоступности
                 (то же, что [post.core]);
  split          крупные полигоны режутся сеткой на куски размером с эталон
                 (то же, что [post.split]): одна маска на сеть траншей иначе
                 засчитывает не больше одного окопа;
  fill-holes     заливка дыр у полигонов класса ([post] fill_holes);
  merge-classes  классы из файла A, остальные из B (ансамбль только по курганам,
                 [models] … ensemble_classes);
  pick           объекты выбранных типов снимков (и классов): файл общей сети
                 (scripts/cache_merge.py) делится по группам порога, как в сдаче.

Примеры (пути — как в docs/AGENT_PLAN.md):

    python scripts/oof_transform.py copy --train F:/expds3/train-expds-3 \
        --src F:/expds3/work/fusion/li_hard2_fused.geojson --src-markup li --dst sp_or \
        --classes kurgany_povrezhdennye kurgany_tselye --group li_sp --out F:/expds3/work/li_sp_copy.geojson
    python scripts/scoreboard.py ... --add F:/expds3/work/li_sp_copy.geojson:li_sp --out notes/li_sp_2026-10-24.json

    python scripts/oof_transform.py core --src F:/expds3/work/spor_gor_oof.geojson \
        --classes gorodishcha --frac 0.3 --min-r 8 --out F:/expds3/work/spor_gor_core.geojson
    python scripts/scoreboard.py ... --replace sp_or=F:/expds3/work/spor_gor_core.geojson

Уверенность копий не меняется: порог новой группы подбирает табло. Файлы
предсказаний — конкурсные данные, в git их не класть.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_feats(path: Path) -> list[dict]:
    """GeoJSON → объекты пайплайна (shapely в EPSG:3857 + свойства)."""
    from shapely.geometry import shape

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for f in data.get("features", []):
        p = dict(f.get("properties") or {})
        try:
            g = shape(f["geometry"])
        except Exception:
            continue
        out.append({**p, "geom": g, "confidence": float(p.get("confidence", 1.0)),
                    "class_name": p.get("class_name")})
    return out


def write_feats(feats: list[dict], path: Path) -> None:
    """Объекты → GeoJSON со всеми свойствами (группа порога сохраняется)."""
    from shapely.geometry import mapping

    out = []
    for f in feats:
        props = {k: v for k, v in f.items() if k != "geom" and isinstance(v, (str, int, float, bool, type(None)))}
        out.append({"type": "Feature", "properties": props, "geometry": mapping(f["geom"])})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"type": "FeatureCollection", "features": out}, ensure_ascii=False),
                          encoding="utf-8")
    print(f"записано {len(out)} объектов → {path}")


def cmd_copy(a) -> None:
    from expds.catalog import scan_root
    from expds.fusion import CopyRule, copy_layers

    feats = [f for src in a.src for f in read_feats(Path(src))]
    rasters = {r.name: r.rasters for r in scan_root(a.train)}
    rule = CopyRule(src=a.src_markup, dst=a.dst, group=a.group, min_conf={c: a.min for c in a.classes},
                    src_groups=tuple(a.src_groups))
    by_region: dict[str, list[dict]] = defaultdict(list)
    for f in feats:
        by_region[f.get("region_name")].append(f)
    out = []
    for region, items in by_region.items():
        rs = rasters.get(region)
        if rs and rs.get(a.dst):
            out += copy_layers(items, rs, [rule])
    print(f"участков с {a.dst}: {sum(1 for r in by_region if (rasters.get(r) or {}).get(a.dst))}, копий {len(out)}")
    write_feats(out, Path(a.out))


def cmd_core(a) -> None:
    from expds.post import shape_cores

    spec = {c: {"frac": a.frac, "min_r_m": a.min_r, "groups": a.groups or None} for c in a.classes}
    feats = read_feats(Path(a.src))
    write_feats(shape_cores(feats, spec), Path(a.out))


def cmd_split(a) -> None:
    from expds.post import split_large

    spec = {c: {"cell_m": a.cell, "min_area_m2": a.min_area, "max_area_m2": a.max_area,
                "groups": a.groups or None} for c in a.classes}
    write_feats(split_large(read_feats(Path(a.src)), spec), Path(a.out))


def cmd_fill(a) -> None:
    from expds.post import fill_holes

    write_feats(fill_holes(read_feats(Path(a.src)), a.classes), Path(a.out))


def cmd_merge(a) -> None:
    keep = set(a.classes)
    fa = [f for f in read_feats(Path(a.a)) if f["class_name"] in keep]
    fb = [f for f in read_feats(Path(a.b)) if f["class_name"] not in keep]
    write_feats(fa + fb, Path(a.out))


def cmd_pick(a) -> None:
    keep_mt = set(a.markups)
    keep_cls = set(a.classes or [])
    feats = [f for f in read_feats(Path(a.src))
             if f.get("markup_type") in keep_mt and (not keep_cls or f["class_name"] in keep_cls)]
    write_feats(feats, Path(a.out))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("copy", help="копии слоя src в слой dst в покрытии снимков")
    c.add_argument("--train", required=True, help="корень трейна (снимки участков)")
    c.add_argument("--src", nargs="+", required=True, help="OOF-файлы источника")
    c.add_argument("--src-markup", default="li")
    c.add_argument("--src-groups", nargs="*", default=["", "li", "li_coarse", "li_mosaic", "li_mask"])
    c.add_argument("--dst", required=True, help="sp_or | or | ae")
    c.add_argument("--classes", nargs="+", required=True)
    c.add_argument("--min", type=float, default=0.0, help="отсечка уверенности источника (порог подберёт табло)")
    c.add_argument("--group", required=True)
    c.add_argument("--out", required=True)
    c.set_defaults(fn=cmd_copy)
    k = sub.add_parser("core", help="полигоны → ядра (круг в полюсе недоступности)")
    k.add_argument("--src", required=True)
    k.add_argument("--classes", nargs="+", required=True)
    k.add_argument("--frac", type=float, default=0.3, help="доля площади полигона")
    k.add_argument("--min-r", type=float, default=0.0, help="минимальный радиус на земле, м")
    k.add_argument("--groups", nargs="*", default=[])
    k.add_argument("--out", required=True)
    k.set_defaults(fn=cmd_core)
    sp = sub.add_parser("split", help="крупные полигоны → куски размером с эталон (сетка cell_m)")
    sp.add_argument("--src", required=True)
    sp.add_argument("--classes", nargs="+", required=True)
    sp.add_argument("--cell", type=float, default=30.0, help="сторона клетки на земле, м")
    sp.add_argument("--min-area", type=float, default=20.0, help="куски мельче, м², отбрасываются")
    sp.add_argument("--max-area", type=float, default=1500.0, help="резать только полигоны крупнее, м²")
    sp.add_argument("--groups", nargs="*", default=[])
    sp.add_argument("--out", required=True)
    sp.set_defaults(fn=cmd_split)
    h = sub.add_parser("fill-holes")
    h.add_argument("--src", required=True)
    h.add_argument("--classes", nargs="+", required=True)
    h.add_argument("--out", required=True)
    h.set_defaults(fn=cmd_fill)
    m = sub.add_parser("merge-classes", help="классы из A, остальное из B")
    m.add_argument("--a", required=True)
    m.add_argument("--b", required=True)
    m.add_argument("--classes", nargs="+", required=True)
    m.add_argument("--out", required=True)
    m.set_defaults(fn=cmd_merge)
    pk = sub.add_parser("pick", help="только объекты этих типов снимков (и классов)")
    pk.add_argument("--src", required=True)
    pk.add_argument("--markups", nargs="+", required=True)
    pk.add_argument("--classes", nargs="*", default=[])
    pk.add_argument("--out", required=True)
    pk.set_defaults(fn=cmd_pick)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
