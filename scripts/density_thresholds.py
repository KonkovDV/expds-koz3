#!/usr/bin/env python3
"""Вариант раздела 5: порог группы там, где на трейне выдаётся в factor раз больше объектов.

На площадке кандидаты лидара и снимков попадают чаще, чем на трейне при тех же порогах.
Табло трейна этого не видит: там порог уже в оптимуме. Вариант фиксируется до посылки по
кривым трейна, одно число на группу, без перебора по публичному баллу. В финал — только
версией B.

    python scripts/density_thresholds.py --stack F:/expds3/work/stack_base.geojson \
        --add F:/expds3/work/gor_phys.geojson:gor_phys --factor 1.5 --out F:/expds3/work/density_1p5.json

Стек читается потоком: из каждого объекта берутся только свойства, геометрия не разбирается.
Геофизика не трогается: её недодачу меряет перепись, а не порог.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Группа табло → таблица `[thresholds_by_markup.*]` сдачи, если имена расходятся.
BOARD_TO_CONFIG = {"sp_focal": "sp_or", "sp_arch": "sp_or", "ae_fort": "ae"}
GEO = {"maggr", "etgr", "terfl"}
_PROPS = re.compile(rb'"properties":\s*\{')


def iter_props(path: Path):
    """Свойства каждого объекта GeoJSON без разбора геометрии."""
    data = Path(path).read_bytes()
    for m in _PROPS.finditer(data):
        start = m.end() - 1
        end = data.find(b"}", start)
        while True:
            try:
                yield json.loads(data[start:end + 1])
                break
            except json.JSONDecodeError:
                end = data.find(b"}", end + 1)
                if end < 0:
                    raise


def table_of(props: dict) -> str:
    group = props.get("threshold_group") or props.get("markup_type") or ""
    return BOARD_TO_CONFIG.get(group, group)


def collect(stack: Path, extra: list[tuple[Path, str]]) -> dict[tuple[str, str], list[float]]:
    conf: dict[tuple[str, str], list[float]] = defaultdict(list)
    for p in iter_props(stack):
        conf[(table_of(p), p.get("class_name"))].append(float(p.get("confidence", 1.0)))
    for path, group in extra:
        have = any(t == group for t, _ in conf)
        if have:
            continue
        for p in iter_props(path):
            conf[(group, p.get("class_name"))].append(float(p.get("confidence", 1.0)))
    return conf


def density_variant(conf: dict[tuple[str, str], list[float]], by_markup: dict, factor: float,
                    skip: set[str] = GEO) -> dict:
    """Порог, при котором объектов группы на трейне в factor раз больше, чем при пороге сдачи.

    Кандидатов на трейне меньше нужного — порог опускается до самого слабого кандидата.
    Группа, которая сейчас не выдаёт ничего, остаётся как есть: выключена намеренно.
    """
    out: dict = {}
    for table, classes in sorted(by_markup.items()):
        if table in skip or not isinstance(classes, dict):
            continue
        for cls, tau in sorted(classes.items()):
            vals = sorted(conf.get((table, cls), ()), reverse=True)
            kept = sum(v >= tau for v in vals)
            rec = {"tau": tau, "kept": kept, "candidates": len(vals)}
            if kept == 0:
                rec.update(tau_new=tau, kept_new=0, note="выключено или нет кандидатов")
            else:
                want = math.ceil(factor * kept)
                cut = vals[min(want, len(vals)) - 1]
                tau_new = math.floor(cut * 1e6) / 1e6
                rec.update(tau_new=tau_new, kept_new=sum(v >= tau_new for v in vals),
                           short=len(vals) < want)
            out.setdefault(table, {})[cls] = rec
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stack", required=True, help="стек табло до порогов (scoreboard.py --dump-stack)")
    ap.add_argument("--add", action="append", default=[],
                    help="путь:группа — группа, которой нет в дампе стека (например, gor_phys)")
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.toml"))
    ap.add_argument("--factor", type=float, default=1.5)
    ap.add_argument("--board", default=None, help="JSON табло с per_class: сверка «оставлено» по группам")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    extra = []
    for spec in args.add:
        path, group = spec.rsplit(":", 1)
        extra.append((Path(path), group))
    cfg = tomllib.loads(Path(args.config).read_text(encoding="utf-8"))
    conf = collect(Path(args.stack), extra)
    variant = density_variant(conf, cfg.get("thresholds_by_markup", {}), args.factor)
    check = {}
    if args.board:
        per = json.loads(Path(args.board).read_text(encoding="utf-8")).get("per_class", {})
        for cls, rec in per.items():
            for group, g in rec.get("groups", {}).items():
                mine = variant.get(BOARD_TO_CONFIG.get(group, group), {}).get(cls)
                if mine is not None:
                    check[f"{group}/{cls}"] = {"board_kept": g["kept"], "stack_kept": mine["kept"],
                                               "board_tau": g["threshold"], "config_tau": mine["tau"]}
    report = {"factor": args.factor, "stack": str(args.stack), "variant": variant, "board_check": check,
              "groups_in_stack": sorted({t for t, _ in conf})}
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    for table, classes in variant.items():
        for cls, r in classes.items():
            print(f"{table:11s} {cls:24s} {r['tau']:.6f} → {r['tau_new']:.6f}  "
                  f"оставлено {r['kept']} → {r['kept_new']} из {r['candidates']}", flush=True)
    for k, v in check.items():
        flag = "" if v["board_kept"] == v["stack_kept"] else "  РАСХОЖДЕНИЕ"
        print(f"сверка {k}: табло {v['board_kept']}, стек {v['stack_kept']}{flag}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
