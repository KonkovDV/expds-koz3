#!/usr/bin/env python3
"""Какой класс геофизики не сработал на площадке: находки или object_poly.

Лог площадки печатает TP/FP геофизики только суммой двух классов. Посылка, в которой
была одна геофизика (23.10: балл 0,0614, геофизика TP 373, FP 197, эталон 1441),
даёт ещё одно уравнение: F1(object_poly) + F1(finds_points) = 0,0614 · W / 10, где W —
сумма весов классов, присутствующих в эталоне выборки (100, если есть все семь).

Числа по умолчанию — самый мелкий набор, который сходится с напечатанной долей ложных, а не единственная перепись лога. Для другого набора передайте `--geo-tp`, `--geo-fp` и `--geo-gt`.

Чистая арифметика оставляет два режима (промежутка нет):
  A — находки дали < 100 TP (почти всё — object_poly, эталон object_poly ≥ 743);
  B — находки дали ≥ 300 TP (object_poly почти пуст, эталон object_poly ≤ 698).
По одному логу их не различить. Различает счёт объектов в своём файле предсказаний
той же версии, что оценивала площадка: сколько выдано находок и сколько object_poly.

    python scripts/geo_regime.py --pred F:/expds3/work/public_result.geojson
    python scripts/geo_regime.py --pred ... --geo-tp 373 --geo-fp 197 --geo-gt 1441 --geo-only-score 0.0614

Файл предсказаний не копируется и в notes/ не пишется; печатаются только счётчики.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

GEO = ("maggr", "etgr", "terfl")


def census(path: Path) -> dict:
    """Счётчики объектов геофизики: класс → число, (класс, слой) → число."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    by_cls: Counter = Counter()
    by_pair: Counter = Counter()
    for f in data.get("features", []):
        p = f.get("properties") or {}
        mt = str(p.get("markup_type") or "").lower()
        if mt not in GEO:
            continue
        by_cls[p.get("class_name")] += 1
        by_pair[(p.get("class_name"), mt)] += 1
    return {"by_class": dict(by_cls), "by_pair": {f"{c}/{m}": n for (c, m), n in sorted(by_pair.items())}}


def feasible(tp: int, fp: int, gt: int, f1_sum: float, tol: float = 6e-4,
             n_op: int | None = None, n_f: int | None = None) -> dict:
    """Все (эталон op, TP op, FP op), согласные с суммой F1. С n_op и n_f — ещё и со счётом выданного."""
    a = np.arange(0, gt + 1, dtype=np.float32)[:, None, None]
    t = np.arange(0, tp + 1, dtype=np.float32)[None, :, None]
    f = np.arange(0, fp + 1, dtype=np.float32)[None, None, :]
    b, tf, ff = gt - a, tp - t, fp - f
    with np.errstate(divide="ignore", invalid="ignore"):
        f_op = np.where(a + t + 0.5 * f > 0, 2 * t / np.maximum(a + t + 0.5 * f, 1e-12), 0.0)
        f_f = np.where(b + tf + 0.5 * ff > 0, 2 * tf / np.maximum(b + tf + 0.5 * ff, 1e-12), 0.0)
    ok = (np.abs(f_op + f_f - f1_sum) < tol) & (tf <= b) & (t <= a)
    if n_op is not None:
        ok &= (t + f) == n_op
    if n_f is not None:
        ok &= (tf + ff) == n_f
    ai, ti, _ = np.nonzero(ok)
    if ai.size == 0:
        return {"solutions": 0}
    t_f = tp - ti
    return {"solutions": int(ai.size),
            "finds_tp": [int(t_f.min()), int(t_f.max())],
            "op_tp": [int(ti.min()), int(ti.max())],
            "op_gt": [int(ai.min()), int(ai.max())],
            "finds_gt": [int(gt - ai.max()), int(gt - ai.min())],
            "f1_op": [round(float(f_op[ok].min()), 3), round(float(f_op[ok].max()), 3)],
            "f1_finds": [round(float(f_f[ok].min()), 3), round(float(f_f[ok].max()), 3)],
            "finds_tp_bands": {"<100": int(np.sum(t_f < 100)), "100-299": int(np.sum((t_f >= 100) & (t_f < 300))),
                               ">=300": int(np.sum(t_f >= 300))}}


def verdict(sol: dict) -> str:
    if not sol.get("solutions"):
        return "решений нет: файл не той версии, что оценивала площадка, или в эталоне выборки не все классы (--weights-present)"
    lo, hi = sol["finds_tp"]
    if sol["finds_tp_bands"]["100-299"] == 0 and sol["finds_tp_bands"][">=300"] == 0:
        return (f"режим A: находки дали {lo}–{hi} TP при эталоне находок {sol['finds_gt'][0]}–{sol['finds_gt'][1]}. "
                "Находки на площадке не работают — scripts/geo_stages.py по публичной выборке покажет, где они теряются")
    if sol["finds_tp_bands"]["<100"] == 0 and sol["finds_tp_bands"]["100-299"] == 0:
        return (f"режим B: находки работают ({lo}–{hi} TP, F1 {sol['f1_finds'][0]}–{sol['f1_finds'][1]}), "
                f"object_poly слаб (F1 {sol['f1_op'][0]}–{sol['f1_op'][1]}, эталон {sol['op_gt'][0]}–{sol['op_gt'][1]}). "
                "Ставка — object_poly: профили и срезы EtGr, TerFl, точка внутри")
    return "оба режима ещё возможны: нужен счёт выданного (--pred) той же версии"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", default=None, help="свой result.geojson публичной выборки той же версии")
    ap.add_argument("--geo-tp", type=int, default=373)
    ap.add_argument("--geo-fp", type=int, default=197)
    ap.add_argument("--geo-gt", type=int, default=1441)
    ap.add_argument("--geo-only-score", type=float, default=0.0614, help="балл посылки, где была одна геофизика")
    ap.add_argument("--weights-present", type=float, default=100.0,
                    help="сумма весов классов, присутствующих в эталоне выборки")
    ap.add_argument("--tol", type=float, default=6e-4, help="допуск суммы F1 (балл напечатан с 4–5 знаками)")
    args = ap.parse_args()

    f1_sum = args.geo_only_score * args.weights_present / 10.0
    out: dict = {"f1_sum": round(f1_sum, 4)}
    n_op = n_f = None
    if args.pred:
        c = census(Path(args.pred))
        out["census"] = c
        n_op = int(c["by_class"].get("object_poly", 0))
        n_f = int(c["by_class"].get("finds_points", 0))
        total = n_op + n_f
        out["census_matches_log"] = total == args.geo_tp + args.geo_fp
        if not out["census_matches_log"]:
            out["warning"] = (f"в файле {total} объектов геофизики, а в логе TP+FP = {args.geo_tp + args.geo_fp}: "
                              "это не та версия или не тот прогон; ниже — без счёта выданного")
            n_op = n_f = None
    sol = feasible(args.geo_tp, args.geo_fp, args.geo_gt, f1_sum, args.tol, n_op, n_f)
    out["feasible"] = sol
    out["verdict"] = verdict(sol)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
