#!/usr/bin/env python3
"""Кривые (ClassCurve) стека по семьям, с малой памятью.

Вход — каталог JSONL по семьям (scripts/split_stack_by_family.py). Каждая семья
собирается пачками по --batch объектов (кэш геометрий чистится после пачки),
затем официальное жадное сопоставление expds.metric.match_collected.
Семьи не пересекаются в пространстве, поэтому сопоставление по семьям совпадает
с общим build_curves (проверка: score_counts in-sample == табло).

    python scripts/family_curves.py --fam work/fam --gt gt_train_v2.geojson --out work/curves
"""
from __future__ import annotations

import argparse
import gc
import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from expds import metric as M  # noqa: E402
from expds.naming import site_family  # noqa: E402

M._SHM_CACHE_DIR = Path("/proc/expds_no_cache")   # ничего не пишем в /dev/shm


def collect_batched(mod, calc, feats_iter, is_gt: bool, batch: int) -> dict:
    out: dict[str, list] = {}
    buf = []

    def flush():
        if not buf:
            return
        part = M._collect(mod, calc, {"type": "FeatureCollection", "features": list(buf)}, is_gt)
        for cls, lst in part.items():
            out.setdefault(cls, []).extend(lst)
        buf.clear()
        M._UTM_GEOM_CACHE.clear()

    for f in feats_iter:
        buf.append(f)
        if len(buf) >= batch:
            flush()
    flush()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fam", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=20000)
    ap.add_argument("--metric", default=None)
    ap.add_argument("--clusters", default=None,
                    help="JSON scripts/family_clusters.py: семьи с пересечением в пространстве сопоставлять вместе")
    ap.add_argument("--drop-groups", default="",
                    help="через запятую: threshold_group, которые выбросить ДО сопоставления (вариант стека без ветки)")
    a = ap.parse_args()
    mod = M.load_official(a.metric)
    drop = {g for g in a.drop_groups.split(",") if g}
    calc = mod.F1Calculator()
    gt_by_fam: dict[str, list] = {}
    for f in json.loads(Path(a.gt).read_text(encoding="utf-8"))["features"]:
        reg = str((f.get("properties") or {}).get("region_name", "default_source"))
        gt_by_fam.setdefault(site_family(reg), []).append(f)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fams = sorted(set(gt_by_fam) | {p.stem for p in Path(a.fam).glob("*.jsonl")})
    units = {f: [f] for f in fams}
    if a.clusters:
        for members in json.loads(Path(a.clusters).read_text(encoding="utf-8"))["clusters"].values():
            for m in members:
                units.pop(m, None)
            units["+".join(sorted(members))] = sorted(members)
    for fam, members in sorted(units.items()):
        dst = out / f"{fam}.pkl"
        if dst.exists():
            continue
        t0 = time.time()

        def it():
            for m in members:
                src = Path(a.fam) / f"{m}.jsonl"
                if not src.exists():
                    continue
                with open(src, encoding="utf-8") as fh:
                    for line in fh:
                        f = json.loads(line)
                        if drop and M.threshold_group(f.get("properties") or {}) in drop:
                            continue
                        yield f

        preds = collect_batched(mod, calc, it(), False, a.batch)
        gts = collect_batched(mod, calc, iter([f for m in members for f in gt_by_fam.get(m, [])]), True, a.batch)
        curves = M.match_collected(calc, preds, gts)
        del preds, gts
        tmp = dst.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(curves, protocol=pickle.HIGHEST_PROTOCOL))
        tmp.replace(dst)
        del curves
        gc.collect()
        print(f"{fam}\t{time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
