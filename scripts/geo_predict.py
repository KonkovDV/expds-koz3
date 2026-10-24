#!/usr/bin/env python3
"""Файл геофизики для табло тем же кодом, что в сдаче (pipeline.geophys_feats).

Табло (`scripts/scoreboard.py --geo`) берёт геофизику готовым файлом. Здесь он
собирается по всем участкам корня с конфигом сдачи и правками поверх него.
Без правок файл обязан дать базу табло (`--expect 0.350588`): это проверка,
что сборка та же, что у `geo_cur.geojson`.

    python scripts/geo_predict.py --root F:/expds3/train-expds-3 --out F:/expds3/work/geo_default.geojson
    python scripts/geo_predict.py --root F:/expds3/train-expds-3 --out F:/expds3/work/geo_k075.geojson \
        --set geophys.maggr.finds_points.peak_k=0.75 --set geophys.maggr.finds_points.min_confidence=0.2

Значение в --set — литерал TOML: числа как есть, строки в кавычках
(--set 'geophys.band_mode="auto"'). Порог выдачи min_confidence после смены
отсечки или каналов берётся из того же прогона geo_tune.py (поле tau).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from expds import post, writer
from expds.catalog import scan_root
from expds.pipeline import TimeGuard, geophys_feats, load_config


def apply_override(cfg: dict, spec: str) -> dict:
    """«a.b.c=литерал TOML» → cfg["a"]["b"]["c"] = значение. Промежуточные таблицы создаются."""
    key, sep, raw = spec.partition("=")
    if not sep or not key.strip():
        raise ValueError(f"нужно ключ=значение: {spec!r}")
    value = tomllib.loads(f"v = {raw.strip()}")["v"]
    node = cfg
    parts = key.strip().split(".")
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = node[part] = {}
        node = nxt
    node[parts[-1]] = value
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="корень трейна (или любой корень с участками)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.toml"))
    ap.add_argument("--set", action="append", default=[], help="правка конфига: ключ.через.точки=литерал TOML")
    ap.add_argument("--regions", nargs="*", default=None, help="только эти участки")
    args = ap.parse_args()

    cfg = load_config(args.config)
    for spec in args.set:
        apply_override(cfg, spec)
    guard = TimeGuard(10 ** 9)
    feats: list[dict] = []
    t0 = time.monotonic()
    for reg in scan_root(args.root):
        if args.regions and reg.name not in args.regions:
            continue
        found = geophys_feats(reg, cfg, guard)
        if found:
            print(f"{reg.name}: {len(found)}", flush=True)
        feats += found
    out = post.finalize(post.merge_within_markup(feats))
    # Табло читает группу порога из свойств: копии по слоям идут своей группой (etgr_copy).
    fc = writer.to_feature_collection(out)
    for feat, src in zip(fc["features"], out):
        for key in ("threshold_group", "source"):
            if src.get(key):
                feat["properties"][key] = src[key]
    Path(args.out).write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")
    print(f"записано {len(out)} объектов за {time.monotonic() - t0:.0f} с → {args.out}")


if __name__ == "__main__":
    main()
