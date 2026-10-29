#!/usr/bin/env python3
"""OOF варианта C (pix_clean) без повторного счёта классики — для бокса с малой памятью.

Тот же результат, что geo_predict.py с [geophys.pixgbm] enabled=true, impl="clean":
классика участка считается один раз (geophys_feats при enabled=false) и кладётся в кэш,
кандидаты pix_clean — отдельно на модели своего фолда, сборка — pipeline.pix_merge
и тот же post/writer, что в geo_predict.py.

    pix_clean_oof.py classic --root TRAIN --config configs/v3_on.toml --cache C
    pix_clean_oof.py pix     --root TRAIN --config ... --set ... --model-dir M/F1_s0 --cache C --tag s0 --regions 200_...
    pix_clean_oof.py compose --root TRAIN --config ... --set ... --cache C --tag s0 --out C_s0.geojson
    (compose --tag "" — только классика, проверка против A.geojson)
"""
from __future__ import annotations

import argparse
import json
import pickle
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from geo_predict import apply_override  # noqa: E402

from expds import pix_clean, post, writer  # noqa: E402
from expds.catalog import scan_root  # noqa: E402
from expds.pipeline import TimeGuard, geophys_feats, load_config, pix_merge  # noqa: E402


def _cfg(args) -> dict:
    cfg = load_config(args.config)
    for spec in args.set:
        apply_override(cfg, spec)
    cfg.setdefault("geophys", {}).setdefault("pixgbm", {}).setdefault("budget_s", 10 ** 9)
    return cfg


def _regions(args):
    for reg in scan_root(args.root):
        if args.regions and reg.name not in args.regions:
            continue
        yield reg


def _peak_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def cmd_classic(args) -> None:
    cfg = _cfg(args)
    cfg["geophys"]["pixgbm"]["enabled"] = False
    d = Path(args.cache) / "classic"
    d.mkdir(parents=True, exist_ok=True)
    guard = TimeGuard(10 ** 9)
    for reg in _regions(args):
        f = d / f"{reg.name}.pkl"
        if f.exists():
            continue
        t = time.monotonic()
        feats = geophys_feats(reg, cfg, guard)
        f.with_suffix(".part").write_bytes(pickle.dumps(feats))
        f.with_suffix(".part").replace(f)
        print(f"classic {reg.name}: {len(feats)} за {time.monotonic() - t:.0f} с, пик {_peak_mb():.0f} МБ", flush=True)


def cmd_pix(args) -> None:
    cfg = _cfg(args)
    pix = cfg["geophys"]["pixgbm"]
    cls = set(pix.get("classes", ["finds_points", "object_poly"]))
    d = Path(args.cache) / "pix" / args.tag
    d.mkdir(parents=True, exist_ok=True)
    guard = TimeGuard(10 ** 9)
    for reg in _regions(args):
        f = d / f"{reg.name}.pkl"
        if f.exists():
            continue
        t = time.monotonic()
        pst: dict = {}
        new = [x for x in pix_clean.region_feats(reg, cfg, Path(args.model_dir), guard=guard, status=pst)
               if x["class_name"] in cls]
        f.with_suffix(".part").write_bytes(pickle.dumps({"new": new, "pst": pst, "model_dir": args.model_dir}))
        f.with_suffix(".part").replace(f)
        print(f"pix {args.tag} {reg.name}: {len(new)} за {time.monotonic() - t:.0f} с, пик {_peak_mb():.0f} МБ",
              flush=True)


def cmd_compose(args) -> None:
    cfg = _cfg(args)
    pix = cfg["geophys"]["pixgbm"]
    feats: list[dict] = []
    for reg in _regions(args):
        found = pickle.loads((Path(args.cache) / "classic" / f"{reg.name}.pkl").read_bytes())
        if args.tag:
            pf = Path(args.cache) / "pix" / args.tag / f"{reg.name}.pkl"
            rec = pickle.loads(pf.read_bytes())
            cls = set(pix.get("classes", ["finds_points", "object_poly"]))
            maps = pix.get("maps", list(pix_clean.METHODS))
            maps = [maps] if isinstance(maps, str) else maps
            # классы — как в geophys_feats, методы (maps) — как в region_feats: без кандидатов метода
            # его классика остаётся (ключ не попадает в have у pix_merge)
            new = [x for x in rec["new"] if x["class_name"] in cls and x["markup_type"] in maps]
            found = pix_merge(found, new, rec["pst"], pix, reg.name, pix_clean)
        feats += found
    out = post.finalize(post.merge_within_markup(feats))
    fc = writer.to_feature_collection(out)
    for feat, src in zip(fc["features"], out):
        for key in ("threshold_group", "source"):
            if src.get(key):
                feat["properties"][key] = src[key]
    Path(args.out).write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")
    print(f"записано {len(out)} объектов → {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["classic", "pix", "compose"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.toml"))
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--regions", nargs="*", default=None)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--model-dir")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out")
    args = ap.parse_args()
    {"classic": cmd_classic, "pix": cmd_pix, "compose": cmd_compose}[args.cmd](args)


if __name__ == "__main__":
    main()
