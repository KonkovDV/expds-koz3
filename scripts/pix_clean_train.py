#!/usr/bin/env python3
"""Обучение моделей pix_clean (чистая перепись PIXGBM, docs/PIXGBM_SPEC_CLEANROOM.md).

1) Выборка пикселей по растрам 200–217 (кэш .npz на растр):
    python scripts/pix_clean_train.py sample --root TRAIN --cache CACHE
2) Модели (все участки или без фолда):
    python scripts/pix_clean_train.py fit --cache CACHE --out models/pixgbm/clean --seed 0 [--exclude 200 201 ...]

Метка: finds_points — пиксели ≤ 0,4 м от GT-точки (кольцо 0,4–1,0 м не учится); object_poly —
внутри GT-полигона своего слоя (кольцо 0,3 м вокруг не учится). Фон — недовыборка: половина
случайно, половина «трудные» (∝ квадрату сигнала).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from expds import pix_clean as pc  # noqa: E402

POS_R_M = 0.4
IGNORE_R_M = 1.0
POLY_RING_M = 0.3
POS_MAX = {"finds_points": 15000, "object_poly": 8000}
NEG_RATIO = 3
NEG_MIN = 3000
LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  max_bin=255, verbose=-1, deterministic=True, force_row_wise=True)
NUM_ROUNDS = 400
TOKEN = {"maggr": "MagGr", "etgr": "EtGr", "terfl": "TerFl"}


def layer_suffix(stem: str, mt: str) -> str | None:
    """Хвост имени после кода метода: «профиль_14», «карта_05_1.6», «цвет_2», «серый», ""."""
    stem = re.sub(r"_(ObjectPoly|FindsPoints)$", "", stem, flags=re.I)
    m = None
    for m in re.finditer(TOKEN[mt], stem, flags=re.I):
        pass
    if m is None:
        return None
    return stem[m.end():].lstrip("_")


def gt_files_for(raster: Path, reg, cls: str, mt: str) -> list[Path]:
    """Файлы разметки того же класса, метода и слоя, что и растр."""
    cand = [p for p, c, t in reg.annotations if c == cls and t == mt]
    if not cand:
        return []
    rs = layer_suffix(raster.stem, mt)
    exact = [p for p in cand if layer_suffix(p.stem, mt) == rs]
    if exact:
        return exact
    generic = [p for p in cand if layer_suffix(p.stem, mt) in ("", None)]
    if mt == "etgr":
        return []          # слои EtGr — отдельные эталоны: без своего файла растр не учится
    return generic


def labels(p: pc.Prep, geoms: list, cls: str) -> np.ndarray:
    """1 — объект, 0 — фон, 255 — не учится."""
    from rasterio.features import rasterize

    lab = np.zeros(p.shape, np.uint8)
    if not geoms:
        return lab
    if cls == "finds_points":
        pos = [g.centroid.buffer(POS_R_M, 8) for g in geoms]
        ign = [g.centroid.buffer(IGNORE_R_M, 8) for g in geoms]
    else:
        pos = list(geoms)
        ign = [g.buffer(POLY_RING_M, 4) for g in geoms]
    r_ign = rasterize([(g, 1) for g in ign], out_shape=p.shape, transform=p.transform, all_touched=True, dtype="uint8")
    r_pos = rasterize([(g, 1) for g in pos], out_shape=p.shape, transform=p.transform, all_touched=False, dtype="uint8")
    if cls == "finds_points" and not r_pos.any():
        pass
    # мелкие объекты меньше пикселя: хотя бы центральный пиксель
    from expds.raster import pixel_to_xy  # noqa: F401
    inv = ~p.transform
    for g in geoms:
        c = g.centroid if cls == "finds_points" else g.representative_point()
        col, row = inv * (c.x, c.y)
        r, k = int(row), int(col)
        if 0 <= r < p.shape[0] and 0 <= k < p.shape[1]:
            r_pos[r, k] = 1
    lab[r_ign > 0] = 255
    lab[r_pos > 0] = 1
    return lab


def _seed_of(name: str, seed: int) -> int:
    return int(hashlib.sha256(f"{name}|{seed}".encode()).hexdigest()[:8], 16)


def sample_raster(p: pc.Prep, lab: np.ndarray, cls: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(_seed_of(p.path.name + cls, seed))
    pos = np.flatnonzero((lab == 1).ravel() & p.valid.ravel())
    neg = np.flatnonzero((lab == 0).ravel() & p.valid.ravel())
    if len(pos) > POS_MAX[cls]:
        pos = rng.choice(pos, POS_MAX[cls], replace=False)
    n_neg = min(len(neg), max(NEG_MIN, NEG_RATIO * len(pos)))
    if n_neg <= 0:
        return pos, np.array([], np.int64)
    half = n_neg // 2
    rnd = rng.choice(neg, half, replace=False)
    s = (np.abs(p.hp.ravel()[neg]) + np.abs(p.luma.ravel()[neg])) ** 2 + 1e-3
    rest = np.setdiff1d(neg, rnd, assume_unique=True)
    s = (np.abs(p.hp.ravel()[rest]) + np.abs(p.luma.ravel()[rest])) ** 2 + 1e-3
    k = min(len(rest), n_neg - half)
    hard = rng.choice(rest, k, replace=False, p=s / s.sum()) if k > 0 else np.array([], np.int64)
    return np.sort(pos), np.sort(np.concatenate([rnd, hard]))


def cmd_sample(args) -> None:
    from expds.catalog import scan_root
    from expds.crs import WEB_MERCATOR, to_crs
    from expds.gt import read_annotation
    from expds.pipeline import region_crs

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    index = []
    for reg in scan_root(args.root):
        num = reg.name.split("_")[0]
        if args.regions and num not in args.regions:
            continue
        crs_region = region_crs(reg, {})
        for mt in pc.METHODS:
            for path in sorted(reg.rasters.get(mt, []) or [], key=lambda f: f.name):
                todo = {}
                for cls in ("finds_points", "object_poly"):
                    if mt not in pc.CLASS_METHODS[cls]:
                        continue
                    files = gt_files_for(path, reg, cls, mt)
                    if files:
                        todo[cls] = files
                if not todo:
                    print(f"  {reg.name} {path.name}: нет разметки своего слоя — пропуск", flush=True)
                    continue
                out = cache / f"{hashlib.sha256(str(path).encode()).hexdigest()[:16]}_s{args.seed}_v{pc.FEATURE_VERSION}.npz"
                if out.exists():
                    index.append(json.loads(out.with_suffix(".json").read_text(encoding="utf-8")))
                    continue
                t = time.monotonic()
                p = pc.prepare(path, mt, crs_region)
                arrays, info = {}, {"raster": path.name, "region": reg.name, "mt": mt, "file": out.name,
                                    "shape": list(p.shape), "step": p.step, "classes": {}}
                for cls, files in todo.items():
                    geoms = []
                    for f in files:
                        for it in read_annotation(f, cls, mt, reg):
                            geoms.append(to_crs(it["geom"], WEB_MERCATOR, p.crs) if p.crs else it["geom"])
                    lab = labels(p, geoms, cls)
                    pos, neg = sample_raster(p, lab, cls, args.seed)
                    idx = np.concatenate([pos, neg])
                    rows, cols = np.unravel_index(idx, p.shape)
                    X = pc.features_at(p, rows, cols)
                    y = np.concatenate([np.ones(len(pos), np.uint8), np.zeros(len(neg), np.uint8)])
                    arrays[f"X_{cls}"], arrays[f"y_{cls}"] = X, y
                    info["classes"][cls] = {"gt_files": [f.name for f in files], "n_gt": len(geoms),
                                            "pos": int(len(pos)), "neg": int(len(neg))}
                np.savez_compressed(out, **arrays)
                info["secs"] = round(time.monotonic() - t, 1)
                out.with_suffix(".json").write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
                index.append(info)
                print(f"  {reg.name} {path.name}: {info['classes']} {info['secs']} с", flush=True)
    (cache / f"index_s{args.seed}.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")


def fit(cache: Path, out: Path, seed: int, exclude: set[str], threads: int, sample_seed: int = 0,
        rounds: int = NUM_ROUNDS) -> dict:
    import lightgbm as lgb

    index = json.loads((cache / f"index_s{sample_seed}.json").read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    meta = {"feature_version": pc.FEATURE_VERSION, "feature_names": pc.feature_names(), "seed": seed,
            "sample_seed": sample_seed, "exclude_regions": sorted(exclude), "decode": dict(pc.DEFAULT_DECODE),
            "params": dict(LGB_PARAMS, num_rounds=rounds), "classes": {},
            "provenance": "clean room: docs/PIXGBM_SPEC_CLEANROOM.md, автор спецификации ≠ исполнитель"}
    for cls, fname in pc.MODEL_FILES.items():
        Xs, ys, regs = [], [], set()
        for info in index:
            if info["region"].split("_")[0] in exclude or cls not in info["classes"]:
                continue
            z = np.load(cache / info["file"])
            Xs.append(z[f"X_{cls}"])
            ys.append(z[f"y_{cls}"])
            regs.add(info["region"])
        if not Xs:
            continue
        X, y = np.concatenate(Xs), np.concatenate(ys)
        params = dict(LGB_PARAMS, seed=seed, num_threads=threads)
        ds = lgb.Dataset(X, label=y, feature_name=pc.feature_names(), free_raw_data=True)
        b = lgb.train(params, ds, num_boost_round=rounds)
        b.save_model(str(out / fname))
        meta["classes"][cls] = {"rows": int(len(y)), "pos": int(y.sum()), "regions": sorted(regs),
                                "sha256": hashlib.sha256((out / fname).read_bytes()).hexdigest()}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta


def cmd_fit(args) -> None:
    meta = fit(Path(args.cache), Path(args.out), args.seed, set(args.exclude or []), args.threads,
               args.sample_seed, args.rounds)
    print(json.dumps({k: v for k, v in meta["classes"].items()}, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--root", required=True)
    s.add_argument("--cache", required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--regions", nargs="*", default=None, help="номера участков (по умолчанию все с геофизикой)")
    f = sub.add_parser("fit")
    f.add_argument("--cache", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--sample-seed", type=int, default=0)
    f.add_argument("--exclude", nargs="*", default=[])
    f.add_argument("--threads", type=int, default=3)
    f.add_argument("--rounds", type=int, default=NUM_ROUNDS)
    args = ap.parse_args()
    {"sample": cmd_sample, "fit": cmd_fit}[args.cmd](args)


if __name__ == "__main__":
    main()
