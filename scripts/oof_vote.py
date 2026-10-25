#!/usr/bin/env python3
"""Голосование моделей по уже готовым предсказаниям вне обучения (без видеокарты).

Приём призёра этапа «Сканирование» (Family): объект подтверждают несколько независимых
моделей. Здесь объекты базового файла остаются на месте, а уверенность пересчитывается
по согласию других моделей того же типа данных:

  rank-mean  среднее рангов уверенности базовой модели и ближайших объектов других моделей
             (ранг — доля объектов той же модели и класса со слабее уверенностью; нет соседа — 0);
  votes      число моделей, у которых рядом есть объект с рангом не ниже --vote-rank, плюс
             ранг базовой модели как дробная часть (порядок внутри одного числа голосов).

Ранги делают шкалы моделей сравнимыми: у карты центров уверенность 0,36 — уже много,
у канала σ — 0,99. Сосед — объект того же участка и класса ближе --radius метров на земле.

    python scripts/oof_vote.py --base F:/expds3/work/spor_focal_oof.geojson `
        --others F:/expds3/work/spor_focal_r05_oof.geojson F:/expds3/work/spor_focal_photo_oof.geojson `
                 F:/expds3/work/spor_focal_scale_oof.geojson F:/expds3/work/spor_focal_r34_oof.geojson `
        --radius 6 --mode rank-mean --out F:/expds3/work/spor_vote.geojson
    python scripts/scoreboard.py ... --replace sp_focal=F:/expds3/work/spor_vote.geojson --honest 10

Голоса разных типов данных (курганы спутника и лидара) — разные признаки: тон и тень снимка
против высоты рельефа. Там, где у другой модели нет данных (участок без LAS, край снимка),
её молчание — не голос «против». --coverage-root <корень участков> включает воздержание:
модель, чей тип данных не покрывает точку, в среднее и в число голосов не входит.
Без флага счёт тот же, что раньше. Голосование шести карт центров уже отвергнуто
(notes/vote_mean_2026-10-25.json, notes/vote_votes_2026-10-25.json). Пример нового опыта,
ещё не измеренного:

    python scripts/oof_vote.py --base F:/expds3/work/spor_focal_oof.geojson `
        --others F:/expds3/work/fusion/li_hard2_fused.geojson --coverage-root F:/expds3/train-expds-3 `
        --radius 8 --mode rank-mean --out F:/expds3/work/spor_vote_li.geojson

Группа порога и геометрия базовых объектов не меняются: табло подбирает новый порог
сданной группе. Файлы предсказаний — конкурсные данные, в git их не класть.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODES = ("rank-mean", "votes")


def load(path: Path) -> list[dict]:
    """GeoJSON → объекты: геометрия shapely, свойства, уверенность."""
    from shapely.geometry import shape

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for f in data.get("features", []):
        p = dict(f.get("properties") or {})
        try:
            g = shape(f["geometry"])
        except Exception:
            continue
        out.append({"props": p, "geom": g, "conf": float(p.get("confidence", 1.0))})
    return out


def ranks(objs: list[dict]) -> np.ndarray:
    """Ранг уверенности внутри класса этой модели: доля объектов не сильнее, (0…1]."""
    out = np.zeros(len(objs))
    by_cls: dict[str, list[int]] = defaultdict(list)
    for i, o in enumerate(objs):
        by_cls[o["props"].get("class_name")].append(i)
    for idx in by_cls.values():
        v = np.array([objs[i]["conf"] for i in idx])
        r = np.searchsorted(np.sort(v), v, side="right") / len(v)
        out[idx] = r
    return out


def ground_xy(objs: list[dict]) -> np.ndarray:
    """Внутренние точки в метрах на земле: зона UTM по каждому объекту (как отбор 12 м в сдаче)."""
    from expds.pipeline import ground_point

    pts = [ground_point(o["geom"]) for o in objs]
    return np.array([(p.x, p.y) for p in pts]) if pts else np.zeros((0, 2))


def neighbour_rank(base: list[dict], bxy: np.ndarray, other: list[dict], oxy: np.ndarray,
                   orank: np.ndarray, radius: float) -> np.ndarray:
    """Для каждого базового объекта — наибольший ранг объекта другой модели того же участка и класса
    ближе radius метров; нет такого — 0."""
    from scipy.spatial import cKDTree

    out = np.zeros(len(base))
    groups: dict[tuple, list[int]] = defaultdict(list)
    for j, o in enumerate(other):
        groups[(o["props"].get("region_name"), o["props"].get("class_name"))].append(j)
    trees = {k: (cKDTree(oxy[v]), np.asarray(v)) for k, v in groups.items() if v}
    for i, b in enumerate(base):
        key = (b["props"].get("region_name"), b["props"].get("class_name"))
        if key not in trees:
            continue
        tree, idx = trees[key]
        hits = tree.query_ball_point(bxy[i], radius)
        if hits:
            out[i] = float(orank[idx[hits]].max())
    return out


def coverage(base: list[dict], others: list[list[dict]], root: Path) -> np.ndarray:
    """[модель, объект] — покрывает ли тип данных другой модели точку базового объекта.

    Тип данных модели — markup_type её объектов. li — рамка облака LAS участка,
    остальные — непустые пиксели снимков участка (как у копий слоёв, fusion.covered).
    """
    from expds import fusion
    from expds.catalog import read_utm_json, scan_root
    from expds.crs import WEB_MERCATOR, las_box, transformer

    regs = {r.name: r for r in scan_root(root)}
    pts = [o["geom"].representative_point() for o in base]
    out = np.zeros((len(others), len(base)), bool)
    for k, other in enumerate(others):
        kinds = {o["props"].get("markup_type") for o in other}
        for i, (b, pt) in enumerate(zip(base, pts)):
            reg = regs.get(b["props"].get("region_name"))
            if reg is None:
                continue
            for mt in kinds:
                if mt == "li":
                    crs = read_utm_json(reg.utm_json)
                    for las in reg.las:
                        box = las_box(las)
                        if crs and box is not None:
                            x, y = transformer(WEB_MERCATOR, crs).transform(pt.x, pt.y)
                            if box[0] <= x <= box[2] and box[1] <= y <= box[3]:
                                out[k, i] = True
                elif reg.rasters.get(mt):
                    fb = read_utm_json(reg.utm_json)
                    out[k, i] |= bool(fusion.covered(
                        np.array([pt.x]), np.array([pt.y]), reg.rasters[mt], fallback_crs=fb)[0])
                if out[k, i]:
                    break
    return out


def fuse(base: list[dict], others: list[list[dict]], radius: float, mode: str, vote_rank: float = 0.5,
         cover: np.ndarray | None = None) -> list[float]:
    """Новая уверенность базовых объектов по согласию моделей.

    cover[модель, объект] = False — модель здесь воздерживается (нет её данных).
    Без cover счёт совпадает с прежним: каждая модель голосует.
    """
    if mode not in MODES:
        raise ValueError(f"неизвестный режим: {mode}")
    brank = ranks(base)
    bxy = ground_xy(base)
    cols = [brank]
    for other in others:
        cols.append(neighbour_rank(base, bxy, other, ground_xy(other), ranks(other), radius))
    m = np.vstack(cols)
    seen = np.vstack([np.ones(len(base), bool), cover if cover is not None else np.ones((len(others), len(base)), bool)])
    n = seen.sum(axis=0)
    if mode == "rank-mean":
        return [float(v) for v in (m * seen).sum(axis=0) / n]
    votes = ((m[1:] >= vote_rank) & seen[1:]).sum(axis=0) if len(m) > 1 else np.zeros(len(base))
    return [float((v + r) / k) for v, r, k in zip(votes, brank, n)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="OOF-файл, чьи объекты идут в выдачу")
    ap.add_argument("--others", nargs="+", required=True, help="OOF-файлы других моделей того же типа данных")
    ap.add_argument("--radius", type=float, default=6.0, help="сосед ближе стольких метров на земле")
    ap.add_argument("--mode", choices=MODES, default="rank-mean")
    ap.add_argument("--vote-rank", type=float, default=0.5, help="votes: ранг соседа, с которого он считается голосом")
    ap.add_argument("--coverage-root", default=None,
                    help="корень участков: модель без своих данных в точке воздерживается, а не голосует против")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = load(Path(args.base))
    others = [load(Path(p)) for p in args.others]
    cover = coverage(base, others, Path(args.coverage_root)) if args.coverage_root else None
    conf = fuse(base, others, args.radius, args.mode, args.vote_rank, cover)
    if cover is not None:
        print(f"доля объектов, где голосует каждая модель: {[round(float(c.mean()), 3) for c in cover]}")
    from shapely.geometry import mapping

    feats = []
    for o, c in zip(base, conf):
        p = dict(o["props"], confidence=round(c, 6), vote_mode=args.mode)
        feats.append({"type": "Feature", "geometry": mapping(o["geom"]), "properties": p})
    Path(args.out).write_text(json.dumps({"type": "FeatureCollection", "features": feats}, ensure_ascii=False),
                              encoding="utf-8")
    print(f"{len(feats)} объектов, моделей в голосовании {len(others) + 1} → {args.out}")


if __name__ == "__main__":
    main()
