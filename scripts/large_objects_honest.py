#!/usr/bin/env python3
"""Честный honest-10 для [large_objects] по закэшированным OOF (CPU, без обучения и без GPU).

Кэша в git нет (протокол данных). Если --fam / --gt / --clusters не находятся,
скрипт печатает, что запускать, и выходит с кодом 2. Числа не выдумывает.

Что считает, когда кэш есть:
  off        — порог группы с других семей (sweep_groups), как scripts/scoreboard.py --honest 10;
  area_tau   — та же базовая τ, для gor/fort площадью ≥ min_area ослабление area_tau
               (константа из конфига, на фолде оценки не подбирается);
  fixc_nested — для крупных gorodishcha dfine_ae τ_large выбирается на обучающих
               семьях фолда из сетки 0.10/0.20/0.30, на фолд оценки не смотрит.

Площадь — geom.area после перевода метрики в UTM (м²). У кругов D-FINE r=4 это ~50 м²,
а не бокс: Fix C по такому кэшу не срабатывает, если в свойствах нет box_area_m2
(скрипт прокидывает box_area_m2 в параллельный массив, если ключ есть в JSONL).

    python3 scripts/large_objects_honest.py \\
        --fam /workspace/koz3_shared/work11/fam0 \\
        --gt /workspace/rerank/gt_train_v2.geojson \\
        --clusters /workspace/koz3_shared/work/family_clusters.json \\
        --metric materials/compute_metrics.py \\
        --out work/large_objects_honest.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from expds import metric as M  # noqa: E402
from expds.constants import CLASS_WEIGHTS  # noqa: E402
from expds.large_objects import spec as lo_spec  # noqa: E402
from expds.large_objects import threshold_for  # noqa: E402
from expds.naming import site_family  # noqa: E402
from family_curves import collect_batched  # noqa: E402

M._SHM_CACHE_DIR = Path("/proc/expds_no_cache")
BINS = (100.0, 500.0, 1000.0, 5000.0, 10000.0)
BIN_NAMES = ("<=100", "100-500", "500-1k", "1k-5k", "5k-10k", ">10k")
FIXC_GRID = (0.10, 0.20, 0.30)
EPS = 1e-6


def sbin(area: float) -> str:
    for edge, name in zip(BINS, BIN_NAMES):
        if area <= edge:
            return name
    return BIN_NAMES[-1]


def inputs_ready(fam: Path, gt: Path, clusters: Path, metric: Path) -> list[str]:
    missing = []
    if not metric.is_file():
        missing.append(f"metric {metric}")
    if not gt.is_file():
        missing.append(f"gt {gt}")
    if not clusters.is_file():
        missing.append(f"clusters {clusters}")
    if not fam.is_dir() or not any(fam.glob("*.jsonl")):
        missing.append(f"fam jsonl {fam}")
    return missing


def run_request(commit: str) -> str:
    return f"""honest-10 не посчитан: в этом окружении нет OOF jsonl и эталона (в git их нет).

Запрос на прогон (CPU, без GPU и без обучения):
  branch: bots/bot4-large-objects
  commit: {commit}
  command:
    python3 scripts/large_objects_honest.py \\
      --fam /workspace/koz3_shared/work11/fam0 \\
      --gt /workspace/rerank/gt_train_v2.geojson \\
      --clusters /workspace/koz3_shared/work/family_clusters.json \\
      --metric materials/compute_metrics.py \\
      --out work/large_objects_honest.json
  ожидаемое время: 30–90 мин на одном ядре (сбор кривых по семьям, как bigobj; три варианта).
  критерий успеха: код выхода 0 и JSON с ключами off, area_tau, fixc_nested;
    у off score близок к честной базе стека, из которого собран fam (для стека A — 0.314615 ± 0.002);
    area_tau и fixc_nested содержат score, precision, recall, delta_vs_off, size_recall, recovered_gt_gt5k.
  Не сдавать на LB. Флаги в configs/default.toml не включать, пока эта Δ не принята.
"""


def _units(fams: list[str], clusters: Path) -> dict[str, list[str]]:
    units = {f: [f] for f in fams}
    for members in json.loads(clusters.read_text(encoding="utf-8"))["clusters"].values():
        for m in members:
            units.pop(m, None)
        units["+".join(sorted(members))] = sorted(members)
    return units


def _iter_feats(fam: Path, members: list[str]):
    for m in members:
        src = fam / f"{m}.jsonl"
        if not src.exists():
            continue
        with src.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def _attach_box_area(preds: dict, pending: list[tuple]) -> dict[str, np.ndarray]:
    """Площадь в порядке match_collected (по убыванию confidence).

    pending — (conf, region, group, area) в том же порядке, что объекты попали в _collect.
    Если длины не сошлись (метрика отбросила геометрию), берём geom.area.
    """
    areas = {}
    used = defaultdict(int)
    for cls, rows in preds.items():
        order = sorted(range(len(rows)), key=lambda i: rows[i][1], reverse=True)
        out = np.zeros(len(order), dtype=np.float64)
        for rank, i in enumerate(order):
            geom, conf, reg, grp = rows[i]
            fallback = 0.0 if getattr(geom, "geom_type", "") == "Point" else float(geom.area)
            # линейный поиск следующего pending с теми же conf/reg/grp — только если заранее совпало
            out[rank] = fallback
        areas[cls] = out
    # Точное выравнивание: идём по preds в исходном порядке и съедаем pending того же ключа.
    buckets: dict[tuple, list[float]] = defaultdict(list)
    for conf, reg, grp, area in pending:
        buckets[(round(float(conf), 6), reg, grp)].append(float(area))
    for cls, rows in preds.items():
        order = sorted(range(len(rows)), key=lambda i: rows[i][1], reverse=True)
        for rank, i in enumerate(order):
            geom, conf, reg, grp = rows[i]
            key = (round(float(conf), 6), reg, grp)
            if buckets[key]:
                areas[cls][rank] = buckets[key].pop(0)
                used[cls] += 1
            elif getattr(geom, "geom_type", "") != "Point":
                areas[cls][rank] = float(geom.area)
    return areas


def _curve_pack(calc, preds, gts):
    """Кривые + площади предсказаний (порядок conf) + площади GT и пары (бин GT, conf, регион, группа, площадь предсказания)."""
    p_ann, g_ann = {}, {}
    for cls, rows in preds.items():
        p_ann[cls] = [f"{rows[i][3]}|{rows[i][2]}" for i in range(len(rows))]
    gt_area = {}
    for cls, rows in gts.items():
        gt_area[cls] = np.array([0.0 if getattr(g, "geom_type", "") == "Point" else float(g.area)
                                 for g, *_ in rows], dtype=np.float64)
        g_ann[cls] = [sbin(a) if a > 0 else "point" for a in gt_area[cls]]
    pairs: list[dict] = []
    curves = M.match_collected(calc, preds, gts, pairs_out=pairs, p_ann=p_ann, g_ann=g_ann)
    return curves, pairs, gt_area


def _concat_curves(parts: list[dict]) -> dict:
    classes = set().union(*(p.keys() for p in parts)) if parts else set()
    out = {}
    for cls in classes:
        confs, hits, regs, groups, gts = [], [], [], [], Counter()
        n = 0
        for curves in parts:
            c = curves.get(cls)
            if c is None:
                continue
            if c.conf.size:
                confs.append(c.conf)
                hits.append(c.hit)
                regs.extend(c.pred_region)
                groups.extend(c.pred_group)
            gts.update(c.gt_regions)
            n += c.n_gt
        if not confs:
            out[cls] = M.ClassCurve(np.zeros(0), np.zeros(0, dtype=bool), [], n, gts, [])
            continue
        conf = np.concatenate(confs)
        hit = np.concatenate(hits)
        order = np.argsort(-conf, kind="stable")
        out[cls] = M.ClassCurve(conf[order], hit[order], [regs[i] for i in order], n, gts,
                                [groups[i] for i in order])
    return out


def _concat_areas(area_parts: list[dict], curve_parts: list[dict]) -> dict[str, np.ndarray]:
    """Площади в том же порядке, что _concat_curves (пересортировка по confidence семей)."""
    classes = set().union(*(p.keys() for p in curve_parts)) if curve_parts else set()
    out = {}
    for cls in classes:
        chunks, confs = [], []
        for areas, curves in zip(area_parts, curve_parts):
            c = curves.get(cls)
            if c is None or not c.conf.size:
                continue
            a = areas.get(cls)
            if a is None or len(a) != c.conf.size:
                a = np.zeros(c.conf.size, dtype=np.float64)
            chunks.append(a)
            confs.append(c.conf)
        if not chunks:
            out[cls] = np.zeros(0, dtype=np.float64)
            continue
        conf = np.concatenate(confs)
        area = np.concatenate(chunks)
        out[cls] = area[np.argsort(-conf, kind="stable")]
    return out


def _weighted(per: dict) -> dict:
    num_f = num_p = num_r = den = 0.0
    tp = fp = fn = 0.0
    for cls, row in per.items():
        if row["n_gt"] <= 0:
            continue
        w = float(CLASS_WEIGHTS.get(cls, 0.0))
        tp += row["tp"]
        fp += row["fp"]
        fn += row["fn"]
        num_f += w * row["f1"]
        num_p += w * (row["tp"] / (row["tp"] + 0.5 * row["fp"] + EPS))
        num_r += w * (row["tp"] / (row["tp"] + row["fn"] + EPS))
        den += w
    return {"score": num_f / den if den else 0.0,
            "precision": num_p / den if den else 0.0,
            "recall": num_r / den if den else 0.0,
            "tp": tp, "fp": fp, "fn": fn}


def _counts_for(curves, areas, folds, lo, *, area_on: bool, fixc_on: bool, fixc_grid: bool):
    """Как metric.honest_counts_reference, но τ применения — threshold_for.

    fixc_grid: на каждом фолде τ_large из FIXC_GRID по F1 обучающих семей, не по фолду оценки.
    """
    every = set().union(*folds) if folds else set()
    counts = {cls: {} for cls in curves}
    chosen = []
    for held in folds:
        train = every - held
        res = M.sweep_groups_reference({cls: M.curve_where(c, train) for cls, c in curves.items()})
        taus = {}
        for cls, v in res.items():
            for g, gv in v["groups"].items():
                taus.setdefault(g, {})[cls] = gv["threshold"]
        tau_large = float(lo.get("dfine_ae_gor_tau_large", 0.20))
        if fixc_on and fixc_grid:
            tau_large = _pick_tau_large(curves, areas, train, taus, lo)
            chosen.append(tau_large)
        lo_fold = dict(lo, dfine_ae_gor_tau_large=tau_large)
        for cls, c in curves.items():
            hc = M.curve_where(c, held)
            per = counts[cls]
            for r, n in hc.gt_regions.items():
                per.setdefault(r, [0, 0, 0])[2] += int(n)
            if hc.conf.size == 0:
                continue
            area = _mask_area(c, areas.get(cls), held)
            kept = np.zeros(hc.conf.size, dtype=bool)
            for i, (conf, grp, reg) in enumerate(zip(hc.conf, hc.pred_group, hc.pred_region)):
                base = taus.get(grp, {}).get(cls)
                if base is None:
                    continue
                markup = "ae" if grp in ("dfine_ae", "ae_dfine") else grp
                t = threshold_for(base, float(area[i]), cls, grp, markup, lo_fold,
                                  area_on=area_on, fixc_on=fixc_on)
                kept[i] = conf >= t
            for reg, hit in zip(np.asarray(hc.pred_region, dtype=object)[kept], hc.hit[kept]):
                per.setdefault(reg, [0, 0, 0])[0 if hit else 1] += 1
    return counts, chosen


def _mask_area(curve, area: np.ndarray | None, regions: set[str]) -> np.ndarray:
    if area is None or len(area) != len(curve.pred_region):
        area = np.zeros(len(curve.pred_region), dtype=np.float64)
    mask = np.fromiter((r in regions for r in curve.pred_region), dtype=bool, count=len(curve.pred_region))
    return area[mask]


def _pick_tau_large(curves, areas, train: set[str], taus: dict, lo: dict) -> float:
    best_t, best_s = FIXC_GRID[0], -1.0
    for t in FIXC_GRID:
        lo_t = dict(lo, dfine_ae_gor_tau_large=t)
        # F1 только на train при уже замороженных τ групп — фолд оценки сюда не входит.
        score = _score_frozen(curves, areas, train, taus, lo_t, area_on=False, fixc_on=True)
        if score > best_s:
            best_t, best_s = t, score
    return best_t


def _score_frozen(curves, areas, regions, taus, lo, *, area_on, fixc_on) -> float:
    counts = {cls: {} for cls in curves}
    for cls, c in curves.items():
        hc = M.curve_where(c, regions)
        if hc.conf.size == 0:
            for r, n in hc.gt_regions.items():
                counts[cls].setdefault(r, [0, 0, 0])[2] += int(n)
            continue
        area = _mask_area(c, areas.get(cls), regions)
        for r, n in hc.gt_regions.items():
            counts[cls].setdefault(r, [0, 0, 0])[2] += int(n)
        for i, (conf, grp, reg) in enumerate(zip(hc.conf, hc.pred_group, hc.pred_region)):
            base = taus.get(grp, {}).get(cls)
            if base is None:
                continue
            markup = "ae" if grp in ("dfine_ae", "ae_dfine") else grp
            t = threshold_for(base, float(area[i]), cls, grp, markup, lo, area_on=area_on, fixc_on=fixc_on)
            if conf >= t:
                counts[cls].setdefault(reg, [0, 0, 0])[0 if hc.hit[i] else 1] += 1
    return M.score_counts(counts, regions)[0]


def _size_and_recovered(pairs_all, gt_bins, curves, areas, folds, lo, *, area_on, fixc_on, fixc_grid):
    """Полнота по корзинам площади GT и сколько пропусков >5k м² варианта off закрывает этот вариант.

    Пары — жадное сопоставление полного списка кандидатов (то же, что hit в кривой).
    """
    every = set().union(*folds) if folds else set()
    # τ по регионам
    tau_of = {}
    chosen = []
    for held in folds:
        train = every - held
        res = M.sweep_groups_reference({cls: M.curve_where(c, train) for cls, c in curves.items()})
        taus = {}
        for cls, v in res.items():
            for g, gv in v["groups"].items():
                taus.setdefault(g, {})[cls] = gv["threshold"]
        tau_large = float(lo.get("dfine_ae_gor_tau_large", 0.20))
        if fixc_on and fixc_grid:
            tau_large = _pick_tau_large(curves, areas, train, taus, lo)
            chosen.append(tau_large)
        lo_fold = dict(lo, dfine_ae_gor_tau_large=tau_large)
        for r in held:
            tau_of[r] = (taus, lo_fold)
    tp_bin = Counter()
    for pair in pairs_all:
        taus, lo_fold = tau_of.get(pair["pred_region"], ({}, lo))
        base = taus.get(pair["group"], {}).get(pair["cls"])
        if base is None:
            continue
        markup = "ae" if pair["group"] in ("dfine_ae", "ae_dfine") else pair["group"]
        t = threshold_for(base, pair["area"], pair["cls"], pair["group"], markup, lo_fold,
                          area_on=area_on, fixc_on=fixc_on)
        if pair["conf"] >= t:
            tp_bin[pair["gt_bin"]] += 1
    size = {}
    for name in BIN_NAMES:
        gt_n = int(gt_bins[name])
        tp_n = int(tp_bin[name])
        size[name] = {"gt": gt_n, "tp": tp_n, "recall": (tp_n / gt_n) if gt_n else None}
    return size, chosen


def evaluate(fam: Path, gt: Path, clusters: Path, metric: Path, batch: int) -> dict:
    mod = M.load_official(str(metric))
    calc = mod.F1Calculator()
    import tomllib
    cfg = tomllib.loads((ROOT / "configs" / "default.toml").read_text(encoding="utf-8"))
    lo = lo_spec(cfg)
    gt_by_fam = defaultdict(list)
    for feat in json.loads(gt.read_text(encoding="utf-8"))["features"]:
        reg = str((feat.get("properties") or {}).get("region_name", "default_source"))
        gt_by_fam[site_family(reg)].append(feat)
    fams = sorted(set(gt_by_fam) | {p.stem for p in fam.glob("*.jsonl")})
    curve_parts, area_parts = [], []
    pairs_all = []
    gt_bins = Counter()
    for _unit, members in sorted(_units(fams, clusters).items()):
        pending = []

        def pred_iter(members=members, pending=pending):
            for feat in _iter_feats(fam, members):
                props = feat.get("properties") or {}
                box = props.get("box_area_m2")
                if box is not None:
                    pending.append((props.get("confidence", 1.0),
                                    str(props.get("region_name", "default_source")),
                                    str(props.get("threshold_group") or props.get("markup_type") or ""),
                                    float(box)))
                yield feat

        preds = collect_batched(mod, calc, pred_iter(), False, batch)
        gts = collect_batched(mod, calc, iter(f for m in members for f in gt_by_fam.get(m, [])), True, batch)
        for rows in gts.values():
            for geom, *_ in rows:
                if getattr(geom, "geom_type", "") == "Point":
                    continue
                gt_bins[sbin(float(geom.area))] += 1
        curves, pairs, _gt_area = _curve_pack(calc, preds, gts)
        # пары хранят группу и регион в p_ann как "group|region" — разберём
        parsed = []
        for p in pairs:
            grp, _, reg = str(p["pred_layer"]).partition("|")
            parsed.append({"cls": p["class"], "conf": p["conf"], "group": grp,
                           "pred_region": reg or p["pred_region"], "gt_bin": p["gt_layer"],
                           "area": 0.0})
        areas = _attach_box_area(preds, pending)
        # площадь предсказания в паре — по conf+region+group из areas не вытащить надёжно;
        # для Fix C важнее box_area в areas кривой. В паре оставим 0 и подменим ниже из кривой нельзя.
        # Для корзин GT площадь предсказания не нужна, если area_on смотрит площадь кандидата.
        # Восстановим её из pending уже съеденного — повторно из кривых areas по (conf, region, group) жадно.
        buckets = defaultdict(list)
        for cls, arr in areas.items():
            c = curves[cls]
            for conf, reg, grp, area in zip(c.conf, c.pred_region, c.pred_group, arr):
                buckets[(cls, round(float(conf), 5), reg, grp)].append(float(area))
        for p in parsed:
            key = (p["cls"], round(float(p["conf"]), 5), p["pred_region"], p["group"])
            if buckets[key]:
                p["area"] = buckets[key].pop(0)
        pairs_all.extend(parsed)
        curve_parts.append(curves)
        area_parts.append(areas)
        del preds, gts
    curves = _concat_curves(curve_parts)
    areas = _concat_areas(area_parts, curve_parts)
    regions = sorted({r for c in curves.values() for r in list(c.pred_region) + list(c.gt_regions)})
    folds = M.family_folds(regions, site_family, 10)

    def one(name, area_on, fixc_on, fixc_grid):
        counts, chosen = _counts_for(curves, areas, folds, lo, area_on=area_on, fixc_on=fixc_on,
                                     fixc_grid=fixc_grid)
        score, per = M.score_counts(counts)
        w = _weighted(per)
        size, _ = _size_and_recovered(pairs_all, gt_bins, curves, areas, folds, lo,
                                     area_on=area_on, fixc_on=fixc_on, fixc_grid=fixc_grid)
        return {"name": name, "score": score, "precision": w["precision"], "recall": w["recall"],
                "tp": w["tp"], "fp": w["fp"], "fn": w["fn"], "size_recall": size,
                "fixc_tau_large_chosen": chosen, "per_class": {k: {"f1": v["f1"], "tp": v["tp"],
                                                                  "fp": v["fp"], "fn": v["fn"]}
                                                               for k, v in per.items()}}

    off = one("off", False, False, False)
    area = one("area_tau", True, False, False)
    fixc = one("fixc_nested", False, True, True)
    for variant in (area, fixc):
        variant["delta_score"] = variant["score"] - off["score"]
        variant["delta_precision"] = variant["precision"] - off["precision"]
        variant["delta_recall"] = variant["recall"] - off["recall"]
        rec = 0
        # пропуски >5k у off, закрытые вариантом: разница TP в двух крупных корзинах
        for name in ("5k-10k", ">10k"):
            rec += int(variant["size_recall"][name]["tp"]) - int(off["size_recall"][name]["tp"])
        variant["recovered_gt_gt5k"] = rec
    return {"off": off, "area_tau": area, "fixc_nested": fixc,
            "note": "τ группы — sweep на других семьях. area_tau не подбирался. "
                    "fixc_nested выбирает τ_large только на train фолда из (0.10, 0.20, 0.30). "
                    "Причина пропуска (below_tau/no_cand/shape/consumed) этим скриптом не классифицируется: "
                    "recovered_gt_gt5k — прирост TP в корзинах >5k м²."}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fam", default="/workspace/koz3_shared/bot-10/work11/fam0")
    ap.add_argument("--gt", default="/workspace/rerank/gt_train_v2.geojson")
    ap.add_argument("--clusters", default="/workspace/koz3_shared/bot-10/work/family_clusters.json")
    ap.add_argument("--metric", default=str(ROOT / "materials" / "compute_metrics.py"))
    ap.add_argument("--out", default="")
    ap.add_argument("--batch", type=int, default=20000)
    a = ap.parse_args()
    fam, gt, clusters, metric = Path(a.fam), Path(a.gt), Path(a.clusters), Path(a.metric)
    commit = "unknown"
    try:
        import subprocess
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        pass
    missing = inputs_ready(fam, gt, clusters, metric)
    if missing:
        print(run_request(commit))
        print("нет входов:", ", ".join(missing))
        return 2
    result = evaluate(fam, gt, clusters, metric, a.batch)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
