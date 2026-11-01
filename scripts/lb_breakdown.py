#!/usr/bin/env python3
"""Офлайн-разбивка как scoring.log площадки (Приложение 4 / compute_metrics.auxiliary).

Формат строк сверяется с reports/lb/scoring_v2_eb268d6_0.41238.log, чтобы diff
кандидата с LB-логом был построчным. Числа — из F1Calculator.calculate_f1
(materials/compute_metrics.py): soft_recall, center_deviation, fp_ratio,
by_data_type, size_categories, contour*.

    python scripts/lb_breakdown.py --pred pred.geojson --gt gt.geojson \\
        --out-dir notes/lb_bd --archive name.zip

    python scripts/lb_breakdown.py --results results.json --out-dir notes/lb_bd

Флаг scoreboard/compare: --lb-breakdown [DIR] (см. scoreboard.py, compare_candidates.py).

honest-10 gate (orch / PROTOCOL):
  python scripts/scoreboard.py --work WORK --train TRAIN --honest 10 --lb-breakdown OUTDIR
  → OUTDIR/scoring_*.log + expected_delta_lb.md (proxy ΔLB per type; class ∂S/∂F1).

final_score = Σ_c w_c·F1_c / Σ w_present (CLASS_WEIGHTS); type/size aux ↛ score.

Формулы (как в compute_metrics; ε = Config.EPSILON ≈ 1e-6):
  soft_recall     = soft_hits / (n_gt + 1e-6)   # soft_hit: IoU>0 или точка в r_tol
  center_deviation = mean(||c_pred − c_gt||) по TP-парам (м, UTM)
  fp_ratio        = Σfp / (Σtp + 1e-6)
  type precision  = tp / (tp + 0.5·fp + ε)      # FP по markup_type предсказания
  type recall     = tp / (tp + fn + ε)          # TP/FN по markup_type GT
  size recall     = tp/gt по корзине площади GT-полигона (м²); точки вне корзин
  size bins       = (0,100], (100,500], (500,1000], (1000,5000], (5000,10000], (10000,+∞)
  li/ae/sp/or/geo → лидар/аэрофото/спутник/ортофото/геофизика (DATA_TYPE_GROUPS)

Неопределённости (документированы в reports/bot7_lb_breakdown.md):
  • подпись «≤ N м²» у корзин 2+ — на самом деле (prev, N]; как в логе площадки.
  • пустая корзина (gt=0) → recall None; в тексте печатаем 0.000000 (как LB при tp=0).
  • «геофизика (maggr, etgr, terfl)» — в DATA_TYPE_GROUPS ещё geogr; в логе LB не упомянут.
  • время инференса / имя архива — только если переданы; offline их нет.
  • contour finds_points всегда «— (контур не применим)»; классы без TP-пар опускаются.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Порядок типов данных как в scoring.log площадки.
DATA_TYPE_ORDER = ("лидар", "аэрофото", "спутник", "ортофото", "геофизика")
GEOPHYSICS_NOTE = "             (сумма всех классов: maggr, etgr, terfl)"
# SIZE_BINS из compute_metrics / Config
DEFAULT_SIZE_BINS = [100.0, 500.0, 1000.0, 5000.0, 10000.0]
POINT_CLASSES = frozenset({"finds_points"})


def size_bucket_label(max_area: float | None, bins: list[float]) -> str:
    """Подпись корзины как в LB-логе: «≤ 100 м²» … «> 10000 м²»."""
    if max_area is None:
        last = int(bins[-1]) if bins else 10000
        return f"> {last} м²"
    # int для 100.0 → 100; пробел после ≤ как в логе
    v = int(max_area) if float(max_area) == int(max_area) else max_area
    return f"≤ {v} м²"


def extract_breakdown(results: dict[str, Any]) -> dict[str, Any]:
    """Нормализованный словарь разбивки из результата F1Calculator.calculate_f1."""
    aux = results.get("auxiliary") or {}
    bins = list((results.get("parameters") or {}).get("size_bins") or DEFAULT_SIZE_BINS)
    by_type_raw = aux.get("by_data_type") or {}
    by_type: dict[str, dict[str, Any]] = {}
    for name in DATA_TYPE_ORDER:
        if name in by_type_raw:
            by_type[name] = dict(by_type_raw[name])
    for name, row in by_type_raw.items():
        if name not in by_type:
            by_type[name] = dict(row)

    size_cats = []
    for i, cat in enumerate(aux.get("size_categories") or []):
        max_a = cat.get("max_area")
        if max_a is None and i < len(bins):
            max_a = bins[i]
        elif i >= len(bins):
            max_a = None
        size_cats.append({
            "max_area": max_a,
            "label": size_bucket_label(max_a, bins),
            "gt": int(cat.get("gt") or 0),
            "tp": int(cat.get("tp") or 0),
            "recall": cat.get("recall"),
        })
    # Если size_categories пуст — всё равно 6 стабильных корзин
    if not size_cats:
        for i, upper in enumerate(bins):
            size_cats.append({
                "max_area": upper, "label": size_bucket_label(upper, bins),
                "gt": 0, "tp": 0, "recall": None,
            })
        size_cats.append({
            "max_area": None, "label": size_bucket_label(None, bins),
            "gt": 0, "tp": 0, "recall": None,
        })

    contour = aux.get("contour") or {}
    contour_by_class = aux.get("contour_by_class") or {}
    return {
        "final_score": float(results.get("final_score") or 0.0),
        "soft_recall": float(aux.get("soft_recall") or 0.0),
        "center_deviation": float(aux.get("center_deviation") or 0.0),
        "fp_ratio": float(aux.get("fp_ratio") or 0.0),
        "contour": {
            "boundary_iou": contour.get("boundary_iou"),
            "hausdorff95_m": contour.get("hausdorff95_m"),
            "polis_m": contour.get("polis_m"),
            "pairs": int(contour.get("pairs") or 0),
        },
        "contour_by_class": {
            cls: {
                "boundary_iou": v.get("boundary_iou"),
                "hausdorff95_m": v.get("hausdorff95_m"),
                "polis_m": v.get("polis_m"),
                "pairs": int(v.get("pairs") or 0),
            }
            for cls, v in contour_by_class.items()
        },
        "by_data_type": by_type,
        "size_categories": size_cats,
        "size_bins": bins,
        "per_class": results.get("per_class") or {},
    }


def _fmt6(x: float | None, empty: str = "0.000000") -> str:
    if x is None:
        return empty
    return f"{float(x):.6f}"


def format_scoring_log(
    results: dict[str, Any],
    *,
    archive: str = "",
    inference_time: str = "",
) -> str:
    """Текст в формате scoring.log площадки (строки diff-able с LB-логом)."""
    bd = extract_breakdown(results)
    arch = archive or "—"
    t = inference_time or "— / 30 мин"
    lines: list[str] = [
        "===== ИТОГИ ОЦЕНКИ · КОЗ-3 =====",
        "Дополнительные метрики. На final_score не влияют.",
        "",
        f"Архив:              {arch}",
        f"Время инференса:    {t}",
        f"final_score:        {bd['final_score']:.10f}",
        "",
        "Сводные показатели:",
        f"  Soft recall:        {bd['soft_recall']:.6f}",
        f"  Center deviation:   {bd['center_deviation']:.6f} м",
        f"  FP ratio:           {bd['fp_ratio']:.6f}",
        "",
        "Контур (по TP-парам):",
        f"  Boundary IoU:       {_fmt6(bd['contour']['boundary_iou'])}",
        f"  Hausdorff95:        {_fmt6(bd['contour']['hausdorff95_m'])} м",
        f"  PoLiS:              {_fmt6(bd['contour']['polis_m'])} м",
        "",
        "Контур по классам:",
    ]
    # Порядок классов = CLASS_WEIGHTS; finds_points всегда; без TP — пропуск
    class_order = list((results.get("parameters") or {}).get("target_classes") or [])
    if not class_order:
        class_order = list(bd["per_class"].keys()) or list(bd["contour_by_class"].keys())
    seen = set()
    for cls in class_order:
        seen.add(cls)
        if cls in POINT_CLASSES:
            lines.append(f"  {cls:22s} — (контур не применим)")
            continue
        row = bd["contour_by_class"].get(cls)
        if not row or not row.get("pairs"):
            continue
        bi = row.get("boundary_iou")
        hd = row.get("hausdorff95_m")
        po = row.get("polis_m")
        if bi is None and hd is None and po is None:
            continue
        lines.append(
            f"  {cls:22s} BIoU {_fmt6(bi)}   HD95 {_fmt6(hd)} м   PoLiS {_fmt6(po)} м"
        )
    for cls, row in bd["contour_by_class"].items():
        if cls in seen:
            continue
        if cls in POINT_CLASSES:
            lines.append(f"  {cls:22s} — (контур не применим)")
            continue
        if not row.get("pairs"):
            continue
        lines.append(
            f"  {cls:22s} BIoU {_fmt6(row.get('boundary_iou'))}   "
            f"HD95 {_fmt6(row.get('hausdorff95_m'))} м   "
            f"PoLiS {_fmt6(row.get('polis_m'))} м"
        )

    lines += ["", "По типам данных:"]
    for name in DATA_TYPE_ORDER:
        row = bd["by_data_type"].get(name)
        if not row:
            # Стабильный каркас: нули, чтобы diff с LB всегда имел 5 строк типов
            lines.append(f"  {name:10s} precision 0.000000   recall 0.000000")
        else:
            lines.append(
                f"  {name:10s} precision {float(row['precision']):.6f}   "
                f"recall {float(row['recall']):.6f}"
            )
        if name == "геофизика":
            lines.append(GEOPHYSICS_NOTE)
    for name, row in bd["by_data_type"].items():
        if name in DATA_TYPE_ORDER:
            continue
        lines.append(
            f"  {name:10s} precision {float(row['precision']):.6f}   "
            f"recall {float(row['recall']):.6f}"
        )

    lines += ["", "По размерам GT-объектов:"]
    for cat in bd["size_categories"]:
        label = cat["label"]
        # LB: "recall" всегда с колонки 15 (после "  " + label шириной 13)
        r = cat["recall"]
        lines.append(f"  {label:<13s}recall {_fmt6(r)}")

    lines.append("")
    return "\n".join(lines)


def breakdown_to_csv_rows(bd: dict[str, Any]) -> list[dict[str, Any]]:
    """Плоские строки для CSV (machine-readable)."""
    rows: list[dict[str, Any]] = [
        {"section": "summary", "key": "final_score", "value": bd["final_score"]},
        {"section": "summary", "key": "soft_recall", "value": bd["soft_recall"]},
        {"section": "summary", "key": "center_deviation_m", "value": bd["center_deviation"]},
        {"section": "summary", "key": "fp_ratio", "value": bd["fp_ratio"]},
        {"section": "contour", "key": "boundary_iou", "value": bd["contour"]["boundary_iou"]},
        {"section": "contour", "key": "hausdorff95_m", "value": bd["contour"]["hausdorff95_m"]},
        {"section": "contour", "key": "polis_m", "value": bd["contour"]["polis_m"]},
    ]
    for name, row in bd["by_data_type"].items():
        rows.append({"section": "data_type", "key": f"{name}.precision", "value": row.get("precision")})
        rows.append({"section": "data_type", "key": f"{name}.recall", "value": row.get("recall")})
        rows.append({"section": "data_type", "key": f"{name}.tp", "value": row.get("tp")})
        rows.append({"section": "data_type", "key": f"{name}.fp", "value": row.get("fp")})
        rows.append({"section": "data_type", "key": f"{name}.fn", "value": row.get("fn")})
    for cat in bd["size_categories"]:
        rows.append({
            "section": "size",
            "key": cat["label"],
            "value": cat["recall"],
            "gt": cat["gt"],
            "tp": cat["tp"],
        })
    return rows


def write_outputs(
    results: dict[str, Any],
    out_dir: Path,
    *,
    stem: str = "scoring",
    archive: str = "",
    inference_time: str = "",
) -> dict[str, Path]:
    """Пишет .log (LB-текст), .json (разбивка), .csv. Возвращает пути."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bd = extract_breakdown(results)
    text = format_scoring_log(results, archive=archive, inference_time=inference_time)
    paths = {
        "log": out_dir / f"{stem}.log",
        "json": out_dir / f"{stem}.json",
        "csv": out_dir / f"{stem}.csv",
    }
    paths["log"].write_text(text, encoding="utf-8")
    paths["json"].write_text(json.dumps(bd, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = breakdown_to_csv_rows(bd)
    with paths["csv"].open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["section", "key", "value", "gt", "tp"], extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return paths


def compute_results(pred: dict, gt: dict, metric_path: str | None = None) -> dict[str, Any]:
    """Полный calculate_f1 через expds.metric.score."""
    from expds.metric import score
    return score(pred, gt, metric_path)


def load_geojson(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_lb_log(text: str) -> dict[str, Any]:
    """Разбор scoring.log (переиспользует lb_counts + size buckets)."""
    import importlib.util
    import re

    spec = importlib.util.spec_from_file_location(
        "lb_counts", Path(__file__).resolve().parent / "lb_counts.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    parsed = mod.parse(text)
    sizes: list[dict[str, Any]] = []
    for m in re.finditer(
        r"(≤\s*\d+\s*м²|>\s*\d+\s*м²)\s+recall\s+([0-9]+(?:\.[0-9]+)?)", text
    ):
        sizes.append({"label": re.sub(r"\s+", " ", m.group(1).strip()), "recall": float(m.group(2))})
    parsed["size_categories"] = sizes
    return parsed


def side_by_side(
    sources: dict[str, dict[str, Any]],
) -> str:
    """Таблица LB vs mock vs honest-10: P/R по типам и recall по корзинам.

    sources: имя → extract_breakdown() или parse_lb_log()-подобный dict.
    Абсолютные шкалы на mock/honest отличаются от LB — печатаем значения и
    относительный ранг внутри набора колонок (1 = лучший recall/precision).
    """
    cols = list(sources.keys())
    lines = [
        "LB breakdown side-by-side (values + rank within row; 1=best)",
        "note: mock/honest absolute scale ≠ public LB; compare shares/ranks, not raw vs LB",
        "",
        f"{'metric':28s}" + "".join(f"  {c:>14s}" for c in cols) + "  ranks",
    ]

    def _rank(vals: list[float | None], higher_better: bool = True) -> list[int]:
        indexed = [(i, v) for i, v in enumerate(vals) if v is not None]
        indexed.sort(key=lambda iv: iv[1], reverse=higher_better)
        ranks = [0] * len(vals)
        for r, (i, _) in enumerate(indexed, 1):
            ranks[i] = r
        return ranks

    def row(label: str, getter, higher_better: bool = True) -> None:
        vals: list[float | None] = []
        for c in cols:
            src = sources[c]
            vals.append(getter(src))
        ranks = _rank(vals, higher_better=higher_better)
        cells = []
        for v in vals:
            cells.append(f"  {v:14.6f}" if v is not None else f"  {'—':>14s}")
        rtxt = ",".join(str(r) if r else "-" for r in ranks)
        lines.append(f"{label:28s}" + "".join(cells) + f"  [{rtxt}]")

    def _get_score(s):
        return s.get("final_score")

    def _get_sr(s):
        return s.get("soft_recall")

    def _get_fp(s):
        return s.get("fp_ratio")

    def _get_cd(s):
        return s.get("center_deviation") or s.get("center_dev")

    row("final_score", _get_score)
    row("soft_recall", _get_sr)
    row("fp_ratio", _get_fp, higher_better=False)
    row("center_deviation_m", _get_cd, higher_better=False)

    for g in DATA_TYPE_ORDER:
        def make_p(name=g):
            def _p(s):
                groups = s.get("by_data_type") or s.get("groups") or {}
                if name in groups:
                    v = groups[name]
                    if isinstance(v, dict):
                        return float(v.get("precision"))
                    if isinstance(v, (tuple, list)) and len(v) >= 1:
                        return float(v[0])
                return None
            return _p

        def make_r(name=g):
            def _r(s):
                groups = s.get("by_data_type") or s.get("groups") or {}
                if name in groups:
                    v = groups[name]
                    if isinstance(v, dict):
                        return float(v.get("recall"))
                    if isinstance(v, (tuple, list)) and len(v) >= 2:
                        return float(v[1])
                return None
            return _r

        row(f"{g}.precision", make_p())
        row(f"{g}.recall", make_r())

    # size buckets: from first source that has labels, else defaults
    labels = []
    for c in cols:
        cats = sources[c].get("size_categories") or []
        if cats and isinstance(cats[0], dict) and cats[0].get("label"):
            labels = [x["label"] for x in cats]
            break
    if not labels:
        labels = [size_bucket_label(b, DEFAULT_SIZE_BINS) for b in DEFAULT_SIZE_BINS] + [
            size_bucket_label(None, DEFAULT_SIZE_BINS)
        ]

    for lab in labels:
        def make_s(label=lab):
            def _s(src):
                for cat in src.get("size_categories") or []:
                    if isinstance(cat, dict):
                        cl = cat.get("label") or size_bucket_label(cat.get("max_area"), DEFAULT_SIZE_BINS)
                        # нормализуем пробелы
                        if re_sub_ws(cl) == re_sub_ws(label):
                            return cat.get("recall")
                return None
            return _s
        row(f"size {lab}", make_s())

    lines.append("")
    return "\n".join(lines)


def re_sub_ws(s: str) -> str:
    import re
    return re.sub(r"\s+", " ", (s or "").strip())


# --- final_score ↔ LB breakdown (bot7, 01.11.2026) ---------------------------------
# CLASS_WEIGHTS mirrors materials/compute_metrics.py F1Calculator.CLASS_WEIGHTS.
CLASS_WEIGHTS: dict[str, float] = {
    "kurgany_povrezhdennye": 30.0,
    "kurgany_tselye": 20.0,
    "gorodishcha": 15.0,
    "arkhitektury": 10.0,
    "object_poly": 10.0,
    "fortifikatsii": 5.0,
    "finds_points": 10.0,
}
CLASS_WEIGHT_SUM = sum(CLASS_WEIGHTS.values())  # 100.0 when all present

# LB type GT shares from scoring_v2_eb268d6_0.41238.log via lb_counts.resolve
# (smallest integer solution; orthophoto hidden-gt variants ignored → shares over
# types with R>0). Recompute with lb_type_shares_from_log() if the log changes.
DEFAULT_LB_TYPE_SHARES: dict[str, float] = {
    "лидар": 964 / 2907,
    "аэрофото": 75 / 2907,
    "спутник": 377 / 2907,
    "ортофото": 50 / 2907,
    "геофизика": 1441 / 2907,
}


def type_f1(precision: float | None, recall: float | None) -> float | None:
    """F1 from type P/R (P already uses 0.5·FP as in compute_metrics)."""
    if precision is None or recall is None:
        return None
    p, r = float(precision), float(recall)
    if p + r <= 0:
        return 0.0
    return 2.0 * p * r / (p + r)


def final_score_from_per_class(
    per_class: dict[str, Any],
    weights: dict[str, float] | None = None,
) -> float:
    """final_score = Σ_c w_c·F1_c / Σ_{c present} w_c  (materials/compute_metrics.py).

    present = classes with gt_count>0 (or with an f1 key if gt_count absent).
    """
    w = dict(weights or CLASS_WEIGHTS)
    present = []
    for cls, row in (per_class or {}).items():
        if not isinstance(row, dict):
            continue
        if "f1" not in row:
            continue
        gt = row.get("gt_count", row.get("gt", None))
        if gt is not None and int(gt) <= 0:
            continue
        if cls not in w:
            continue
        present.append(cls)
    if not present:
        return 0.0
    tw = sum(w[c] for c in present)
    if tw <= 0:
        return 0.0
    return sum(float(per_class[c]["f1"]) * w[c] for c in present) / tw


def class_partial_derivatives(
    present: list[str] | None = None,
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """∂final_score/∂F1_c = w_c / Σ w_present. Default: all CLASS_WEIGHTS present."""
    w = dict(weights or CLASS_WEIGHTS)
    cls = list(present) if present is not None else list(w.keys())
    tw = sum(w[c] for c in cls if c in w)
    if tw <= 0:
        return {}
    return {c: w[c] / tw for c in cls if c in w}


def reconcile_final_score_with_lb_log(
    lb_log_text: str,
    *,
    expected: float = 0.4123779189,
    tol: float = 5e-4,
) -> dict[str, Any]:
    """Can type/size P/R from the LB log reproduce final_score?

    Answer: NO. The log itself says auxiliary metrics do not affect final_score.
    final_score is the CLASS_WEIGHTS-weighted mean of per-class F1; the public
    log does not print per-class F1, only type P/R + size recall + soft/fp/center.

    Returns a structured verdict for tests/report.
    """
    parsed = parse_lb_log(lb_log_text)
    scored = float(parsed.get("final_score") or 0.0)
    # Type-share-weighted F1 from resolved counts (or from P/R directly)
    import importlib.util as _iu
    from pathlib import Path as _P
    spec = _iu.spec_from_file_location("lb_counts", _P(__file__).resolve().parent / "lb_counts.py")
    lc = _iu.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(lc)
    sols = lc.resolve(parsed)
    type_wmean = None
    shares: dict[str, float] = {}
    if sols:
        sol = sols[0]
        total = sum(v[2] for v in sol.values()) or 1
        f_acc = 0.0
        for g, (tp, fp, ngt) in sol.items():
            f1, _, _ = lc.edge(tp, fp, ngt)
            shares[g] = ngt / total
            f_acc += f1 * ngt
        type_wmean = f_acc / total
    else:
        # fallback: harmonic from printed P/R, equal weight
        vals = []
        for g, pr in (parsed.get("groups") or {}).items():
            if isinstance(pr, (tuple, list)) and len(pr) >= 2:
                f = type_f1(pr[0], pr[1])
                if f is not None:
                    vals.append(f)
                    shares[g] = 1.0
        type_wmean = sum(vals) / len(vals) if vals else None

    reproduces = False  # by design
    gap = None if type_wmean is None else abs(type_wmean - scored)
    return {
        "log_final_score": scored,
        "expected": expected,
        "log_matches_expected": abs(scored - expected) <= tol,
        "type_share_weighted_f1": type_wmean,
        "type_weighted_gap": gap,
        "reproduces_from_type_size": reproduces,
        "reason": (
            "final_score = Σ_c w_c·F1_c / Σ w_present (CLASS_WEIGHTS); "
            "type/size/soft/fp/center are auxiliary and do not enter final_score "
            "(stated in the LB log header and compute_metrics.py)."
        ),
        "lb_type_shares": shares,
        "class_partials_all_present": class_partial_derivatives(),
    }


def lb_type_shares_from_log(lb_log_text: str) -> dict[str, float]:
    """GT shares per data type from LB log (lb_counts.resolve, smallest sol)."""
    verdict = reconcile_final_score_with_lb_log(lb_log_text)
    shares = verdict.get("lb_type_shares") or {}
    return shares if shares else dict(DEFAULT_LB_TYPE_SHARES)


def expected_delta_lb_from_type_deltas(
    delta_f1_by_type: dict[str, float],
    *,
    lb_shares: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Surrogate: ΔLB_hat(X) ≈ s_X^{LB} · ΔF1_X.

    Assumptions (LABELLED — print with every table):
      1. Type P/R do NOT enter final_score; this is a first-order planning proxy.
      2. Scale honest-10 (or mock) ΔF1_type by LB GT shares s_X (not honest shares):
         honest-10 type mix ≠ public LB mix.
      3. Ignores cross-type interactions and class-weight reallocation when a
         class appears/disappears.
      4. Exact gate ΔLB still comes from class-weighted F1:
         ΔLB = Σ_c (w_c/W)·ΔF1_c  with  ∂S/∂F1_c = w_c/W.

    Returns per-type rows + sum proxy.
    """
    shares = dict(lb_shares or DEFAULT_LB_TYPE_SHARES)
    rows = []
    total = 0.0
    for name in DATA_TYPE_ORDER:
        d = float(delta_f1_by_type.get(name) or 0.0)
        s = float(shares.get(name) or 0.0)
        contrib = s * d
        total += contrib
        rows.append({
            "type": name,
            "lb_share": s,
            "delta_f1_type": d,
            "expected_delta_lb": contrib,
        })
    return {
        "rows": rows,
        "sum_expected_delta_lb": total,
        "class_partials": class_partial_derivatives(),
        "assumptions": [
            "type metrics are auxiliary; ∂final/∂F1_type = 0 exactly",
            "proxy ΔLB_hat(X)=s_X^LB·ΔF1_X uses LB GT shares (not honest-10 shares)",
            "exact ΔLB = Σ_c (w_c/W)·ΔF1_c ; W=Σ w_present (usually 100)",
            "no cross-type / appearance effects",
        ],
    }


def delta_f1_by_type(base_bd: dict[str, Any], exp_bd: dict[str, Any]) -> dict[str, float]:
    """ΔF1 per data type from two extract_breakdown() dicts."""
    out: dict[str, float] = {}
    for name in DATA_TYPE_ORDER:
        b = (base_bd.get("by_data_type") or {}).get(name) or {}
        e = (exp_bd.get("by_data_type") or {}).get(name) or {}
        fb = type_f1(b.get("precision"), b.get("recall"))
        fe = type_f1(e.get("precision"), e.get("recall"))
        if fb is None and fe is None:
            continue
        out[name] = float(fe or 0.0) - float(fb or 0.0)
    return out


def format_expected_delta_lb(table: dict[str, Any]) -> str:
    """Human-readable expected_delta_lb table for gate stdout / .md."""
    lines = [
        "expected_delta_lb (proxy; type aux ↛ final_score)",
        "assumptions: " + "; ".join(table.get("assumptions") or []),
        "",
        f"{'type':10s}  {'s_LB':>8s}  {'ΔF1_type':>10s}  {'ΔLB_hat':>10s}",
    ]
    for row in table.get("rows") or []:
        lines.append(
            f"{row['type']:10s}  {row['lb_share']:8.4f}  "
            f"{row['delta_f1_type']:+10.6f}  {row['expected_delta_lb']:+10.6f}"
        )
    lines.append(
        f"{'SUM':10s}  {'':>8s}  {'':>10s}  {table.get('sum_expected_delta_lb', 0.0):+10.6f}"
    )
    lines.append("")
    lines.append("exact class coefficients ∂final/∂F1_c (all classes present, W=100):")
    for cls, coef in (table.get("class_partials") or {}).items():
        w = CLASS_WEIGHTS.get(cls, 0.0)
        lines.append(f"  {cls:22s}  w={w:5.1f}  ∂S/∂F1={coef:.4f}")
    lines.append("")
    return "\n".join(lines)


def write_expected_delta_lb(
    out_dir: Path,
    base_bd: dict[str, Any],
    exp_bd: dict[str, Any] | None = None,
    *,
    lb_shares: dict[str, float] | None = None,
    stem: str = "expected_delta_lb",
) -> Path:
    """Write expected_delta_lb.md (+ .json). If exp_bd is None, coeffs-only (ΔF1=0)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    deltas = delta_f1_by_type(base_bd, exp_bd) if exp_bd is not None else {n: 0.0 for n in DATA_TYPE_ORDER}
    table = expected_delta_lb_from_type_deltas(deltas, lb_shares=lb_shares)
    # also record exact final_score delta if per_class present
    if exp_bd is not None and base_bd.get("per_class") and exp_bd.get("per_class"):
        b = final_score_from_per_class(base_bd["per_class"])
        a = final_score_from_per_class(exp_bd["per_class"])
        table["exact_delta_final_score"] = a - b
        table["exact_final_base"] = b
        table["exact_final_exp"] = a
    text = format_expected_delta_lb(table)
    if table.get("exact_delta_final_score") is not None:
        text += (
            f"exact Δfinal_score (class-weighted): "
            f"{table['exact_delta_final_score']:+.6f} "
            f"({table['exact_final_base']:.6f} → {table['exact_final_exp']:.6f})\n"
        )
    md = out_dir / f"{stem}.md"
    js = out_dir / f"{stem}.json"
    md.write_text(text, encoding="utf-8")
    js.write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    return md


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", default=None, help="predictions GeoJSON")
    ap.add_argument("--gt", default=None, help="ground truth GeoJSON")
    ap.add_argument("--results", default=None, help="готовый JSON F1Calculator.calculate_f1")
    ap.add_argument("--metric", default=None, help="materials/compute_metrics.py")
    ap.add_argument("--out-dir", default=None, help="куда писать scoring.log/.json/.csv")
    ap.add_argument("--stem", default="scoring", help="префикс файлов")
    ap.add_argument("--archive", default="", help="имя архива в шапке лога")
    ap.add_argument("--inference-time", default="", help="строка времени инференса")
    ap.add_argument("--lb-log", default=None, help="с эталонным LB-логом: дописать side_by_side.md")
    ap.add_argument("--compare-json", action="append", default=[],
                    help="имя=path.json разбивки для side-by-side (повторяемо)")
    ap.add_argument("--reconcile-lb", default=None,
                    help="только сверка final_score с LB-логом (type/size ↛ score); путь к scoring.log")
    ap.add_argument("--expected-delta-from", default=None, metavar="BASE.json",
                    help="с --compare-json exp=...: писать expected_delta_lb.md из BASE vs exp")
    args = ap.parse_args(argv)

    if args.reconcile_lb:
        verdict = reconcile_final_score_with_lb_log(
            Path(args.reconcile_lb).read_text(encoding="utf-8", errors="replace"))
        print(json.dumps(verdict, ensure_ascii=False, indent=2))
        print("reproduces_from_type_size:", verdict["reproduces_from_type_size"])
        print("reason:", verdict["reason"])
        return 0 if verdict["log_matches_expected"] and not verdict["reproduces_from_type_size"] else 1

    if not args.out_dir:
        ap.error("нужен --out-dir (или только --reconcile-lb)")
    if args.results:
        results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    elif args.pred and args.gt:
        results = compute_results(load_geojson(args.pred), load_geojson(args.gt), args.metric)
    else:
        ap.error("нужен --results или пара --pred/--gt")

    paths = write_outputs(
        results, Path(args.out_dir), stem=args.stem,
        archive=args.archive, inference_time=args.inference_time,
    )
    print(f"wrote {paths['log']}", flush=True)
    print(f"wrote {paths['json']}", flush=True)
    print(f"wrote {paths['csv']}", flush=True)
    print(format_scoring_log(results, archive=args.archive, inference_time=args.inference_time), end="")

    sources: dict[str, dict[str, Any]] = {"offline": extract_breakdown(results)}
    if args.lb_log:
        sources = {"LB": parse_lb_log(Path(args.lb_log).read_text(encoding="utf-8", errors="replace")),
                   **sources}
    for spec in args.compare_json:
        name, _, path = spec.partition("=")
        if not path:
            ap.error("--compare-json имя=path.json")
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        sources[name] = raw if "soft_recall" in raw else extract_breakdown(raw)
    if len(sources) > 1:
        table = side_by_side(sources)
        tp = Path(args.out_dir) / "side_by_side.md"
        tp.write_text(table + "\n", encoding="utf-8")
        print(f"wrote {tp}", flush=True)
        print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
