#!/usr/bin/env python3
"""Целые TP, FP и эталоны по типам данных из scoring.log площадки.

Лог печатает по типу данных precision = tp / (tp + 0,5·fp) и recall = tp / (tp + fn)
с шестью знаками. Этого хватает, чтобы восстановить целые числа. Кратные решения
(всё ×2, ×3) дают те же доли; выбор между ними делает FP ratio = Σfp / Σtp
из сводки. Эталон ортофото без попаданий из долей не виден, его число оценивает
soft recall.

    python scripts/lb_counts.py Evaluation_logs_…/scoring.log [второй.log ...]

Пороги по этим числам не подбираются. Это диагностика веток: сколько эталонов
у типа данных, сколько из выданного попало и где точка «выгодно добавить»
F / (4 − F) относительно доли попаданий среди выданного.
"""
from __future__ import annotations

import itertools
import re
import sys
from pathlib import Path

TOL = 6e-7
GROUPS = ("лидар", "аэрофото", "спутник", "ортофото", "геофизика")


def parse(text: str) -> dict:
    """Числа из текста лога: final_score, soft recall, FP ratio и P/R по типам данных."""
    num = r"([0-9]+(?:\.[0-9]+)?)"
    out: dict = {"groups": {}}
    for key, pat in (("final_score", r"final_score:\s*" + num), ("soft_recall", r"Soft recall:\s*" + num),
                     ("fp_ratio", r"FP ratio:\s*" + num), ("center_dev", r"Center deviation:\s*" + num),
                     ("time", r"Время инференса:\s*([^\n/]+)")):
        m = re.search(pat, text)
        if m:
            out[key] = m.group(1).strip() if key == "time" else float(m.group(1))
    for g in GROUPS:
        m = re.search(g + r"\s+precision\s+" + num + r"\s+recall\s+" + num, text)
        if m:
            out["groups"][g] = (float(m.group(1)), float(m.group(2)))
    return out


def base_solutions(p: float, r: float, max_gt: int = 20000) -> list[tuple[int, int, int]]:
    """Все (tp, fp, n_gt) с n_gt ≤ max_gt, дающие эти доли с точностью печати."""
    sols = []
    if r == 0.0:
        return sols
    for n_gt in range(1, max_gt + 1):
        for tp in {int(r * n_gt), int(r * n_gt) + 1}:
            if tp < 1 or tp > n_gt or abs(tp / n_gt - r) > TOL:
                continue
            fp_f = 2 * (tp / p - tp)
            for fp in {int(fp_f), int(fp_f) + 1}:
                if fp >= 0 and abs(tp / (tp + 0.5 * fp) - p) <= TOL:
                    sols.append((tp, fp, n_gt))
    return sols


def resolve(parsed: dict, max_mult: int = 12) -> list[dict[str, tuple[int, int, int]]]:
    """Наборы чисел по типам данных, согласованные с FP ratio. Сначала самые мелкие."""
    groups = {g: pr for g, pr in parsed["groups"].items() if pr[1] > 0}
    base = {}
    for g, (p, r) in groups.items():
        sols = base_solutions(p, r)
        if not sols:
            return []
        base[g] = sols[0]
    names = sorted(base)
    found = []
    for mult in itertools.product(range(1, max_mult + 1), repeat=len(names)):
        tp = sum(base[g][0] * k for g, k in zip(names, mult))
        fp = sum(base[g][1] * k for g, k in zip(names, mult))
        if tp and abs(fp / tp - parsed.get("fp_ratio", fp / tp)) <= TOL:
            found.append({g: tuple(v * k for v in base[g]) for g, k in zip(names, mult)})
    found.sort(key=lambda s: sum(v[2] for v in s.values()))
    return found


def hidden_gt(parsed: dict, sol: dict, limit: int = 3000) -> list[tuple[int, int]]:
    """Эталоны типов данных без попаданий (ортофото), при которых soft recall целый."""
    sr = parsed.get("soft_recall")
    if sr is None:
        return []
    known = sum(v[2] for v in sol.values())
    out = []
    for extra in range(limit):
        n = known + extra
        hits = round(sr * n)
        if abs(hits / n - sr) <= TOL:
            out.append((extra, hits))
    return out


def edge(tp: int, fp: int, n_gt: int) -> tuple[float, float, float]:
    """F1 типа данных, доля попаданий среди выданного и точка безубыточности F / (4 − F)."""
    f1 = 2 * tp / (n_gt + tp + 0.5 * fp) if n_gt else 0.0
    return f1, tp / (tp + fp) if tp + fp else 0.0, f1 / (4 - f1)


def report(path: Path) -> None:
    parsed = parse(path.read_text(encoding="utf-8", errors="replace"))
    print(f"== {path}")
    print(f"final_score {parsed.get('final_score')}, время {parsed.get('time')}, FP ratio {parsed.get('fp_ratio')}, "
          f"soft recall {parsed.get('soft_recall')}")
    sols = resolve(parsed)
    if not sols:
        print("целых чисел не нашлось: лог другой формы")
        return
    sol = sols[0]
    if len(sols) > 1:
        print(f"согласованных наборов {len(sols)}; ниже самый мелкий, остальные кратны ему по долям")
    print(f"{'тип данных':10s} {'эталон':>7s} {'TP':>6s} {'FP':>6s} {'FN':>6s} {'выдано':>7s} {'попад.':>7s} "
          f"{'F1':>6s} {'выгодно от':>10s}")
    for g in GROUPS:
        if g in sol:
            tp, fp, n_gt = sol[g]
            f1, hit, e = edge(tp, fp, n_gt)
            print(f"{g:10s} {n_gt:7d} {tp:6d} {fp:6d} {n_gt - tp:6d} {tp + fp:7d} {hit:7.1%} {f1:6.3f} {e:10.1%}")
        elif g in parsed["groups"]:
            print(f"{g:10s} {'?':>7s} {0:6d} {'?':>6s}   эталон есть, попаданий нет")
    tp = sum(v[0] for v in sol.values())
    fp = sum(v[1] for v in sol.values())
    print(f"всего TP {tp}, FP {fp}, FP/TP {fp / tp:.6f}")
    extra = hidden_gt(parsed, sol)
    if extra:
        print("эталон типов без попаданий (ортофото и т. п.), варианты по soft recall: "
              + ", ".join(f"{e} (мягких попаданий {h})" for e, h in extra[:6]))


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    for arg in sys.argv[1:]:
        report(Path(arg))


if __name__ == "__main__":
    main()
