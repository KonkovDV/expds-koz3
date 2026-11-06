#!/usr/bin/env python3
"""Сравнение логов площадки на одном тесте: где разница в целых TP и FP.

    python scripts/lb_diff.py BASE.log OTHER.log [OTHER2.log ...] [--json OUT.json]

Тест публичной выборки один, поэтому разница двух посылок — это разница их выдачи
на тех же эталонах. Лог печатает доли с шестью знаками; из них восстанавливаются
целые числа (`scripts/lb_counts.py`), а дальше видно, сколько попаданий и ложных
добавила или сняла правка.

По каждому типу данных печатается ΔTP, ΔFP и доля попаданий в разнице выдачи h.
Блок из a попаданий и b ложных повышает F1 типа (FP весит 0,5) тогда и только
тогда, когда h = a/(a+b) больше F/(4−F) того прогона, к которому блок добавляют.
Значит:

* вариант выдаёт меньше базы, а снятая выдача попадала чаще F/(4−F) варианта —
  вариант слишком осторожен на этом тесте (порог высок, модель недооценивает
  уверенность на новых участках);
* вариант выдаёт больше, а добавленная выдача попадает реже F/(4−F) базы —
  вариант шумит.

Корзины площади: точки (`finds_points`) в них не входят (`compute_metrics`,
`size_categories`), поэтому изменение TP геофизики, которого нет в корзинах,
приходится на находки. Целые числа корзин — самый мелкий набор, согласованный
со всеми переданными логами; настоящие числа могут быть кратны ему.

Пороги по этим числам не подбираются: это разбор уже сделанных посылок.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _lb_counts():
    spec = importlib.util.spec_from_file_location("lb_counts", _HERE / "lb_counts.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


LC = _lb_counts()
TOL = LC.TOL

_SIZE_RE = re.compile(r"([≤>])\s*([0-9]+)\s*м²\s+recall\s+([0-9]+(?:\.[0-9]+)?)")
_CONTOUR_RE = re.compile(
    r"^\s+([a-z_]+)\s+BIoU\s+([0-9.]+)\s+HD95\s+([0-9.]+)\s*м\s+PoLiS\s+([0-9.]+)", re.M)
_NO_PAIRS_RE = re.compile(r"^\s+([a-z_]+)\s+—\s+\(нет TP-пар\)", re.M)
_ARCHIVE_RE = re.compile(r"Архив:\s*(\S+)")


def parse(text: str) -> dict:
    """lb_counts.parse + корзины площади, контур по классам и имя архива."""
    out = LC.parse(text)
    out["sizes"] = {f"{m.group(1)}{m.group(2)}": float(m.group(3)) for m in _SIZE_RE.finditer(text)}
    contour = {m.group(1): (float(m.group(2)), float(m.group(3)), float(m.group(4)))
               for m in _CONTOUR_RE.finditer(text)}
    for m in _NO_PAIRS_RE.finditer(text):
        contour[m.group(1)] = None
    out["contour"] = contour
    m = _ARCHIVE_RE.search(text)
    out["archive"] = m.group(1) if m else None
    return out


def resolve_any(parsed: dict, max_mult: int = 12) -> list[dict[str, tuple[int, int, int]]]:
    """Целые наборы по типам данных; если лог не сходится по FP ratio, допускает
    неизвестные ложные у типов без попаданий (precision 0 и recall 0, например
    ортофото с порогом выше всех уверенностей): Σfp известных ≤ FP ratio·Σtp."""
    sols = LC.resolve(parsed, max_mult)
    if sols:
        return sols
    zero = [g for g, (p, r) in parsed["groups"].items() if p == 0.0 and r == 0.0]
    if not zero:
        return []
    groups = {g: pr for g, pr in parsed["groups"].items() if pr[1] > 0}
    base = {}
    for g, (p, r) in groups.items():
        s = LC.base_solutions(p, r)
        if not s:
            return []
        base[g] = s[0]
    names = sorted(base)
    ratio = parsed.get("fp_ratio")
    if ratio is None:
        return []
    import itertools
    found = []
    for mult in itertools.product(range(1, max_mult + 1), repeat=len(names)):
        tp = sum(base[g][0] * k for g, k in zip(names, mult))
        fp = sum(base[g][1] * k for g, k in zip(names, mult))
        rest = ratio * tp - fp
        k = round(rest)
        # FP ratio печатается с 6 знаками: допуск на остаток ~ tp·5e-7
        if k >= 0 and abs(rest - k) <= max(TOL * tp, 1e-6):
            sol = {g: tuple(v * m for v in base[g]) for g, m in zip(names, mult)}
            sol["_fp_unknown"] = (0, k, 0)
            found.append(sol)
    found.sort(key=lambda s: sum(v[2] for g, v in s.items() if not g.startswith("_")))
    return found


def matched_solutions(parsed_list: list[dict]) -> list[dict] | None:
    """По набору на лог, с одинаковым эталоном у общих типов данных (тест один)."""
    cand = [resolve_any(p) for p in parsed_list]
    if not all(cand):
        return None
    first = cand[0]
    for s0 in first:
        gt0 = {g: v[2] for g, v in s0.items() if not g.startswith("_")}
        picks = [s0]
        for other in cand[1:]:
            hit = None
            for s in other:
                gt = {g: v[2] for g, v in s.items() if not g.startswith("_")}
                if all(gt0[g] == gt[g] for g in gt0.keys() & gt.keys()):
                    hit = s
                    break
            if hit is None:
                break
            picks.append(hit)
        if len(picks) == len(cand):
            return picks
    return None


def joint_bucket_counts(values: list[float], max_n: int = 20000) -> tuple[int | None, list[int | None]]:
    """Самый мелкий эталон корзины n, при котором все ненулевые recall — k/n
    с точностью печати. Нулевые recall совместимы с любым n."""
    nz = [v for v in values if v > 0]
    if not nz:
        return None, [0 if v == 0 else None for v in values]
    for n in range(1, max_n + 1):
        ks = [round(v * n) for v in nz]
        if all(abs(k / n - v) <= TOL for k, v in zip(ks, nz)):
            return n, [round(v * n) for v in values]
    return None, [None for _ in values]


def block_verdict(dtp: int, dfp: int, f_base: float, f_other: float) -> dict:
    """Что значит разница выдачи (ΔTP, ΔFP) варианта против базы."""
    out = {"dtp": dtp, "dfp": dfp}
    if dtp == 0 and dfp == 0:
        out["kind"] = "same"
        return out
    if dtp >= 0 and dfp <= 0:
        out["kind"] = "dominates"           # больше попаданий и не больше ложных
        return out
    if dtp <= 0 and dfp >= 0:
        out["kind"] = "dominated"
        return out
    a, b = abs(dtp), abs(dfp)
    h = a / (a + b)
    out["hit_share"] = h
    if dtp > 0:                              # вариант выдаёт больше
        e = f_base / (4 - f_base)
        out.update(kind="adds", break_even=e, pays=h > e)
    else:                                    # вариант выдаёт меньше
        e = f_other / (4 - f_other)
        out.update(kind="drops", break_even=e, pays=h > e)
    return out


def compare(texts: list[str], names: list[str] | None = None) -> dict:
    parsed = [parse(t) for t in texts]
    names = names or [p.get("archive") or f"log{i}" for i, p in enumerate(parsed)]
    res: dict = {"logs": [{"name": n, "final_score": p.get("final_score"), "archive": p.get("archive")}
                          for n, p in zip(names, parsed)]}
    sols = matched_solutions(parsed)
    res["types"] = {}
    res["fp_unknown"] = [s.get("_fp_unknown", (0, 0, 0))[1] for s in sols] if sols else []
    if sols:
        base = sols[0]
        for g in LC.GROUPS:
            if g not in base:
                continue
            tp0, fp0, n0 = base[g]
            f0 = LC.edge(tp0, fp0, n0)[0]
            row = {"gt": n0, "runs": []}
            for name, s in zip(names, sols):
                if g not in s:
                    row["runs"].append({"name": name, "tp": 0, "fp": None})
                    continue
                tp, fp, _ = s[g]
                f = LC.edge(tp, fp, n0)[0]
                run = {"name": name, "tp": tp, "fp": fp, "f1": f}
                if s is not base:
                    run["vs_base"] = block_verdict(tp - tp0, fp - fp0, f0, f)
                row["runs"].append(run)
            res["types"][g] = row
    res["sizes"] = {}
    keys = []
    for p in parsed:
        for k in p["sizes"]:
            if k not in keys:
                keys.append(k)
    for k in keys:
        vals = [p["sizes"].get(k) for p in parsed]
        if any(v is None for v in vals):
            continue
        n, tps = joint_bucket_counts(vals)
        res["sizes"][k] = {"gt_min": n, "tp": tps, "recall": vals}
    res["contour"] = {}
    classes = []
    for p in parsed:
        for c in p["contour"]:
            if c not in classes:
                classes.append(c)
    for c in classes:
        res["contour"][c] = [p["contour"].get(c) for p in parsed]
    return res


def _fmt_verdict(v: dict) -> str:
    kind = v["kind"]
    if kind == "same":
        return "без изменений"
    if kind == "dominates":
        return "лучше базы без оговорок"
    if kind == "dominated":
        return "хуже базы без оговорок"
    h, e = v["hit_share"], v["break_even"]
    if kind == "adds":
        tail = "окупается" if v["pays"] else "не окупается"
        return f"добавляет выдачу: попадает {h:.1%} при безубыточности базы {e:.1%} — {tail}"
    tail = ("вариант слишком осторожен: снятое стоило оставить" if v["pays"]
            else "снятое было шумом: правка верна")
    return f"снимает выдачу: снятое попадало {h:.1%} при безубыточности варианта {e:.1%} — {tail}"


def render(res: dict) -> str:
    lines = []
    for i, lg in enumerate(res["logs"]):
        tag = "база" if i == 0 else f"вариант {i}"
        lines.append(f"{tag}: {lg['name']}  final_score {lg['final_score']}")
    lines.append("")
    if not res["types"]:
        lines.append("по типам данных целые числа не согласовались")
    for g, row in res["types"].items():
        lines.append(f"{g} (эталон {row['gt']}):")
        for run in row["runs"]:
            if run.get("fp") is None:
                lines.append(f"  {run['name']}: попаданий нет")
                continue
            s = f"  {run['name']}: TP {run['tp']} FP {run['fp']} F1 {run['f1']:.3f}"
            if "vs_base" in run:
                v = run["vs_base"]
                s += f"  ΔTP {v['dtp']:+d} ΔFP {v['dfp']:+d} — {_fmt_verdict(v)}"
            lines.append(s)
    if any(res.get("fp_unknown") or []):
        lines.append("ложные у типов без попаданий (из FP ratio): "
                     + " / ".join(str(k) for k in res["fp_unknown"]))
    if res["sizes"]:
        lines.append("")
        lines.append("корзины площади (точки сюда не входят; эталон — самый мелкий согласованный):")
        for k, b in res["sizes"].items():
            n = b["gt_min"] if b["gt_min"] is not None else "?"
            tps = " / ".join("?" if t is None else str(t) for t in b["tp"])
            d = ""
            if len(b["tp"]) > 1 and None not in b["tp"][:1] and b["tp"][0] is not None:
                d = "  Δ " + " ".join("?" if t is None else f"{t - b['tp'][0]:+d}" for t in b["tp"][1:])
            lines.append(f"  {k:>7s} м²: эталон ≥ {n}, TP {tps}{d}")
    changed = {c: v for c, v in res["contour"].items() if len({repr(x) for x in v}) > 1}
    if changed:
        lines.append("")
        lines.append("контур по TP-парам, где он изменился (BIoU / HD95 м / PoLiS м):")
        for c, vals in changed.items():
            cells = ["нет TP-пар" if v is None else f"{v[0]:.4f} / {v[1]:.2f} / {v[2]:.2f}" for v in vals]
            lines.append(f"  {c}: " + "  →  ".join(cells))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("logs", nargs="+", type=Path, help="первый — база")
    ap.add_argument("--json", type=Path, help="записать разбор в JSON")
    args = ap.parse_args(argv)
    if len(args.logs) < 2:
        ap.error("нужно хотя бы два лога")
    texts = [p.read_text(encoding="utf-8", errors="replace") for p in args.logs]
    res = compare(texts, [p.name for p in args.logs])
    print(render(res))
    if args.json:
        args.json.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
