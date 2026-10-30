#!/usr/bin/env python3
"""Golden e2e: сравнить два прогона inference/main.py (эталон и сборка) по счётчикам стадий и ответу.

    python scripts/golden_compare.py RUN_A RUN_B [--json out.json]

RUN_* — папка с run.log (лог main.py) и result.geojson. Код архивов не меняется: счётчики
берутся из строк, которые пишет любая сборка (pipeline «<регион>: <фаза> +N», протокол сторожа,
health), и из самого ответа (по классам, по слоям, геометрии). Код выхода 0 — совпало всё,
1 — есть расхождение (первая расходящаяся стадия печатается первой).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

TS = re.compile(r"^\d{4}-\d\d-\d\d [\d:,]+ ")
# Строки со временем/путями/средой — не сравниваются.
VOLATILE = re.compile(r"прошло|сек|\d+(\.\d+)? с\b|среда:|сторож сработает|потолок|найдено регионов|время|RSS|байт|cpu родителя|MemAvailable|конфиг SHA256")
NUM = re.compile(r"(\d+(?:\.\d+)?) с\b")


def stage_lines(log: Path) -> list[tuple[str, str]]:
    """[(стадия, значение)] в порядке лога: добавления фаз и протокол сторожа."""
    out = []
    for raw in log.read_text(encoding="utf-8", errors="replace").splitlines():
        line = TS.sub("", raw)
        m = re.match(r"\w+ expds\.pipeline: (\S+): (.+?) ([+-]\d+)", line)
        if m:
            out.append((f"{m.group(1)}:{m.group(2)}", m.group(3)))
            continue
        m = re.match(r"\w+ expds\.watchdog: (фазы|до порога по классам|до порога|записано|после порога[^:]*|вход):? (.*)", line)
        if m:
            val = NUM.sub("", m.group(2)) if m.group(1) == "фазы" else m.group(2)
            if m.group(1) in ("записано", "вход"):
                val = val.split(", файл")[0]
            out.append((f"protocol:{m.group(1)}", val.strip()))
            continue
        m = re.match(r"(WARNING|ERROR) (\S+): (.*)", line)
        if m:
            out.append((f"{m.group(1)}:{m.group(2)}", m.group(3)[:200]))
    return out


def geo_summary(path: Path, nd: int = 3) -> dict:
    fc = json.loads(path.read_text(encoding="utf-8"))
    per_class, per_layer, geoms = Counter(), Counter(), Counter()
    for f in fc.get("features", []):
        p = f.get("properties") or {}
        cls = p.get("class_name") or p.get("class") or "?"
        per_class[cls] += 1
        per_layer[str(p.get("markup_type") or p.get("layer") or "?")] += 1
        g = json.dumps(_round(f.get("geometry"), nd), sort_keys=True)
        geoms[(cls, hashlib.sha1(g.encode()).hexdigest()[:16])] += 1
    return {"n": sum(per_class.values()), "per_class": dict(sorted(per_class.items())),
            "per_layer": dict(sorted(per_layer.items())), "geoms": geoms,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _round(x, nd):
    if isinstance(x, float):
        return round(x, nd)
    if isinstance(x, list):
        return [_round(v, nd) for v in x]
    if isinstance(x, dict):
        return {k: _round(v, nd) for k, v in x.items()}
    return x


def compare(a: Path, b: Path) -> dict:
    sa, sb = stage_lines(a / "run.log"), stage_lines(b / "run.log")
    da, db = dict(sa), dict(sb)
    stages = []
    for k in list(dict.fromkeys([k for k, _ in sa] + [k for k, _ in sb])):
        stages.append({"stage": k, "A": da.get(k), "B": db.get(k), "same": da.get(k) == db.get(k)})
    ga, gb = geo_summary(a / "result.geojson"), geo_summary(b / "result.geojson")
    classes = sorted(set(ga["per_class"]) | set(gb["per_class"]))
    only_a, only_b = ga["geoms"] - gb["geoms"], gb["geoms"] - ga["geoms"]
    per_class = {c: {"A": ga["per_class"].get(c, 0), "B": gb["per_class"].get(c, 0),
                     "onlyA": sum(v for (k, _), v in only_a.items() if k == c),
                     "onlyB": sum(v for (k, _), v in only_b.items() if k == c)} for c in classes}
    first = next((s["stage"] for s in stages if not s["same"]), None)
    return {"A": str(a), "B": str(b), "identical_bytes": ga["sha256"] == gb["sha256"],
            "sha256": {"A": ga["sha256"], "B": gb["sha256"]},
            "geom_equal": not only_a and not only_b, "first_divergent_stage": first,
            "per_class": per_class, "stages": stages}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--json", type=Path)
    r = compare(ap.parse_args().a, ap.parse_args().b)
    args = ap.parse_args()
    print(f"A={r['A']}\nB={r['B']}\nбайты равны: {r['identical_bytes']}; геометрии равны: {r['geom_equal']}; "
          f"первая расходящаяся стадия: {r['first_divergent_stage']}")
    for c, v in r["per_class"].items():
        print(f"  {c:24s} A {v['A']:5d}  B {v['B']:5d}  только A {v['onlyA']:4d}  только B {v['onlyB']:4d}")
    for s in r["stages"]:
        if not s["same"]:
            print(f"  ≠ {s['stage']}: A={s['A']!r} B={s['B']!r}")
    if args.json:
        args.json.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0 if r["identical_bytes"] or (r["geom_equal"] and r["first_divergent_stage"] is None) else 1


if __name__ == "__main__":
    sys.exit(main())
