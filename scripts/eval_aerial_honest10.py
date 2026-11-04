#!/usr/bin/env python3
"""Честный honest-10 прогон с акцентом на слой аэрофото (bot-3).

Обёртка над scripts/scoreboard.py --honest 10. Печатает ALL и (если есть в JSON)
группы ae / li_ae; критерий приёмки bot-3: рост ae recall/F1 без провала ALL
слоёв > 0,01 и честная приёмка scoreboard (ΔF1≥0,002, p_gt0≥0,8).

Пример (пути ПК Дмитрия):
  python scripts/eval_aerial_honest10.py \\
    --work F:/expds3/work --train F:/2/train-expds-3 \\
    --out notes/bot3_aerial_honest10.json

С ослабленными порогами: скопировать configs/default.toml → /tmp/ae_relax.toml,
поставить [ae_recall_relax] enabled=true, прогнать пайплайн с этим конфигом,
затем этот скрипт на новых OOF/предсказаниях (--add / --replace).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCOREBOARD = ROOT / "scripts" / "scoreboard.py"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, help="папка с предсказаниями / стеком scoreboard")
    ap.add_argument("--train", required=True, help="корень трейна")
    ap.add_argument("--gt", default=None)
    ap.add_argument("--geo", default=None)
    ap.add_argument("--out", default=None, help="JSON результат (по умолчанию <work>/bot3_aerial_honest10.json)")
    ap.add_argument("--honest", type=int, default=10)
    ap.add_argument("--honest-by", choices=("family", "region"), default="family")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--add", action="append", default=[], help="как у scoreboard --add")
    ap.add_argument("--replace", action="append", default=[], help="как у scoreboard --replace")
    ap.add_argument("--base-add", action="append", default=[], help="как у scoreboard --base-add")
    ap.add_argument("--expect", type=float, default=None)
    args, extra = ap.parse_known_args()

    out = Path(args.out) if args.out else Path(args.work) / "bot3_aerial_honest10.json"
    cmd = [sys.executable, str(SCOREBOARD), "--work", args.work, "--train", args.train,
           "--honest", str(args.honest), "--honest-by", args.honest_by,
           "--bootstrap", str(args.bootstrap), "--out", str(out)]
    if args.gt:
        cmd += ["--gt", args.gt]
    if args.geo:
        cmd += ["--geo", args.geo]
    if args.expect is not None:
        cmd += ["--expect", str(args.expect)]
    for a in args.base_add:
        cmd += ["--base-add", a]
    for a in args.add:
        cmd += ["--add", a]
    for a in args.replace:
        cmd += ["--replace", a]
    cmd += extra

    print("aerial honest-10:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if proc.returncode != 0:
        return proc.returncode
    if not out.is_file():
        print("нет JSON:", out, file=sys.stderr)
        return 2
    data = json.loads(out.read_text(encoding="utf-8"))
    honest = data.get("honest") or {}
    print("--- bot-3 aerial summary ---", flush=True)
    print("honest.accept:", honest.get("accept"), "folds:", honest.get("folds"), flush=True)
    if honest.get("base") is not None and honest.get("after") is not None:
        labels = ("F1", "P", "R")
        for i, lab in enumerate(labels):
            b, a = honest["base"][i], honest["after"][i]
            print(f"  honest {lab}: {b:.6f} → {a:.6f} (Δ {a - b:+.6f})", flush=True)
    # Подсказка по слою ae, если scoreboard положил разбивку
    for key in ("by_markup", "markup", "groups", "ae"):
        if key in data:
            print(f"extra[{key}]:", json.dumps(data[key], ensure_ascii=False)[:500], flush=True)
    print("JSON:", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
