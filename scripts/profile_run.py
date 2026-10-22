#!/usr/bin/env python3
"""Хронометраж inference/main.py: время фаз по логу и пиковая память всего дерева процессов.

    python scripts/profile_run.py F:/expds3/mock_val F:/expds3/work/mock_out/result.geojson \
        --report F:/expds3/work/mock_out/profile.json

Фазы берутся из строк лога «геофизика», «лидар», «оптика» и контрольных записей.
Память — сумма RSS процесса и потомков, опрос раз в 0,5 с.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]


def tree_rss(proc: psutil.Process) -> int:
    total = 0
    for p in [proc] + proc.children(recursive=True):
        try:
            total += p.memory_info().rss
        except psutil.Error:
            pass
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir")
    ap.add_argument("output")
    ap.add_argument("--report", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = out.with_suffix(".log")
    t0 = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen([sys.executable, "-X", "utf8", str(ROOT / "inference" / "main.py"),
                                 args.input_dir, str(out)], stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        ps = psutil.Process(proc.pid)
        peak = {"rss": 0, "t": 0.0}

        def sample():
            while proc.poll() is None:
                try:
                    rss = tree_rss(ps)
                except psutil.Error:
                    break
                if rss > peak["rss"]:
                    peak.update(rss=rss, t=time.monotonic() - t0)
                time.sleep(0.5)

        th = threading.Thread(target=sample, daemon=True)
        th.start()
        code = proc.wait()
        th.join(timeout=2)
    total = time.monotonic() - t0
    text = log_path.read_text(encoding="utf-8", errors="replace")
    marks = {}
    for line in text.splitlines():
        m = re.search(r"контрольная запись.*?(геофизика|лидар|конец)", line)
        if m:
            marks[m.group(1)] = line[:23]
    per_region = {}
    for m in re.finditer(r"(region_\w+): (геофизика|лидар|оптика) ([+-]\d+)(?:, ([\d.]+) с)?", text):
        per_region.setdefault(m.group(2), []).append(
            {"region": m.group(1), "objects": int(m.group(3)), "seconds": float(m.group(4) or 0)})
    report = {"exit_code": code, "seconds_total": round(total, 1), "peak_rss_gb": round(peak["rss"] / 2**30, 2),
              "peak_at_s": round(peak["t"], 1), "checkpoints": marks,
              "lidar_slowest": sorted(per_region.get("лидар", []), key=lambda r: -r["seconds"])[:5],
              "log": str(log_path)}
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
