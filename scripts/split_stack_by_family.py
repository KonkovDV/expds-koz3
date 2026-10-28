#!/usr/bin/env python3
"""Потоково режет большой стек (FeatureCollection) на JSONL по семьям участков.

Память ~ размер куска чтения (64 МБ), не размер файла. Нужен, когда стек
(например, дамп `scoreboard.py --dump-stack`, 1,1 ГБ) не помещается в ОЗУ.

    python scripts/split_stack_by_family.py --stack stack_arch.geojson --out work/fam
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def iter_features(path: Path, chunk: int = 64 << 20):
    dec = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as fh:
        buf = fh.read(chunk)
        start = buf.index('"features"')
        pos = buf.index("[", start) + 1
        eof = False
        while True:
            while pos < len(buf) and buf[pos] in " \n\r\t,":
                pos += 1
            if pos < len(buf) and buf[pos] == "]":
                return
            try:
                obj, end = dec.raw_decode(buf, pos)
                if end >= len(buf) and not eof:
                    raise ValueError("на границе куска")
            except ValueError:
                if eof:
                    raise
                more = fh.read(chunk)
                eof = not more
                buf = buf[pos:] + more
                pos = 0
                continue
            yield obj, buf[pos:end]
            pos = end


def main() -> None:
    from expds.naming import site_family

    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    handles, cnt = {}, Counter()
    for feat, raw in iter_features(Path(a.stack)):
        reg = str((feat.get("properties") or {}).get("region_name", "default_source"))
        fam = site_family(reg)
        fh = handles.get(fam)
        if fh is None:
            fh = handles[fam] = open(out / f"{fam}.jsonl", "a", encoding="utf-8")
        fh.write(raw + "\n")
        cnt[fam] += 1
    for fh in handles.values():
        fh.close()
    (out / "_counts.json").write_text(json.dumps(dict(cnt.most_common()), ensure_ascii=False, indent=1), encoding="utf-8")
    print("семей", len(cnt), "объектов", sum(cnt.values()))


if __name__ == "__main__":
    main()
