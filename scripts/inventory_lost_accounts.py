#!/usr/bin/env python3
"""Опись ресурсов Kaggle на заблокированных 04.11.2026 аккаунтах: что это было и есть ли копия.

    KAGGLE_TOKEN=... python scripts/inventory_lost_accounts.py [--drive-names FILE] > out.md

Источник списка — упоминания `<аккаунт>/<slug>` в git (кроме snapshot/). Для каждого slug: первая строка контекста из
репозитория, есть ли ресурс с тем же slug у dmitrijkv (ядро или датасет), есть ли файл/папка с этим slug в списке имён
Google Drive (--drive-names: по одному имени на строку) и в боксе /workspace/koz3. Токен не печатается.
"""
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOST = ("elenakonkova", "aaaa0000004", "usertest1234", "aaaadg4gg4g4g")
NOT_RESOURCES = {"cpu", "cpu-2", "gpu-1", "gpu-2", "dmitrijkv", *LOST}   # слоты аренды и пары аккаунтов в журналах


def git_refs():
    out = subprocess.run(["git", "-C", str(ROOT), "grep", "-nE", rf"({'|'.join(LOST)})/[A-Za-z0-9_.-]+", "--", ".", ":!snapshot"],
                         capture_output=True, text=True).stdout
    refs = defaultdict(list)
    for line in out.splitlines():
        path, _, text = line.split(":", 2)
        for acc, slug in re.findall(rf"({'|'.join(LOST)})/([A-Za-z0-9_.-]+)", text):
            slug = slug.rstrip(".")
            if slug in NOT_RESOURCES:
                continue
            refs[(acc, slug)].append((path, text.strip()))
    return refs


def kaggle_names():
    tok = os.environ.get("KAGGLE_TOKEN")
    if not tok:
        return set()
    names = set()
    for url in ["https://www.kaggle.com/api/v1/kernels/list?user=dmitrijkv&pageSize=100&page={p}",
                "https://www.kaggle.com/api/v1/datasets/list?group=my&pageSize=100&page={p}"]:
        for p in range(1, 6):
            out = subprocess.run(["curl", "-sS", "-m", "60", "-H", f"Authorization: Bearer {tok}", url.format(p=p)], capture_output=True, text=True).stdout
            try:
                L = json.loads(out)
            except Exception:
                break
            if not isinstance(L, list) or not L:
                break
            names |= {x["ref"].split("/", 1)[1] for x in L if x.get("ref")}
    return names


def main():
    drive = set()
    if "--drive-names" in sys.argv:
        drive = {l.strip() for l in open(sys.argv[sys.argv.index("--drive-names") + 1], encoding="utf-8") if l.strip()}
    kag = kaggle_names()
    box = Path("/workspace/koz3")
    box_names = {p.name for p in box.rglob("*")} if box.exists() else set()
    refs = git_refs()
    print("| аккаунт | ресурс | упоминаний | копия на dmitrijkv | на Drive | в боксе | первое упоминание |")
    print("|---|---|---|---|---|---|---|")
    for (acc, slug), hits in sorted(refs.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        in_drive = any(slug in n for n in drive)
        in_box = any(slug in n for n in box_names)
        path, text = hits[0]
        ctx = re.sub(r"\s+", " ", text)[:140].replace("|", "/")
        print(f"| {acc} | `{slug}` | {len(hits)} | {'да' if slug in kag else '—'} | {'да' if in_drive else '—'} | {'да' if in_box else '—'} | `{path}`: {ctx} |")


if __name__ == "__main__":
    main()
