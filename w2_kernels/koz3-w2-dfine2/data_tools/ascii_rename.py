#!/usr/bin/env python3
"""Kaggle strips non-ASCII from dataset filenames: rename files_<part>/<region>__<file>.tif -> <NNN>__<kk>_SpOr.tif
and write manifest_<part>.json {ascii_name: {"region": region, "file": original file name}} (UTF-8 content is kept)."""
import json, os, sys, collections
part = sys.argv[1]; d = f"files_{part}"; man = {}; k = collections.Counter()
for n in sorted(os.listdir(d)):
    if not n.endswith(".tif") or "__" not in n: continue
    if n.split("__", 1)[0].isdigit(): continue  # already renamed
    reg, orig = n.split("__", 1); nnn = reg.split("_")[0]; k[nnn] += 1
    a = f"{nnn}__{k[nnn]:02d}_SpOr.tif"; assert a.isascii() and not os.path.exists(os.path.join(d, a))
    os.rename(os.path.join(d, n), os.path.join(d, a)); man[a] = {"region": reg, "file": orig}
old = json.load(open(f"{d}/manifest_{part}.json")) if os.path.exists(f"{d}/manifest_{part}.json") else {}
old.update(man); json.dump(old, open(f"{d}/manifest_{part}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
print(len(old), "files in manifest")
