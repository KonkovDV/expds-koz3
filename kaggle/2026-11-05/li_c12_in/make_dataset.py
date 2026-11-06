#!/usr/bin/env python3
"""Приватный датасет dmitrijkv/koz3-cnx-li-prep12 из выхода prep12.py: cnx_relief/{участок}.npy|.json + manifest.json.

Проверка (PREREG c12-in, «Стоп до гейта»): первые 5 каналов каждого участка побайтно равны выходу ядра
dmitrijkv/koz3-cnx-li-prep (вход c5-in). Если хоть один участок не равен — датасет не собирается.
Манифест повторяет формат prep-выхода ядра: vendor = sha вендора каналов (как в manifest.json prep-выхода),
теги — 12 каналов c12. Ядро c12 сверяет и то, и другое.

    python make_dataset.py RELIEF12_DIR PREP5_DIR OUT_DIR      # PREP5_DIR — cnx_relief выхода koz3-cnx-li-prep
    kaggle datasets create -p OUT_DIR --dir-mode zip          # isPrivate: true в dataset-metadata.json
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np

TAGS12 = ("lrm_s", "lrm_l", "slope", "svf_r", "opp_r", "opn_r", "tpi5_r", "tri3_r", "curv_r", "slrm8_r", "dens_r", "int_r")
SLUG = "dmitrijkv/koz3-cnx-li-prep12"


def epsg(crs) -> str:
    """'urn:ogc:def:crs:EPSG::32637' → 'EPSG:32637' (как region_crs в ядре)."""
    m = re.search(r"EPSG:*\s*(\d+)", str(crs), re.I)
    return f"EPSG:{m.group(1)}" if m else str(crs)


def build(src: Path, ref: Path, out: Path) -> dict:
    man5 = json.loads((ref / "manifest.json").read_text(encoding="utf-8"))
    sites = sorted(p.stem for p in src.glob("*.npy"))
    if sorted(man5["regions"]) != sites:
        raise SystemExit(f"участки не совпали с prep-выходом c5: лишние {sorted(set(sites) - set(man5['regions']))}, "
                         f"нет {sorted(set(man5['regions']) - set(sites))}")
    vend_full = json.loads((ref / f"{sites[0]}.json").read_text(encoding="utf-8"))["vendor"]
    d = out / "cnx_relief"
    d.mkdir(parents=True, exist_ok=True)
    report = {}
    for s in sites:
        a = np.load(src / f"{s}.npy", mmap_mode="r")
        r = np.load(ref / f"{s}.npy", mmap_mode="r")
        if a.shape[0] != 12 or r.shape != a[:5].shape or not all(np.array_equal(r[i], a[i]) for i in range(5)):
            raise SystemExit(f"{s}: первые 5 каналов не равны выходу koz3-cnx-li-prep ({a.shape} vs {r.shape}) — стоп")
        js = json.loads((src / f"{s}.json").read_text(encoding="utf-8"))
        meta = dict(region=s, crs=epsg(js["crs"]), left=js["left"], top=js["top"], step=js["step"], shape=list(a.shape),
                    tags=list(TAGS12), site_stats=js["site_stats"], dem_res=js["dem_res"], dem_fill="delaunay",
                    backend=js["backend"], rvt="2.2.1 (inference/cnx/_deps)", vendor=vend_full,
                    valid_px=int((np.asarray(a[0]) > 0).sum()), t_s=None)
        dst = d / f"{s}.npy"
        if dst.exists():
            dst.unlink()
        try:
            os.link(src / f"{s}.npy", dst)                 # без копии: тот же файл
        except OSError:
            np.save(dst, np.asarray(a))
        (d / f"{s}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        report[s] = {"shape": list(a.shape), "first5_equal_c5_prep": True}
    man = dict(regions=sites, tags=list(TAGS12), vendor=man5["vendor"], backend=man5.get("backend"),
               rvt=man5.get("rvt"), res=man5.get("res", 0.5), dem_res=man5.get("dem_res", 0.1), dem_fill="delaunay",
               note="c12: каналы 0–4 побайтно = koz3-cnx-li-prep; 5–11 — relief_ext.EXTRA_TAGS того же вендора")
    (d / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "dataset-metadata.json").write_text(json.dumps(
        {"title": "koz3-cnx-li-prep12", "id": SLUG, "licenses": [{"name": "other"}], "isPrivate": True},
        ensure_ascii=False, indent=1), encoding="utf-8")
    return report


if __name__ == "__main__":
    rep = build(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
    print(json.dumps({"sites": len(rep), "all_first5_equal": all(v["first5_equal_c5_prep"] for v in rep.values())}))
