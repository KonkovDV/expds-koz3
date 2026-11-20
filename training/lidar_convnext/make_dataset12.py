#!/usr/bin/env python3
"""Папка входов c12-in для ядра обучения из выхода prep12.py: cnx_relief/{участок}.npy|.json и manifest.json.

Манифест повторяет формат выхода
режима prep ядра: `vendor` — sha256 файлов вендора, от которых зависят каналы (как RELIEF_SHA в ядре), `tags` —
двенадцать каналов c12. Ядро сверяет и то, и другое. С `--ref` (выход prep для c5) каждый участок проверяется:
первые пять каналов должны быть побайтно равны входу c5, иначе папка не собирается.

    python training/lidar_convnext/make_dataset12.py work/relief12 work/prep12_dataset [--ref work/cnx_relief5]
    # Kaggle: kaggle datasets create -p work/prep12_dataset --dir-mode zip (приватный датасет)
    # Локально: CNX_PREP_DIR=work/prep12_dataset/cnx_relief
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TAGS12 = ("lrm_s", "lrm_l", "slope", "svf_r", "opp_r", "opn_r", "tpi5_r", "tri3_r", "curv_r", "slrm8_r", "dens_r", "int_r")


def epsg(crs) -> str:
    """'urn:ogc:def:crs:EPSG::32637' → 'EPSG:32637' (как region_crs в ядре)."""
    m = re.search(r"EPSG:*\s*(\d+)", str(crs), re.I)
    return f"EPSG:{m.group(1)}" if m else str(crs)


def relief_sha() -> tuple[dict, dict]:
    """(sha256 файлов каналов — как RELIEF_SHA ядра, полный манифест вендора)."""
    full = json.loads((HERE / "vendor" / "vendor_manifest.json").read_text(encoding="utf-8"))["sha256"]
    return {k: v for k, v in full.items() if k.startswith("expds") or k == "relief_ext.py"}, full


def build(src: Path, out: Path, ref: Path | None = None, slug: str = "") -> dict:
    sites = sorted(p.name[:-4] for p in src.glob("*.npy"))
    if not sites:
        raise SystemExit(f"{src}: нет *.npy (выход prep12.py)")
    if ref is not None:
        man5 = json.loads((ref / "manifest.json").read_text(encoding="utf-8"))
        if sorted(man5["regions"]) != sites:
            raise SystemExit(f"участки не совпали с выходом prep c5: лишние {sorted(set(sites) - set(man5['regions']))}, "
                             f"нет {sorted(set(man5['regions']) - set(sites))}")
    vend, vend_full = relief_sha()
    d = out / "cnx_relief"
    d.mkdir(parents=True, exist_ok=True)
    report = {}
    for s in sites:
        a = np.load(src / f"{s}.npy", mmap_mode="r")
        if a.shape[0] != 12:
            raise SystemExit(f"{s}: {a.shape[0]} каналов вместо 12")
        if ref is not None:
            r = np.load(ref / f"{s}.npy", mmap_mode="r")
            if r.shape != a[:5].shape or not all(np.array_equal(r[i], a[i]) for i in range(5)):
                raise SystemExit(f"{s}: первые 5 каналов не равны входу c5 ({a.shape} vs {r.shape}) — стоп")
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
        report[s] = {"shape": list(a.shape), "first5_equal_c5_prep": True if ref is not None else None}
    man = dict(regions=sites, tags=list(TAGS12), vendor=vend, backend=json.loads((d / f"{sites[0]}.json").read_text(
        encoding="utf-8"))["backend"], rvt="2.2.1", res=0.5, dem_res=0.1, dem_fill="delaunay",
               note="c12: каналы 0–4 = вход c5; 5–11 — relief_ext.EXTRA_TAGS того же вендора")
    (d / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    if slug:
        (out / "dataset-metadata.json").write_text(json.dumps(
            {"title": slug.split("/")[-1], "id": slug, "licenses": [{"name": "other"}], "isPrivate": True},
            ensure_ascii=False, indent=1), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", type=Path, help="выход prep12.py --root … --out SRC")
    ap.add_argument("out", type=Path)
    ap.add_argument("--ref", type=Path, help="выход режима prep ядра c5 (cnx_relief) для сверки первых пяти каналов")
    ap.add_argument("--slug", default="", help="OWNER/имя — записать dataset-metadata.json для Kaggle")
    a = ap.parse_args(argv)
    rep = build(a.src, a.out, a.ref, a.slug)
    print(json.dumps({"sites": len(rep), "checked_against_c5": a.ref is not None}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
