#!/usr/bin/env python3
"""Модальностный гейт W4 (docs/PREREG_w4_pix_clean.md, «GEO-ORG3 — модальностный гейт»): A против C.

    python scripts/pix_modality_gate.py --gt GT --a A18.geojson --c C.geojson --etgr 212 213 214 215 216 217 --out R.json

(a) участки без EtGr: sha256 канонической сериализации объектов участка у C и A совпадают;
(b) Δ_global (все участки, честные пороги) ≥ +0,009 и p_region ≥ 0,95 (бутстрэп по участкам среди EtGr);
(c) F1 каждого класса не ниже A.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from shapely.geometry import shape  # noqa: E402

import pix_clean_eval as ev  # noqa: E402
from expds import metric  # noqa: E402


def region_hashes(path: str) -> dict[str, str]:
    fc = json.loads(Path(path).read_text(encoding="utf-8"))
    rows: dict[str, list[str]] = {}
    for f in fc["features"]:
        pr = f.get("properties") or {}
        g = shape(f["geometry"]).wkb_hex if f.get("geometry") else ""
        rows.setdefault(pr.get("region_name", ""), []).append(json.dumps(pr, sort_keys=True, ensure_ascii=False) + "|" + g)
    return {r: hashlib.sha256("\n".join(sorted(v)).encode()).hexdigest() for r, v in rows.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--c", required=True)
    ap.add_argument("--etgr", nargs="+", required=True, help="номера участков с EtGr")
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    gt = ev._load(args.gt)
    regions = {f["properties"]["region_name"] for f in gt["features"]}
    gt = ev._load(args.gt, regions)
    folds = ev.folds_of(regions)
    et = {r for r in regions if r.split("_")[0] in set(args.etgr)}
    non = regions - et
    ha, hc = region_hashes(args.a), region_hashes(args.c)
    a_ok = {r: ha.get(r, "EMPTY") == hc.get(r, "EMPTY") for r in sorted(non)}

    scale = ev.CLASS_WEIGHTS and sum(ev.CLASS_WEIGHTS.get(c, 0.0) for c in ev.CLASSES) / sum(ev.CLASS_WEIGHTS.values())
    ac, _ = ev.honest(ev._load(args.a, regions), gt, folds)
    cc, _ = ev.honest(ev._load(args.c, regions), gt, folds)
    sa, sc = ev.summary(ac, folds), ev.summary(cc, folds)
    d_global = (sc["score2"] - sa["score2"]) * scale
    d_f1 = {c: sc["f1"].get(c, 0.0) - sa["f1"].get(c, 0.0) for c in ev.CLASSES}
    b_reg = metric.paired_bootstrap_counts(cc, ac, lambda r: r, args.boot, 0, regions=et)
    b_fam = metric.paired_bootstrap_counts(cc, ac, ev.site_family, args.boot, 0)
    d_et = (metric.score_counts(cc, et)[0] - metric.score_counts(ac, et)[0]) * scale
    d_non = (metric.score_counts(cc, non)[0] - metric.score_counts(ac, non)[0]) * scale
    gate = {"a_non_etgr_identical": all(a_ok.values()),
            "b_d_ge_0.009": d_global >= 0.009,
            "b_p_region_etgr_ge_0.95": b_reg.get("p_gt0", 0.0) >= 0.95,
            "c_f1_not_below_A": all(v >= 0 for v in d_f1.values())}
    gate["PASS"] = all(gate.values())
    res = {"a": args.a, "c": args.c, "etgr_regions": sorted(et), "non_etgr_identical": a_ok,
           "hashes_A": ha, "hashes_C": hc, "A_score2": sa["score2"], "C_score2": sc["score2"],
           "d_global": d_global, "d_f1": d_f1, "d_etgr_subset_global": d_et, "d_non_etgr_subset_global": d_non,
           "d_folds_global": {k: (sc["folds"][k] - sa["folds"][k]) * scale for k in sc["folds"]},
           "boot_region_etgr": b_reg, "boot_family_ref": b_fam, "gate": gate}
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("d_global", "d_f1", "d_etgr_subset_global", "d_non_etgr_subset_global",
                                          "d_folds_global", "gate")}, ensure_ascii=False))
    print("p_region_etgr", b_reg.get("p_gt0"), "p_family", b_fam.get("p_gt0"))


if __name__ == "__main__":
    main()
