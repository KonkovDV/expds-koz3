#!/usr/bin/env python3
"""Decode dumped li prob maps (EXPDS_LI_DUMP=...) and splice them into the family JSONL stack.

Replaces ONLY the features of one threshold group (default: li) by the re-decoded ones; every other
feature is copied unchanged. Output = family JSONL dir for bot-10 scripts/family_curves.py.

  --mode legacy  as pipeline._li_with_model today: kurgans = relief candidates (dumped cand_x/cand_y)
                 as circle_m circles in both classes with the net prob (5 px max); other classes = masks.
  --mode peaks   expds.li_decode.decode_li (net-peak kurgans + centre-NMS + dual, gor/fort points).
  --mode hybrid  relief candidates kept as kurgan locations, but class choice/dual and confidence of
                 li_decode (share rule); gor/fort points as in peaks.

    python scripts/li_decode_dump.py --dump D --fam-src FAM --fam-dst OUT --mode peaks [--config cfg.toml]
"""
from __future__ import annotations

import argparse
import json
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from expds import li_decode as L  # noqa: E402
from expds.naming import site_family  # noqa: E402


def affine(t):
    from rasterio import Affine
    return Affine(*[float(v) for v in t[:6]])


def hybrid(probs, classes, tr, crs, region, source, cand, li_cfg, circle_m):
    """Relief candidates as locations + li_decode share/dual rule; non-kurgan classes as in peaks."""
    from expds import candidates
    from expds.crs import WEB_MERCATOR, to_crs
    from shapely import buffer, points

    prm = L.params(li_cfg)
    kur = [c for c in L.KURGANS if c in classes]
    other = [c for c in classes if c not in kur]
    feats = L.decode_li(probs[[classes.index(c) for c in other]], other, tr, crs, float(tr.a), region, "li",
                        source, li_cfg) if other else []
    if not kur or cand is None or len(cand["x"]) == 0:
        return feats
    kp = probs[[classes.index(c) for c in kur]].astype(np.float32)
    s = candidates.sample_probs(cand, kp, tr)                       # N x 2, as legacy
    share = s / np.maximum(s.sum(1, keepdims=True), 1e-12)
    geoms = buffer(points(np.asarray(cand["x"]), np.asarray(cand["y"])), float(circle_m), quad_segs=4)
    if crs and crs != WEB_MERCATOR:
        geoms = to_crs(geoms, crs, WEB_MERCATOR)
    best = share.argmax(1)
    for i, g in enumerate(geoms):
        for k, cls in enumerate(kur):
            if k == best[i] or share[i, k] >= float(prm["dual"]):
                feats.append({"geom": g, "class_name": cls, "confidence": float(s[i].max() * share[i, k]),
                              "region_name": region, "markup_type": "li", "source": source, "original_crs": crs})
    return feats


def legacy(probs, classes, tr, crs, region, source, cand, circle_m):
    from expds import candidates
    from expds.infer_nn import probs_to_features

    kur = [c for c in L.KURGANS if c in classes] if cand is not None else []
    rest = [c for c in classes if c not in kur]
    feats = probs_to_features(probs[[classes.index(c) for c in rest]], rest, tr, crs, tr.a, region, "li",
                              source) if rest else []
    if kur:
        s = candidates.sample_probs(cand, probs[[classes.index(k) for k in kur]].astype(np.float32), tr)
        p = candidates.CandidateParams(circle_m=circle_m)
        feats += candidates.to_features(cand, np.column_stack([np.zeros(len(s), np.float32), s]), crs, region, p,
                                        classes=tuple(kur), source=source)
    return feats


def main() -> None:
    from shapely.geometry import mapping

    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--fam-src", required=True)
    ap.add_argument("--fam-dst", required=True)
    ap.add_argument("--mode", choices=["legacy", "peaks", "hybrid"], required=True)
    ap.add_argument("--group", default="li", help="threshold group of the stack to replace")
    ap.add_argument("--config", default=None, help="TOML with [li] / [li.peaks] (default: configs/default.toml)")
    ap.add_argument("--circle-m", type=float, default=2.0, help="legacy kurgan circle radius (default.toml:210)")
    a = ap.parse_args()
    cfg = tomllib.loads(Path(a.config or ROOT / "configs" / "default.toml").read_text(encoding="utf-8"))
    li_cfg = {**cfg.get("li", {}), "decode": "peaks"}
    new: dict[str, list] = defaultdict(list)
    n_maps = 0
    for z in sorted(Path(a.dump).glob("*.npz")):
        d = np.load(z, allow_pickle=False)
        probs = d["probs"].astype(np.float32)
        classes = [str(c) for c in d["classes"]]
        tr, crs = affine(d["transform"]), (str(d["crs"]) or None)
        region, source = str(d["region"]), str(d["source"])
        cand = {"x": d["cand_x"], "y": d["cand_y"]} if "cand_x" in d.files else None
        if a.mode == "legacy":
            feats = legacy(probs, classes, tr, crs, region, source, cand, a.circle_m)
        elif a.mode == "peaks":
            feats = L.decode_li(probs, classes, tr, crs, float(tr.a), region, "li", source, li_cfg)
        else:
            feats = hybrid(probs, classes, tr, crs, region, source, cand, li_cfg, a.circle_m)
        for f in feats:
            pr = {"class_name": f["class_name"], "confidence": round(float(f["confidence"]), 6),
                  "region_name": region, "crs": "EPSG:3857", "markup_type": "li", "original_crs": crs}
            if a.group != "li":
                pr["threshold_group"] = a.group
            new[site_family(region)].append({"type": "Feature", "geometry": mapping(f["geom"]), "properties": pr})
        n_maps += 1
    dst = Path(a.fam_dst)
    dst.mkdir(parents=True, exist_ok=True)
    src_fams = {p.stem for p in Path(a.fam_src).glob("*.jsonl")}
    kept = dropped = 0
    for fam in sorted(src_fams | set(new)):
        with open(dst / f"{fam}.jsonl", "w", encoding="utf-8") as fo:
            sp = Path(a.fam_src) / f"{fam}.jsonl"
            if sp.exists():
                for line in open(sp, encoding="utf-8"):
                    pr = json.loads(line).get("properties") or {}
                    if str(pr.get("threshold_group") or pr.get("markup_type") or "") == a.group:
                        dropped += 1
                        continue
                    fo.write(line if line.endswith("\n") else line + "\n")
                    kept += 1
            for f in new.get(fam, []):
                fo.write(json.dumps(f, ensure_ascii=False) + "\n")
    print(json.dumps({"maps": n_maps, "new": sum(map(len, new.values())), "kept": kept, "dropped": dropped,
                      "mode": a.mode, "group": a.group}))


if __name__ == "__main__":
    main()
