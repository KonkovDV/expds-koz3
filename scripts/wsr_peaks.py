#!/usr/bin/env python3
"""All sp_or peaks of one centre map, region by region (PROMPT_WITHIN_SITE_RANK_20261031).

Same path as `scripts/nn_oof.py --weights` (predict_region -> merge_within_markup -> finalize), but without the
20000-object output cap of writer.enforce_limits, which otherwise drops whole families from a train-wide dump.
usage: python scripts/wsr_peaks.py --root ROOT --weights map.pt --out peaks.geojson [--families F ...] [--exclude-families F ...]"""
from __future__ import annotations
import argparse, json, logging, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from expds import post, writer
from expds.catalog import scan_root
from expds.infer_nn import Segmenter
from expds.naming import site_family


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--weights", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--families", nargs="*", default=None); ap.add_argument("--exclude-families", nargs="*", default=[])
    ap.add_argument("--tta", type=int, default=1, help="[tta] sp_or of the submission is 1 (no entry)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.path.insert(0, str(ROOT / "scripts"))
    from nn_oof import predict_region
    seg = Segmenter(Path(a.weights)); seg.tta = a.tta
    feats, n_reg, t0 = [], 0, time.time()
    for reg in scan_root(a.root):
        fam = site_family(reg.name)
        if not reg.rasters.get("sp_or") or fam in a.exclude_families or (a.families and fam not in a.families):
            continue
        try:
            f = predict_region(seg, reg, "sp_or", None)
        except Exception:
            logging.exception("%s: skipped", reg.name); continue
        feats += post.finalize(post.merge_within_markup(f)); n_reg += 1
    fc = writer.to_feature_collection(feats)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"written": len(fc["features"]), "regions": n_reg, "weights": a.weights, "s": round(time.time() - t0)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
