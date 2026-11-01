#!/usr/bin/env python3
"""koz3-w4-li-post-gate — honest leave-one-fold Li post levers on ConvNeXt-Li OOF.

CPU-only, internet-off. Dataset: elenakonkova/koz3-w2-oof (s0 per-fold OOF+GT).
Variants: OFF (base τ=0.15) | class_tau nested | self_nms@8 | self_nms@12 |
          class_tau+nms8 | class_tau+nms12.

Matching: soft cov>=0.5 + representative_point in GT (official materials/compute_metrics.py).
P = tp/(tp+0.5*fp). Selection only on train folds; report on held-out folds.

Dry-run on box:
  LI_POST_OOF_ROOT=/workspace/kaggle/datasets/koz3-w2-oof-light \
  LI_POST_OUT=/workspace/kaggle/push/li-post-gate/dryrun \
  python3 koz3-w4-li-post-gate.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from collections import defaultdict
from glob import glob
from pathlib import Path

from shapely.geometry import shape
from shapely.strtree import STRtree

OUT = Path(os.environ.get("LI_POST_OUT", "/kaggle/working"))
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "run.log"

SIZE_BINS = [
    (0, 100, "(0,100]"),
    (100, 500, "(100,500]"),
    (500, 1000, "(500,1000]"),
    (1000, 5000, "(1000,5000]"),
    (5000, 10000, "(5000,10000]"),
    (10000, 1e18, "(10000,+inf)"),
]
BASE_TAU = 0.15
GRID = [0.15, 0.17, 0.19, 0.21, 0.25, 0.30, 0.35, 0.45]
CLASSES = (
    "kurgany_povrezhdennye",
    "kurgany_tselye",
    "gorodishcha",
    "fortifikatsii",
    "arkhitektury",
)


def log(msg: str) -> None:
    line = msg.rstrip()
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def find_root() -> Path:
    env = os.environ.get("LI_POST_OOF_ROOT")
    if env and Path(env).exists():
        return Path(env)
    # Kaggle layouts
    cands = []
    for pat in (
        "/kaggle/input/koz3-w2-oof",
        "/kaggle/input/datasets/elenakonkova/koz3-w2-oof",
        "/kaggle/input/**/koz3-w2-oof",
    ):
        cands.extend(glob(pat, recursive=True))
    for c in cands:
        p = Path(c)
        if (p / "s0").exists() or list(p.glob("**/oof_cnx_c3-sc_f*.geojson")):
            return p
    # any oof under input
    hits = glob("/kaggle/input/**/oof_cnx_c3-sc_f0.geojson", recursive=True)
    if hits:
        return Path(hits[0]).parents[2]  # .../s0/<slug>/file → root with s0
    raise FileNotFoundError("koz3-w2-oof not found under /kaggle/input or LI_POST_OOF_ROOT")


def discover_folds(root: Path) -> list[tuple[str, Path, Path]]:
    """Return [(fold_id, oof_path, gt_path), ...] for s0 (prefer), else any seed."""
    seeds = ["s0", "s1", "s2"]
    for seed in seeds:
        base = root / seed
        if not base.exists():
            # light layout may be root/s0 already flattened differently
            base = root
        oofs = sorted(base.glob("**/oof_cnx_c3-sc_f*.geojson"))
        if not oofs and seed == "s0":
            oofs = sorted(root.glob("**/oof_cnx_c3-sc_f*.geojson"))
        pairs = []
        for op in oofs:
            name = op.name
            # oof_cnx_c3-sc_f{K}.geojson
            if "_f" not in name:
                continue
            fid = name.rsplit("_f", 1)[-1].replace(".geojson", "")
            gt = op.with_name(f"gt_li_cnx_c3-sc_f{fid}.geojson")
            if not gt.exists():
                # alt next to oof
                alts = list(op.parent.glob(f"gt_*_f{fid}.geojson"))
                if not alts:
                    continue
                gt = alts[0]
            pairs.append((fid, op, gt))
        # unique by fold id
        by = {}
        for fid, op, gt in pairs:
            by[fid] = (fid, op, gt)
        if len(by) >= 2:
            log(f"SEED={seed if (root/seed).exists() else 'auto'} folds={sorted(by)}")
            return [by[k] for k in sorted(by)]
    raise FileNotFoundError(f"no per-fold OOF+GT under {root}")


def load_fc(path: Path) -> list[dict]:
    g = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for feat in g.get("features") or []:
        try:
            geom = shape(feat["geometry"])
        except Exception:
            continue
        if geom.is_empty:
            continue
        props = feat.get("properties") or {}
        c = geom.centroid
        out.append(
            {
                "geom": geom,
                "cls": props.get("class_name") or props.get("class") or "",
                "conf": float(props.get("confidence", 1.0)),
                "cx": float(c.x),
                "cy": float(c.y),
                "area": float(geom.area),
            }
        )
    return out


def size_bucket(area: float) -> str:
    for lo, hi, name in SIZE_BINS:
        if lo < area <= hi or (lo == 0 and area <= hi):
            # (0,100] includes area==0 edge
            if area <= lo and lo > 0:
                continue
            if area > lo and area <= hi:
                return name
            if lo == 0 and area <= hi:
                return name
    return SIZE_BINS[-1][2]


def soft_match(preds, gts, cov_thr=0.5):
    """Official soft: cov>=thr AND rep-point in GT; greedy by conf; same class."""
    used = set()
    matched_gt = set()
    tp = fp = 0
    tp_by_cls = defaultdict(int)
    fp_by_cls = defaultdict(int)
    fn_by_cls = defaultdict(int)
    tp_by_sz = defaultdict(int)
    fp_by_sz = defaultdict(int)
    fn_by_sz = defaultdict(int)
    gt_cls_count = defaultdict(int)
    for g in gts:
        gt_cls_count[g["cls"]] += 1

    if not gts:
        for p in preds:
            fp += 1
            fp_by_cls[p["cls"]] += 1
            fp_by_sz[size_bucket(p["area"])] += 1
        return {
            "tp": 0,
            "fp": fp,
            "fn": 0,
            "by_cls": {},
            "by_size": {},
        }

    tree = STRtree([g["geom"] for g in gts])
    order = sorted(range(len(preds)), key=lambda i: -preds[i]["conf"])
    for i in order:
        p = preds[i]
        cand = tree.query(p["geom"])
        hit = None
        best = -1.0
        for j in cand:
            if j in used:
                continue
            g = gts[j]
            if g["cls"] != p["cls"]:
                continue
            inter = p["geom"].intersection(g["geom"]).area
            denom = min(p["geom"].area, g["geom"].area)
            if denom <= 0:
                continue
            cov = inter / denom
            if cov < cov_thr:
                continue
            rp = p["geom"].representative_point()
            if not g["geom"].covers(rp):
                continue
            if cov > best:
                best = cov
                hit = j
        if hit is not None:
            used.add(hit)
            matched_gt.add(hit)
            tp += 1
            tp_by_cls[p["cls"]] += 1
            tp_by_sz[size_bucket(gts[hit]["area"])] += 1
        else:
            fp += 1
            fp_by_cls[p["cls"]] += 1
            fp_by_sz[size_bucket(p["area"])] += 1

    fn = 0
    for j, g in enumerate(gts):
        if j not in matched_gt:
            fn += 1
            fn_by_cls[g["cls"]] += 1
            fn_by_sz[size_bucket(g["area"])] += 1

    by_cls = {}
    for cls in sorted(set(list(tp_by_cls) + list(fp_by_cls) + list(fn_by_cls) + list(gt_cls_count))):
        by_cls[cls] = {
            "tp": tp_by_cls[cls],
            "fp": fp_by_cls[cls],
            "fn": fn_by_cls[cls],
        }
    by_size = {}
    for _, _, name in SIZE_BINS:
        by_size[name] = {
            "tp": tp_by_sz[name],
            "fp": fp_by_sz[name],
            "fn": fn_by_sz[name],
        }
    return {"tp": tp, "fp": fp, "fn": fn, "by_cls": by_cls, "by_size": by_size}


def prf(tp, fp, fn):
    P = tp / (tp + 0.5 * fp) if (tp + 0.5 * fp) > 0 else 0.0
    R = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
    return {"P": P, "R": R, "F1": F1, "tp": tp, "fp": fp, "fn": fn}


def apply_thr(preds, floors):
    return [p for p in preds if p["conf"] >= floors.get(p["cls"], BASE_TAU)]


def nms(preds, radius):
    if radius <= 0:
        return preds
    r2 = radius * radius
    order = sorted(range(len(preds)), key=lambda i: -preds[i]["conf"])
    by = defaultdict(list)
    for i in order:
        by[preds[i]["cls"]].append(i)
    keep = set()
    for _cls, idxs in by.items():
        taken = []
        for i in idxs:
            p = preds[i]
            if any((p["cx"] - q["cx"]) ** 2 + (p["cy"] - q["cy"]) ** 2 <= r2 for q in taken):
                continue
            taken.append(p)
            keep.add(i)
    return [preds[i] for i in sorted(keep)]


def fit_nested(train):
    floors = {c: BASE_TAU for c in CLASSES}
    for cls in ("kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha", "fortifikatsii"):
        best_t, best_f = BASE_TAU, -1.0
        for t in GRID:
            floors[cls] = t
            tp = fp = fn = 0
            for pr, gt in train:
                m = soft_match(apply_thr(pr, floors), gt)
                tp += m["tp"]
                fp += m["fp"]
                fn += m["fn"]
            F = prf(tp, fp, fn)["F1"]
            if F > best_f:
                best_f = F
                best_t = t
        floors[cls] = best_t
    return floors


def merge_counts(dst, src):
    dst["tp"] += src["tp"]
    dst["fp"] += src["fp"]
    dst["fn"] += src["fn"]
    for cls, d in src["by_cls"].items():
        b = dst["by_cls"].setdefault(cls, {"tp": 0, "fp": 0, "fn": 0})
        b["tp"] += d["tp"]
        b["fp"] += d["fp"]
        b["fn"] += d["fn"]
    for sz, d in src["by_size"].items():
        b = dst["by_size"].setdefault(sz, {"tp": 0, "fp": 0, "fn": 0})
        b["tp"] += d["tp"]
        b["fp"] += d["fp"]
        b["fn"] += d["fn"]


def summarize(counts, off_f1):
    s = prf(counts["tp"], counts["fp"], counts["fn"])
    s["delta_F1"] = s["F1"] - off_f1
    s["by_cls"] = {
        cls: prf(d["tp"], d["fp"], d["fn"]) for cls, d in sorted(counts["by_cls"].items())
    }
    s["by_size"] = {
        sz: prf(d["tp"], d["fp"], d["fn"]) for sz, d in sorted(counts["by_size"].items())
    }
    return s


def main() -> int:
    t0 = time.monotonic()
    log("STATUS=START li-post-gate")
    try:
        root = find_root()
        log(f"OOF_ROOT={root}")
        folds = discover_folds(root)
        data = []
        for fid, op, gp in folds:
            log(f"load fold={fid} oof={op.name} gt={gp.name}")
            data.append((fid, load_fc(op), load_fc(gp)))

        base_floors = {c: BASE_TAU for c in CLASSES}
        variants = ("off", "thr_nested", "nms8", "nms12", "thr_nms8", "thr_nms12")
        agg = {
            v: {"tp": 0, "fp": 0, "fn": 0, "by_cls": {}, "by_size": {}} for v in variants
        }
        thr_maps = []

        for hold in range(len(data)):
            train = [(data[i][1], data[i][2]) for i in range(len(data)) if i != hold]
            floors = fit_nested(train)
            thr_maps.append({"hold": data[hold][0], "map": floors})
            pr, gt = data[hold][1], data[hold][2]
            pipelines = {
                "off": lambda p, fl=floors: apply_thr(p, base_floors),
                "thr_nested": lambda p, fl=floors: apply_thr(p, fl),
                "nms8": lambda p, fl=floors: nms(apply_thr(p, base_floors), 8.0),
                "nms12": lambda p, fl=floors: nms(apply_thr(p, base_floors), 12.0),
                "thr_nms8": lambda p, fl=floors: nms(apply_thr(p, fl), 8.0),
                "thr_nms12": lambda p, fl=floors: nms(apply_thr(p, fl), 12.0),
            }
            for name, fn in pipelines.items():
                merge_counts(agg[name], soft_match(fn(pr), gt))
            log(f"hold={data[hold][0]} floors={floors}")

        off_f1 = prf(agg["off"]["tp"], agg["off"]["fp"], agg["off"]["fn"])["F1"]
        results = {name: summarize(agg[name], off_f1) for name in variants}
        payload = {
            "protocol": "honest leave-one-fold on s0 cnx OOF; base_tau=0.15; soft cov>=0.5+rep",
            "n_folds": len(data),
            "fold_ids": [d[0] for d in data],
            "thr_maps": thr_maps,
            "variants": results,
            "elapsed_s": round(time.monotonic() - t0, 2),
            "note": "Li-layer only; not full stack A weighted score",
        }
        out_json = OUT / "li_post_honest.json"
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"WROTE {out_json}")
        for name, s in results.items():
            log(
                f"{name}: P={s['P']:.4f} R={s['R']:.4f} F1={s['F1']:.4f} "
                f"dF1={s['delta_F1']:+.4f} tp={s['tp']} fp={s['fp']} fn={s['fn']}"
            )
        log(f"STATUS=DONE elapsed_s={payload['elapsed_s']}")
        return 0
    except Exception:
        log("STATUS=FAIL")
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
