#!/usr/bin/env python3
"""Merge Fix C / area_tau ae candidates into stack_arch; leave non-ae layers untouched.

Reads optional gzip jsonl of pre-threshold ae_dfine preds (with box_area_m2 stash).
baseline: identity copy of stack.
area_tau: inject large ae gor/fort that pass relaxed base τ.
fixc: inject large ae_dfine gorodishcha with conf >= tau_large (default 0.20).

Threshold math inlined from expds.large_objects (no post.py / shapely needed here).
"""
from __future__ import annotations

import argparse, gzip, json, sys
from pathlib import Path

DFINE_AE_GROUPS = frozenset({"dfine_ae", "ae_dfine"})


def iter_stack_features(path: Path, chunk: int = 64 << 20):
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
                    raise ValueError("boundary")
            except ValueError:
                if eof:
                    raise
                more = fh.read(chunk)
                eof = not more
                buf = buf[pos:] + more
                pos = 0
                continue
            yield obj
            pos = end


def load_ae_oof(path: Path):
    if not path:
        return []
    opener = gzip.open if str(path).endswith(".gz") else open
    mode = "rt" if str(path).endswith(".gz") else "r"
    out = []
    with opener(path, mode, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def thr_area_tau(base: float, area: float, lo: dict) -> float:
    min_area = float(lo["min_area_m2"])
    relax = float(lo["tau_relax"])
    floor = float(lo["tau_floor"])
    if area < min_area or relax <= 0.0:
        return base
    frac = min(1.0, (area - min_area) / max(min_area, 1.0))
    return max(floor, base - relax * frac)


def thr_fixc(base: float, area: float, cn: str, tg: str, mk: str, lo: dict) -> float:
    if cn not in set(lo.get("dfine_ae_gor_classes") or ("gorodishcha",)):
        return base
    if mk not in set(lo.get("dfine_ae_gor_markups") or ("ae",)):
        return base
    if tg not in set(lo.get("dfine_ae_gor_groups") or DFINE_AE_GROUPS):
        return base
    if area < float(lo["min_area_m2"]):
        return base
    return min(base, float(lo["dfine_ae_gor_tau_large"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stack_in")
    ap.add_argument("stack_out")
    ap.add_argument("--mode", choices=("baseline", "area_tau", "fixc"), default="baseline")
    ap.add_argument("--ae-oof", default="")
    ap.add_argument("--tau-large", type=float, default=0.20)
    ap.add_argument("--min-area", type=float, default=5000.0)
    ap.add_argument("--base-gor-tau", type=float, default=1.01)
    ap.add_argument("--base-fort-tau", type=float, default=0.35)
    ap.add_argument("--tau-relax", type=float, default=0.15)
    ap.add_argument("--tau-floor", type=float, default=0.05)
    a = ap.parse_args()

    lo = {
        "min_area_m2": a.min_area,
        "tau_relax": a.tau_relax,
        "tau_floor": a.tau_floor,
        "dfine_ae_gor_tau_large": a.tau_large,
        "dfine_ae_gor_classes": ["gorodishcha"],
        "dfine_ae_gor_markups": ["ae"],
        "dfine_ae_gor_groups": ["dfine_ae", "ae_dfine"],
    }

    feats = list(iter_stack_features(Path(a.stack_in)))
    n_in = len(feats)
    if a.mode != "baseline":
        kept = []
        n_drop_exist = 0
        for f in feats:
            p = f.get("properties") or f
            tg = str(p.get("threshold_group") or "")
            mk = str(p.get("markup_type") or "")
            cn = p.get("class_name")
            if cn == "gorodishcha" and (
                tg in ("dfine_ae", "ae_dfine") or (mk == "ae" and "dfine" in tg)
            ):
                n_drop_exist += 1
                continue
            kept.append(f)
        feats = kept
    else:
        n_drop_exist = 0

    n_add = 0
    if a.mode != "baseline" and a.ae_oof:
        for f in load_ae_oof(Path(a.ae_oof)):
            p = dict(f.get("properties") or {})
            cn = p.get("class_name")
            conf = float(p.get("confidence") or 0)
            area = float(p.get("box_area_m2") or 0)
            tg = str(p.get("threshold_group") or "ae_dfine")
            mk = str(p.get("markup_type") or "ae")
            if a.mode == "fixc":
                if cn != "gorodishcha":
                    continue
                thr = thr_fixc(a.base_gor_tau, area, cn, tg, mk, lo)
            else:  # area_tau
                if cn not in ("gorodishcha", "fortifikatsii"):
                    continue
                base = a.base_gor_tau if cn == "gorodishcha" else a.base_fort_tau
                thr = thr_area_tau(base, area, lo)
            if conf < thr:
                continue
            p.setdefault("threshold_group", tg)
            p.setdefault("markup_type", mk)
            p["box_area_m2"] = area
            p["_large_gate_inject"] = a.mode
            feats.append({"type": "Feature", "geometry": f["geometry"], "properties": p})
            n_add += 1

    out = {"type": "FeatureCollection", "features": feats}
    Path(a.stack_out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    stats = {
        "mode": a.mode,
        "n_in": n_in,
        "n_out": len(feats),
        "n_drop_exist_ae_gor": n_drop_exist,
        "n_add": n_add,
        "tau_large": a.tau_large,
        "min_area": a.min_area,
    }
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
