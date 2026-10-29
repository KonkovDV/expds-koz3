"""GEO-ORG3 OOF: EtGr object_poly по бейзлайну организаторов 02 (expds/pix_org_et.py).

    python scripts/pix_org_et_oof.py cand --root TRAIN --cache OOF            # oof/orget/cand/{reg}.pkl
    python scripts/pix_org_et_oof.py fold --root TRAIN --cache OOF --seed 0   # geo5 F3–F5 → oof/orget/s0/{reg}.pkl
    python scripts/pix_org_et_oof.py full --cache OOF --out models/pixgbm/clean/org --seed 0
Порог каждой модели («карта»/«профиль») — внутренний LOSO по участкам обучения. Затем compose --orget-tag s0.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from expds import pix_org_et as E  # noqa: E402
from expds.catalog import scan_root  # noqa: E402
from expds.crs import to_crs  # noqa: E402
from expds.gt import read_annotation  # noqa: E402
from pix_clean_train import gt_files_for  # noqa: E402

FOLDS = {"F3": ["208", "209", "210", "215"], "F4": ["211", "212", "216"], "F5": ["213", "214", "217"]}


def cmd_cand(a):
    d = Path(a.cache) / "orget" / "cand"
    d.mkdir(parents=True, exist_ok=True)
    for reg in scan_root(Path(a.root)):
        rs = reg.rasters.get("etgr") or []
        if not rs or (a.regions and reg.name.split("_")[0] not in a.regions):
            continue
        f = d / f"{reg.name}.pkl"
        if f.exists():
            continue
        t = time.monotonic()
        cs = E.region_cands(rs, reg.boundaries)
        for c in cs:
            gts = []
            for gf in gt_files_for(c["path"], reg, "object_poly", "etgr"):
                for o in read_annotation(gf, "object_poly", "etgr", reg):
                    g = to_crs(o["geom"], "EPSG:3857", c["crs"]).buffer(0)
                    if g.geom_type in ("Polygon", "MultiPolygon") and not g.is_empty:
                        gts.append(g)
            c["gts"], c["y"] = gts, E.label(c["polys"], gts)
        f.write_bytes(pickle.dumps(cs))
        print(f"cand {reg.name}: растров {len(cs)}, кандидатов {sum(len(c['polys']) for c in cs)} "
              f"(+{sum(int(c['y'].sum()) for c in cs)}), GT {sum(len(c['gts']) for c in cs)}, "
              f"{time.monotonic() - t:.0f} с", flush=True)


def _load(cache):
    return {p.stem: pickle.loads(p.read_bytes()) for p in sorted((Path(cache) / "orget" / "cand").glob("*.pkl"))}


def _fit_kind(C, regs, kind, seed, threads):
    cs = [c for r in regs for c in C[r] if c["kind"] == kind]
    if not cs:
        return None
    return E.fit(np.concatenate([c["X"] for c in cs]), np.concatenate([c["y"] for c in cs]), seed, threads)


def _inner_thr(C, regs, kind, seed, threads):
    """Внутренний LOSO: OOF-скоры участков обучения → порог по пулу (как у организаторов)."""
    use = [r for r in regs if any(c["kind"] == kind for c in C[r])]
    cs, ss, gs = [], [], []
    for r in use:
        m = _fit_kind(C, [x for x in use if x != r], kind, seed, threads)
        for c in C[r]:
            if c["kind"] == kind:
                cs.append(c); ss.append(E.predict(m, c["X"])); gs.append(c["gts"])
    return E.pick_threshold(cs, ss, gs) if cs else 0.5


def cmd_fold(a):
    C = _load(a.cache)
    out = Path(a.cache) / "orget" / f"s{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    log = {}
    for k, ex in FOLDS.items():
        test = [n for n in C if n.split("_")[0] in ex]
        train = [n for n in C if n not in test]
        t = time.monotonic()
        res = {n: [] for n in test}
        for kind in E.KINDS:
            m = _fit_kind(C, train, kind, a.seed, a.threads)
            thr = _inner_thr(C, train, kind, a.seed, a.threads)
            log[f"{k}_{kind}"] = thr
            for n in test:
                for c in C[n]:
                    if c["kind"] == kind:
                        res[n] += E.to_feats(c, E.predict(m, c["X"]), thr, n)
        for n in test:
            (out / f"{n}.pkl").write_bytes(pickle.dumps({"new": res[n], "pst": {}}))
        print(f"fold {k} s{a.seed}: {test} пороги {[log[f'{k}_{x}'] for x in E.KINDS]} {time.monotonic() - t:.0f} с",
              flush=True)
    (out / "thr.json").write_text(json.dumps(log))


def cmd_full(a):
    C = _load(a.cache)
    regs = sorted(C)
    o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
    thr = {}
    for kind in E.KINDS:
        m = _fit_kind(C, regs, kind, a.seed, a.threads)
        if m is not None:
            m.save_model(str(o / f"org_etgr_{kind}.txt"))
        thr[kind] = _inner_thr(C, regs, kind, a.seed, a.threads)
    (o / "org_etgr_meta.json").write_text(json.dumps({"seed": a.seed, "regions": regs, "thr": thr,
                                                       "spec": "GEO-ORG3, docs/PREREG_w4_pix_clean.md"},
                                                      ensure_ascii=False))
    print("ok", o, thr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for n in ("cand", "fold", "full"):
        p = sp.add_parser(n)
        p.add_argument("--root", default="/workspace/koz3/w4geo/train")
        p.add_argument("--cache", required=True)
        p.add_argument("--regions", nargs="*")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--threads", type=int, default=2)
        p.add_argument("--out")
    a = ap.parse_args()
    {"cand": cmd_cand, "fold": cmd_fold, "full": cmd_full}[a.cmd](a)


if __name__ == "__main__":
    main()
