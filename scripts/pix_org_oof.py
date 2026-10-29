"""GEO-ORG1 OOF: кандидаты MagGr по бейзлайну организаторов (expds/pix_org.py), фолдовые модели, признаки для compose.

    python scripts/pix_org_oof.py cand  --root TRAIN --cache OOF            # кандидаты + метки, oof/org/cand/{reg}.pkl
    python scripts/pix_org_oof.py fold  --root TRAIN --cache OOF --seed 0   # модель без участков фолда → oof/org/s0/{reg}.pkl
    python scripts/pix_org_oof.py full  --cache OOF --out models/pix_clean --seed 0   # модели на всех участках
Затем: scripts/pix_clean_oof.py compose ... --org-tag s0.
"""
from __future__ import annotations

import argparse
import json
import pickle
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from expds import pix_org  # noqa: E402
from expds.catalog import scan_root  # noqa: E402
from expds.crs import to_crs  # noqa: E402
from expds.gt import read_annotation  # noqa: E402

FOLDS = {"F1": ["200"], "F2": [str(i) for i in range(201, 208)], "F3": ["208", "209", "210", "215"],
         "F4": ["211", "212", "216"], "F5": ["213", "214", "217"]}


def _peak():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _regions(root, only=None):
    regs = [r for r in scan_root(Path(root)) if r.rasters.get("maggr")]
    return [r for r in regs if not only or r.name in only or r.name.split("_")[0] in only]


def cmd_cand(a):
    d = Path(a.cache) / "org" / "cand"
    d.mkdir(parents=True, exist_ok=True)
    for reg in _regions(a.root, a.regions):
        f = d / f"{reg.name}.pkl"
        if f.exists():
            continue
        t = time.monotonic()
        c = pix_org.candidates(reg)
        if c is None:
            print("нет MagGr:", reg.name); continue
        pts, polys = [], []
        for path, cls, mt in reg.annotations:
            if mt != "maggr":
                continue
            for o in read_annotation(path, cls, mt, reg):
                g = to_crs(o["geom"], "EPSG:3857", c["crs"])
                (pts if cls == "finds_points" else polys).append(g)
        gxy = np.array([(g.centroid.x, g.centroid.y) for g in pts]).reshape(-1, 2)
        polys = [g.buffer(0) for g in polys if g.geom_type in ("Polygon", "MultiPolygon")]
        c["py"] = pix_org.label_points(c["pxy"], gxy)
        c["cy"] = pix_org.label_polys(c["polys"], polys)
        c["n_gt"] = (len(gxy), len(polys))
        f.with_suffix(".part").write_bytes(pickle.dumps(c)); f.with_suffix(".part").replace(f)
        print(f"cand {reg.name}: точек {len(c['pxy'])} (+{int(c['py'].sum())}), полигонов {len(c['polys'])} "
              f"(+{int(c['cy'].sum())}), GT {c['n_gt']}, {time.monotonic() - t:.0f} с, пик {_peak():.0f} МБ", flush=True)


def _load_cands(cache):
    return {p.stem: pickle.loads(p.read_bytes()) for p in sorted((Path(cache) / "org" / "cand").glob("*.pkl"))}


def _fit(cs, seed, threads):
    m = {}
    for cls, kx, ky in (("finds_points", "pX", "py"), ("object_poly", "cX", "cy")):
        X = np.concatenate([c[kx] for c in cs]); y = np.concatenate([c[ky] for c in cs])
        m[cls] = pix_org.fit(X, y, seed, threads)
    return m


def cmd_fold(a):
    C = _load_cands(a.cache)
    out = Path(a.cache) / "org" / f"s{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    for k, ex in FOLDS.items():
        test = [n for n in C if n.split("_")[0] in ex]
        if not test:
            continue
        t = time.monotonic()
        m = _fit([C[n] for n in C if n not in test], a.seed, a.threads)
        for n in test:
            c = C[n]
            feats = pix_org.to_feats(c, pix_org.predict(m["finds_points"], c["pX"]),
                                     pix_org.predict(m["object_poly"], c["cX"]), n, a.point_r, a.t_ref,
                                     {"finds_points", "object_poly"})
            (out / f"{n}.pkl").write_bytes(pickle.dumps({"new": feats, "pst": {}}))
        print(f"fold {k} s{a.seed}: {test} {time.monotonic() - t:.0f} с", flush=True)


def cmd_full(a):
    C = _load_cands(a.cache)
    m = _fit(list(C.values()), a.seed, a.threads)
    o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
    for cls, b in m.items():
        if b is not None:
            b.save_model(str(o / f"org_maggr_{cls}.txt"))
    (o / "org_maggr_meta.json").write_text(json.dumps({"seed": a.seed, "regions": sorted(C), "n_iter": pix_org.N_ITER,
                                                        "spec": "GEO-ORG1, docs/PREREG_w4_pix_clean.md"}, ensure_ascii=False))
    print("ok", o)


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
        p.add_argument("--point-r", type=float, default=0.3)
        p.add_argument("--t-ref", type=float, default=0.5)
        p.add_argument("--out")
    a = ap.parse_args()
    {"cand": cmd_cand, "fold": cmd_fold, "full": cmd_full}[a.cmd](a)


if __name__ == "__main__":
    main()
