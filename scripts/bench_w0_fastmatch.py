"""Сверка и замер EXPDS_RUST_MATCH: официальное жадное сопоставление по классам (как metric._build_curves_impl)
через STRtree vs Rust-префильтр. Вход: stack_arch (table.pkl + geom_wkb.pkl + gt_wkb.pkl) или geojson OOF + GT.
Печатает время, число пар, sha1 списков hit и пар (i, j, score) для обоих путей."""
import hashlib, json, os, pickle, sys, time
from pathlib import Path
import numpy as np, shapely
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from expds import metric as M, fastmatch as FM

METRIC = os.environ.get("EXPDS_METRIC", str(Path(__file__).resolve().parents[1] / "materials" / "compute_metrics.py"))
mod = M.load_official(METRIC); calc = mod.F1Calculator()

def greedy(pg, gg, q):
    q_pred, q_gt = q
    hits = np.zeros(len(pg), bool); pairs = []
    if len(q_pred):
        splits = np.flatnonzero(np.diff(q_pred)) + 1
        matched = set()
        for rank, js in zip(q_pred[np.r_[0, splits]], np.split(q_gt, splits)):
            r = int(rank); g = pg[r]
            if g.geom_type != "Polygon":
                continue
            best, bj = float("-inf"), -1
            for j in sorted(int(k) for k in js):
                ok, s = calc._match_quality(g, gg[j])
                if ok:
                    pairs.append((r, j, float(s)))
                if j in matched:
                    continue
                if ok and s > best + 1e-9:
                    best, bj = s, j
            if bj >= 0:
                matched.add(bj); hits[r] = True
    return hits, pairs

def run(name, classes):
    tot = {"py": 0.0, "rs": 0.0}; H = {"py": hashlib.sha1(), "rs": hashlib.sha1()}; P = {"py": hashlib.sha1(), "rs": hashlib.sha1()}
    npairs = {"py": 0, "rs": 0}; nh = 0
    for c, pg, gg in classes:
        if not len(pg) or not len(gg):
            continue
        for mode in ("py", "rs"):
            t = time.perf_counter()
            if mode == "py":
                os.environ["EXPDS_RUST_MATCH"] = "0"
                q = shapely.STRtree([x.buffer(calc.r_tolerance) if x.geom_type == "Point" else x for x in gg]).query(pg)
            else:
                os.environ["EXPDS_RUST_MATCH"] = "1"
                q = FM.prefilter(pg, gg, calc); assert q is not None, "rust path unavailable"
            hits, pairs = greedy(pg, gg, q)
            tot[mode] += time.perf_counter() - t; npairs[mode] += len(q[0])
            H[mode].update(c.encode() + hits.tobytes()); P[mode].update(repr(pairs).encode())
            if mode == "py": nh += int(hits.sum()); ref = (hits, pairs)
            else:
                assert np.array_equal(hits, ref[0]) and pairs == ref[1], f"MISMATCH {name}/{c}"
        print(f"  {c}: n={len(pg)} gt={len(gg)} hits={int(ref[0].sum())} py={tot['py']:.1f}s rs={tot['rs']:.1f}s", flush=True)
    print(json.dumps({"set": name, "py_s": round(tot["py"], 1), "rs_s": round(tot["rs"], 1),
                      "speedup": round(tot["py"] / max(tot["rs"], 1e-9), 1), "pairs_py": npairs["py"], "pairs_rs": npairs["rs"],
                      "hits": nh, "hits_sha_py": H["py"].hexdigest()[:12], "hits_sha_rs": H["rs"].hexdigest()[:12],
                      "pairs_sha_py": P["py"].hexdigest()[:12], "pairs_sha_rs": P["rs"].hexdigest()[:12]}), flush=True)

def stack_arch(work):
    T = pickle.load(open(f"{work}/table.pkl", "rb")); W = pickle.load(open(f"{work}/geom_wkb.pkl", "rb"))
    G = pickle.load(open(f"{work}/gt_wkb.pkl", "rb"))
    for k, c in enumerate(T["CL"]):
        idx = np.flatnonzero(T["cls"] == k); idx = idx[np.argsort(-T["conf"][idx], kind="stable")]
        yield c, list(shapely.from_wkb(W[idx])), list(shapely.from_wkb(G[c])) if c in G else []

def geojson_set(pred, gt):
    pr = M._collect(mod, calc, json.loads(Path(pred).read_bytes()), False)
    gs = M._collect(mod, calc, json.loads(Path(gt).read_bytes()), True)
    for c in calc.target_classes:
        p = pr.get(c, []); order = sorted(range(len(p)), key=lambda i: p[i][1], reverse=True)
        yield c, [p[i][0] for i in order], [x[0] for x in gs.get(c, [])]

if __name__ == "__main__":
    if sys.argv[1] == "stack":
        run("stack_arch", stack_arch(sys.argv[2]))
    else:
        run(Path(sys.argv[2]).name, geojson_set(sys.argv[2], sys.argv[3]))
