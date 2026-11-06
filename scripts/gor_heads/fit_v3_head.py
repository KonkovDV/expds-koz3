""". Fit gor_v3_head.npz from extract_feats outputs + hillfort labels.

Reproduces scripts/gor_patch/gor_v3.py MODE=fit arithmetic (PCA-64 + L2-logreg
C=1 + Landauer 0.85/0.05 cleaning on inner 5 family folds) without gor_scene.py.
Labels from gor_dense_inputs.json (same SpOr hillfort polygons as dense export).

Env: OUT (feats.npz + meta.json dir), HEAD, INPUTS (gor_dense_inputs.json),
     FEATS_TAG, SEED (default 0). Prints JSON summary with md5.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import Polygon

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from expds.metric import family_folds  # noqa: E402
from expds.naming import site_family  # noqa: E402

SEED = int(os.environ.get("SEED", "0"))
PC, CREG, CLEAN_HI, CLEAN_LO = 64, 1.0, 0.85, 0.05
OD = Path(os.environ.get("OUT", "work/gor_heads/v3"))
FT = os.environ.get("FEATS_TAG", "")
HEAD = Path(os.environ.get("HEAD", str(OD / "gor_v3_head.npz")))
INPUTS = Path(
    os.environ.get(
        "INPUTS",
        str(Path(__file__).resolve().parent / "fixtures" / "gor_dense_inputs.json"),
    )
)


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main() -> None:
    F = np.load(OD / f"feats{FT}.npz")["F"].astype(np.float32)
    META = json.load(open(OD / f"meta{FT}.json", encoding="utf-8"))
    J = json.load(open(INPUTS, encoding="utf-8"))
    PREG = np.array([m["reg"] for m in META])
    POLY = [Polygon(m["corners"]) for m in META]
    hf = {
        r: [(shapely.from_wkt(w), bool(s)) for w, s in v]
        for r, v in J["hf"].items()
    }
    # labels: pos = ≥50% of an sp_or hillfort inside vignette; ignore = intersects any HF; else neg
    lab = np.zeros(len(META), np.int8)
    for k, P in enumerate(POLY):
        st = 0
        for g, is_sp in hf.get(PREG[k], []):
            gg = g if g.geom_type != "Point" else g.buffer(5)
            if not P.intersects(gg):
                continue
            if is_sp and P.intersection(gg).area >= 0.5 * gg.area:
                st = 1
                break
            if st == 0:
                st = -1
        lab[k] = st

    layouts = J["layouts"]["fam10"]
    every = set().union(*layouts)
    ALLIDX = np.arange(len(META))
    tri = ALLIDX[np.isin(PREG, list(every))]

    # PCA-64 on train (label-free), same rng(0) subsample as gor_v3.py
    Xt = F[tri]
    mu = Xt.mean(0)
    sd = Xt.std(0) + 1e-6
    sub = tri[np.random.default_rng(SEED).permutation(len(tri))[:4000]]
    W = np.linalg.svd((F[sub] - mu) / sd, full_matrices=False)[2][:PC].T
    qsd = (((F[tri] - mu) / sd) @ W).std(0) + 1e-6
    QQ = np.c_[np.ones(len(F)), (((F - mu) / sd) @ W) / qsd]

    def fit(idx):
        yt = (lab[idx] == 1).astype(float)
        if len(idx) == 0 or yt.min() == yt.max():
            return ("const", float(yt.mean()) if len(idx) else 0.0)
        Q = QQ[idx]
        sw = np.where(yt == 1, 0.5 / yt.mean(), 0.5 / (1 - yt.mean()))
        w = np.zeros(Q.shape[1])
        lam = np.r_[0, np.ones(Q.shape[1] - 1)] / CREG
        for _ in range(50):
            p = 1 / (1 + np.exp(-Q @ w))
            gr = Q.T @ (sw * (p - yt)) + lam * w
            H = (Q * (sw * p * (1 - p))[:, None]).T @ Q + np.diag(lam) + 1e-9 * np.eye(len(w))
            st = np.linalg.solve(H, gr)
            w -= st
            if np.abs(st).max() < 1e-8:
                break
        return ("lr", w)

    def pred(mdl, idx):
        if mdl[0] == "const":
            return np.full(len(idx), mdl[1])
        return 1 / (1 + np.exp(-QQ[idx] @ mdl[1]))

    def inner_oof(idx, nf=5):
        regs = sorted(set(PREG[idx]))
        fo = family_folds(regs, site_family, nf)
        p = np.zeros(len(idx))
        for h in fo:
            te = np.isin(PREG[idx], list(h))
            if te.any():
                p[te] = pred(fit(idx[~te]), idx[te])
        return p

    def fit_clean(idx):
        li = idx[lab[idx] >= 0]
        p = inner_oof(li)
        yl = lab[li] == 1
        keep = ~((yl & (p < CLEAN_HI)) | (~yl & (p > CLEAN_LO)))
        nrem = int((~keep).sum())
        if keep.sum() < 50 or (lab[li[keep]] == 1).sum() < 5:
            return fit(li), 0
        return fit(li[keep]), nrem

    mdl, nrem = fit_clean(tri)
    assert mdl[0] == "lr", mdl[0]
    HEAD.parent.mkdir(parents=True, exist_ok=True)
    np.savez(HEAD, mu=mu, sd=sd, W=W, qsd=qsd, w=mdl[1], feat_dim=np.int64(F.shape[1]))
    Qc = np.c_[np.ones(len(tri)), (((F[tri] - mu) / sd) @ W) / qsd]
    assert np.allclose(1 / (1 + np.exp(-Qc @ mdl[1])), pred(mdl, tri), atol=1e-6)
    summ = dict(
        head=str(HEAD),
        md5=md5(HEAD),
        bytes=HEAD.stat().st_size,
        n_train_patches=int(len(tri)),
        removed_by_cleaning=int(nrem),
        feat_dim=int(F.shape[1]),
        label_counts=dict(collections.Counter(lab.tolist())),
        seed=SEED,
        expected_legacy_md5="acb95c52956d712a16c3dd389237a691",
        md5_matches_legacy=md5(HEAD) == "acb95c52956d712a16c3dd389237a691",
    )
    print(json.dumps(summ, ensure_ascii=False))


if __name__ == "__main__":
    main()
