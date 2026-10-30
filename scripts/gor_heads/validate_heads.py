"""Validate gor_v3_head.npz + gor_dense_head.npz against loader expectations + sanity metric."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from expds.gor_dense import head_p as dense_head_p  # noqa: E402
from expds.gor_gate import head_p as gate_head_p  # noqa: E402
from expds.gor_gate import load_head  # noqa: E402

V3 = Path(os.environ.get("V3_HEAD", "models/gor_gate/gor_v3_head.npz"))
DENSE = Path(os.environ.get("DENSE_HEAD", "models/gor_dense/gor_dense_head.npz"))
LEGACY_V3_MD5 = "acb95c52956d712a16c3dd389237a691"
LEGACY_DENSE_MD5 = "e35570e10d17edac32bea8d26fa0f5f4"
LEGACY_V3_BYTES = 407736
LEGACY_DENSE_BYTES = 204984


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_one(path: Path, feat_dim: int, legacy_md5: str, legacy_bytes: int, head_p_fn) -> dict:
    assert path.exists(), path
    H = load_head(path, None)
    assert int(H["feat_dim"]) == feat_dim, (path, int(H["feat_dim"]), feat_dim)
    for k, shape in (
        ("mu", (feat_dim,)),
        ("sd", (feat_dim,)),
        ("W", (feat_dim, 64)),
        ("qsd", (64,)),
        ("w", (65,)),
    ):
        a = np.asarray(H[k])
        assert a.shape == shape, (path, k, a.shape, shape)
        assert np.isfinite(a).all(), (path, k)
    # dtypes: mu/sd/W/qsd float32-compatible; w float64-compatible
    rng = np.random.default_rng(0)
    F = rng.standard_normal((128, feat_dim)).astype(np.float32)
    p = head_p_fn(H, F)
    assert p.shape == (128,) and np.isfinite(p).all()
    assert float(p.min()) >= 0.0 and float(p.max()) <= 1.0
    # sanity: constant input → low variance; random → spread
    spread = float(p.std())
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    return dict(
        path=str(path),
        bytes=path.stat().st_size,
        md5=md5,
        sha256=sha256(path),
        feat_dim=feat_dim,
        p_mean=float(p.mean()),
        p_std=spread,
        legacy_md5_match=md5 == legacy_md5,
        legacy_bytes_match=path.stat().st_size == legacy_bytes,
        ok=True,
    )


def main() -> None:
    out = {
        "gor_v3": check_one(V3, 1536, LEGACY_V3_MD5, LEGACY_V3_BYTES, gate_head_p),
        "gor_dense": check_one(DENSE, 768, LEGACY_DENSE_MD5, LEGACY_DENSE_BYTES, dense_head_p),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    assert out["gor_v3"]["ok"] and out["gor_dense"]["ok"]


if __name__ == "__main__":
    main()
