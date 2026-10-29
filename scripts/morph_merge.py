"""Задача 10: сборка morph_li_kurgan.parquet из частей по регионам + внутрирегиональные rz-версии + meta.json."""
import json, sys, glob, subprocess, hashlib
from pathlib import Path
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).resolve().parent))
import morph_li as M

RZ = ["morph_bg_rough", "morph_h0", "morph_hmax", "morph_ring", "morph_r_half", "morph_r_edge", "morph_slope_edge",
      "morph_vol", "morph_tpi2", "morph_tpi5", "morph_tpi8", "morph_tpi15", "morph_tpi30", "morph_std3", "morph_std8",
      "morph_curv1_5", "morph_curv3"]

def main(cands, tmp, out):
    C = np.load(cands, allow_pickle=False)
    sidx, reg = C["stack_idx"], C["region"]
    pos = {int(s): i for i, s in enumerate(sidx)}
    F = np.full((sidx.size, len(M.NAMES)), np.nan, np.float32)
    seen = np.zeros(sidx.size, bool); infos = []
    for f in sorted(glob.glob(f"{tmp}/*.npz")):
        d = np.load(f); ii = np.array([pos[int(s)] for s in d["stack_idx"]])
        F[ii] = d["feats"]; seen[ii] = True
        infos.append(json.loads(Path(f[:-4] + ".json").read_text()))
    assert seen.all(), f"нет строк: {(~seen).sum()}"
    cols = {"stack_idx": pa.array(sidx.astype(np.int32))}
    for j, n in enumerate(M.NAMES):
        cols[n] = pa.array(F[:, j])
    rz = np.full((sidx.size, len(RZ)), np.nan, np.float32)
    for r in np.unique(reg):
        m = reg == r
        for k, n in enumerate(RZ):
            x = F[m, M.NAMES.index(n)].astype(np.float64)
            if np.isfinite(x).sum() >= 10:
                med = np.nanmedian(x); mad = 1.4826 * np.nanmedian(np.abs(x - med))
                rz[m, k] = (x - med) / max(mad, 1e-6)
    for k, n in enumerate(RZ):
        cols["morph_rz_" + n[6:]] = pa.array(rz[:, k])
    pq.write_table(pa.table(cols), out, compression="zstd")
    return F, rz, infos

if __name__ == "__main__":
    cands, tmp, out = sys.argv[1:4]
    F, rz, infos = main(cands, tmp, out)
    print("ok", F.shape, rz.shape)
