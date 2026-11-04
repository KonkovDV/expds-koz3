#!/usr/bin/env python3
"""CPU smoke on tiny synthetic data (box-safe): produce both heads + validate shapes.

Does NOT need train-expds-3 or dinov2. Runs fit_v3_head + export_head arithmetic
on synthetic feats/dense with a minimal gor_dense_inputs fixture.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import shapely

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))


def main() -> int:
    rng = np.random.default_rng(0)
    td = Path(tempfile.mkdtemp(prefix="gor_heads_smoke_"))
    try:
        v3 = td / "v3"
        dense = td / "dense"
        heads = td / "heads"
        v3.mkdir()
        dense.mkdir()
        heads.mkdir()

        # --- synthetic vignette feats (1536-d) ---
        regs = [f"r{i:03d}_FamA" for i in range(8)] + [f"r{i:03d}_FamB" for i in range(8, 16)]
        n = 240
        F = rng.standard_normal((n, 1536)).astype(np.float16)
        # plant a separable signal on first 32 dims for "pos" patches
        meta = []
        for i in range(n):
            reg = regs[i % len(regs)]
            ux, uy = 1000 + (i % 20) * 200.0, 2000 + (i // 20) * 200.0
            half = 192.0
            corners = [
                (ux - half, uy - half),
                (ux + half, uy - half),
                (ux + half, uy + half),
                (ux - half, uy + half),
            ]
            meta.append(dict(reg=reg, file=f"{reg}.tif", ux=ux, uy=uy, corners=corners, gsd=0.5))
            if i % 8 == 0:
                F[i, :32] += 3.0  # pos-ish
        np.savez(v3 / "feats.npz", F=F)
        json.dump(meta, open(v3 / "meta.json", "w"))

        # --- inputs: one sp_or hillfort per "pos" region near those patches ---
        hf = {}
        for ri, reg in enumerate(regs):
            if ri % 4 != 0:
                continue
            # cover the planted dense token block around (1000+9*28 .. 1000+12*28, same uy)
            poly = shapely.geometry.box(1000 + 3 * 28 - 20, 2000 + 1 * 28 - 20, 1000 + 6 * 28 + 20, 2000 + 3 * 28 + 20)
            hf[reg] = [[shapely.to_wkt(poly, rounding_precision=2), 1]]
        # also mark vignette-centred hillforts for v3 labels
        for i, m in enumerate(meta):
            if i % 8 != 0:
                continue
            r = m["reg"]
            ux, uy = m["ux"], m["uy"]
            poly = shapely.geometry.box(ux - 40, uy - 40, ux + 40, uy + 40)
            hf.setdefault(r, []).append([shapely.to_wkt(poly, rounding_precision=2), 1])
        # fam10-like layouts: two folds covering all regs
        layouts = {
            "fam10": [regs[:8], regs[8:]],
            "LOFO": [[r] for r in regs],
            "seed0": [regs[::2], regs[1::2]],
        }
        inputs = dict(hf=hf, scn=regs, layouts=layouts)
        inputs_path = td / "gor_dense_inputs.json"
        json.dump(inputs, open(inputs_path, "w"))

        env = {
            **os.environ,
            "OUT": str(v3),
            "HEAD": str(heads / "gor_v3_head.npz"),
            "INPUTS": str(inputs_path),
            "SEED": "0",
            "PYTHONPATH": str(REPO),
        }
        r = subprocess.run([sys.executable, str(HERE / "fit_v3_head.py")], env=env, cwd=str(REPO))
        assert r.returncode == 0, "fit_v3 failed"

        # --- synthetic dense tokens (768-d) ---
        # 16 files × 50 tokens (5×10 local grid) = 800
        per = 50
        nt = len(regs) * per
        DF = rng.standard_normal((nt, 768)).astype(np.float16)
        fid = np.repeat(np.arange(len(regs), dtype=np.int32), per)
        local = np.arange(per)
        row = np.tile(local // 10, len(regs)).astype(np.int32)  # 0..4
        col = np.tile(local % 10, len(regs)).astype(np.int32)   # 0..9
        ux = np.zeros(nt); uy = np.zeros(nt)
        vf = np.ones(nt, np.float32)
        files = []
        for fi, reg in enumerate(regs):
            files.append(dict(reg=reg, file=f"{reg}.tif", H=5, W=10, gsd=2.0, epsg=32636))
        for i in range(nt):
            ux[i] = 1000 + col[i] * 28.0
            uy[i] = 2000 + row[i] * 28.0
            if 3 <= col[i] <= 6 and 1 <= row[i] <= 3 and fid[i] % 4 == 0:
                DF[i, :32] += 4.0
        np.savez(
            dense / "dense.npz",
            F=DF,
            fid=fid,
            row=row,
            col=col,
            ux=ux.astype(np.float64),
            uy=uy.astype(np.float64),
            vf=vf,
        )
        json.dump(dict(files=files, model="smoke", gsds=[2.0, 4.0]), open(dense / "files.json", "w"))
        shutil.copy2(inputs_path, dense / "gor_dense_inputs.json")

        env = {
            **os.environ,
            "DOUT": str(dense),
            "OUT": str(heads / "gor_dense_head.npz"),
            "PYTHONPATH": str(REPO),
            "REPO": str(REPO),
        }
        env.pop("FOLD", None)
        r = subprocess.run([sys.executable, str(REPO / "scripts/gor_dense/export_head.py")], env=env, cwd=str(REPO))
        assert r.returncode == 0, "export_head failed"

        env = {
            **os.environ,
            "V3_HEAD": str(heads / "gor_v3_head.npz"),
            "DENSE_HEAD": str(heads / "gor_dense_head.npz"),
            "PYTHONPATH": str(REPO),
        }
        r = subprocess.run([sys.executable, str(HERE / "validate_heads.py")], env=env, cwd=str(REPO))
        assert r.returncode == 0, "validate failed"
        print(json.dumps({"smoke": "PASS", "work": str(td), "heads": os.listdir(heads)}, indent=2))
        return 0
    finally:
        shutil.rmtree(td, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
