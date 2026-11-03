#!/usr/bin/env python3
"""W4 01.11: перевод двух .npz RR1s-cf в «только числовые 1-D» (условие бота 1), значения массивов — те же.

ref.npz: убираются alpha (0-D; α берётся из [w5_cloudfeat] конфига) и classes (строки; классы — из имён ключей).
gq_ref.npz: строковый keys переносится в meta.json["gq_keys"] (порядок тот же).
Пример: w5cf_npz_numeric.py --ref SRC/ref.npz --infer-dir SRC_INFER --out OUT
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ref", type=Path, required=True)
ap.add_argument("--infer-dir", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
(a.out / "w5cf").mkdir(parents=True, exist_ok=True)
z = np.load(a.ref, allow_pickle=False)
np.savez_compressed(a.out / "w5cf" / "ref.npz", **{k: z[k] for k in z.files if k not in ("alpha", "classes")})
d = a.out / "w5cf_infer"
shutil.copytree(a.infer_dir, d, dirs_exist_ok=True)
g = np.load(a.infer_dir / "gq_ref.npz", allow_pickle=False)
meta = json.loads((a.infer_dir / "meta.json").read_text(encoding="utf-8"))
meta["gq_keys"] = [str(k) for k in g["keys"].tolist()]
(d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
np.savez_compressed(d / "gq_ref.npz", **{k: g[k] for k in g.files if k != "keys"})
for p in (a.out / "w5cf" / "ref.npz", d / "gq_ref.npz"):
    q = np.load(p, allow_pickle=False)
    assert all(q[k].ndim == 1 and q[k].dtype.kind in "biuf" for k in q.files), p
print("ok", a.out)
