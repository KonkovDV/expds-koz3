"""Сухой прогон ноутбука 01 на CPU на синтетическом npz той же схемы (bot9_kurgan_subclass.py crops).
Запуск: OMP_NUM_THREADS=2 nice -n 10 venv/bin/python tests/k01/run_local.py"""
import json, os, sys, shutil, traceback
from pathlib import Path
import numpy as np
B = Path(__file__).resolve().parents[2]; T = Path("/tmp/bot9_k01"); shutil.rmtree(T, ignore_errors=True)
rng = np.random.default_rng(0)
def make(path, N=240, tile=64, mutate=None):
    fams = [f"Сем{i}" for i in range(12)]
    region = np.array([f"{i % 24:03d}_{fams[i % 12]}_уч{(i // 12) % 2}" for i in range(N)])
    y = rng.choice([-1, 0, 1], N, p=[0.1, 0.5, 0.4]).astype(np.int8)
    X = rng.integers(0, 255, (N, 3, tile, tile), dtype=np.uint8)
    kind = np.where(np.arange(N) % 3 == 0, "gt", "cand")
    d = dict(X=X, y=y, region=region, family=np.array([r.split("_")[1] for r in region]), kind=kind,
             stack_idx=np.where(kind == "cand", np.arange(N), -1),
             cand_class=np.where(kind == "cand", rng.choice(["kurgany_povrezhdennye", "kurgany_tselye"], N), ""),
             res=np.float32(0.5), tile=np.int32(tile), li_tags=np.array(["li_tin_59"]), markup=np.array(["v2"]))
    if mutate: mutate(d)
    path.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(path, **d)
nb = json.loads((B / "notebooks/01_kurgan_subclass.ipynb").read_text(encoding="utf-8"))
cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
def run(inp, n_cells=None):
    os.environ.update(KOZ3_BOT9_LOCAL_TEST="1", KOZ3_BOT9_INPUT=str(inp), KOZ3_BOT9_WORK=str(T / "work"), KOZ3_BOT9_CODE=str(T / "nobundle"))
    g = {"__name__": "__main__"}
    for c in cells[:n_cells]:
        exec(compile(c, "<cell>", "exec"), g)
    return g
res = {}
# 1) негативные проверки схемы/наличия — только ячейки 1–2
cases = {
  "нет_датасета": (T / "absent", None),
  "нет_y": (T / "bad1", lambda d: d.pop("y")),
  "NHWC": (T / "bad2", lambda d: d.__setitem__("X", np.ascontiguousarray(d["X"].transpose(0, 2, 3, 1)))),
  "y=2": (T / "bad3", lambda d: d["y"].__setitem__(0, 2)),
  "float_X": (T / "bad4", lambda d: d.__setitem__("X", d["X"].astype(np.float32))),
}
for name, (dirp, mut) in cases.items():
    if name != "нет_датасета": make(dirp / "kurgan_subclass_crops.npz", mutate=mut)
    try:
        run(dirp, 2); res[name] = "НЕ УПАЛО (плохо)"
    except Exception as e:
        res[name] = f"{type(e).__name__}: {str(e)[:220]}"
# 2) полный сухой прогон: все ячейки, 1 батч на фолд, 2 модели, финальные веса
good = T / "good"; make(good / "kurgan_subclass_crops.npz")
g = run(good)
rep = json.loads((T / "work/kurgan_subclass_report.json").read_text(encoding="utf-8"))
res["full"] = {"files": sorted(p.name for p in (T / "work").rglob("*") if p.is_file() and "hf_" not in str(p)),
               "folds": len(rep["folds"]), "models": list(rep["models"]),
               "ens_auc_all(синтетика, шум)": rep["models"]["ensemble"]["auc_oof_all"],
               "by_family_rows": len(rep["models"]["ensemble"]["by_family"]),
               "oof_rows": sum(1 for _ in open(T / "work/kurgan_subclass_oof.csv")) - 1}
# 3) gpu_guard на CPU обязан упасть
try:
    g["gpu_guard"](2); res["gpu_guard_cpu"] = "НЕ УПАЛО (плохо)"
except RuntimeError as e:
    res["gpu_guard_cpu"] = str(e)[:160]
print(json.dumps(res, ensure_ascii=False, indent=1))
