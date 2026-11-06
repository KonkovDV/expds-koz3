"""Задача 14: сборка morph/morph_ringw_li_kurgan.parquet из per-region.npz morph_fast --ringw --save ringw.
Порядок строк и ключ stack_idx — как в morph_li_kurgan.parquet (1 209 594). Метки не читаются.
python morph_ringw_merge.py TMPDIR OUT.parquet"""
import hashlib, json, sys, glob
from pathlib import Path
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).resolve().parent))
import morph_fast as MF

tmp, outp = Path(sys.argv[1]), Path(sys.argv[2])
ref = pq.read_table("/workspace/koz3_shared/bot-2/morph/morph_li_kurgan.parquet", columns=["stack_idx"]).column(0).to_numpy()
pos = {int(s): i for i, s in enumerate(ref)}
V = np.full((ref.size, 4), np.nan, np.float32); seen = np.zeros(ref.size, bool); regs = []
for f in sorted(glob.glob(str(tmp / "*.npz"))):
    d = np.load(f); assert list(d["names"]) == MF.RW_NAMES
    ii = np.array([pos[int(s)] for s in d["stack_idx"]]); assert not seen[ii].any()
    V[ii] = d["ringw"]; seen[ii] = True
    regs.append(json.loads(Path(f[:-4] + ".json").read_text()))
assert seen.all(), f"нет строк: {(~seen).sum()}"
cols = {"stack_idx": pa.array(ref.astype(np.int32))}
for k, n in enumerate(MF.RW_NAMES):
    cols[n] = pa.array(V[:, k], from_pandas=True)
pq.write_table(pa.table(cols), outp, compression="zstd")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
here = Path(__file__).resolve().parent
dist = {}
for k, n in enumerate(MF.RW_NAMES):
    x = V[:, k].astype(np.float64); f_ = np.isfinite(x)
    q = np.percentile(x[f_], [1, 10, 25, 50, 75, 90, 99]) if f_.any() else []
    dist[n] = {"non_null": int(f_.sum()), "mean": float(x[f_].mean()), "std": float(x[f_].std()),
               "min": float(x[f_].min()), "max": float(x[f_].max()),
               "q01_10_25_50_75_90_99": [round(float(v), 4) for v in q]}
ok = [r for r in regs if r.get("status") == "ok"]
n_ok = sum(r["n"] for r in ok)
t_rw = sum(r.get("t_ringw_s", 0) for r in ok); t_feat = sum(r.get("t_feat_s", 0) for r in ok)
meta = {
    "task": "bot-2 задача 14: оконные признаки кольца для лидарных кандидатов курганов (для PREREG_MORPH_RING бота 4)",
    "key": "stack_idx", "rows": int(ref.size), "row_order": "как morph_li_kurgan.parquet",
    "label_free": True, "tuned_on_gt": False,
    "params": {"shift_radius_m": MF.RW_R, "shift_step_m": MF.RW_STEP, "n_shifts": int(MF.RW_SY.size),
               "dem": "0,5 м, LAS класс 2, accel.las_grid + fill_interior_holes(pyramid) — как morph_fast/morph_li",
               "detrend": "плоскость по решётке 2 м в кольце 16–32 м вокруг ИСХОДНОГО центра (как morph_*)",
               "bottom": "среднее z' в r<1 м от сдвинутого центра (кольца 0,5 м №0–1)",
               "rim_annulus": "1,5 ≤ r < 5 м (кольца 0,5 м №3–9), кольцо 1–1,5 м не используется",
               "background": "среднее z' в кольце 12–20 м вокруг исходного центра (= фон morph_hmax)",
               "origin": "радиусы шаблона — из пред-декларации/дизайна задачи 13 (дно r≤1, вал 1,5–5, фон 12–20); радиус сдвигов 3 м — запрос бота 4; шаг 1 м = радиус дна; без подбора по GT",
               "validity": "NaN там же, где у morph_* (valid<0,05 или нет высоты в центре), и вне LAZ"},
    "features": [
        {"name": "morph_ringw_ring", "unit": "м", "desc": "max по 29 сдвигам s (|s|≤3 м, шаг 1 м) от [max профиля колец 0,5 м на 1,5–5 м − дно r<1 м]"},
        {"name": "morph_ringw_depth", "unit": "м", "desc": "max по сдвигам от [среднее z' в кольце 1,5–5 м − дно r<1 м] (перевёрнутый TPI / кольцевой шаблон)"},
        {"name": "morph_ringw_rimbg", "unit": "м", "desc": "вал при s* (argmax morph_ringw_ring) − фон 12–20 м"},
        {"name": "morph_ringw_off", "unit": "м", "desc": "|s*| — смещение лучшего центра кольца от центра кандидата (0…3)"}],
    "distribution": dist,
    "timing": {"regions_ok": len(ok), "cands_ok": n_ok, "t_ringw_total_s": round(t_rw, 2),
               "ms_per_cand_ringw": round(1000 * t_rw / max(n_ok, 1), 4),
               "t_feat45_plus_ringw_total_s": round(t_feat, 2),
               "max_rss_mb": max(r.get("maxrss_mb", 0) for r in regs)},
    "regions": regs,
    "sha256": {"morph_fast.py": sha(here / "morph_fast.py"), "morph_ringw_merge.py": sha(here / "morph_ringw_merge.py"),
               "test_morph_ringw.py": sha(here / "test_morph_ringw.py"),
               "candidates": "/workspace/koz3_shared/bot-4/work/cropverif/li_kurgan_candidates.npz",
               "parquet": sha(outp)},
}
Path(str(outp).replace(".parquet", ".meta.json")).write_text(
    json.dumps(meta, ensure_ascii=False, indent=1))
print(json.dumps({"rows": int(ref.size), "timing": meta["timing"], "non_null": {n: dist[n]["non_null"] for n in dist}}, ensure_ascii=False))
