"""Весь поток `scoreboard.py --honest 10 --bootstrap 1000` с опытом (как --replace sp_focal2=пусто)
на готовом стеке (без build_stack: отдельных файлов work/ здесь нет). Функции evaluate/honest —
из scripts/scoreboard.py выбранного репо, остальное — копия main().
python scoreboard_flow.py PRED GT --repo R --out OUT.json
«34 новых» = notes/regions_las_new.txt; «без лидара» = участки эталона без предсказаний markup li."""
import argparse, json, sys, time, io, contextlib, importlib.util
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("pred"); ap.add_argument("gt")
ap.add_argument("--repo", default="/workspace/koz3_shared/bot-8/repo")
ap.add_argument("--out", default=None)
ap.add_argument("--cache-dir", default=None, help="EXPDS_GEOM_CACHE для нового кода")
a = ap.parse_args()
T0 = time.perf_counter()
sys.path.insert(0, a.repo)
import expds.metric as M
import os
CACHE = Path(a.cache_dir or "/workspace/koz3_shared/bot-8/shm_cache")
if hasattr(M, "_SHM_CACHE_DIR"):
    M._SHM_CACHE_DIR = CACHE   # не /dev/shm (правило); старый код кладёт сюда *.wkb.pkl, новый — *.geomcache
os.environ["EXPDS_GEOM_CACHE"] = str(CACHE)
spec = importlib.util.spec_from_file_location("sb", Path(a.repo) / "scripts" / "scoreboard.py")
sb = importlib.util.module_from_spec(spec); spec.loader.exec_module(sb)
METRIC = str(Path(a.repo) / "materials" / "compute_metrics.py")
T = {}
def tic(k, t0): T[k] = T.get(k, 0.0) + time.perf_counter() - t0
log = io.StringIO()
with contextlib.redirect_stdout(log):
    t = time.perf_counter(); gt = sb.load(Path(a.gt)); feats = sb.load(Path(a.pred))["features"]; tic("load_json", t)
    new = {ln.strip() for ln in (Path(a.repo) / "notes" / "regions_las_new.txt").read_text(encoding="utf-8").splitlines() if ln.strip()}
    li_regs = {(f.get("properties") or {}).get("region_name") for f in feats if (f.get("properties") or {}).get("markup_type") == "li"}
    optical = {(f.get("properties") or {}).get("region_name") for f in gt["features"]} - li_regs - {None}
    t = time.perf_counter(); base, _, kept_b, per_b = sb.evaluate("база", feats, gt, new, optical, METRIC); tic("evaluate_base", t)
    t = time.perf_counter(); honest_b = sb.honest("база", new, optical, 10, "family"); tic("honest_base", t)
    mixed = [f for f in feats if (f.get("properties") or {}).get("threshold_group") != "sp_focal2"]
    t = time.perf_counter(); after, taus, kept_a, per_a = sb.evaluate("опыт", mixed, gt, new, optical, METRIC); tic("evaluate_exp", t)
    t = time.perf_counter()
    boot = M.paired_bootstrap(M.build_curves(kept_a, gt, METRIC), M.build_curves(kept_b, gt, METRIC), {}, {}, 1000)["score"]
    tic("paired_bootstrap+2curves", t)
    t = time.perf_counter(); honest_a = sb.honest("опыт", new, optical, 10, "family"); tic("honest_exp", t)
    from expds.naming import site_family
    t = time.perf_counter(); hboot = M.paired_bootstrap_counts(honest_a[1], honest_b[1], site_family, 1000); tic("paired_bootstrap_counts", t)
T["total_flow"] = time.perf_counter() - T0
out = {"base": base, "after": after, "per_b": per_b, "per_a": per_a, "boot": boot, "honest_b": honest_b[0],
       "honest_a": honest_a[0], "hboot": hboot, "taus": taus, "log": log.getvalue(), "times": T}
print(json.dumps({k: (round(v, 3)) for k, v in T.items()}))
if a.out: Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
