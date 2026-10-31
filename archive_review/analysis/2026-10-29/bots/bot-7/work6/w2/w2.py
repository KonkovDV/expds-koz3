"""bot-7 W2: пик RSS дерева процессов main.py (сумма) и по процессам, 217, PIXGBM procs=1 и procs=2.
Пакет — распакованный zip v3 (флаги выкл. = v2 по байтам), меняется только [geophys.pixgbm] procs.
taskset 2 ядра, nice 10, OMP/MKL=2 (main.py ставит OMP=8 сам — ядра ограничивает taskset). Замер /proc каждые 0,1 с."""
import json, os, re, shutil, subprocess, sys, time, zipfile
from pathlib import Path
Z, INP, W = Path(sys.argv[1]), Path(sys.argv[2]).resolve(), Path(__file__).parent
PY = "/workspace/koz3_shared/bot-8/venv/bin/python"
def rss(pid):
    try:
        for l in open(f"/proc/{pid}/status"):
            if l.startswith("VmRSS"): return int(l.split()[1]) // 1024
    except OSError: return 0
    return 0
def kids(pid):
    out = []
    try:
        for t in os.listdir(f"/proc/{pid}/task"):
            out += [int(x) for x in open(f"/proc/{pid}/task/{t}/children").read().split()]
    except OSError: pass
    return out
def tree(pid):
    s, seen = [pid], []
    while s:
        p = s.pop(); seen.append(p); s += kids(p)
    return seen
res = {}
for procs in (1, 2):
    d = W / f"pkg_p{procs}"; shutil.rmtree(d, ignore_errors=True); zipfile.ZipFile(Z).extractall(d)
    c = (d / "configs/default.toml").read_text(encoding="utf-8")
    c2, n = re.subn(r"(?m)^procs = 2\b", f"procs = {procs}", c); assert n == 1
    (d / "configs/default.toml").write_text(c2, encoding="utf-8")
    out = W / f"out_p{procs}"; shutil.rmtree(out, ignore_errors=True); out.mkdir()
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
    t0 = time.time()
    p = subprocess.Popen(["taskset", "-c", "0,1", "nice", "-n", "10", PY, "inference/main.py", str(INP), str(out)],
                         cwd=d, env=env, stdout=open(W / f"run_p{procs}.log", "w"), stderr=subprocess.STDOUT)
    peak_sum, peak_proc, nmax, per = 0, 0, 0, {}
    while p.poll() is None:
        pids = tree(p.pid); r = {q: rss(q) for q in pids}
        tot = sum(r.values()); peak_sum = max(peak_sum, tot); nmax = max(nmax, len(pids))
        for q, v in r.items(): per[q] = max(per.get(q, 0), v)
        peak_proc = max([peak_proc] + list(r.values()))
        if peak_proc > 1900 or tot > 3800:
            p.kill(); print("KILL", procs, peak_proc, tot)
        time.sleep(0.1)
    res[procs] = {"rc": p.returncode, "wall_s": round(time.time() - t0, 1), "peak_tree_sum_mib": peak_sum,
                  "peak_single_proc_mib": peak_proc, "max_procs_in_tree": nmax,
                  "per_process_peak_mib": sorted(per.values(), reverse=True)[:6],
                  "result_bytes": (out / "result.geojson").stat().st_size if (out / "result.geojson").exists() else None}
    print(procs, res[procs], flush=True)
    shutil.rmtree(d)
json.dump(res, open(W / "w2.json", "w"), indent=1)
