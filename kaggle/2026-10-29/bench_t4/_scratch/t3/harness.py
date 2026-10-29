import os, sys, json, time, subprocess
src = open("../../src/bench_main.py").read()
pre, tail = src.split("# ── Шаг 4.", 1)
# минимальный контекст из шагов 0–3
W = os.getcwd(); LOGS = os.path.join(W, "logs"); T0 = time.time(); SUMMARY = {"steps": []}
def log(*a): print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)
def save_summary(): json.dump(SUMMARY, open("run_summary.json", "w"), ensure_ascii=False, indent=1)
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="0", HF_HUB_OFFLINE="1")
exec("# ── Шаг 4." + tail)
