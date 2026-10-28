"""Сквозной прогон pipeline.run на выборке под cProfile (prefetch=0 — всё в главном потоке).
usage: prof_run.py REPO SAMPLE OUT_PREFIX [cfg_override_toml_text]
"""
import cProfile, json, logging, os, pstats, sys, time, tomllib
from pathlib import Path
repo, sample, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(repo / "inference" / "lib")); sys.path.insert(0, str(repo))
import torch
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s",
                    stream=open(str(out) + ".log", "w"))
text = (repo / "configs" / "default.toml").read_text(encoding="utf-8")
text = text.replace("prefetch = 2", "prefetch = " + os.environ.get("PREFETCH", "0"))
if os.environ.get("NO_WATCHDOG", "1") == "1":
    text = text.replace("watchdog = true", "watchdog = false")
cfgp = Path(str(out) + ".toml"); cfgp.write_text(text, encoding="utf-8")
tomllib.loads(text)
from expds import pipeline
prof = cProfile.Profile()
t = time.perf_counter()
if os.environ.get("NOPROF") != "1":
    prof.enable()
s = pipeline.run(sample, Path(str(out) + ".geojson"), cfgp, repo)
prof.disable()
s["wall"] = time.perf_counter() - t
json.dump(s, open(str(out) + ".summary.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(s, ensure_ascii=False))
if os.environ.get("NOPROF") != "1":
    prof.dump_stats(str(out) + ".prof")
    with open(str(out) + ".txt", "w") as f:
        st = pstats.Stats(prof, stream=f)
        st.sort_stats("cumulative").print_stats(70)
        st.sort_stats("tottime").print_stats(50)
