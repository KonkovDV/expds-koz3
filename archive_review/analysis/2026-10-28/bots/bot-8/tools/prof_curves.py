import cProfile, pstats, json, sys, gc
from pathlib import Path
sys.path.insert(0, "/workspace/koz3_shared/bot-8/repo")
import expds.metric as M
M._SHM_CACHE_DIR = Path("/workspace/koz3_shared/bot-8/shm_cache_off_" + str(__import__("os").getpid()))

pred = json.loads(Path(sys.argv[1]).read_bytes()); gt = json.loads(Path(sys.argv[2]).read_bytes())
METRIC = "/workspace/koz3_shared/bot-8/repo/materials/compute_metrics.py"
M.load_official(METRIC)
pr = cProfile.Profile(); pr.enable(); M.build_curves(pred, gt, METRIC); pr.disable()
st = pstats.Stats(pr); st.sort_stats("tottime").print_stats(25); st.sort_stats("cumulative").print_stats(25)
