"""Пик памяти и время одного pix_clean.region_feats (без классики): pix_mem.py <region> <model_dir> <procs>."""
import resource, sys, time
sys.path.insert(0, "/workspace/koz3/w4clean")
from expds.catalog import scan_root
from expds.pipeline import load_config
from expds import pix_clean
reg_name, md, procs = sys.argv[1], sys.argv[2], int(sys.argv[3])
cfg = load_config("/workspace/koz3/w4clean/configs/v3_on.toml")
pix = cfg["geophys"]["pixgbm"]
pix.update(enabled=True, impl="clean", model_dir=md, replace=True, union_name_re="", ground_metres=False,
           procs=procs, budget_s=10**9)
reg = [r for r in scan_root("/workspace/koz3/w4geo/train") if r.name == reg_name][0]
base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
t = time.time(); st = {}
out = pix_clean.region_feats(reg, cfg, md, None, st)
print(f"{reg_name}: {len(out)} кандидатов, {time.time()-t:.0f} с, пик RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f} МБ (до {base/1024:.0f}), fell={st.get('fell')}")
