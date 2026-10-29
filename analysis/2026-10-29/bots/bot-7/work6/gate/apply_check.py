import sys, glob, logging, time
from pathlib import Path
from types import SimpleNamespace
import torch; torch.set_num_threads(2)
sys.path.insert(0, "/workspace/koz3_shared/bot-7/work6/vendor_test/lib"); sys.path.insert(0, "/workspace/koz3_shared/bot-7/repo")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
from expds import gor_gate
D = Path("/workspace/data/train-expds-3")
regs = {n: SimpleNamespace(name=n, utm_json=D / n / "UTM.json", rasters={"sp_or": sorted(glob.glob(str(D / n / "*SpOr*/*.tif*")))})
        for n in ("046_Тыва_1", "185_Изборск")}
feats = [{"markup_type": "sp_or", "class_name": "gorodishcha", "region_name": n, "confidence": 0.995} for n in regs] + \
        [{"markup_type": "sp_or", "class_name": "kurgany", "region_name": "185_Изборск", "confidence": 0.7}]
cfg = {"gor_gate": {"enabled": True, "device": "cpu"}, "thresholds_by_markup": {"sp_or_gated": {"gorodishcha": 0.992648}}}
t = time.time(); out = gor_gate.apply(feats, regs, cfg, Path("root")); print(round(time.time() - t, 1), "s")
for f in out: print(f["region_name"], f["class_name"], round(f["confidence"], 6), f.get("threshold_group"))
