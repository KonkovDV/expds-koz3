"""predict_region с фоновым DEM и без: одинаковые предсказания на двух малых участках."""
import sys
from pathlib import Path

ROOT = Path(r"C:\Sloi\expds-koz3_v0.2.0\expds-koz3")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from nn_oof import predict_region  # noqa: E402

from expds import candidates  # noqa: E402
from expds.catalog import read_utm_json, scan_root  # noqa: E402
from expds.infer_nn import Segmenter  # noqa: E402
from expds.lidar import DemPrefetch  # noqa: E402
from expds.pipeline import _params, load_config  # noqa: E402

cfg = load_config(ROOT / "configs" / "default.toml")
cand_p = _params(candidates.CandidateParams, cfg.get("candidates"))
seg = Segmenter(ROOT / "models" / "li_r18_hard2.pt")
regs = [r for r in scan_root(r"F:\expds3\train-expds-3") if r.name in ("179_Гнёздово_Лесная_1", "180_Гнёздово_Лесная_2")]
plain = [predict_region(seg, r, "li", cand_p, cfg.get("li")) for r in regs]
pf = DemPrefetch([(las, read_utm_json(r.utm_json)) for r in regs for las in r.las], 2)
try:
    fetched = [predict_region(seg, r, "li", cand_p, cfg.get("li"), pf) for r in regs]
finally:
    pf.close()
key = lambda f: (f["class_name"], f["confidence"], f["geom"].wkb)
print("одинаково:", all(sorted(map(key, a)) == sorted(map(key, b)) for a, b in zip(plain, fetched)),
      [len(a) for a in plain])
