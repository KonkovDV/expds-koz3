import sys, json
from pathlib import Path
sys.path.insert(0, "/workspace/koz3_shared/bot-7/repo/tests"); sys.path.insert(0, "/workspace/koz3_shared/bot-7/repo")
import conftest as C
t = Path(sys.argv[1])
r1 = t / "region_0001"; C.write_las(r1 / "01_region_0001_La_точки" / "01_region_0001_La_точки.las")
(r1 / "UTM.json").write_text(json.dumps({"utm": "36N"}), encoding="utf-8")
r2 = t / "region_0002"; C.write_mag_tif(r2 / "07_region_0002_MagGr_карты" / "01_region_0002_MagGr.tif")
print("ok")
