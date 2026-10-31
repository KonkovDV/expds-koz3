"""bot-6 crsfix: сцена 008_Герите Or так, как её строит pipeline (crs_for = read_utm_json(UTM.json)) — до/после."""
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from expds import dfine_aeor as A, sp_dfine as S
from expds.catalog import read_utm_json
R = Path("/workspace/data/train-expds-3/008_Герите")
tif = next(R.glob("05_*/*_Or_*.tif"))
fb = read_utm_json(R / "UTM.json")
crs = S.utm_for(tif, fb)
sc = A.Scene(tif, crs, 0.5)
w = A.windows(sc.W, sc.H, 1024, 288)
print("SCENE008 " + json.dumps({"code": sys.argv[1], "utm_json": fb, "crs": crs, "WxH": [sc.W, sc.H], "res": abs(sc.T.a),
      "overview_level": sc.overview_level, "windows": len(w)}))
sc.close()
