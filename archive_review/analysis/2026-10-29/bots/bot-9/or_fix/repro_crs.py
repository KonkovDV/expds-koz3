# Воспроизведение 1×1 px: UTM.json Герите = EPSG:4326 → WarpedVRT с resolution=0.5 (градусы!). Только заголовки/обзоры.
import sys, json, re
sys.path.insert(0, "/workspace/koz3_shared/bot-9/or_fix/bot6_snapshot")
from pathlib import Path
import rasterio
from rasterio.warp import calculate_default_transform
from pyproj import CRS, Transformer
R = Path("/workspace/data/train-expds-3/008_Герите")
tif = R / "05_Герите_Or_ортофото/01_Герите_Or_ортофото.tif"
d = json.load(open(R / "UTM.json")); m = re.search(r"EPSG::?(\d+)", d["crs"]); crs_json = f"EPSG:{m.group(1)}"
with rasterio.open(tif) as s:
    lon, lat = Transformer.from_crs(s.crs, "EPSG:4326", always_xy=True).transform(*s.xy(s.height // 2, s.width // 2))
    utm = f"EPSG:{(32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1}"
    for c in (crs_json, utm):
        tr, w, h = calculate_default_transform(s.crs, c, s.width, s.height, *s.bounds, resolution=0.5)
        print(f"bot9 Scene: target {c} projected={CRS.from_user_input(c).is_projected} → {w}x{h}")
from expds import catalog, sp_dfine as S, dfine_aeor as A
c6 = S.utm_for(tif, catalog.read_utm_json(R / "UTM.json"))
sc = A.Scene(tif, c6, 0.5); print("bot6 pipeline crs_for=read_utm_json →", c6, "Scene", sc.W, "x", sc.H, "ovr", sc.overview_level); sc.close()
c6b = S.utm_for(tif, None); sc = A.Scene(tif, c6b, 0.5); print("bot6 utm_for(None) →", c6b, "Scene", sc.W, "x", sc.H, "ovr", sc.overview_level); sc.close()
