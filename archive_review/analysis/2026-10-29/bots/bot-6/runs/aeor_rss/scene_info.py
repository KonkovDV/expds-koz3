import sys, json
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from expds import dfine_aeor as A, sp_dfine as S
for t in sys.argv[2:]:
    p = Path(t); crs = S.utm_for(p, None); sc = A.Scene(p, crs, 0.5)
    import rasterio
    with rasterio.open(p) as s0:
        full = (s0.width, s0.height, s0.count, s0.overviews(1))
    print(json.dumps({"tif": p.name, "full_wh_bands_ovr": full, "src_m": round(sc.src_m, 4), "overview_level": sc.overview_level,
                      "scene_WH_0.5m": [sc.W, sc.H], "windows_1024_288": A.n_windows(sc.W, sc.H, 1024, 288)}, ensure_ascii=False))
    sc.close()
