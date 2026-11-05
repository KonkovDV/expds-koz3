"""12-channel cnx input from one LAS (vendor bot-9 code, read-only): base lrm_s/lrm_l/slope + 9 relief_ext extras.
Channel order: TAGS12 below. First five channels = cnx_prep TAGS5 (same vendor functions, accel=None -> rvt)."""
import json, sys, time
from pathlib import Path
import numpy as np
REPO = Path("/home/user/expds-koz3")
VENDOR = REPO / "inference/cnx/vendor"
sys.path.insert(0, str(VENDOR)); sys.path.insert(0, str(REPO / "inference/cnx/_deps"))
from expds import accel as XA, lidar as XL                      # vendor copies
from expds.candidates import to_work_grid
import relief_ext as RX
TAGS12 = ("lrm_s", "lrm_l", "slope", "svf_r", "opp_r", "opn_r", "tpi5_r", "tri3_r", "curv_r", "slrm8_r", "dens_r", "int_r")
CELL_M, DEM_RES = 0.5, 0.1

def prep(las, crs, out):
    t = time.time()
    d = XL.build_dem(str(las), res=DEM_RES, fill="delaunay", crs=crs)
    base, tr = XL.relief_channels(d, CELL_M, TAGS12[:3])
    z, step = to_work_grid(d.dem, d.res, CELL_M)
    z = np.where(base[0] > 0, z, np.nan).astype(np.float32)
    ext, stats = RX.extra_channels(z, step, d.left, d.top, las_path=str(las), accel=None, tags=TAGS12[3:])
    arr = np.concatenate([base, ext]).astype(np.uint8)
    geo = dict(left=float(d.left), top=float(d.top), step=float(step), crs=d.crs, shape=list(arr.shape), tags=list(TAGS12),
               dem_res=float(d.res), dem_factor=int(d.info.get("factor", 1)), backend=XA.backend(), site_stats=stats)
    np.save(out.with_suffix(".npy"), arr)
    out.with_suffix(".json").write_text(json.dumps(geo, ensure_ascii=False, default=str))
    return arr.shape, round(time.time() - t, 1)

if __name__ == "__main__":
    las, crs, out = Path(sys.argv[1]), (sys.argv[2] or None), Path(sys.argv[3])
    print(json.dumps({"las": las.name, "shape_s": prep(las, crs, out)}, ensure_ascii=False), flush=True)
