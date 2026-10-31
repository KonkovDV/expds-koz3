# Kernel prep_tif (or-final, lines 390-430) applied from /vsis3; writes buggy (kernel) and fixed (or_alpha_any_dtype) outputs in one pass.
import os, sys, math, time, numpy as np, rasterio
from urllib.parse import urlparse
from rasterio.enums import Resampling, ColorInterp
from rasterio.windows import Window
from pyproj import Geod, Transformer
ep=urlparse(os.environ['S3_ENDPOINT']).netloc
env=dict(AWS_S3_ENDPOINT=ep, AWS_VIRTUAL_HOSTING='FALSE', AWS_HTTPS='YES', GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', AWS_REGION='ru-7',
         CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif', GDAL_HTTP_MAX_RETRY='5', GDAL_HTTP_RETRY_DELAY='2', GDAL_CACHEMAX=512)
key, out_bug, out_fix = sys.argv[1:4]; PREP_RES=0.25
with rasterio.Env(**env):
  with rasterio.open(key if key.startswith("/") else f"/vsis3/{os.environ['S3_BUCKET']}/{key}") as s:
    t=time.time()
    lon0, lat0 = Transformer.from_crs(s.crs, "EPSG:4326", always_xy=True).transform(*s.xy(s.height // 2, 0))
    lon1, lat1 = Transformer.from_crs(s.crs, "EPSG:4326", always_xy=True).transform(*s.xy(s.height // 2, s.width - 1))
    nat = Geod(ellps="WGS84").inv(lon0, lat0, lon1, lat1)[2] / max(1, s.width - 1)
    f = max(1, int(PREP_RES / max(nat, 1e-6)))
    W, H = math.ceil(s.width / f), math.ceil(s.height / f); nb = min(3, s.count); R = max(1, int(64e6 // (s.width * f * 2 * s.count)))
    prof = dict(driver="GTiff", width=W, height=H, count=nb, dtype=s.dtypes[0], crs=s.crs, nodata=0, tiled=True,
                blockxsize=256, blockysize=256, compress="deflate", predictor=2, BIGTIFF="IF_SAFER",
                transform=s.transform * s.transform.scale(s.width / W, s.height / H))
    alpha_bug = False
    if s.count >= 4 and s.dtypes[3] == "uint8": alpha_bug = True  # (kernel also checks values; int16 never reaches it)
    v = np.unique(s.read(4, out_shape=(min(512, s.height), min(512, s.width)), resampling=Resampling.nearest)) if s.count >= 4 else []
    alpha_fix = s.count >= 4 and np.issubdtype(np.dtype(s.dtypes[3]), np.integer) and (s.colorinterp[3] == ColorInterp.alpha or set(np.asarray(v).tolist()) - ({s.nodata} if s.nodata is not None else set()) <= {0, 255})
    print('nat',nat,'f',f,'W,H',W,H,'R',R,'alpha_bug',alpha_bug,'alpha_fix',alpha_fix,'band4 vals',np.asarray(v)[:8],flush=True)
    with rasterio.open(out_bug+'.part','w',**prof) as db, rasterio.open(out_fix+'.part','w',**prof) as dfx:
      for oy in range(0, H, R):
        r = min(R, H - oy); sy = oy * f; sh = min(s.height - sy, r * f); win = Window(0, sy, s.width, sh)
        a = s.read(list(range(1, nb + 1)), window=win, out_shape=(nb, r, W), resampling=Resampling.average)
        m = s.dataset_mask(window=win, out_shape=(r, W), resampling=Resampling.nearest)
        ab = a.copy(); mb = m
        if alpha_bug: mb = np.minimum(m, s.read(4, window=win, out_shape=(r, W), resampling=Resampling.nearest))
        ab[:, mb == 0] = 0; db.write(ab, window=Window(0, oy, W, r))
        mf = m
        if alpha_fix: mf = np.minimum(m, s.read(4, window=win, out_shape=(r, W), resampling=Resampling.nearest).astype(m.dtype))
        a[:, mf == 0] = 0; dfx.write(a, window=Window(0, oy, W, r))
        if (oy // R) % 20 == 0: print(oy, H, round(time.time()-t), flush=True)
    os.replace(out_bug+'.part', out_bug); os.replace(out_fix+'.part', out_fix); print('done', round(time.time()-t), flush=True)
