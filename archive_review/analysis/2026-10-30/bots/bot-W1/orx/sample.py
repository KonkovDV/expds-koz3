import os, rasterio, time, numpy as np, sys
from urllib.parse import urlparse
from rasterio.windows import Window
from rasterio.enums import Resampling
ep=urlparse(os.environ['S3_ENDPOINT']).netloc
env=dict(AWS_S3_ENDPOINT=ep, AWS_VIRTUAL_HOSTING='FALSE', AWS_HTTPS='YES', GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', AWS_REGION='ru-7', CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif', GDAL_HTTP_MAX_RETRY='5', GDAL_HTTP_RETRY_DELAY='2')
B=os.environ['S3_BUCKET']; k=sys.argv[1]; step=int(sys.argv[2]); out=sys.argv[3]
with rasterio.Env(**env):
  with rasterio.open(f'/vsis3/{B}/{k}') as s:
    print(s.block_shapes, s.profile, flush=True)
    rows=list(range(0,s.height,step)); W=-(-s.width//step); a=np.zeros((s.count,len(rows),W),dtype=s.dtypes[0]); t=time.time()
    for i,r in enumerate(rows):
      x=s.read(window=Window(0,r,s.width,1))
      a[:,i,:]=x[:,0,::step][:,:W]
      if i%100==0: print(i,len(rows),round(time.time()-t),flush=True)
    np.save(out,a); print('saved',a.shape, a.dtype, flush=True)
