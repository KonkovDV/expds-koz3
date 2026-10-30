import numpy as np, json, sys, glob
from PIL import Image, ImageDraw
from pyproj import Transformer
from affine import Affine
name,npy,step,a0,c0,e0,f0,gt,mode=sys.argv[1:10]
step=int(step); T=Affine(float(a0)*step,0,float(c0),0,float(e0)*step,float(f0))
a=np.load(npy)
if mode=='int16_kernel':   # kernel: alpha not detected (int16), dataset_mask all-valid, stretch (9,255) from band1>0
    x=a[:3].astype(np.float32); x=np.clip((x-9)/(255-9)*255,0,255)
elif mode=='int16_fixed':  # fix: alpha band4 (0/255) and nodata applied -> 0
    m=(a[3]==255)&(a[0]!=-32768); x=a[:3].astype(np.float32); x[:,~m]=0
else:
    m=a[3]>0; x=a[:3].astype(np.float32); x[:,~m]=0
img=Image.fromarray(x.transpose(1,2,0).astype(np.uint8))
gw=img.width*abs(T.a); gh=img.height*abs(T.e); s=1400/max(gw,gh)
img=img.resize((int(gw*s),int(gh*s))); sx=img.width/a.shape[2]; sy=img.height/a.shape[1]
tr=Transformer.from_crs(3857,32636,always_xy=True); inv=~T; dr=ImageDraw.Draw(img)
for g in glob.glob(gt):
    for f in json.load(open(g))['features']:
        if not f['geometry']: continue
        cs=f['geometry']['coordinates']; polys=cs if f['geometry']['type']=='MultiPolygon' else [cs]
        for p in polys:
            pts=[(lambda c:(c[0]*sx,c[1]*sy))(inv*tr.transform(x_,y_)) for x_,y_ in p[0]]
            dr.line(pts+[pts[0]],fill=(255,0,0),width=2)
img.save(f'{name}_{mode}.png'); print(img.size)
