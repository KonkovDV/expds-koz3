import numpy as np, json, sys, glob
from PIL import Image, ImageDraw
from pyproj import Transformer
from affine import Affine
name,npy,step,a0,b0,c0,d0,e0,f0,gt=sys.argv[1:11]
step=int(step); T=Affine(float(a0)*step,0,float(c0),0,float(e0)*step,float(f0))
a=np.load(npy).astype(np.float32)
nod=(a[:3]==(-32768 if a.min()<-30000 else 0)).all(0) if a.shape[0]>=3 else None
print(name,'shape',a.shape,'nodata frac',nod.mean() if nod is not None else None)
for b in range(a.shape[0]):
    v=a[b][~nod]; print(' band',b,'min',v.min(),'p1',np.percentile(v,1),'p50',np.median(v),'p99',np.percentile(v,99),'max',v.max())
rgb=a[:3].copy()
for b in range(3):
    v=rgb[b][~nod]; lo,hi=np.percentile(v,[1,99]); rgb[b]=np.clip((rgb[b]-lo)/(hi-lo+1e-6)*255,0,255)
rgb[:, nod]=0
img=Image.fromarray(rgb.transpose(1,2,0).astype(np.uint8))
# resize to ground aspect
gw=img.width*abs(T.a); gh=img.height*abs(T.e); s=1600/max(gw,gh)
img=img.resize((int(gw*s),int(gh*s))); sx=img.width/a.shape[2]; sy=img.height/a.shape[1]
tr=Transformer.from_crs(3857,32636,always_xy=True); inv=~T
dr=ImageDraw.Draw(img); n=0
for g in glob.glob(gt):
    for f in json.load(open(g))['features']:
        geom=f['geometry']; cs=geom['coordinates']
        polys=cs if geom['type']=='MultiPolygon' else [cs]
        for p in polys:
            pts=[]
            for x,y in p[0]:
                X,Y=tr.transform(x,y); c,r=inv*(X,Y); pts.append((c*sx,r*sy))
            dr.line(pts+[pts[0]],fill=(255,0,0),width=2); n+=1
print('gt polys',n)
img.save(f'{name}.png'); print(img.size)
