"""Визуальная проверка: 8 эталонов ae на участок, окно 120 м, контур эталона красный, кандидаты D<=400 зелёные, прочие жёлтые."""
import numpy as np, pickle, shapely, rasterio
from rasterio.windows import from_bounds
from PIL import Image, ImageDraw
Z=np.load('cands_all.npz',allow_pickle=True); A=Z['A']; reg=Z['reg']; Dv=Z['Dv']; G=pickle.load(open('gt_kur.pkl','rb'))
TIF={'043_Гочево':'/workspace/data/train-expds-3/043_Гочево/03_Гочево_Ae_немецкая/01_Гочево_Ae_немецкая.tif',
     '185_Изборск':'/workspace/data/train-expds-3/185_Изборск/03_Изборск_Ae_немецкая/01_Изборск_Ae_немецкая.tif',
     '103_Гнёздово_Ольшанская':None}
H=120; PX=240
for r,f in TIF.items():
    if f is None: continue
    ws=[w for c in G for w,rr,mk in zip(G[c]['wkb'],G[c]['reg'],G[c]['mk']) if rr==r and mk=='ae']
    gg=shapely.from_wkb(np.array(ws,dtype=object)); allg=gg
    rng=np.random.default_rng(0); pick=rng.choice(len(gg),8,replace=False)
    m=reg==r; P=A[m,:2]; top=Dv[m]<=400
    tiles=[]
    with rasterio.open(f) as d:
        assert d.crs.to_epsg()==32600+int(str(d.crs.to_epsg())[-2:])
        for i in pick:
            cx,cy=shapely.get_coordinates(shapely.centroid(gg[i]))[0]
            b=(cx-H/2,cy-H/2,cx+H/2,cy+H/2); win=from_bounds(*b,d.transform)
            a=d.read(1,window=win,out_shape=(PX,PX),boundless=True,fill_value=0).astype(float)
            lo,hi=np.percentile(a[a>0],[2,98]) if (a>0).any() else (0,1); a=np.clip((a-lo)/(hi-lo+1e-6)*255,0,255).astype(np.uint8)
            im=Image.fromarray(a).convert('RGB'); dr=ImageDraw.Draw(im); s=PX/H
            tf=lambda x,y:((x-b[0])*s,(b[3]-y)*s)
            for g in allg:
                if g.distance(shapely.Point(cx,cy))<H:
                    for poly in getattr(g,'geoms',[g]): dr.line([tf(x,y) for x,y in poly.exterior.coords],fill=(255,0,0),width=1)
            q=(np.abs(P[:,0]-cx)<H/2)&(np.abs(P[:,1]-cy)<H/2)
            for (x,y),t in zip(P[q],top[q]):
                u,v=tf(x,y); dr.ellipse([u-2,v-2,u+2,v+2],outline=(0,255,0) if t else (255,255,0))
            tiles.append(im)
    W=Image.new('RGB',(PX*4,PX*2))
    for k,t in enumerate(tiles): W.paste(t,((k%4)*PX,(k//4)*PX))
    W.save(f'diag_gi_{r[:3]}.png'); print(r,'saved')
