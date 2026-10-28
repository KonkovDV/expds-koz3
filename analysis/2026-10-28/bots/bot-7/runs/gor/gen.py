"""bot-7, task 6. Генераторы кандидатов городищ без лидара (только оптика: SpOr/Ae/Or).
Без меток: меток здесь нет вообще. Выход: raw/{i:03d}.npz по сцене (x,y UTM, score, gen-id, R).
Генераторы:
  frst_abs/bright/dark — радиальная симметрия (Loy&Zelinsky 2003) по градиенту, радиусы RADII м: кольца/дуги валов и рвов;
  log_bright/dark     — LoG-пятна (площадка/впадина) на масштабах RADII;
  grad                — плотность сильных градиентов (текстура валов) в окне R;
Пики: максимум-фильтр NMS 40 м, до CAP пиков на км^2 на генератор-радиус (минимум 30 на сцену)."""
import os, sys, json, math, time, resource
import numpy as np, rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from scipy import ndimage as ndi
from pyproj import Transformer
OUT='/workspace/koz3_shared/bot-7/work/gor/raw'; os.makedirs(OUT,exist_ok=True)
D='/workspace/data/train-expds-3'
S=json.load(open('/workspace/koz3_shared/bot-7/work/gor/scenes.json'))
RADII=[15,25,40,60,90,140]; NMS=40.; CAP=60
GENS=['frst_abs','frst_bright','frst_dark','log_bright','log_dark','grad']
def frst(gx,gy,mag,n,thr):
    H,W=mag.shape; ys,xs=np.nonzero(mag>thr)
    if len(ys)==0: return np.zeros((H,W),np.float32),np.zeros((H,W),np.float32)
    m=mag[ys,xs]; ux=gx[ys,xs]/m; uy=gy[ys,xs]/m
    out=[]
    for sgn in (1,-1):
        px=np.clip(np.round(xs+sgn*n*ux).astype(int),0,W-1); py=np.clip(np.round(ys+sgn*n*uy).astype(int),0,H-1)
        O=np.zeros(H*W,np.float32); M=np.zeros(H*W,np.float32)
        np.add.at(O,py*W+px,1.0); np.add.at(M,py*W+px,m)
        O=O.reshape(H,W); M=M.reshape(H,W)
        F=M*np.minimum(O,9.9)/9.9
        out.append(ndi.gaussian_filter(F,0.25*n+1))
    return out  # [+]: градиент к центру=светлый центр, [-]: тёмный центр
def peaks(A,valid,core,res,cap_n):
    mf=ndi.maximum_filter(A,size=int(2*NMS/res)+1)
    pk=(A==mf)&valid&core&(A>0)
    ys,xs=np.nonzero(pk); v=A[ys,xs]; o=np.argsort(-v)[:cap_n]
    return ys[o],xs[o],v[o]
only=[int(x) for x in os.environ.get('ONLY','').split(',') if x]
t0=time.time()
for si,s in enumerate(S):
    if only and si not in only: continue
    of=f'{OUT}/{si:03d}.npz'
    if os.path.exists(of): continue
    if s['gsd']<0.01: print('skip gsd',s['file']); continue
    res=3.0 if s['kind']=='Ae' else 2.0
    f=f"{D}/{s['reg']}/" ; import glob; f=[p for p in glob.glob(f+'*/'+s['file'])][0]
    rec=[]
    with rasterio.open(f) as d:
        tr=Transformer.from_crs(d.crs,s['epsg'],always_xy=True)
        sc=res/s['gsd']  # исходных пикселей на рабочий
        TILE=1600; MG=int(200/res)  # рабочих пикселей; поле 200 м
        Wt=int(d.width/sc); Ht=int(d.height/sc)
        for ty in range(0,max(Ht,1),TILE):
            for tx in range(0,max(Wt,1),TILE):
                x0=max(tx-MG,0); y0=max(ty-MG,0); x1=min(tx+TILE+MG,Wt); y1=min(ty+TILE+MG,Ht)
                if x1-x0<20 or y1-y0<20: continue
                win=Window(x0*sc,y0*sc,(x1-x0)*sc,(y1-y0)*sc)
                a=d.read(window=win,out_shape=(d.count,y1-y0,x1-x0),resampling=Resampling.average,boundless=True,fill_value=0).astype(np.float32)
                if d.count>=4 and d.colorinterp[3].name=='alpha': valid=a[3]>0
                else: valid=np.any(a[:min(3,d.count)]!=0,0)
                if valid.mean()<0.05: continue
                g=a[:min(3,d.count)].mean(0)
                v=g[valid]; lo,hi=np.percentile(v,[2,98]); g=np.clip((g-lo)/max(hi-lo,1e-6),0,1); g[~valid]=np.median(g[valid])
                g=ndi.gaussian_filter(g,1.0)
                gx=ndi.sobel(g,1); gy=ndi.sobel(g,0); mag=np.hypot(gx,gy)
                validE=ndi.binary_erosion(valid,iterations=3)
                thr=np.percentile(mag[validE],80) if validE.any() else 1e9
                mag[~validE]=0
                core=np.zeros_like(valid); core[ty-y0:ty-y0+TILE, tx-x0:tx-x0+TILE]=True
                km2=valid[core].sum()*res*res/1e6; cap_n=max(30,int(CAP*km2))
                for R in RADII:
                    n=R/res
                    if n<3: continue
                    Fp,Fm=frst(gx,gy,mag,n,thr)
                    maps={'frst_bright':Fp,'frst_dark':Fm,'frst_abs':Fp+Fm}
                    L=-ndi.gaussian_laplace(g,R/res/math.sqrt(2))*(R/res)**2/2
                    maps['log_bright']=L; maps['log_dark']=-L
                    maps['grad']=ndi.uniform_filter((mag>thr).astype(np.float32),int(2*R/res)+1)
                    for gi,nm in enumerate(GENS):
                        A=maps[nm].astype(np.float32); A[~validE]=0
                        # нормировка внутри тайла: доля к 99-му перцентилю (без меток)
                        q=np.percentile(A[validE],99) if validE.any() else 1
                        ys,xs,vv=peaks(A,validE,core,res,cap_n)
                        if len(ys)==0: continue
                        X0,Y0=d.xy(((ys+y0+0.5)*sc).tolist(),((xs+x0+0.5)*sc).tolist(),offset='ul')
                        ux,uy=tr.transform(np.array(X0),np.array(Y0))
                        rec.append(np.c_[ux,uy,vv/max(q,1e-9),np.full(len(ys),gi),np.full(len(ys),R)])
    A=np.concatenate(rec) if rec else np.zeros((0,5))
    np.savez(of,A=A.astype(np.float64))
    print(f'[{time.time()-t0:.0f}s] {si} {s["reg"]} {s["file"]} n={len(A)} rss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f}MB',flush=True)
print('done')
