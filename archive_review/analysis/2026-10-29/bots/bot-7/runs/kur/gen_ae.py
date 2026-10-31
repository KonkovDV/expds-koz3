"""bot-7 task7. Генератор кандидатов курганов на НЕлидарных Ae-сценах (немецкая аэрофотосъёмка), без меток.
Рабочая сетка RES=1 м (усреднение), тайлы 2000 м + поле 60 м. Детектор: max по масштабам |LoG_норм| (r=2.5..14 м),
NMS 7 м, до CAP пиков на км^2 ядра тайла. Признаки в пике (все без меток): LoG по масштабам со знаком, лучший масштаб,
mean/std яркости 5/15/45 м, градиент 5/15 м, гессиан (λ1, λ2) на 2 масштабах, диполь тени (|∇G_r|) на 2 масштабах,
FRST(abs) n=3,5,8 м, доля валидных. Выход raw/{si:03d}.npz: A (N x 5+F) = ux, uy, score, scale_idx, si; F признаков."""
import os, json, math, time, resource, glob
import numpy as np, rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from scipy import ndimage as ndi
from pyproj import Transformer
OUT='/workspace/koz3_shared/bot-7/work/kur/raw'; os.makedirs(OUT,exist_ok=True)
D='/workspace/data/train-expds-3'
S=json.load(open('/workspace/koz3_shared/bot-7/work/gor/scenes.json'))
RES=float(os.environ.get('RES','0.5')); RADII=[2,3,4.5,6.5,9,13]; NMS=float(os.environ.get('NMS','4')); DD=3.; CAP=int(os.environ.get('CAP','1500')); TILE=1000; MG=60
FN=[f'log_{r}' for r in RADII]+['best_r','m5','m15','m45','s5','s15','s45','gm5','gm15','l1_5','l2_5','l1_10','l2_10','dip5','dip10','frst2','frst4','frst7','validf']
def frst(gx,gy,mag,n,thr):
    H,W=mag.shape; ys,xs=np.nonzero(mag>thr)
    if len(ys)==0: return np.zeros((H,W),np.float32)
    m=mag[ys,xs]; ux=gx[ys,xs]/m; uy=gy[ys,xs]/m; F=np.zeros((H,W),np.float32)
    for sgn in (1,-1):
        px=np.clip(np.round(xs+sgn*n*ux).astype(int),0,W-1); py=np.clip(np.round(ys+sgn*n*uy).astype(int),0,H-1)
        O=np.bincount(py*W+px,minlength=H*W).astype(np.float32).reshape(H,W)
        F+=ndi.gaussian_filter(np.minimum(O,9.9)/9.9,0.25*n+0.5)
    return F
def hess(g,s):
    xx=ndi.gaussian_filter(g,s,order=(0,2)); yy=ndi.gaussian_filter(g,s,order=(2,0)); xy=ndi.gaussian_filter(g,s,order=(1,1))
    tr=(xx+yy)/2; dt=np.sqrt(((xx-yy)/2)**2+xy**2)
    return (tr+dt)*s*s,(tr-dt)*s*s
SC=[i for i,s in enumerate(S) if s['kind']=='Ae' and not s['lidar']]
only=[int(x) for x in os.environ.get('ONLY','').split(',') if x]
t0=time.time()
for si in SC:
    s=S[si]
    if only and si not in only: continue
    of=f'{OUT}/{si:03d}.npz'
    if os.path.exists(of): continue
    f=glob.glob(f"{D}/{s['reg']}/*/{s['file']}")[0]
    rec=[]; feats=[]
    with rasterio.open(f) as d:
        tr=Transformer.from_crs(d.crs,s['epsg'],always_xy=True)
        res=max(RES,s['gsd']); sc=res/s['gsd']; mg=int(MG/res); tl=int(TILE/res)
        Wt=int(d.width/sc); Ht=int(d.height/sc)
        for ty in range(0,max(Ht,1),tl):
            for tx in range(0,max(Wt,1),tl):
                x0=max(tx-mg,0); y0=max(ty-mg,0); x1=min(tx+tl+mg,Wt); y1=min(ty+tl+mg,Ht)
                if x1-x0<40 or y1-y0<40: continue
                win=Window(x0*sc,y0*sc,(x1-x0)*sc,(y1-y0)*sc)
                a=d.read(window=win,out_shape=(d.count,y1-y0,x1-x0),resampling=Resampling.average,boundless=True,fill_value=0).astype(np.float32)
                if d.count>=4 and d.colorinterp[3].name=='alpha': valid=a[3]>0
                else: valid=np.any(a[:min(3,d.count)]!=0,0)
                g=a[:min(3,d.count)].mean(0); del a
                # чёрные рамки (не nodata): почти нулевая яркость сплошными полями
                if valid.any():
                    lo0=np.percentile(g[valid],1); bl=min(lo0,0.03*np.percentile(g[valid],99))+1e-6; valid&=~ndi.binary_opening(g<=bl,iterations=5)
                if valid.mean()<0.05: continue
                v=g[valid]; lo,hi=np.percentile(v,[2,98]); g=np.clip((g-lo)/max(hi-lo,1e-6),0,1); g[~valid]=float(np.median(g[valid]))
                g=ndi.gaussian_filter(g,0.7).astype(np.float32)
                validE=ndi.binary_erosion(valid,iterations=int(15/res))
                core=np.zeros_like(valid); core[ty-y0:ty-y0+tl, tx-x0:tx-x0+tl]=True
                L=np.stack([(-ndi.gaussian_laplace(g,r/res/math.sqrt(2))*(r/res)**2/2).astype(np.float32) for r in RADII])
                bi=np.abs(L).argmax(0).astype(np.int8)
                km2=(validE&core).sum()*res*res/1e6; capn=max(50,int(CAP*km2))
                P=[]
                for k,r in enumerate(RADII):
                    for sg in (1,-1):
                        A=sg*L[k]; A[~validE]=0
                        mf=ndi.maximum_filter(A,size=int(2*max(NMS,r)/res)+1)
                        yy,xx=np.nonzero((A==mf)&validE&core&(A>0)); P.append(np.c_[yy,xx,A[yy,xx]])
                P=np.concatenate(P); P=P[np.argsort(-P[:,2])]
                # дедуп: в клетке DD м остаётся сильнейший
                key=(P[:,0]*res//DD).astype(np.int64)*100000+(P[:,1]*res//DD).astype(np.int64)
                _,first=np.unique(key,return_index=True); P=P[np.sort(first)][:capn]
                ys,xs,vv=P[:,0].astype(int),P[:,1].astype(int),P[:,2]; del A,mf
                if len(ys)==0: continue
                F=[L[:,ys,xs].T, bi[ys,xs][:,None].astype(np.float32)]; del L
                for w in (5,15,45):
                    k=int(2*w/res)+1; m=ndi.uniform_filter(g,k); m2=ndi.uniform_filter(g*g,k)
                    F.append(np.c_[m[ys,xs],np.sqrt(np.maximum(m2-m*m,0))[ys,xs]])
                F=[F[0],F[1],np.c_[F[2][:,0],F[3][:,0],F[4][:,0],F[2][:,1],F[3][:,1],F[4][:,1]]]
                gx=ndi.sobel(g,1); gy=ndi.sobel(g,0); mag=np.hypot(gx,gy)
                F.append(np.c_[ndi.uniform_filter(mag,int(5/res)*2+1)[ys,xs],ndi.uniform_filter(mag,int(15/res)*2+1)[ys,xs]])
                hl=[]
                for r in (5,10):
                    l1,l2=hess(g,r/res/math.sqrt(2)); hl+= [l1[ys,xs],l2[ys,xs]]
                F.append(np.c_[tuple(hl)])
                dp=[]
                for r in (5,10):
                    s_=r/res; dp.append(np.hypot(ndi.gaussian_filter(g,s_,order=(0,1)),ndi.gaussian_filter(g,s_,order=(1,0)))[ys,xs]*s_)
                F.append(np.c_[tuple(dp)])
                thr=np.percentile(mag[validE],80); mag[~validE]=0
                F.append(np.c_[tuple(frst(gx,gy,mag,n/res,thr)[ys,xs] for n in (2,4,7))])
                F.append(ndi.uniform_filter(valid.astype(np.float32),int(30/res)+1)[ys,xs][:,None])
                F=np.concatenate(F,1).astype(np.float32); assert F.shape[1]==len(FN)
                X0,Y0=d.xy(((ys+y0+0.5)*sc).tolist(),((xs+x0+0.5)*sc).tolist(),offset='ul')
                ux,uy=tr.transform(np.array(X0),np.array(Y0))
                rec.append(np.c_[ux,uy,vv,bi[ys,xs],np.full(len(ys),si)]); feats.append(F)
    R=np.concatenate(rec) if rec else np.zeros((0,5)); Fz=np.concatenate(feats) if feats else np.zeros((0,len(FN)),np.float32)
    np.savez(of,A=R.astype(np.float64),F=Fz,FN=np.array(FN))
    print(f'[{time.time()-t0:.0f}s] {si} {s["reg"]} {s["file"]} n={len(R)} rss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f}MB',flush=True)
print('done',flush=True)
