# CI2-8: fit organizers' Li_g / Li_i hillshade style against our LAS DEM (vendor build_dem, Li grid).
import sys,glob,json,math,time,numpy as np,rasterio
sys.path.insert(0,'/workspace/koz3/w2/vendor')
from expds import lidar, accel
from rasterio.windows import from_bounds
from scipy import ndimage as ndi
site=sys.argv[1]; MAXPIX=float(sys.argv[2]) if len(sys.argv)>2 else 60e6
gp=glob.glob(f'{site}/02_*/*_Li_g.tif')[0]; ip=glob.glob(f'{site}/02_*/*_Li_i.tif')[0]; las=glob.glob(f'{site}/01_*/*.las')[0]
src=rasterio.open(gp); lres=src.res[0]
t=time.time(); d=lidar.build_dem(las,res=lres,max_pixels=int(MAXPIX)); tb=time.time()-t
H,W=d.dem.shape; b=(d.left,d.top-H*d.res,d.left+W*d.res,d.top)
win=from_bounds(*b,transform=src.transform).round_offsets().round_lengths()
from rasterio.enums import Resampling
rd=lambda ds:ds.read(1,window=win,boundless=True,fill_value=255,out_shape=(H,W),resampling=Resampling.average if d.res>lres+1e-9 else Resampling.nearest)
G=rd(src); I=rd(rasterio.open(ip))
valid=np.isfinite(d.dem); z=np.where(valid,d.dem,np.nanmin(d.dem)).astype(np.float32)
ok=valid&(G<255)&ndi.binary_erosion(valid,iterations=3)
rng=np.random.default_rng(0); sel=np.flatnonzero(ok.ravel()); sel=rng.choice(sel,min(100000,len(sel)),replace=False)
def grads(zz,res,horn):
    if horn:
        k=np.array([[1,0,-1],[2,0,-2],[1,0,-1]],np.float32)/(8*res)
        gx=-ndi.correlate(zz,k,mode='nearest'); gy=ndi.correlate(zz,k.T,mode='nearest')  # gy: north-positive
    else:
        gx=np.gradient(zz,res,axis=1); gy=-np.gradient(zz,res,axis=0)
    return gx.ravel()[sel],gy.ravel()[sel]
def shade(gx,gy,az,alt,zf):
    sl=np.arctan(np.hypot(gx,gy)*zf); asp=np.arctan2(-gx,-gy); zen=math.radians(90-alt)
    return math.cos(zen)*np.cos(sl)+math.sin(zen)*np.sin(sl)*np.cos(math.radians(az)-asp)
def fit(target,gx,gy,grid):
    y=target.ravel()[sel].astype(np.float64); best=None
    for az in grid['az']:
        for alt in grid['alt']:
            for zf in grid['z']:
                s=shade(gx,gy,az,alt,zf); A=np.c_[s,np.ones_like(s)]; coef,*_=np.linalg.lstsq(A,y,rcond=None)
                p=np.clip(A@coef,0,255); mae=float(np.abs(p-y).mean()); r=float(np.corrcoef(s,y)[0,1])
                if best is None or mae<best['mae']: best=dict(az=az,alt=alt,z=zf,a=round(coef[0],2),b=round(coef[1],2),mae=round(mae,3),r=round(r,4))
    return best
out={'site':site,'li_res':lres,'dem_res':d.res,'shape':[H,W],'build_s':round(tb,1)}
coarse=dict(az=list(range(0,360,15)),alt=[15,25,35,45,55,65,75],z=[0.5,1,2,3,5])
for horn in (False,True):
    gx,gy=grads(z,d.res,horn)
    for nm,T in (('g',G),('i',I)):
        bc=fit(T,gx,gy,coarse)
        fine=dict(az=[bc['az']+k for k in (-10,-5,0,5,10)],alt=[max(5,bc['alt']+k) for k in (-8,-4,0,4,8)],z=[bc['z']*f for f in (0.7,0.85,1,1.2,1.4)])
        out[f'{nm}_{"horn" if horn else "grad"}']=fit(T,gx,gy,fine)
    gx0,gy0=gx,gy
# our default as-is (ESRI 315/45/z1, a=255,b=0) MAE
s=np.clip(np.rint(255*shade(gx0,gy0,315,45,1)),0,255); out['ours_default_g_mae']=round(float(np.abs(s-G.ravel()[sel]).mean()),3); out['ours_default_g_eq1']=round(float((np.abs(s-G.ravel()[sel])<=1).mean()),3)
out['ours_default_i_mae_255minus']=round(float(np.abs((255-s)-I.ravel()[sel]).mean()),3)
# smoothing sensitivity at best g params
bg=out['g_horn']
for sg in (1,2,3,5):
    gx,gy=grads(ndi.gaussian_filter(z,sg),d.res,True); s=shade(gx,gy,bg['az'],bg['alt'],bg['z']); y=G.ravel()[sel]; A=np.c_[s,np.ones_like(s)]; c,*_=np.linalg.lstsq(A,y,rcond=None)
    out[f'g_horn_smooth{sg}px']=dict(mae=round(float(np.abs(np.clip(A@c,0,255)-y).mean()),3),a=round(float(c[0]),1),b=round(float(c[1]),1),r=round(float(np.corrcoef(s,y)[0,1]),4))
gi=G.ravel()[sel].astype(int)+I.ravel()[sel]; out['g_plus_i']=dict(med=float(np.median(gi)),p5=float(np.percentile(gi,5)),p95=float(np.percentile(gi,95)))
out['li_nodata255_frac']=round(float((G==255).mean()),3); out['our_valid_frac']=round(float(valid.mean()),3)
print(json.dumps(out,ensure_ascii=False)); json.dump(out,open(f'/workspace/koz3/w2/ci2_8/{site}.json','w'),ensure_ascii=False,indent=1)
