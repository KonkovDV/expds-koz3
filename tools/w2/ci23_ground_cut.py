# CI2-3: does ground classification (class 2) cut mound fill? Compare per-kurgan heights on DEMs from different class sets.
import sys,json,glob,numpy as np,laspy
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform as stf
from scipy import ndimage as ndi
site=sys.argv[1]; R=0.5
las=glob.glob(f'{site}/01_*/*.las')[0]; epsg=json.load(open(f'{site}/UTM.json'))['crs'].split(':')[-1]
tr=Transformer.from_crs(3857,int(epsg),always_xy=True).transform
K=[]
for gj in sorted(glob.glob(f'{site}/06_*/*_Li_курганы_*.geojson')):
    cls='tselye' if 'целые' in gj else 'povr'
    for f in json.load(open(gj))['features']:
        g=stf(tr,shape(f['geometry'])); K.append((cls,g))
hdr=laspy.open(las).header; x0,y0=hdr.mins[0],hdr.mins[1]; W=int((hdr.maxs[0]-x0)/R)+1; H=int((hdr.maxs[1]-y0)/R)+1
lab=np.zeros((H,W),np.int32)  # 0 none, i+1 = kurgan i polygon or its annulus area
roi=np.zeros((H,W),bool)
for i,(c,g) in enumerate(K):
    rr=max(3.0,0.75*np.sqrt(g.area/np.pi)); b=g.buffer(rr).bounds
    c0,c1=int((b[0]-x0)/R),int((b[2]-x0)/R)+1; r0,r1=int((b[1]-y0)/R),int((b[3]-y0)/R)+1
    roi[max(r0,0):r1,max(c0,0):c1]=True
idx=np.flatnonzero(roi.ravel()); pos=-np.ones(H*W,np.int64); pos[idx]=np.arange(len(idx)); n=len(idx)
SETS={'g2':[2],'g2+1':[1,2],'nonveg_all':[1,2,3,8],'all_but_noise':[1,2,3,4,5,6,8,9]}
acc={k:dict(s=np.zeros(n),c=np.zeros(n),mn=np.full(n,np.inf)) for k in SETS}
cc=np.zeros((n,10))
with laspy.open(las) as fh:
    for p in fh.chunk_iterator(3_000_000):
        x=np.asarray(p.x); y=np.asarray(p.y); z=np.asarray(p.z); cl=np.asarray(p.classification)
        col=((x-x0)/R).astype(np.int64); row=((y-y0)/R).astype(np.int64); ok=(col>=0)&(col<W)&(row>=0)&(row<H)
        lin=np.where(ok,row*W+col,0); q=np.where(ok,pos[lin],-1); m=q>=0; q=q[m]; z=z[m]; cl=cl[m]
        np.add.at(cc,(q,np.minimum(cl,9)),1)
        for k,cs in SETS.items():
            s=np.isin(cl,cs); a=acc[k]; np.add.at(a['s'],q[s],z[s]); np.add.at(a['c'],q[s],1); np.minimum.at(a['mn'],q[s],z[s])
def grid(v):
    G=np.full(H*W,np.nan); G[idx]=v; return G.reshape(H,W)
D={}
for k,a in acc.items():
    D[k+'_mean']=grid(np.where(a['c']>0,a['s']/np.maximum(a['c'],1),np.nan)); D[k+'_min']=grid(np.where(a['c']>0,a['mn'],np.nan))
CC=np.zeros((H*W,10)); CC[idx]=cc; CC=CC.reshape(H,W,10)
from rasterio.features import rasterize
from rasterio.transform import Affine
T=Affine(R,0,x0,0,R,y0)  # row increases with y
rows=[]
for i,(c,g) in enumerate(K):
    rr=max(3.0,0.75*np.sqrt(g.area/np.pi)); b=g.buffer(rr).bounds
    c0,c1=max(int((b[0]-x0)/R),0),min(int((b[2]-x0)/R)+1,W); r0,r1=max(int((b[1]-y0)/R),0),min(int((b[3]-y0)/R)+1,H)
    t=Affine(R,0,x0+c0*R,0,R,y0+r0*R); sh=(r1-r0,c1-c0)
    pin=rasterize([(g,1)],sh,transform=t).astype(bool); pann=rasterize([(g.buffer(rr),1)],sh,transform=t).astype(bool)&~rasterize([(g.buffer(0.5),1)],sh,transform=t).astype(bool)
    if pin.sum()<4 or pann.sum()<4: continue
    r={'cls':c,'area':round(g.area,1)}
    for k,G in D.items():
        w=G[r0:r1,c0:c1]; a=w[pin]; b2=w[pann]
        if np.isfinite(a).sum()<3 or np.isfinite(b2).sum()<3: r['h_'+k]=None; continue
        r['h_'+k]=round(float(np.nanpercentile(a,95)-np.nanmedian(b2)),3)
        if k in('g2_mean',): r['hole_in']=round(float(np.isnan(a).mean()),3); r['hole_ann']=round(float(np.isnan(b2).mean()),3)
    ci=CC[r0:r1,c0:c1][pin].sum(0); ca=CC[r0:r1,c0:c1][pann].sum(0)
    r['cls_in']={str(j):round(float(ci[j]/max(ci.sum(),1)),3) for j in range(10) if ci[j]}; r['cls_ann']={str(j):round(float(ca[j]/max(ca.sum(),1)),3) for j in range(10) if ca[j]}
    r['g2_frac_in']=round(float(ci[2]/max(ci.sum(),1)),3); r['g2_frac_ann']=round(float(ca[2]/max(ca.sum(),1)),3)
    rows.append(r)
json.dump(rows,open(f'/workspace/koz3/w2/ci2_3/{site}.json','w'),ensure_ascii=False,indent=0)
import statistics as st
for c in ('tselye','povr'):
    R_=[r for r in rows if r['cls']==c and r.get('h_g2_mean') is not None and r.get('h_nonveg_all_mean') is not None]
    if not R_: continue
    f=lambda k:np.array([r[k] if r.get(k) is not None else np.nan for r in R_])
    print(site,c,'n',len(R_),'med h g2_mean',np.nanmedian(f('h_g2_mean')).round(3),'g2_min',np.nanmedian(f('h_g2_min')).round(3),'g2+1_min',np.nanmedian(f('h_g2+1_min')).round(3),'nonveg_min',np.nanmedian(f('h_nonveg_all_min')).round(3),'all_min',np.nanmedian(f('h_all_but_noise_min')).round(3))
    d=f('h_nonveg_all_min')-f('h_g2_min'); print('   Δh(nonveg_min - g2_min) med',np.nanmedian(d).round(3),'q90',np.nanquantile(d,.9).round(3),'max',np.nanmax(d).round(3),' rel med',np.nanmedian(d/np.maximum(f('h_nonveg_all_min'),0.05)).round(3))
    print('   g2 frac in/ann med',np.median(f('g2_frac_in')).round(3),np.median(f('g2_frac_ann')).round(3),'hole in/ann med',np.nanmedian(f('hole_in')).round(3),np.nanmedian(f('hole_ann')).round(3))
