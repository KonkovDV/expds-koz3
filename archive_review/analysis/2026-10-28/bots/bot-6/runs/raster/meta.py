import json, sys, glob, os, numpy as np, rasterio
from rasterio.enums import Resampling
out=[]
for p in sorted(glob.glob("/workspace/data/train-expds-3/*/03_*_Ae_*/*.tif")+glob.glob("/workspace/data/train-expds-3/*/05_*_Or_*/*.tif")):
    with rasterio.open(p) as ds:
        f=max(1,int(np.ceil(max(ds.width,ds.height)/1024)))
        a=ds.read(out_shape=(ds.count,max(1,ds.height//f),max(1,ds.width//f)),resampling=Resampling.nearest)
        nd=ds.nodata
        row=dict(file=os.path.relpath(p,"/workspace/data/train-expds-3"),size_mb=round(os.path.getsize(p)/1e6,1),
                 crs=ds.crs.to_string() if ds.crs else None, res=[round(ds.res[0],4),round(ds.res[1],4)],
                 wh=[ds.width,ds.height], count=ds.count, dtype=ds.dtypes[0], nodata=nd,
                 compress=ds.compression.value if ds.compression else None,
                 colorinterp=[c.name for c in ds.colorinterp], overview_factor=f,
                 tags_area=ds.tags().get("AREA_OR_POINT"))
        bands=[]
        for b in range(a.shape[0]):
            x=a[b].astype(np.float64); valid=x!=0 if nd is None else x!=nd
            v=x[valid]
            bands.append(dict(band=b+1, valid_share=round(float(valid.mean()),4),
                 min=float(v.min()) if v.size else None, p1=float(np.percentile(v,1)) if v.size else None,
                 p50=float(np.percentile(v,50)) if v.size else None, p99=float(np.percentile(v,99)) if v.size else None,
                 max=float(v.max()) if v.size else None, share_255=round(float((v>=255).mean()),4) if v.size else None,
                 share_le1=round(float((v<=1).mean()),4) if v.size else None))
        if a.shape[0]>=3:
            row["bands_identical"]=bool(np.array_equal(a[0],a[1]) and np.array_equal(a[1],a[2]))
        row["bands"]=bands
    out.append(row); print(json.dumps(row,ensure_ascii=False))
json.dump(out,open(sys.argv[1],"w"),ensure_ascii=False,indent=1)
