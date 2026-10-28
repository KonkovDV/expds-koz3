"""bot-7. Инвентарь оптических сцен (SpOr/Ae/Or) с границами в UTM региона -> scenes.json"""
import glob, json, os, math, re
import rasterio
from pyproj import Transformer
D='/workspace/data/train-expds-3'; out=[]
boxes=set(json.load(open('/workspace/expds-koz3/notes/las_boxes_59.json')))
for f in sorted(glob.glob(f'{D}/*/*/*.tif')):
    b=os.path.basename(f)
    m=re.search(r'_(SpOr|Ae|Or)_',b)
    if not m: continue
    reg=f.split('/')[-3]; epsg=int(json.load(open(f'{D}/{reg}/UTM.json'))['crs'].split(':')[-1])
    try:
        with rasterio.open(f) as d:
            tr=Transformer.from_crs(d.crs,epsg,always_xy=True)
            L,B,R,Tp=d.bounds
            cs=[tr.transform(x,y) for x,y in ((L,Tp),(R,Tp),(R,B),(L,B))]
            xa,ya=tr.transform([L,L+d.transform.a*100],[Tp,Tp]); gsd=math.hypot(xa[1]-xa[0],ya[1]-ya[0])/100
            out.append(dict(reg=reg,file=b,kind=m.group(1),lidar=reg in boxes,w=d.width,h=d.height,gsd=gsd,corners=cs,epsg=epsg,count=d.count))
    except Exception as e: print('ERR',b,e)
json.dump(out,open('scenes.json','w'),ensure_ascii=False)
import collections; print(len(out), collections.Counter((s['kind'],s['lidar']) for s in out))
