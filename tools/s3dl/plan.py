import json, os
from pathlib import Path
DEST=Path('/workspace/data/train-expds-3')
objs=[]
for l in open('/workspace/data/s3_listing.jsonl'):
    d=json.loads(l)
    if d.get('type')=='directory': continue
    objs.append((d['key'].replace('s3://train-expds-3/',''), int(d['size'])))
def dest(rel): return DEST/(rel[:-4]+'.laz' if rel.lower().endswith('.las') else rel)
ext=lambda r: Path(r).suffix.lower()
tif=[o for o in objs if ext(o[0]) in ('.tif','.tiff')]; las=[o for o in objs if ext(o[0])=='.las']; oth=[o for o in objs if o not in tif and o not in las]
print(len(objs), sum(s for _,s in objs)/1e9, 'tif',len(tif),sum(s for _,s in tif)/1e9,'las',len(las),sum(s for _,s in las)/1e9,'oth',len(oth),sum(s for _,s in oth)/1e6,'MB')
todo=[o for o in tif+oth+las if not dest(o[0]).exists()]
for g,name in ((tif,'tif'),(oth,'oth'),(las,'las')):
    t=[o for o in g if not dest(o[0]).exists()]; print('todo',name,len(t),sum(s for _,s in t)/1e9)
big=sorted(objs,key=lambda o:-o[1])[:8]; print(big)
