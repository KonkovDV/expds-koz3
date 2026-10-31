import json,sys,glob
sys.path.insert(0,'/workspace/koz3_shared/bot-2/repo/materials'); import compute_metrics as cm
from shapely.geometry import shape
c=cm.F1Calculator(); u=lambda g: cm._to_utm(g,'webmercator')
pr=[(f['properties'],u(shape(f['geometry']).buffer(0))) for f in json.load(open('/workspace/kaggle_out/koz3-dfine-or/oof_dfine_or.geojson'))['features']]
for r,pat in [('185_Изборск','*_городища.geojson'),('033_Елизаветинское','*_городища.geojson')]:
  for f in sorted(glob.glob(f'/workspace/data/train-expds-3/{r}/06_*/{pat}')):
    for i,ft in enumerate(json.load(open(f))['features']):
      if not ft.get('geometry'): print(f.split('/')[-1],i,'no geom'); continue
      g=u(shape(ft['geometry']).buffer(0))
      hits=sorted([(round(p['confidence'],4),p['class_name'],j) for j,(p,q) in enumerate(pr) if p['region_name']==r and c._match_polygon_gt(q,g)[0]],reverse=True)
      print(f.split('/')[-1],i,round(g.area,1),'gor_hits',[h for h in hits if h[1]=='gorodishcha'][:4],'n_hits_any',len(hits),'top_any',hits[:2])
