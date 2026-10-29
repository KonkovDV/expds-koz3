import json, collections
PIX = [f for f in json.load(open('/workspace/koz3_shared/bot-3/pixgbm/pix_lofo.geojson'))['features'] if f['properties']['region_name'].split('_')[1]=='Гнёздово']
CL = ['finds_points', 'object_poly']; grp = lambda p: p.get('threshold_group') or p.get('markup_type')
cov = {c: {(f['properties']['region_name'], f['properties']['markup_type']) for f in PIX if f['properties']['class_name']==c} for c in CL}
out=[]; c=collections.Counter()
for l in open('fam_A/Гнёздово.jsonl', encoding='utf-8'):
    f=json.loads(l); p=f['properties']
    if p['class_name'] in CL and (p['region_name'], grp(p)) in cov[p['class_name']]: c['rm']+=1; continue
    out.append(f)
add=[f for f in PIX if f['properties']['class_name'] in CL]; out+=add; c['add']=len(add)
c['etgr_left_classic']=sum(1 for f in out if f['properties'].get('markup_type')=='etgr' and f['properties'].get('etgr_layers'))
import os; os.makedirs('fam_S', exist_ok=True)
with open('fam_S/Гнёздово.jsonl','w',encoding='utf-8') as fh:
    for f in out: fh.write(json.dumps(f, ensure_ascii=False)+'\n')
print(dict(c))
