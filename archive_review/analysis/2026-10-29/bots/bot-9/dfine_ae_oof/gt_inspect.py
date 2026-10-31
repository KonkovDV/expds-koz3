import json,glob,collections,sys
d=json.load(open('/workspace/kaggle_out/koz3-dfine-ae/dfine_report.json'))
regs=sorted({r for f in d['folds'].values() for r in f['test_regions']})
print(len(regs))
for r in regs:
    u=json.load(open(f'/workspace/data/train-expds-3/{r}/UTM.json')) if glob.glob(f'/workspace/data/train-expds-3/{r}/UTM.json') else None
    fs=glob.glob(f'/workspace/data/train-expds-3/{r}/06_*/*_Ae_*.geojson')
    types=collections.Counter(); crs=set()
    for f in fs:
        j=json.load(open(f)); crs.add(json.dumps(j.get('crs'))[:60])
        for ft in j['features']: types[(ft.get('geometry') or {}).get('type')]+=1
    print(r,u,len(fs),dict(types),crs)
