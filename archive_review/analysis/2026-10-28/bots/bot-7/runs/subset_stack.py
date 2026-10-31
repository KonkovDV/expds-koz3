"""Потоковая выборка стека: на (класс, группа) не больше TOPN самых уверенных кандидатов."""
import ijson, numpy as np, json, sys
TOPN=int(sys.argv[1]) if len(sys.argv)>1 else 20000
confs=np.load('work/confs.npy',allow_pickle=True).item()
cut={k:(v[TOPN-1] if len(v)>TOPN else -1.0) for k,v in confs.items()}
n=0
with open('/workspace/rerank/stack_arch.geojson','rb') as f, open('work/stack_sub%d.geojson'%TOPN,'w') as o:
    o.write('{"type": "FeatureCollection", "features": [\n')
    first=True
    for feat in ijson.items(f,'features.item', use_float=True):
        p=feat['properties']; k=p['class_name']+'|'+(p.get('threshold_group') or p.get('markup_type'))
        if p['confidence']<cut[k]: continue   # ties at cut kept
        o.write(('' if first else ',\n')+json.dumps(feat,ensure_ascii=False)); first=False; n+=1
    o.write('\n]}\n')
print('kept',n, {k:c for k,c in cut.items() if c>0})
