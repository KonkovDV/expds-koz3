"""bot-7. Общие загрузки для task 6 (городища без лидара)."""
import json, pickle, sys
import numpy as np, shapely
from shapely.geometry import Polygon, Point
sys.path.insert(0,'/workspace/expds-koz3')
C='gorodishcha'
def load_gt():
    from expds import metric as M
    mod=M.load_official('/workspace/expds-koz3/materials/compute_metrics.py')
    T=pickle.load(open('/workspace/rerank/work/table.pkl','rb'))
    G=shapely.from_wkb(pickle.load(open('/workspace/rerank/work/gt_wkb.pkl','rb'))[C]); greg=np.array(T['greg'][C])
    gt=json.load(open('/workspace/rerank/gt_train_v2.geojson')); gmk=[]
    for f in gt['features']:
        p=f.get('properties') or {}
        if mod.normalize_class_name(p.get('class_name',''))==C and f.get('geometry'): gmk.append(p.get('markup_type'))
    assert len(gmk)==len(G)
    boxes=set(json.load(open('/workspace/expds-koz3/notes/las_boxes_59.json')))
    lid=np.array([r in boxes for r in greg])
    return T,G,greg,np.array(gmk),lid,mod
def load_scenes():
    S=json.load(open('/workspace/koz3_shared/bot-7/work/gor/scenes.json'))
    for s in S: s['poly']=Polygon(s['corners'])
    return S
