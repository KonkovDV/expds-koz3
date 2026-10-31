"""Диагностика: почему кандидаты выше порога не попадают и распределение ошибок центра/размера."""
from common7 import *
import expds.metric as _m
import numpy as np, shapely, json
from collections import defaultdict
mod=_m.load_official(METRIC); calc=mod.F1Calculator()
st=load(W/'stack_sub6000.geojson'); gt=load('/workspace/rerank/gt_train_v2.geojson')
cur=build_curves(st,gt,METRIC); ins,res=insample(cur)
P=_m._collect(mod,calc,st,False); G=_m._collect(mod,calc,gt,True)
del st
out={}
for c in ['kurgany_povrezhdennye','kurgany_tselye','gorodishcha','object_poly','arkhitektury','fortifikatsii']:
    gg=[x[0] for x in G.get(c,[])]; greg=[x[2] for x in G.get(c,[])]
    tree=shapely.STRtree(gg)
    p=P[c]; order=sorted(range(len(p)),key=lambda i:p[i][1],reverse=True)
    hits=cur[c].hit
    taus={g:v['threshold'] for g,v in res[c]['groups'].items()}
    stats=defaultdict(lambda: defaultdict(int)); err=defaultdict(lambda: {'d_rel':[],'s_rel':[],'d_abs':[]})
    for r,i in enumerate(order):
        geom,conf,reg,grp=p[i]
        if conf<taus.get(grp,np.inf): continue
        s=stats[grp]; s['kept']+=1
        if hits[r]: s['hit']+=1
        part=geom if geom.geom_type=='Polygon' else max(geom.geoms,key=lambda q:q.area)
        rp=part.representative_point()
        cand=tree.query(part)
        cand=[j for j in cand if greg[j]==reg]
        inG=[j for j in cand if gg[j].covers(rp)]
        if not hits[r]:
            if inG:
                # почему не прошло: покрытие или эталон уже занят
                cov=max(gg[j].intersection(part).area/min(part.area,gg[j].area) for j in inG)
                s['miss_rp_in_gt_cov<0.5' if cov<0.5 else 'miss_rp_in_gt_taken']+=1
            elif cand: s['miss_intersects_only']+=1
            else: s['miss_no_overlap']+=1
        # ошибки относительно ближайшего эталона (в пределах 3 r_eq)
        near=tree.query_nearest(rp,max_distance=200)
        near=[j for j in near if greg[j]==reg]
        if near:
            j=near[0]; R=np.sqrt(gg[j].area/np.pi); d=rp.distance(gg[j].centroid)
            if d<3*R:
                e=err[grp]; e['d_rel'].append(d/R); e['d_abs'].append(d); e['s_rel'].append(np.sqrt(part.area/np.pi)/R)
    out[c]={g:dict(v) for g,v in stats.items()}
    print('\n==',c,'F1 ins',round(res[c]['f1'],4))
    for g,v in stats.items():
        e=err[g]; pc=lambda a:' '.join(f'{x:.2f}' for x in np.percentile(a,[10,25,50,75,90])) if a else '-'
        print(f'  {g:10s} {dict(v)}\n     d/R p10..90: {pc(e["d_rel"])} | d m: {pc(e["d_abs"])} | r_pred/R: {pc(e["s_rel"])}')
        out[c][g]['err']={k:np.percentile(a,[10,25,50,75,90]).round(3).tolist() if a else None for k,a in e.items()}
json.dump(out,open(W/'diag.json','w'),ensure_ascii=False,indent=1)
