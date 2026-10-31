"""bot-7 task7 шаг 3: почему Гочево и Изборск почти не находятся. Без обучения, только геометрия.
(а) расстояние от центра эталона ae до ближайшего кандидата (все D), (б) поиск систематического сдвига ±40 м шаг 1 м
(сколько эталонов получают кандидата в 1,5 м от центра+сдвиг), (в) Изборск: эталон ae против sp_or того же участка, (г) эталон внутри кадра Ae."""
import json,pickle,numpy as np,shapely
from scipy.spatial import cKDTree
Z=np.load('cands_all.npz',allow_pickle=True); A=Z['A']; reg=Z['reg']; Dv=Z['Dv']
G=pickle.load(open('gt_kur.pkl','rb')); S=json.load(open('../gor/scenes.json'))
REGS=['043_Гочево','185_Изборск','103_Гнёздово_Ольшанская','012_Сож','059_Шишкино','060_Сельцо','029_Вышегжа_3','110_Саратов']
out={}
def gt(r,mk):
    ws=[];cl=[]
    for c in G:
        m=(G[c]['reg']==r)&(G[c]['mk']==mk); ws+=list(G[c]['wkb'][m]); cl+=[c[8:12]]*int(m.sum())
    g=shapely.from_wkb(np.array(ws,dtype=object)) if ws else np.array([]); return g,np.array(cl)
sh=np.arange(-40,41)
for r in REGS:
    g,cl=gt(r,'ae')
    if len(g)==0: continue
    ctr=shapely.get_coordinates(shapely.centroid(g)); rad=np.sqrt(shapely.area(g)/np.pi)
    fp=shapely.union_all([shapely.Polygon(s['corners']) for s in S if s['reg']==r and s['kind']=='Ae'])
    inside=shapely.contains_xy(fp,ctr[:,0],ctr[:,1])
    m=reg==r; P=A[m,:2]; top=Dv[m]<=400
    res={'n_gt':len(g),'gt_eq_radius_p50':float(np.median(rad)),'gt_inside_Ae_frame':int(inside.sum()),'n_cand':int(m.sum())}
    for nm,PP in (('all',P),('D400',P[top])):
        t=cKDTree(PP); d,_=t.query(ctr); res[f'nn_{nm}']=dict(p10=float(np.percentile(d,10)),p50=float(np.median(d)),le2=int((d<=2).sum()),le5=int((d<=5).sum()),le_rad=int((d<=rad).sum()),gt50=int((d>50).sum()))
        H=np.zeros((len(sh),len(sh)),int)
        for i,dx in enumerate(sh):
            for j,dy in enumerate(sh):
                H[i,j]=int((t.query(ctr+[dx,dy],distance_upper_bound=1.5)[0]<np.inf).sum())
        a=np.unravel_index(H.argmax(),H.shape)
        res[f'shift_{nm}']=dict(at0=int(H[40,40]),best=int(H[a]),best_dxdy=[int(sh[a[0]]),int(sh[a[1]])],bg_median=float(np.median(H)),bg_p99=float(np.percentile(H,99)))
    if r=='185_Изборск':
        g2,_=gt(r,'sp_or'); c2=shapely.get_coordinates(shapely.centroid(g2)); t2=cKDTree(c2); d,j=t2.query(ctr)
        v=c2[j]-ctr; close=d<=30
        res['ae_vs_sp_or']=dict(n_sp_or=len(g2),nn_dist_p50=float(np.median(d)),n_within30=int(close.sum()),median_vec_within30=np.median(v[close],0).round(2).tolist() if close.any() else None,
            n_overlap=int(sum(shapely.intersects(g[i],g2[j[i]]) for i in range(len(g)))))
        # поиск сдвига ae->sp_or
        H=np.zeros((len(sh),len(sh)),int)
        for i,dx in enumerate(sh):
            for k,dy in enumerate(sh): H[i,k]=int((t2.query(ctr+[dx,dy],distance_upper_bound=3)[0]<np.inf).sum())
        a=np.unravel_index(H.argmax(),H.shape); res['ae_to_sp_or_shift']=dict(at0=int(H[40,40]),best=int(H[a]),best_dxdy=[int(sh[a[0]]),int(sh[a[1]])],bg_median=float(np.median(H)))
    out[r]=res; print(r,json.dumps(res,ensure_ascii=False),flush=True)
json.dump(out,open('diag_gi.json','w'),ensure_ascii=False,indent=1)
