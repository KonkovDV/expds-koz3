import pickle, numpy as np, json
T = pickle.load(open('/workspace/rerank/work/table.pkl','rb'))
print(list(T.keys()))
cls=T['cls'].astype(int); print('N',len(cls),'cls-1',int((cls<0).sum()))
print('mk', {m:int((T['mk']==m).sum()) for m in np.unique(T['mk'])})
A=T['CL'].index('arkhitektury'); a=cls==A
print('arch grp', {g:int(((T['grp']==g)&a).sum()) for g in np.unique(T['grp'][a])})
print('regions', len(np.unique(T['reg'])))
fg=np.load('../orgate/feats_gate.npz'); X=fg['X']; n=list(fg['names'])
print('nan lat rows', int(np.isnan(X[:,n.index('lat')]).sum()), 'las rows', int(X[:,n.index('r_las')].sum()))
