"""bot-7 task7: LightGBM OOF по фолдам honest-10 стека (семья участка не видна модели). Отдельно povr и tselye.
Метка = y>0 (официальное попадание круга 2 м в эталон класса того же участка). Выход oof.npz: p_povr, p_tsel."""
import numpy as np, lightgbm as lgb, json, os
from sklearn.metrics import roc_auc_score, average_precision_score
Z=np.load('cands_all.npz',allow_pickle=True); X=Z['X']; fold=Z['fold']; FN=list(Z['FN'])
P=dict(objective='binary',learning_rate=0.05,num_leaves=31,min_data_in_leaf=50,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,
       lambda_l2=1.0,num_threads=2,verbose=-1,seed=0)
NR=int(os.environ.get('NR','300'))
out={}; rep={}
for nm in ('povr','tsel'):
    y=(Z['y_'+nm]>0).astype(int); p=np.full(len(y),np.nan)
    for k in np.unique(fold):
        te=fold==k; tr=~te
        if y[tr].sum()<5: p[te]=0; continue
        b=lgb.train(P,lgb.Dataset(X[tr],y[tr],feature_name=FN),NR); p[te]=b.predict(X[te])
    out['p_'+nm]=p
    base=-np.log(Z['Dv'])
    rep[nm]=dict(n=int(len(y)),pos=int(y.sum()),auc_oof=float(roc_auc_score(y,p)),ap_oof=float(average_precision_score(y,p)),
                 auc_rawrank=float(roc_auc_score(y,base)),ap_rawrank=float(average_precision_score(y,base)))
    b=lgb.train(P,lgb.Dataset(X,y,feature_name=FN),NR); imp=b.feature_importance('gain')
    rep[nm]['top_gain']=[FN[i] for i in np.argsort(-imp)[:12]]
    b.save_model(f'lgb_{nm}_alldata.txt')
    print(nm,rep[nm],flush=True)
np.savez('oof.npz',**out); json.dump(rep,open('train_oof.json','w'),ensure_ascii=False,indent=1)
