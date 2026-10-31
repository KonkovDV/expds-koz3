import json,sys
for t in sys.argv[1:]:
  d=json.load(open('cstar/R_%s.json'%t))
  for c in d['C']:
    ks={k:c[k] for k in c if k not in('summary','info','d_f1','d_folds_global','file') and not isinstance(c[k],(dict,list))}
    print(t,c['file'].replace('C_CetO3_Sr8G_noTerFl_L_',''),'dG %.4f'%c['d_global'],'finds %.4f'%c['summary']['f1']['finds_points'],ks)
