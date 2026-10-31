"""bot-9: пороги групп *_pix на всём трейне (вариант S, pix_lofo), тем же M.sweep_groups, что в eval_pix.kept_mask, но по всем участкам."""
import os, sys, json
W='/workspace/koz3_shared/bot-3/pixgbm'; os.environ.setdefault('PART','none'); os.environ['PAIRS']='S:stack'
sys.argv=['eval_pix.py']
src=open(f'{W}/eval_pix.py').read().split('# ───────── (1) адресная')[0]
exec(src)
out={}
every=set().union(*FOLDS)
for vn in ['S','stack']:
  for scope in ['global','site']:
    cv=curves_of(VARS[vn], scope)
    for c in CL:
        res=M.sweep_groups({c: M.curve_where(cv[c], every)})
        out[f'{vn}|{scope}|{c}']={g: gv['threshold'] for g, gv in res[c]['groups'].items()}
        print(vn, scope, c, json.dumps(out[f'{vn}|{scope}|{c}'], ensure_ascii=False), flush=True)
json.dump(out, open(f'{W}/pc/thr_pix_full.json','w'), ensure_ascii=False, indent=1, default=float)
print('DONE')
