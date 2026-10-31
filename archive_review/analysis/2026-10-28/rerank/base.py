from common import *
t=time.time(); st = load(W/'stack_arch.geojson'); gt = load(W/'gt_train_v2.geojson'); print('load', time.time()-t, len(st['features']), flush=True)
new, optical = sets(gt)
t=time.time(); curves = build_curves(st, gt, METRIC); print('curves', time.time()-t, flush=True)
ins, res = insample(curves); print('insample', ins)
s, counts, info, folds = honest(curves, new, optical); print('honest', s, info)
for c,v in score_counts(counts)[1].items(): print(c, {k: round(x,4) for k,x in v.items()})
pickle.dump({'counts': counts, 'folds': folds, 'scores': s}, open(W/'work/base_honest.pkl','wb'))
