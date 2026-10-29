exec(open('thr_v2.py').read().split("res = {}")[0])
t = {('etgr_pix','object_poly'):0.0,('maggr_pix','finds_points'):0.93802,('maggr_pix','object_poly'):0.68606,('terfl_pix','object_poly'):0.0}
c='object_poly'; cv=CV[c]
print(cv.n_gt, sum(cv.gt_regions.values()), len(cv.conf), cv.hit.dtype, cv.hit.sum())
print('fast', class_f1_fast(c,t), f1c(kept_counts(t)[c]))
xs,f=scan('maggr_pix',c,t); j=np.searchsorted(xs,0.68606,side='right')-1; print(xs[j], f[j], f.max())
