import numpy as np, json, csv, sys
D='/workspace/koz3/w5data/work/'; X=np.load(D+'X3.npy'); nm=json.load(open(D+'X3_names.json'))
rows=list(csv.reader(open(sys.argv[1]))); hdr=rows[0]; F=np.full((X.shape[0],len(hdr)-1),np.nan,np.float32)
for r in rows[1:]: F[int(r[0])]=[float(v) if v else np.nan for v in r[1:]]
print('filled', int(np.isfinite(F[:,0]).sum()), 'of', X.shape[0])
np.save('w5d/work/X4.npy', np.hstack([X, F]).astype(X.dtype)); json.dump(nm+hdr[1:], open('w5d/work/X4_names.json','w'))
