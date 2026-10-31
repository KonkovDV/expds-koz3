import pickle, sys, numpy as np
sys.path.insert(0, '/workspace/koz3_shared/bot-10/repo')
from pathlib import Path
A = Path('/workspace/koz3_shared/bot-10/work/curves'); D = Path(sys.argv[1])
bad = []
for p in sorted(D.glob('*.pkl')):
    x = pickle.load(open(p, 'rb')); y = pickle.load(open(A / p.name, 'rb'))
    ok = set(x) == set(y) and all(np.array_equal(np.sort(x[k].conf), np.sort(y[k].conf)) and int(np.sum(x[k].hit)) == int(np.sum(y[k].hit)) for k in x)
    if not ok: bad.append(p.stem)
print(len(list(D.glob('*.pkl'))), 'differ', bad)
