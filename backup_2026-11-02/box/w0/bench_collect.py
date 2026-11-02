"""Сравнение build_curves: официальный путь (EXPDS_FASTGEOM=0) vs Rust fastgeom; холодный кэш (без /dev/shm)."""
import json, os, sys, time
sys.path.insert(0, '/workspace/koz3/w0/pylib'); sys.path.insert(1, '/workspace/koz3/wt/w0')
import numpy as np, shapely
from expds import metric, fastgeom
metric._todo_batch_hash = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('no shm'))
gt = json.load(open('/workspace/koz3/ds/gt/gt_train_v2.geojson'))
pred = json.load(open(sys.argv[1]))
mod = metric.load_official(os.environ['EXPDS_METRIC'])
res = {}
for flag in ('0', '1', '0', '1'):
    os.environ['EXPDS_FASTGEOM'] = flag
    metric._UTM_GEOM_CACHE.clear()
    t = time.time(); cur = metric.build_curves(pred, gt); dt = time.time() - t
    res.setdefault(flag, []).append(dt)
    if flag == '1': cur1 = cur
    else: cur0 = cur
print('fastgeom.enabled', fastgeom.enabled(), 'threads', fastgeom.metric_threads())
for k, v in res.items(): print('FASTGEOM=%s build_curves s:' % k, ['%.2f' % x for x in v])
same = sorted(cur0) == sorted(cur1)
for c in cur0:
    a, b = cur0[c], cur1[c]
    ok = (np.array_equal(a.conf, b.conf) and np.array_equal(a.hit, b.hit) and a.pred_region == b.pred_region
          and a.n_gt == b.n_gt and a.gt_regions == b.gt_regions and a.pred_group == b.pred_group)
    same &= ok
    print(c, len(a.conf), int(np.sum(a.hit)), 'EQUAL' if ok else 'DIFF')
calc = mod.F1Calculator()
wk = {}
for flag in ('0', '1'):
    os.environ['EXPDS_FASTGEOM'] = flag; metric._UTM_GEOM_CACHE.clear()
    t = time.time(); o = metric._collect(mod, calc, pred, False); dt = time.time() - t
    wk[flag] = {c: (shapely.to_wkb([x[0] for x in v]).tolist(), [x[1:] for x in v]) for c, v in o.items()}
    print('_collect pred FASTGEOM=%s %.2fs' % (flag, dt))
print('WKB_IDENTICAL', wk['0'] == wk['1'])
print('CURVES_IDENTICAL', same)
