import sys, time, os, json
sys.path.insert(0, '/workspace/koz3/w0/pylib'); sys.path.insert(1, '/workspace/koz3/wt/w0')
from expds import metric, fastjson
from expds.naming import site_family
metric._todo_batch_hash = lambda *a, **k: (_ for _ in ()).throw(RuntimeError())  # холодный кэш
T = time.time; t0 = T()
t = T(); gt = fastjson.load_json('/workspace/koz3/ds/gt/gt_train_v2.geojson'); pred = fastjson.load_json(sys.argv[1]); tl = T() - t
t = T(); cur = metric.build_curves(pred, gt); tb = T() - t
t = T(); thr = metric.sweep_groups(cur)
regs = set(); [regs.update(c.gt_regions) or regs.update(c.pred_region) for c in cur.values()]
folds = metric.family_folds(sorted(regs), site_family, 10); hc = metric.honest_counts(cur, folds); th = T() - t
print(json.dumps({'mode': os.environ.get('W0_MODE'), 'load': round(tl, 2), 'build_curves': round(tb, 2),
                  'sweep+honest': round(th, 2), 'total': round(T() - t0, 2), 'thr': repr(thr)[:200], 'hc': repr(hc)[:300]}))
