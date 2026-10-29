"""bot-10 work11 C1: post.site_calibration аудита @d4edf2f (код без изменений, git archive → audit2_src) по семьям стека.
    python xform_calib.py --src ../work/fam --dst fam_cal1 --groups li sp_or sp_focal [--mode mix]
Логика = oof_transform.py cmd_calib: spec groups/classes(None→курганы)/mode/alpha 0.5/q_level 0.99/floor 0.05/min_n 30."""
import argparse, json, sys, os
from pathlib import Path
sys.path.insert(0, '/workspace/koz3_shared/bot-10/work11/audit2_src')
from expds import post
assert 'audit2_src' in post.__file__, post.__file__
ap = argparse.ArgumentParser(); ap.add_argument('--src', required=True); ap.add_argument('--dst', required=True)
ap.add_argument('--groups', nargs='+', required=True); ap.add_argument('--mode', default='mix')
a = ap.parse_args()
spec = {'groups': a.groups, 'classes': None, 'mode': a.mode, 'alpha': 0.5, 'q_level': 0.99, 'floor': 0.05, 'min_n': 30}
src, dst = Path(a.src).resolve(), Path(a.dst); dst.mkdir(parents=True, exist_ok=True); st = {}
for fp in sorted(src.glob('*.jsonl')):
    lines = fp.read_text(encoding='utf-8').splitlines()
    feats = []
    for i, ln in enumerate(lines):
        p = dict(json.loads(ln).get('properties') or {}); p['confidence'] = float(p.get('confidence', 1.0)); p['_i'] = i; feats.append(p)
    out = post.site_calibration(feats, dict(spec))
    newconf = {o['_i']: o['confidence'] for o in out if 'conf_raw' in o and o['confidence'] != o['conf_raw']}
    st[fp.stem] = {'n': len(feats), 'calibrated': sum(1 for o in out if 'conf_raw' in o), 'changed': len(newconf)}
    if (dst / fp.name).exists() or (dst / fp.name).is_symlink(): (dst / fp.name).unlink()
    if not newconf:
        os.symlink(fp, dst / fp.name)
    else:
        with open(dst / fp.name, 'w', encoding='utf-8') as fh:
            for i, ln in enumerate(lines):
                if i in newconf:
                    f = json.loads(ln); f['properties']['confidence'] = newconf[i]; ln = json.dumps(f, ensure_ascii=False)
                fh.write(ln + '\n')
    print(fp.stem, st[fp.stem], flush=True)
(dst / '_xform_counts.json').write_text(json.dumps({'spec': spec, 'fam': st}, ensure_ascii=False, indent=1), encoding='utf-8')
print('total', {k: sum(v[k] for v in st.values()) for k in ('n', 'calibrated', 'changed')})
