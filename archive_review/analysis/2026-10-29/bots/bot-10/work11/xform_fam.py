"""bot-10 work11: пофайловое применение oof_transform.py gormask|neigh (аудит @6e26062, код без изменений) к семьям стека.
Логика = cmd_gormask / cmd_neigh из wt_audit/scripts/oof_transform.py (аргументы по умолчанию), только вход — jsonl семьи.
    python xform_fam.py gormask --src ../work/fam --dst fam_gm --taus taus_insample_A.json
    python xform_fam.py neigh   --src ../work/fam --dst fam_nb --taus taus_insample_A.json [--radius 200]"""
import argparse, json, sys, os
from pathlib import Path
sys.path.insert(0, '/workspace/koz3_shared/bot-10/work11/wt_audit')
from shapely.geometry import shape
from expds import post
assert 'wt_audit' in post.__file__, post.__file__
ap = argparse.ArgumentParser(); ap.add_argument('mode'); ap.add_argument('--src', required=True); ap.add_argument('--dst', required=True)
ap.add_argument('--taus', required=True); ap.add_argument('--radius', type=float, default=200.0)
a = ap.parse_args()
KUR = {'kurgany_povrezhdennye', 'kurgany_tselye'}
TH = json.loads(Path(a.taus).read_text(encoding='utf-8')).get('thresholds', {})
src, dst = Path(a.src).resolve(), Path(a.dst); dst.mkdir(parents=True, exist_ok=True); st = {}
for fp in sorted(src.glob('*.jsonl')):
    lines = fp.read_text(encoding='utf-8').splitlines()
    feats = []
    for i, ln in enumerate(lines):
        f = json.loads(ln); p = dict(f.get('properties') or {})
        if p.get('class_name') not in KUR and not (a.mode == 'gormask' and p.get('class_name') == 'gorodishcha'):
            continue
        feats.append({**p, 'geom': shape(f['geometry']), 'confidence': float(p.get('confidence', 1.0)), '_i': i})
    drop, newconf = set(), {}
    if a.mode == 'gormask':
        kur = [f for f in feats if f['class_name'] in KUR]
        gor = [f for f in feats if f['class_name'] == 'gorodishcha']
        gor = [f for f in gor if f['confidence'] >= TH.get(f.get('threshold_group') or f.get('markup_type') or '', {}).get('gorodishcha', float('inf'))]
        spec = {'min_gor_conf': 0.0, 'buffer_m': 0.0, 'groups': None, 'gor_groups': None, 'min_area_m2': 500.0, 'max_area_m2': 400_000.0}
        kept = post.kurgans_outside_hillforts(kur + gor, spec)
        keep_i = {f['_i'] for f in kept}
        drop = {f['_i'] for f in kur if f['_i'] not in keep_i}
        st[fp.stem] = {'kur': len(kur), 'gor_masks': len(gor), 'removed': len(drop)}
    else:
        tau = post.threshold_fn({}, float('inf'), TH)
        spec = {'groups': ['li', 'li_coarse', 'li_mosaic'], 'count_groups': None, 'radius_m': a.radius, 'k': 3.0,
                'floor': 0.5, 'slope': 0.5, 'cap': 2.0, 'support_conf': None}
        out = post.kurgan_neighbour_prior(feats, spec, tau)
        assert len(out) == len(feats)
        for f, o in zip(feats, out):
            if o['confidence'] != f['confidence']: newconf[f['_i']] = o['confidence']
        st[fp.stem] = {'kur': len(feats), 'changed': len(newconf)}
    if (dst / fp.name).exists() or (dst / fp.name).is_symlink(): (dst / fp.name).unlink()
    if not drop and not newconf:
        os.symlink(fp, dst / fp.name)
    else:
        with open(dst / fp.name, 'w', encoding='utf-8') as fh:
            for i, ln in enumerate(lines):
                if i in drop: continue
                if i in newconf:
                    f = json.loads(ln); f['properties']['confidence'] = newconf[i]; ln = json.dumps(f, ensure_ascii=False)
                fh.write(ln + '\n')
    print(fp.stem, st[fp.stem], flush=True)
    del feats, lines
(dst / '_xform_counts.json').write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding='utf-8')
print('total', {k: sum(v.get(k, 0) for v in st.values()) for k in ('kur', 'gor_masks', 'removed', 'changed')})
