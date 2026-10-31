"""bot-10 work11 C3: post.annotate_scene_drel + post.frame_prior @d4edf2f (код без изменений, audit2_src) по семьям стека.
Рамки сцен — frame_boxes.json (scan_root + map_footprint, как oof_transform.py frame). Spec по умолчанию аудита:
groups [sp_or, sp_focal, sp_focal2], classes все, d0 0,5, edge 0,5.
    python xform_frame.py --src fam0 --dst fam_fr1"""
import argparse, json, sys, os
from pathlib import Path
sys.path.insert(0, '/workspace/koz3_shared/bot-10/work11/audit2_src')
from expds import post
assert 'audit2_src' in post.__file__
from shapely.geometry import shape, box
ap = argparse.ArgumentParser(); ap.add_argument('--src', required=True); ap.add_argument('--dst', required=True)
ap.add_argument('--groups', nargs='+', default=['sp_or', 'sp_focal', 'sp_focal2']); ap.add_argument('--d0', type=float, default=0.5); ap.add_argument('--edge', type=float, default=0.5)
a = ap.parse_args()
B = {r: {mt: [box(*b) for b in bs] for mt, bs in d.items()} for r, d in json.load(open(Path(__file__).with_name('frame_boxes.json'), encoding='utf-8'))['boxes'].items()}
spec = {'groups': a.groups, 'classes': None, 'd0': a.d0, 'edge': a.edge}
src, dst = Path(a.src).resolve(), Path(a.dst); dst.mkdir(parents=True, exist_ok=True); st = {}
for fp in sorted(src.glob('*.jsonl')):
    lines = fp.read_text(encoding='utf-8').splitlines(); feats = []; idx = []
    for i, ln in enumerate(lines):
        f = json.loads(ln); p = dict(f.get('properties') or {})
        if post._group_of(p) in spec['groups'] and (p.get('markup_type') or '') in B.get(p.get('region_name'), {}):
            p['geom'] = shape(f['geometry']); p['confidence'] = float(p.get('confidence', 0.0)); p['_i'] = i
            feats.append(p)
    by_reg = {}
    for p in feats: by_reg.setdefault(p.get('region_name'), []).append(p)
    for r, fs in by_reg.items(): post.annotate_scene_drel(fs, B[r])
    out = post.frame_prior(feats, dict(spec))
    newconf = {o['_i']: o['confidence'] for o, f0 in zip(out, feats) if o['confidence'] != f0['confidence']}
    st[fp.stem] = {'n': len(lines), 'cand': len(feats), 'with_drel': sum(1 for p in feats if 'scene_drel' in p), 'changed': len(newconf)}
    t = dst / fp.name
    if t.exists() or t.is_symlink(): t.unlink()
    if not newconf: os.symlink(fp, t)
    else:
        with open(t, 'w', encoding='utf-8') as fh:
            for i, ln in enumerate(lines):
                if i in newconf:
                    f = json.loads(ln); f['properties']['confidence'] = newconf[i]; ln = json.dumps(f, ensure_ascii=False)
                fh.write(ln + '\n')
    print(fp.stem, st[fp.stem], flush=True)
(dst / '_xform_counts.json').write_text(json.dumps({'spec': spec, 'fam': st}, ensure_ascii=False, indent=1), encoding='utf-8')
print('total', {k: sum(v[k] for v in st.values()) for k in ('n', 'cand', 'with_drel', 'changed')})
