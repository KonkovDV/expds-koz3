"""bot-10 (29.10): ORSP bot-6 на семьях стека A: post.sp_score_fusion из bot-6/repo @315dc91 (radius_m 8) на кандидатах
семьи ДО порогов (как в watchdog.finish 315dc91: only_known_classes → sp_score_fusion → apply_thresholds).
Семьи без изменённых уверенностей — симлинк на исходник.  python orsp_fam.py --src ../work/fam --dst fam_orsp"""
import argparse, json, sys, os
from pathlib import Path
sys.path.insert(0, '/workspace/koz3_shared/bot-6/repo')
from shapely.geometry import shape
from expds import post
assert 'bot-6/repo' in post.__file__, post.__file__
ap = argparse.ArgumentParser(); ap.add_argument('--src', required=True); ap.add_argument('--dst', required=True)
ap.add_argument('--floor', action='store_true', help='сначала выкинуть курганы sp_focal < 0,15 и sp_focal2 < пола выдачи (pipeline.output_floors)')
ap.add_argument('--nofuse', action='store_true'); a = ap.parse_args()
FL = {'sp_focal': {'kurgany_povrezhdennye': 0.15, 'kurgany_tselye': 0.15}, 'sp_focal2': {'kurgany_povrezhdennye': 0.501645, 'kurgany_tselye': 0.645505}}
def below(d):
    g = d.get('threshold_group') or d.get('markup_type'); t = FL.get(g, {}).get(d.get('class_name'))
    return t is not None and float(d.get('confidence', 0)) < t
src, dst = Path(a.src).resolve(), Path(a.dst); dst.mkdir(parents=True, exist_ok=True); st = {}
for fp in sorted(src.glob('*.jsonl')):
    lines = fp.read_text(encoding='utf-8').splitlines()
    feats = []
    for ln in lines:
        f = json.loads(ln); d = dict(f.get('properties') or {}); d['geom'] = shape(f['geometry']); feats.append(d)
    nb = 0
    if a.floor:
        keep = [i for i, d in enumerate(feats) if not below(d)]; nb = len(feats) - len(keep)
        feats = [feats[i] for i in keep]; lines = [lines[i] for i in keep]
    out = feats if a.nofuse else post.sp_score_fusion(feats, {'radius_m': 8.0})
    assert len(out) == len(feats)
    ch = [i for i, (x, y) in enumerate(zip(feats, out)) if x.get('confidence') != y.get('confidence')]
    st[fp.stem] = {'n': len(feats), 'changed': len(ch), 'dropped_below_floor': nb}
    if (dst / fp.name).exists() or (dst / fp.name).is_symlink(): (dst / fp.name).unlink()
    if not ch and not nb:
        os.symlink(fp, dst / fp.name)
    else:
        with open(dst / fp.name, 'w', encoding='utf-8') as fh:
            for i, ln in enumerate(lines):
                if out[i].get("confidence") != feats[i].get("confidence"):
                    f = json.loads(ln); f['properties']['confidence'] = out[i]['confidence']; ln = json.dumps(f, ensure_ascii=False)
                fh.write(ln + '\n')
    print(fp.stem, st[fp.stem], flush=True)
    del feats, out, lines
(dst / '_orsp_counts.json').write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding='utf-8')
print('total changed', sum(v['changed'] for v in st.values()))
