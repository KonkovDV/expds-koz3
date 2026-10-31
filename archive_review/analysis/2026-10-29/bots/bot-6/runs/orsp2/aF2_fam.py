# Контроль: база A с полом = пороги A на всех участках (как в сдаче: sp_or 0,3645), без слияния.
"""bot-6 ORSP v2 (c050028) на семьях стека A bot-10 work/fam, по образцу bot-10 work9/orsp_fam.py --floor:
сырые кандидаты (в стеке от 0,05) → post.sp_score_fusion → пол выдачи = новые пороги
[post.sp_score_fusion.thresholds] (как post.sp_fusion_thresholds + apply_thresholds в watchdog.finish).
Пол остальных групп не трогается (как у bot-10: sp_focal и sp_focal2 — только курганы)."""
import json, sys, os
from pathlib import Path
sys.path.insert(0, '/workspace/koz3_shared/bot-6/repo')
from shapely.geometry import shape
from expds import post
assert 'bot-6/repo' in post.__file__
TH = {'sp_focal': {'kurgany_povrezhdennye': 0.364475, 'kurgany_tselye': 0.372525},
      'sp_focal2': {'kurgany_povrezhdennye': 0.501645, 'kurgany_tselye': 0.645505}}
def grp(d):
    g = d.get('threshold_group') or d.get('markup_type'); return 'sp_focal' if g == 'sp_or' else g
src, dst = Path('/workspace/koz3_shared/bot-10/work/fam'), Path(sys.argv[1]); dst.mkdir(parents=True, exist_ok=True); st = {}
for fp in sorted(src.glob('*.jsonl')):
    lines = fp.read_text(encoding='utf-8').splitlines(); feats = []
    for ln in lines:
        f = json.loads(ln); d = dict(f.get('properties') or {}); d['geom'] = shape(f['geometry']); feats.append(d)
    out = feats; assert len(out) == len(feats)
    keep, ch = [], 0
    for i, (x, y) in enumerate(zip(feats, out)):
        t = TH.get(grp(x), {}).get(x.get('class_name'))
        if t is not None and float(y['confidence']) < t: continue
        ln = lines[i]
        if y.get('confidence') != x.get('confidence'):
            f = json.loads(ln); f['properties']['confidence'] = y['confidence']; ln = json.dumps(f, ensure_ascii=False); ch += 1
        keep.append(ln)
    st[fp.stem] = {'n': len(feats), 'changed': ch, 'dropped_below_new_floor': len(feats) - len(keep)}
    if (dst / fp.name).exists() or (dst / fp.name).is_symlink(): (dst / fp.name).unlink()
    if len(keep) == len(lines) and not ch: os.symlink(fp, dst / fp.name)
    else: (dst / fp.name).write_text('\n'.join(keep) + ('\n' if keep else ''), encoding='utf-8')
    print(fp.stem, st[fp.stem], flush=True); del feats, out, lines, keep
(dst / '_orsp_counts.json').write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding='utf-8')
print('total changed', sum(v['changed'] for v in st.values()), 'dropped', sum(v['dropped_below_new_floor'] for v in st.values()))
