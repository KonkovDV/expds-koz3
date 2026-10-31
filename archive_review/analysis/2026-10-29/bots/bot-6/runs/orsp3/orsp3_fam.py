"""bot-6 ORSP rework: новый код ветки bot6-orsp (post.sp_score_fusion из bot-6/repo, assert) на семьях стека A
bot-10 (work/fam удалён bot-10 в 08:21; пересоздан тем же split_stack_by_family.py в work11/fam0) БЕЗ полов выдачи (пороги подбираются только вложенно в paired.py по фолдам) → честное число нового кода."""
import json, sys, os, hashlib
from pathlib import Path
sys.path.insert(0, '/workspace/koz3_shared/bot-6/repo')
from shapely.geometry import shape
from expds import post
assert 'bot-6/repo' in post.__file__, post.__file__
src, dst = Path('/workspace/koz3_shared/bot-10/work11/fam0'), Path(sys.argv[1]); dst.mkdir(parents=True, exist_ok=True); st = {}
for fp in sorted(src.glob('*.jsonl')):
    lines = fp.read_text(encoding='utf-8').splitlines(); feats = []
    for ln in lines:
        f = json.loads(ln); d = dict(f.get('properties') or {}); d['geom'] = shape(f['geometry']); feats.append(d)
    out = post.sp_score_fusion(feats, {'radius_m': 8.0}); assert len(out) == len(feats)
    res, ch = [], 0
    for ln, x, y in zip(lines, feats, out):
        if y.get('confidence') != x.get('confidence'):
            f = json.loads(ln); f['properties']['confidence'] = y['confidence']; ln = json.dumps(f, ensure_ascii=False); ch += 1
        res.append(ln)
    st[fp.stem] = {'n': len(feats), 'changed': ch}
    if (dst / fp.name).exists() or (dst / fp.name).is_symlink(): (dst / fp.name).unlink()
    if not ch: os.symlink(fp, dst / fp.name)
    else: (dst / fp.name).write_text('\n'.join(res) + ('\n' if res else ''), encoding='utf-8')
    print(fp.stem, st[fp.stem], flush=True); del feats, out, lines, res
st['_code'] = {'post_file': post.__file__, 'post_sha256': hashlib.sha256(Path(post.__file__).read_bytes()).hexdigest()}
(dst / '_orsp_counts.json').write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding='utf-8')
print('total changed', sum(v['changed'] for k, v in st.items() if not k.startswith('_')))
