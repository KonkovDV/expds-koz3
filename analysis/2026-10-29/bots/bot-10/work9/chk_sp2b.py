"""bot-10: доля sp_focal ниже 0,15 и участки sp_focal2 с LAS (las_boxes_59.json) в стеке оценки work/fam."""
import json
from pathlib import Path
from collections import Counter
las = set(json.load(open('/workspace/expds-koz3/notes/las_boxes_59.json', encoding='utf-8')))
st = Counter(); r2 = set()
for fp in sorted(Path('/workspace/koz3_shared/bot-10/work/fam').glob('*.jsonl')):
    if Path('fam_orsp', fp.name).is_symlink(): continue
    for ln in open(fp, encoding='utf-8'):
        if '"sp_focal' not in ln: continue
        p = json.loads(ln)['properties']; g = p.get('threshold_group') or p.get('markup_type')
        if p['class_name'] not in ('kurgany_povrezhdennye', 'kurgany_tselye'): continue
        if g == 'sp_focal': st['sp'] += 1; st['sp_below_0.15'] += float(p['confidence']) < 0.15
        if g == 'sp_focal2': r2.add(p['region_name']); st['sp2_on_las_region'] += p['region_name'] in las
print(dict(st), 'sp2 regions', len(r2), 'with LAS', len(r2 & las))
