"""bot-10 work11 C2b: стеки ConvNeXt по произвольному OOF и набору фолдов (обобщение cnx_fam.py).
  python cnx_fam2.py <имя> <R|X> <folds через запятую> <oof.geojson> [<oof2> ...]
  R — на участках этих фолдов курганы группы li (threshold_group или markup_type = li) убраны, добавлены курганы li_cnx;
  X — добавлены все объекты li_cnx. Прочие семьи — симлинки на work11/fam0. Пишет work11/fam_<имя>/ и cnx_fam_counts_<имя>.json"""
import json, os, sys, collections
from pathlib import Path
sys.path.insert(0, '/workspace/koz3_shared/bot-10/repo')
from expds.naming import site_family
W = Path('/workspace/koz3_shared/bot-10'); SRC = W / 'work11/fam0'; OUT = W / 'work11'
name, v, folds, srcs = sys.argv[1], sys.argv[2], {int(x) for x in sys.argv[3].split(',')}, sys.argv[4:]
RF = json.load(open('/workspace/koz3_shared/bot-9/kaggle/convnext_lidar/c3sc_f01_out/region_fold.json', encoding='utf-8'))['region_fold']
COV = {r for r, f in RF.items() if f in folds}
KUR = {'kurgany_povrezhdennye', 'kurgany_tselye'}
cnx = collections.defaultdict(list)
for p in srcs:
    fs = json.load(open(p, encoding='utf-8'))['features']; assert fs, p
    for f in fs:
        r = f['properties']['region_name']; assert r in COV, r
        assert f['properties'].get('threshold_group') == 'li_cnx'
        cnx[site_family(r)].append(f)
fams_cov = {site_family(r) for r in COV}
d = OUT / f'fam_{name}'; d.mkdir(exist_ok=True); st = {}
for fp in sorted(SRC.glob('*.jsonl')):
    fam = fp.stem; tgt = d / fp.name
    if tgt.exists() or tgt.is_symlink(): tgt.unlink()
    if fam not in fams_cov:
        os.symlink(fp, tgt); continue
    nrm = nadd = 0
    with open(tgt, 'w', encoding='utf-8') as fh:
        for ln in open(fp, encoding='utf-8'):
            if v == 'R':
                p = json.loads(ln)['properties']
                g = p.get('threshold_group') or p.get('markup_type')
                if p.get('class_name') in KUR and g == 'li' and p.get('region_name') in COV:
                    nrm += 1; continue
            fh.write(ln if ln.endswith('\n') else ln + '\n')
        for f in cnx.get(fam, []):
            if v == 'R' and f['properties']['class_name'] not in KUR: continue
            fh.write(json.dumps(f, ensure_ascii=False) + '\n'); nadd += 1
    st[fam] = {'removed_li_kur': nrm, 'added_cnx': nadd}
tot = {k: sum(x[k] for x in st.values()) for k in ('removed_li_kur', 'added_cnx')}
json.dump({'variant': v, 'folds': sorted(folds), 'sources': srcs, 'covered_sites': sorted(COV), 'families_covered': sorted(fams_cov), 'per_fam': st, 'total': tot},
          open(OUT / f'cnx_fam_counts_{name}.json', 'w'), ensure_ascii=False, indent=1)
print(name, 'sites', len(COV), 'families', len(fams_cov), tot)
