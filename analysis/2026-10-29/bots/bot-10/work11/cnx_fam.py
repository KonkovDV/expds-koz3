"""bot-10 work11 C2: стеки с ConvNeXt c3-sc (бот 9, фолды 0–3) для honest-10.
  R  — на участках фолдов 0–3 курганы группы li (threshold_group или markup_type = li, без своей группы) убраны, добавлены курганы li_cnx;
  X  — добавлены ВСЕ объекты li_cnx (повтор бота 9: база + cnx);
  Сабсет S27 делается потом симлинками кривых юнитов (cnx_subset.py).
Прочие семьи — симлинки на work11/fam0. Запуск: cnx_fam.py R | X"""
import json, os, sys, collections
from pathlib import Path
sys.path.insert(0, '/workspace/koz3_shared/bot-10/repo')
from expds.naming import site_family
W = Path('/workspace/koz3_shared/bot-10'); SRC = W / 'work11/fam0'; OUT = W / 'work11'
RF = json.load(open('/workspace/koz3_shared/bot-9/kaggle/convnext_lidar/c3sc_f01_out/region_fold.json', encoding='utf-8'))['region_fold']
COV = {r for r, f in RF.items() if f in (0, 1, 2, 3)}
KUR = {'kurgany_povrezhdennye', 'kurgany_tselye'}
cnx = collections.defaultdict(list)
for p in ('/workspace/kaggle_out/koz3-cnx-li-c3sc-f01/oof_cnx_c3-sc.geojson', '/workspace/kaggle_out/koz3-cnx-li-c3sc-f23/oof_cnx_c3-sc.geojson'):
    for f in json.load(open(p, encoding='utf-8'))['features']:
        r = f['properties']['region_name']; assert r in COV, r
        assert f['properties'].get('threshold_group') == 'li_cnx'
        cnx[site_family(r)].append(f)
fams_cov = {site_family(r) for r in COV}
st = {}
VV = sys.argv[1:] or ['R', 'X']
for v in VV:
    d = OUT / f'fam_cnx{v}'; d.mkdir(exist_ok=True)
    for fp in sorted(SRC.glob('*.jsonl')):
        fam = fp.stem; tgt = d / fp.name
        if tgt.exists() or tgt.is_symlink(): tgt.unlink()
        if v.endswith('_sub') and fam not in fams_cov: continue
        if v == 'A_sub' or fam not in fams_cov:
            os.symlink(fp, tgt); continue
        nrm = nadd = 0
        with open(tgt, 'w', encoding='utf-8') as fh:
            for ln in open(fp, encoding='utf-8'):
                if v.startswith('R'):
                    p = json.loads(ln)['properties']
                    g = p.get('threshold_group') or p.get('markup_type')
                    if p.get('class_name') in KUR and g == 'li' and p.get('region_name') in COV:
                        nrm += 1; continue
                fh.write(ln if ln.endswith('\n') else ln + '\n')
            for f in cnx.get(fam, []):
                if v.startswith('R') and f['properties']['class_name'] not in KUR: continue
                fh.write(json.dumps(f, ensure_ascii=False) + '\n'); nadd += 1
        st[f'{v}/{fam}'] = {'removed_li_kur': nrm, 'added_cnx': nadd}
json.dump({'covered_sites': sorted(COV), 'families_covered': sorted(fams_cov), 'per_fam': st}, open(OUT / f'cnx_fam_counts_{"_".join(VV)}.json', 'w'), ensure_ascii=False, indent=1)
print('covered sites', len(COV), 'families', len(fams_cov))
for k, x in st.items():
    if True: print(k, x)
