"""W2: проверка и склейка OOF ConvNeXt по фолдам → полный OOF на вариант×сид (вход для honest-10 у W4).
Проверки: каждый участок region_fold покрыт ровно своим фолдом, threshold_group == li_cnx, детекции только на участках своего фолда.
Выход: <oof>/merged/oof_cnx_<variant>_<seed>.geojson (+ .json сводка). Ансамбль сидов НЕ делается (только после PREREG P-W2-2)."""
import json, sys, glob, collections
from pathlib import Path
root = Path(sys.argv[1]); out = root / 'merged'; out.mkdir(exist_ok=True)
REF = None; summary = {}
for seed_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith('s')):
    per = collections.defaultdict(dict)   # variant -> fold -> path
    rf_all = {}
    for f in glob.glob(str(seed_dir / '**/oof_cnx_*_f[0-9].geojson'), recursive=True):
        name = Path(f).stem; var, fold = name[len('oof_cnx_'):].rsplit('_f', 1)
        per[var][int(fold)] = f
        for r in glob.glob(str(Path(f).parent / 'region_fold.json')):
            rf = json.load(open(r, encoding='utf-8'))['region_fold']
            if REF is None: REF = rf
            assert rf == REF, f'region_fold расходится: {r}'
    for var, folds in sorted(per.items()):
        feats, errs, nreg = [], [], collections.Counter()
        for fold, f in sorted(folds.items()):
            g = json.load(open(f, encoding='utf-8'))
            for ft in g['features']:
                p = ft['properties']; reg = p.get('region') or p.get('site') or p.get('region_name')
                if p.get('threshold_group') != 'li_cnx': errs.append(f'tg {p.get("threshold_group")} f{fold}')
                if REF and reg is not None and REF.get(reg) != fold: errs.append(f'{reg} не в фолде {fold}')
                nreg[fold] += 1; feats.append(ft)
            crs = g.get('crs')
        missing = sorted(set(range(5)) - set(folds))
        key = f'{var}_{seed_dir.name}'
        summary[key] = dict(folds=sorted(folds), missing=missing, n_det=dict(nreg), errors=sorted(set(errs))[:20], complete=not missing and not errs)
        if not missing and not errs:
            json.dump(dict(type='FeatureCollection', crs=crs, features=feats), open(out / f'oof_cnx_{key}.geojson', 'w', encoding='utf-8'), ensure_ascii=False)
        print(key, summary[key] if errs or missing else f'COMPLETE 5/5, n={len(feats)}')
json.dump(summary, open(out / 'merge_summary.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
