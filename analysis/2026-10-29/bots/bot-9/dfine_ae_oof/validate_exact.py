# Проверка: официальный _soft_match на всех пред. участка == результат analyse_ae_oof.py
import json, sys, numpy as np, collections
_regs = sys.argv[1:]; sys.argv=[sys.argv[0]]
exec(open('analyse_ae_oof.py').read().split("# --- кандидатные пары")[0])
A = json.load(open('ae_oof_analysis.json'))
for r in (_regs or ['036_Тимерево', '181_Гнёздово_Лесная_3', '103_Гнёздово_Ольшанская', '015_Демидовка']):
    ri = REG.index(r); idx = np.nonzero(reg == ri)[0]; P = []
    with open(SRC, 'rb') as f:
        for p in idx:
            f.seek(int(off[p])); bt = f.read(int(ln[p])).rstrip()
            bt = bt[:-2].rstrip() if bt.endswith(b']}') else bt
            bt = bt[:-1] if bt.endswith(b',') else bt
            ft = json.loads(bt); gg = cm._extract_geometry(ft)
            if gg is not None: P.append((ft['properties']['class_name'], float(ft['properties']['confidence']), utm(gg)))
    for t in (0.05, 0.3, 0.5):
        mine = A['per_region'][f'{r}@{t}']; ok = True
        for c in CLS:
            G = [(g['geom'], 1.0) for g in gt if g['region'] == r and g['cls'] == c and g['geom'] is not None]
            PP = [(g, s) for cc, s, g in P if cc == c and s >= t]
            tp, fp, fn, _ = calc._soft_match(PP, G)
            m = mine.get(c, dict(tp=0, fp=0, fn=0))
            if (tp, fp, fn) != (m['tp'], m['fp'], m['fn']): ok = False; print('DIFF', r, t, c, (tp, fp, fn), m)
        print(r, t, 'OK' if ok else 'MISMATCH', len(P), flush=True)
