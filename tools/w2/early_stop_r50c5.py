#!/usr/bin/env python3
"""Ранний стоп P-W2-9 (r50c5) по f0/f1 — без гейта.
usage: python w2_tools/early_stop_r50c5.py <каталог output ядра elenakonkova/koz3-w2-cnx-r50c5-f01> [--materials <путь к materials>]
Нужны в каталоге: oof_cnx_c5-sc_f{0,1}.geojson (или oof_cnx_*_f{0,1}.geojson) и gt_li_cnx_*_f{0,1}.geojson.
Правила (PREREG P-W2-9, базы — c3-sc s2, w2_notes/r50c5_early_stop_baseline.json):
 (1) mean(Li-only final_score f0, f1) r50c5 <= 0.3492 -> FAIL
 (2) F1 курганов на f0 (среднее F1 kurgany_tselye и kurgany_povrezhdennye, compute_metrics per_class)
     r50c5 <= c3-sc s2 f0 (0.2369; povr 0.3144, tselye 0.1593) -> FAIL
Печатает JSON с verdict PASS/FAIL. PASS = ни одно правило не сработало (тогда ждать f234 и гейт)."""
import sys, json, glob, logging, argparse, os
ap = argparse.ArgumentParser(); ap.add_argument('d'); ap.add_argument('--materials', default=os.path.join(os.path.dirname(__file__), '..', 'materials'))
a = ap.parse_args(); sys.path.insert(0, a.materials)
import compute_metrics as CM
logging.getLogger("compute_metrics").setLevel(logging.ERROR)
BASE = json.load(open(os.path.join(os.path.dirname(__file__), '..', 'w2_notes', 'r50c5_early_stop_baseline.json')))
def one(pat):
    g = sorted(glob.glob(os.path.join(a.d, '**', pat), recursive=True)); assert g, f'нет {pat} в {a.d}'; return json.load(open(g[0]))
res = {}
for f in (0, 1):
    r = CM.F1Calculator().calculate_f1(one(f'oof_cnx_*_f{f}.geojson'), one(f'gt_li_cnx_*_f{f}.geojson'))
    pc = r.get('per_class', {})
    res[f] = dict(final=r.get('final_score'), kurgany_povrezhdennye=pc.get('kurgany_povrezhdennye', {}).get('f1', 0.0), kurgany_tselye=pc.get('kurgany_tselye', {}).get('f1', 0.0))
mean01 = (res[0]['final'] + res[1]['final']) / 2
kur0 = (res[0]['kurgany_povrezhdennye'] + res[0]['kurgany_tselye']) / 2
r1 = mean01 <= BASE['rule1_threshold_mean_f0_f1']; r2 = kur0 <= BASE['rule2_c3sc_s2_f0_kurgan_mean_f1']
print(json.dumps(dict(r50c5=res, mean_f0_f1=mean01, kurgan_mean_f1_f0=kur0, rule1_fail=r1, rule2_fail=r2,
                      verdict='FAIL' if (r1 or r2) else 'PASS'), ensure_ascii=False, indent=1))
