"""Пороги для сборки (bot-4 THR): выбор стратегии по классу вложенно (внутр. фолды = canon-10 на всех участках),
τ подобраны на всех участках. Две версии: на сырых уверенностях (N) и с OOF-оценками реранкера arch canon (RN)."""
import sys, json, numpy as np
sys.argv = [sys.argv[0], '--layouts', 'none']
exec(open('thr_eval.py').read().split('t0 = time.time(); res =')[0])
out = {}
oc = np.load(W / 'work/nested_v2.npz')['outer']
for name, cv in (('N_raw_conf', base_cv), ('RN_arch_rerank_oof', None)):
    if cv is None:
        cv = dict(base_cv); S = conf.copy(); m = cls == ARCH; S[m] = oc[m]; cv['arkhitektury'] = arch_curve(S)
    ch = nested_choice(cv, regions, canon); ST = strat_taus(cv, regions)
    out[name] = {'choice': ch, 'no_lidar_regions_rule': 'region not in notes/las_boxes_59.json',
                 'thresholds': {c: {'strategy': ch[c], 'tau': ST[ch[c]][0].get(c, {}),
                                    'tau_no_lidar': (ST['S2'][1].get(c, {}) if ch[c] == 'S2' else None)} for c in CL}}
    print(name, ch, flush=True)
json.dump(out, open('thresholds_thr.json', 'w'), ensure_ascii=False, indent=1, default=float)
