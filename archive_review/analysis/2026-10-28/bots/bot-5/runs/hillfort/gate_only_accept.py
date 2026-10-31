"""Пост-хок абляция (после просмотра gor_scene.json): только gate (conf sp_or-городищ × p сцены), без новых строк.
Тот же протокол П1-П5; вывод помечен post_hoc=True."""
src = open(__file__.replace('gate_only_accept.py', 'gor_scene.py'), encoding='utf-8').read()
exec(src.split("VARS = [")[0])
per = gor_counts(folds10, None, True); vf = dict(base_full); vf[G] = per
D = {'post_hoc': True, 'delta10': score_counts(vf)[0] - A,
     'delta_new34': score_counts(vf, new)[0] - score_counts(base_full, new)[0],
     'delta_optical': score_counts(vf, optical)[0] - score_counts(base_full, optical)[0],
     'bootstrap': paired_bootstrap_counts(vf, base_full, site_family, n=1000, seed=0), 'gor': gsc(per)}
log(D); json.dump(D, open(OUT / 'gate_only_accept.json', 'w'), ensure_ascii=False, indent=1, default=float)
