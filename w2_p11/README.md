# P-W2-11: аудит s1 f4 (OOF / карты li_cnx c3-sc)
Итог: OOF и Kaggle-карты s1 f4 **корректны**. Расхождение P-W2-10 (190: 0,666 vs 0,515) — тест считал «s1f4» веса **s2f4**
(coord-ядро s2f4 переиспользовало `id_no` ядра s1f4 → версия 2 того же ядра; `kernels output` отдал последнюю версию).
Настоящие s1f4 = `kaggle kernels output elenakonkova/koz3-w2-cnx-c3sc-s2f4/1` (weights md5 efbff0f3…).
- `m2_provenance.py/json` — M2: seed/n_det/md5 всех 15 единиц.
- `run_m1.sh`, `run_m3.sh` — CPU-инференс (`w2/demfill/infer_cpu.py`), фолд 4 целиком + споты s1 f0–f3, s0/s2 f4.
- `m1_compare.py` → `result_P-W2-11_m1.json`: MAD карт (≤1e-5), n_det, Li F1 по участкам (OOF / декод Kaggle-карт / CPU).
- `oof_cnx_c3-sc_s012avg_p11.geojson.json` — пересобранный s012avg (s1 f4 из CPU-карт), отличия ±1 детекция на 7 участках.
- `run_gate_p11.sh`, `gate/` — honest-10: 0,307574 (A 0,314615, Δ −0,00704, FAIL; против прежнего s012 Δ +0,00001).
  `gate/UNREG_replace_mode_*` — ошибочный прогон в режиме replace, вне PREREG.
Подробно: `w2_notes/p11_s1f4_audit.json`, PREREG P-W2-11 (раздел «Результат»).
