# w2_gate — малые результаты гейтов (большие стеки/pkl — в Kaggle elenakonkova/koz3-w2-gate)
- `gate_s2/<tag>/work/w6c_gate_<tag>.json` — итог гейта (score, d, boot.p_gt0, per_class, d_new34) для tag = s2, s012, varb, varb_kur.
- `gate_s2/detail_<tag>.json` — разбор по классам и типам данных (`w2_tools/gate_detail.py`).
- `gate_s2/logs/` — логи build/prep/gate/detail; `base_repro.log` — воспроизведение A=0,314615.
- `gate_s2/run_varb.sh` — образец полного прогона (stack_replace_li → prep.py → gate_v6c.py → gate_detail.py).
- `gate_lihm/early_stop.json` — итог раннего стопа li_hm (P-W2-7) = шаблон для r50c5; для r50c5 есть готовый скрипт `w2_tools/early_stop_r50c5.py`. Как воспроизвести — HANDOFF, «Данные и как ими пользоваться».
