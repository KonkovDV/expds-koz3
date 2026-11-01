# w2_notes — итоговые JSON по замерам W2
| Файл | Что |
|---|---|
| `w2_c3sc_s2_gate.json` | гейт A+c3-sc s2 и s012avg (P-W2-2), FAIL |
| `w2_c3sc_varb_gate.json` | вариант b (P-W2-6), FAIL |
| `w2_lihm_gate.json` | ранний стоп li_hm (P-W2-7), FAIL; шаблон для r50c5 |
| `w2_ci23_ci28_watch.json` | итоги аудита земли (CI2-3/P-W2-8) и hillshade parity (CI2-8) |
| `w2_hsparity_ci28_rows.json` | parity по 59 LAS: MAE vs Li, лучшие az/alt/z/blur |
| `r50c5_early_stop_baseline.json` | базы раннего стопа r50c5 (c3-sc s2: f0 0,2198 / f1 0,4786; курганы f0 povr 0,3144, tselye 0,1593, среднее 0,2369), посчитано 30.10 |
| `box_misc/` | мелочи с бокса: `t4_center_snap_stage1.json` (итог P-W2-4 center snap, нетто −6), `maps_pull.log` (список ядер, откуда тянулись maps), логи проб рельефа `prep_probe_*`, `probe182_*` (разбор 182_Красносельская: 0 детекций) |
| `lidar_aug/lidar_aug_f34.json` | закрытие линии lidar-aug (c7-sc, kernel elenakonkova/koz3-lidar-aug-f0 v5 f3/f4 + f2 dmitrijkv): Li-only F1 @0.3 против c3-sc s2 тех же фолдов, f2 0,343/0,401, f3 0,313/0,361, f4 0,308/0,319 → CLOSE. Скрипт `lidar_aug/eval_lidar_aug_f34.py` (venv /workspace/koz3/venv, данные /workspace/koz3/out/lidar-aug-f0_v5 и out/*c3sc*; копии OOF c3-sc в Kaggle koz3-w2-oof). 01.11 |
