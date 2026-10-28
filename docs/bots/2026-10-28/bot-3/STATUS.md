# bot-3 STATUS (geophys)
- 2026-10-28 19:22:02 МСК: COMMON.md прочитан; клон repo/ ветка bot3-geophys от 24bd916.
- Данные: в /workspace/data/train-expds-3 пока только *.geojson/UTM.json (555 geojson, 210 json), растров геофизики НЕТ (find -type f | sed ext | uniq -c).

## 2026-10-28 19:37 MSK
- GT-статистика: gt_stats_geophys.json (sim/gt_stats.py). Главное: у Гнёздова 59 % находок имеют соседа ближе 2 м → NMS радиусом 2 м (peak_radius_m) сохраняет ≤66 % при случайном порядке силы.
- Нашёл в repo уже готовый post.multipole_object_poly (коммит 01cc769), заявленные выигрыши в notes/sota_stack6_2026-10-28.json — константы там подобраны на всём трейне, проверяю.
- Патч в repo (ветка bot3-geophys): GeoParams.split/split_h/pole_circle_m, geophys.split_merged (h-максимумы + watershed_ift, только scipy), label_poles, split_spec_params; geo_tune.py --splits/--pole-circle. tests/test_geo_split.py 8/8, соседние geo-тесты 32/32.
- Идёт симуляция sim/sim_poly.py (sigma 0.25/0.5/1.0 × шум 0/0.15), лог sim_run.log.

## 2026-10-28 20:23 MSK — ГОТОВО
- Все симуляции завершены (sim_run.log: ALLDONE). Сводки: sim_poly_summary.json, sim_analyze_out.txt, sim_poly_summary_patchcode.json.
- Код патча, σ0,5 n0,15: object_poly F1 0,7477 → 0,784 (LOFO). multipole_post 0,7008 (хуже базы).
- REPORT.md написан. Растров геофизики на боксе по-прежнему нет (только LiDAR/аэро tif): оценка — симуляция из GT, на ПК нужна проверка geo_tune + scoreboard --honest 10.

## 2026-10-28 21:30 MSK — audit01 ИЗМЕРЕН, object_poly — в работе
- Ветка bot3-audit01 (worktree bot-3/repo_audit01, c315a9b), diff bot-3/audit01.diff.
- A2 (maps=all, dedupe_maps=false): honest-10 +0,0039 (drop/keep, global/site), П1–П4 ✓, П5 — только пакетом. Детали: GEO_GAP_REPORT.md раздел A, geo_gap/a01/eval_all.out.
- band_mode в [geophys.maggr] игнорируется (A3m == A2 побайтно); нужен [geophys.maggr.finds_points].
- 200 и 212(auto) не измерены (OOM при 2 ГБ) — команды в отчёте.
- Разбор FN object_poly: geo_gap/gap_poly.json; геометрия стека G1–G3: geo_gap/eval_geom.out (≤ +0,0014, не проходят П1/П2).
- Идут растровые R1–R3 (split dt:0.5 / pole_circle 0.05) на 201–217: geo_gap/run_r.sh, 1 поток.
