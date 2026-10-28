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

## 2026-10-28 22:05 МСК — ehimen (находки MagGr по рецепту EN)
- Отчёт: bot-3/ehimen/EHIMEN_REPORT.md, вердикт: EHIMEN_VERDICT.md, diff: bot-3/ehimen/ehimen_port.diff (ветка bot3-ehimen, 88cc077, без push).
- «0,72» EN = leave-one-region-out. У нашего порта под LORO F1 находок 0,688 (+0,0135), под leave-family-out Δ итога от −0,0026 до +0,0037, ни один вариант не проходит П2 (work/eval_*.out). Лучший — 0,05 м/px: +0,0037, P 0,625.
- Аудит 01 (A3): +0,0060, П1–П4 ✔ — лучше порта. В сдачу порт не брать.
- D8-аугментации в train.py:246–252/:332–336 (flips=True для maggr, tiles.py:635) ломают диполь N−S, но только у сети maggr, а её в сдаче нет.
- census_finds.zip — посылка-перепись на LB (только на ПК), на метки не влияет; не отправлялась.

## 2026-10-28 22:20 MSK — GEO_GAP_REPORT.md готов (разделы A–E)
- Лучший пакет: A2 (finds, maps=all без склейки) + R3 (pole_circle_m=0.05): honest-10 global drop +0,004037 / keep +0,006048, site +0,004077 / +0,006117; П1–П4 ✓, P(Δ>0) 0,869 (geo_gap/eval_pkg.out).
- Якоря по рецепту EhimenNathan (H1): keep +0,0021, drop −0,0002, П2 ✗ (geo_gap/eval_anchor.out). layer_copy R4 −0,016, split R2 −0,0029 (geo_gap/r/eval_r34.out, eval_r2.out). R1 — OOM на 211.
- Фильтр EtGr > 2 м: 0 из 76 tif на трейне.

## 2026-10-28 23:05 МСК — PIXGBM готов
- Отчёт: bot-3/PIXGBM_REPORT.md. Ветка bot3-pixgbm (b1ea251, без push), diff: bot-3/pixgbm/pixgbm.diff.
- S (pix заменяет классику): honest-10 +0,0110 / +0,0119 / +0,0099 / +0,0141, P 0,95, П1–П4 ✔ во всех 4 режимах, Δ ≥ MDE (pixgbm/honest_lofo.json). Выигрыш в основном от находок (Sf +0,0091…+0,0098). U и AU: П2 ✘.
- Адресно (site keep, S): object_poly A 37/153 (Гн 10), D 91/406 (Гн 87), +482 FP; finds A 255/624, B 365/803, C 261/436; Гнёздово finds ΔTP −1 (pixgbm/targeted_lofo.json).
- OOF для bot-6: pixgbm/oof/ (21 файл, READY_<участок>, READY_ALL, README.md).
- 200 и 212 цвет_2 измерены. Пик RSS 1,0 ГБ.

## 28.10 ~23:35 МСК — H и бюджет PIXGBM
- Вариант H (S вне Гнёздово, U в Гнёздово) **отклонён**: H vs stack в режимах drop не проходит П2 (P = 0.737 / 0.827); в drop H уступает S. Подробности — `PIXGBM_REPORT.md` («Дополнение … вариант H») и `pixgbm/honest_lofo_H.json`. Код: ветка bot3-pixgbm, коммит 7cdace5, `union_name_re`; `pixgbm.diff` обновлён.
- Бюджет: `pixgbm/BUDGET_PIXGBM.md`. [изм] 7 регионов: 420 с при 1 потоке, 275 с при 2, пик 706 МБ, predict = 73 %. [оц] Площадка при 2 потоках: +258–431 с.
