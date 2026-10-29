# bot-6 STATUS — канал оптики / аэрофото 1940-х
- 19:2x МСК: прочитан COMMON.md; копия кода: bot-6/repo, ветка bot6-optics (от 24bd916).
- Источник «0,09 / 0,013»: docs/LB_LOG.md — справочный блок by_data_type «аэрофото» 0,090909 / 0,013333 (мин. целый набор 75 эталонов, 1 TP, 20 FP — scripts/lb_counts.py).
- Что выдаёт ae в сдаче: configs/default.toml [models] ae = ae_fort_mask_r18.pt (только fortifikatsii) + копии курганов лидара li_to_ae (группы ae / li_ae). Своей сети курганов на аэрофото нет.
- Растров в /workspace/data/train-expds-3 пока нет (только geojson разметки и UTM.json) — проверки растров невозможны; делаю проверку сдвига слоёв Ae vs Li по эталону.
- Сделано: work/ae_audit.json (курганы ae 1620, достижимо копией li 592, 545 на участках без li), work/layer_shift_gt*.json (сдвиг li→ae: устойчив у 4 из 11, 608→631 из 1276).
- Патч: ветка bot6-optics (5618d13), bot-6/bot6-optics.diff — scripts/ae_layer_audit.py. Итог: REPORT.md. ГОТОВО.
- 20:5x МСК: начаты проверки растров. Пришло 7 растров ae (001,005,012,015,017,019,020; zstd, xfer_state.json), ортофото нет. Скрипт work/raster/meta.py написан, прогон прерван — ПАУЗА по запросу бота «1» (аудит типов данных → TYPES.md). Продолжить с meta.py.
- 20:5x МСК: TYPES.md (частично, п.1–3) записан; дальше — размеры полигонов vs GT, потеря классов/типов, затем растры.
- 21:2x МСК: twins перепроверен честно (A=0,314615 воспроизведён). F +0,00285 [0,00061; 0,00581], П1–П4 да, П5 пакет. TYPES.md §5. Порядок дальше: COPY → LOCALIZE. copy_model.py запущен 21:23.
- 21:4x МСК: по запросу бота «1»: сверка с bot-7, аудит twins-only, ветка bot6-twins-only (ad8ae9b) + bot6-twins-only.diff, точный замер после порогов (work/twins/twins_exact.*) → TYPES.md §6. COPY: copy_model.py готов (work/copy/copy_model.log), COPY.md — следующим.
- 21:5x МСК: COPY.md (закрыто ботом «1»: модель −0,0020). LOCALIZE: localize.py запущен 21:47.
22:29 МСК: LOCALIZE.md готов. Точный пост-порог: лучший сверх bot-7 — fort 0,5 м +0,00026 (П1 нет, пакет); gor −0,0015; привязка Li/DEM отрицательна. Ветка bot6-localize 6b58d99, bot6-localize.diff.
22:4x МСК: (a) TYPES.md §7: потерь полей, классов и геометрии нет. Целиком недостижимые группы — каждая < 0,005 (arch ae 0,0040; сумма ≈ 0,0074). 211 (0 кандидатов): потолок 0,0106, реально ≈ 0,0007. Гнёздово etgr/terfl — слепая зона честного CV (unseen → drop), а не потеря сдачи. Файлы: work/lost/{geomcheck,lost,cand}.*.
22:4x МСК: (b) ae_layer_audit.py исправлен (cb17611: кириллица «Ае» + пустые фигуры), тест 1 passed, bot6-optics.diff 182 строки. REPORT.md §3/§4 на v3 (курганы ae 1749, достижимо li 595 = 34,0 %; сдвиг 611 → 634 из 1322) + таблица v1/v2/v3.
22:4x МСК: (c) REPORT.md §5: 36 снимков Ae, 32 участка (work/raster/ae_raster_check.*, 366 МБ). 2 снимка в 3857 (015, 029); 7 в int16 (у 106 растяжка ×3,04); у 017/036 чёрные поля не nodata; эталон ae на валидных пикселях 2170/2179. ВСЕ ТРИ ПУНКТА ГОТОВЫ.
22:56 МСК: п.1 GEO.md: честно (LOSO nested) +0,0190 к honest-10 drop [CI +0,0068; +0,0358] = keep-all +0,0186; порог 0,096635 бесполезен; pixgbm +0,0007…+0,0028 к стеку н.з. Далее п.2 (211), п.3 (растры).
23:00 МСК: bot6-twins-only 78add0e: починка масштаба twins; регресс-тест (старый код 0/20); на объектах bot-10 ae_twin 85, sp 113 (work/twinsfix/check.log). bot6-geo 67f84bd: [post] geo_convex_hull, bot6-geo.diff. EXPERIMENTS.md: запись о честной базе 0,333245.
23:3x МСК: GEO.md переписан (§0/§3 без круга: стек nested 0,332604, hull 0,332532; §4 pixgbm по одному t_pix, 12 строк, work/geo/compare_tpix.log: S/U t0,3–0,5 +0,0046…+0,0075 к keep-all, лучшая U t0,5 +0,0075 [+0,0040; +0,0110]; M не помогает; §6 приёмка оболочки для bot-10). Старое — work/geo/old_mixed/.
23:3x МСК: п.2 211 — REPORT.md раздел «211» (bound211: +0,0013 нечестно). п.3 ae-патч: bot6-optics 62145d6 ([raster_fix] reproject_3857/int16_identity/zero_nodata, выкл.), тест 4 passed, регресса нет; bot6-optics.diff 483 строки; real_check: +≈5 с CPU, 029 окон 72→56; эффект на трейне ∈ [−0,00009; +0,00005] (bound.json). ВСЕ ПУНКТЫ ГОТОВЫ.
23:47 МСК: or/sp ПАУЗА (по указанию бота 1). Диагностика work/orsp/res_diag.json; PREREG_ORSP.md 23:37; SPF замерен (res_spf.json): canon +0,00336 P 0,995, 5/5 семян >0, LOFO +0,00353 → П1–П4 ✓, по П5 только пакет. Фоновых процессов нет. Осталось: ORSP.md, ветка bot6-orsp + diff, site-scope, ПК. Итог — REPORT.md «Ортофото/спутник: текущее состояние (пауза)».
- 29.10 ORSP закрыт: ORSP.md; ветка bot6-orsp 315dc91, флаг [post] sp_score_fusion (выкл.), bot6-orsp.diff; в пределах участка Δ +0,00466 (P 0,998). Стоп.

## 29.10 01:5x МСК — сводка по запросу бота «1» (ПАУЗА до утра по квоте, новых задач не беру)
- Гнёздово etgr/terfl — ГОТОВО (GEO.md). keep-all ≈ сдача (честная база 0,333245, запись в EXPERIMENTS.md). Оболочка: bot6-geo 67f84bd, против keep-all −0,0007 н.з. → у бота 10. pixgbm бота 3 (U, t_pix 0,5, etgr/terfl): +0,0075 к keep-all (лучшая из 12, честно ≈ +0,005) → решение по PREREG бота 10; пресет и порт инференса делает бот 3. Мешает: время инференса pixgbm не замерено. Следующий шаг: за ботами 3/10.
- 211 — ГОТОВО (REPORT.md «211»): серая MagGr постеризована; maps=all +0,0013 (не честно). Команды на ПК поставлены ботом 1 в очередь. Следующий шаг: перемер на ПК.
- Растры ae — ГОТОВО: bot6-optics 62145d6, [raster_fix] выкл.; эффект на трейне ∈ [−0,00009; +0,00005], ≈ +5 с CPU. Держим выключенным (решение бота 1).
- ORSP — НЕ ПРИНЯТ ботом 10 (bot-10/REPORT9_v2.md, (C)): с порогами сдачи −0,0017, 0/5 (sp_focal2 почти весь ниже выходного порога). Следующий шаг УТРОМ: исправить по (C) и отдать на перемер — ночью не запускаю (указание бота 1).
- 29.10 08:02 МСК: ORSP v2 по bot-10 §C: bot6-orsp c050028 (сырые кандидаты, партнёры 12 м, алиас sp_or, пороги sp), bot6-orsp.diff пересобран; честно +0,002523 (P 0,984, 5/5), v1 −0,00169 воспроизведено; orspR +0,0055 оптимистично (all-data пол). Только в пакете. Стоп.

## 08:35 ORGATE ПРИПАРКОВАН (приоритет ORSP по указанию bot 1 08:24)
- Готово: k10 (Rrepro,RULE,G,G_noGeo,G_noRG), seed0–seed4 (G,G_noRG) — work/orgate/res/{k10,seed0..4}.log, res_*.json.
- Прерван: lofo (запущен 08:32, убит 08:35, частичных результатов нет). Возобновить: `cd work/orgate && LAYOUTS=lofo VARS=G,G_noRG ./run.sh`.
- Не начато: --scope site, Gext (эмбеддинги bot-4), ORGATE.md.

## 09:10 ORSP v3 (переделка по REPORT10 §3) — готово
- Коммит abae911 (bot6-orsp): пороги сдачи = медиана per-fold k10 (sp_focal povr 0,2665015; tselye 0,3696; sp_focal2 0,514733/0,701621); алиас оставлен (обоснован в ORSP.md).
- Новый код без полов: +0,002523 (P 0,984), LOFO +0,003163, 5/5 — work/orsp3/orsp3_global.{log,json}; кривые побайтно = v1 bot-10. Прежнее «честное v2» было кэшем v1 — исправлено в ORSP.md.
- Δ(v2−v1)=0: стек уже сырой от 0,05, sp_focal2 после отбора 12 м, группа подписана sp_focal (stackcheck.json). Проверка — только end-to-end на ПК (список в ORSP.md).
- Далее: возобновляю ORGATE (lofo).

## 10:1x 29.10 ORGATE — ГОТОВО (ORGATE.md)
- PREREG_ORGATE.md 08:06:47 < первый результат G res/k10.log 08:15:25 (оговорка: правило LAS из разведки на всех данных; G_noLAS ≥ G — правило не нужно).
- G против A: k10 +0,011470 (P 0,985), LOFO +0,011394, site k10 +0,011537, LOFO site +0,011779, 5/5 семян. Против R (принятый rerank_arch bot-10): k10 +0,001933 (P_R 0,993), LOFO +0,003525, семена 5/5 (+0,0014…+0,0062) → доп. правило PREREG выполнено; по П5 — только пакет (замена R).
- Or (k10): A 5/646 → R 16/148 → G 19/165; прирост G над R — из sp_arch (216→162 FP).
- Gext (эмбеддинги bot-4, PCA в фолде): k10 +0,012810, dR +0,003274; не входит в бюджет (≈4 683 с).
- Фоновых процессов bot-6 нет. Команда для bot-10 — ORGATE.md §6.

## 11:59 29.10 ORGATE_S — ГОТОВО (НЕ ПРИНЯТО), ORGATE.md обновлён
- **ORGATE.md §7:** по вердикту bot-10 (REPORT12_orgate.md) рекомендация изменена на **G с правилом LAS**. Условия: r_las и правило LAS на тесте считать как в трейне; гейт переобучать при смене стека кандидатов arch (D-FINE/FR1h).
- **ORGATE_S.md:** Gext_S_rule (G + PCA16 эмбеддингов DINOv2-S/14-reg) против G.
  - Δ: k10 +0,002035 (P 0,819); LOFO −0,000310 / −0,000392 (site); семена 4/5 > 0 (среднее +0,00094 global, +0,00085 site).
  - П2 ✗ и П3 ✗ → **не принято**, остаётся G.
- **Время S:** 41 мс/кроп всё включено → 7105 × 0,041 ≈ 291 с ≤ 561 с, укладывается. Сеть отдельно: 12,4–31,4 мс/кроп.
- **Веса и данные:** официальные Apache-2.0, sha256 78e7f952…abae7. Использованы только данные бокса, правило организаторов от 29.10 соблюдено (PREREG_ORGATE_S.md, аддендум 2).

## 12:5x 29.10 AEOR_DETECT — ГОТОВО (ветка bot6-aeor-detect @ 3617be6, AEOR_DETECT.md)
- Флаг `[dfine_aeor] enabled=false` стоит между sp_dfine и gor_gate. Группы dfine_ae/dfine_or, кэш общий с sp_dfine, импорт только через sp_dfine.import_dfine, маркер DFINE_AEOR_UNAVAILABLE.
- tests/test_dfine_aeor.py: 29 passed (work/aeor_pytest_final.log).
- Побайтовая идентичность (база / флаг выкл / флаг вкл без весов): одинаковый sha 9222f5d1…, 245 объектов (work/aeor_identity/run2/identity.json).
- Diff: bot6-aeor-detect.diff. Ждём от bot-9 веса и время на 4090, от bot-10 OOF-пороги. Фоновых процессов нет.

## 13:1x 29.10 AEOR v2 — ГОТОВО (bot6-aeor-detect @ f4e733b, не запушен; AEOR_DETECT.md §v2)
- Ключ слияния изолирует dfine_ae/dfine_or (merge_union выкл.), у dfine отдельная цепочка finish. Scene с обзорами + общий буфер батча.
- RSS: Герите Or 605 308 КБ / 4,9 с; Елизаветинское 749 392 КБ / 90,7 с; Фалхан 791 660 КБ / 89,5 с (work/aeor_rss/*.log).
- Риск по времени не снят: Or без обзоров ≈ 89 с при потолке фазы 120 с.
- check_submission: AEOR_UNAVAILABLE → FAIL. Смоук n_det>0 на 181_Гнёздово_Лесная_3 (Ae) и 185_Изборск (Or); без весов → FAIL.
- Тесты: test_dfine_aeor 38, вместе с флагами/check_submission 82, регресс 139 passed. Identity sha 9222f5d1… (work/aeor_identity/run4).
- Diff bot6-aeor-detect.diff (sha a45f5e4d…). Весов bot-9 (bot-9/models) на момент проверки не было.

## 13:1x 29.10 ORGATE G в инференсе — ГОТОВО (bot6-orgate @ e0a6709, не запушен; ORGATE_INFER.md)
- Флаг `[orgate] enabled=false`; гейт заменяет R на его месте (до FR1h и порогов). dfine_* и sp_dfine не оцениваются. LAS берётся из входного каталога.
- Отказ даёт маркер ORGATE_UNAVAILABLE и откат на R, check_submission → FAIL. Смоук на 008_Герите: удаляет 11 из 43, опорный вход вне git (п. 13.11).
- Модель 676f4c1b… (532 514 Б), meta d3980320…; пороги or 0,004086 / sp_arch 0,008176 (медиана k10); REPRO_OK 54/327.
- Эквивалентность: весь трейн — 0 расхождений из 134 участков, maxrss 891 888 КБ; путь пайплайна на 4 участках точный. Оговорка: решения зависят от состава входа (gq/пул), как у R.
- Тесты 13 + регресс 190 passed; identity off / on-без-модели / on = база 9222f5d1… (work/orgate_identity/run1).
- Diff bot6-orgate.diff, sha 1d30c8fb…. Фоновых процессов нет.

## 13:4x 29.10 follow-ups bot-7 13:12 — ГОТОВО (bot6-orgate @ ca63a9d, родитель e0a6709, не запушен)
- check_submission: если опорных данных смоука нет (`--orgate-smoke` / растры dfine_aeor / `--no-weights`), выдаётся WARN «смоук не выполнен», а не PASS и не FAIL. Дефекты архива по-прежнему дают FAIL.
- dfine_aeor: срок проверяется перед каждым чтением окна и каждым масштабом, перерасход ≤ 1 чтения окна; skip через soft_ok подтверждён тестом. В AEOR_DETECT.md v3 — заметка о перекалибровке под GTX 1650.
- ORGATE_INFER.md (g): orgate_g обучен на table.pkl **до** D-FINE R. R выбрасывает 46 735 из 47 518 arch-строк (все sp_arch) и 205 из 228 OK-пар (`work/orgate_stack/stack_check.json`). Совместного замера G+R по arch в REPORT14/12/10 нет, нужен перемер.
- Тесты: 56 (42 aeor + 14 orgate), регресс 195 passed (`work/orgate_regress3.log`); identity run2 4/4 b5f6b41c… (bot-7 venv; bot-8 venv нет после рестора).
- Diff bot6-followups-1312.diff (e0a6709..ca63a9d), sha 2a75a4de…; bot6-orgate.diff без изменений (1d30c8fb…). Фоновых процессов нет.
- 14:06 МСК 29.10: v4 — веса dfine_or bot-9 (sha OK), смоук 185 PASS (n_det 94, 2,8–3,2 с/окно, RSS ≈1,1 ГБ; batch 4 под 2 ГБ — OOM, нужен 1), пороги-заглушка все 1,01; check_submission с весами PASS после фикса TypeError (770abda / c67c385). CRS-фикс 008_Герите: ветка bot6-crsfix от 885758e — f86a31a (+ c67c385), bot6-crsfix.diff, bot6-v4-checksub.diff; 008 1×1/0 дет. → 2070×1883/9 окон/397 дет.; по 210 участкам меняется только 008. AEOR_DETECT.md §v4. Не запушено.
