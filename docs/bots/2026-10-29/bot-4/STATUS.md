# STATUS bot-4 (28.10.2026, MSK)
- [x] COMMON.md прочитан; клон bot-4/repo (24bd916), venv bot-4/venv
- [x] база воспроизведена (work/res_base.json): честно 0,05041 (gor 0,2762, fort 0,1796)
- [x] путаница и потери: work/res_confusion.json, res_pool.json, res_ceiling.json
- [x] арбитр fort2gor + грубый проход по стеку: expds/gorfort.py, замеры work/res_gorfort_nms_s0.4.json — честный Δ<0, не включать
- [x] грубый проход по ЦМР: hillfort.COARSE / --preset coarse + тесты; замер только на ПК (нет LAS)
- [x] ветка bot4-gorfort (ab8ad26), diff bot-4/bot4-gorfort.diff, REPORT.md
- [ ] сообщить bot «1» путь к REPORT.md — у меня нет инструмента сообщений, передано родителю

## 28.10 20:54 МСК — GAP (срочно от бота 1)
- [x] GAP.md: класс×тип, FN/FP-потери, без лидара, Geo, IoU/размер, потолки (1а/1б/2/3) — `work/gap/gap.py`, `work/gap/res_gap.json`
- [ ] Перепроверка/бутстреп потолков (по желанию бота 1)
- [ ] DEM грубый проход на 21 LAZ-участке — на паузе, не запускался
- [ ] Уведомить бота «1» (через родителя)

## 28.10 20:58 МСК — DEM грубый проход ПРИОСТАНОВЛЕН (по команде бота 1)
- Готово 25 участков с LAS: `work/dem/out/*.geojson` (кандидаты gor_coarse_dem), лог `work/dem/run.log`. Не обработаны (крупные LAZ): 004, 007, 028 и всё, что докачается позже.
- Возобновление: `nohup work/dem/run.sh >> work/dem/run.log &` (пропускает готовые), затем `venv/bin/python work/dem/dem_eval.py` (оценка написана, не запускалась).
- Процессов DEM не осталось.

## 28.10 22:22 МСК — пороги класс×группа поверх реранкера (R) + OFF-пары
- [x] THR.md (21:31): A/R/N/RN/P1/P2 на 5 раскладках, `work/thr/res_thr_seed0_..._seed4.json`, `log_seeds.txt`. RN−R по семенам: +0,00067 / −0,00392 / −0,00100 / +0,00221 / −0,00276 → вложенные пороги поверх R не стабильны.
- [~] thr_off.py (OFF-пары, бот 1): идёт (pid 272296, nice 10). Готово canon и seed0 → `work/thr/res_off_canon_seed0_..._seed4.json` (частично), `log_off.txt`.
  - canon: R 0,324359; R_OFFn +0,00002; R_EXp −0,02863; R_HT −0,00469; R_OrBroad −0,00255.
  - seed0: R_OFFn +0,00125; R_HT −0,00229; R_EXp −0,02491.
  - Предварительно: фикс. выключения как у EXpedition/huntertomo вредят; вложенный OFFn ≈0.
- [~] LOFO (thr_eval.py --layouts lofo, pid 184326): 80+ семей за 69 мин.
- Блокеров нет. ETA: OFF seed1–4 ≈ 45 мин (~23:10 МСК), затем обновление THR.md; потом POST (п. 3–5 из COMPETITORS_DEEP.md).

## 28.10 23:25 МСК — THR/OFF/LOFO досчитаны, THR.md обновлён
- [x] LOFO (pid 184326) завершился штатно в 22:38 ("done 5316 s"), `work/thr/res_thr_lofo.json`: R +0,00863, RN−R +0,00160, P2 +0,00816, N +0,00222.
- [x] OFF canon+seed0..4 завершились в 23:06, `work/thr/res_off_canon_seed0_..._seed4.json`.
- Вердикт: ship только R. RN−R, P2 (=E3p бота 2), N, OFFn, EXp, HT, OrBroad — отклонены (см. THR.md «РЕЗУЛЬТАТЫ»).
- [ ] Новая база: пакет бота 7 без twins — разбираюсь с воспроизведением на table.pkl.
- [ ] POST поверх новой базы (или R как запасной).

## 28.10 23:28 МСК — новая база: пакет bot-7 (без twins) на table.pkl
- Воспроизведение на коробке — ЧАСТИЧНОЕ (без crop): crop verify (`stack_verify_sp_arch.geojson`) есть только на F: ПК (bot-9/PC_RESULTS.md п. 3c), на коробке файла нет (`find /workspace -name 'stack_verify*'` пусто). Поэтому база здесь = stack_arch + курганы 0,5 м (bot-7/work2/fam_kur) + stack6 separate, без twins, без crop (= «честный замер на коробке» bot-7 REPORT3 §3, +0,016428).
- Сборка: `work/pkg/s1_stack6.py` (post.multipole_object_poly + filter_finds_points_regimes из bot-7/repo bot7-candidate; счётчики групп совпали с bot-7/work4/fam_pkg.log: finds|maggr 3732→2928, object_poly|etgr 403→721, maggr 146→0, maggr_poly 0→247, terfl 319→95) → `work/pkg/s2_table.py` → `work/pkg/work/table.pkl` (1 390 700 строк; строки 0..N-1 = исходная table.pkl, курганы с геометрией 0,5 м и пересчитанными парами, старые geo-строки выключены, новые дописаны). Пик RSS шага 2 — 2,25 ГБ (чуть выше лимита 2 ГБ, ~2,5 мин), дальше считаю в ≤1,2 ГБ на процесс.
- Запущено (nice 10, 1 поток каждый): thr_eval canon на пакете (pid 407972, лог `work/pkg/log_thr_canon.txt`, выход `work/pkg/thr/`), post_eval canon на пакете (pid 407973, `work/pkg/log_post_canon.txt`, выход `work/pkg/post/`). ETA ~20–30 мин.

## 23:36 МСК — пакет bot-7 на коробке: воспроизведён (кроме crop); THR и POST canon на пакете
- База пакета (коробка, без twins/crop, honest-10 наш протокол): PKG_A 0.322689 vs A 0.314615 → Δ +0.00807, бутстреп P 0.993 [p5 +0.00059; p95 +0.02028] (work/pkg/cmp_base_canon.json). PKG_R 0.332433 (= R-база 0.324359 + тот же Δ).
  Почему меньше чем у bot-7 (+0.0164 коробка / +0.0204 ПК): курганы и finds совпадают с bot-7 до tp (povr 943, finds F1 0.6058); расхождение только object_poly — у bot-7 протокол honest_bootstrap «unseen keep» (база obj 261/607), у нас honest_counts (78/68). Crop verify нет на коробке (stack_verify_sp_arch.geojson только на F: ПК).
- THR на пакете (work/pkg/thr/res_thr_canon.json): R +0.00974 (класс arch не затронут пакетом → seeds/LOFO R те же), N +0.00218, RN−R +0.00147, P1 +0.00144, P2 +0.00498. Вердикт тот же: только R.
- POST canon на пакете+R (work/pkg/post/res_post_canon.json): NB +0.00089 (P .79), IN −0.00501 (P .054), DUALli +0.00133 (P .966), DUALsp −0.00149, DUAL −0.00014, PKG +0.00009.
- Проверка «2% внутри / 45% вне» для (4): НЕ подтверждается. Точность курганов, прошедших порог, внутри наших предсказанных городищ: povr 12/60 = 20%, tselye 9/58 = 16%; вне: povr 931/3628 = 26%, tselye 686/1828 = 38%. GT: внутри GT-городищ лишь 6 из 5197 GT-курганов.
- Запущено: post_eval seeds0-4+lofo на пакете (pid 414466, log work/pkg/log_post_seeds_lofo.txt, ETA ~23:42).

## 23:42 МСК — POST seeds на пакете готовы; утечка памяти в LOFO исправлена
- POST seeds0-4 (Δ к PKG_R; work/pkg/post/res_post_seed0_seed1_seed2_seed3_seed4_lofo.json, log work/pkg/log_post_seeds_lofo.txt):
  NB +.00313/+.00222/+.00040/+.00123/+.00103 (5/5); IN −.00012/+.00020/−.00185/−.00088/−.00186 (1/5); DUALli +.00225/+.00243/+.00138/+.00288/+.00180 (5/5);
  DUALsp все <0; DUAL +.00195/+.00023/−.00022/+.00142/−.00087; PKG +.00519/+.00410/−.00068/+.00311/+.00029.
- LOFO в том прогоне убит мной на старте: RSS 5.4 ГБ (утечка — кэш кодов регионов в post_eval держал каждую временную кривую). Исправлено (кэш только для постоянных кривых), перезапуск LOFO: RSS 1.16 ГБ стабильно.
- Правила для (5): лимит 20000 — на ВЕСЬ result.geojson (expds/writer.py:108, constants.MAX_FEATURES; compute_metrics MAX_FEATURES_PER_FILE=20000). Запрета на объект двух классов в одном месте в writer/метрике нет (сопоставление по классам). Объектов на всём трейне (сумма по held-out, canon): R 9986, DUALli 10261 (+275), DUAL 10375, PKG 10277 — далеко от 20000.
- Запущено (последовательно, по одному процессу ради ≤2 ГБ): post LOFO pid 439934 (ETA ~23:48, log work/pkg/log_post_lofo.txt) → thr_off canon на пакете (цепочка pid 444124, log work/pkg/log_off_canon.txt) → POST NB+DUALli (post-hoc пакет, цепочка pid 476146, log work/pkg/log_post_nbdli.txt).

## 23:45 МСК — ОСТАНОВ по команде bot 1 (квота). Все фоновые задачи убиты.
Убиты: post LOFO pid 439934 (дошёл до семьи 70 из ~105, результат НЕ сохранён); цепочки pid 444124 (thr_off canon на пакете — не стартовал) и pid 476146 (POST NB+DUALli — не стартовал). Процессов post_eval/thr_off нет.
ГОТОВО:
- THR/OFF/LOFO на базе R — THR.md §«РЕЗУЛЬТАТЫ (23:25)»: в прод только R (canon +0.00974, P 0.966, seeds 5/5, LOFO +0.00863); N, RN−R, P1, P2, OFFn, EXp, HT, OrBroad — отклонены.
- Пакет bot-7 на коробке (без crop/twins): Δ +0.00807, P 0.993 к базе (work/pkg/cmp_base_canon.json); курганы/finds = bot-7, разница только object_poly из-за протокола (unseen keep). THR на пакете: work/pkg/thr/res_thr_canon.json (R +0.00974, N +0.00218, RN−R +0.00147, P1 +0.00144, P2 +0.00498) — вердикт тот же.
- POST на пакете+R, canon+seeds: work/pkg/post/res_post_canon.json, res_post_seed0_seed1_seed2_seed3_seed4_lofo.json (только seeds, lofo там нет). Предварительно: (3) NB +0.00089 P .79 seeds 5/5 — П1/П2 не проходит; (4) IN −0.00501 P .054 — отклонить, «2%/45%» не подтверждено (внутри 20%/16%, вне 26%/38%); (5) DUALli +0.00133 P .966 seeds 5/5, но П1 и П4 new34 (−0.00219) не проходят; DUALsp −0.00149, DUAL −0.00014, PKG(3+4+5) +0.00009 P .53 — отклонить. Лимит 20000 не угрожает (≈10.3k на весь трейн).
НЕ СДЕЛАНО: POST LOFO, thr_off на пакете, POST NB+DUALli (post-hoc), POST.md не написан, раздел THR-на-пакете в THR.md.
ВОЗОБНОВЛЕНИЕ (из /workspace/koz3_shared/bot-4/work, по одному процессу, ≤2 ГБ):
  P=$PWD/pkg
  KOZ_W=$P KOZ_OUT=$P/post OMP_NUM_THREADS=1 nohup nice -n 10 ../venv/bin/python post/post_eval.py --layouts lofo > $P/log_post_lofo.txt 2>&1 &   # ~11 мин
  KOZ_W=$P KOZ_OUT=$P/thr OMP_NUM_THREADS=1 nohup nice -n 10 ../venv/bin/python thr/thr_off.py --layouts canon > $P/log_off_canon.txt 2>&1 &
  POST_VARS=NBDli KOZ_W=$P KOZ_OUT=$P/post OMP_NUM_THREADS=1 nohup nice -n 10 ../venv/bin/python post/post_eval.py --layouts canon,seed0,seed1,seed2,seed3,seed4,lofo > $P/log_post_nbdli.txt 2>&1 &
  затем написать POST.md по res_post_*.json.

## 29.10 01:58 МСК — ПАУЗА ДО УТРА (по указанию пользователя через бота 1, 23:44: квота, новые задачи не запускать)
- Пороги/OFF: итог в THR.md «РЕЗУЛЬТАТЫ (23:25)» — принимается только R; N, RN, P1, P2, OFFn, EXp, HT, OrBroad отклонены.
- POST (поверх пакет+R, canon+семена): всё отклонено пока; LOFO прерван — см. секцию 23:45.
- Процессов bot-4 нет. Блокер: пауза пользователя. Следующий шаг после снятия паузы: POST LOFO, thr_off на пакете, POST.md (команды — секция 23:45). Новые задачи из GAP/AUDIT не беру до явного снятия паузы.

## 07:36 МСК 29.10 — пауза снята; задача (1) ревью check_submission готова
- REVIEW_CHECK_SUBMISSION.md. Вердикт: для 428257f OK; формат, CRS, классы, лимит и время не тронуты. При PIXGBM enabled=true два ослабления: папка моделей без проверки poly.txt и finds.txt; ленивые импорты без проверки фолбэка. Тесты: work/review/cs_test/test_cs.py (T3, T4, T8).
- Далее: (2) RED_TEAM v2 eb268d6, затем (3) GAP.

## 29.10 07:27 МСК — задачи (2) RED_TEAM v2 eb268d6 и (3) GAP §6 готовы
- (2) `RED_TEAM_v2_eb268d6.md`: **GO условный, блокеров 0, предупреждений 6.**
  - Оригинальный check: FAIL 3 — все ложные под PIXGBM (model_dir-каталог, psutil/tl2cgen с фолбэком, maps-список); format PASS, 230 Polygon.
  - Время [оц]: 360–630 с; на объёме mock_val ≈1200–1270 с, защиты 1500/1620 держат 1800.
  - Лицензии: 9 U-Net resnet18 с ImageNet-инициализацией (теперь разрешено), sp_crop без предобучения, LightGBM свои.
  - Утечек и секретов нет; в docstring'ах примеры путей D:/, C:/, в архиве недостижимые модули gt/blobs/coarse/det_boxes.
  - Лог: `work/review/check_orig_eb268d6_synth.log`.
- (3) `GAP.md` §6: приоритеты бэкбонов (оценки): ранжирование → arch Or → povr Ae/Or → gor Or → object_poly Geo (последний).
- Фоновых процессов нет (pid 764380 завершился). `cs_tmp` удалён.
