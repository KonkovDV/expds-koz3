# STATUS bot-10
- 2026-10-28 19:25 МСК: начато; репозиторий склонирован в bot-10/repo (HEAD 24bd916).
- 19:35: прочитаны scoreboard.py, expds/metric.py (honest_counts, paired_bootstrap_counts), docs/LB_LOG.md, PLAN.md, notes/*.json.
- Найден дамп стека /workspace/rerank/stack_arch.geojson (1,39 млн объектов, стек сданного архива arch, база 0,367363 / честно 0,314615 по /workspace/rerank/stack_arch_base.json и surf_arch_board_2026-10-27.json). План: потоково разрезать по семьям, посчитать кривые по семьям (≤2 ГБ ОЗУ, 1 поток), получить посемейные честные счётчики и бутстреп.
- 19:55: стек разрезан по 103 семьям (work/fam, 26 с, пик ОЗУ 725 МБ), кривые по семьям (work/curves, пик 471 МБ). Найдены 2 кластера семей с пересечением рамок (work/family_clusters.json) — сопоставляются вместе. Честный балл 10 фолдов по семьям воспроизведён точно: 0.3146153226828423 = /workspace/rerank/surf_arch_board_2026-10-27.json honest.after[0]. Строятся варианты без gor_phys и без архитектуры (sp_arch, or).
- 19:47: готов noise_measured.json (556 с). Парная SE ≈0,003, MDE ≈0,006 (1-ст.) / 0,008 (2-ст.); SD уровня по бутстрепу семей 0,027, по раскладкам фолдов 0,0107; модель площадки (48 участков): SD уровня 0,056, SD Δ 0,004–0,007. ae_fort честно по семьям +0,00524 (P 0,926).
- 19:50: коммит a839da2 (ветка bot10-honest-bootstrap, без push), diff honest_bootstrap.diff; tests/test_metric_exact.py 5 passed.
- 19:52: записаны /workspace/koz3_shared/EXPERIMENTS.md, REVIEW.md (отчёты на 19:50: bot-2, 4, 5, 6, 9; нет: bot-3, 7, 8), REPORT.md. Готово. Сообщений никому не отправлял (по указанию пользователя).
- 20:14: задача 2 (keep/drop бота 7): артефакт честного табло. etgr/terfl object_poly только в Гнёздово, в сдаче уже есть. keep−drop +0,018622 (P 0,64), парные Δ не меняются. Ветка bot10-unseen-keep 4409c5b, unseen_keep_tooling.diff, REPORT3.md, строка в EXPERIMENTS.md. Сообщений никому не отправлял.
- 21:07 (промежуточно, задача 3; всё в work4/):
  - **stack6 separate, глобально** (work4/stack6_global.json): drop Δ +0,006703, keep Δ +0,015057, P 0,948 (структурное: геофизика есть только в 3 семьях, нижняя граница CI 0,0000); LOFO > 0, 5/5 раскладок, тяжёлые 0. unified под drop даёт +0,033687; из них +0,01863 — артефакт unseen, под keep то же, что separate (+0,015057). **На участках** (work4/stack6_site.json): sep keep +0,014454, drop +0,006699, не отрицательно.
  - **ae_fort, глобально keep** (work4/aefort_global.json): +0,00524, P 0,926, LOFO +0,00684, 5/5. На участках (work4/aefort_site.json): +0,00526, P 0,923, не отрицательно. **crop**: стека и счётчиков на коробке нет, ждёт ПК.
  - **Реранкер arch, строгий П3** (вложенный: пороги на внутреннем 5-фолдовом OOF обучения; work4/rerank/res_*_nested5_micro.json):
    - k10 +0,009536, P 0,957; LOFO +0,007869, P 0,934; семена 0–4: +0,0024 / +0,0084 / +0,0105 / +0,0063 / +0,0074 (5/5) → П3 ✓.
    - Без Елизаветинского: k10 +0,0068.
    - Без rg_n/rg_pct: k10 +0,0044 (P 0,859), LOFO +0,0037. Без всех rg_*/conf_rel: k10 −0,0031, LOFO −0,0006 → выигрыш держится на признаках уровня участка.
    - На участках: k10 +0,00959, P 0,980.
  - **Микрокруги arch bot-7**: моя таблица пар (work4/rerank/pairs_arch_micro.pkl) воспроизводит bot-7 точно: +0,001076, P 0,867. На участках только +0,000227 → почти весь выигрыш — межучастковые совпадения. Реранк + микро k10 +0,010585 (P 0,985).
  - Идут: bot-7 kur/arch/bundle глобально и на участках (work4/bot7_*.json); реранк на участках, семена и LOFO; сверка site-кривых с патчем bot-8 (work4/xcheck_bot8.log). Затем пакет (work4/compose.py → work4/package.json), обзор bot-3/8/9/5, REPORT4.md, EXPERIMENTS.md.
- 21:14: задача 5 (лидар 59 / EhimenNathan): REPORT5_lidar59.md. Сообщений никому не отправлял.

## 28.10 21:17 МСК — задача 3 почти закрыта
- Пакет без crop: R+kur +0,010907 P 0,979 LOFO +0,007933 5/5; +archM +0,011955 (`work4/package.log`). bot-7 воспроизведён точно (`work4/bot7_global.json`). Rerank site LOFO +0,008152, семена 5/5 (`work4/rerank/*_site.log`).
- EXPERIMENTS.md: П6 (одинаковый unseen), вторичная site-проверка, перенос строки задачи 2, исправлен П3 у rerank_arch, 9 новых строк, сводка ACCEPTED/REJECTED/PENDING с командами.
- COMMON.md: дописана строка-поправка про потолок 0,985. REPORT4.md, REVIEW.md обновлены.
- В работе (не блокирует вывод): site-кривые bot-7 kur (queue2), xcheck bot-8 (queue3).

## 28.10 22:14 МСК — задача 4 (twins-only bot-6) закрыта
- REPORT6_twins.md.
- Код ветки `bot6-twins-only` ad8ae9b как есть: +0,000347, P 0,766 (П1 и П2 ✗). Причина — ошибка масштаба `_ground_scale(c.y)` для каждой точки в `post.kurgan_cross_modal_twins`: ae-двойников 5 вместо 85 (`work6/diag.log`).
- С починкой: +0,001718, P 0,986, LOFO +0,0016, 5/5 (= заявка bot-6). Полная с починкой: +0,001969 (заявка +0,00285 не подтверждена). С kur 0,5 м складываются: +0,003065, P 1,0.
- Site-scope подтверждает (`work6/stats_site.log`). Фоновые: queue2 (bot-7 site) готово — kur site +0,001574, LOFO +0,001482; xcheck bot-8 убит по памяти после 2/8 семей, обе совпали.
- EXPERIMENTS.md: 4 новые строки, обновлена строка полной twins, сводка (+2 строки, kur site). Копия до правки: `work6/EXPERIMENTS.bak_*.md`.
- Начата задача 5 (A2+R3 bot-3): `work7/`.
