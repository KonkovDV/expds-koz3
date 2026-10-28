# bot-7 STATUS (геометрия под метрику)
- [x] COMMON.md прочитан; лимиты: OMP_NUM_THREADS=2, nice 10, RSS ≤ 1.5 ГБ (подвыборка top-6000 на (класс, группу) = тот же балл, work/base_sub.py).
- [x] Аналитика + МК форм: work/mc_shapes.{py,txt,json}
- [x] Статистика GT/предсказаний: work/geom_stats.txt; диагностика промахов: work/diag.txt
- [x] Дубли в эталоне: work/gt_dups.txt (объясняет, почему склейка дублей вредна)
- [x] Генератор: repo/expds/geom_opt.py (ветка bot7-geom-opt)
- [x] Точная честная оценка: work/eval_geom.py → eval_geom.pkl/log; сводка work/summarize.py → eval_geom_summary.txt
- [x] REPORT.md (готово)

## Задача 2 (28.10.2026, МСК): пакет мелких правок — ГОТОВО (20:45 МСК)
- [x] ae-маска фортификаций bot-6 — уже в базе (default.toml `[models] ae`, группа ae_fort; абляция bot-10 +0,00524) → не применялась
- [x] Ветка bot7-bundle 42ada8d (post.micro_circles bot-5 + радиус по классу, хук watchdog, oof_transform micro --radius-map, блок конфига) → bundle.diff
- [x] Ветка замера bot7-bundle-measure (инструменты bot-10 + scripts/micro_fam.py) → bundle_measure_tools.diff
- [x] База воспроизведена: honest-10 0,3146153226828423 (кривые побитово = bot-10/work/curves)
- [x] Замер по EXPERIMENTS.md: work2/hb_{kur,arch,bundle}.json, breakdown.json, fixed_tau.json, summary.txt
- [x] Пакет +0,00245 [+0,00076; +0,00447], P 0,995, LOFO +0,0016, 5/5 → принят; курганы +0,00137 (P 0,953), архитектура +0,00108 (P 0,867) — только пакетом
- [x] REPORT2.md
- Нужно от ПК: перекалибровка порогов на OOF после oof_transform micro (REPORT2 §5); без push, без LB
- [x] Обзор конкурентов (только чтение): NOTES_recara-cv-hackathon-nsk.md, NOTES_KateBuikina-expds-solution.md — оба решения КОЗ2, заявленных 0,5–0,6 в коде нет

## Задача 3 (28.10.2026, МСК): кандидат на посылку — ГОТОВО (~21:20 МСК)
- [x] Состав изменён по просьбе бота 1 (21:02): курганы 0,5 м без микрокругов архитектуры; crop-fix bot-8; реранк архитектуры ВКЛЮЧЁН
- [x] Ветка bot7-candidate: 42ada8d, 07bed29, dd7982e, 733d3ed, fe0e570 (дерево e75e311); прежний вариант — bot7-candidate-v1 (913e2be)
- [x] Доказательства: bot-5 analysis/joint/summary.json (RM−R в среднем −0,00035); bot-10 work4/rerank/*.log (k10 +0,0095, LOFO +0,0079, 5/5); EXPERIMENTS rerank +0,00974
- [x] pytest 315 passed; toml diff = 5 ключей; сборщик репо → zip; check_submission PASS 28; время реранка 6,6 с / 697 МБ на 206 тыс. объектов
- [x] candidate.diff, candidate.mbox, REPORT3.md
- Заблокировано (F:): честный перемер пакетом с crop verify, platform_sim на mock_val
