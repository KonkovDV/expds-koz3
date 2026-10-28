# Отчёты ботов 2–10 за 28.10.2026

Снимок рабочих каталогов `/workspace/koz3_shared/bot-N` на 28.10.2026 ~21:40 МСК. Таблицы, CSV и JSON с результатами — в `bot-N/results/` (структура каталогов сохранена; `work`→`runs`, `data`→`data_`, `cache`→`cache_`, чтобы не попасть под `.gitignore`). Код ботов — в [`analysis/2026-10-28/bots/`](../../../analysis/bots_2026-10-28/), патчи — в [`patches/2026-10-28/`](../../../patches/2026-10-28/), Kaggle-ядра — в [`kaggle/2026-10-28/`](../../../kaggle/2026-10-28/).

Не перенесены: пообъектные выгрузки геометрии и OOF (производные от эталона), веса и бинарники, файлы >5 МБ, клоны чужих репозиториев.

## bot-2 — SOTA-обзор, рельеф, морфометрия, merge_across_sources

- [`EXPERIMENTS.md`](bot-2/EXPERIMENTS.md) — bot-2 — заранее объявленные эксперименты
- [`NOTES_L0thlorien-ds-expedition.md`](bot-2/NOTES_L0thlorien-ds-expedition.md) — NOTES: L0thlorien-ds-expedition (bot-2, 28.10.2026)
- [`NOTES_yakirill99-ds-expedition.md`](bot-2/NOTES_yakirill99-ds-expedition.md) — NOTES: yakirill99-ds-expedition (bot-2, 28.10.2026)
- [`RELIEF.md`](bot-2/RELIEF.md) — RELIEF.md — задание боту 9: многоканальный рельеф из LAS в Kaggle-ядре
- [`REPORT.md`](bot-2/REPORT.md) — REPORT bot-2 — обзор SOTA и топ-5 приёмов до 11.11.2026
- [`REPORT2.md`](bot-2/REPORT2.md) — REPORT2 (bot-2): морфометрический классификатор «повреждённый / целый» курган (топ-5, п. 2)
- [`STATUS.md`](bot-2/STATUS.md) — STATUS bot-2
- `results/` — 24 файлов замеров

## bot-3 — геофизика (object_poly, finds_points), MataNerdy

- [`GEO_GAP_REPORT.md`](bot-3/GEO_GAP_REPORT.md) — GEO_GAP_REPORT — bot-3, КОЗ №3 (28.10.2026, MSK)
- [`MATANERDY_VERDICT.md`](bot-3/MATANERDY_VERDICT.md) — MataNerdy — вердикт (v2, приоритетная проверка KOZ2 vs KOZ3)
- [`REPORT.md`](bot-3/REPORT.md) — bot-3 · REPORT: геофизика (object_poly, finds_points) — 2026-10-28
- [`STATUS.md`](bot-3/STATUS.md) — bot-3 STATUS (geophys)
- `results/` — 17 файлов замеров

## bot-4 — городища ↔ фортификации, пороги, карта потерь

- [`GAP.md`](bot-4/GAP.md) — GAP — карта потерь балла на честной базе A (bot-4, 28.10.2026 20:54 МСК)
- [`POST.md`](bot-4/POST.md) — POST — три постобработки курганов из COMPETITORS_DEEP §6 P0 п. 3–5 (bot-4)
- [`REPORT.md`](bot-4/REPORT.md) — bot-4: городища (вес 15) ↔ фортификации (вес 5)
- [`STATUS.md`](bot-4/STATUS.md) — STATUS bot-4 (28.10.2026, MSK)
- [`THR.md`](bot-4/THR.md) — THR — честные пороги (класс × группа) с вложенным выбором, поверх реранкера архитектуры (bot-4)
- `results/` — 46 файлов замеров

## bot-5 — архитектура, «без лидара»

- [`NOLIDAR.md`](bot-5/NOLIDAR.md) — NOLIDAR — дыра «без лидара» (bot-5, 28.10.2026, МСК)
- [`REPORT.md`](bot-5/REPORT.md) — bot-5 · Архитектура (`arkhitektury`, вес 10, F1 ≈ 0,08–0,09)
- [`STATUS.md`](bot-5/STATUS.md) — bot-5 STATUS (28.10.2026, 21:40 МСК)
- `results/` — 32 файлов замеров

## bot-6 — оптика/аэрофото 1940-х, двойники между типами данных

- [`REPORT.md`](bot-6/REPORT.md) — bot-6 — канал оптики / аэрофото 1940-х (ae): разбор «0,09 / 0,013»
- [`STATUS.md`](bot-6/STATUS.md) — bot-6 STATUS — канал оптики / аэрофото 1940-х
- [`TYPES.md`](bot-6/TYPES.md) — bot-6 — аудит «один объект в разных типах данных» (ЧАСТИЧНЫЙ, п. 1–3 из 4; размеры полигонов — ещё не сделано)
- `results/` — 21 файлов замеров

## bot-7 — геометрия полигонов под метрику, пакет-кандидат

- [`NOTES_KateBuikina-expds-solution.md`](bot-7/NOTES_KateBuikina-expds-solution.md) — NOTES — KateBuikina-expds-solution (bot-7, только чтение)
- [`NOTES_recara-cv-hackathon-nsk.md`](bot-7/NOTES_recara-cv-hackathon-nsk.md) — NOTES — recara-cv-hackathon-nsk (bot-7, только чтение)
- [`REPORT.md`](bot-7/REPORT.md) — bot-7 — геометрия полигонов под метрику koz3-1.5.0
- [`REPORT2.md`](bot-7/REPORT2.md) — bot-7 · REPORT2: пакет мелких правок (курганы 0,5 м + микрокруги архитектуры bot-5), честный парный замер по EXPERIMENTS.md
- [`REPORT3.md`](bot-7/REPORT3.md) — bot-7 · REPORT3 · кандидат на посылку (28.10.2026, МСК)
- [`STATUS.md`](bot-7/STATUS.md) — bot-7 STATUS (геометрия под метрику)
- `results/` — 53 файлов замеров

## bot-8 — надёжность и скорость сдачи, Rust-метрика, бюджет 4090

- [`BUDGET.md`](bot-8/BUDGET.md) — BUDGET.md — бюджет контейнера на RTX 4090 (24 ГБ), лимит 30 мин
- [`CORRECTNESS.md`](bot-8/CORRECTNESS.md) — КОЗ №3, бот 8: корректность сопоставления в build_curves (global против per-site)
- [`CRS_AUDIT.md`](bot-8/CRS_AUDIT.md) — CRS_AUDIT.md — поиск бага бейзлайна `crs = region_crs or file_crs`
- [`NOTES_Egor-git-cloud-ds-expedition.md`](bot-8/NOTES_Egor-git-cloud-ds-expedition.md) — NOTES: Egor-git-cloud-ds-expedition
- [`NOTES_Iskatel-potenzialov-gis.md`](bot-8/NOTES_Iskatel-potenzialov-gis.md) — NOTES: Iskatel-potenzialov-gis
- [`NOTES_KateBuikina-expds-solution.md`](bot-8/NOTES_KateBuikina-expds-solution.md) — NOTES: KateBuikina-expds-solution
- [`NOTES_KoshelevDV-ds-expedition-infra.md`](bot-8/NOTES_KoshelevDV-ds-expedition-infra.md) — NOTES: KoshelevDV-ds-expedition-infra
- [`NOTES_iwannasome-expds-rubleincorp-KOZ2.md`](bot-8/NOTES_iwannasome-expds-rubleincorp-KOZ2.md) — NOTES: iwannasome-expds-rubleincorp-KOZ2
- [`REPORT.md`](bot-8/REPORT.md) — bot-8 — надёжность и скорость сдачи КОЗ №3
- [`RUST_REPORT.md`](bot-8/RUST_REPORT.md) — bot-8 — ускорение метрики КОЗ №3 (Rust + векторизация NumPy)
- [`STATUS.md`](bot-8/STATUS.md) — bot-8 — STATUS (надёжность и скорость сдачи)
- `results/` — 165 файлов замеров

## bot-9 — очередь Kaggle (2×T4): верификатор, D-FINE, подкласс кургана

- [`NOLIDAR_VERIFIER.md`](bot-9/NOLIDAR_VERIFIER.md) — NOLIDAR_VERIFIER — верификатор «объект/фон» для участков без лидара (bot-9, 28.10.2026, MSK)
- [`REPORT.md`](bot-9/REPORT.md) — bot-9 · Очередь Kaggle (2×T4): диагноз edge/centers, ноутбуки, план GPU-часов
- [`REPORT2.md`](bot-9/REPORT2.md) — REPORT2 bot-9: 01 подкласс кургана, ядра s4/s5 (28.10.2026, МСК)
- [`REPRO.md`](bot-9/REPRO.md) — REPRO MataNerdy — ОТМЕНЕНО
- [`STATUS.md`](bot-9/STATUS.md) — STATUS bot-9 (Kaggle-очередь)
- [`VERIFIER.md`](bot-9/VERIFIER.md) — Верификатор кандидатов — ОТЛОЖЕНО
- `results/` — 17 файлов замеров

## bot-10 — статистика табло, приёмка, стратегия посылок

- [`REPORT.md`](bot-10/REPORT.md) — bot-10: статистика табло и стратегия посылок (28.10.2026, МСК)
- [`REPORT3.md`](bot-10/REPORT3.md) — REPORT3 — бот 10: «keep» групп без порога (находка бота 7), 28.10.2026
- [`REPORT4.md`](bot-10/REPORT4.md) — REPORT4 bot-10 — задача 3 (28.10.2026, 21:17 МСК)
- [`REPORT5_lidar59.md`](bot-10/REPORT5_lidar59.md) — REPORT5 — лидар на 59 LAS и сопоставимость с EhimenNathan (bot-10, 28.10.2026)
- [`REVIEW.md`](bot-10/REVIEW.md) — REVIEW bot-10: проверка отчётов ботов 2–9
- [`STATUS.md`](bot-10/STATUS.md) — STATUS bot-10
- `results/` — 70 файлов замеров
