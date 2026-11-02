# HANDOFF KOZ3 — как продолжить работу только по GitHub (обновлено 02.11.2026 14:05 MSK, coord/бот 1)

## АКТУАЛЬНО 02.11.2026 14:05 MSK (передача №2: у coord ~18% недельной квоты, процесс переходит к другому боту/аккаунту)
**Финал без изменений:** «смелый» v3_on **b447649** (sha `1406c01a…ef41d`), «надёжный» v2clean (`545ef32a…`). Оба Kaggle-parity PASS. Сабмитит только Елена: v3_on → при LB < 0.4124 откат на v2clean. Лучший LB 0.41238.

**Кандидат v4 = v3_on b447649 + C1F (+ возможно FP1) — ЖДЁТ независимого перегейта W4. Не финал, не сабмитить, пока W4 не PASS + Kaggle parity.**
- C1F (W1, bots/w1-satq): PREREG 1532475 → результат 5cc5ed7 (`docs/RESULTS_W1_FRZ.md`). Приор «центр сцены» для спутника (sp_dfine) при **замороженных на базе порогах всех не-спутниковых групп**: **Δ+0.0159 P1.000**, new34 0.0000, nolidar +0.0285, все классы ≥0 (kt +0.0003) → PASS. P2F (gor+arkh) Δ+0.0149 PASS (запасной). A1F FAIL (kt −0.0033).
- Финальный конфиг C1F: **af6b827** (`docs/W1_C1F_HANDOFF.md`, `configs/v3_on.toml [post.w1_c1f]/[post.w1_p2f]`, выбор на всех данных d0 0.25, s 0.75; флаги **по умолчанию off**; tests/test_sat_recall.py).
- W1-SATQ (PREREG 0614536 + Amendment 1 d425325, результаты **e574579**, `docs/RESULTS_W1_SATQ.md`): **FP1 PASS** — починка frame_prior (параметры v3_on 0.5/0.5 перепривязаны к sp_dfine, без подбора): **Δ+0.00858 P0.999**, new34 +0.0009, nolidar +0.019, полы ok. Q1a/b/c (arkh-приор/τ) FAIL по P; FP2 nested и Q2 (≥4 Мпикс) FAIL по new34. C1F и FP1 вместе ещё **не гейтились** — нужен совместный гейт.
- W4 перегейт C1F: PREREG **e6b5a31** (bots/w4-v3on, `docs/bots/2026-11-02/bot-W4/c1f_regate/PREREG.md`), слот box/c1f до 19:57. **Найден баг порта af6b827:** в проде `_sp_dfine_phase` не вызывает `_with_scene_drel`, поэтому в архиве C1F-приор был бы no-op (работали бы только пониженные τ). W4 чинит под флагом (off = побайтно как b447649) + unit-тест. При PASS W4 собирает архив v4 и Kaggle parity (как v5 для b447649), sha256 → FINAL_CANDIDATES.md.
- Оговорка: правило заморозки придумано после просмотра new34 (W4 Q0), так что проверка new34 не полностью слепая — записано в PREREG.

**Урок дня (обязателен для всех гейтов):** при гейте приёма для одной группы **замораживать пороги всех остальных групп на базе** (taus_V3P[fold]). Nested-перевыбор чужих порогов (лидарные gor: gor_phys f4, li_coarse f5/f7) давал ложный минус new34 (W4 Q0 7c33027, `docs/bots/2026-11-02/bot-W4/q0_new34/Q0_NEW34.md`). Именно это превратило C1 (FAIL) в C1F (PASS). Старые «почти PASS» линии стоит перегейтить с заморозкой.

**Закрыто 02.11 после утра (FAIL, флаги off):**
- W1 SATREC2 A1–A3/B1–B3 (3dcd07e); W1 SATQ Q1a/b/c, FP2, Q2 (e574579); W1 SATDET трек P1/P2 (PREREG 843d807) — см. RESULTS в bots/w1-satq / bots/w1-satdet.
- W2 P-W2-12 мягкий бленд сидов лидара (PREREG d338318, результат 3dbe4ce): blend3 Δ−0.0035, blend4 Δ−0.0003, целые курганы −0.015.
- W3 KURG-SUBCLS (30c12b3, AUC 0.689), KURG-SUBDROP (3c92fd7); Q3 gor-swap Δ+0.0031 P0.787 new34 −0.0076, Q4 li→sp_or Δ+0.0006, Q5 спутн. хедж Δ−0.0003 kt −0.0063 (bots/w3-kurgan-seeds 1028b70, 8af2969, f7e5545).
- W4 Q6 GBM находок (clean room; PREREG 870433c, A1 a4da8a9, 3b7a097): прод-реплей все FAIL (UNI Δ−0.0075, REP −0.016), хотя new34 у UNI +0.02 — разрыв gate↔прод из-за merge_within_markup/filter_finds_points_regimes. Детали `docs/bots/2026-11-02/bot-W4/q6_finds_gbm/HANDOFF.md`.

**В работе (на 14:05 MSK):**
- Бот 2 (W1): SATDET Faster R-CNN R50-FPN v2 OOF на GPU **elenakonkova** (ETA ~17:25 MSK) → гейт D1/D2 (tools/satq, run_det.sh, bots/w1-satdet 7cbf4e4).
- Бот 3 (W2): P-W2-13 **второй D-FINE (sp_or)** на GPU **dmitrijkv**, 2 кернела (PREREG 1c0ddc0 + amend 34f97d3, bots/w2-lidar a614dfa), аренды acc2/gpu-1 до 17:48, gpu-2 до 18:48; ранний стоп f0+f3, затем гейт поверх v3_on.
- Бот 4 (W3): перегейт KDUAL / LI-KURG-TAU с замороженными чужими порогами (slot box/q2 до 15:54); Q6 передан от W4 (продолжать только при идее, как закрыть разрыв gate↔прод).
- Бот 5 (W4): независимый перегейт C1F (+ фикс scene_drel) → v4 архив + Kaggle parity.

**Баг:** frame_prior в v3_on b447649 фактически выключен (привязан к спутниковым группам, которые D-FINE выкидывает). Сабмиту не вредит; починка = FP1 (PASS выше).

**Разбор конкурентов:** `orch/COMPETITOR_GAPS_2026-11-02.md` (d2fe1de; Drive 11K6QztmfE40YJJHgCtDfF2BxfCb3fMIC) — 10 разрывов vs EN (0.502), только clean-room спецификации.

**Резервные копии №2:** Drive KOZ3-final/backup_2026-11-02b, GitHub bots/backup-2026-11-02b, Kaggle приватный `<acc>/koz3-handoff-20261102b` на 5 аккаунтах — см. `orch/BACKUP_2026-11-02b.md`. Утренний пакет (веса, results) — `orch/BACKUP_2026-11-02.md`, не дублируется.


## 02.11.2026 утро 09:45 (передача №1, история)
**Финал (не меняется):**
- «смелый» = **v3_on b447649** `koz3_v3on_b447649_6d4764f8_w6dfd2e07.zip`, sha256 `1406c01a9aaafc996862a14f12cde14f6bf05b313ad0bbce14428df1d10ef41d`, 722 697 913 B. honest-10 0.333787 (Δ+0.0198 P0.976 vs v2clean). Kaggle parity v5 PASS 4/4 среды (D-FINE грузится, V3_DEGRADED нет), вывод побайтно = старому ebc8e4ba.
- «надёжный» = **v2clean 545ef32a** `koz3_v2clean_70d335a_c028f4eb_wb042594f.zip`, sha256 `545ef32a70b4994f4d694828ae5eb28f27c9161d7f19a96f5a9d3833b87e093f`, 568 488 239 B. Kaggle parity v5 PASS 5/5 сред (PREREG 8844d49, W4 9e711f6), вывод = REF_check4.
- Оба parity-проверены. **Сабмитит только Елена:** сначала v3_on; если LB < 0.4124 → финал v2clean. Старый ebc8e4ba НЕ сабмитить (молча откатывался на v2).
- Где лежат: Drive KOZ3-final (приватная) — папки v3_on_b447649_1406c01a, v2clean_545ef32a, backup_2026-11-02; Kaggle приватный датасет `<acc>/koz3-handoff-20261102` (финальные zip + веса/результаты + orch-доки + SHA256SUMS; на всех 5: dmitrijkv, elenakonkova, usertest1234, aaaa0000004, aaaadg4gg4g4g — приватность проверена 02.11 ~09:55 MSK: чужой аккаунт 403, аноним 403; zip лежат как `.zip.bin`, т.к. Kaggle распаковывает .zip/.tar).

**Гейты, закрытые 02.11 как FAIL (флаги off):**
- W1 SATREC R1/R2/R3/C1/C2 — лучший C1 (приор «центр сцены») Δ+0.0142 P0.9995, режется только new34 −0.0011 (пол ≥0); R1 (единый τ) Δ+0.0100 P0.997, new34 −0.0023. Итог: `docs/RESULTS_W1_SATREC.md` (bots/w1-satrec 06ef4ec). 65% потерь спутника — пороги, не детектор.
- W3 KDUAL (двойная выдача курганов) Δ+0.0031 P0.87 — PREREG 6e1d33d, результат dae47a7, runs/gate_KDUAL.json.
- W3 KDUAL-damaged-only Δ+0.0023 P0.968, new34 −0.0007 — PREREG 5a804b9, b2a5a4d.
- W3 LI-KURG-TAU (4 варианта, Бонферрони P≥0.975): A Δ+0.0044 P0.81; B1 −0.060; B2 −0.055; C +0.0015 — PREREG c96a0d6, результаты 0b35b80 (runs/gate_LIKT.json). Вывод: рост damaged = потеря intact → рычаг в разделении подклассов, не в порогах.

**Сейчас идут (box слоты q1/q2):**
- W1 **SATREC2** (бот 2, ветка bots/w1-satrec2, wt/w1sr): C1+/R1+ с силой приора по классам, расширенная сетка, заранее заданная защита new34; новый PREREG.
- W3 **KURG-SUBCLS** (бот 4, bots/w3-kurgan-seeds): LightGBM-разделение damaged/intact по геоморфометрическим признакам.
- Oracle-gap разбор W3: `runs/gap_breakdown.md` / `.json` (bots/w3-kurgan-seeds b2a5a4d). Потолок 0.690 при базе 0.3220; нет кандидата +0.173, ниже порога +0.173, не тот класс +0.095, геометрия +0.042.
- Конкуренты: `COMPETITORS_2026-11-02.md` (EN 0.502 vs наш 0.412; ⅔ отставания — полнота спутника).

**Безопасность (02.11):**
- 13 публичных датасетов на dmitrijkv **удалены** (xfer s000–s009, gt-train-v2, fr1h-wheels, fr1h-chk-20261101); с другого аккаунта не видны. Все 105 ноутбуков на 5 аккаунтах приватные.
- Drive-папка **KOZ3-pixgbm всё ещё открыта по ссылке** — закрыть может только Елена вручную (коннектор не снимает доступ).
- Ключ S3 организаторов захардкожен в коде 5 **приватных** кернелов (публично не виден; сменить могут только организаторы).
- 31 git-конфиг вне KOZ3 содержит GitHub-токен — ждём решения Елены.
- Kaggle-токены лежат только в `~/.secrets/kpool/<user>` на боксе, **никогда не в git/датасетах**. Вызов: `env -u KAGGLE_USERNAME -u KAGGLE_KEY KAGGLE_API_TOKEN=$(cat ~/.secrets/kpool/<user>) ~/.local/bin/kaggle …` (токены не печатать).
- **ПРАВИЛО: любой Kaggle-датасет создавать только с `"isPrivate": true` в dataset-metadata.json.** Публичный датасет нельзя сделать приватным (API отвечает «успешно», но не меняет) — только удалить. После создания проверять с другого аккаунта (`datasets files` должен падать).

**Резервные копии:** Drive KOZ3-final/backup_2026-11-02 (27 частей, 2.4 ГБ, BACKUP_MANIFEST.md с sha256 и инструкцией восстановления); GitHub bots/backup-2026-11-02 @ 78ffd05 (947 текстовых файлов); этот манифест — `BACKUP_2026-11-02.md`.

## История (до 02.11)
> **ФИНАЛ 01.11 23:50 MSK:** «смелый» = **v3_on b447649** (sha 1406c01a…ef41d, 722 697 913 B, Kaggle parity v5 PASS 4/4 среды, вывод побайтно = ebc8e4ba; Drive KOZ3-final/v3_on_b447649_1406c01a), заменяет ebc8e4ba; «надёжный» = v2clean 545ef32a (Kaggle parity v5 PASS 5/5 сред, PREREG 8844d49, W4 9e711f6). План: Елена сабмитит v3_on b447649 → при LB < 0.4124 откат на v2clean.
> *(прежняя запись)* **ФИНАЛ 01.11 22:35 MSK:** «смелый» v3_on (sha ebc8e4ba…, Δ+0.0198 P0.976, прод Δ+0.0194), «надёжный» v2clean (sha 545ef32a…). Все sha, ссылки на Drive (KOZ3-final), закрытые линии и риск regex — в [`FINAL_CANDIDATES.md`](FINAL_CANDIDATES.md). План: фикс regex (бот 5) → сабмит v3_on Еленой → при LB < 0.4124 откат на v2clean.
> **АКТУАЛЬНО 01.11 18:47:** см. [`HANDOFF_NOW_20261101.md`](HANDOFF_NOW_20261101.md) (живой Kaggle + Drive + next steps).

> Правило: HANDOFF.md обновляется после КАЖДОГО значимого шага (гейт PASS/FAIL, новый прогон, смена железа, решение Елены).

## Цель
- Соревнование KOZ3 (археология: оптика + лидар + геофизика). Цель LB **>0.5**, stretch **0.61**.
- Лучший официальный LB: **0.4124** (29.10, v2 eb268d6). Внутренняя база honest-10 **A=0.314615**.
- Дедлайн сдачи **11.11.2026**. Все новые гейты должны пройти **до 08.11**; 08–11.11 — сборка, check_submission, документы.
- Репо: `KonkovDV/expds-koz3`. Координация: ветка `bots/orch-protocol`, папка `orch/`. **main не пушить никогда.**

## Схема оркестрации (боты)
| Бот | Роль | Ветки |
|---|---|---|
| 1 | coord: план, гейт-решения, очередь, аренды, железо, сводка Елене | bots/orch-protocol |
| 2 | W1: D-FINE / оптика (ae, or, облачные признаки CI2-1) | bots/w1-aeor |
| 3 | W2: лидар (li_*, heatmap lihm, r50c5, CI2-3/CI2-8) | bots/w2-lidar |
| 4 | W3: курганы / городища (сиды, WS-G1) | bots/w3-kurgan-seeds |
| 5 | W4: гейт-перепроверки, геофизика (ORG3), архив, DOC-1 | bots/w4-gate, bots/w4-pix-clean, bots/w4-aidecl |
| helper coord | W5: rerank / пороги | bots/w5-rerank |
Боты пишут статус в `orch/bot-WN/STATUS.md`; coord сводит в STATE.md и COORD_LOG.md. Сообщения бот↔бот асинхронны.

## Железо
- Kaggle (5 аккаунтов): **GPU/TPU только elenakonkova и dmitrijkv**; CPU-only: usertest1234, aaaa0000004, aaaadg4gg4g4g — **у последних нет интернета/S3** (только приватные датасеты Kaggle; никаких ключей в датасетах).
- dmitrijkv GPU: квота до Сб 31.10 03:00. TPU: кернелы часто висят в очереди Kaggle → писать «TPU не занят, простой из-за очереди Kaggle».
- Бокс (компьютер бота): 8 ядер, слоты `slot:box/qN`.

## Где секреты (только имена, значений здесь нет)
- env: `GITHUB_TOKEN`, `KAGGLE_USERNAME`, `KAGGLE_KEY`, `KAGGLE_KEY_1`, `KAGGLE_KEY_2`, `KAGGLE_API_TOKEN`; конфиг `~/.config/kaggle/`.
- Пул Kaggle-токенов: `~/.secrets/kpool/{dmitrijkv,elenakonkova,usertest1234,aaaa0000004,aaaadg4gg4g4g}` (права 600, не в git).
- S3 организаторов: `/workspace/koz3/s3_keys.tsv` (на боксе, не в git). Датасет с cred.json на aaaadg4gg4g4g удалён 30.10 09:40.
- Перед любым push — secret-scan (паттерны KGAT_, ghp_, github_pat_, AWS-ключи). Скрипты кернелов с токенами не пушить, сначала вынести в env.

## Протокол аренд / очереди
- `orch/lease.sh` (take/release/list) → LEASES.json; `orch/lease.sh sync` зеркалит LEASES.json+COORD_LOG.md в bots/orch-protocol. Аренда coord `branch:bots/orch-protocol` до 13.11 — не сокращать.
- `QUEUE.json` + `enqueue.py`: задачи со статусами RUNNING/hold/queued; hold снимает только coord.
- `watcher.py` / `gpu_watcher.py` / `cpu_watcher.sh`: снимают аренды по завершённым кернелам (баг «снимает box-слоты посреди расчёта» исправлен 30.10).
- `COORD_LOG.md`: журнал решений coord (время MSK, что/почему).

## Правила гейта
- honest-10 против A, bootstrap 2000; PASS = **Δ≥+0.003 и P(B>A)≥0.9**, полы по классам не нарушены, new34/nolidar без регрессий.
- PREREG коммитится ДО замера. Модальностный гейт для модальностных сигналов (напр. EtGr только 212–217, остальные байт-в-байт).
- Пост-hoc выбор из k вариантов → **Бонферрони** (P≥1−0.1/k) и независимая перепроверка ботом 5. Пороги после гейта — только nested-fold.

## Жёсткие правила
- Сабмиты на LB делает **только Елена**. Боты не сабмитят.
- Не пушить main. Только рабочие ветки.
- В финале **нет кода, производного от EN** (geo_pixgbm и т.п.). DOC-1 guard: `bots/w4-pix-clean 70d335a`, порт в `bots/w4-gate ab9fd82` — сборка падает при EN-файле. Файлы EN в main/analysis и bots/bot3-v4-patches-nomodels не трогать, в сборку не берутся. Прогоны на EN-коде — только для сравнения.
- Удаления/внешние действия — только с одобрения Елены.

## Рутина «KOZ3 ночной контроль» (каждые 15 мин)
Проверяет кернелы всех 5 аккаунтов и бокс, снимает/выдаёт аренды, дозагружает свободные слоты из QUEUE (не hold), пингует простаивающих ботов, пишет в COORD_LOG, Елене — только при приросте или блокере.

## Экономия токенов
Лимит на исходе (обновится Вс 01.11 19:21). Минимум сообщений, без подтверждений-квитанций; при смене аккаунта новый ассистент читает HANDOFF.md → STATE.md → MASTER_PLAN.md → хвост COORD_LOG.md → QUEUE.json/LEASES.json.

## PROMPT для нового ассистента (скопировать целиком; обновлено 02.11.2026 14:05 MSK)
```
Ты продолжаешь проект KOZ3 «Экспедиция DS. Комплексирование» (соревнование по археологии: оптика/спутник + лидар + геофизика)
для Елены Коньковой. Работай на русском, коротко. Цель LB > 0.5 (лучший наш 0.41238; EN 0.502). Дедлайн 11.11.2026,
все новые гейты — до 08.11; 08–11.11 только сборка, check_submission, документы, декларация ИИ.
Елене пиши ТОЛЬКО при приросте балла (PASS гейта / готовый архив) или блокере. Экономь токены.

0. ПРАВИЛА БЕЗОПАСНОСТИ (нарушать нельзя):
   - Никогда не печатать и не коммитить токены/ключи (Kaggle KGAT, GITHUB_TOKEN, ghp_/github_pat_, S3/AKIA, cred.json).
     Ключи только в ~/.secrets на боксе. Перед каждым push/выгрузкой — secret-scan (KGAT_, ghp_, github_pat_, AKIA, aws_secret, cred.json).
   - Сабмиты на LB делает только Елена. main не пушить, без force-push. Перед push брать lease (orch/lease.sh acquire).
   - Ничего не удалять/не перемещать (ветки, PR, файлы, датасеты, ПК Елены) без её явного «да».
   - Не трогать чужие RUNNING/QUEUED кернелы Kaggle.
   - В финале нет кода, производного от EN (DOC-1 guard: сборка падает на EN-файле). Идеи EN — только clean-room спецификации + гейт.
   - Kaggle: любой новый датасет ТОЛЬКО с "isPrivate": true в dataset-metadata.json (без поля библиотека делает его публичным;
     public→private Kaggle не умеет). После создания проверить с токена ДРУГОГО аккаунта: `datasets files` → 403/404.
     Вызов: env -u KAGGLE_USERNAME -u KAGGLE_KEY KAGGLE_API_TOKEN=$(cat ~/.secrets/kpool/<user>) ~/.local/bin/kaggle ...
     GPU только elenakonkova и dmitrijkv; usertest1234, aaaa0000004, aaaadg4gg4g4g — CPU без интернета/S3.
   - Drive: всё в приватной папке KOZ3-final (1GzW2XGWIMVybFV2pP3WLryeIIhQeN8NB); доступ не расширять, новые файлы не anyone-with-link.

1. Клонируй KonkovDV/expds-koz3 (приватный), ветка bots/orch-protocol. Читай по порядку: orch/HANDOFF.md (раздел
   «АКТУАЛЬНО 02.11 14:05») → orch/STATE.md → orch/FINAL_CANDIDATES.md → orch/COMPETITOR_GAPS_2026-11-02.md
   → хвост orch/COORD_LOG.md → orch/QUEUE.json, orch/LEASES.json → orch/BACKUP_2026-11-02.md и BACKUP_2026-11-02b.md.
   Бокс: /workspace/koz3 (orch/, wt/coord-orch — worktree bots/orch-protocol, w4geo/dist — архивы).

2. ФИНАЛЫ: «смелый» v3_on b447649 (koz3_v3on_b447649_6d4764f8_w6dfd2e07.zip, sha256 1406c01a…ef41d, 722 697 913 B),
   «надёжный» v2clean (koz3_v2clean_70d335a_c028f4eb_wb042594f.zip, sha256 545ef32a…). Оба Kaggle parity PASS.
   Елена: сначала v3_on; LB < 0.4124 → v2clean. ebc8e4ba НЕ сабмитить. Проверь в OFFICIAL_SCORES.md, сабмитила ли она.
   КАНДИДАТ v4 = v3_on + C1F (+FP1?) — ждёт перегейта W4 (см. п.5.1).

3. АРТЕФАКТЫ: Drive KOZ3-final (v3_on_b447649_1406c01a, v2clean_545ef32a, backup_2026-11-02, backup_2026-11-02b, reports);
   GitHub bots/backup-2026-11-02 и bots/backup-2026-11-02b; приватные Kaggle-датасеты <acc>/koz3-handoff-20261102 и
   <acc>/koz3-handoff-20261102b на всех 5 аккаунтах (zip лежат как .zip.bin; сверяй SHA256SUMS).
   Восстановление бокса: см. BACKUP_2026-11-02.md (cat partNN > tar, sha256sum, tar -xf в /workspace/koz3);
   сырые данные организаторов — из S3 (ключи у Елены, в репо их нет).

4. ПРАВИЛА ГЕЙТА: honest-10 поверх v3_on b447649 (прод-база 0.345439, гейт-база 0.333787), bootstrap 2000 (site-family),
   PASS = Δ≥+0.003 и P≥0.9 (k вариантов → Бонферрони P≥1−0.1/k), полы по классам, new34≥0 и nolidar≥0.
   PREREG коммитится ДО замера; выбор параметров только nested; PASS проверяет независимо бот 5 (W4), затем прод-реплей
   и Kaggle parity. НОВОЕ ПРАВИЛО: при гейте приёма для одной группы ЗАМОРАЖИВАТЬ пороги всех остальных групп на базе
   (taus_V3P[fold]) — иначе перевыбор лидарных gor-порогов даёт ложный минус new34 (W4 Q0, bots/w4-v3on 7c33027).

5. СЛЕДУЮЩИЕ ШАГИ ПО ПРИОРИТЕТУ:
   5.1 Довести v4: W4 (бот 5) перегейтирует C1F (PREREG e6b5a31) и чинит баг порта (sp_dfine без scene_drel → приор no-op).
       При PASS: совместный гейт C1F+FP1 (FP1 = починка frame_prior, PASS Δ+0.0086 P0.999, e574579) — если вместе PASS,
       v4 = v3_on + C1F + FP1, иначе v3_on + C1F. Сборка архива, check_submission, Kaggle parity в 4+ средах,
       sha256 → FINAL_CANDIDATES.md, копии в Drive KOZ3-final и приватный Kaggle-датасет. Потом Елене: «v4 готов к сабмиту».
   5.2 Дождаться и гейтировать GPU-линии: W1 SATDET Faster R-CNN (elenakonkova, ETA ~17:25 02.11) → D1/D2;
       W2 P-W2-13 второй D-FINE (dmitrijkv, аренды до 17:48/18:48) → ранний стоп f0+f3 → гейт поверх v4/v3_on.
   5.3 Перегейт «почти PASS» линий с заморозкой чужих порогов: KDUAL (Δ+0.0031), LI-KURG-TAU A (+0.0044), P2F как альтернатива,
       Q1 arkh (+0.0047, не хватило P), Q3 gor-swap (+0.0031) — бот 4 уже начал KDUAL/LI-KURG-TAU.
   5.4 Остальное из COMPETITOR_GAPS (только clean-room): fold-ансамбль D-FINE на инференсе (+0.003…0.010, гейтом не меряется —
       только с согласия Елены), framing gate на больших сценах в форме без потерь new34, Q6 GBM находок — лишь если
       есть способ закрыть разрыв gate↔прод (merge_within_markup / filter_finds_points_regimes).
   Не повторять закрытое: SATREC/SATREC2, KURG-SUBCLS/SUBDROP, бленд сидов лидара, lidar-aug, r50c5, J2/CTXc, Q4, Q5, Q6 REP.

6. ЧТО У КАКОГО БОТА (на 02.11 14:05): бот 2 = W1 (спутник/оптика, bots/w1-satq, bots/w1-satdet), бот 3 = W2 (лидар +
   второй D-FINE, bots/w2-lidar), бот 4 = W3 (курганы/городища, bots/w3-kurgan-seeds), бот 5 = W4 (гейт, архив,
   bots/w4-v3on). Статусы: orch/bot-WN/STATUS.md и STATUS в их ветках. Рутина «KOZ3 ночной контроль» каждые 15 мин.

7. ОТКРЫТЫЕ ВОПРОСЫ К ЕЛЕНЕ: (a) закрыть доступ по ссылке к Drive-папке KOZ3-pixgbm (коннектор не умеет);
   (b) GitHub-токен в 31 git-конфиге вне KOZ3 — чистить ли; (c) ключ S3 организаторов захардкожен в 5 приватных кернелах;
   (d) ФИО/роли/ИИ-инструменты для декларации ИИ до 11.11; (e) чистить ли общий клон /workspace/koz3/repo
   (81 изменённый отслеживаемый файл, check_conflicts ERROR); (f) рутину с 15 мин на раз в час для экономии квоты.

8. После каждого значимого шага обновляй orch/HANDOFF.md и STATE.md в /workspace/koz3/wt/coord-orch
   (git -c user.name=coord-lease -c user.email=coord@koz3.local commit, lease → push в bots/orch-protocol),
   затем cd /workspace/koz3/orch && ./lease.sh sync.
```
