# STATE KOZ3 — актуально на 02.11.2026 14:40 MSK (coord, передача №3, финальная)
## ФИНАЛ (без изменений)
- «смелый» = **v3_on b447649** sha 1406c01a…ef41d (основной); «надёжный» = v2clean 545ef32a… (запасной). *(УСТАРЕЛО: v4 собран 15:05 — koz3_v4c1f_07795d2 sha f0def982…, см. FINAL_CANDIDATES §0.)*
## C1F / FP1 — PASS у W4 (607edef)
- **W4, правило метрики, санити 0/50: C1F-all PASS Δ+0.0162 P1.0 new34 +0.0007 (лучший); C1F-gor/arkh PASS Δ+0.0134 new34 0; FP1 PASS Δ+0.0086.**
- C1F: W1 FRZ PASS Δ+0.0159 (5cc5ed7); W4 перегейт 5d0dd4c (FAIL Δ+0.0122) ОТМЕНЁН AMEND1 c60e50f8. Причина — правило порогов у W4 (FP=1, порог на скоре) вместо метрики (FP=0.5, середина): W1 3929737, docs/W1_FRZ_DISCREPANCY.md; с правилом прода Δ+0.0149/+0.0152, new34/kt ≥0. W4 подтвердил санити 0/50 и закоммитил PREREG_AMEND1 (bots/w4-v3on c60e50f8); замер выполнен (607edef).
- Баг scene_drel (C1F no-op в b447649/af6b827) → фикс 98cdf56 (474 теста).
- FP1 W1 PASS Δ+0.0086 (e574579), подтверждён W4. C1F+FP1 не аддитивна (kt −0.0035) — только отдельный гейт.
## FAIL после 14:00
- Q6 GBM находок (3b7a097, нужен новый PREREG с maggr/фильтром finds); LI-KURG-TAU с заморозкой (e6089c2, d4c550f); Q1 arkh, FP2, Q2 (e574579). KDUAL не перегейчивали (перевыбора не было).
## УРОК
- Замораживать чужие пороги; подбирать затронутые правилом метрики (FP=0.5, середина между скорами); санити — воспроизведение базовых порогов.
## RUNNING (без присмотра)
- W1 SATDET elenakonkova/koz3-w1-satdet-frcnn v2, ETA ~17:25 → tools/satdet/run_det.sh (bots/w1-satdet; STATUS_W1 a0c10733) (orch/bot-W1/STATUS_W1.md).
- W2 D-FINE-2 a/b RUNNING на dmitrijkv (STATUS_W2 5b5e3251) → ранний стоп/гейт по orch/bot-W2/STATUS_W2.md.
- Рутина «KOZ3 ночной контроль» ПРИОСТАНОВЛЕНА 02.11 14:27. Боты свёрнуты, leases сняты. pool_kaggle.sh `<username> <kaggle args>`.
## NEXT
1) собрать v4 = v3_on + 98cdf56 + флаг C1F-all, parity 4+ сред, sha256, копии; 2) гейты SATDET и D-FINE-2; 3) 08–11.11 сборка, декларация.

---
# STATE KOZ3 — актуально на 02.11.2026 14:05 MSK (coord)
## ФИНАЛ (без изменений, см. FINAL_CANDIDATES.md)
- «смелый» = **v3_on b447649** sha 1406c01a…ef41d; «надёжный» = v2clean 545ef32a…. Оба Kaggle parity PASS. Елена: v3_on → при LB < 0.4124 v2clean. Лучший LB 0.41238.
## КАНДИДАТ v4 (история; перегейт W4 выполнен — PASS 607edef)
- **C1F PASS** (W1, bots/w1-satq PREREG 1532475 → 5cc5ed7): Δ+0.0159 P1.000, new34 0, nolidar +0.0285, классы ≥0. Конфиг af6b827 (docs/W1_C1F_HANDOFF.md, d0 0.25, s 0.75, флаги off). P2F Δ+0.0149 PASS (запас). A1F FAIL (kt).
- **FP1 PASS** (W1-SATQ e574579): починка frame_prior → sp_dfine, Δ+0.00858 P0.999, new34 +0.0009. С C1F совместно не гейтился.
- W4 перегейт C1F: PREREG e6b5a31 (bots/w4-v3on), slot box/c1f до 19:57; найден баг порта (sp_dfine без scene_drel → приор no-op), чинится под флагом. При PASS → архив v4 + Kaggle parity.
- Урок: при гейте замораживать пороги чужих групп на базе (W4 Q0 7c33027).
## ЗАКРЫТО 02.11 (FAIL, флаги off)
- Утро: W1 SATREC R1–R3/C1/C2 (06ef4ec); W3 KDUAL (dae47a7), KDUAL-damaged (b2a5a4d), LI-KURG-TAU (0b35b80).
- День: W1 SATREC2 (3dcd07e); W1 SATQ Q1a/b/c, FP2, Q2 (e574579); W1 SATDET P1/P2 (843d807); W2 P-W2-12 бленд сидов (3dbe4ce); W3 KURG-SUBCLS (30c12b3), SUBDROP (3c92fd7), Q3/Q4/Q5 (f7e5545); W4 Q6 GBM находок (3b7a097).
## RUNNING
- Бот 2 W1 SATDET Faster R-CNN, GPU elenakonkova, ETA ~17:25 → гейт D1/D2.
- Бот 3 W2 P-W2-13 второй D-FINE, GPU dmitrijkv (acc2/gpu-1 до 17:48, gpu-2 до 18:48), PREREG 1c0ddc0.
- Бот 4 W3 перегейт KDUAL/LI-KURG-TAU с заморозкой (box/q2 до 15:54).
- Бот 5 W4 перегейт C1F → v4.
## АНАЛИЗ
- COMPETITOR_GAPS_2026-11-02.md (d2fe1de), oracle gap (W3 runs/gap_breakdown.md), Q0 new34 (W4 7c33027).
## БЕЗОПАСНОСТЬ / ОТКРЫТО У ЕЛЕНЫ
- KOZ3-pixgbm на Drive открыта по ссылке; 31 git-конфиг вне KOZ3 с токеном; S3-ключ в 5 приватных кернелах; декларация ИИ; общий клон /workspace/koz3/repo (81 файл); рутина 15 мин → 1 ч?
- Датасеты Kaggle только `"isPrivate": true` + проверка с другого аккаунта.
## КОПИИ
- Утро: backup_2026-11-02 (Drive / GH bots/backup-2026-11-02 / Kaggle koz3-handoff-20261102). День: backup_2026-11-02b (BACKUP_2026-11-02b.md).
## NEXT
1) W4 перегейт C1F → (C1F+FP1 совместно) → v4 архив + parity → Елене; 2) гейты SATDET и D-FINE-2; 3) перегейт почти-PASS линий с заморозкой; 4) 08–11.11 сборка, декларация.

---
# STATE KOZ3 — актуально на 02.11.2026 09:45 MSK (coord)
## ФИНАЛ (см. FINAL_CANDIDATES.md)
- «смелый» = **v3_on b447649** sha **1406c01a…ef41d** (722 697 913 B), honest-10 0.333787; Kaggle parity v5 PASS 4/4.
- «надёжный» = **v2clean 545ef32a…** (568 488 239 B), honest-10 0.313967; Kaggle parity v5 PASS 5/5 (PREREG 8844d49, W4 9e711f6).
- Оба parity-проверены. Сабмитит Елена: v3_on → при LB < 0.4124 откат на v2clean. Лучший LB пока 0.4124.
- Копии: Drive KOZ3-final (приватная) + backup_2026-11-02; GitHub bots/backup-2026-11-02 @78ffd05; Kaggle приватный `<acc>/koz3-handoff-20261102` на dmitrijkv, elenakonkova, usertest1234, aaaa0000004, aaaadg4gg4g4g (проверено: 403 извне).
## ЗАКРЫТО 02.11 (FAIL, флаги off)
- W1 SATREC R1/R2/R3/C1/C2: лучший C1 Δ+0.0142 P0.9995 (режет только new34 −0.0011); R1 Δ+0.0100 P0.997 (new34 −0.0023). bots/w1-satrec 06ef4ec, docs/RESULTS_W1_SATREC.md.
- W3 KDUAL Δ+0.0031 P0.87 (6e1d33d/dae47a7); KDUAL-damaged Δ+0.0023 P0.968 (5a804b9/b2a5a4d); LI-KURG-TAU 4×FAIL, лучший A Δ+0.0044 P0.81 (c96a0d6/0b35b80).
## RUNNING
- W1 SATREC2 (бот 2, bots/w1-satrec2, slot box/q1 до 02.11 21:23): C1+/R1+ сила приора по классам + защита new34.
- W3 KURG-SUBCLS (бот 4, bots/w3-kurgan-seeds, slot box/q2): LightGBM damaged/intact по геоморфометрии.
## АНАЛИЗ
- Oracle gap (W3, runs/gap_breakdown.md): база 0.3220, оракул 0.690; нет кандидата +0.173, ниже порога +0.173, не тот класс +0.095, геометрия +0.042, 1:1 +0.016.
- Конкуренты: COMPETITORS_2026-11-02.md (EN 0.502; наш LB 0.412).
## БЕЗОПАСНОСТЬ
- 13 публичных датасетов dmitrijkv удалены 02.11. KOZ3-pixgbm на Drive открыта по ссылке — закрывает Елена вручную. S3-ключ захардкожен в 5 приватных кернелах. 31 git-конфиг вне KOZ3 с GitHub-токеном — ждём Елену.
- Kaggle-токены только в ~/.secrets/kpool (не в git). Датасеты — только `"isPrivate": true`.
## NEXT
1) Елена сабмитит v3_on; 2) итоги SATREC2 / KURG-SUBCLS → гейт до 08.11; 3) 08–11.11 сборка, check_submission, декларация ИИ.

---
# STATE KOZ3 — актуально на 01.11.2026 23:50 MSK (coord)
## ФИНАЛ (см. FINAL_CANDIDATES.md)
- «смелый» = **v3_on b447649** `koz3_v3on_b447649_6d4764f8_w6dfd2e07.zip` sha **1406c01a…ef41d** (722 697 913 B) — honest-10 0.333787 (Δ+0.0198 P0.976 vs v2clean), прод Δ+0.0194 P0.964; Kaggle parity v5 PASS во всех 4 средах, вывод побайтно = ebc8e4ba. Drive KOZ3-final/v3_on_b447649_1406c01a. Заменяет ebc8e4ba (молча откатывался на v2). На LB ещё не был.
- «надёжный» = **v2clean** sha **545ef32a…** — 0.313967, check PASS (Kaggle parity v5 PASS 5/5 сред, PREREG 8844d49, W4 9e711f6).
- Копии: Drive KOZ3-final (приватная), Kaggle aaaa0000004/koz3-w4-dfine-ref103.
## РИСК
- regex/huggingface_hub → D-FINE не импортировался → ebc8e4ba молча = v2. ЗАКРЫТО в b447649 (Kaggle v5 PASS).
## ЗАКРЫТО 01.11
J2/RR1s-cf (прод +0.0032 P0.793), CTXc (−0.0046), v4_on, lidar-aug (−0.039), r50c5 на V3P (+0.0024 P0.66), G1 sat2-kp (+0.0019 P0.94), G2 li nms8 (−0.003), large-gate, ae-dfine, v2+sat2 vs v3_on (−0.010), M_rp на V3P (+0.00276).
## NEXT
1) ✅ фикс b447649 (1406c01a) проверен и выложен в Drive; 2) Елена сабмитит v3_on b447649; 3) если LB < 0.4124 → v2clean; 4) новые рычаги только с PREREG, гейт до 08.11.

---
# STATE KOZ3 — актуально на 30.10.2026 09:55 MSK (coord)
## PASS (в финал)
- M_rp (пороги, micro_circles→rp): **+0.0044** — bots/w5-rerank c4e18a9
- ORG3 EtGr (212–217): **+0.0232** — PREREG 5ea8aea, nested-fold порог, TerFl off
- Архив v2clean sha **545ef32a…** (DOC-1 guard ok), check4 в очереди (lease bot-W4-chk3 до 21:20); прошлый 15d59e9f check_submission PASS
- Подтверждено ≈ +0.028 к A=0.314615
## REJECTED
V7 D-FINE (Бонферрони/7 FAIL), c3-sc (аддитивно и замена), WS-RR1/2, K1-5/KT, H1-5, WS-A1, V8 max_det 300, гипотеза «кроны в DEM»
## RUNNING
lihm heatmap (elenakonkova GPU, ~10:15–10:45); dino-emb; CI2-1 cloudfeat на боксе (бот 2); CI2-3 ground audit / CI2-8 hillshade parity (бот 3, Kaggle CPU)
## NEXT
r50c5: снять hold Сб 31.10 03:00 на dmitrijkv GPU; WS-G1 после dino-emb; фикс сидов курганов 12:00 (бот 3)
## OPEN
Декларация ИИ: ФИО/роли/ИИ-инструменты — от Елены (позже, до 11.11). S3 cred датасет удалён 09:40.

---
# STATE KOZ3: единый источник правды (обновлено 29.10.2026 22:55 MSK, coord / бот 1)
Правит только coord или держатель аренды `branch:bots/orch-protocol`. Потоки пишут в `orch/bot-WN/STATUS.md`, coord сводит их сюда.

## Числа
| Что | Значение | Источник |
|---|---|---|
| Лучший официальный LB | **0.4124** (29.10, v2 eb268d6) | OFFICIAL_SCORES.md |
| Предыдущий | 0.3480 (27.10, arch) | OFFICIAL_SCORES.md |
| Gate-база honest-10 A | **0.314615** | scoreboard.py --honest 10 |
| Потолки (honest) | идеальные пороги 0.367; оракул-реранкер 0.658 | REVIEW_SOTA §2.2 |
| Цель | 0.61 | Елена |
| Дедлайн / freeze | **11.11** / 10.11 23:59 | PROTOCOL v2 |
| Сдачи LB сегодня | 0 из 3 | — |

## Потоки
| Поток | Бот | Ветка / worktree | Активные аренды | Состояние | Блокеры | Следующее |
|---|---|---|---|---|---|---|
| W0/coord | бот 1 | bots/w0-rust-speed (697409e) / wt/w0; bots/orch-protocol / wt/coord-orch | branch:w0, orch-protocol | Rust-ускорение gate готово (e2e ×2.5) | — | профиль постпроцесса инференса; миграция watcher-ов на lease.sh |
| W1 aeor | бот 2 | нет ветки (orch/bot-W1, w1/) | slot acc1/gpu-1: koz3-w1-dfine-ae-final (до ~04:30) | ae proxy 0.0979, or 0.0371; брак ae-ft fold1 (1 итерация) | базового OOF-стека нет на боксе | D1: ae all-class + window-grid fix; or-final в очереди |
| W2 lidar | бот 3 | нет ветки (w2/repo — отдельный клон) | slot acc1/gpu-2: s1f23 (до 05:00); acc2/tpu: QUEUED smoke | c3-sc s0 OOF 85647 склеен; c5 закрыт | — | merge s1 → W4; center-snap п.5; heatmap п.8 после 31.10 |
| W3 bigobj | бот 4 | нет ветки (orch/bot-W3) | 3 CPU-акка: kurgan seeds (до 06:00); acc1/tpu: QUEUED smoke | PREREG H1-3/K1-3; r18/r34 фолды 0.59–0.87 | GPU до 31.10 | Fix C gor (п.9) |
| W4 gate | бот 5 | bots/w4-gate (22a32c3) / w4/repo | acc2/cpu: koz3-w4-pixgbm-s3 (до 08:00, NON-CLEANROOM research) | v3 D-FINE R−A +0.0198 по gate, но mock-e2e −0.0023 | **table.pkl / OOF-стек A на ПК (ПК offline)**; паритет PIXGBM | **п.1 паритет v2 (критический путь)**, затем рычаги 3/4/6/10/12/14 |
| W5 rerank | (назначить) | bots/w5-rerank (8a76da4) / wt/w5 | branch:w5 | V1–V4 + nested gate, PREREG, ядро dino-emb не запущено | GPU acc1 — после ae; acc2 — после 31.10 03:00 | смоук эмбеддингов, D4 03.11 |

## LB-очередь (ведёт W4 + бот 1, сдаёт только Елена)
1. (заблокировано до PASS паритета v2) v3a = v2 + рычаги п.3/4/6/10/14, принятые gate.
2. v3b = v3a + ae all-class (W1), если gate и docker согласны.

## Открытые решения
- D0: отменить 2 бесцельных QUEUED TPU-смоука (acc1 koz3-acc2-tpu-smoke, acc2 koz3-tpu-cnx-c3sc-train).
- config.py метрики отсутствует (используются дефолты) — уточнить у Елены.
- D-FINE sp/or: вне архива до согласия gate ↔ docker.
- Правило EN: только clean-room идеи.
- PR #11 (`orch` → main) дублирует bots/orch-protocol — закрыть?

## Инфраструктура
- Аренды: `orch/lease.sh` (flock). Конфликты: `orch/check_conflicts.sh --all --log` (последний прогон 22:32: ERROR 0, WARN 2 — отдельные клоны w2/repo, w4/repo).
- Rust: `rust/expds_core` в bots/w0-rust-speed; колесо в /workspace/koz3/w0/pylib (PYTHONPATH). `inference/lib/expds_core.abi3.so` архива **не** обновлён (нужна проверка glibc контейнера).
- Данные ПК: `/workspace/koz3/pc_data/` ещё не создан; ПК сейчас offline. S3 организаторов (train-expds-3) — сырые данные, OOF/table.pkl не заменяет.
