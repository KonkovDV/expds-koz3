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
