# KOZ3: протокол работы ботов (v2, 29.10.2026 22:45 MSK)
**Дедлайн: 11.11.2026** (официально 14.11, буфер не планируем). Code freeze: 10.11 23:59. Финал: 2 успешные сдачи — архив A («надёжный») и B («смелый»).
Единый источник правды: `orch/STATE.md` (ведёт coord, бот 1). Очередь: `orch/QUEUE.json` + `orch/SCHEDULE.md`. Аренды: `orch/LEASES.json` только через `orch/lease.sh`.

## 1. Потоки, владельцы, пути
| Поток | Бот | Тема | Пишет только в | Ветки | Префикс Kaggle |
|---|---|---|---|---|---|
| W0 / coord | бот 1 | оркестрация, Rust, STATE/QUEUE | `orch/` (общие файлы), `/workspace/koz3/w0/` | `bots/w0-*`, `bots/orch-protocol` | `koz3-w0-` |
| W1 | бот 2 | аэро/орто D-FINE | `/workspace/koz3/w1/`, `orch/bot-W1/` | `bots/w1-*` | `koz3-w1-` |
| W2 | бот 3 | лидар | `/workspace/koz3/w2/`, `orch/bot-W2/` | `bots/w2-*` | `koz3-w2-` |
| W3 | бот 4 | крупные объекты, подклассы курганов | `/workspace/koz3/w3/`, `orch/bot-W3/` | `bots/w3-*` | `koz3-w3-` |
| W4 | бот 5 | gate, сборка, compliance, LB-очередь | `/workspace/koz3/w4/`, `orch/bot-W4/` | `bots/w4-*` | `koz3-w4-` |
| W5 | (назначает бот 1) | реранкер DINOv2+LGBM (п.7) | `/workspace/koz3/w5/`, `orch/bot-W5/` | `bots/w5-*` | `koz3-w5-` |
Общие файлы (QUEUE.json, LEASES.json, SCHEDULE.md, STATE.md, LB_LOG, PROTOCOL.md, BRIEF_COMMON.md) правит только coord или держатель аренды `branch:bots/orch-protocol`. Остальные пишут в свой `bot-WN/STATUS.md` (время MSK | бот | что | число | ссылка).

## 2. Git: общий клон только для чтения
- `/workspace/koz3/repo` — общий клон, **всегда на main, только чтение**. `checkout`/`switch`/`commit` в нём запрещены (инцидент 29.10: два агента переключали ветки в общем клоне).
- Каждый агент работает в своём worktree: `git -C /workspace/koz3/repo worktree add /workspace/koz3/wt/<owner> -b bots/<owner>-<тема> origin/<база>`, где `<owner>` = w0…w5 / coord-orch.
- Одна ветка — один worktree. В чужие ветки не коммитить и не пушить (в т.ч. «починить заодно»), только PR/сообщение владельцу. В main не пушить никому.
- Отдельные клоны (`w2/repo`, `w4/repo`) — допустимы до 31.10, затем перейти на worktree.
- Перед каждым коммитом: `grep -rE 'KGAT_[A-Za-z0-9]{8,}|ghp_|github_pat_'` по добавляемым файлам; пул-json, s3_train.env, веса и разметку не коммитить.

## 3. Аренды (lease.sh, атомарно через flock)
```
orch/lease.sh acquire slot:acc1/gpu-1 bot-W1-aeor 8 --kernel elenakonkova/koz3-w1-x --note "..."
orch/lease.sh renew   slot:acc1/gpu-1 bot-W1-aeor 4
orch/lease.sh release slot:acc1/gpu-1 bot-W1-aeor
orch/lease.sh list | who <res> | sync   # sync = зеркало в bots/orch-protocol (только coord)
```
- Ресурсы: `slot:<acc>/<gpu-1|gpu-2|tpu|cpu>`, `kernel:<acc>/<slug>`, `dataset:<acc>/<slug>`, `branch:<name>`. TTL обязателен; истёкшая аренда считается свободной.
- **До** `kaggle kernels push` — аренда слота и ядра; после COMPLETE/ERROR — сразу `release`. Чужие RUNNING/QUEUED не трогать и не удалять. Отменять чужое может только coord.
- lease.sh отказывает, если slug ядра/датасета не с префиксом потока или ветка не `bots/w<N>-*` (legacy — только coord `--force`).
- Квоты: acc1 = elenakonkova, acc2 = dmitrijkv: GPU 30 ч + TPU 20 ч в неделю, сброс Сб 03:00 MSK (31.10, 07.11). usertest1234, aaaa0000004, aaaadg4gg4g4g — только CPU (данные — приватные датасеты, шаринг между аккаунтами разрешён Еленой).
- Watcher-ы (gpu_watcher, cpu_watcher) обязаны ходить через lease.sh (миграция — бот 1).
- `orch/check_conflicts.sh --all --log` каждые 15 мин: двойные слоты/ядра, истёкшие аренды, ядра RUNNING/QUEUED без аренды, общий клон не на main или грязный, worktree вне `wt/<owner>` или на чужой ветке, чужие авторы в ветке. ERROR → бот 1 разбирает в течение 15 мин.

## 4. Gate (принимает только W4, не автор)
- honest-10 по семьям (`scripts/scoreboard.py --honest 10`, база A = 0.314615), метрика `materials/compute_metrics.py`.
- Приём: Δ ≥ +0.003 и P(A>B) ≥ 0.9; без просадки на 34 новых LAS и на участках без лидара; **плюс** взвешенный по составу теста gate (п.12 REVIEW_SOTA: публичный и «закрытый» сценарии, оба ≥ 0).
- **Время инференса:** docker e2e на полном моке ≤ 22 мин (лимит 30), архив ≤ 2 ГБ, выход ≤ 100 МБ, ≤ 20000 объектов, ≤ 3000 вершин; один объект = один полигон внутри своей модальности, между модальностями не склеивать.
- Варианты в PREREG до замера. Новое только за флагом (по умолчанию выкл). Негатив-лист REVIEW_SOTA §2.4 не повторять.
- D-FINE sp/or вне архива, пока gate и docker mock-e2e не совпадут.

## 5. LB
- Одна очередь сдач, её ведёт W4 (бот 5) вместе с ботом 1 (раздел LB в STATE.md). Нажимает «сдать» только Елена.
- Квалификация: ≤ 10 попыток и ≤ 3 успешных в день. **Новых сдач нет, пока W4 не воспроизвёл v2 (п.1).**
- Слот 1 — контроль/кандидат, 2–3 — A/B «вкл/выкл» одной ветви, уже принятой gate. Пороги по LB не двигать (шум ±0.034). Каждая сдача — в docs/LB_LOG.md и orch/OFFICIAL_SCORES.md.

## 6. Ресурсы бокса
8 ядер, 15 ГБ RAM, без swap, общий для всех: ≤ 3 тяжёлых процесса и ≤ 8 ГБ на бота, `nice -n 10`. Тяжёлое — на Kaggle CPU. ПК Елены только чтение (сейчас offline).

## 7. Чистая комната
Конкурент EN: только идеи, без кода и весов. Реализатор PIXGBM (W4) не открывает `geo_pixgbm.py`; спецификация — `docs/PIXGBM_SPEC_CLEANROOM.md`.

## Watcher v2 (29.10 22:46 MSK) — автоподача слотов
- Один процесс `orch/watcher.py` на box (pid в `orch/.watcher.pid`, лог `orch/logs/watcher.out`), заменил gpu_watcher/cpu_watcher.
- Аренды — только через `lease.sh` (flock). Каждые 5 мин: завершённые kernels → release + pull (`pull_to`); живые с <1 ч до конца аренды → renew +2 ч; свободный слот (5 аккаунтов × gpu-1/gpu-2/tpu/cpu + box/q1,q2) → первый `ready` из `QUEUE.json["dispatch"]` с выполненным `after`.
- Каждые 15 мин: `check_conflicts.sh --all --log`.
- Ставить работу в очередь: `orch/enqueue.py add --id … --owner bot-WN-… --accel gpu|tpu|cpu|box --account elenakonkova|dmitrijkv|any-cpu --kernel-dir … [--after …] [--pull-to …]`; `enqueue.py set <id> status=hold|ready`; `enqueue.py list`. Руками `kaggle kernels push` в слоты, которые ведёт watcher, не делать.
- `cooldown` в QUEUE.json: слот на паузе до сброса квоты (acc2 GPU до 31.10 03:00); при ошибке квоты watcher ставит его сам.
