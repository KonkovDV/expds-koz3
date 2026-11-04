# bot8 — GPU-5 OOF-инференс honest-10 (Kaggle T4)

**Цель:** получить OOF-предсказания всех моделей из `[models]` на honest-10 для калибровки порогов по слою (scoreboard `--honest 10`, density_thresholds, oof_transform).
**Гипотеза:** веса-режим (`nn_oof --weights` / predict-only / resume) даёт стабильные `*_oof.geojson` имён consumer'а без переобучения; resumable STATUS/sha256 переживает обрыв T4.
**Критерий приёмки:** полный набор `oof/*_oof.geojson` + `STATUS.txt=DONE` + `SHA256SUMS`; CPU smoke green; `box/run_tests.sh` green. Честный Δ порогов — после прогона (бот 7 / ПК), не в этом PR.
**Бюджет:** Kaggle GPU T4 ~8–12 ч (не ПК GPU); CPU box ≤ 0.1 ч на smoke/гейт. **Push/run Kaggle — QUEUED-AFTER-RESET** (31.10 03:00 МСК); staged `/workspace/kaggle/push/koz3-bot8-oof-honest10`.
**Ветка:** `bots/bot8-oof-kaggle` @ **`a6f8992`** (база main `70dedf4`). Worktree: `/workspace/koz3-wt/bot8-oof-kaggle`. Draft PR: https://github.com/KonkovDV/expds-koz3/pull/7

Обновлено: 29.10.2026 16:23 МСК (Europe/Moscow).

---

## Статус

| Ворота | Результат |
|---|---|
| `pytest tests/test_oof_kaggle_smoke.py` | **PASS** 5 passed |
| `box/run_tests.sh bots/bot8-oof-kaggle` | **PASS** 671 passed / 43 skipped @ `a6f8992` → `box/results/bots-bot8-oof-kaggle_a6f8992.*` |
| Kaggle push/run | **QUEUED-AFTER-RESET** 31.10 03:00 МСК — `kaggle/QUEUE.md` (8–12ч не влезают в remaining) |
| honest-10 + пороги | pending (после COMPLETE ядра) |

## Сделано в коммите `a6f8992`

1. `scripts/oof_kaggle_jobs.py` — каталог jobs (`spor_focal`, `spor_gor`, mask/lidar/ae/…), STATUS/sha256 skip, `run_smoke`.
2. `kaggle/2026-10-29/oof_honest10/` — `koz3-bot8-oof-honest10.py`, копия lib, `kernel-metadata.json` (NvidiaTeslaT4, `dmitrijkv/koz3-s3-cred`), README.
3. `tests/test_oof_kaggle_smoke.py` — CPU smoke без torch/S3/GPU; `OOF_SMOKE=1`.

## Compliance hotfix (смежная ветка)

Бот 6: `run_tests` exit 4 из‑за явного пути `tests/test_check_compliance.py` на ветке без файла (stale name; официальный `box/run_tests.sh` сейчас путь **не** передаёт). На `bots/bot8-orsp-tta` @ **`35ca3df`**: module-level `pytest.skip` если нет `scripts/check_compliance.py` — тест самодостаточен, чужие ветки не роняет. Патч синхронизирован в `bot8_patches/`. Shared runner не правили.

## Следующий шаг

Квота-агент: `kaggle kernels push` по QUEUE; после COMPLETE — забрать geojson → scoreboard honest-10 / калибровка порогов (бот 7, ПК CPU). Bot8 idle до результатов ядра или следующей строки BACKLOG.
