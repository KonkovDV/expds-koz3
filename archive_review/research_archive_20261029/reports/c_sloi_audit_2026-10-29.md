# Аудит C:\Sloi — 2026-10-29 (Europe/Moscow)

Машина: DESKTOP `08e3443c-68f7-475f-b73d-bf817fdb8138`.  
Корни: только `C:\Sloi` и `C:\Sloi Trash`. Ничего не удалялось — только `Move-Item`.  
Секреты (KAGGLE_API_TOKEN / GITHUB_TOKEN) не читались и не писались.

## 1. Свободное место и итоги

| Метрика | Значение |
|--------|----------|
| C: свободно (на момент аудита/после) | ~13.3–13.4 GB |
| C:\Sloi до переноса (этот аудит) | **~51.2 GB** (файлы; часть «размеров» orphan-wt была junction на models) |
| Уже до нас runner перенёс на F: | `v3e2e`, `lihalf`, hotfix, bot4 и др. → `F:\Sloi_pc_jobs_archive` (отсюда ~13 GB free) |
| Перенесено в `C:\Sloi Trash` этим аудитом | **~21.78 GB** (56 позиций в MANIFEST) |
| C:\Sloi сейчас (без Trash) | **~23.5 GB** (оценка по top-level после хода) |
| `C:\Sloi Trash` | **~21.78 GB**, ~19438 файлов |
| Освобождение C: от Move на том же томе | **0** (ожидаемо: rename) |

Манифест на ПК: `C:\Sloi Trash\MANIFEST.txt` (+ `MOVE_LOG.txt`).

## 2. Инвентаризация top-level (до хода, с оговорками)

Крупное на C:\Sloi до переноса:

| Имя | ~GB | Заметка |
|-----|-----|---------|
| expds3c | 12.16 | старый measure/bench — **перенесён в Trash** |
| expds-koz3_v0.2.0 | 11.58 | основной checkout + `.venv` 4.8 + models 2.1 + dist 2.7 + rust/target 1.2 |
| models_59 | 4.39 | веса fold/raw — **оставлен** |
| work59 | 2.64 | старые stack/acc — **в Trash** |
| wt_b10 / wt_geo / wt_lidump | ~2 each apparent | `models` = **junction** на main; real ~5–7 MB — **в Trash** |
| sim | 1.29 | sim done — **в Trash** |
| остальные wt_bot* | ~0.8 each | зарегистрированные worktree |
| pc_jobs | 0.63 | после архивации runner’ом крупные job dir уже на F: |

### Уже на F: (не трогали)

- `F:\Sloi_pc_jobs_archive\`: `v3e2e`, `lihalf`, `bot2_v3_hotfix_1ec2d7c`, `bot2_a2_dfine_stages_fb66c89`, `bot4_lo_mock_337a27c`, `bot5_gor_heads_4dadeab`, `bot6_pixgbm`, `bot7_diff_scope_aa3694f`
- `F:\Sloi_pc_jobs\`: активные/свежие `bot5_gor_heads_9f62a96`, `bot7_compare_candidates_b6f835d`
- `F:\Sloi_wt\`: `wt_bot2_829ea7d`, `wt_bot5_gor_heads`, `wt_bot6_3f61ea8`, `wt_bot7_compare_candidates`

## 3. Что IN USE — оставлено

### Активные mount / lease (не трогать)

- Docker bot5 gor_heads монтировал **`C:\Sloi\wt_bot5_50f6f27\models\dinov2-small`** (ro) + F: wt/jobdir — **оставлен**, путь на месте.
- JobDir/WT активных полос на **F:** (`F:\Sloi_wt`, `F:\Sloi_pc_jobs`) — вне корня хода по C:\Sloi Trash, не трогали.
- Контейнеры talent/kontur/microphoenix — не трогали.

### Обязательный keep (runners / compare / веса)

- `C:\Sloi\expds-koz3_v0.2.0\expds-koz3` — main (DIRTY: много M/?? включая untracked models_59 folds в models/)
- `.venv` runners (`...\expds-koz3\.venv\Scripts\python.exe`)
- `C:\Sloi\wt_bots` — реестр worktree (DIRTY)
- `C:\Sloi\wt_bot5_50f6f27` — dinov2 + dirty untracked model dir
- `C:\Sloi\pc_jobs\_vendor_zips\`, `_runners\`
- **`C:\Sloi\pc_jobs\bot5_v4_off_50f6f27\out\result.geojson`** — ссылка bot7 compare — **на месте (0.41 MB)**
- `models_59` — есть файлы сверх копии в main (raw_*/ens) — **оставлен**
- `files\`, `kurgan_subclass_crops.npz`, `DS26_report.pdf`
- Зарегистрированные wt на C: bot2_1ec2d7c, bot2_fb66c89, bot3_aerial, bot4_lo_337a27c, bot6_*, bot7_*, bot8_orsp_tta, wt_lihalf

### Dirty / unpushed worktrees (оставлены)

| Worktree | HEAD | Почему оставить |
|----------|------|-----------------|
| main expds-koz3 | 931be93 master | DIRTY: код + untracked веса/доки; ahead origin 0 |
| wt_bots | 931be93 | DIRTY: код/конфиги; используется `git worktree add` |
| wt_bot3_aerial | 475bafa | DIRTY: ?? pixgbm finds/poly |
| wt_bot5_50f6f27 | 50f6f27 | DIRTY: dinov2-small; **docker mount** |
| wt_bot6_d173a75 | d173a75 | DIRTY: pixgbm models + scripts; fallback в runners |
| wt_bot6_pixgbm | 70dedf4 | DIRTY: AI_DECLARATION |
| wt_bot7_base_70dedf4 | 70dedf4 | DIRTY: AI_DECLARATION |
| wt_bot7_honest_layer | aa3694f | DIRTY: AI_DECLARATION |
| wt_bot2_* / bot4 / bot8 / wt_lihalf | clean на origin | оставлены как зарегистрированные / очередь (bot8 waiting) |

Перенесён только дубль **`wt_bot2_hotfix`** (тот же SHA `1ec2d7c`, что `wt_bot2_1ec2d7c`) → Trash + `git worktree prune`.

## 4. Что перенесено в C:\Sloi Trash (группы)

| Группа | ~GB | Причина |
|--------|-----|---------|
| expds3c | 12.16 | старый локальный measure; не в lease; только stale v2prof |
| dist (submission_*.zip) | 2.73 | старые LB zip-дубли |
| work59 | 2.64 | старые stack/acc; GT также на F:\expds3\work |
| rust/expds_core/target | 1.19 | build cache, воспроизводим cargo |
| sim | 1.29 | sim_done; не в очереди |
| wt_bot2_hotfix | 0.80 | дубль worktree DONE |
| kur / rerank / xfer* / venvs / geo* / pix / orphans wt_* | ~0.9 | ad-hoc / orphan / temp |
| корневые audit/zip/md | ~0.01 | старые архивы/доки |
| пустой C:\Sloi\TRASH | 0 | слит в `C:\Sloi Trash` |

Полный список: `C:\Sloi Trash\MANIFEST.txt`.

## 5. Очередь / пути job’ов

- **Менять пути в QUEUE из‑за этого аудита не нужно.**
- Уже до нас: `C:\Sloi\pc_jobs\v3e2e` отсутствует (лежит в `F:\Sloi_pc_jobs_archive\v3e2e`). Старые команды QUEUE с `C:\Sloi\pc_jobs\v3e2e\...` устарели относительно диска; активные job’ы runner’а уже на F:.  
- Сравнительные артефакты bot7: `bot5_v4_off_...\out\result.geojson` на C: сохранён.
- Мы **не** двигали ничего из текущего docker mount и не трогали F:\Sloi_wt / F:\Sloi_pc_jobs.

## 6. Что осталось на C:\Sloi (после)

- `expds-koz3_v0.2.0` (~7.7 GB без dist/target)
- `models_59` (~4.4 GB)
- зарегистрированные `wt_bot*` + `wt_bots` + `wt_lihalf` (~8–9 GB)
- `pc_jobs` (~0.6 GB, в т.ч. vendor zip + bot5_v4_off + runners)
- `files`, `kurgan_subclass_crops.npz`, `DS26_report.pdf`

## 7. Рекомендации на удаление позже (только с согласия)

Удаление из Trash **реально** освободит место на C: (~21.8 GB). Кандидаты «безопасно после подтверждения»:

1. **Высокая уверенность:** `dist\` submission zip’ы; `rust\...\target`; orphan wt_*; корневые мелкие zip/md; xfer/xfer_venv/venvs/bench/pkg; sim (если sim больше не нужен).
2. **Средняя:** `expds3c` (12 GB) — локальные прогоны; убедиться, что ничего уникального не понадобится.
3. **Осторожно:** `work59` (есть крупные geojson и gt_* — копии GT есть на F:\expds3\work, но лучше сверить перед purge).
4. **Не удалять без отдельного решения:** всё ещё на C:\Sloi — main, `.venv`, `models_59`, dinov2 wt, `bot5_v4_off`, vendor zips, dirty worktrees.

Дополнительно (не в Trash): при нехватке места рассмотреть перенос **чистых** DONE worktree с C: на F:\Sloi_wt (после проверки dirty) — сейчас консервативно оставлены.

## 8. Технические заметки

- Перенос: PowerShell `Move-Item` same-volume.
- После хода: `git -C C:\Sloi\expds-koz3_v0.2.0\expds-koz3 worktree prune` убрал запись `wt_bot2_hotfix`.
- Скрипты аудита на ПК: `C:\Sloi\pc_jobs\_audit_sizes.ps1`, `_move_to_sloi_trash.ps1` (можно потом убрать вручную).
