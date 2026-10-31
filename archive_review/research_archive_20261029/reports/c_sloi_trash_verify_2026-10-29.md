# Доскональная перепроверка C:\Sloi\TRASH — 2026-10-29 (Europe/Moscow)

Машина: DESKTOP `08e3443c-68f7-475f-b73d-bf817fdb8138`.  
Режим: **только чтение** (ничего не удалялось, не переносилось, не правилось).  
Секреты не читались и не печатались. Контейнеры talent / kontur / microphoenix не трогались (inspect только имён; mounts проверены у остальных).  
Манифест: `C:\Sloi\TRASH\MANIFEST_2026-10-29.txt`.  
Свободно на C: на момент проверки ≈ **9.46 GB**.

Проект: `KonkovDV/expds-koz3` (main checkout `C:\Sloi\expds-koz3_v0.2.0\expds-koz3`, HEAD `931be93` master, behind origin 13, dirty). Worktree’ы активны на C:\Sloi\wt_* и F:\Sloi_wt. GitHub: private; **релизов нет**; тегов `v0.2.0` нет (только `archive/*`).

---

## Сводка вердиктов

| Папка | Размер | Вердикт | Можно освободить сейчас |
|-------|--------|---------|-------------------------|
| `expds3c` | 12.16 GB | **KEEP часть** → остальное SAFE | ≈ **11.0 GB** после сохранения 2 файлов |
| `work59` | 2.63 GB | **KEEP** (уникальные стеки/кривые) | **0** (пока нет копии на F:) |
| `sim` | 1.29 GB | **KEEP zip** → остальное SAFE | ≈ **0.79 GB** |
| `dist` | 2.73 GB | **KEEP** (единственные копии LB-архивов) | **0** |
| `rust/.../target` | 1.19 GB | **SAFE TO DELETE** | **1.19 GB** |
| **Итого безопасно сразу** | | | **≈ 13.0 GB** |
| **Итого после бэкапа уникального на F:** | | | **≈ 20.0 GB** (все 5 папок) |

Консервативно: **нельзя** просто снести все пять папок. Сразу безопасно: `target` целиком + большая часть `expds3c` + модели/`wt_sim` в `sim`.  
`dist` и крупные geojson из `work59` / `expds3c\in\stack_base.geojson` / `sim\submission_sim_pkg7tw.zip` — **сначала скопировать на F:**, потом удалять.

---

## 1. `C:\Sloi\TRASH\expds3c` (12.16 GB)

### Что это
Локальный стенд measure/bench bot-7 (не git-репозиторий, не worktree main).

| Подпапка | ≈GB | Содержание |
|----------|-----|------------|
| `v2meas` | 5.01 | Кривые/jsonl семейного measure v2 |
| `regions` | 3.42 | 1× GeoTIFF участка M047 + UTM.json |
| `in` | 2.20 | Входные стеки/GT geojson |
| `wt_bot7` | 0.80 | Снимок дерева репо **без** `.git` + models |
| `bot7v2` | 0.49 | Скрипты measure v2 + `submission_bot7cand_v2_eb268d6.zip` |
| `pkg` | 0.24 | Кривые pkg/crop |
| `bot7` | ~0 | mbox, SHA256SUMS, ps1 |

Даты файлов: 20.10–29.10.2026 (последняя запись ~08:05 29.10 — v2meas/peak).  
Git: **нет** (ни корня, ни `wt_bot7`). Не зарегистрирован в `git worktree list` main.

### Уникальный / воспроизводимый контент

| Файл | Размер | SHA256 (полный) | Копия вне TRASH? |
|------|--------|-----------------|------------------|
| `regions\...\*.tif` (M047 Or) | 3504.95 MB | `B4BF7913228E459F0B2C47489E847BC3E30FFF09BD96CE1DFE231FE1E5ED6F56` | **ДА** — hash= у `F:\expds3\mock_val\region_M047\...tif`; тот же размер у `F:\2\train-expds-3\056_*` |
| `in\stack_verify_sp_arch.geojson` | 1117.66 MB | `5816328CE3F253189DCFDD97A946B0F875A737E98A17EE5AB864A98461777169` | **ДА** — hash= `F:\expds3\work\stack_verify_sp_arch.geojson` |
| `in\stack_base.geojson` | 1079.73 MB | `B412E29934DD292BCC343627690CD246FD9048948A85E8A00181C86417F8E253` | **НЕТ точной** — на F: есть `stack_base.geojson` **другой** (1191943434 байт, 26.10). SHA из `bot7\SHA256SUMS` = именно trash-версия |
| `in\gt_train_v2.geojson` | 37.58 MB | `D4BC1A3B01350C9B3785155DC4D7C13403FDFE7A098664BCB259F96BC168FF86` | **ДА** — hash= `F:\expds3\work\gt_train_v2.geojson` |
| `in\pix_lofo.geojson` | 20.77 MB | `EFEC055209E45DC6A1F17E30F88CD6513517FFEA0657E5EA3C7D6C2ADB0AC650` | **НЕТ** (есть только `pix_lofo.stats.json` в docs wt) |
| `bot7v2\submission_bot7cand_v2_eb268d6.zip` | 500.83 MB | `8F3F267107CE1BF5AC2C0B2D15346749521B0CE9C809AC4F339649085422D589` | **ДА** — hash= `F:\Sloi_pc_jobs_archive\v3e2e\kit\...` |
| `wt_bot7\models\*.pt` (выборочно hard2) | ~55 MB×N | `0311E877…` и др. | **ДА** — в `C:\Sloi\expds-koz3_v0.2.0\expds-koz3\models\` |
| `v2meas\**\*.pkl`, jsonl | ~5 GB | — | **НЕТ копий**; воспроизводимы долгим measure |

### Ссылки
- Активные `pc/QUEUE.md`, `pc/runners/*.ps1`, leases, docker mounts (**koz3-bot2-a3** и др.): **не ссылаются** на `expds3c` / TRASH.
- Устаревшее: `C:\Sloi\pc_jobs\v2prof\sim_eb.ps1` → `C:\Sloi\expds3c\bot7v2\...` и `C:\Sloi\sim\wt_sim` (пути до переноса; задача не в активной очереди).
- В analysis worktree’ов (`F:\Sloi_wt\wt_bot6_3f61ea8\...pc_measure_*.ps1`) исторические пути `C:\expds3c\...` (без `\Sloi`) — документация, не runtime.
- shared/koz3: только упоминания в `reports/c_sloi_audit_2026-10-29.md`.

### Вердикт: **KEEP (2 файла) / остальное SAFE TO DELETE**
Оставить до копирования на F: (или оставить на C:):
1. `in\stack_base.geojson` (~1.08 GB) — канон measure по SHA256SUMS, отличается от F:
2. `in\pix_lofo.geojson` (~21 MB) — единственная копия

После этого удаление остального **~11.0 GB** безопасно.  
`v2meas`/`pkg` — артефакты прогона; в очереди не нужны; при сомнении можно сначала заархивировать на F: (~5 GB).

---

## 2. `C:\Sloi\TRASH\work59` (2.63 GB)

### Что это
Рабочая папка экспериментов 28.10 (стеки fold-59, micro/imnet, acc_s4). **Не git.**

Крупное:
- `stack_s4_imnet.geojson` 636.51 MB — SHA `0E406425B52618B8473318662916E375AB4564D69E2B3111A3B495B3FCEA3948`
- `stack_base_59.geojson` 616.91 MB — SHA `452B7D8C5B46D5B0FF1A77BFD3DA9F99E1EDE6A6D9F11D723C04F1CCB79CD4EB`
- `acc_s4\` ≈ 1.29 GB (jsonl/pkl кривые base vs cand)
- `surf_arch_sp_micro.geojson` 70.92 MB — SHA `169EC3E9D254BBC6D96B855BAF0F48730BDB6A44178B6F73166D90A4CDC60F90`
- `gt_train_v2.geojson` 37.58 MB — тот же SHA, что на F: (`D4BC1A3B…`)
- `gt_raw.geojson` 17.70 MB — SHA `8B3D0FC870942CD2AA80A946E5D134FFE3BA46AB7F74FB68A02DFE1D5C66DFBA` — **копий нет**

Даты: 22.10–29.10 (pix_done.flag 29.10 00:10).

### Уникальность
Поиск по `C:\Sloi` (без TRASH), `F:\expds3`, `F:\2`, `F:\Sloi_*`:  
**нет копий** `stack_s4_imnet`, `stack_base_59`, `surf_arch_sp_micro`, `gt_raw`, `pix_lofo`.  
На F: есть `surf_arch_sp.geojson` (другое имя/файл), не micro.

Весов `.pt` нет. Это выходные стеки/оценки — теоретически пересобираемы из моделей, но **дорого и без гарантии бит-в-бит**.

### Ссылки
Нет в активной QUEUE/runners/docker. Audit помечал «осторожно».  
`gt_train_v2` дублируется на F:; остальное — нет.

### Вердикт: **KEEP**
Пока нет бэкапа на F: — **не удалять**. Минимум сохранить (~1.35 GB файлов + лучше весь `acc_s4` 1.29 GB):
- `stack_s4_imnet.geojson`, `stack_base_59.geojson`, `surf_arch_sp_micro.geojson`, `gt_raw.geojson`, `acc_s4\`  
Рекомендация: скопировать всю папку на `F:\expds3\work\work59_backup_2026-10-29\`, затем можно удалить с C: (**+2.63 GB**).

---

## 3. `C:\Sloi\TRASH\sim` (1.29 GB)

### Что это
Прогон platform sim v2 (флаг `sim_done.flag` 29.10 00:21). **Не git.**  
`wt_sim` — снимок дерева репо без `.git` + models; `platform_sim_v2\result.geojson`; `submission_sim_pkg7tw.zip`.

### Уникальность
| Файл | Размер | Копия? |
|------|--------|--------|
| `submission_sim_pkg7tw.zip` | 498.98 MB, SHA `AD330D338FCE5B86BBBF596905839B71C14CD06A4CE1016636321F5DC526C2F1` | **НЕТ** нигде вне TRASH |
| `wt_sim\models\*.pt` | ~0.79 GB | **ДА** — размеры/hash совпали с main `models\` (li_r18, surf_arch, sp_crop, hard2, …); `rerank_arch_lgbm.*` есть в wt_bot* / F:\Sloi_wt |
| `platform_sim_v2\result.geojson` | 3.69 MB | локальный результат сима; не в QUEUE |
| CSV/логи | мелочь | симуляция завершена |

### Ссылки
Только stale `v2prof\sim_eb.ps1`. Docker/QUEUE — нет.

### Вердикт: **KEEP zip / остальное SAFE**
- Оставить: `submission_sim_pkg7tw.zip` (~499 MB) — единственная копия sim-пакета.  
- Удалить можно: `wt_sim` (~0.80 GB), `platform_sim_v2`, логи/скрипты.  
Итого сразу ≈ **0.79 GB**. После копии zip на F: — вся папка SAFE (**1.29 GB**).

---

## 4. `...\expds-koz3\dist` (2.73 GB)

### Что это
Ровно **6 submission zip** (не wheels/sdist):

| Файл | MB | SHA256 | Заметка из docs/LB_LOG |
|------|-----|--------|------------------------|
| `submission_crop.zip` | 498.75 | `BD52AD71A1FBA6782B304612D7A63E13A277308CD94A839BAA3C38FCC5B3B774` | собран, на LB не грузили (verify crop) |
| `submission_arch.zip` | 459.12 | `0D1837A27BB10CE1C1D9515FA60DB2DBE126AD0FF0062D53A0258A7A21EC920E` | **публичный LB 0,3479576040** (27.10) |
| `submission_density15.zip` | 459.02 | `B367F402D1A96D9009A44C9FA839769674DEA1FEB50CF7E46D6B1F8931FE055D` | собран, не грузили |
| `census_finds.zip` | 459.02 | `5B27E102A6AB6B2CC615E90EDA2A01002310A7A56DAF72C6C4DC5FF3B175B571` | only_classes finds; не грузили |
| `submission_hillfort.zip` | 459.02 | `D7B25A6974320502DE502FD311AEA414197A59657BBD490EA58C109F4F818F4E` | LB 0,33352 |
| `submission.zip` | 458.97 | `C1A3F87B9FE53A402467A671A84C65EE0BC69ADE139D24C7B4D9984EA5F821EF` | ранний |

В main сейчас **`dist\` отсутствует** (перенесён в TRASH).  
На F:\expds3\work\lb_* — только `scoring.log`, **без zip**.  
Корневые zip на C:\Sloi — **другие размеры**, не дубли.  
GitHub Releases: **пусто**. Тега `v0.2.0` нет.

Воспроизводимость: теоретически через `scripts/build_submission.py` + models в main, но **единственные бинарные копии с зафиксированным SHA — здесь**. Для аудита LB это критично.

`.venv` / runners **не** ставят пакеты из этого `dist` (там нет `.whl`; только submission zip).  
`expds_core` берётся из `.venv\...\expds_core.pyd` и `inference\lib\expds_core.abi3.so`, не из dist.

### Ссылки
Docs (`LB_LOG.md`, `PLAN.md`, `ARCHITECTURE.md`, `AI_WORK_PLAN`) активно ссылаются на `dist\submission_*.zip` и SHA.  
Активные runners на эти пути не монтируют.

### Вердикт: **KEEP**
Не удалять, пока не скопированы на F: (например `F:\expds3\work\lb_submission_zips_2026-10-29\`) с проверкой SHA256. После бэкапа — SAFE (**2.73 GB**).

---

## 5. `...\rust\expds_core\target` (1.19 GB)

### Что это
Классический Cargo build cache: `debug\`, `release\`, `maturin\`, incremental, deps `.rlib`/`.pdb`/макросы.  
Даты: 20–22.10.2026.

Артефакт `release\expds_core.dll` / `deps\expds_core.dll`:  
SHA256 `13B1011FE9A14A40…`, 1126400 байт — **бит-в-бит =**  
`C:\Sloi\expds-koz3_v0.2.0\expds-koz3\.venv\Lib\site-packages\expds_core\expds_core.pyd`.

Также Linux `expds_core.abi3.so` (1.38 MB) уже в:
- main `inference\lib\`
- wt_bots, F:\Sloi_wt\*, zip job’ов на F:

В main `rust\expds_core\target` **уже нет** (перенесён). Pip/maturin из этого target runners не читают.

### Вердикт: **SAFE TO DELETE**
Восстанавливается `cargo build` / maturin. Освобождает **1.19 GB**. Удаление не ломает `.venv` и inference.

---

## Docker / очереди / runners

- Активный контейнер `koz3-bot2-a3-*`: mounts только `F:\Sloi_pc_jobs\...` и `F:\Sloi_pc_jobs_archive\v3e2e\regions` — **не TRASH**.
- Ни один mount любого проверенного контейнера не указывает на `expds3c` / `work59` / `sim` / `dist` / `target` / `TRASH`.
- `pc/QUEUE.md`, `STATUS_LIVE`, shared runners: **нет** рабочих ссылок на эти пять путей.
- Единственные живые текстовые ссылки — audit-отчёт и устаревший `pc_jobs\v2prof\sim_eb.ps1`.

---

## Рекомендуемый порядок действий (только с согласия пользователя)

1. **Сейчас удалить (SAFE):** `...\rust\expds_core\target` → **+1.19 GB**.
2. **Скопировать на F: затем удалить:**  
   - весь `dist\` (2.73 GB)  
   - весь `work59\` (2.63 GB)  
   - `sim\submission_sim_pkg7tw.zip`  
   - `expds3c\in\stack_base.geojson` + `pix_lofo.geojson`
3. **После п.2 удалить:** остаток `expds3c`, остаток `sim` → суммарно с п.1 ≈ **20 GB** с C:.
4. По желанию поправить/удалить stale `C:\Sloi\pc_jobs\v2prof\sim_eb.ps1` (не блокирует удаление TRASH).

**Не трогать** без отдельного решения: main `expds-koz3_v0.2.0` (без dist/target), `.venv`, `models_59`, активные wt_*, `pc_jobs` на C:/F:, docker talent/kontur/microphoenix.

---

## Хеши ключевых файлов (для сверки после бэкапа)

```
5816328CE3F253189DCFDD97A946B0F875A737E98A17EE5AB864A98461777169  stack_verify_sp_arch.geojson
B412E29934DD292BCC343627690CD246FD9048948A85E8A00181C86417F8E253  expds3c/in/stack_base.geojson (UNIQUE vs F)
D4BC1A3B01350C9B3785155DC4D7C13403FDFE7A098664BCB259F96BC168FF86  gt_train_v2.geojson
EFEC055209E45DC6A1F17E30F88CD6513517FFEA0657E5EA3C7D6C2ADB0AC650  pix_lofo.geojson (UNIQUE)
8F3F267107CE1BF5AC2C0B2D15346749521B0CE9C809AC4F339649085422D589  submission_bot7cand_v2_eb268d6.zip
0E406425B52618B8473318662916E375AB4564D69E2B3111A3B495B3FCEA3948  stack_s4_imnet.geojson (UNIQUE)
452B7D8C5B46D5B0FF1A77BFD3DA9F99E1EDE6A6D9F11D723C04F1CCB79CD4EB  stack_base_59.geojson (UNIQUE)
169EC3E9D254BBC6D96B855BAF0F48730BDB6A44178B6F73166D90A4CDC60F90  surf_arch_sp_micro.geojson (UNIQUE)
8B3D0FC870942CD2AA80A946E5D134FFE3BA46AB7F74FB68A02DFE1D5C66DFBA  gt_raw.geojson (UNIQUE)
AD330D338FCE5B86BBBF596905839B71C14CD06A4CE1016636321F5DC526C2F1  submission_sim_pkg7tw.zip (UNIQUE)
BD52AD71A1FBA6782B304612D7A63E13A277308CD94A839BAA3C38FCC5B3B774  submission_crop.zip (sole)
0D1837A27BB10CE1C1D9515FA60DB2DBE126AD0FF0062D53A0258A7A21EC920E  submission_arch.zip (sole, LB)
B367F402D1A96D9009A44C9FA839769674DEA1FEB50CF7E46D6B1F8931FE055D  submission_density15.zip (sole)
5B27E102A6AB6B2CC615E90EDA2A01002310A7A56DAF72C6C4DC5FF3B175B571  census_finds.zip (sole)
D7B25A6974320502DE502FD311AEA414197A59657BBD490EA58C109F4F818F4E  submission_hillfort.zip (sole)
C1A3F87B9FE53A402467A671A84C65EE0BC69ADE139D24C7B4D9984EA5F821EF  submission.zip (sole)
B4BF7913228E459F0B2C47489E847BC3E30FFF09BD96CE1DFE231FE1E5ED6F56  regions M047 tif (= mock_val)
13B1011FE9A14A40… (1126400 B)  target release expds_core.dll = .venv expds_core.pyd
```

Отчёт подготовлен read-only агентом-верификатором 29.10.2026 ~18:30–18:40 Europe/Moscow.
