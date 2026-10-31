# Бот 1 — диагностика sp_or F1=0 на mock v3 (29.10.2026 ~16:11 МСК)

| Поле | Значение |
|---|---|
| **Цель** | Почему слой `sp_or` на mock e2e v3 даёт F1=0 (было 0,2429 у v2). Только существующие выходы на ПК, без checkout/retrain. |
| **Данные** | `C:\Sloi\pc_jobs\v3e2e\out_v3fix2\` (res_v2/res_v3/gt.geojson, main_v3.log, zip_v3). Анализ в `C:\Sloi\pc_jobs\v3diag\` (`diag_spor.py` → `diag_report.json`, `diag_offset.py`). |
| **Ограничения** | CopyToBox с ПК отказан; после ~15:50 МСК Shell на machineId `08e3443c-…` стабильно `spawn EPERM`. Числа ниже — из успешных прогонов до отказа. |

## Корневая причина (наиболее вероятная)

**Механика регрессии слоя (уверенность ≈0,95):** вариант R (`[sp_dfine] drop_groups = ["sp_or","sp_focal","sp_arch","sp_focal2"]`) на 2 полностью пройденных участках **снимает рабочие U-Net объекты (−46 = все v2 sp_or)** и оставляет только D-FINE. После порога остаётся **13** `sp_or` с **soft_tp = 0/13** → F1 sp_or = 0; ALL 0,3125→0,2880.

**Почему 13 D-FINE не матчятся (уверенность ≈0,85 — не CRS-катастрофа, а промах по месту):**
- CRS выходных полигонов — **корректный EPSG:3857**, регион Гнёздово (не пиксели, не 4326, не swap осей). Bounds v3/v2/GT совпадают по порядку величин.
- Soft-match скорера (coverage≥0,5 меньшей площади + внутренняя точка; матч **по классу**): **0/13** даже при пороге coverage **0,01**.
- Медиана расстояния центроида pred→ближайший GT sp_or: **~191 м** (min≈35, max≈884). В пределах 50 м от любого GT: **1/13**. Средний сдвиг облака центроидов участка: **~347 м / ~392 м**.
- Глобальные сдвиги ±half-tile (±320 м), ±overlap (±160), ±tile **не восстанавливают** soft_tp (кроме единичного −half_xy → 1/4 на участке 103).
- Дистанция до ближайшего **рабочего** v2 pred: медиана **~285 м / ~142 м** — D-FINE не сидит на тех же курганах, что U-Net.
- Размер боксов **не причина F1=0**: v3≈v2 ≈ **0,78 м²** (display-круги ~r=0,5 м); у GT медиана ~145 м². v2 при том же размере даёт soft_tp **32/46**.
- Классы/лейблы ОКны: все 13 — `kurgany_povrezhdennye`; `id2label` модели = CLASSES в `sp_dfine.py`. Conf 0,56–0,73 при thr=0,555 → порог объясняет 482→13, но **не** нулевой матч выживших.
- Сырые 482 GeoJSON на диске не сохранены; после кластеризации/порога в выдаче только 13.

**drop_groups намерен?** Да. Вариант R в `configs/default.toml` / harvest `make_on.py`, принят bot-10: при включённом D-FINE **заменяет** U-Net SpOr, а не дополняет.

**e3c4f41 («3857» / geophys ground_metres):** к пути D-FINE **не относится** (см. также `reports/bot2_v3_regression.md`). Релевантный CRS-коммит D-FINE — **e460b49** (`crs_for=read_utm_json`).

## Числа (чек-лист)

| # | Проверка | Результат |
|---|---|---|
| 1 | CRS / extents | v3/v2/GT: EPSG:3857 Гнёздово; preds внутри/у края site bbox; нет признаков pixel/4326/axis-swap |
| 2 | Площади | v3 p50≈0,78 м² ≈ v2; GT p50≈145 м² — размер OK для soft-match при верном центре |
| 3 | IoU / soft | soft_tp 0/13; matched_at_cov@{0.01…0.75}=0; med NN GT ≈191 м |
| 4 | Scores / labels | 482 сырых → 13 после thr; conf∈[0.56,0.73]; класс kurgany_povrezhdennye; id2label OK |
| 5 | drop_groups | Намеренный вариант R; лог: −46 на 2 sites |

## Минимальный фикс (гипотеза для патч-бота)

1. **Hotfix (восстановить слой ≈v2):** в конфиге сдачи / harvest  
   `[sp_dfine] drop_groups = []`  
   **или** `enabled = false`.  
   **Да: отключение drop_groups с высокой вероятностью вернёт sp_or ~v2 (0,24):** U-Net (−46) останутся; 13 D-FINE добавят FP (precision чуть хуже, F1 слоя ≈v2 или чуть ниже). Проверить, что ALL не падает >0,01 из‑за FP.

2. **Safety valve в коде** (`expds/pipeline.py` `_sp_dfine_phase`): не вызывать `drop_groups`, пока нет доказанных TP / пока `n_dfine` после порога ≪ ожидаемого (план бота 2 §5 commit 1).

3. **Настоящий фикс D-FINE (если оставлять R):**  
   - дамп сырых xyxy + центров 3857 vs GT (`EXPDS_DFINE_DUMP`);  
   - сверка `UTM.json` mock и CRS WarpedVRT (гард geographic EPSG:4326 = `0037c0e`);  
   - перекалибровка `[thresholds_by_markup.sp_dfine]` **OOF**, не in-sample;  
   - зона кода: `expds/sp_dfine.py` `detect_scene` / `boxes_to_ellipses` / `utm_for` + конфиг `drop_groups` / thresholds.

## Артефакты на ПК

- `C:\Sloi\pc_jobs\v3diag\diag_report.json`, `diag_spor.py`, `diag_offset.py`, копии `sp_dfine.py`/`pipeline.py`/`default.toml` из zip_v3.
- Повторный Shell на ПК после ~15:50 МСК — EPERM (числа не переснимались).

