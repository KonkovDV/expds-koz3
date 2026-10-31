# Бот 2 — регрессия sp_or v3 (mock e2e) — 29.10.2026 ~16:14 МСК

| Поле | Значение |
|---|---|
| **Цель** | Вернуть слой sp_or на mock e2e v3 к уровню v2 после регрессии 0,2429→0 (вариант R + 0 TP у D-FINE). |
| **Гипотеза** | Механика слоя — R (`drop_groups`) снимает рабочие U-Net (−46); 13 D-FINE — FP с верным CRS, но промахом по месту (~191 м). Hotfix: `drop_groups=[]` (A) или `enabled=false` (B). Настоящий фикс D-FINE (localization / OOF thr) — отдельно, не в этой ветке. |
| **Критерий приёмки** | Mock e2e: ALL F1 ≥ **0,3125 − 0,01** и ни один слой не падает >0,01 vs v2; sp_or ≈ **0,2429** (±0,01). Предпочтение A; если A роняет ALL >0,01 — B. |
| **Бюджет** | Код+тесты: ~1 ч CPU box. Mock e2e A+B: ≤15 мин GPU ×2 (GTX 1650). |
| **Ветка** | `bots/bot2-e2e-v3-prep` @ **`fb66c89`** (hotfix tip `1ec2d7c` + stages/A≡B/geo-guard). |

Связанные отчёты: `reports/bot1_v3_spor_diag.md` (диагноз ПК: **не CRS**). Заявка ПК: `pc/QUEUE.md` «mock e2e v3 hotfix A/B».

---

## 1. Итог диагноза (bot1 + bot2)

| Факт | Значение |
|---|---|
| Механика регрессии | R −46 v2 SpOr; после thr 13 `kurgany_povrezhdennye` conf 0,56–0,73; soft_tp **0/13** |
| CRS | **OK** EPSG:3857; размеры ≈ v2 (~0,78 м²); half-tile сдвиги не лечат |
| Локализация | медиана pred→GT ~**191 м**; 1/13 <50 м; v2 при тех же размерах 32/46 |
| e3c4f41 / 0037c0e | к D-FINE path **не** относится; порт geographic UTM **не делаем** |

---

## 2. Коммиты на `bots/bot2-e2e-v3-prep`

| SHA | Что |
|---|---|
| `5614d0db1b5b915e5899e907627ad555123b9114` | **Safety valve:** `_sp_dfine_phase` — `drop_groups` только на участках с ≥`drop_min_det` (default **1**, багфикс ВКЛ.) объектов `sp_dfine`; иначе v2 сохраняются. |
| `daeeed99ab9d265305e6d25a5bcb42e8c2b3dfa9` | **Harvest hotfix A:** `make_on.py` дефолт **MODE=C** (без R); overlay `pipeline.py`; MODE=R — явный opt-in. |
| `8eacb68e0b0cf1ca1a3201f418aea430df1d129a` | **Dump (OFF by default):** `EXPDS_DFINE_DUMP` → JSON pixel+3857 + GT GeoJSON. |
| `1ec2d7cec6c66066a2ef88cae9080cbdf7e163df` | **Docs hotfix A/B** в `configs/default.toml`: A=`enabled`+`drop_groups=[]`; B=`enabled=false`; R запрещён до localization. |

Tip ветки = `1ec2d7c`.

**Не сделано (намеренно):** порт `0037c0e`; OOF-перекалибровка порогов; правки `detect_scene` / `boxes_to_ellipses` / `utm_for` под localization; PR / LB.

---

## 3. Тесты (box, 29.10 ~16:14 МСК)

```
pytest tests/test_bot2_dfine_dump.py tests/test_bot2_make_on_modes.py \
  tests/test_bot7_v3_flags.py::test_pipeline_phase_drop_skips_regions_without_dfine \
  tests/test_bot7_v3_flags.py::test_pipeline_phase_drop_min_det_zero_allows_empty \
  tests/test_bot7_v3_flags.py::test_pipeline_phase_drop_only_complete \
  tests/test_bot7_v3_flags.py::test_default_flags_off
→ 10 passed
```

`/home/box/shared/koz3/box/run_tests.sh` — узкий набор bot8; для этой ветки использован точечный pytest в `/workspace/koz3-venv`.  
`test_flag_modules_not_packaged_when_off` / collect pixgbm — падает на ветке **nomodels** (нет `models/pixgbm/...`) — вне scope.

---

## 4. ПК: заявка A/B

См. **`pc/QUEUE.md`**: «[bot2] 29.10 16:14 МСК — mock e2e v3 hotfix A/B».

- **A:** zip `make_on … C` + опционально `EXPDS_DFINE_DUMP=…/dump_A`
- **B:** `enabled=false` (flags-off той же ветки)
- Критерий: ALL ≥ 0,3025 и без провала слоя >0,01; sp_or ≈ 0,2429; предпочесть A.

EXPERIMENTS.csv: строка `1ec2d7c` / `hotfix_A_drop_groups_empty+…` / verdict **pending**.

---

## 5. Позже (BACKLOG)

Настоящий фикс D-FINE SpOr (после приёмки A/B):

1. OOF-калибровка `[thresholds_by_markup.sp_dfine]` (не in-sample R_max).
2. Разбор сдвига ~191 м: `detect_scene` / `boxes_to_ellipses` / кластеризация C (`radius_m`) — **не** CRS-порт 0037c0e.
3. Пока soft_tp=0 на mock — **не** возвращать вариант R.

Пункт внесен в `BACKLOG.md` (Kaggle GPU §4 и «Боты → 2»).


---

## 5. A≡B и localization (29.10 ~16:41 МСК)

PC A/B @ 1ec2d7c: **A ≡ B** ALL 0.312873 / sp_or 0.245064. Принято: D-FINE не в scored output (не «FP без эффекта на F1»).
Коммит `fb66c89`: stages dump, синтетические тесты пути, geographic CRS guard, localization audit (math OK / ~191 м = domain).
Отчёт: `reports/bot2_dfine_localization.md`. Hotfix v4 = B. PC next: A2 dump+stages.
