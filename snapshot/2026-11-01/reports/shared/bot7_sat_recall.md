# bot-7 — satellite recall levers (sp_or / spor / sat)

Обновлено: 01.11.2026 ~20:00 МСК (Europe/Moscow).

## Контракт
- **Цель:** 2–3 рычага recall спутника при приемлемом P; флаги OFF; suite без новых fails; draft PR; пакет Kaggle без push.
- **Исключения:** NOT sat2 (bot2/W1), NOT CLOSED `sp_large` / sat 8m cluster (`sp_score_fusion` r=8) / near_kept / peak-rerank / WSR / r50c5 / dino-gor, NOT large_objects (bot4).
- **Ветка:** `bots/bot7-sat-recall` (от main `f127db8`).
- **Gate:** honest-10 vs B′ **0.315800**, Δ≥0.003, P≥0.9, damaged не хуже.

## Диагноз (LB + mock FN)

| источник | числа |
|---|---|
| LB type спутник (`scoring_v2_eb268d6_0.41238`) | P **0.456** R **0.255** → GT≈**377** TP≈**96** FP≈**229** FN≈**281** |
| s_sat proxy (lb-breakdown) | **0.1297**; ceiling R→0.6 → **+0.025** proxy / **+0.049** clsmap |
| break-even hit новых dets | **8.9%** |
| mock FN (`fn_sat_mock.json`, 83 GT / 67 FN) | loc 2–10м **14**; loc 10–50м **31**; far **2**; **undetected >200м 19** |
| bot2 `sat_thr_relax` | уже на main OFF; scale 0.85 E[ΔLB] +0.003…0.010 — не переоткрываем |
| bot8 orsp/TTA | код OFF; PC honest-10 **pending** → TTA/MS **не** первичный рычаг |
| D-FINE SpOr | локализация сломана (bot2) — не для recall |

## Рычаги (ранг по E[ΔLB])

| # | Флаг | Суть | E[ΔLB] proxy | Риск |
|---|---|---|---|---|
| **L1** | `[sat_band_verify]` | thr×0.90 mid-band только при `verify_p`≥0.40 | **+0.004…0.008** (prior mid **+0.005**) | без verify_p mid-band reject |
| **L2** | `[sat_agree_accept]` | thr×0.85 если партнёр sp_focal↔sp_focal2 в **16 м** (≠ CLOSED 8 м) | **+0.002…0.005** (prior **+0.003**) | мало объёма |
| **L3** | `[sat_cross_nms]` | NMS 4 м между sat-группами, max conf | **+0.001…0.003** (prior **+0.0015**) | режет дубли FP → P↑ |
| ALL | L1+L2+L3 | | prior **+0.007** | нужен honest-10 |

`radius_m=8.0` в agree → **ValueError** (защита от CLOSED 8m cluster).

## Код
- `expds/sat_recall.py` — L1/L2/L3
- `expds/watchdog.py` — finish/safe_finish hook при `any_enabled`
- `configs/default.toml` — три секции `enabled=false`
- `tests/test_sat_recall.py` — 11 passed

## Suite
`pytest tests/` @ tip: **1007 passed / 65 skipped / 2 failed** (`test_bot7_dfine_import` — prefails как geo-gate; **0 new**).

## Пакет Kaggle (не пушить)
- Путь: `/workspace/kaggle/push/sat-recall/`
- Slug: `dmitrijkv/koz3-bot7-sat-recall`
- Helper: `common/koz3_s3.py` (reuse)
- Push (bot1): `orch/pool_kaggle.sh dmitrijkv kernels push -p /workspace/kaggle/push/sat-recall`
- Secrets: `KOZ3_S3_ACCESS_KEY_ID` / `KOZ3_S3_SECRET_ACCESS_KEY` (+ dataset `dmitrijkv/koz3-s3-cred`)
- Runtime: smoke <1 с; S3 pull ~10–40 мин CPU; full re-infer TBD (`SAT_RECALL_REINFER=1`)

## PR / SHA
- Ветка: `bots/bot7-sat-recall`
- Tip: `3adc7f461c227f66f2b13a143142507ab9940d3a`
- DRAFT PR: https://github.com/KonkovDV/expds-koz3/pull/27

