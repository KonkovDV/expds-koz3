# Кандидат v2 + FR1h — отчёт 01.11.2026 (MSK)

**Проект:** KonkovDV/expds-koz3 (KOZ №3). База LB **0.41238** = v2 `eb268d6`. Дедлайн 11.11.  
**Владелец волны:** bot-W4-gate. **LB не сабмитили** (только Елена). `main` не пушили.

## 1. Что запущено / где

| Аккаунт | Kernel | Слот | Задача | Статус | wall (ядро) |
|---|---|---|---|---|---|
| usertest1234 | `koz3-w4-fr1h-check` | CPU | `check_submission` + smoke | **COMPLETE** RC=0 | ~63 с проверок |
| aaaadg4gg4g4g | `koz3-w4-fr1h-gate` | CPU | thr verify + honest-10 vs A / B′ / v2-approx | **COMPLETE** | ~7 с |
| aaaa0000004 | `koz3-w4-fr1h-geo` | CPU | geo floors + static compliance | **COMPLETE** | ~10 с |

Датасеты: `dmitrijkv/koz3-fr1h-chk-20261101` (+ `dmitrijkv/koz3-fr1h-wheels`), READER на 3 CPU.  
Аренды: `kernel:` + `slot:` ×3 → bot-W4-gate до 01.11 23:21 MSK (после отчёта — release).  
Защищённые ядра **не трогали:** `elenakonkova/koz3-lidar-aug-f0` RUNNING, `dmitrijkv/koz3-w1-lidar-aug-f2` RUNNING; `koz3-ae-dfine-gate` не трогали.

### TPU
- `dmitrijkv/koz3-tpu-smoke` COMPLETE; `elenakonkova/koz3-acc2-tpu-smoke` QUEUED.
- Готового недублирующего TPU-джоба под FR1h/convnext (мимо уже идущих lidar-aug) **нет** → TPU **не запускали**.

## 2. Артефакт кандидата

- Zip: `/workspace/koz3/out/fr1h-pack/for_elena/submission_fr1h_on_eb268d6.zip`  
  (копия: `…/out/fr1h-pack/submission_fr1h_on_eb268d6.zip`)
- **sha256:** `fe8d60b9032474760f02d4bcfa08fb43269ae3ffb872efa932d2f1d7692944b4`
- Размер: **500.8 МиБ** (525 160 654 Б), 50 файлов; PIXGBM + `rerank_arch` **в архиве**; `[post.frame_prior]` ON (только курганы, d0=0.5, edge=0.5).
- База = archive v2 eb268d6 + overlay FR1h (release `fr1h-pack-20261031`). Пороги после re-sweep вшиты в zip.

## 3. Honest-10 и gate

Источник чисел: pack `w6c_gate_fr1h.json` (триада honest-10, re-sweep порогов по train-fold) + верификация на Kaggle CPU.

| Сравнение | base | score | Δ | P(>0) | verdict |
|---|---:|---:|---:|---:|---|
| vs A (SpOr) | 0.314615 | 0.318705 | **+0.00409** | **0.999** | **PASS** (hard) |
| vs B′ | **0.315800** | 0.318705 | **+0.002905** | **0.999** | **PASS_SOFT** (hard Δ≥0.003 — нет; soft P≥0.8 — да) |
| vs v2 offline approx | ~0.3256 (A+~0.011 PIXGBM) | 0.318705 | −0.00691 | — | не apple-to-apple: gate FR1h на A-стеке без PIXGBM; инкремент FR1h = +0.00409 на SpOr-курганах |

Дополнительно: damaged dF1 **+0.0107**; floors OK; triad vs A **PASS**.

## 4. Threshold re-sweep (в zip; verify на gate-ядре)

| group | class | archive | FR1h | verify |
|---|---|---:|---:|---|
| sp_or | kurgany_povrezhdennye | 0.3645 | 0.360843 | OK |
| sp_or | kurgany_tselye | 0.3725 | 0.336139 | OK |
| sp_focal2 | kurgany_povrezhdennye | 0.501645 | 0.37981 | OK |
| sp_focal2 | kurgany_tselye | 0.645505 | 0.572335 | OK |

`thr_ok=true`. Городища / архитектуры без изменений.

## 5. Геофизика (P/R floors 0.769 / 0.656)

- Black-box 31.10 на `pantum/n7q`, 18 geo-регионов: выход FR1h **байт-идентичен** REF_v2 (`9dcb6e71…`).
- Вердикт Kaggle geo-ядра: **PASS** (floors сохранены; полный geo re-run в этой волне не повторяли — нужны geo-rasters+image, результат уже в pack).
- PIXGBM в zip: **да** (`models/pixgbm/all/{poly,finds}.txt` + `geo_pixgbm.py`).

## 6. Compliance (`check_submission`)

Ядро `usertest1234/koz3-w4-fr1h-check`: **RC=0**, **ИТОГ PASS** (FAIL 0, WARN 1 = unshare offline), 29 проверок, wall **62.8 с**.

Ключевые PASS: layout, forbidden, size **500.8 МиБ < 2048 МиБ**, metadata image=`pantum/n7q:20261024`, все 15 весов, PIXGBM на месте, imports без сети, weights open on CPU, smoke main.py **15.5 с**, RSS peak 1119 МиБ, FeatureCollection OK.

Примечание про «≤100 МБ / без .pt в kit-zip»: это kit без моделей. Данный zip — **полный submission** с `.pt` (лимит 2 ГиБ) — **PASS**.

## 7. Готовность для Елены

| Критерий | Статус |
|---|---|
| v2-артефакты (PIXGBM + rerank) в zip | да |
| FR1h ON (kurgans-only) | да |
| thr re-sweep + verify | PASS |
| geo floors / byte-id vs v2 | PASS |
| check_submission | PASS |
| honest-10 vs A | PASS (+0.00409, P0.999) |
| honest-10 vs B′ 0.315800 | **PASS_SOFT** (+0.002905 < 0.003 hard) |

**Вердикт:** кандидат **готов к решению Елены о сабмите** (GO с оговоркой soft vs B′). Ожидаемый эффект на LB небольшой (сателлитные курганы); geo/прочие модальности как у 0.41238.  
**Сабмит не выполнялся.** Rollback: archive v2 zip.

### Путь и sha для сабмита
```
/workspace/koz3/out/fr1h-pack/for_elena/submission_fr1h_on_eb268d6.zip
sha256: fe8d60b9032474760f02d4bcfa08fb43269ae3ffb872efa932d2f1d7692944b4
```
Также: GH release `fr1h-pack-20261031`, датасет `dmitrijkv/koz3-fr1h-chk-20261101`.

## 8. Артефакты прогона
- `/workspace/koz3/out/fr1h-pack/kaggle/{check,gate,geo}/` — STATUS, result.json, логи
- Dry-run на box CPU перед push — PASS
