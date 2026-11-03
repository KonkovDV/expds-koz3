# bot-3 — recall аэрофото (LB ae R≈0,013)

Обновлено: 29.10.2026 16:06 МСК (Europe/Moscow).

## Контракт (PROTOCOL §1)
- **Цель:** поднять recall слоя ae без регрессии остальных слоёв; default поведение байт-в-байт.
- **Гипотеза:** главный провал — слишком жёсткие пороги ae_fort / li_ae / fusion + отсутствие ae-курганного детектора на участках без лидара; D-FINE Ae и raster_fix уже есть как OFF-флаги на базе v4.
- **Критерий приёмки:** honest-10 (`scripts/eval_aerial_honest10.py`): ΔF1_ALL ≥ +0,002 и p_gt0≥0,8; ни один слой не падает >0,01; ae R и/или F1 растут vs база ветки. Пока **не пройдено** (ждёт прогон на ПК).
- **Бюджет:** ПК CPU/GPU mock+honest ~15–40 мин; обучение не требуется для порогового эксперимента. Следующий шаг после приёмки порогов — Kaggle D-FINE Ae (BACKLOG GPU-3).
- **Ветка:** `bots/bot3-aerial-recall` (от **cbd4804**, не main — слот `[dfine_aeor]` / `expds/dfine_aeor.py` есть только на v4/cbd4804).

## Ветка / коммиты
- Base: `cbd48049ceaa6b56d0e4b85c72f4d35e839b2d3c` (bots/bot5-candidate-v4 tip)
- Tip: `475bafad60504cd6058366f522fd9cd178a93b6a`
- Коммиты:
  - `e6ee68f` — cherry-pick bot6 `[raster_fix]` (код в, секция в TOML закомментирована = OFF)
  - `6cdceb6` — `[ae_recall_relax]` + `expds/ae_recall_relax.py` + тесты, default OFF
  - `475bafa` — `scripts/eval_aerial_honest10.py`
- URL: https://github.com/KonkovDV/expds-koz3/tree/bots/bot3-aerial-recall
- PR (draft): https://github.com/KonkovDV/expds-koz3/pull/3
- Worktree: `/workspace/koz3-wt/bot3-aerial-recall`

## Корневые причины (диагноз)
1. **Модель ae = только форты** (`models/ae_fort_mask_r18.pt`, classes=`fortifikatsii`) при τ≈**0,998215** → почти ничего не проходит на LB.
2. **li_to_ae копии** режутся группой `li_ae` (povr **0,993061**, tsel **1,000001**) и `fusion.povr_min=0,99` → копии почти мёртвые.
3. **Участки ae без лидара** (большинство публичного LB ae): li_to_ae нечем копировать; **нет ae-курганной сети** → ~554 GT курганов в no-li регионах недостижимы порогами.
4. **`[dfine_aeor]`** уже в базе ветки, **enabled=false** (веса koz3-dfine-ae COMPLETE на Kaggle — отдельный вход).
5. **`[raster_fix]`** (CRS 3857→UTM, int16 identity, zero_nodata) чинит часть битых ae-растров; без секции = OFF.

Train keep ae раньше: P≈0,39 R≈0,20 vs LB R=0,013 — пороги/домен, не «нулевой» сигнал.

## Флаги (все default OFF)
| Флаг | Где | OFF = | ON делает |
|---|---|---|---|
| `ae_recall_relax` | `[ae_recall_relax] enabled=false` | пороги/fusion как в default.toml | понижает ae/li_ae τ и fusion.povr/tselye_min (см. DEFAULTS в `expds/ae_recall_relax.py`) |
| `dfine_aeor` | `[dfine_aeor] enabled=false` | фаза не бежит | D-FINE всех классов на Ae/Or (нужны веса + thresholds_file) |
| `raster_fix` | секция закомментирована | configure_raster_fix(None) | раскомментировать блок + ключи true |

## Честный eval
```
python scripts/eval_aerial_honest10.py \
  --work <work_stack> --train F:/2/train-expds-3 \
  --out notes/bot3_aerial_honest10.json
```
Опыт с relax: конфиг с `enabled=true` → прогон пайплайна/OOF → scoreboard `--add`/`--replace` vs база.

## Заявка на ресурс
См. `pc/QUEUE.md` [bot-3] (pytest flags-off + mock e2e flags-off identity + опционально honest-10 на имеющемся стеке). **Kaggle не нужен** для порогового прохода; обучение D-FINE Ae — следующий пункт BACKLOG GPU-3.

## Следующий пункт BACKLOG (для «1»)
После закрытия этой задачи бот 3 берёт **«D-FINE Ae ядро Kaggle (п. GPU-3)»**:
- BACKLOG §Kaggle GPU п.3: «D-FINE Ae: дообучение / доп. эпохи на аэрофото (recall ae 0,013 — главный провал)».
- §Боты: «3: recall аэрофото → D-FINE Ae ядро Kaggle (п. GPU-3)».
- koz3-dfine-ae уже COMPLETE — уточнить у «1», нужен ли fine-tune / доп. эпохи или только калибровка порогов + `[dfine_aeor] enabled=true` на честном honest-10.
