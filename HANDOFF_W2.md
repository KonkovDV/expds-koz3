# HANDOFF W2 (бот 3, лидар / ConvNeXt, тег [li_cnx]) — 30.10.2026 ~11:20 MSK

Ветка: `bots/w2-lidar` (worktree бокса `/workspace/koz3/wt/w2`). Координатор — бот 1. В `main` не пушить, LB-сабмитов нет, токены не коммитить.
Время везде MSK. База A = 0.314615 (стек `/workspace/koz3/w5data/stack_arch.geojson`).

## Что делать дальше (по порядку)

ВАЖНО: на аккаунте Grok Bot бота 3 стояли 2 задачи по расписанию (запуск 03:10 и наблюдение/гейт 05:00–15:30 в Сб 31.10). На другом аккаунте их НЕТ — делать вручную по шагам ниже.

Обозначения: `K=/workspace/koz3`, `PK=$K/orch/pool_kaggle.sh`, `L=$K/orch/lease.sh`, owner аренд `bot-W2-lidar`.

1. **Сб 31.10, 03:05–03:30 — проверка квоты после сброса (03:00).**
   `$PK elenakonkova kernels list --mine -s koz3 | head` и `$PK dmitrijkv kernels list --mine | head` (убедиться, что нет чужих RUNNING на нужных слотах).
   Квоту GPU смотреть на kaggle.com/settings каждого аккаунта (нужно ≥ 7 ч на elenakonkova, ≥ 12 ч на dmitrijkv; после сброса — по 30 ч).
   Hold в `$K/orch/QUEUE.json` по `w2-r50c5-f01-acc1` и `w2-r50c5-f234` бот 1 снял/снимает; если там ещё `hold` — решение бота 1 (31.10) было «запускать оба параллельно», статус ставить `pushed` после push.
2. **Аренды (до push).**
   `$L acquire slot:acc1/gpu-1 bot-W2-lidar 7 --kernel elenakonkova/koz3-w2-cnx-r50c5-f01 --note "P-W2-9 r50c5 f01"`
   `$L acquire slot:acc2/gpu-1 bot-W2-lidar 12 --kernel dmitrijkv/koz3-w2-cnx-r50c5-f234 --note "P-W2-9 r50c5 f234"`
   Если «ЗАНЯТО» — `$L who <res>` и спросить бота 1, не трогать чужие ядра.
3. **Push обоих ядер параллельно.**
   `$PK elenakonkova kernels push -p $K/w2/kernels/elenakonkova/koz3-w2-cnx-r50c5-f01`
   `$PK dmitrijkv kernels push -p $K/w2/kernels/dmitrijkv/koz3-w2-cnx-r50c5-f234`
   (копии в git: `w2_kernels/...`). Статус: `$PK <acc> kernels status <acc>/<slug>`. Строку в `orch/bot-W2/STATUS.md` + `orch/COORD_LOG.md`.
4. **≈05:40–06:30 — f01 готов** (оценка 2,6–3,3 ч). Проверять раз в 30 мин:
   `$PK elenakonkova kernels status elenakonkova/koz3-w2-cnx-r50c5-f01`; при COMPLETE —
   `$PK elenakonkova kernels output elenakonkova/koz3-w2-cnx-r50c5-f01 -p $K/w2/oof/r50c5/koz3-w2-cnx-r50c5-f01` (read-only).
   Ошибка/таймаут до f0+f1 → одна повторная попытка тем же кодом, затем стоп.
5. **Ранний стоп (P-W2-9, без гейта).** Из лога/`report_cnx_full.json` ядра (Li-only, materials/compute_metrics.py):
   (1) среднее Li-only f0,f1 r50c5 ≤ **0,3492** (c3-sc s2: f0 0,2198 / f1 0,4786) → FAIL;
   (2) F1 курганов (kurgany_tselye+kurgany_povrezhdennye) на f0 r50c5 не выше c3-sc s2 на f0 → FAIL.
   Шаблон проверки: `w2_gate/gate_lihm/early_stop.json`, `w2_notes/w2_lihm_gate.json`.
6. **f234 по итогу:**
   FAIL → `$PK dmitrijkv kernels ...` отменить/не использовать выход f234 (на сайте Kaggle: Cancel run), снять аренды `$L release slot:acc2/gpu-1 bot-W2-lidar` (+ kernel:, slot:acc1/gpu-1), ничего нового не запускать, квота → W1; отчёт боту 1.
   PASS → ждать f234 (до ~11,3 ч, ≈ до 14:30), забрать в `$K/w2/oof/r50c5/koz3-w2-cnx-r50c5-f234`, снять аренды.
7. **Гейт (только PASS, после 5/5 фолдов).** Склейка: `$K/w2/tools/merge_cnx.py` (как для c3-sc, проверки: threshold_group, region_fold, 0 вне фолда, EPSG:3857), группа переименовывается в `li_r50c5`.
   Прогон по образцу `w2_gate/gate_s2/run_varb.sh` (prep.py → gate_v6c.py → gate_detail.py), см. раздел «Гейт honest-10». H1 аддитивно; H2 (замена курганов) только если H1 FAIL, P ≥ 0,95. Результат → боту 1 (принимают W4/W5).
8. **P-W2-10 (DEM fill), если не закончен.** Код: флаг `lidar.dem_fill="tin_lowveg"` в `expds/lidar.py` (по умолчанию "delaunay"); скрипты `$K/w2/demfill/` (копия `w2_demfill/`): `build_relief.py` (C/T рельеф 015/021/186/190) → `run_inf.sh`/`infer_cpu.py` (c3-sc s0/s1/s2 веса своего фолда, CPU, `nice -n 10`) → `swap_sites.py` → `run_gate.sh`. Условия PASS — PREREG P-W2-10. Логи `$K/w2/demfill/*.log`, выход `$K/w2/demfill/out`.
9. **li_match (P-W2-5)** — после r50c5 и только если бот 1 даст GPU-квоту: нужно пофолдовое обучение c3-sc с `lidar.hillshade_style="li_match"` (5 фолдов, ~как s2: f01 + f23 + f4). Иначе закрыть как непроверяемое.

## Состояние линий W2
| Линия | Итог |
|---|---|
| c3-sc s0/s1/s2 аддитивно (P-W2-2/3) | FAIL: s2 Δ−0,0162 (P=0,000); s012avg Δ−0,0071 (P=0,106); s01avg (V6c W5) Δ−0,0103 |
| вариант (b) замена li-слоя на c3-sc s012avg (P-W2-6) | FAIL: B=0,29513 Δ−0,0195 P=0,12; new34 −0,050; без лидара +0,009 (сдвиг порогов); b-kur Δ−0,0112. c3-sc закрыт, сиды s3+ в hold |
| li_hm heatmap (P-W2-7) | ранний стоп FAIL: f0 0,2099 / f1 0,4175, среднее 0,3137 ≤ 0,3492 |
| CI2-3 / P-W2-8 ground-clip | закрыт: DEM строится только по классу 2; все точки — только 037/101/102/187 (там лишь класс 8) |
| CI2-8 hillshade parity | 59/59: MAE vs Li_g медиана 4,46/255 (p90 6,73, max 11,1 — 038); лучшее Horn az315 alt 41–53 z 0,7–1,2, blur 1 px в 34/59; 28/59 у Li другой шаг растра |
| Center snap курганов (P-W2-4) | закрыт, нетто −6 |
| c5 (P-W2-1) | закрыт решением бота 1 |
| r50c5 (P-W2-9) | ждёт запуска Сб 31.10 03:00 (выше) |
| DEM fill (P-W2-10) | в работе (CPU) |
| li_match (P-W2-5) | флаг есть (выкл.), замер заблокирован — нет пофолдовых весов |

182_Красносельская_1.7км — 0 детекций (распаханные насыпи вне распределения), не чинится.

## PREREG (orch_bot-W2/PREREG_W2_cnx.md)
P-W2-1 c5 — закрыт; P-W2-2 seed-ансамбль — FAIL; P-W2-3 seed1 контроль — выполнен; P-W2-4 center snap — FAIL; P-W2-5 li_match — заблокирован; P-W2-6 (b) — FAIL; P-W2-7 li_hm — FAIL (ранний стоп); P-W2-8 ground-clip — закрыт (условие не выполнено); P-W2-9 r50c5 — зарегистрирован, ждёт запуска; P-W2-10 dem_fill — в работе.

## Гейт honest-10
- База A=0.314615 (`materials/compute_metrics.py`), пайплайн W5 `$K/wt/w5/analysis/2026-10-30/w5_rerank` (`prep.py`, `gate_v6c.py --vs $K/w5data/work/base_counts.pkl`), детали `$K/w2/tools/gate_detail.py`; сборка стека `$K/w2/tools/stack_replace_li.py` / `seed_merge_cnx.py`. Пример: `w2_gate/gate_s2/run_varb.sh`.
- PASS ⇔ Δ ≥ +0,003 И P(B>A) ≥ 0,9 (paired bootstrap 2000 по 103 семьям) И полы: −0,002 kurgany_povr/tselye/gorodishcha, −0,005 прочие (классы с n_GT<5 исключены) И new LAS ≥ −0,001 И без лидара ≥ −0,001. Принимают только W4/W5. PREREG до замера, новые фичи за флагами (выкл.).
- CPU бокса: ≤ 3 тяжёлых процесса, 8 ГБ, `nice -n 10`.

## Kaggle и аренды (без секретов)
- GPU/TPU: только elenakonkova (acc1) и dmitrijkv (acc2); сброс квоты Сб 03:00 MSK. CPU-пул: aaaa0000004 и др. (без интернета — данные через приватные датасеты).
- `orch/pool_kaggle.sh <user> <kaggle args>` — токен берётся из env/локального файла, не печатается. Обычный `kaggle` CLI не авторизован.
- Аренды: `orch/lease.sh acquire|renew|release|who|list`, ресурсы `slot:<acc>/<gpu-1|gpu-2|tpu|cpu>`, `kernel:<acc>/<slug>`, `branch:<name>`; файл `orch/LEASES.json`. Префикс ядер `koz3-w2-`. Чужие RUNNING/QUEUED не трогать. Известный баг: аренда иногда сразу GONE — у бота 1.

## Где лежат большие файлы (бэкап 30.10 ~11:50 MSK)

Все датасеты ПРИВАТНЫЕ, на аккаунте elenakonkova, статус `ready`, список файлов и размеры сверены с боксом 1:1 (скрипт сверки — `datasets files --csv` vs `os.walk`). Скачать: `$PK elenakonkova datasets download elenakonkova/<slug> -p $K/w2/restore/<slug> --unzip`. Лимиты Kaggle: 200 ГБ на датасет и 200 ГБ приватных на аккаунт, ≤ 50 файлов в корне (поэтому всё по подпапкам). Kaggle сам распаковывает загруженные .gz/.zip — файлы лежат несжатыми.

| Артефакт на боксе | Где хранится | Как достать / пересобрать |
|---|---|---|
| `w2/oof/merged/*` (s0, s1, s2, s01avg, s012avg + merge_summary), `w2/oof/{s0,s1,s2,lihm}/<ядро>/` (oof по фолдам, gt, region_fold, report_cnx_full, логи; без .pt и maps), `w2/seedmerge/` | **elenakonkova/koz3-w2-oof** — 92 файла, 0,99 ГБ | download; также output ядер (`$PK <acc> kernels output <acc>/<slug>`); merged пересобирается `w2/tools/collect_cnx.sh` → `merge_cnx.py` / `seed_merge_cnx.py` |
| Веса c3-sc `weights_cnx_c3-sc_f{0..4}.pt` для s0, s1, s2 (15 шт.), `ckpt_*` s2 f2–f4, li_hm `ckpt_cnx_c3-sc_f{0..4}.pt` (у li_hm только ckpt) | **elenakonkova/koz3-w2-weights** (v2) — 24 файла, 1,47 ГБ; структура `s0|s1|s2|lihm/<ядро>/` | download; первоисточник — output ядер dmitrijkv/koz3-cnx-li-c3sc-{f01,f23,f4-v2}, dmitrijkv/koz3-cnx-li-c3sc-s1f01, elenakonkova/koz3-cnx-li-c3sc-{s1f23,s1f4}, dmitrijkv/koz3-w2-cnx-c3sc-{s2f01,s2f23}, elenakonkova/koz3-w2-cnx-c3sc-s2f4, elenakonkova/koz3-w2-lihm-c3sc-f01234 (`--file-pattern '^weights_.*\.pt$'`) |
| Карты `maps_*.npz`: `w2/maps/{s0,s1,s2}` (+симлинки), `oof/s2/*/maps`, `oof/lihm/*/maps` | **elenakonkova/koz3-w2-maps** — 295 файлов, 2,11 ГБ; `w2maps/…`, `koz3-w2-cnx-c3sc-s2f23/`, `koz3-w2-cnx-c3sc-s2f4/`, `koz3-w2-lihm-c3sc-f01234/` | download; или output соответствующих ядер (`--file-pattern 'maps/'`) |
| Гейт: `w2/gate_s2/{s2,s012,varb,varb_kur}` (стеки + work/*.pkl), `logs`, `detail_*.json`, `check_s2only.geojson`; база W5: `w5data/stack_arch.geojson`, `gt_train_v2.geojson`, `work/base_counts.pkl`, `work/table.pkl` | **elenakonkova/koz3-w2-gate** — 54 файла, 5,43 ГБ; `gate_s2/…`, `w5base/…` | download; либо пересобрать: `w2/gate_s2/run_varb.sh` (stack_replace_li.py → prep.py → gate_v6c.py → gate_detail.py; для s2/s012 — тот же шаблон с OOF s2/s012avg, аддитивно через `seed_merge_cnx.py`), ~10–20 мин CPU на вариант, ~2 ГБ на вариант |
| Рельеф `w2/relief/cnx_relief` (119 .npy/.json, 1,6 ГБ) | уже был: **elenakonkova/koz3-cnx-li-prep** (119/119 совпали по размеру) | download; пересборка — ядро dmitrijkv/koz3-cnx-li-prep (S3 → рельеф) |
| LAS/Li-карты/разметка `w2/las` (54 файла, 1,9 ГБ; 7 участков) | S3 бакет `train-expds-3` (endpoint `https://s3.ru-7.storage.selcloud.ru`) — 54/54 совпали по пути и размеру со снимком листинга `$K/s3_keys.tsv` (29.10) | boto3 как в ядрах (ключи — приватный датасет elenakonkova/koz3-s3-cred, не печатать) |
| `w2/gaudit/ds` LAS 015/021/186/190 + Li-разметка (10 ГБ) | S3 (те же LAS по размеру) + **aaaa0000004/koz3-w2-gaudit-las** (16 файлов, ready) | download с aaaa0000004 или S3 |
| `w2/kernels` (скрипты ядер, 9 МБ) | git, эта ветка `w2_kernels/` | — |

**Не выгружено (и почему):**
- `w2/demfill/{rel,maps,out}` (≈125 МБ) — тест P-W2-10 ещё идёт, частичные результаты не бэкапились. Когда закончится: `mkdir -p $K/w2/bk/stage/demfill && cp -al $K/w2/demfill/{rel,maps,out} $K/w2/bk/stage/demfill/`, `dataset-metadata.json` с `"id":"elenakonkova/koz3-w2-demfill"`, затем `orch/lease.sh acquire dataset:elenakonkova/koz3-w2-demfill bot-W2-lidar 4` и `cd $K/w2/bk/stage/demfill && $PK elenakonkova datasets create -p . --dir-mode zip` (приватный по умолчанию). Веса для демфилла (`demfill/w`) уже в koz3-w2-weights. Рельеф пересобирается `w2/demfill/build_relief.py` из LAS в S3.
- `w2/repo` (1,6 ГБ) — клон `KonkovDV/expds-koz3`, есть на GitHub.
- `w2/bak/merged_pre_s2_0334` — дубль merged s0/s1/s01avg до s2 (те же файлы в koz3-w2-oof).
- `w5data/work/X*.npy, geom_wkb.pkl, nested_*` — данные W5 (не W2), пересобираются `prep.py` из stack_arch + gt (см. «Гейт honest-10»).
- Стейджинг бэкапа: `$K/w2/bk/stage/{oof,weights,maps,gate}` (хардлинки + dataset-metadata.json). Новая версия: `cd $K/w2/bk/stage/<s> && $PK elenakonkova datasets version -p . --dir-mode zip -m "..."` (сначала аренда `dataset:elenakonkova/koz3-w2-<s>`).

## Большие артефакты (только на боксе, не в git) — см. раздел выше, всё кроме перечисленного выгружено
- `$K/w2/oof/merged/` — oof_cnx_c3-sc_{s0 64M, s1 62M, s2 71M, s01avg 62M (отдан W5), s012avg 63M}.geojson; s0 129M, s1 126M, s2 677M (`oof/s*`); li_hm `oof/lihm` 1,7G (карты npz)
- `$K/w2/gate_s2/{s2,s012}` по 2,0G (стеки 1,1G, pkl), `varb` 206M, `varb_kur` 213M
- `$K/w2/las` 5,2G, `$K/w2/gaudit/las` 6,7G (015/021/186/190), `$K/w2/relief/cnx_relief` 1,6G
- `$K/w2/demfill/{rel 100M, w 368M (веса c3-sc), maps 13M}`
- `$K/w5data/stack_arch.geojson` 1,1G (.gz 185M, sha 42ff3087…), `gt_train_v2.geojson` 38M, `work/table.pkl` 348M
- Веса c3-sc: в output ядер Kaggle (elenakonkova/dmitrijkv, слаги koz3-w2-cnx-c3sc-*).

## Что в этой ветке
`orch_bot-W2/` (STATUS, PREREG, TASKS), `w2_kernels/` (скрипты+metadata ядер), `w2_tools/`, `w2_notes/`, `w2_gate/` (gate_s2 json/логи/скрипт, gate_lihm), `w2_gaudit/` (сводки аудита), `w2_merge/` (merge_summary.json), `w2_demfill/` (код P-W2-10), `tools/w2`, `notes/w2`, `expds/lidar.py` (флаги hillshade_style, dem_fill).
