# План работ ИИ-исполнителя с 25.10.2026

Этот файл — задание исполнителю и заменяет `docs/AI_WORK_PLAN_2026-10-23.md` как текущий план. Почему он такой — `docs/AUDIT_2026-10-24.md`. Правила `docs/AGENT_PLAN.md`, раздел 0, действуют, кроме приёмки: она теперь по честному баллу (раздел 0 ниже). Окружение прежнее: Windows, PowerShell без `&&`, `python -X utf8`, корень `C:\Sloi\expds-koz3_v0.2.0\expds-koz3`, `.venv\Scripts\python.exe`, данные `F:\expds3`. Код чужих команд не копировать. Коммит и push — только по прямой просьбе. Колонку «кто проверил» в `docs/AI_DECLARATION.md` не заполнять.

## 0. Что изменилось в правилах

1. **Приёмка — по честному баллу.** Табло подбирает пороги групп на других семьях и применяет к своей (`--honest 10`). Решение о конфиге — строка `честная приёмка: проходит`: прирост честного балла ≥ 0,002, бутстреп по семьям A > B ≥ 0,8, 34 новых и участки без лидара не хуже чем на 0,001. Старая строка `приёмка:` печатается для сравнения, но решения не принимает. Пороги, которые идут в конфиг, по-прежнему из перебора на всех участках.
2. **Срок загрузки — 11.11, 23:59 МСК.** Так пишет страница площадки 24.10. ТР п. 5.3 продлевает этап до 14.11, но загрузку закрывает площадка. Финальная версия — в течение трёх дней после конца квалификации (ТР 8.3.1), на закрытой выборке две успешные попытки, в рейтинг идёт лучшая (8.3.5).
3. **Посылки — инструмент замера.** Квалификация даёт 3 успешные посылки в сутки, место на призы не влияет. Каждая посылка отличается от предыдущей одной веткой, уже принятой честным табло. После каждой — `python scripts\lb_counts.py <scoring.log>` и строка в `docs/LB_LOG.md` с целыми TP/FP по типам данных. Порог по публичному баллу не подбирается: публичный балл отвечает только «ветка вкл/выкл».
4. **Порядок веток — по весу и цене.** Геофизика (20 % балла, половина эталона выборки) → архитектура (10 %, сейчас 0) → городища и фортификации (точка внутри) → курганы спутника (на арендованной карте).
5. **Новая база честного табло.** Первый же прогон с `--honest 10` даёт честную базу текущего стека. Записать её в `docs/PLAN.md` и дальше сравнивать с ней. 0,329489 считался одной семьёй на фолд (`--honest -1`) и на стеке до пакета.

Команда базы (дополняет `docs/AGENT_PLAN.md`, раздел 0, п. 5):

```powershell
$env:PYTHONPATH = "C:\Sloi\expds-koz3_v0.2.0\expds-koz3"
.venv\Scripts\python -X utf8 -u scripts\scoreboard.py --work F:\expds3\work --train F:\expds3\train-expds-3 `
  --metric C:\Sloi\files\materials\compute_metrics.py --gt F:\expds3\work\gt_train_v2.geojson `
  --geo F:\expds3\work\geo_cur.geojson `
  --base-add F:\expds3\work\ae_fort_mask_oof.geojson:ae_fort:fortifikatsii:ae `
  --base-add F:\expds3\work\sp_arch_oof.geojson:sp_arch:arkhitektury:sp_or `
  --base-add F:\expds3\work\li_fort_mask_oof.geojson:li_fort:fortifikatsii:li `
  --base-add F:\expds3\work\li_dev_far.geojson:li_dev `
  --base-add F:\expds3\work\spor_focal_wide_opt.geojson:sp_focal2 `
  --expect 0.353793 --honest 10 --out notes/honest_base_2026-10-25.json
```

Если файлы пакета называются иначе — взять те, что дали 0,353793 (`notes/pack_code_2026-10-23.json`). `--expect` обязан сойтись до любого опыта.

## 0.5. Новые места или соседние планшеты (первым делом, 5 минут)

От ответа зависит, чем мерить опыты. Если участки выборки лежат рядом с участками трейна, честная оценка по семьям занижает всё, что выучивает конкретный памятник, и тогда правильна оценка по участкам.

```powershell
.venv\Scripts\python -X utf8 scripts\site_overlap.py --train F:\expds3\train-expds-3 --val <папка публичной выборки> --out notes\site_overlap_public.json
```

- Больше половины участков выборки ближе 2 км к участку трейна (`<= 2000 m` в сводке) → во всех командах табло `--honest 10 --honest-by region`.
- Иначе → `--honest-by family` (по умолчанию).
- Отдельно смотреть `median_gap_m_by_type` для `maggr`, `etgr`, `terfl`: если геофизика выборки — те же памятники, что трейн, сеть на картах геофизики выигрывает у классики заведомо (см. `docs/AUDIT_2026-10-24.md`, раздел 11: у evgps сеть MagGr даёт +0,054).

Имена участков выборки скрипт в вывод не пишет.

## 1. Проверка пакета 23.10 честным баллом (первым делом, 10 минут)

Тот же прогон без двух последних `--base-add` и с `--expect 0.350588`, затем пакет через `--add`. Если `честная приёмка: не проходит` — пакет окна 256 м и канала σ на площадке ничего не дал (так и вышло: 0,32850 против честных 0,329489 до пакета). Тогда его можно убрать из конфига: минус две сети, четыре хрупких порога у 0,99+, минус минута инференса. Решение записать в `docs/PLAN.md`. Если проходит — оставить.

## 2. Геофизика без видеокарты (25–26.10)

**G0. Паспорт растров.** Без эталона на публичной выборке, с эталоном на трейне:

```powershell
.venv\Scripts\python -X utf8 scripts\geo_probe.py --root F:\expds3\train-expds-3 --gt F:\expds3\work\gt_train_v2.geojson --out notes\geo_probe_train.json
.venv\Scripts\python -X utf8 scripts\geo_probe.py --root <папка публичной выборки> --out notes\geo_probe_public.json
```

Что смотреть:
- `channels_differ` > 0,01 у карт, которые читает детектор (`used_by_detector`) → опыт G3 обязателен. На публичной выборке по одной карте MagGr на участок: если это цветная или «2D» карта, первый канал — не поле.
- `recall_ceiling` на трейне при отсечке 1,0 против 0,5 → сколько эталонных находок вообще не становится кандидатом.
- `peaks_per_1000m2` публичной выборки против трейна при той же отсечке → площадка беднее кандидатами или нет.
- `crs` и `identity_transform` у 14 профилей EtGr публичной выборки → без привязки объекты профиля уходят в пиксельных координатах.

Имена участков публичной выборки в `notes/` не переносить: скрипт сам заменяет их номерами, когда эталона нет.

**G1. База геофизики воспроизводится текущим кодом.**

```powershell
.venv\Scripts\python -X utf8 scripts\geo_predict.py --root F:\expds3\train-expds-3 --out F:\expds3\work\geo_default.geojson
# табло из раздела 0 с --geo F:\expds3\work\geo_default.geojson: база обязана совпасть с --expect
```

Если не совпала — стоп, сравнить с `geo_cur.geojson` по участкам: файл табло собран не тем кодом.

**G2. Отсечка пиков и радиус подавления.** Сравнение — по `f1_lofo_pooled`, не по `f1_in_sample`:

```powershell
.venv\Scripts\python -X utf8 scripts\geo_tune.py --root F:\expds3\train-expds-3 --gt F:\expds3\work\gt_train_v2.geojson `
  --only maggr/finds_points --k-min 1.0 --out F:\expds3\work\geo_k10.json      # база: f1_lofo_pooled 0,4614
.venv\Scripts\python -X utf8 scripts\geo_tune.py --root F:\expds3\train-expds-3 --gt F:\expds3\work\gt_train_v2.geojson `
  --only maggr/finds_points --k-min 0.5 0.75 1.0 --out F:\expds3\work\geo_k.json
```

Если `f1_lofo_pooled` выросла: взять из `best_cfg` фон, радиус, `peak_k`, `use_noise`, `confidence` и `tau` и собрать файл:

```powershell
.venv\Scripts\python -X utf8 scripts\geo_predict.py --root F:\expds3\train-expds-3 --out F:\expds3\work\geo_k.geojson `
  --set geophys.maggr.finds_points.peak_k=<k> --set geophys.maggr.finds_points.peak_radius_m=<r> `
  --set geophys.maggr.finds_points.background_m=<bg> --set geophys.maggr.finds_points.min_confidence=<tau>
```

Табло с `--geo F:\expds3\work\geo_k.geojson` вместо `geo_cur`. Принимается по честной приёмке. Затем — одна посылка с этим изменением и `lb_counts.py`: на площадке у геофизики было TP 373 / FP 197 / эталон 1441.

**G3. Каналы карты.** Только если G0 показал различающиеся каналы:

```powershell
... scripts\geo_tune.py ... --only maggr/finds_points --band-mode auto --k-min <лучшее из G2> --out F:\expds3\work\geo_band.json
... scripts\geo_tune.py ... --only maggr/object_poly --band-mode auto --out F:\expds3\work\geo_band_poly.json
```

В конфиг — `[geophys] band_mode = "auto"` (действует на все модальности геофизики) и пороги из того же прогона.

**G4. object_poly.** По очереди, каждый своим опытом:
- полярность ям: `geo_tune.py --only maggr/object_poly --polarities abs pos --out ...`;
- точка внутри: `python scripts\oof_transform.py core --src <файл геофизики> --classes object_poly --frac 0.3 --min-r 1 --out ...`, затем табло с этим файлом в `--geo`;
- то же для `etgr/object_poly` и `terfl/object_poly` (`--polarities abs pos neg`).

Аналитический сигнал (`--signals abs asa`) — последним и только если остаётся время. На модели диполя он локализацию не улучшает (`tests/test_geophys_signal.py`).

## 2.5. Голосование моделей по готовым файлам (без видеокарты, 1–2 часа)

Приём победителя «Сканирования» (Family): объект подтверждают несколько независимых моделей. У нас отвергнутые поодиночке модели уже посчитаны вне обучения на тех же фолдах. Голосование не добавляет новых объектов и групп порогов — меняется только уверенность сданных кругов, поэтому оптимизм выбора порога здесь мал.

```powershell
.venv\Scripts\python -X utf8 scripts\oof_vote.py --base F:\expds3\work\spor_focal_oof.geojson `
  --others F:\expds3\work\spor_focal_r05_oof.geojson F:\expds3\work\spor_focal_r34_oof.geojson `
           F:\expds3\work\spor_focal_photo_oof.geojson F:\expds3\work\spor_focal_scale_oof.geojson `
           F:\expds3\work\spor_focal_hrrep_oof.geojson `
  --radius 6 --mode rank-mean --out F:\expds3\work\spor_vote_mean.geojson
# табло из раздела 0 с --replace sp_focal=F:\expds3\work\spor_vote_mean.geojson --honest 10
```

- Второй опыт — `--mode votes --vote-rank 0.5` (ближе к «2 из 4» Family).
- Радиус 6 м: курган на спутнике — медиана 491 м², эквивалентный радиус около 12 м.
- Если принято: в сдачу идут веса этих моделей на всех тайлах и тот же пересчёт уверенности на инференсе (`pipeline` — отдельная правка после приёмки; время площадки есть: занято 3 из 30 минут).
- То же для курганов лидара: база `fusion\li_hard2_fused.geojson`, голоса `li_dev_oof.geojson`, `li_focal_oof.geojson`, `li_ema_oof.geojson`; в табло — `--fused`.

## 3. Сеть на картах MagGr и EtGr (26–30.10, ночи GTX 1650) — главная ветка

Это самая подтверждённая ставка: у конкурента сеть MagGr по его замеру даёт +0,054 к баллу, та же сеть идёт у него и на EtGr, находки выдаются с порогом 0,02. У нас вся геофизика даёт 0,061 при полноте 26 %. Задача — та же, что локализация объектов точками: карта центров с фокальной потерей, как у сданной карты курганов спутника. Всё уже есть в коде; сдача запускает сеть строкой `[models] maggr` (`pipeline.geo_model_feats`, группа `maggr_nn`).

```powershell
.venv\Scripts\python -X utf8 -m expds.tiles --root F:\expds3\train-expds-3 --gt F:\expds3\work\gt_train_v2.geojson `
  --markup maggr --out F:\expds3\work\cache\maggr_finds_r01_t256 --res 0.1 --tile 256 --classes finds_points
.venv\Scripts\python -X utf8 scripts\nn_oof.py --markup maggr --root F:\expds3\train-expds-3 `
  --cache F:\expds3\work\cache\maggr_finds_r01_t256 --work F:\expds3\work\maggr_finds `
  --out F:\expds3\work\maggr_finds_oof.geojson --folds 3 --epochs 30 --batch 16 `
  --classes finds_points --target centers --center-loss focal --center-sigma 0 `
  --center-radius 0.3 --center-min-dist 1.0
```

- `--folds 3`: у MagGr три семьи (Гнёздово, Семибратнее, Судьбищи), фолд — семья.
- Шаг: конкурент читает MagGr на 0,05 м. Если `geo_probe` (поле `res`) показывает 0,05 м и мельче — `--res 0.05 --tile 256` (окно 12,8 м); если 0,1 м — как в команде выше.
- Порог группы `maggr_nn` подбирает табло; ожидать низкий (у конкурента 0,02): при F1 около 0,4 добавлять выгодно уже при 11 % попаданий.
- Второй прогон — тот же рецепт на EtGr (`--markup etgr --classes object_poly`, `--center-radius 0.3 --center-min-dist 0.5`): каждый срез и профиль — отдельный эталон, `select_rasters` уже отбрасывает срезы глубже 2 м.
- Табло: `--add F:\expds3\work\maggr_finds_oof.geojson:maggr_nn:finds_points:maggr` поверх лучшей классики. Если сеть лучше классики целиком — второй опыт `--replace maggr=...` с классикой находок, выключенной в файле геофизики.
- Принято → веса на всех тайлах (`python -m expds.train ... --all-data`, эпох — медиана лучших эпох фолдов), в конфиг:

```toml
[models]
maggr = {path = "models/maggr_finds_r18.pt", classes = ["finds_points"]}
[thresholds_by_markup.maggr_nn]
finds_points = <порог из табло>
# если сеть заменяет классику находок целиком:
# [geophys.maggr]
# classic = false        # выключает и object_poly классики MagGr — тогда нужна вторая сеть или отдельный опыт
```

Без секции `[thresholds_by_markup.maggr_nn]` сеть выдаёт всё выше нуля — сначала пороги, потом строка модели.

Тот же рецепт потом для `object_poly` MagGr (`--classes object_poly`, `--center-radius 0.5 --center-min-dist 2.0`): маленький круг внутри ямы засчитывается при внутренней точке в эталоне.

## 4. Архитектура и ортофото (28.10–04.11)

Вес 10, на площадке 0 попаданий. На трейне архитектура — 210 объектов на спутнике (медиана 182 м²) и 178 на ортофото четырёх участков (медиана 15 м², десятая перцентиль 0,1 м²). На публичной выборке одно ортофото 2,7 ГБ с эталоном не меньше 40–50 объектов.

```powershell
.venv\Scripts\python -X utf8 -m expds.tiles --root F:\expds3\train-expds-3 --gt F:\expds3\work\gt_train_v2.geojson `
  --markup or --out F:\expds3\work\cache\or_arch_r01_t256 --res 0.1 --tile 256 --classes arkhitektury
.venv\Scripts\python -X utf8 scripts\nn_oof.py --markup or --root F:\expds3\train-expds-3 `
  --cache F:\expds3\work\cache\or_arch_r01_t256 --work F:\expds3\work\or_arch `
  --out F:\expds3\work\or_arch_oof.geojson --folds 3 --epochs 30 --batch 16 `
  --classes arkhitektury --target centers --center-loss focal --center-sigma 0 `
  --center-radius 0.5 --center-min-dist 2.0
```

- Сдача уже читает ортофото (`optical_feats`, тип `or`) — нужна строка `[models] or = {path = ..., classes = ["arkhitektury"]}` и `[thresholds_by_markup.or]`.
- Семей ортофото три–четыре: честный балл здесь шумный. Решение — честная приёмка плюс одна посылка «ортофото вкл»: `lb_counts.py` покажет TP ортофото напрямую (сейчас 0).
- Второй шаг — архитектура спутника высокого разрешения на шаге 0,25 м (сейчас 0,5 м, F1 0,026): тот же рецепт с `--markup sp_or --res 0.25`, только участки со снимком мельче 0,35 м.

## 5. Городища и фортификации — точка внутри (1 час CPU)

Опыт 24.10 «ядро» дал +0,0018 и не прошёл старую приёмку. Он не добавляет групп порогов, поэтому честный балл должен держать прирост лучше, чем у новых групп. Повторить с `--honest 10` теми же файлами (`notes/post_core_2026-10-24.json`). Корзина площадей 5000–10 000 м² на площадке — 0 % найдено: крупные объекты не находятся вовсе.

## 6. Курганы, городища и архитектура на снимках мелким шагом — на арендованной карте (с 28.10)

Конкурент читает спутник и аэрофото на шаге 0,3 м тайлами 1024 (окно около 300 м), у нас курганы спутника — 1 м и окно 128 м, аэрофото — 0,5 м. Опыт 23.10 «0,5 м, окно 128 м» (`sp_focal_r05`) этого не проверял: окно то же, сеть та же маленькая. Опыты на GTX 1650 исчерпаны: ResNet34 с батчем 8 «не оставил кругов» — это недообучение, а не потолок. Аренда GPU разрешена ответами организаторов, если доступ только у команды, копии под контролем и удаляются после (`docs/DATA_PROTOCOL.md`, п. 3). На карте 24 ГБ:

- карта центров курганов спутника: `--encoder resnet50`, `--batch 16–32`, шаг 0,3 м и тайл 1024 (окно ~300 м), затем для сравнения 0,5 м и тайл 512; 40 эпох, `--ema 0.999`; на инференсе сдача сама приводит снимок к шагу модели (`_raster_with_model`), время площадки позволяет — сейчас занято 3 из 30 минут;
- аэрофото: тот же рецепт, шаг 0,3 м;
- «общая» модель на спутник и аэрофото вместе с отдельной группой порога — у конкурента так и сделано; у GigaFlex по открытому описанию — общая сеть, затем узкие модели по классу;
- проверка — те же пять фолдов, честная приёмка.

Лицензия весов: если организаторы запретят ImageNet, запасной путь — энкодер DINOv2 (Apache-2.0 на код и веса), а не обучение с нуля. DINOv3, SAM 3, SegFormer `mit_*`, ultralytics — нет.

## 7. Календарь

| Даты | CPU | GPU ночью | Посылки |
|---|---|---|---|
| 25.10 | разделы 0.5, 1; G0, G1, G2; голосование 2.5 | кэш `maggr_finds` | G2 или голосование, если приняты |
| 26–27.10 | G3, G4, раздел 5 | сеть MagGr, фолды | G3/G4, если приняты |
| 28–30.10 | сеть MagGr в табло | кэш и фолды ортофото | сеть MagGr, если принята |
| 31.10–04.11 | ортофото в табло; письмо по ImageNet | веса всех тайлов | ортофото вкл |
| 01–07.11 | — | аренда: курганы спутника, аэрофото | ветки по одной |
| 08–09.11 | заморозка двух версий A и B | — | A и B |
| 10–11.11 | только исправления | — | последние посылки до 23:59 11.11 |
| 12–17.11 | финальная версия, пакет экспертизы | — | 2 успешные на закрытой выборке |

## 8. Ожидания и стоп-правила

| Ветка | Пунктов балла, оценка |
|---|---|
| геофизика: отсечка, каналы, object_poly (классика) | +1…+3 |
| сеть MagGr и EtGr вместо классики | +3…+6 (у конкурента одна MagGr — +5,4) |
| архитектура: ортофото и снимки мелким шагом | +1…+4 |
| городища и фортификации: точка внутри | +0,2…+0,5 |
| курганы спутника и аэрофото на шаге 0,3 м, аренда | +1…+3 |
| **итого** | **0,43–0,50**, если ветки сработают; 0,37–0,42 — без сетей на геофизике и снимках |

0,6 этими ветками не набирается: для неё нужен F1 около 0,6 почти по всем классам (`docs/AUDIT_2026-10-24.md`, раздел 2). 0,5 — достижима: её набрали пять команд, и разрыв с ними — в ветках выше, а не в лидаре.

- База табло не совпала с `--expect` — стоп, разбор сборки.
- Честная приёмка не прошла — опыт записать и идти дальше; порог не снижать.
- Посылка упала — читать лог, тот же архив второй раз не грузить.
- До 11.11 23:59 нет посылки с принятыми ветками — всё остальное остановить за 48 часов.
- Публичный балл ниже предыдущего при том же честном балле — ветку не откатывать сразу: сначала `lb_counts.py`, какой тип данных просел.

## 9. Записи

После каждого опыта: JSON в `notes/` (с `honest`), строка в `docs/PLAN.md`, строка в `docs/AI_DECLARATION.md` без колонки «кто проверил». После каждой посылки: `lb_counts.py` и строка в `docs/LB_LOG.md`.
