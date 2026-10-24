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

## 3. Сеть на картах MagGr (26–30.10, ночи GTX 1650)

Задача — та же, что локализация объектов точками: карта центров с фокальной потерей, как у сданной карты курганов спутника. Всё уже есть в коде; сдача запускает сеть строкой `[models] maggr` (`pipeline.geo_model_feats`, группа `maggr_nn`).

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
- Шаг 0,1 м — проверить по `geo_probe` (поле `res`); если карты 0,05 м — `--res 0.05 --tile 256`.
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

## 6. Курганы спутника и аэрофото — только на арендованной карте (с 01.11)

Опыты на GTX 1650 исчерпаны: ResNet34 с батчем 8 «не оставил кругов» — это недообучение, а не потолок. Аренда GPU разрешена ответами организаторов, если доступ только у команды, копии под контролем и удаляются после (`docs/DATA_PROTOCOL.md`, п. 3). На карте 24 ГБ:

- карта центров курганов спутника: `--encoder resnet50`, `--batch 32`, тайл 256 на шаге 1 м и отдельно 512 на 0,5 м, 40 эпох, `--ema 0.999`;
- аэрофото: тот же рецепт, шаг 0,5 м;
- проверка — те же пять фолдов, честная приёмка.

Лицензия весов: если организаторы запретят ImageNet, запасной путь — энкодер DINOv2 (Apache-2.0 на код и веса), а не обучение с нуля. DINOv3, SAM 3, SegFormer `mit_*`, ultralytics — нет.

## 7. Календарь

| Даты | CPU | GPU ночью | Посылки |
|---|---|---|---|
| 25.10 | раздел 1; G0, G1, G2 | кэш `maggr_finds` | G2, если принят |
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
| геофизика: отсечка, каналы, object_poly | +1…+3 |
| сеть MagGr | +1…+2 сверх классики |
| архитектура: ортофото и снимки высокого разрешения | +1…+3 |
| городища и фортификации: точка внутри | +0,2…+0,5 |
| курганы спутника и аэрофото на аренде | +0,5…+2 |
| **итого** | **0,37–0,42** |

0,6 этими ветками не набирается: для неё нужен F1 около 0,6 почти по всем классам (`docs/AUDIT_2026-10-24.md`, раздел 2).

- База табло не совпала с `--expect` — стоп, разбор сборки.
- Честная приёмка не прошла — опыт записать и идти дальше; порог не снижать.
- Посылка упала — читать лог, тот же архив второй раз не грузить.
- До 11.11 23:59 нет посылки с принятыми ветками — всё остальное остановить за 48 часов.
- Публичный балл ниже предыдущего при том же честном балле — ветку не откатывать сразу: сначала `lb_counts.py`, какой тип данных просел.

## 9. Записи

После каждого опыта: JSON в `notes/` (с `honest`), строка в `docs/PLAN.md`, строка в `docs/AI_DECLARATION.md` без колонки «кто проверил». После каждой посылки: `lb_counts.py` и строка в `docs/LB_LOG.md`.
