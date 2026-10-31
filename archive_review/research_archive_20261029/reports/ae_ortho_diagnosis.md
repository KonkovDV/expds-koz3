# Диагноз: почему ae/ortho recall ≈ 0 на LB (база eb268d6 / 0.41238)

_МСК: 2026-10-29 20:27 (Europe/Moscow). Репо tip на момент разбора: local `/workspace/koz3-repo` @ `eb268d6`. LB: `reports/lb/scoring_v2_eb268d6_0.41238.log`._

## Краткий вывод

На сдаче **eb268d6 аэрофото почти пустое по конструкции**, а не из‑за одного порога:

1. **Единственная ae‑сеть в `configs/default.toml` — `ae_fort_mask_r18.pt` (только `fortifikatsii`) с τ = 0.998215.** Курганов/городищ/архитектуры на ae эта сеть не выдаёт.
2. **D-FINE Ae в пути инференса eb268d6 нет** (`dfine_aeor.py` / `[dfine_aeor]` появились на ветках bot5/bot3, default OFF). Веса `/workspace/kaggle/out/dfine-ae-ft*` в сабмит не входят.
3. **Ортофото (`ортофото` → markup `or`) обрабатывается отдельно**, не маршрутизируется в ae‑модель. Модель `or` = `surf_arch_r18.pt` **только `arkhitektury`** (τ 0.065565). Курганы/форты/городища на Or в пайплайне не ищутся.
4. **Копии li→ae почти убиты порогами** (`li_ae.tselye = 1.000001`, `li_ae.povr = 0.993061`, `fusion.povr_min = 0.99`). На LB‑участках ae без LAS копии всё равно нечем делать.
5. **Крупные объекты (>5k м²):** не «max‑size фильтр» ( `post.split` выключен). Потеря из‑за отсутствия ae/or‑детектора крупных классов + крошечного окна ae_fort (128×0.5 м ≈ 64 м) + `gorodishcha` у D-FINE OFF (τ=1.01).

Mock ae F1 **0.0956** (59 pred / 252 GT) vs LB ae R **0.013** — mock всё же видит часть ae (форты + копии на совместном покрытии); публичный LB ae почти без совместного лидара → потолок текущего пайплайна ≈ ноль.

---

## 1. Путь кода ae/ortho @ eb268d6

| Шаг | Что происходит |
|---|---|
| Имя файла | `expds/naming.py`: коды `Ae`/`Or` + ключи `аэрофото`/`немецк`/`luft` → Ae; `ортофото` → Or. Markup: `ae` / `or`. |
| Каталог | `catalog.Region.rasters[markup]` — ae и or **раздельные** списки. |
| Оптика | `pipeline.optical_feats`: цикл `("ae", "sp_or", "or")` → сегментеры из `[models].<markup>`. |
| Модели | **ae** = `{ae_fort_mask_r18.pt, classes=[fortifikatsii]}`; **or** = `{surf_arch_r18.pt, classes=[arkhitektury]}`; **sp_or** = gor+focal+arch. |
| Пороги early | `output_floors(cfg, mt)` = `[thresholds_by_markup.<mt>]` → в `probs_to_features(..., min_conf=floors)` отсечка **до** полигонизации. |
| Fusion | `li_to_ae=true`: уверенные li‑курганы копируются в markup `ae` (группы `ae` / `li_ae`). |
| Post | `apply_thresholds` по группам; `cap_features` ≤ 20000; `split_large` **не включён** в default. Микрокруги курганов 0.5 м, twins, rerank_arch — не про ae‑recall. |
| D-FINE | **Отсутствует** на eb268d6. На bot5: `[dfine_aeor] enabled=false`, окна Ae 640/160 @0.5 м, Or 1024/288 @0.5 м, NMS IoU 0.5, `max_det=100`, `max_objects=20000`, fuse WBF/NMS. |

### Пороги (default.toml), релевантные ae/or

```
[thresholds_by_markup.ae]   fort=0.998215  povr=0.99053  tsel=0.802255
[thresholds_by_markup.or]   arch=0.065565
[thresholds_by_markup.li_ae] povr=0.993061  tsel=1.000001   # «выкл»
[fusion] li_to_ae=true  povr_min=0.99  tselye_min=0.8
```

`ae_r18.pt` лежит в `models/`, но **в конфиге не подключён** (заменён fort‑mask; notes: «вне обучения не мерилась»).

### Тайлинг / масштаб

| Источник | res | tile | окно на земле |
|---|---|---|---|
| ae_fort (кэш обучения) | 0.5 м | 128 | **≈64 м** |
| D-FINE Ae (когда включён) | 0.5 м | 640, overlap 160 | **≈320 м** |
| D-FINE Or | 0.5 м | 1024, overlap 288 | **≈512 м** |
| min_area_px в `probs_to_features` | — | — | 3 px (~0.75 м² @0.5 м) — не душит крупные |
| MAX_FEATURES | 20000 | — | на LB ae ~10–20 pred — не лимит |

Крупный объект 5–10 тыс. м² (~70–100 м сторона) **не помещается в один тайл ae_fort** → фрагменты с низкой conf + τ 0.998 → 0. D-FINE 320 м закрывает большинство, но gorodishcha сейчас OFF.

### Class mapping (D-FINE Ae FT)

`id2label`: 0 tselye, 1 povr, 2 gorodishcha, 3 fortifikatsii, 4 arkhitektury — совпадает с метрикой. OOF calib: gor/arch **τ=1.01 OFF** (bot‑10).

---

## 2. LB vs mock vs OOF

| Источник | ae | ortho | крупные |
|---|---|---|---|
| **LB** scoring_v2_eb268d6 | P 0.091 / **R 0.013** (~1 TP на ~75–100 GT) | P 0.37 / **R 0.14** | 5–10k **R=0**; >10k R=0.29 |
| **Mock** v2 docker eb268d6 | F1 **0.0956** (59/252) | в mock‑строках отдельно нет | — |
| **D-FINE Ae OOF** (bot‑9 analysis на koz3-dfine-ae; FT detail согласуется) | при τ=0.05: R_all **0.47** (P 0.0005!); τ=0.5: R 0.156 P 0.05; calib F2: tselye **0.028**, povr **0.173**, fort **0.184** | — | >5k ae: **43/52** hit @0.05, **21/52** @0.3; все 52 имеют пересечение с pred |

**Потолок модели vs пороги:** D-FINE **находит** большую долю ae‑GT при низком τ (в т.ч. 43/52 крупных), но при калиброванных τ полнота падает, а tselye почти мёртв (F2 0.028). Это **и потолок качества (особенно tselye/gor), и убийственные пороги/отсутствие модели в сдаче**. Текущий eb268d6 вообще не использует этот сигнал → LB ae R≈0 — **потолок пайплайна сдачи**, не D-FINE.

Исторический ae_fort @ τ 0.998: 714 kept / 94 TP из 696 fort GT на train (notes/ae_fort_mask) — на LB ae почти нет попаданий ⇒ либо мало fort‑GT в публичном ae, либо conf на LB ниже полки.

---

## 3. Корневые причины (ранг по ожидаемому Δ LB)

Оценка Δ — порядок величины к `final_score` 0.412; честно валидировать mock + honest‑10, не пушить на LB без гейта.

| # | Причина | Ожид. Δ LB | Уверенность |
|---|---|---|---|
| **1** | **Нет ae‑детектора курганов/мультикласса в сдаче** (только fort‑mask @0.998). Включить `[dfine_aeor]` Ae + веса f4pack/`dfine_ae` + OOF τ | **+0.02…+0.05** (ae R 0.013→~0.10–0.25 при контроле FP; mock ae 0.10→0.15–0.25) | средняя: F2 tselye слаб, риск FP |
| **2** | **Or только архитектура** — нет kurgan/fort/gor на ортофото; LB Or R=0.14 уже не ноль (arch), но классы с весом 30+20+15+5 % пустые | **+0.01…+0.03** после Or‑весов / общего optical FT | средняя; dfine_or ещё не в прод‑гейте |
| **3** | **Крупные (>5k)**: gor OFF у D-FINE; окно ae_fort 64 м; нет large_objects в сдаче (bot4 flag OFF; mock ON дал −0.016!) | **+0.005…+0.02** на bin 5–10k (сейчас 0); осторожно — bot4 ON регрессил mock | низкая–средняя |
| **4** | **li_ae / fusion полки** убивают копии; на LB ae‑only почти не помогают | **+0…+0.005** на LB; больше на mock/train с LAS | высокая как факт, низкая как LB‑рычаг |
| **5** | **ae_fort τ 0.998** — можно чуть опустить (0.99–0.995) без D-FINE | **+0.002…+0.01** | низкая на LB (мало fort?) |

Не главные на ae: `cap_features`, `min_area_px=3`, `split_large` (OFF), NMS D-FINE (ещё не в сдаче), нехватка времени (инференс 3m48s / 30 мин).

---

## 4. Три быстрых фикса (с валидацией)

### Fix A — включить D-FINE Ae в кандидате (не main), веса f4

**Зачем:** единственный мультиклассный ae‑сигнал с OOF R≫0.013.

**Изменение (ветка bot3/bot5, не main):**
```toml
[dfine_aeor]
enabled = true
markups = ["ae"]          # сначала только ae
weights = { ae = ["models/dfine_ae"] }
thresholds_file = "models/dfine_aeor/thresholds_oof.json"
# стартовые τ из thresholds_ae_oof.json (tselye 0.45, povr 0.5, fort 0.35; gor/arch 1.01)
```
Положить веса из `/workspace/kaggle/out/dfine-ae-ft-f4pack/` (sha `249b86fb…`, полный FT fold4) предпочтительнее final `3e619878…` (1 FT‑итерация с old f3).

**Честная проверка:**
1. Docker mock OFF vs ON (`pantum/n7q:20261024`): Δ ae F1, ALL не −>0.01; `assert` n_raw ae > 0.
2. honest‑10 global: Δ ALL > 0, ни один слой не −>0.01.
3. OOF sweep: кривая P/R по классам на `oof_dfine_ae.geojson` / fold files — не только in‑sample τ.
4. **Не** пушить LB и **не** мержить в main до ACCEPT.

Ожидание: mock ae +0.03…+0.10; LB ae R ×5–10 при удачном τ; риск — FP ratio (сейчас LB FP 1.14).

### Fix B — ослабить ae_fort τ + (опц.) li_ae на mock‑гейтах

**Зачем:** 0‑code‑risk рычаг на уже сданной сети; быстрый A/B.

**Изменение:**
```toml
[thresholds_by_markup.ae]
fortifikatsii = 0.995   # было 0.998215; сетка 0.99 / 0.995 / 0.997
# опционально для участков с LAS (больше mock, чем LB):
[thresholds_by_markup.li_ae]
kurgany_povrezhdennye = 0.95
kurgany_tselye = 0.85
[fusion]
povr_min = 0.9
tselye_min = 0.7
```

**Проверка:** scoreboard / mock e2e OFF→ON; смотреть ae F1 и FP; honest‑10. Если Δ≈0 на mock без LAS — ожидаемо (копии не работают) → не тратить слот LB.

### Fix C — крупные: не оставлять gor OFF + большее окно / не резать крупные при merge

**Зачем:** bin 5–10k R=0; OOF D-FINE бьёт 43/52 крупных @0.05, но gor @τ≥0.3 = 0/16.

**Изменение (поверх Fix A):**
- В `thresholds_oof.json` для `gorodishcha` поставить τ из OOF F_β sweep (не 1.01), **только для площади ≥5000 м²** (area‑aware, идея bot4) — иначе FP.
- Не включать целиком `[large_objects] enabled=true` без ablation: mock ON дал **−0.016**. Брать только под‑флаги, которые bot4 ablation подтвердит.
- Убедиться, что ae_fort / merge не дробит крупный fort (D-FINE окно 320 м уже лучше 64 м).

**Проверка:** OOF recall по `_recall_by_area` и `gt_gt5000_ae`; mock + honest‑10; отдельно метрика bin 5–10k / >10k если есть в локальном скорере.

---

## 5. Обрабатывает ли пайплайн «ортофото»?

**Да.** `naming`: ключ `ортофото` → код `Or` → markup `or`. `optical_feats` гоняет `or` своей моделью. LB печатает **аэрофото** и **ортофото** раздельно — это те же markup `ae` / `or`.

Ограничение: **or‑ветка узкая (только arch)**. Исторически перенос spor_arch на Or давал 0 попаданий (STRATEGY); текущий `surf_arch` + низкий τ даёт LB Or R=0.14. Чтобы поднять Or дальше — нужен мультикласс (dfine_or / общий optical), не понижение arch‑τ в одиночку.

---

## 6. Артефакты, на которых стоит диагноз

- Код/конфиг: `/workspace/koz3-repo` @ eb268d6 (`configs/default.toml`, `expds/{naming,pipeline,fusion,infer_nn,post,constants}.py`)
- LB: `/home/box/shared/koz3/reports/lb/scoring_v2_eb268d6_0.41238.log`
- Mock: `pc/results/bot2_v2_base_docker_eb268d6/scores_mock.json` (ae 0.0956)
- OOF D-FINE: `/workspace/kaggle/out/dfine-ae-ft/{thresholds_ae_oof*.json,dfine_report.json,oof_*.geojson}`; сводка bot‑9 `…/dfine_ae_oof/ae_oof_analysis.json` (R@τ, area bins, 52 крупных)
- Веса: `/workspace/kaggle/out/dfine-ae-ft-f4pack/` (рекомендуемый pack)
- Предшествующие отчёты: `reports/bot3_aerial_recall.md`, `bot3_dfine_aeor_eval.md`, `bot4_large_objects.md`

## 7. Что не делать

- Не коммитить `expds/geo_pixgbm.py`, не пушить main, не слать LB из этого разбора.
- Не включать `[large_objects]` целиком без ablation (уже −0.016 mock).
- Не считать mock ae 0.0956 «починенным LB» — домен LB ae жёстче (мало LAS‑копий).
- Не печатать токены.
