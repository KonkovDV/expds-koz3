# HEAD_INTERFACE — ядро ConvNeXt-T U-Net (bot-9) ↔ голова центров (bot-2)

Итоговый согласованный интерфейс (28.10.2026, ~22:40 МСК). Код: `src/cnx_head.py` (адаптер, HEAD_API = 1),
`src/convnext_main.py` (ядро), снимок головы bot-2 — `src/vendor/centerhead/` (из git, не правится руками).

## 1. Источник головы bot-2 (вендоринг по sha)
- Репозиторий `/workspace/koz3_shared/bot-2/repo`, ветка `bot-2/centerhead`, коммит `788202a1b3b8eebdef70b6728faaea3154ee2031`
  (короткий `788202a`, по умолчанию в `build.py`: `CH_SHA = os.environ.get("CNX_CENTERHEAD_SHA", "788202a")`).
- `build.py` берёт файлы через `git -C bot-2/repo archive <sha> centerhead` (без `tests/`), то есть только чтение.
  В манифест вендора (`VENDOR_B64`, sha256 по файлам) пишутся `source "centerhead/*" = "git … 788202a1…:centerhead"`
  и `centerhead_commit`. Diff вендор-копии с рабочим деревом bot-2 = 0 (logs/build.log).
- Обновление от бота 2: новый коммит → `CNX_CENTERHEAD_SHA=<sha> python3 build.py` → пересобираются все 12 ядер.
  **prep перезапускать не нужно**: prep-манифест сверяет только `RELIEF_SHA` (sha `expds/*`, `expds_core.abi3.so`,
  `relief_ext.py`) и теги каналов (`convnext_main.py`, строка 330). Проверено: prep собран заново,
  затем smoke-из-prep с centerhead — rc=0 (logs/dry_prep.log, logs/dry_prep_centerhead.log).

## 2. Контракт головы (HEAD_API = 1, `cnx_head.py`)
Атрибуты: `name`, `stride`, `circle_r_m`, `out_keys`, `prob_keys`, `vec_keys`.

| метод | вход | выход |
|---|---|---|
| `build(ch, encoder, weights)` | число каналов, `"tu-convnext_tiny"`, `"imagenet"`/`None` | `nn.Module`, `forward(x)` → dict: `hm` (B,2,S/s,S/s) логиты; `wh` (B,1,·,·) log-радиус в пикс выхода; `off` (B,2,·,·); `mask` (B,3,S,S) логиты |
| `targets(objs, S)` | `objs` (N,4) `[x, y, diam_px, cls]` в пикселях кропа **после** D8 | numpy на сетке S/s: `hm`, `wh`, `off`, `reg_mask` |
| `loss(out, tgt, valid_fullres)` | выход модели, цели, valid (B,1,S,S) | `(Tensor, parts)`; для centerhead: `center_loss(...)["loss_center"]` (valid → `valid_to_stride`) |
| `untta(o, k, fl)` | dict без `mask` | обратный D8: hm/wh — flip + rot90(−k), off — `torch_untta_offsets` |
| `decode(maps, left, top, step·stride, valid_h, suppress_h)` | карты на сетке головы | таблица `X, Y, cls, score` (+ `radius_m`, `row`, `col`) |

Ядро владеет: seg-головой `mask` (3 класса: городища, укрепления, arkhitektury; BCE+Dice, полное разрешение),
её decode/untta, D8-аугментацией, copy-paste, тайлингом, сшивкой, OOF-экспортом.
Итоговый лосс: `head.loss(...) + mask_loss(out["mask"], m3, valid)`.

Головы: `centerhead` (по умолчанию; `UnetCenterNet` bot-2, крючок на `decoder.blocks[2]` smp: 64 канала,
128×128 при кропе 512 — проверено прямым прогоном `build_unet_centernet(5, "tu-convnext_tiny", None, cfg_with(stride=4))`:
`block 2 feat_ch 64`, hm (1,2,128,128), wh (1,1,128,128), off (1,2,128,128), mask (1,3,512,512)) и `heat`
(старая встроенная, stride 1, оставлена для сравнения). Выбор: `CNX_HEAD=centerhead|heat`, по умолчанию при сборке
`CNX_HEAD_DEFAULT` (= centerhead).

Конфиг centerhead: `centerhead.config.DEFAULTS` ← общие ключи ядра (`sigma_k` 0.15, `sigma_min_px` 1.0,
`default_diam_m` 8.0, `min_score` 0.05, `topk = max_det` 20000; совпадают с DEFAULTS bot-2, кроме topk)
← env `CNX_CH_<KEY>` (например `CNX_CH_SIGMA_MODE=linear`, `CNX_CH_STRIDE=1`).

## 3. Пайплайн ядра (что где вызывается)
- `WinDS.__getitem__`: D8 кропа → `d8_points` (≡ `d8_objs` bot-2, max |Δ| = 0.0, logs/test_head.log) → `head.targets`.
- Тайлинг инференса: `tile_origins` bot-2 (последний тайл кратен stride); ядро проверяет, что `crop` и
  `infer_stride` кратны stride головы.
- Два аккумулятора: полное разрешение (mask, `cnt`) и сетка головы (hm/wh/off, `cnt4`).
  `valid = (a[0]>0) & (cnt>0)`, `valid_h = block_any(valid, s) & (cnt4>0)` (≡ `valid_to_stride`, logs/test_head.log).
- sigmoid применяется к `mask` и `prob_keys` (hm); `wh/off` сохраняются сырыми.
- Подавление курганов внутри городищ: `sup = block_mean(gor, stride) > 0.5` → `head.decode(..., valid_h, sup)`.
- Круг-выход: `CIRCLE_R` = env `CNX_CIRCLE_R` → иначе `head.circle_r_m` (0.5 м у centerhead) → иначе `CFG` 1.0 (heat).

## 4. OOF-карты для реранкера (обязательный выход)
Файл на участок и задание: `maps/maps_cnx_{tag}_{region}.npz` (`np.savez_compressed`, `CNX_SAVE_MAPS=1` по умолчанию),
`tag` = `{variant}_f{fold}`, например `c5-in_f4`. Карты пишутся только для тест-фолда модели (честный OOF).

| ключ | dtype | форма | смысл |
|---|---|---|---|
| `valid` | u8 | (H,W) | покрытие лидаром и тайлами, полное разрешение |
| `valid_h` | u8 | (H/s,W/s) | то же на сетке головы |
| `hm` | float32 | (2,H/s,W/s) | вероятности центров (0 = tselye, 1 = povr) |
| `wh` | float16 | (1,·,·) | log-радиус, пикс выхода (сырой) |
| `off` | float16 | (2,·,·) | субпиксельный сдвиг (сырой, уже un-TTA) |
| `mask` | u8 | (3,H,W) | p·255: городища, укрепления, arkhitektury |
| `meta` | JSON-строка | — | `region, fold, family, tag, head, head_api, stride, left, top, step, step_head, crs, H, W, tags, enc, keys` |

`left/top` — левый верхний угол пикселя (0,0) в `crs` (родная UTM участка, например EPSG:32637); `step` = 0.5 м,
`step_head` = `step·stride` = 2.0 м. hm хранится в float32: в u8 decode давал 1699 вместо 428 кандидатов, в f16 ≈ 1 % лишних,
в f32 — точное совпадение (tests/check_maps.py → logs/check_maps.log: 428/428 и 1669/1669).
Размер в сухом прогоне: 0.1 и 0.4 МБ на участок (logs/dry_run_centerhead.log).

`region_fold.json` (пишется до старта заданий): `region_fold` {участок → фолд}, `region_family` {участок → семья},
`jobs`. 59 участков, фолды {0: 6, 1: 1, 2: 13, 3: 18, 4: 21} (logs/check_maps.log). Для `cnn_fold_of` бота 2.

Векторный OOF: `oof_cnx_{tag}.geojson`, EPSG:3857, Polygon, поля как в `/workspace/rerank/stack_arch.geojson`
(`class_name, confidence, crs, fid, markup_type, original_crs, region_name`; logs/merge_test.log); слияние — `src/merge_oof.py`.

## 5. Самопроверка и тесты
- В каждом ядре при старте: `check_head` (атрибуты, формы, конечный лосс). Лог centerhead:
  `голова: centerhead {… stride 4, hm [2,2,16,16], mask [2,3,64,64] …}` (logs/dry_run_centerhead.log).
- `tests/test_head.py` → logs/test_head.log (rc=0): d8 hm/off Δ = 0; `tile_origins` идентичны bot-2; decode 8/8 объектов
  на 0.0 м (centerhead) / 8/8 при max 0.2693 м (heat); лосс 2.4814 (hm 2.1431, wh 0.5971, off 0.2786); без mask_loss
  градиента нет только у decoder.blocks.3/4 и mask_head (ожидаемо).
- `tests/check_maps.py` → logs/check_maps.log: decode функцией bot-2 по npz воспроизводит OOF.
- `src/validate_oof.py` (env `VAL_R`) → logs/validate_oof_v2.log: схема без ошибок; контроль «идеальные круги» r = 0.5 м:
  0.9999994444 (18 GT) и 0.9999991667 (3 GT).

## 6. Ответ боту 2 по п.5
1. **Признак stride 4, `decoder.blocks[2]`, 64 канала, 128×128 при 512 — ПОДТВЕРЖДАЮ.** Проверено прямым прогоном (см. §2).
2. **sigma — ПОДТВЕРЖДАЮ ваш дефолт `centernet` (min_overlap 0.7), минимум 1 пикс выхода (2 м), радиус GT = √(S/π).**
   `linear` 0.15·D оставлен переключателем `CNX_CH_SIGMA_MODE=linear`; на stride 4 для типичных D 8–20 м обе формулы
   дают σ порядка 1 пикс выхода, поэтому разница несущественна (оценка, не измерено). heat-голова сохраняет linear.
3. **min_score 0.05, peak_pool 3 (6 м), NMS `max(1 м, 0.8·min(r_i,r_j))` в одном классе, обе гипотезы класса в точке —
   ПОДТВЕРЖДАЮ** (используется ваш `decode`, `nms_cross_class=False`). Отличие только у legacy-головы heat:
   peak_k 11 пикс и argmax класса — для сравнения, не для реранкера.
4. **Круг-выход 0.5 м — ПОДТВЕРЖДАЮ** для centerhead (`head.circle_r_m`); heat остаётся 1.0 м; переопределение `CNX_CIRCLE_R`.
5. **OOF-карты — ДА, с отличием по dtype:** hm хранится в **float32**, а не float16 (f16 даёт ≈ 1 % лишних пиков из-за плато,
   logs/check_maps.log); wh/off — float16; маски — u8 (p·255) на полном разрешении; плюс valid/valid_h и meta с
   `left, top, step, step_head, crs`. Формат — §4.
6. **Фолды — ДА, `region_fold.json` с явными `region_fold` и `region_family`** (честные семейные фолды, те же, что в FOLDS ядра).
7. **Каналы c3 (lrm_s, lrm_l, slope) и c5 (+svf_r, opp_r) — ОКОНЧАТЕЛЬНЫ** для этого раунда; имена тегов — в `meta.tags`.
   12 каналов не делаем.
8. **AMP fp16 — ПОДТВЕРЖДАЮ**; адаптер приводит выход к `.float()` перед `center_loss`, то есть весь лосс (и wh/off) в float32.
9. **arkhitektury — ПОДТВЕРЖДАЮ**, остаётся каналом 2 seg-головы `mask`.
10. **SpOr-согласие — ОТЛИЧИЕ: в этом ядре SpOr нет.** SpOr-выход — у бота 10 (его стек) и в ядре
    dfine_spor; соединять по участкам на этапе реранка у бота 2/10, не здесь.

По §6 INTERFACE.md (wh 1 канал, dict-лосс, маски на полном разрешении, кратный stride последний тайл, `floor(x)`) — всё принято
без возражений и реализовано.
