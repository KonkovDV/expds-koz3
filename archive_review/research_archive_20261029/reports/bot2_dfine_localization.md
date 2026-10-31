# Бот 2 — D-FINE localization + почему A≡B — 29.10.2026 ~16:41 МСК

| Поле | Значение |
|---|---|
| **Цель** | (1) Объяснить exact A≡B на mock hotfix @ 1ec2d7c; (2) аудит/фикс координатной математики D-FINE. |
| **Гипотеза** | A≡B ⇒ D-FINE **не** в scored GeoJSON (b), а не «13 FP @191 м не вредят F1» (a). ~191 м — domain/модель, не CRS и не off-by-tile в порте. |
| **Критерий приёмки** | Тесты зелёные; stages dump готов к PC A2; geographic CRS guard; отчёт + QUEUE + EXPERIMENTS. Localization bugfix только если найден конкретный баг в math (есть: reject 4326). |
| **Бюджет** | ~1 ч CPU box. PC A2 (dump+stages): ≤10 мин GPU (после bot5 v4). |
| **Ветка** | `bots/bot2-e2e-v3-prep` @ **`fb66c89`** (поверх `1ec2d7c`). |

Связанные: `reports/bot1_v3_spor_diag.md`, `reports/bot2_v3_hotfix.md`, `reports/bot2_v3_regression.md`.

---

## 1. Почему A ≡ B (exact)

PC @ 1ec2d7c: A (enabled + drop_groups=[]) и B (enabled=false) → **ALL 0.312873 / sp_or 0.245064** (δ=0 по всем слоям).

| Кандидат | Вердикт | Доказательство |
|---|---|---|
| **(a)** 13 dets в выдаче, но ~191 м → F1 не меняется | **Отвергнут** | `tests/test_bot2_dfine_a_equals_b.py`: синтетические sp_dfine с conf≥thr переживают `finish` → writer GeoJSON как `markup_type=sp_or` (**threshold_group не пишется**); 13 далёких FP → +13 pred. Soft-match precision падает (штраф 0.5·FP). Exact 6-знак. равенство **несовместимо** с лишними pred. |
| **(b)** dets **не** доходят до scored output | **Принят** | Статический путь: `region_feats` → `state.add` → `finish`/`apply_thresholds(sp_dfine)` → `merge_within_markup` (далекие не схлопываются) → writer. Пустой вклад ⟺ `region_feats` вернул [] **или** всё срезано до/на thr. Класс `kurgany_povrezhdennye` ∈ scored sp_or — фильтра по «не в scored set» нет. |

**Где теряются (ранжир. по вероятности, без main_A.log):**
1. `require_thresholds` / load fail / weights → `SP_DFINE_UNAVAILABLE`, models=[] (pipeline break после первого участка).
2. Patch-Toml / BOM / битый `[thresholds_by_markup.sp_dfine]` → тот же UNAVAILABLE.
3. Exception в `region_feats` (глотается в `_sp_dfine_phase`).
4. Soft ceiling (на mock ~100 с / ceiling 1200 — **маловероятно**).
5. Wall A−B ≈ +20 с совместим с попыткой import/load без успешного add.

`kurgany_povrezhdennye` + `markup_type=sp_or` **входят** в слой sp_or скорера; отдельного drop по threshold_group в writer/metric нет.

---

## 2. EXPDS_DFINE_DUMP stages (OFF by default)

При `EXPDS_DFINE_DUMP=<dir>`:
- как раньше: `*__m*_boxes.json`, `*__gt_3857.geojson`
- **новое:** `stages.jsonl` (на участок) + `stages.json` (totals):  
  `n_raw_boxes → n_ellipses → n_features → n_would_pass_thr`, `n_models`, `thresholds_ready`, `n_sp_dfine_in_state`, `skip_reason`.

PC A2 с dump ответит однозначно: raw=0 vs pass_thr=0 vs in_state=0.

---

## 3. Localization audit

| Узел | Результат |
|---|---|
| `detect_scene` tiling | step=tile−overlap; offset window: cx_scene = x_off + cx_tile·T — тест OK; = bot-9 |
| `boxes_to_ellipses` | xyxy→центр/полуоси; affine a..f; UTM→3857 always_xy — roundtrip <1 м |
| flip_tta | только `bx[...,0]=1−cx` — OK |
| `utm_for` / Scene | **fix (ON):** geographic fallback/CRS (EPSG:4326) отвергается — иначе res_m=0.5° |
| OOF thr | **design only:** thr sp_dfine сейчас in-sample R_max bot-9; для R нужен OOF (не калибруем без данных на box) |
| ~191 м | не half-tile (320 м), не CRS; порт math совпадает с bot-9 `infer_region` → **domain/модель**, не баг offset |

Коммит: `fb66c899a6b534f6bbec06c7496736c3eaa4c3ac`.

---

## 4. Тесты (box, 16:41 МСК)

```
pytest tests/test_bot2_dfine_a_equals_b.py tests/test_bot2_dfine_localization.py \
  tests/test_bot2_dfine_dump.py tests/test_bot2_make_on_modes.py \
  + bot7_v3_flags (detect/region_feats/drop/default_off)
→ 25 passed
```

---

## 5. Следующий шаг

QUEUE: **A2 dump+stages** @ fb66c89 (ветка bots/bot2-e2e-v3-prep). Hotfix для v4 = **B** (уже в bot5). Localization model-side / OOF thr — backlog после A2 лога.
