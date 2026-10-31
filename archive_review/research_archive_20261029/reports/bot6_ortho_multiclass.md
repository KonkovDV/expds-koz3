# Бот 6 — Or multiclass D-FINE (анализ + дизайн, без коммитов) — 29.10.2026 ~20:40 МСК

| Поле | Значение |
|---|---|
| **Цель** | Поднять LB > 0.41238 за счёт Or multiclass D-FINE за флагом OFF по умолчанию; честный nested OOF τ; сравнение с surf_arch; оценка e2e при `merge_union=false`. |
| **Гипотеза** | (а) OOF geojson **без** систематического CRS/pixel-bug (в отличие от SpOr A3 как domain-риск); (б) provisional p25 τ **не** honest; (в) полезен только `arkhitektury`, остальное OFF; (г) добавление dfine_or рядом с surf_arch при низком τ портит Or F1. |
| **Критерий приёмки (будущий код)** | Ветка `bots/bot6-ortho-*` от **`ed73c5e`** (не ветка bot5). Cherry-pick в v5: **ΔALL>0** и **ни один слой не падает >0.01** (PROTOCOL §5). Новое OFF by default. |
| **Бюджет** | Box CPU-only анализ (~1 ч); без train / PC-Kaggle e2e / push / сообщений другим ботам. |
| **Владение** | Or-модель + honest OOF + multiclass — **бот 6** (steering 29.10). |

Связанные: `reports/bot2_dfine_a3_localization.md`, `reports/ae_ortho_diagnosis.md`, `reports/bot5_v4.md`, `scripts/analyze_dfine_aeor_or_localization.py`, артефакты `reports/bot6_ortho_work/`.

---

## 0. RED FLAG первым: локализация OOF vs A3

| | OOF D-FINE **Or** (этот отчёт) | bot2 A3 SpOr D-FINE |
|---|---:|---:|
| n dets | **1228** | 13 pass_thr (из 2506 raw) |
| median dist_any → GT | **239 м** | **≈388 м** |
| soft_tp@2 м | **31** | **0** |
| within 50 / 200 м | 277 / 528 | 1 / 4 (pass_thr) |
| mean (dx,dy) | (+84, +47) м, ‖mean‖≈96 | (+71, −200), ‖mean‖≈212 |
| std (dx,dy) | **(216, 212)** ≫ mean | (376, 420) ≫ mean |
| residual median после mean | 230 м | ≈306 м |

**Вердикт geo:** **нет** постоянного CRS / axis-swap / half-tile / 1024→geo бага на пути Or OOF. Soft TP@2 м > 0 и std ≫ mean → разброс **domain/модель/порог**, как у A3 по характеру, но Or заметно лучше (есть близкие попадания). Сравнение A3↔OOF: A3 — **SpOr** domain fail; Or OOF — тот же класс риска, не доказательство бага в `boxes_to_ellipses` / WarpedVRT 1024/288@0.5 м.

Детали: `reports/bot6_ortho_work/loc/summary.json`, `offset_deep.json`.

**Следствие:** threshold work **разрешён** (geo не блокирует). Главный риск — качество матча по официальному soft_match и FP при низком τ.

---

## 1. Веса: SHA256 verify

Источник: Kaggle `koz3-dfine-or`; копия `/workspace/kaggle/pulls/koz3-dfine-or` + `SHA256SUMS`. Проверка `/workspace/kaggle/out/dfine-or` против pulls:

| path | bytes | sha256 | out vs pulls |
|---|---:|---|---|
| `dfine_f0/model.safetensors` | 125001980 | `82c8a47a2ad0d0b12fd5467b38592d31910114ad4029242d933f2e807c517d9c` | **OK** |
| `dfine_f1/model.safetensors` | 125001980 | `7fcb58645366819e83f125b0078a5fdc0b4b896f1c2a1b0b3c995820586b6744` | **OK** |
| `dfine_f2/model.safetensors` | 125001980 | `29dcd3c633c31aa153abe420bde59766de25fa662b4b2b3065f7a78dacd0ed2e` | **OK** |
| `oof_dfine_or.geojson` | — | `5c8758f33e6c7e1fa79b3dfb9576201a2574dbbe7d72e545838f47ea502fd45f` | **OK** |

`sha256sum -c SHA256SUMS` — все файлы OK. Совпадает с `docs/WEIGHTS_MANIFEST_v4.md`.

Folds leave-region-out: **f0** 008/014/093; **f1** 033/056; **f2** 185. Train 3×2400 iters ~5333 с. OOF 1228: arch 1110 / fort 50 / gor 44 / tsel 17 / povr 7.

---

## 2. Пороги: provisional vs nested-CV (официальный soft_match)

GT: `gt_train_v2.geojson` markup=`or` → **184** объекта (arch 178 / gor 4 / fort 2) на 6 Or-участках. Метрика: `materials/compute_metrics.F1Calculator._soft_match` (cov≥0.5 + representative_point ∈ GT, UTM).

**Honest nested** (τ на фолдах ≠ te → score на te; pooled):

| | wF1 (веса классов GT) | arch F1_koz | gor/fort/kurgan |
|---|---:|---:|---|
| Nested pooled | **0.0367** | 0.110 (tp32 fp744) | **0 TP** → держать OFF |
| Deploy = median fold τ | **0.0487** | **0.146** (tp25 fp279) @ arch **0.2** | все **1.01** |
| Provisional p25 | **0.0360** | 0.108 (tp33 fp801) | FP: gor33 fort38 tsel13 povr6, **0 TP** |
| Uniform arch-only @0.2 | 0.0487 | 0.146 | — |
| Uniform arch-only @0.5 | 0.0423 | 0.127 (P0.32 R0.07) | — |

**Provisional** (`models/dfine_aeor/thresholds_or_oof_provisional.json`: tsel 0.0544, povr 0.05311, gor 0.05943, fort 0.05845, arch 0.07682) = **p25 conf, не honest F1**. Как baseline рядом с nested: **хуже** deploy и засоряет FP не-arch классами.

**Рекомендуемые honest τ (черновик, не в TOML пока):**
```json
{"dfine_or": {
  "arkhitektury": 0.2,
  "gorodishcha": 1.01,
  "fortifikatsii": 1.01,
  "kurgany_tselye": 1.01,
  "kurgany_povrezhdennye": 1.01
}}
```
(альтернатива e2e-safe рядом с surf_arch: arch **0.5**, см. §4).

JSON: `reports/bot6_ortho_work/oof_nested_scores.json`.

---

## 3. Сравнение с surf_arch (Falhan + Герите, n_GT arch=98)

| Источник | thr | tp / fp / fn | P / R | F1_koz |
|---|---|---:|---:|---:|
| **surf_arch** (baseline or) | 0.065565 | 32 / 81 / 66 | 0.28 / 0.33 | **0.375** |
| dfine_or deploy | arch 0.2 | 25 / 83 / 73 | 0.23 / 0.26 | 0.304 |
| dfine_or provisional | p25 | 33 / 405 / 65 | 0.08 / 0.34 | 0.198 |
| **union** surf+dfine@0.2 | — | 36 / 185 / 62 | 0.16 / 0.37 | **0.318** (−0.057 vs surf) |
| **union** surf+dfine@0.5 | — | 34 / 98 / 64 | 0.26 / 0.35 | **0.376** (~0 vs surf) |

Перекрытие TP: surf 32, dfine@0.2 25, **overlap 21**, surf_only 11, dfine_only **4**. При @0.5: dfine_only **2**.

**Вывод:** один D-FINE Or **слабее** surf_arch на arch. Уникальный recall мал. При `merge_union=false` (оба источника остаются) низкий τ **роняет** Or arch F1; высокий τ (~0.5) ≈ нейтрален.

---

## 4. Fusion e2e (`merge_union=false`)

Код (`expds/dfine_aeor.py`, `pipeline._dfine_aeor_phase`, `watchdog._aeor_split`):

- `dfine_or` идёт **изолированной** короткой цепочкой (свои пороги → `merge_within_markup` → cap остатка лимита).
- `surf_arch` / rerank / twins **не** видят dfine-кандидатов.
- Склейка D-FINE только внутри своих кругов: `cluster_circles merge_m=4`.
- **Нет** межисточникового merge с surf_arch.

**Эффект по умолчанию:** детекции **добавляются** к Or-слою. При provisional / arch@0.2 — ожидаемый **Δ Or arch < 0** на train-подобных участках (см. union −0.057). Для гейта PROTOCOL это риск падения optical/arch и, через вес arch=10, final_score.

**Дизайн включения (когда дойдём до кода):**
1. `[dfine_aeor] enabled=false` по умолчанию; `markups=["or"]` для Or-only слота.
2. τ: arch ≥ **0.5** (или nested 0.2 **только** если e2e mock покажет Δ>0 — маловероятно без dedupe).
3. Не-arch классы **1.01** (gor уже 1.01 в default.toml).
4. Опционально позже: dedupe vs surf_arch / `merge_union`+dedupe — отдельный эксперимент, не default.
5. Не пушить в ветку bot5; база **`ed73c5e`**.

---

## 5. Оценка вклада в LB

База LB `eb268d6` = **0.41238** (`reports/lb/scoring_v2_eb268d6_0.41238.log`):

- ортофото (auxiliary): **P 0.37 / R 0.14**
- ae R **0.013** (отдельная проблема; Or multiclass её не лечит)
- final_score = class-weighted F1; Or сейчас почти только **arkhitektury** через `surf_arch_r18` (τ 0.065565)

| Сценарий | Ожид. Δ LB | Уверенность |
|---|---:|---|
| Or ON + provisional τ | **≤ 0 … −0.005** (FP flood) | высокая как риск |
| Or ON + arch@0.2 + rest OFF | **−0.002 … +0.001** | средняя; union вреден |
| Or ON + arch@0.5 + rest OFF | **≈ 0 … +0.001** | средняя; почти нейтрально |
| Or замена surf_arch (не в коде) | ? | не оценивали e2e |
| Multiclass gor/fort/kurgan на Or | **≈ 0** сейчас (0 soft_match TP) | высокая |

Потолок «+0.01…0.03» из `ae_ortho_diagnosis` для Or multiclass **не подтверждён** текущим OOF: non-arch мёртв по официальному матчу; arch не бьёт surf_arch.

---

## 6. Дизайн (когда писать код)

| Элемент | Решение |
|---|---|
| Ветка | `bots/bot6-ortho-<short>` от **`ed73c5e`** |
| Флаг | `[dfine_aeor] enabled=false`; для прогона Or-only: `markups=["or"]`, `weights.or=[dfine_f0..f2]` |
| thresholds_file | новый `thresholds_or_oof_nested.json` (не provisional); TOML `dfine_or.gorodishcha=1.01` уже есть |
| assert | сохранить hard-FAIL `assert_dfine_aeor_or_nonzero` при ON |
| Гейт | Docker `pantum/n7q:20261024`: OFF vs ON; ΔALL>0; no layer drop>0.01; wall ≤ budget |
| Не делать | push в bot5; train на box; включать Ae slot без весов; provisional в прод |

---

## 7. Missing / вопросы к bot5 (и владельцу)

**Missing**
1. PC mock e2e Or-only ON с nested τ (QUEUE c′) — box не гоняет Docker GPU.
2. Полный surf_arch OOF на всех 6 Or-участках (есть только Falhan+Герите dumps).
3. Honest-10 / layer score после включения.
4. Ae FT веса — вне скоупа Or; не блокируют Or-only слот.

**Вопросы bot5**
1. Согласны ли заменить provisional в `models/dfine_aeor/` на nested arch@0.2/0.5 + rest 1.01 **до** Docker c′, или c′ сначала гонять с provisional только как negative control?
2. Нужен ли в v5 слот Or-only при ожидаемом Δ≈0, или ждать dedupe/replace-surf дизайн от bot6?
3. `gor` на Or остаётся 1.01 (бот-10/gor_dense) — подтвердить, что multiclass Or не должен снимать этот запрет без нового замера.

---

## 8. Файлы работы

| Путь | Содержание |
|---|---|
| `reports/bot6_ortho_work/loc/` | localization summary + offset_deep |
| `reports/bot6_ortho_work/oof_nested_scores.json` | nested / deploy / provisional / uniform |
| `reports/bot6_ortho_work/surf_vs_dfine_fal_ger.json` | surf vs dfine vs union |
| `reports/bot6_ortho_work/gt_or_only.geojson` | Or GT extract |
| `/workspace/kaggle/out/dfine-or/` | веса + OOF (SHA OK) |

**Коммитов нет** (analysis+design only).
