# bot-2 — satellite (sp_*) recall 0.25→0.4+ (Rule P)

_обновлено 29.10 20:40 МСК_

| поле | значение |
|------|----------|
| **Цель** | Поднять LB выше **0.41238** (`eb268d6`); слой **спутник** P=0.456 / R=0.255 → **recall ≥ 0.4** при удержании P. |
| **Гипотеза** | Пороги `[thresholds_by_markup.sp_*]` уже = **A_max F1-opt** (bot-9 thr_sweep ≡ default.toml). Ослабление ×`scale=0.85` (`[sat_thr_relax]`, default OFF) выдаст пики ниже thr; hit-rate новых dets > break-even **8.9%** → type F1↑ → вклад в weighted class F1. D-FINE / A4 **не** для recall (локализация сломана; A4 INVALID). |
| **Критерий** | (1) pytest `test_sat_thr_relax` green; (2) PC Docker OFF≡v2 mock ALL **0.3125**/sp_or **0.2429**; ON: sp_or не падает >0.01 **или** ΔALL>0 при layer drop ≤0.01; (3) ожидаемый вклад в LB записан. **Не** LB submit / main / PR. |
| **Бюджет** | 1× mock Docker OFF+ON ~20–40 мин GPU; JobDir `C:\Sloi\pc_jobs\…` only. |
| **Ветка / SHA** | `bots/bot2-sat-recall` / **`491c63ec246a30949c21cd2c7675ba1ef75a0a34`** (base **`eb268d6`**, LB zip). |

## 1. Вклад спутника в LB (`scoring_v2_eb268d6_0.41238.log`)

Type-level (вторичные; final_score = weighted **class** F1):

| тип | P | R |
|-----|---|---|
| спутник | **0.456057** | **0.254642** |
| лидар | 0.516 | 0.504 |
| геофизика | 0.769 | 0.656 |
| орто | 0.368 | 0.140 |
| аэро | 0.091 | 0.013 |

Метрика type P = `TP / (TP + 0.5·FP)` (soft FP). При R=0.254642 → **GT≈377, TP=96, FP≈229, FN=281** (проверка: P=96/(96+114.5)=0.456).

**Арифметика цели R: 0.255 → 0.40 при том же P=0.456:**

- TP' = 0.4 × 377 ≈ **151** (ΔTP ≈ **+55**)
- FP' = 2·TP'·(1/P − 1) ≈ **360** (ΔFP ≈ **+131**)
- F1_type = 2PR/(P+R): **0.327 → 0.426** (ΔF1_type ≈ **+0.099**)
- Break-even hit-rate новых dets: F/(4−F) ≈ **8.9%** (текущий hit среди выданных sp_focal povr A_max ≈ **20%** ≫ 8.9%)

**Ожидаемый вклад в LB (final_score):** type F1 не входит в final напрямую; sat объекты в основном `kurgany_povrezhdennye` (w=30), `tselye` (20), `gorodishcha` (15), `arkhitektury` (10). Доля sat в matching mass ≈ mid. Банд:

| сценарий | ΔLB |
|----------|-----|
| полный R→0.4 @P const, sat share 8–25% of weighted F1 mass | **+0.008 … +0.025** (mid ≈ **+0.015**) |
| частичный (scale 0.85 → R≈0.30) | **+0.003 … +0.010** |

Per-class TP/FP/FN на LB **не** в scoring.log (только type P/R + contour). Offline lb_counts из лога: type counts выше.

## 2. OOF / пороги (без leakage)

Источник: `docs/bots/2026-10-29/bot-9/submission_dfine/thr/thr_sweep.json` (A_max / R_max).

| группа | класс | A_max thr (= default.toml) | kept / tp (A_max) | hit |
|--------|-------|----------------------------|-------------------|-----|
| sp_focal | kurgany_povrezhdennye | **0.3645** | 1014 / 203 | 20.0% |
| sp_focal | kurgany_tselye | **0.3725** | 127 / 27 | 21.3% |
| sp_focal2 | povr / tselye | 0.5016 / 0.6455 | 193/37 · 10/3 | ~19% |
| sp_or | gorodishcha | **0.994085** | 33 / 3 | 9.1% |
| sp_arch | arkhitektury | **0.286725** | 227 / 21 | 9.3% |

- **A_max F1-opt ≡ уже в default.toml** — крутить thr выше не даст F1; для **recall** нужен сдвиг **ниже** A_max (или denser peaks / D-FINE).
- **R_max** в sweep опирается на **sp_dfine** (in-sample thr ~0.55–0.69) — на mock D-FINE даёт 0 TP / сдвиг сотни метров → **не использовать** для sat recall.
- Честный LOFO sat (bot-10 taus_S / lbv2_keep): train OOF sat R≈0.11 P≈0.22 — ещё ниже LB; подтверждает, что запас recall на U-Net thr есть, но OOF осторожнее in-sample.
- **Выбранный thr для эксперимента:** `scale=0.85` на `sp_or/sp_focal/sp_focal2/sp_arch` (без sp_dfine). Эквивалент sp_focal povr thr **0.3098**. Полной PR-кривой сырых OOF на box нет → PR аппроксимируется A_max hit + scale; PC mock OFF/ON измерит фактический Δ.

## 3. FN-анализ (mock, post-threshold)

Файл: `/workspace/uploads/fn_sat_mock.json` (центры, nearest sat pred; **не** полный official soft_match).

| | |
|--|--|
| sat GT / pred | 83 / 46 |
| TP / FP / FN (2 м proxy) | 16 / 30 / 67 |

**Корзины FN (по nearest pred center):**

| bucket | n | смысл |
|--------|---|--------|
| loc_shift 2–10 м | 14 | детект рядом, soft_match / radius / ellipse? |
| loc_shift 10–50 м | 31 | сдвиг / неверный пик |
| far 50–200 м | 2 | |
| **undetected >200 м** | **19** | нет пика / срезано порогом / нет ответа модели |
| should_have_matched | 1 | |

Offsets FN→nearest pred: p50≈**27.6 м**, p90≈**469 м**, mean≈**165 м**.

**Ниже порога vs undetected:** post-threshold preds only → **нельзя** отделить «был raw ниже thr» от «модель молчала» без `EXPDS_*_DUMP` / stages. Гипотеза relax: часть из 19+ сдвиговых FN получит второй пик / более низкий conf peak.

**Не реqueue A4:** agreement_m не создаёт TP; D-FINE localization std≫mean; A4 INVALID (wall~28 с, exit=-1, stale geo).

## 4. Код / runners / очередь

- **Код:** `scale_sat_thresholds` + `sat_thr_relax_by_markup` в `expds/post.py`; `finish`/`safe_finish` берут by_markup через hook; `[sat_thr_relax] enabled=false` в `configs/default.toml`.
- **Тесты:** `tests/test_sat_thr_relax.py` — 5 passed (`koz3-venv`).
- **Runners (shared `pc/runners/`):** a3 / v2_base — wipe stale + DONE только при exit==0 и mtime≥start (как A4); a2 / v3_hotfix — не DONE при exit≠0. A4 уже был hardened.
- **PC queue:** `run_bot2_sat_thr_relax_491c63e.ps1` — mock Docker OFF vs ON scale=0.85, image `pantum/n7q:20261024`.

## 5. Ожидаемый LB delta (для QUEUE)

| | |
|--|--|
| Целевой (R→0.4 @P) | **+0.008 … +0.025** |
| Этот прогон (scale 0.85) | **+0.003 … +0.010** (если hit новых >8.9% и слой не роняет ALL) |
| Риск | ON роняет sp_or P сильнее, чем поднимает R → ΔALL≤0 → FAIL, флаг остаётся OFF |

