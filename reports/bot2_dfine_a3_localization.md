# Бот 2 — A3 D-FINE localization (Docker @829ea7d) — 29.10.2026 ~18:51 МСК

| Поле | Значение |
|---|---|
| **Цель** | Разобрать dump/stages/locdump A3 vs GT: scores 482 in_state, NN-дистанции, systematic offset?, thr-sweep (mock-optimistic); гипотеза CRS vs domain/thr; при ясном фиксе — малый commit OFF-by-default. |
| **Гипотеза** | **Не CRS/pixel-origin.** 13 pass_thr — high-conf FP со **случайным** смещением (std ≫ mean); median dist_any ≈ **388 м** (не константа ~191 м). Thr-sweep не даёт TP@2 м → калибровка thr alone не спасёт. `agreement_m` (OFF) нейтрализует далёкие FP рядом с U-Net. |
| **Критерий приёмки** | Отчёт + EXPERIMENTS + QUEUE A4; pytest agreement зелёный; A4 Docker: mock ALL ≥ 0.31287 и sp_or ≥ 0.245−0.01, без layer drop >0.01 vs B. |
| **Бюджет** | ~1 ч CPU box (анализ PC dump); A4 GPU Docker ≤15–20 мин (как A3 ~15 мин). |
| **Ветка** | `bots/bot2-e2e-v3-prep` (tip **`f9ad045`**; agreement_m default OFF). |

Связанные: `reports/bot2_dfine_localization.md`, `reports/bot2_runtime_env.md`, A3 `pc/results/bot2_a3_docker_829ea7d/`, locdump `pc/results/bot2_a3_dfine_locdump/`.

---

## 1. A3 Docker результат (ENV=docker:pantum/n7q:20261024)

| | A3 (D-FINE ON, drop_groups=[]) | B (OFF) | Δ |
|---|---:|---:|---:|
| ALL F1 | **0.310530** | 0.312873 | −0.00234 |
| sp_or | **0.231735** | 0.245064 | −0.0133 |
| ae / li | 0.095556 / 0.336983 | same | 0 |

Stages: n_raw 101=1497 / 103=1009 **total 2506**; n_features=**482**; n_would_pass_thr=**13**; written sp_or=57 (=44 U-Net + 13 D-FINE). D-FINE **загрузился** (не A≡B).

---

## 2. Score distribution

| Набор | n | score p50 | p90 | max |
|---|---:|---:|---:|---:|
| raw boxes | 2506 | 0.075 | 0.161 | 0.734 |
| pass class thr (make_on) | 13 | 0.64 | 0.70 | 0.73 |

Класс thr (kurgany_povrezhdennye 0.555…) режет почти всё; из 482 in_state после convert/cluster только **13** проходят. Сырые скоры низкие (p90≈0.16) — thr не «чуть завышен», а отсекает хвост, в котором почти нет локально верных детекций.

---

## 3. Дистанции до GT (EPSG:3857, nearest)

### 3.1 Все raw / pass_thr

| | dist_any p50 | within 2 м | within 50 м | within 200 м |
|---|---:|---:|---:|---:|
| all raw 2506 | 534 м | **2** | 122 | 403 |
| pass_thr 13 | **388 м** | **0** | 1 | 4 |

13 прошедших (sorted dist_any м):  
`[14.1, 96.7, 144.5, 191.1, 258.1, 327.4, 388.2, 439.7, 456.2, 598.8, 706.1, 891.8, 1508.1]`  
→ **не** все ~191 м (Kaggle R-only); один близок (14 м), один ≈191 м, остальные сотни–тысячи метров.

### 3.2 Systematic offset? (pass_thr vs nearest any-GT)

- mean (dx, dy) ≈ **(+71, −200)** м, ‖mean‖≈212 м  
- **std (dx, dy) ≈ (376, 420)** ≫ mean  
- residual после вычитания mean: p50≈306 м  

**Вердикт:** нет постоянного вектора (CRS / pixel-origin / half-tile). Разброс случайный → **domain/модель**, согласуется с prior roundtrip tests + geo CRS guard @ fb66c89.

### 3.3 Locdump vs R-only

Locdump (все 57 written sp_or): median same-class **2.02 м**, within_2=23 — **доминирует U-Net**. 13 D-FINE к ближайшему U-Net sp_or: min **70 м**, within_50=**0**. Итого locdump «хороший» ≠ D-FINE локализован.

---

## 4. Thr-sweep (mock-optimistic, center ≤2 м)

Offline по dump+GT (не official soft-match; **нужен OOF/honest-10** для честного thr):

- all raw: max F1_opt ≈ **0.00135** (thr=0.05; tp2=2 / fp=2504)  
- при thr≈0.55: tp2=**0**  

**Вывод:** снижение порога на mock **не создаёт** TP у GT; только больше FP. Калибровка thr без улучшения localization бесполезна для F1.

---

## 5. Гипотеза (итог)

1. **Primary:** domain/model localization failure на mock SpOr (high-conf dets не на GT).  
2. **Secondary:** in-sample make_on thr пропускает 13 FP → sp_or 0.245→0.232, ALL −0.002.  
3. **Отвергнуто:** CRS / affine / pixel-origin bug (std≫mean; unit roundtrip OK).  
4. **Mitigation (не «лечение модели»):** `sp_dfine.agreement_m` (default **0=OFF**) — оставлять D-FINE только ≤ N м от уже найденного не-sp_dfine sp_or (фаза после оптики). На A3 все 13 ≥70 м от U-Net → `agreement_m=50` → 0 добавок → ожидаемо ≡ B.

---

## 6. Код (commit на ветке)

- `expds/sp_dfine.py`: `agreement_m` в DEFAULTS; `filter_agreement()`  
- `expds/pipeline.py`: после `region_feats`, до `state.add`  
- `tests/test_bot2_dfine_agreement.py` (5 passed)  
- Полный набор D-FINE тестов: 50 passed (2 pre-existing packaging fail без models/pixgbm — не регрессия)

---

## 7. A4 (очередь)

Docker `pantum/n7q:20261024`, runner `pc/runners/run_bot2_a4_dfine_agreement_docker.ps1`:  
enabled=true, drop_groups=[], **agreement_m=50**, overlay sp_dfine+pipeline.  
PASS: mock ALL ≥ **0.31287** и sp_or ≥ **0.245−0.01**, no layer drop >0.01 vs B.  
Ожидание: ALL/sp_or ≈ B (нейтрализация FP). Не claim «D-FINE помогает» — только safety gate.

Дальше (backlog): OOF thr / domain adapt / soft re-rank U-Net proposals — не в A4.
