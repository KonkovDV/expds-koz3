# bot7 — offline eval as on LB (scoring.log breakdown)

**Дата:** 01.11.2026 ~19:20 МСК (Europe/Moscow)  
**Бот:** bot7 (executor)  
**Ветка:** `bots/bot7-lb-breakdown` (rebased on `origin/main`; PR #2 MERGED, #8 CLOSED → base **main**)  
**Worktree:** `/workspace/koz3-wt/bot7-lb-breakdown`  
**PR base:** `main` (DRAFT)

## Контракт

| | |
|---|---|
| **Цель** | Офлайн-табло печатает разбивку как LB `scoring.log` + `expected_delta_lb` (proxy ΔLB per type); сверка final_score; флаг для всех honest-10 gate. |
| **Gate flag** | `python scripts/scoreboard.py --work WORK --train TRAIN --honest 10 --lb-breakdown OUTDIR [--lb-log reports/lb/scoring_v2_eb268d6_0.41238.log]` |
| **Не делать** | main push, LB upload, secrets in text. |

## soft_ceiling

| | |
|---|---|
| **Статус** | PASS alone на `origin/main`, `42b4bf8` (honest-eval-layer), `5e43510` (compare-candidates). Допуск `1e-3` уже на main. |
| **SHA** | soft_ceiling fix на main (через #2); исторические tip: compare `5e43510`, honest-eval `42b4bf8`. |
| **Bot1 FAIL** | Устаревший BOARD tip до фикса; текущий main/ветка — зелёные. Suite full: 0 fail soft_ceiling. |

## MAIN: сделано 01.11

- `scripts/lb_breakdown.py` — extract/format/write; `side_by_side`; **reconcile** (`--reconcile-lb`); **expected_delta_lb** (proxy + class ∂S/∂F1).
- `scripts/lb_breakdown_table.py` — CLI LB vs mock vs honest-10.
- `scripts/scoreboard.py` — `--lb-breakdown DIR` / `--lb-archive` / `--lb-log` → scoring_*.log + **expected_delta_lb.md**.
- `scripts/compare_candidates.py` — `--lb-breakdown` / `--lb-log`.
- `tests/test_lb_breakdown.py` — synthetic + reconcile + ΔLB proxy.

## final_score reconciliation (task 2)

**Вердикт: type/size P/R НЕ воспроизводят 0.41238.**

```
final_score = Σ_c w_c · F1_c / Σ_{c present} w_c
CLASS_WEIGHTS = {
  kurgany_povrezhdennye: 30, kurgany_tselye: 20, gorodishcha: 15,
  arkhitektury: 10, object_poly: 10, fortifikatsii: 5, finds_points: 10
}  # Σ=100
```

- LB log header: «Дополнительные метрики. На final_score не влияют.»
- type-share-weighted F1 (lb_counts.resolve, GT=2907) ≈ **0.56687** ≠ **0.4123779189** (gap ≈ 0.154).
- CLI: `python scripts/lb_breakdown.py --reconcile-lb reports/lb/scoring_v2_eb268d6_0.41238.log` → exit 0, `reproduces_from_type_size: false`.

## ΔLB-per-type (task 3)

**Точно (классы):**
```
∂final/∂F1_c = w_c / W ,  W = Σ w_present (обычно 100)
ΔLB = Σ_c (w_c/W) · ΔF1_c
```
Коэффициенты (все классы present):  
povrezhdennye **0.30**, tselye **0.20**, gorodishcha **0.15**, arkhitektury/object_poly/finds_points **0.10**, fortifikatsii **0.05**.

**Proxy по типу данных (планирование; ∂final/∂F1_type = 0 точно):**
```
ΔLB_hat(X) ≈ s_X^{LB} · ΔF1_X
F1_X = 2PR/(P+R)   # P уже с 0.5·FP
```
LB GT shares (v2 eb268d6, lb_counts smallest sol, n_gt=2907):

| type | s_LB |
|---|---|
| лидар | 964/2907 ≈ 0.3316 |
| аэрофото | 75/2907 ≈ 0.0258 |
| спутник | 377/2907 ≈ 0.1297 |
| ортофото | 50/2907 ≈ 0.0172 |
| геофизика | 1441/2907 ≈ 0.4957 |

**Assumptions (печатаются в expected_delta_lb.md):**  
(1) type aux ↛ final_score; (2) scale honest ΔF1_type by **LB** shares (honest mix ≠ LB); (3) exact gate Δ = class-weighted; (4) no cross-type / appearance effects.

## Gate integration

Команда honest-10 в orch/PROTOCOL/STATE = `scripts/scoreboard.py --honest 10`.  
Добавить `--lb-breakdown OUTDIR` (и опц. `--lb-log …`) — печатает scoring.log + expected_delta_lb table. Default stdout/JSON `--out` не меняются.

## PC run request

Нужен CPU ≤4 thr: v2 mock + honest-10 metric matching → `scoring_mock.log` + `scoring_honest10.log` + `side_by_side.md` + `expected_delta_lb.md`.  
Файл через `orch/enqueue.py` (не старый pc/QUEUE.md).

## Suite / SHA

- soft_ceiling: PASS (`1e-3` on main / 42b4bf8 / 5e43510)
- full suite: **1011 passed, 64 skipped, 0 failed** (~5.5 min)
- tip SHA: `91f472d7` · PR https://github.com/KonkovDV/expds-koz3/pull/25
