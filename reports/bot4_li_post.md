# bot-4: Li post levers (class_tau + self_nms)

## Results (2026-11-01) — honest leave-one-fold on s0 ConvNeXt-Li OOF

Флаги в `configs/default.toml` **выключены** (`[li_post] enabled=false`, `class_tau=false`, `self_nms=false`).
При выключенном мастере `finish` / `safe_finish` — no-op (байты как без модуля).

Данные: локальный light-срез `elenakonkova/koz3-w2-oof` s0 per-fold OOF+fold GT
(`/workspace/kaggle/datasets/koz3-w2-oof-light`). Метрика мягкая официальная:
`cov≥0.5` + representative point ∈ GT; `P=tp/(tp+0.5·fp)`. База OFF = τ=0.15
(production fallback, когда нет `[thresholds_by_markup.li_cnx]`).
**Это Li-layer only**, не взвешенный score стека A. Не дублирует bot-7 геофизику.
Закрытые рычаги не трогались: r50c5, WSR, ring, peak-rerank, sp_large, dino-gor, sat 8m.

### FN/FP (local OOF analysis)

| наблюдение | число |
|---|---|
| FP почти все в корзине площади **(0,100] м²** | 14 589 FP @τ=0.15; raw-OOF 83 716 все в (0,100] |
| Доминируют FP-классы | `kurgany_povrezhdennye` (8758 FP), `kurgany_tselye` (4432 FP) @τ=0.15 |
| fortifikatsii | R≈0.197, P≈0.081 (FN=212) — порог не поднимаем |
| TP conf p50 / FP conf p50 (raw) | 0.36 / 0.08 |
| center offset TP p50 | **1.43 m** |
| preds с соседом &lt;12 m (фрагментация) | **21 412** |

### Honest leave-one-fold (dry-run = gate script)

| вариант | P | R | F1 | ΔF1 |
|---|---|---|---|---|
| OFF (τ=0.15) | 0.1788 | 0.6487 | 0.2803 | — |
| thr_nested (class_tau) | 0.3477 | 0.4661 | 0.3983 | **+0.1180** |
| self_nms@8 | 0.1817 | 0.6438 | 0.2834 | +0.0031 |
| self_nms@12 | 0.1867 | 0.5993 | 0.2847 | +0.0044 |
| thr + nms8 | 0.3505 | 0.4653 | 0.3998 | **+0.1195** |
| thr + nms12 | 0.3507 | 0.4449 | 0.3922 | +0.1119 |

Nested floors (median по hold): `kurgany_povrezhdennye≈0.30`, `kurgany_tselye≈0.30`;
gorodishcha/fortifikatsii остаются 0.15.

Per-class Δ от thr_nested (kurgany): povr F1 0.269→0.413; tselye 0.343→0.540.

### Kaggle COMPLETE (`elenakonkova/koz3-w4-li-post-gate`)

Выход: `/workspace/kaggle_out/lipost/li_post_honest.json` (2026-11-01).
Числа **совпали** с dry-run (Li-layer only, не full-stack):

| вариант | F1 | ΔF1 |
|---|---|---|
| OFF (τ=0.15) | **0.2803** | — |
| thr_nested | **0.3983** | **+0.1180** |
| thr_nms8 | **0.3998** | **+0.1195** |
| nms12 alone | 0.2847 | +0.0044 |

Full-stack honest-10 vs B′=0.315800 — пакет `/workspace/kaggle/push/li-post-full/` (см. ниже / COORD).

### Выбранные рычаги (1–2, дешёвые, без retrain)

1. **`class_tau`** — абсолютные полы conf по классу для `li*`-группы после `apply_thresholds`.
2. **`self_nms`** — same-class NMS радиус `self_nms_m=8` (фрагментация &lt;12 m).

Почему: FP курганов с низкой conf и мелкой площадью; NMS режет дубли без обучения.
Не closed. Не geo.

### Ожидаемый вклад в LB (**estimate**, не verified на стеке A)

Li ≈ 33 % LB, публичный P/R≈0.50. OOF c3-sc после τ=0.15 даёт F1_li≈0.28 → 0.40
(+0.12 layer). Перенос на полный стек/LB сильно меньше (уже есть thr/dedupe_m у li_cnx,
слой в стеке A слабее аддитивно historically FAIL): **+0.003…+0.015** к LB
(**labelled estimate**; full-stack → `li-post-full` Kaggle package, box lacks stack_arch/bprime_counts/fam0 jsonl).

### Kaggle package

`/workspace/kaggle/push/li-post-gate/` (slug `elenakonkova/koz3-w4-li-post-gate`).
CPU, internet off. dataset_sources: `elenakonkova/koz3-w2-oof`.
Dry-run box: exit 0, STATUS=DONE, 17.4 s → `dryrun/li_post_honest.json`.
**Не пушить без coord.** Lease на packaging не брался (coord takes at push).

```bash
orch/pool_kaggle.sh elenakonkova kernels push -p /workspace/kaggle/push/li-post-gate
```

### Full-stack package (vs B′ 0.315800)

`/workspace/kaggle/push/li-post-full/` slug `elenakonkova/koz3-w4-li-post-full`.
CPU, **internet off**. Datasets: `elenakonkova/koz3-w2-gate` + `elenakonkova/koz3-w2-oof`.
Variants: baseline stack | thr_nested | thr_nms8 (floors from nested LOFO; flags ON only in kernel).
На боксе `stack_arch` / `base_counts.pkl` / `bprime_counts.pkl` / fam0 jsonl **нет** (fam0 только `_counts.json`); w2-gate download 403 → full gate только на Kaggle.

```bash
orch/pool_kaggle.sh elenakonkova kernels push -p /workspace/kaggle/push/li-post-full
```

### Код

- `expds/li_post.py` + hook в `expds/watchdog.py` (finish + safe_finish)
- `configs/default.toml` `[li_post]` все false
- `tests/test_bot4_li_post.py` (7 passed)
- ветка `bots/bot4-li-post`

## Контракт
- **Цель:** поднять Li-layer F1 дешёвым post без retrain (FN/FP курганов / фрагментация), не трогая closed levers и не дублируя bot-7 geo.
- **Флаги OFF по умолчанию.**
- **Честный протокол:** подбор τ только на train-фолдах OOF; отчёт на hold-out.
- **Сдача:** только после positive honest на полном стеке / Kaggle gate + lease.
