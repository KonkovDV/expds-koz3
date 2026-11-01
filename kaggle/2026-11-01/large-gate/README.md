# koz3-w4-large-gate (CPU honest-10 · large_objects vs B′)

Plan **(b)**: без fam0. S3/xfer fam0 нет; `stack_arch` post-thr — rebuild fam0
из него невалиден для thr-рычагов. Вместо этого: инъекция D-FINE ae OOF
gorodishcha (slim, embedded) в `w5base/stack_arch` → prep → gate_v6c vs B′=0.315800.

Варианты: `baseline` (off) / `area_tau` / `fixc_nested` (τ_large=0.20 из default.toml).
**Не сдавать на LB.**

## Почему не (a)
- fam0 ≈1.39M objs / 103 fam pre-thr jsonl — 0 hits в S3 index (2132 objs = raw train).
- `split_stack_by_family.py` + `family_clusters` по post-thr stack не вернёт preds,
  убитые τ=1.01 на ae gorodishcha.

## Fix C effect check
Embedded `data/ae_dfine_gor_oof_slim.jsonl.gz`: 4520 gor (area≥5k OR conf≥0.15);
**698** Fix-C-pass (area≥5k ∩ conf≥0.20); conf range ≈0.05–0.44 — **все ≪ 1.01**.

## Датасеты
- `elenakonkova/koz3-w2-gate` (w5base: stack_arch, gt_train_v2, base/bprime_counts)
- internet **off** (S3 не нужен; OOF slim в бандле)

## Push (человек / orch; бот не пушит)
```bash
orch/pool_kaggle.sh elenakonkova kernels push -p /workspace/kaggle/push/large-gate
```

## Контракт
Δ≥+0.003 vs B′, p_gt0≥0.9, layer drop ≤0.01; CPU ~20–60 min; branch `bots/bot4-large-objects`.
