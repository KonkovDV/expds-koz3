# bot-3 · GPU-3 — Kaggle kernel D-FINE Ae fine-tune

**МСК:** 2026-10-29 16:14 (Europe/Moscow)  
**Ветка:** `bots/bot3-dfine-ae-kernel` @ `6e42cf6` (`6e42cf605be3a6eab5bb9a92f113dd7faae98183`)  
**База:** `bots/bot3-aerial-recall` @ `475bafa` (lineage `cbd4804`)  
**Worktree:** `/workspace/koz3-wt/bot3-dfine-ae-kernel`

## Контракт (PROTOCOL §1)

| | |
|---|---|
| **Цель** | Подготовить (не запускать) возобновляемое Kaggle-ядро дообучения D-FINE-L Ae поверх COMPLETE `dmitrijkv/koz3-dfine-ae`, чтобы поднять ae recall (~0.013). |
| **Гипотеза** | +~2000 ит/фолд с resume + all-data final 1500 + recall-oriented τ (F_β=2) даст детекции, пригодные для `[dfine_aeor]` и честного honest-10. |
| **Критерий приёмки ядра** | `STATUS=DONE`; `dfine_ae/{config.json,model.safetensors}`; `thresholds_ae_oof.json` с группами `dfine_ae`/`ae_dfine`; folds без `error`. |
| **Критерий приёмки эффекта (после PC)** | honest-10: Δ ALL_F1 > 0 и слой ae не проседает >0.01 vs база; флаг `[dfine_aeor]` остаётся default-OFF в коде до ACCEPT. |
| **Бюджет** | **~2.5–3.0 GPU-ч** T4×2 (deadline 3.0 ч). Квота ~9.4 ч до 31.10 03:00 МСК. |
| **RULES** | Без Yandex/Google/Bing/Esri; без данных КОЗ №2; база `ustc-community/dfine-large-obj2coco-e25` (Apache-2.0). |

## Что сделано

1. Скопирован и пропатчен completed kernel `kaggle/.../ae_o2c/koz3-dfine-ae.py` → `kaggle/2026-10-29/dfine_ae_ft/koz3-dfine-ae-ft.py`.
2. Lean defaults: `FULL_ITERS=2000`, `DEADLINE_H=3.0`, `FINAL_ITERS=1500`, `CKPT_EVERY=500`.
3. Resume: `find_fold_ckpt` — working mid-ckpt → `/kaggle/input/**/dfine_f{k}` (kernel_sources `dmitrijkv/koz3-dfine-ae`).
4. `STATUS.txt`/`STATUS.json`; mid-ckpt в builtin и bot5 trainer.
5. Final `dfine_ae/` (all-data FT from best fold) + `models/dfine_ae/` + `PACK_HINT.toml` для `[dfine_aeor]` (enabled=false).
6. Threshold calib на OOF: F_β=2; `gorodishcha`/`arkhitektury` = 1.01 (OFF, bot-10); выход `thresholds_ae_oof.json` + `models/dfine_aeor/thresholds_oof.json`.
7. `kernel-metadata.json`, README; QUEUE + lease + EXPERIMENTS строка; BACKLOG GPU-3 → QUEUED.
8. **Инференс не трогали** — только train kernel; `[dfine_aeor]` уже OFF на базе ветки.

## Коммиты

- `ea440c3` feat(bot3): Kaggle kernel koz3-dfine-ae-ft — lean resume of D-FINE Ae
- `6e42cf6` docs(bot3): README for koz3-dfine-ae-ft kernel

## Пути ядра

- `kaggle/2026-10-29/dfine_ae_ft/koz3-dfine-ae-ft.py`
- `kaggle/2026-10-29/dfine_ae_ft/kernel-metadata.json`
- `kaggle/2026-10-29/dfine_ae_ft/README.md`

## Оценка GPU

| этап | ч |
|---|---|
| setup + chips Ae | 0.3–0.5 |
| 5×2000 ит / 2 GPU | ~1.4 |
| OOF infer | ~0.3 |
| final 1500 | 0.4–0.5 |
| calib | ≪0.1 |
| **итого** | **2.5–3.0** |

## Очередь

См. `/home/box/shared/koz3/kaggle/QUEUE.md` секция `[bot-3] D-FINE Ae fine-tune`.  
Lease: `leases/kaggle_bot3_dfine_ae_ft.lease` (QUEUED; bot-3 **не** запускает Kaggle сам).

## Следующий шаг

1. «1» / kaggle-quota: push+run `koz3-dfine-ae-ft` когда слот T4×2 свободен (≥3 ч квоты).
2. Забрать `dfine_ae/` + thresholds → PC pack с `[dfine_aeor]` ON только для honest-10.
3. При Δ>0 — отдельный PR включения (не этот).

## Не сделано / риски

- Ядро не прогонялось (по заданию). Smoke на боксе не запускался (нужен S3/GPU).
- Calib — in-sample OOF (не nested fam10 bot-10); после прогона желателен PC re-sweep / honest-10.
- Mid-ckpt bot5 сохраняет веса, но оптимизатор не сериализуется — resume продолжит с тех же весов и нового OneCycle (приемлемо для FT).
