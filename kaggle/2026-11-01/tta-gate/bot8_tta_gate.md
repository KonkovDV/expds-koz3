# bot-8 · TTA/MS gate package (`koz3-ae-dfine-gate`)

**Когда (Europe/Moscow):** 2026-11-01 ~19:15 MSK  
**Ветка пакета:** `bots/bot8-tta-gate` @ **`f47ea449`** (based on `f3aafa42` = tip `bots/bot8-orsp-tta`)
**Draft PR:** https://github.com/KonkovDV/expds-koz3/pull/23 (base `bots/bot8-orsp-tta`)  
**Пуш-пакет:** `/workspace/kaggle/push/tta-gate/`  
**Slug:** `dmitrijkv/koz3-ae-dfine-gate`  
**Не пушим / не гоняем Kaggle.** Bot 1. LB-submit запрещён (правило P).

## Branch state (steering: PR #19 / orsp_tta)

| ref | note |
|---|---|
| `origin/bots/bot8-orsp-tta` | tip **`f3aafa42`** — cloud agent: `[orsp_tta]` U-Net + D-FINE d4/d8/gsd WBF, `scripts/orsp_tta_honest10.py`, `check_compliance`, tests |
| earlier bot8 on same name | `35ca3df` / `81b1d8d` / `82a167c` / `73b1fee` — **all ancestors of f3aafa4** (continuous history, **not** a rewrite) |
| `73b1fee..f3aafa4` | ~799 commits ahead, 0 behind tip (integrated mainline + cloud tip) |
| **PR #4** | MERGED 2026-10-30 — head was `bots/bot8-orsp-tta` (old TTA/MS sp_or + compliance) |
| **PR #19** | OPEN draft — **same branch name** `bots/bot8-orsp-tta`, tip commit `f3aafa42` (new `[orsp_tta]` section). Same branch tip after merge of #4 + later history; **not** a divergent force-push of #4 |
| local old wt `bot8-orsp-tta` | still parked at `73b1feed` (stale); **do not force-push** |
| this package branch | `bots/bot8-tta-gate` cut from **`f3aafa42`** — reuses `expds/orsp_tta.py` (`apply_unet`, `unet_tta`/`unet_scales`, markups `sp_or`/`sp_focal2`) and Ae `rot90_tta` in `expds/dfine_aeor.py` |

## Sources (slug mirror)

`koz3-ae-dfine-gate` **не найден** как готовый пакет на боксе / GH. Metadata зеркалит `koz3-dfine-ae-ft`:

- `dataset_sources`: `dmitrijkv/koz3-s3-cred`
- `kernel_sources`: `dmitrijkv/koz3-dfine-ae`
- GPU T4, internet, тот же private BYOD docker digest

## Variants vs B′ = 0.315800

| id | overlays |
|---|---|
| `bprime` | `[orsp_tta] enabled=false`, `rot90_tta=false` |
| `tta_ae` | Ae `rot90_tta=true` (+ flip) |
| `tta_ms_sat` | `[orsp_tta] enabled`, `unet_tta=4`, `unet_scales=[0.75,1.0,1.25]` на sp_or/sp_focal2; D-FINE views off |
| `both` | Ae rot90 + sat TTA/MS |

Gate: Δ≥0.003, P≥0.9, layer drop ≤0.01. Wall target ≤20 min LB (budget env `TTA_GATE_LB_BUDGET_S=1200`).

Resumable: `STATUS.txt`/`STATUS.json`, `variants/<id>/{done,score,wall,gate,overlay}.{json,toml}`, `SHA256SUMS`.

## DRY-RUN (box CPU, 2026-11-01 Europe/Moscow)

```bash
TTA_GATE_DRY=1 TTA_GATE_OUT=/tmp/tta-gate-dry nice -n 10 python3 koz3-tta-gate.py
```

- Exit 0; `STATUS=DONE`; resume skip verified (2nd run skips all four).
- Synthetic scores only (no GPU/weights/S3). Real path raises `NotImplementedError` until bot1 attaches repo+data+f4pack.
- Dry decisions (illustrative): bprime=BASELINE; tta_ae/tta_ms_sat=FAIL_DELTA; both=PASS (Δ=0.0034). **Not** a real honest-10 claim.
- `slot:box/cpu` held by **coord** until **01.11 23:07 MSK** — dry run under `nice -n 10` without exclusive lease.

## Lease

```
orch/lease.sh acquire branch:bots/bot8-tta-gate bot-8 …
→ bad owner 'bot-8'
```

`OWNERS = ^(coord|bot-W[1-9]…)$` — classic `bot-8` not allowed. No lease taken under a foreign W-id. Documented blocker for coord.

## Deliverables

- `/workspace/kaggle/push/tta-gate/{koz3-tta-gate.py,kernel-metadata.json,README.md}`
- wt: `kaggle/2026-11-01/tta-gate/` + this report
- Draft PR #23 from `bots/bot8-tta-gate` (package only; no Kaggle push)

## RU (6–10 строк)

1. Пакет TTA-gate собран на ветке `bots/bot8-tta-gate` от tip `f3aafa42` (`bots/bot8-orsp-tta` / draft PR #19).  
2. PR #4 (merged) и PR #19 (draft) — **одна и та же** ветка `bots/bot8-orsp-tta`; коммиты 35ca3df…73b1fee — предки tip, расхождения/rewrite нет.  
3. Спутниковый U-Net берём из `expds/orsp_tta.py` (unet_tta=4, scales 0.75/1/1.25); Ae — `rot90_tta` в `dfine_aeor`.  
4. Slug `koz3-ae-dfine-gate` зеркалит metadata `koz3-dfine-ae-ft` (s3-cred + kernel koz3-dfine-ae).  
5. Пакет лежит в `/workspace/kaggle/push/tta-gate/`; **Kaggle не пушили**.  
6. DRY-RUN на CPU (`nice -n 10`) прошёл: STATUS DONE + resume; синтетические скоры.  
7. Аренда `bot-8` отклонена (OWNERS); `box/cpu` до 23:07 MSK у coord.  
8. Draft PR #23; ждём Ae-gate + bot1 GO на honest-10; LB не сабмитим.
