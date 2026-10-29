# Манифест весов — release v5 (bot-5)

База ветки: `bots/bot5-release-v5` ← `eb268d6` (LB base ALL **0.41238**, wall **3:48**/30).  
Сборка v5: **tooling** (validator, sha256-at-load, budget guard, манифест). **Score-фичи из bot2/3/4/8 не включены** — ни одна не прошла PROTOCOL §5 (Δ>0, no layer drop >0.01, honest eval). См. `reports/bot5_v5.md`.

Правило: большие веса D-FINE / PIXGBM / DINOv2 в git **не** добавляем (`[nomodels]`). Хеши — эталон для проверки на ПК/в Docker.

Источники sha256:
- **PC** `C:\Sloi`, бот 1, 29.10 — TSV `/home/box/shared/koz3/pc/results/weights_sha256_2026-10-29.tsv`
- **v4 манифест** `docs/WEIGHTS_MANIFEST_v4.md` (переиспользован)
- **Kaggle** `koz3-dfine-or`, `koz3-dfine-ae-ft` (Ae/Or; флаги OFF в v5)

## PC TSV 2026-10-29 (обязательные для v2/v5 runtime)

| path | bytes | sha256 | source | роль v5 |
|---|---:|---|---|---|
| `models/pixgbm/all/finds.txt` | 2111596 | `5fa8c09d77d0914cc925fb6bb925a0ab1b64ccecf3c823d7f3546e71c4ceb04a` | PC TSV + zip v2 eb268d6 | **ON** — проверяется в `expds/geo_pixgbm.load_models` |
| `models/pixgbm/all/poly.txt` | 2122321 | `d1fe3efafcd4377d41430d8828e881576c9024f9d7a5c89f55e23554e02ec800` | PC TSV + zip v2 eb268d6 | **ON** — то же |
| `models/li_gor_mask_r18.pt` | 57426355 | `39702cd8a5d92d0c0c06132690febc265519ed7a35b3c6ff02ed1fd7c1fdd8ef` | PC TSV | base lidar gor |
| `models/spor_gor_r18.pt` | 57425599 | `00d5c08539d1bced03d373c018d9e31432d987c7d848196a4afac6b0d44f1cbe` | PC TSV | base satellite |
| `models/dfine_f0/model.safetensors` | 125001980 | `81ef82280c43dd5e48255587f449683603ece375c976981db4f9a243e8509ed8` | PC TSV (zip v3) | SpOr D-FINE — **OFF** в v5 |


## Machine-readable expected hashes

См. `docs/weights_expected_sha256.json` — читается лоадерами / тестами. Env:
- `EXPDS_PIXGBM_SHA_STRICT=0` — отключить fail-loud для PIXGBM (локальные stub-тесты).
- `EXPDS_PIXGBM_EXPECTED_SHA256` — JSON override `{"poly":"...","finds":"..."}`.

## D-FINE Or (Kaggle `koz3-dfine-or`) — **EXCLUDED from v5** (bot6 honest); flag OFF/absent

| path | bytes | sha256 |
|---|---:|---|
| `models/dfine_or/dfine_f0/model.safetensors` | 125001980 | `82c8a47a2ad0d0b12fd5467b38592d31910114ad4029242d933f2e807c517d9c` |
| `models/dfine_or/dfine_f1/model.safetensors` | 125001980 | `7fcb58645366819e83f125b0078a5fdc0b4b896f1c2a1b0b3c995820586b6744` |
| `models/dfine_or/dfine_f2/model.safetensors` | 125001980 | `29dcd3c633c31aa153abe420bde59766de25fa662b4b2b3065f7a78dacd0ed2e` |

(c′) Or-only @`2b745df` — **WITHDRAWN 20:45 МСК** (PC QUEUE). bot6 `reports/bot6_ortho_multiclass.md`: provisional wF1≈0.036, nested non-arch 0 TP, arch@0.5 ΔLB≤0 within noise → **not proven**. Do not enable `[dfine_aeor]` in v5.

## D-FINE Ae (Kaggle `koz3-dfine-ae-ft`) — OFF

| path | sha256 |
|---|---|
| `models/dfine_ae/model.safetensors` (final) | `3e61987893b5da499482dc1af8520b0793894f73149fa3f2cddab8d1c9dfacad` |
| `models/dfine_f4/model.safetensors` (fold4) | `249b86fb9713cc1cb234cfbf11d239414092ef1af8178f434df2a38edfa94efa` |

## gor heads (retrained @9f62a96) — gor_dense **REJECTED** (Δ=0)

| path | bytes | sha256 | md5 |
|---|---:|---|---|
| `models/gor_gate/gor_v3_head.npz` | 407736 | `5bd88f69b3ffbfdc5312214b4d6012c92be12c66539b7bf8ad35564c76a75e7d` | `487c48e49b5945300cefeed70bc2df00` |
| `models/gor_dense/gor_dense_head.npz` | 204984 | `feb1591bb1e9ada63b32829178eae44a7e19fdfd7e99d7d222e3b1389c5b2cb7` | `d92f38f3cbf916a198b363b3f7c84158` |

## В git (box models/, не коммитить изменения rerank)

| path | sha256 | note |
|---|---|---|
| `models/rerank_arch_lgbm.txt` | `0bc2e924b7a7beb79706eae275d516a2668406294e99b6caf44e454030b6d97b` | **не трогать** |
| `models/li_gor_mask_r18.pt` | `39702cd8a5d92d0c0c06132690febc265519ed7a35b3c6ff02ed1fd7c1fdd8ef` | MATCH PC |
| `models/spor_gor_r18.pt` | `00d5c08539d1bced03d373c018d9e31432d987c7d848196a4afac6b0d44f1cbe` | MATCH PC |

Полный список lidar/ae/spor из v4 — см. WEIGHTS_MANIFEST_v4.md (без изменений для v5 tooling).

## Флаги v5 (defaults = eb268d6 / accepted-only)

| секция | enabled | причина |
|---|---|---|
| `[geophys.pixgbm]` / `[rerank_arch]` | `true` | база LB 0.41238 |
| `[sp_dfine]` | `false` | SpOr D-FINE OFF (bot2 FAIL / not accepted) |
| `[dfine_aeor]` | `false` | (c′) pending; Ae unproven |
| `[gor_dense]` / `[gor_gate]` | `false` | REJECTED Δ=0 |
| TTA / aerial / large_objects | **не в дереве** | unproven / REJECTED — см. exclude table в отчёте |
