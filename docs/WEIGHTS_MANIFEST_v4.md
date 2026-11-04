# Манифест весов — кандидат v4 (bot-5)

Историческая база манифеста (29.10): `cbd48049ceaa6b56d0e4b85c72f4d35e839b2d3c`.  
Актуальная ветка `bots/bot5-candidate-v4` идёт от `main` @ `f127db8` (этот коммит уже содержит `cbd4804`).  
Сборка v4 **не включает** D-FINE SpOr, gor_dense и D-FINE Ae/Or: флаги выключены, пока нет фикса локализации sp_or.

Правило 29.10: большие веса D-FINE / PIXGBM в git не добавляем (`[nomodels]`).

Источники sha256:
- **box worktree** `/workspace/koz3-wt/bot5-candidate-v4/models` (29.10) — файлы в git;
- **PC** `C:\Sloi`, бот 1, 29.10 — TSV `/home/box/shared/koz3/pc/results/weights_sha256_2026-10-29.tsv`.

## PC (бот 1, C:\Sloi, 29.10) — полные sha256 из TSV

| path | bytes | sha256 | source | сверка |
|---|---:|---|---|---|
| `models/dfine_f0/model.safetensors` | 125001980 | `81ef82280c43dd5e48255587f449683603ece375c976981db4f9a243e8509ed8` | PC `C:\Sloi`, bot 1, 29.10 (`pc_jobs\v3e2e\out_v3fix2\zip_v3\`) | новый (ранее неизвестно) |
| `models/pixgbm/all/finds.txt` | 2111596 | `5fa8c09d77d0914cc925fb6bb925a0ab1b64ccecf3c823d7f3546e71c4ceb04a` | PC `C:\Sloi`, bot 1, 29.10 (zip v2 eb268d6, zip v3, `C:\Sloi\wt_pix`) | **MATCH** stub `MODELS_NOT_IN_GIT.md` / прежний манифест |
| `models/pixgbm/all/poly.txt` | 2122321 | `d1fe3efafcd4377d41430d8828e881576c9024f9d7a5c89f55e23554e02ec800` | PC `C:\Sloi`, bot 1, 29.10 (same) | **MATCH** stub / прежний манифест |
| `models/li_gor_mask_r18.pt` | 57426355 | `39702cd8a5d92d0c0c06132690febc265519ed7a35b3c6ff02ed1fd7c1fdd8ef` | PC `C:\Sloi`, bot 1, 29.10 | **MATCH** box worktree |
| `models/spor_gor_r18.pt` | 57425599 | `00d5c08539d1bced03d373c018d9e31432d987c7d848196a4afac6b0d44f1cbe` | PC `C:\Sloi`, bot 1, 29.10 | **MATCH** box worktree |

Несовпадений PIXGBM / li_gor / spor_gor с эталонами манифеста **нет**.

## Pending (нет на C:\Sloi)

| path | loader | статус | source |
|---|---|---|---|
| `models/dinov2-small/` | `gor_gate` / `gor_dense` backbone | **OK on PC** `C:\Sloi\wt_bot5_50f6f27\models\dinov2-small` (md5 `ea251be6…`); box `/tmp/dinov2-small` | PC bot1; DEFAULTS |
| `models/gor_gate/gor_v3_head.npz` | `gor_gate.load_head` | **READY (retrained)** — see gor heads table below | PC Docker @`9f62a96`, bot1, 29.10 |
| `models/gor_dense/gor_dense_head.npz` | `gor_dense.load_dense_head` | **READY (retrained)** — see gor heads table below; centre_s=250 | same |
| `models/dfine_or/dfine_f{0,1,2}/` | `dfine_aeor` (or) | **READY** — see Or table below (Kaggle `koz3-dfine-or`) | SHA256SUMS verified 29.10 |
| `models/dfine_ae/` | `dfine_aeor` (ae) | **pending** — bot3 fine-tune kernel `koz3-dfine-ae-ft` still running | Kaggle |



## D-FINE Or (Kaggle `koz3-dfine-or`, pull `/workspace/kaggle/pulls/koz3-dfine-or`) — 29.10.2026

SHA256SUMS recomputed on box: **all OK**. Source: **Kaggle koz3-dfine-or**.

| path | bytes | sha256 |
|---|---:|---|
| `models/dfine_or/dfine_f0/model.safetensors` | 125001980 | `82c8a47a2ad0d0b12fd5467b38592d31910114ad4029242d933f2e807c517d9c` |
| `models/dfine_or/dfine_f1/model.safetensors` | 125001980 | `7fcb58645366819e83f125b0078a5fdc0b4b896f1c2a1b0b3c995820586b6744` |
| `models/dfine_or/dfine_f2/model.safetensors` | 125001980 | `29dcd3c633c31aa153abe420bde59766de25fa662b4b2b3065f7a78dacd0ed2e` |
| `models/dfine_or/dfine_f{0,1,2}/config.json` | 4222 | `ff8bb6305c227c5e52ba7f7cc16bf0edaa7a9ac085104452b0aec3b6b9e90789` (identical) |
| `oof_dfine_or.geojson` | — | `5c8758f33e6c7e1fa79b3dfb9576201a2574dbbe7d72e545838f47ea502fd45f` |
| `dfine_report.json` | — | `43413de64f7ab153c594ef185efc5b906a8b4a27e96025f1a9b6d8dd77e8ed07` |

**Quality note (from `dfine_report.json` + OOF, not yet honest-10 F1):** 3 folds × 2400 iters, total_sec≈5333 (~1.48 h), peak VRAM 7.76 GB. OOF union `oof_dfine_or.geojson`: **1228** dets (f0=258, f1=940, f2=30). Class mix: arkhitektury 1110, fortifikatsii 50, gorodishcha 44, kurgany_tselye 17, kurgany_povrezhdennye 7. Confidence quartiles ≈ 0.071 / 0.113 / 0.202. Loss last ≈ 26.3 / 31.4 / 26.5. **No platform F1 in kernel report** — needs PC/Kaggle honest eval. Ae still pending (`koz3-dfine-ae-ft`).

**dfine_aeor Or-only:** possible — set `[dfine_aeor] markups = ["or"]` + `weights.or = ["models/dfine_or/dfine_f0", …]` + thresholds for `dfine_or`; leave ae out until FT finishes. Full Ae+Or mock waits for Ae weights.

## gor heads (retrain PC Docker `pantum/n7q:20261024` @ `9f62a96`) — 29.10.2026

Source: `/home/box/shared/koz3/pc/results/bot5_gor_heads_9f62a96/` (STATUS DONE 14:58:23Z = 17:58 МСК; ckpt stages extract_v3/fit_v3/extract_dense/inputs_dense/export_dense/validate all DONE). sha256 recomputed on box: **MATCH** SUMMARY.json/HEADS_SHA256.txt; sizes = legacy.

| path | bytes | sha256 | md5 | legacy md5 |
|---|---:|---|---|---|
| `models/gor_gate/gor_v3_head.npz` | 407736 | `5bd88f69b3ffbfdc5312214b4d6012c92be12c66539b7bf8ad35564c76a75e7d` | `487c48e49b5945300cefeed70bc2df00` | `acb95c52956d712a16c3dd389237a691` (≠) |
| `models/gor_dense/gor_dense_head.npz` | 204984 | `feb1591bb1e9ada63b32829178eae44a7e19fdfd7e99d7d222e3b1389c5b2cb7` | `d92f38f3cbf916a198b363b3f7c84158` | `e35570e10d17edac32bea8d26fa0f5f4` (≠) |

Not byte-identical to legacy (features re-extracted; recipe + SEED=0 match). DEFAULTS md5 in `expds/gor_dense.py` point to legacy → e2e ON must override `gate_head_md5` / `head_md5` with the new md5. τ=0.824066 was tuned on legacy heads — e2e decides. `.npz` gitignored (not in git); on PC: `C:\Sloi\pc_jobs\bot5_gor_heads_9f62a96\heads\`.

## В git (box worktree models/)

| path | loader | sha256 | source |
|---|---|---|---|
| `models/li_r18_hard2.pt` | lidar candidates | `0311e877e148e0ec3acb6f3b13a7fb12affe5e0b58a60d3524a2df956b3793b1` | box worktree 29.10 |
| `models/li_r18_hard.pt` | lidar | `100bcc21afe8fc09be5362e19e2621ec815db28a38cad82a6bd9a5dffa7cc7d9` | box worktree 29.10 |
| `models/li_r18_all.pt` | lidar | `2ed166cd9b37c3514438b885ec196cd6c54aeba9331bb5a4e81c55faa71ef531` | box worktree 29.10 |
| `models/li_r18.pt` | lidar (ранний) | `c36149aba08bc619bdae67169fb51437bf4d08776f6f524b46d2a58b68ba90b5` | box worktree 29.10 |
| `models/li_dev_r18.pt` | lidar σ | `2b655c809cb89c9394fb36ae9fec1d710b1d17d35a671834bdd80404b28265ab` | box worktree 29.10 |
| `models/li_fort_mask_r18.pt` | lidar fort | `eb11cff8be474501d26fb6755268150abeaed42289512a6dd507b0571d4934e9` | box worktree 29.10 |
| `models/li_gor_mask_r18.pt` | lidar gor | `39702cd8a5d92d0c0c06132690febc265519ed7a35b3c6ff02ed1fd7c1fdd8ef` | box + PC MATCH |
| `models/ae_fort_mask_r18.pt` | ae fort | `36d64e1a8a14d850dbf1494a11629e2a4ff7cea218b10dd462c5b827cf58bba7` | box worktree 29.10 |
| `models/ae_r18.pt` | ae (ранний) | `83b4cc32b17f5c17604ffee530f4eec23f37b5017449390c6a70f80ea43cca98` | box worktree 29.10 |
| `models/spor_gor_r18.pt` | satellite | `00d5c08539d1bced03d373c018d9e31432d987c7d848196a4afac6b0d44f1cbe` | box + PC MATCH |
| `models/spor_focal_r18.pt` | satellite | `88bbd1a24bf4bd4a2dae8c7855a95767e117124955273c9df1b2255c96fdd88b` | box worktree 29.10 |
| `models/spor_arch_r18.pt` | satellite | `9478716f2b41567a25733919a92d55f2f199c8ca1d09c337ac2571705bb30cac` | box worktree 29.10 |
| `models/spor_focal_wide_r18.pt` | satellite | `aa09855cb95babc8bfaf8018e6d32763ff0fd8356db5c4fb41fa31333a4f742c` | box worktree 29.10 |
| `models/surf_arch_r18.pt` | surface | `8266add934346e5c675cda0c4aeb84ee39be3652e9308136a789673d768011a7` | box worktree 29.10 |
| `models/sp_crop.pt` | crop / candidates | `4a41b0caada94d387721eb13526071469d5c70504b9dfb6743e7923a1f39cbeb` | box worktree 29.10 |
| `models/kurgan_lgbm.txt` | запасной lgbm | `f249a47e3305f21ac6bb45f3f35169019c52b93b53574d572b12ea28542e796c` | box worktree 29.10 |
| `models/orgate_g.txt` | orgate | `676f4c1bbc8ff9d71b0bb86b39c9b870c10dc5a327bcf38d81c08e9d64db697a` | box worktree 29.10 |
| `models/orgate_g.json` | orgate meta | `d398032068f1abc3460083dcba4066b0bd2edcb757303144a465686686632c5d` | box worktree 29.10 |
| `models/rerank_arch_lgbm.txt` | rerank_arch | `0bc2e924b7a7beb79706eae275d516a2668406294e99b6caf44e454030b6d97b` | ON — **не трогать** в коммитах bot-5 |
| `models/rerank_arch_lgbm.json` | rerank meta | `d76c7edc54fc1c377a652122dcc8c7022e62244d48a6c0bbe85ecc19d2215ab0` | box worktree 29.10 |

## Флаги на `cbd4804` (снимок 29.10, не текущий `default.toml`)

| секция | enabled default | примечание |
|---|---|---|
| `[sp_dfine]` | `false` | **отдельный** D-FINE SpOr; веса `dfine_f0` на PC подтверждены; не зависит от `[dfine_aeor]` |
| `[dfine_aeor]` | `false` | Or weights **READY** (table above); Ae **pending** bot3 FT |
| `[gor_gate]` / `[gor_dense]` | `false` | backbone OK on PC; heads **retrained** @ `9f62a96` (new md5 → set `gate_head_md5`/`head_md5` when enabling) |
| `[orgate]` | `false` | — |
| `[geophys.pixgbm]` / `[rerank_arch]` | `true` на том снимке | PIXGBM sha256 = stub v2 eb268d6 |

## Сверка клона 2026-11-01 (ветка `bots/bot5-candidate-v4`, база `main` @ `f127db8`)

Файл `models/rerank_arch_lgbm.txt` в этом клоне **нет**. Его не восстанавливали: этим файлом занимается другой агент. sha256 из таблицы выше (`0bc2e924…`) снят 29.10, здесь не пересчитан.

Пересчитано sha256 файлов, которые в клоне есть. Совпало с таблицей «В git»:

| path | sha256 | в git на этой ветке |
|---|---|---|
| `models/li_r18_hard2.pt` | `0311e877e148e0ec3acb6f3b13a7fb12affe5e0b58a60d3524a2df956b3793b1` | да |
| `models/li_gor_mask_r18.pt` | `39702cd8a5d92d0c0c06132690febc265519ed7a35b3c6ff02ed1fd7c1fdd8ef` | да |
| `models/spor_gor_r18.pt` | `00d5c08539d1bced03d373c018d9e31432d987c7d848196a4afac6b0d44f1cbe` | да |
| `models/spor_focal_r18.pt` | `88bbd1a24bf4bd4a2dae8c7855a95767e117124955273c9df1b2255c96fdd88b` | да |
| `models/ae_fort_mask_r18.pt` | `36d64e1a8a14d850dbf1494a11629e2a4ff7cea218b10dd462c5b827cf58bba7` | да |
| `models/sp_crop.pt` | `4a41b0caada94d387721eb13526071469d5c70504b9dfb6743e7923a1f39cbeb` | да |
| `models/kurgan_lgbm.txt` | `f249a47e3305f21ac6bb45f3f35169019c52b93b53574d572b12ea28542e796c` | да |
| `models/orgate_g.txt` | `676f4c1bbc8ff9d71b0bb86b39c9b870c10dc5a327bcf38d81c08e9d64db697a` | да |
| `models/orgate_g.json` | `d398032068f1abc3460083dcba4066b0bd2edcb757303144a465686686632c5d` | да |
| `models/rerank_arch_lgbm.json` | `d76c7edc54fc1c377a652122dcc8c7022e62244d48a6c0bbe85ecc19d2215ab0` | да |
| `models/dfine_aeor/thresholds_or_oof_provisional.json` | `380274a11c1875e6898ce5465fff6359591650624698c17e68cd07e70354683d` | да, 178 байт, не веса сети |

Нет в этом клоне (sha256 неизвестен здесь; числа с ПК/Kaggle — в таблицах выше, файлы не пересчитывались):

| path | зачем в коде | источник |
|---|---|---|
| `models/pixgbm/all/poly.txt` | `[geophys.pixgbm]`, сейчас `enabled=false` | обучение bot-3; sha256 в `MODELS_NOT_IN_GIT.md` и в таблице PC выше |
| `models/pixgbm/all/finds.txt` | то же | то же |
| `models/rerank_arch_lgbm.txt` | `[rerank_arch] model`, сейчас `enabled=false` | `scripts/rerank_arch_train.py`; файл в git не класть этим PR |
| `models/dfine_f0/` (`config.json` + `model.safetensors`) | `[sp_dfine] weights`, флаг выкл. | bot-9, Kaggle `koz3-dfine-spor`, фолд f0; sha256 PC `81ef8228…` |
| `models/dfine_ae/` | `[dfine_aeor] weights.ae` | bot-9 / дообучение `koz3-dfine-ae-ft`; в манифесте 29.10 — pending |
| `models/dfine_or/dfine_f{0,1,2}/` | `[dfine_aeor] weights.or` | Kaggle `koz3-dfine-or`; sha256 в таблице Or выше |
| `models/dinov2-small/` | `[gor_gate]` / `[gor_dense] backbone` | HF `facebook/dinov2-small` @ `ed25f3a3`; md5 safetensors `ea251be6…` |
| `models/gor_gate/gor_v3_head.npz` | голова сцены | `scripts` bot-5 / ретрейн PC @ `9f62a96`; sha256 `5bd88f69…` |
| `models/gor_dense/gor_dense_head.npz` | плотная голова, `centre_s=250` | `scripts/gor_dense/export_head.py`; sha256 `feb1591b…` |

Флаги в текущем `configs/default.toml` (не в архивном снимке 29.10):

| секция | enabled |
|---|---|
| `[sp_dfine]` | false, отдельно от `[dfine_aeor]` и от `[aeor_dfine]` |
| `[dfine_aeor]` | false |
| `[aeor_dfine]` | false |
| `[gor_dense]` / `[gor_gate]` | false; пакет gor_dense: τ=0.824066, `centre_s=250` только в закомментированном блоке включения |
| `[orgate]` | false |
| `[geophys.pixgbm]` | false |
| `[rerank_arch]` | false |
