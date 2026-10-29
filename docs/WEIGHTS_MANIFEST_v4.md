# Манифест весов — кандидат v4 (bot-5)

База ветки: `bots/bot5-candidate-v4` ← `cbd48049ceaa6b56d0e4b85c72f4d35e839b2d3c`  
Tip после манифеста: см. git log. Сборка v4 **ещё не** сливает/включает sp_or / gor_dense / D-FINE Ae/Or — ждём фикс sp_or бота 2.

Правило 29.10: большие веса D-FINE / PIXGBM в git не добавляем (`[nomodels]`). sha256 ниже посчитаны локально на worktree `/workspace/koz3-wt/bot5-candidate-v4` (29.10 МСК), кроме отмеченных «не на диске».

## В git (worktree models/)

| path | loader | sha256 | source |
|---|---|---|---|
| `models/li_r18_hard2.pt` | lidar candidates | `0311e877e148e0ec3acb6f3b13a7fb12affe5e0b58a60d3524a2df956b3793b1` | models/README + `[models]` |
| `models/li_r18_hard.pt` | lidar | `100bcc21afe8fc09be5362e19e2621ec815db28a38cad82a6bd9a5dffa7cc7d9` | в git |
| `models/li_r18_all.pt` | lidar | `2ed166cd9b37c3514438b885ec196cd6c54aeba9331bb5a4e81c55faa71ef531` | в git |
| `models/li_r18.pt` | lidar (ранний) | `c36149aba08bc619bdae67169fb51437bf4d08776f6f524b46d2a58b68ba90b5` | в git, в конфиге не стоит |
| `models/li_dev_r18.pt` | lidar σ | `2b655c809cb89c9394fb36ae9fec1d710b1d17d35a671834bdd80404b28265ab` | в git |
| `models/li_fort_mask_r18.pt` | lidar fort | `eb11cff8be474501d26fb6755268150abeaed42289512a6dd507b0571d4934e9` | в git |
| `models/li_gor_mask_r18.pt` | lidar gor | `39702cd8a5d92d0c0c06132690febc265519ed7a35b3c6ff02ed1fd7c1fdd8ef` | в git |
| `models/ae_fort_mask_r18.pt` | ae fort | `36d64e1a8a14d850dbf1494a11629e2a4ff7cea218b10dd462c5b827cf58bba7` | в git |
| `models/ae_r18.pt` | ae (ранний) | `83b4cc32b17f5c17604ffee530f4eec23f37b5017449390c6a70f80ea43cca98` | в git, в конфиге не стоит |
| `models/spor_gor_r18.pt` | satellite | `00d5c08539d1bced03d373c018d9e31432d987c7d848196a4afac6b0d44f1cbe` | в git |
| `models/spor_focal_r18.pt` | satellite | `88bbd1a24bf4bd4a2dae8c7855a95767e117124955273c9df1b2255c96fdd88b` | в git |
| `models/spor_arch_r18.pt` | satellite | `9478716f2b41567a25733919a92d55f2f199c8ca1d09c337ac2571705bb30cac` | в git |
| `models/spor_focal_wide_r18.pt` | satellite | `aa09855cb95babc8bfaf8018e6d32763ff0fd8356db5c4fb41fa31333a4f742c` | в git |
| `models/surf_arch_r18.pt` | surface | `8266add934346e5c675cda0c4aeb84ee39be3652e9308136a789673d768011a7` | в git |
| `models/sp_crop.pt` | crop / candidates | `4a41b0caada94d387721eb13526071469d5c70504b9dfb6743e7923a1f39cbeb` | `weights = "models/sp_crop.pt"` |
| `models/kurgan_lgbm.txt` | запасной lgbm | `f249a47e3305f21ac6bb45f3f35169019c52b93b53574d572b12ea28542e796c` | в git |
| `models/orgate_g.txt` | orgate | `676f4c1bbc8ff9d71b0bb86b39c9b870c10dc5a327bcf38d81c08e9d64db697a` | `[orgate]` OFF |
| `models/orgate_g.json` | orgate meta | `d398032068f1abc3460083dcba4066b0bd2edcb757303144a465686686632c5d` | в git |
| `models/rerank_arch_lgbm.txt` | rerank_arch | `0bc2e924b7a7beb79706eae275d516a2668406294e99b6caf44e454030b6d97b` | ON — **не трогать** в коммитах bot-5 |
| `models/rerank_arch_lgbm.json` | rerank meta | `d76c7edc54fc1c377a652122dcc8c7022e62244d48a6c0bbe85ecc19d2215ab0` | в git |

## Не в git / не на этом диске

| path | loader | sha256 / контроль | source |
|---|---|---|---|
| `models/pixgbm/all/poly.txt` | `geo_pixgbm.load_models` | `d1fe3efafcd4377d41430d8828e881576c9024f9d7a5c89f55e23554e02ec800` | stub `MODELS_NOT_IN_GIT.md`; файл на box **отсутствует** |
| `models/pixgbm/all/finds.txt` | то же | `5fa8c09d77d0914cc925fb6bb925a0ab1b64ccecf3c823d7f3546e71c4ceb04a` | то же |
| `models/dinov2-small/` | `gor_gate` / `gor_dense` backbone | md5 `ea251be6e151ba8609ad0bc9b2e13a78` (sha256 неизвестно; каталога нет) | DEFAULTS |
| `models/gor_gate/gor_v3_head.npz` | `gor_gate.load_head` | md5 `acb95c52956d712a16c3dd389237a691` (sha256 неизвестно; файла нет) | DEFAULTS |
| `models/gor_dense/gor_dense_head.npz` | `gor_dense.load_dense_head` | неизвестно (`head_md5` пуст; файла нет) | DEFAULTS |
| `models/dfine_f0/` | `sp_dfine.load_models` | неизвестно (нет на диске) | комментарий `[sp_dfine]`; флаг **OFF**, `weights=[]` |
| `models/dfine_ae/`, `models/dfine_or/` | `dfine_aeor` | неизвестно (нет на диске) | комментарий `[dfine_aeor]`; флаг **OFF** |

## Флаги (все новые OFF; SpOr отдельно)

| секция | enabled default | примечание |
|---|---|---|
| `[sp_dfine]` | `false` | **отдельный** D-FINE SpOr; не зависит от `[dfine_aeor]` |
| `[dfine_aeor]` | `false` | D-FINE Ae/Or |
| `[gor_gate]` / `[gor_dense]` | `false` | — |
| `[orgate]` | `false` | — |
| `[geophys.pixgbm]` / `[rerank_arch]` | `true` | уже в базе v2/v3 |

Поиск на box (29.10): `/workspace/koz3-wt/bot5-candidate-v4/models`, `/workspace/koz3-repo/models`, `/home/box/shared/koz3` (вкл. `bot6_cleanroom`, `box/`) — PIXGBM poly/finds, dinov2, gor_*.npz, D-FINE HF **не найдены**.
