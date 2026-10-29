# Манифест весов — кандидат v4 (bot-5)

База ветки: `bots/bot5-candidate-v4` ← `cbd48049ceaa6b56d0e4b85c72f4d35e839b2d3c`
(сборка v4 **ещё не** сливает v3-патчи / gor_dense / D-FINE Ae/Or — ждём фикс sp_or бота 2).

Правило 29.10: большие веса D-FINE / PIXGBM в git не добавляем (`[nomodels]`). Ниже — пути из кода/конфига на базе `cbd4804`, загрузчик, sha256 (или «неизвестно»), источник.

| path | loader | sha256 / контроль | source |
|---|---|---|---|
| `models/pixgbm/all/poly.txt` | `expds.geo_pixgbm.load_models` (LightGBM / lleaves) | `d1fe3efafcd4377d41430d8828e881576c9024f9d7a5c89f55e23554e02ec800` | не в git; stub `models/pixgbm/all/MODELS_NOT_IN_GIT.md`; оригинал `/workspace/koz3_shared/bot-3/pixgbm/models/all/` |
| `models/pixgbm/all/finds.txt` | то же | `5fa8c09d77d0914cc925fb6bb925a0ab1b64ccecf3c823d7f3546e71c4ceb04a` | то же |
| `models/dinov2-small/` | `gor_gate.load_backbone` / `gor_dense` (DEFAULTS `backbone`) | md5 каталога/артефакта `ea251be6e151ba8609ad0bc9b2e13a78` (sha256 неизвестно) | DEFAULTS `expds/gor_dense.py`; общий снапшот с gor_gate |
| `models/gor_gate/gor_v3_head.npz` | `gor_gate.load_head` | md5 `acb95c52956d712a16c3dd389237a691` (sha256 неизвестно) | DEFAULTS `gate_head` |
| `models/gor_dense/gor_dense_head.npz` | `gor_dense.load_dense_head` | неизвестно (`head_md5` в DEFAULTS пуст) | DEFAULTS `head`; export `scripts/gor_dense/export_head.py` |
| `models/dfine_f0/` (HF: config + model.safetensors) | `sp_dfine.load_models` | неизвестно | комментарий вкл. в `configs/default.toml` `[sp_dfine]`; по умолчанию `weights = []`, флаг **OFF** |
| `models/dfine_ae/`, `models/dfine_or/` | `dfine_aeor` → общий кэш с `sp_dfine` | неизвестно | комментарий вкл. `[dfine_aeor]`; флаг **OFF**, веса пусты до чекпойнтов bot-9 |
| `models/sp_crop.pt` | crop / `[candidates]` | неизвестно (файл в git) | `configs/default.toml` `weights = "models/sp_crop.pt"` |
| `models/rerank_arch_lgbm.txt` (+ `.json`) | `rerank_arch` | неизвестно (файл в git) | `[rerank_arch] enabled=true` — **не трогать** этот файл в коммитах bot-5 |
| `models/orgate_g.txt` (+ `.json`) | `orgate` | неизвестно (файл в git) | `[orgate] enabled=false` |
| `models/kurgan_lgbm.txt` | запасной оценщик лидара | неизвестно (файл в git) | `models/README.md` |
| `models/li_*.pt`, `models/spor_*.pt`, `models/ae_*.pt`, `models/surf_arch_r18.pt` | кандидаты лидара / спутника / аэро | неизвестно (файлы в git) | перечень в `models/README.md`; подключение через `[models]` / `[candidates]` |

## Флаги, от которых зависят веса (все OFF по умолчанию, кроме PIXGBM / rerank_arch / crop)

| секция | enabled default | независимость |
|---|---|---|
| `[sp_dfine]` | `false` | **отдельный** флаг D-FINE SpOr; не зависит от `[dfine_aeor]` |
| `[dfine_aeor]` | `false` | D-FINE Ae/Or; не включает SpOr |
| `[gor_gate]` | `false` | — |
| `[gor_dense]` | `false` | — |
| `[geophys.pixgbm]` | `true` | PIXGBM (нужны poly/finds вне git) |
| `[rerank_arch]` | `true` | — |
| `[orgate]` | `false` | — |

## Примечания

1. На базе `cbd4804` отдельный флаг SpOr уже есть: `[sp_dfine] enabled = false` (см. `tests/test_bot7_v3_flags.py::test_default_flags_off`). Новый флаг не добавлялся.
2. sha256 для `.pt` / HF-чекпойнтов на этой машине не считались (веса не скачивались); md5 взяты из DEFAULTS кода.
3. Сборка полного v4 (слияние патчей бота 3, gor_dense package, Ae/Or) **отложена** до фикса регрессии sp_or ботом 2.
