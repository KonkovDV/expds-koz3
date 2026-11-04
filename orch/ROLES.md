# Роли и передача работы (orch)

Источник задач — `orch/board.yaml`; статус — `orch/state.json` (пишет только диспетчер).
Любой бот берёт задачу, только если `owner` = он; GPU/TPU занимает диспетчер (аренда слота), не вручную.

| Бот | Роль | Ресурс | Основа |
|---|---|---|---|
| bot-1 | координатор LB, murder board, выбор 2 финальных сабмитов | — | docs/bots/2026-10-29/bot-1, LB_LOG |
| bot-2 | фьюжн/калибровка, ADAF | cpu | bot-2/STATUS.md |
| bot-3 | геофизика, geosnap | cpu | bot-3/STATUS.md |
| bot-4 | crop-verifier, kurgan-subclass | gpu (через диспетчер) | bot-4/CROPVERIF.md |
| bot-5 | городища (gor dense), RF-DETR | gpu | bot-5/* |
| bot-6 | aero/ortho D-FINE, ORGATE | gpu | bot-6/AEOR_DETECT.md, ORGATE.md |
| bot-7 | кандидат сабмита, флаги, конфиг | cpu | bot-7/REPORT4.md, MERGE_2026-10-29.md |
| bot-8 | honest-10 OOF (PR #7), бюджет, clean-room | cpu/gpu | bot-8/BUDGET.md |
| bot-9 | ConvNeXt лидар (GPU+TPU), контракт cnx | gpu/tpu | bot-9/CONVNEXT*.md |
| bot-10 | преррегистрации, red team, гейт #8 | cpu | bot-10/PREREG_*.md |
| Grok «2» | очередь TPU: держит оба TPU-слота занятыми, smoke→train | tpu@acc1/2 | CONVNEXT_TPU.md |
| Grok «3» | CPU: аудит recall аэро (GSD/CRS, площадь, NMS, пороги, dedupe, 20 000) | cpu-box | PLAN_PIVOT п.1, п.17 |
| Grok «4» | приёмка: honest-10 + гейт #8 по каждому COMPLETE-ядру | cpu-box | PR #7, #8 |
| Grok «5» | сборка сабмита, check_submission, архив ≤2 ГБ | cpu-box | RULES.md, check_submission |
| Grok Bot (основной) | диспетчер, доска, квоты, PR | box cron | orch/* |

## Передача
1. Ядро COMPLETE → диспетчер ставит `complete` → «4» скачивает выход, гоняет honest-10, пишет вердикт в PR/issue.
2. Прошло гейт → владелец открывает PR с фичей **за флагом (off)**; bot-7 включает в кандидате.
3. «5» собирает кандидата, `check_submission`, bot-1 решает о посылке (≤10 попыток, 2 успешные).
4. ERROR → диспетчер ретраит ≤2 раз; дальше `failed`, владелец чинит и меняет каталог (новый sha = новая версия).
5. Новые задачи — PR в `board.yaml` (id, owner, resource, kernel, dir, priority, est_hours, depends_on).
