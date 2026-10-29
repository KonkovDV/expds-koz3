# orch — оркестрация вычислений и ботов KOZ3

* `board.yaml` — единая доска: задачи, владельцы, ресурс (`cpu-box|gpu@acc1|gpu@acc2|tpu@acc1|tpu@acc2`),
  ядро и каталог, priority, depends_on, приёмка (honest-10). Политика квот и ретраев.
* `dispatcher.py` — раз в 10 мин (cron на боксе Grok Bot): опрос Kaggle → `state.json` (runtime, не в git) →
  освобождение слотов → оценка недельной квоты (по логам; API квоту не отдаёт) → пуш следующей задачи.
* `ROLES.md` — роли ботов и правила передачи.

Гарантии: не трогает RUNNING/QUEUED; не перепушивает ту же версию (sha каталога); ≤2 ретраев;
preflight доступа к датасетам (приватные `dmitrijkv/*` недоступны с acc1 → задача уходит на alt-ресурс или ждёт);
запас квоты `safety_margin_h`; лок-файл от параллельных запусков; `--dry-run`.

```bash
KAGGLE_JSON_1=… KAGGLE_JSON_2=… python orch/dispatcher.py --dry-run
```
Токены: `KAGGLE_JSON_n` = JSON kaggle.json или новый токен (`KAGGLE_API_TOKEN`). Не логируются.

## Данные конкурса на CPU-боксе (S3 организаторов)
- Креды S3 (публикуются организаторами участникам) — **только** в `~/.config/koz3-orch/s3.env` (chmod 600, в git не коммитятся). Там же remote rclone `expds:` через переменные `RCLONE_CONFIG_EXPDS_*`.
- Синхронизация (возобновляемая, идемпотентная): `~/.config/koz3-orch/s3_sync.sh` → `/workspace/data/train-expds-3`, лог `/workspace/data/s3_sync.log`. По умолчанию тянет разметку (`*.geojson`) + метаданные (`*.json`, UTM) + аэрофото `*_Ae*` (~7 ГиБ). Весь бакет ~208 ГиБ (Li 97, Or 61, La 37) на бокс не влезает.
- CPU-задачи доски (`aero-recall-audit`, `honest10-gate`, `submission-assembly`) читают данные через `KOZ3_DATA=/workspace/data/train-expds-3`; перед запуском: `set -a; . ~/.config/koz3-orch/s3.env; set +a`. Докачка конкретного региона: `rclone copy expds:train-expds-3/<NNN_регион> $KOZ3_DATA/<NNN_регион>`.
