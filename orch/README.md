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
