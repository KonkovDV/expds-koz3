# koz3-s3-index — индекс S3 bucket `train-expds-3`

CPU-only, `enable_internet=true`. Листинг Selectel S3 (`https://s3.ru-7.storage.selcloud.ru`)
через тот же `s3_client()`, что в `docs/bots/2026-10-29/bot-9/kaggle/dfine_spor/or_o2c/koz3-dfine-or.py`.

**Секреты Kaggle (имена):** `KOZ3_S3_ACCESS_KEY_ID`, `KOZ3_S3_SECRET_ACCESS_KEY`
(аккаунты `dmitrijkv` / `elenakonkova`). Значения секретов не печатаются.

## Выходы (`/kaggle/working/`)
| Файл | Содержание |
|---|---|
| `s3_index.txt` | `key\tsize\tlast_modified` (UTC ISO) на объект |
| `s3_prefix_summary.txt` | count + total_bytes по префиксам глубины ≤3 |
| `s3_hits_fam0.txt` | ключи с `fam0`, `stack_A`, `stack_arch`, `family_clusters`, `oof`, `jsonl` |
| `STATUS.txt` | `DONE` / `FAILED` |

Листинг: `list_objects_v2` с пагинацией по всему бакету и отдельно по top-level
префиксам (`Delimiter='/'`).

## Push (не пушить без coord)
```bash
orch/pool_kaggle.sh elenakonkova kernels push -p /workspace/kaggle/push/s3-index
```

## Статус / скачать выход
```bash
orch/pool_kaggle.sh elenakonkova kernels status elenakonkova/koz3-s3-index
orch/pool_kaggle.sh elenakonkova kernels output elenakonkova/koz3-s3-index -p /workspace/kaggle_out/koz3-s3-index
```

## Dry-run на box (mock boto3, без сети/секретов)
```bash
python /workspace/kaggle/push/s3-index/dryrun_mock.py
```
