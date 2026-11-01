# koz3-w4-large-gate (CPU honest-10 · large_objects)

Обертка над `scripts/large_objects_honest.py` (tip **33791bcc**, PR #18).
Варианты: `off` / `area_tau` / `fixc_nested`. **Не сдавать на LB.**

## Lease
- `slot:acc1/cpu` → **bot-W4-gate** до **02.11 07:14 Europe/Moscow**
- kernel: `elenakonkova/koz3-w4-large-gate`

## Что в xfer-s000..s009
| Shard | Содержимое |
|---|---|
| s000 | `small_files.tar.zst` — разметка регионов (geojson), UTM.json |
| s001–s009 | крупные блобы растров (`f####` → относительный путь в train tree) |
| + `koz3-gt-train-v2` | сводный `gt_train_v2.geojson` |

**fam0 OOF jsonl в xfer нет** (протокол данных). Ядро:
1. (опц.) `rebuild_train_tree.py` → `/kaggle/working/train-expds-3`
2. GT из `koz3-gt-train-v2` (fallback — консолидация labels из дерева)
3. fam0 — из прикреплённого `*fam0*`/`*fam*` датасета, либо `split_stack_by_family` от `stack*.geojson`, либо `LARGE_GATE_SMOKE=1` (синтетика, не stack A)
4. `family_clusters.py` → clusters
5. `large_objects_honest.py` → JSON

## Успех
exit 0 + `large_gate/large_objects_honest.json` с ключами `off`, `area_tau`, `fixc_nested`
(score, precision, recall, delta_*, size_recall, recovered_gt_gt5k).
Полный stack A: `off.score ≈ 0.314615 ± 0.002`; wall **30–90 мин** / 1 ядро.
SMOKE: секунды, score ≠ 0.314615.

## Push (делает coord, этот пакет не пушим сами)
```bash
orch/lease.sh who slot:acc1/cpu   # bot-W4-gate
orch/pool_kaggle.sh elenakonkova kernels push -p /workspace/kaggle/push/large-gate
```

Для полного замера приложите датасет fam0 (stack A OOF jsonl) в `dataset_sources`
(например `elenakonkova/koz3-stack-a-fam0`) — без него ядро уйдёт в SMOKE, если
не выставить явно иное.

## Dry-run на боксе
```bash
LARGE_GATE_SMOKE=1 KAGGLE_INPUT=/tmp/lg_empty \
  /workspace/koz3-venv-dry/bin/python koz3-w4-large-gate.py
```
Лог: `dryrun/dryrun.log`.
