# EXPDS_MATCH_SCOPE=layer — сопоставление по слою данных

Диагностический режим честного offline-eval: пара pred↔gt допускается только если
нормализованный `markup_type` совпадает (`expds.metric.layer_key`).

## Включение

По умолчанию **выключено** (`global` — как официальный `F1Calculator.calculate_f1`).

```bash
# через env
EXPDS_MATCH_SCOPE=layer python scripts/scoreboard.py --work ... --train ... --honest 10

# или флаг scoreboard
python scripts/scoreboard.py --work ... --train ... --honest 10 --match-scope layer
```

Допустимые значения: `global` (default) | `site` | `layer`. `site`/`layer` печатают
предупреждение в stderr — не для сдачи.

## Ключ слоя

| markup_type | layer_key |
|-------------|-----------|
| la, li      | li        |
| ae, german_ae | ae      |
| sp_or, spor | sp_or     |
| or, maggr, etgr, geogr, terfl | как есть (отдельные) |
| пусто / нет | `?`       |
| неизвестный | lowercase как есть |

Счётчики TP/FP/FN по слоям **складываются** (микро-F1), аналогично `scope=site` по участкам.

## Зачем

Сверка гипотезы «honest выше LB, потому что матчит между слоями». Ожидание: layer
**не поднимает** honest относительно global (скорее снизит или оставит); разрыв
honest↔LB объясняется train-OOF vs public val, а не cross-layer matching.

Тесты: `tests/test_bot7_layer_scope.py`.

## Диагностика Δ global↔layer

```bash
python scripts/diff_match_scope.py --work F:/expds3/work --train F:/2/train-expds-3 \
    --out notes/bot7_diff_scope/
# или --pred stack.geojson --gt gt.geojson
```

Пишет `summary.json`, `cross_matches.csv` (pred_layer→gt_layer, class, quality),
`tp_by_class_layers.csv`, `delta_by_class.csv`. API: `expds.metric.build_curves_with_pairs`.
