# Архивные ветки (29.10.2026)

Ветки **не влиты** в `main` и **не удалены**. На их головах поставлены аннотированные теги `archive/*`
(запушены), чтобы содержимое не потерялось, если ветки потом удалят. Удаление веток — только решение владельца.

| Ветка | Голова | Тег | Впереди `main` 164707a |
|---|---|---|---|
| `bots/bot8-cpu-a-clean` | `95bc0a6a357d039872fa03769fa3265240c93e89` | `archive/bots-bot8-cpu-a-clean` | 8 коммитов |
| `bots/bot8-cpu-a-clean-on-v2` | `18b2f66acafd1d65329460376a58af0dd810bcef` | `archive/bots-bot8-cpu-a-clean-on-v2` | 9 коммитов |
| `speedup-2026-10-28` | `6602bb23a88076969e909e674f9a896070db1c38` | `archive/speedup-2026-10-28` | 1 коммит |

Почему не влиты:

* `bots/bot8-cpu-a-clean`, `bots/bot8-cpu-a-clean-on-v2` — **не clean room**: `docs/CLEANROOM_SPEC_lidar_a.md`
  написан тем же агентом по поведению EhimenNathan-EXpedition @d434e56; `tests/lidar_a_oracle.py` /
  `tests/test_lidar_a_oracle.py` загружают репозиторий конкурента. Плюс бинарь
  `rust/expds_fast/wheels/expds_fast-0.1.0-cp312-abi3-manylinux…whl`.
* `speedup-2026-10-28` — меняет пути CPU по умолчанию («байт-в-байт с 14d419…» заявлено, но не проверено
  на v2/main) и добавляет тот же `.whl`.

Восстановить: `git checkout -b <имя> archive/<тег>`.
