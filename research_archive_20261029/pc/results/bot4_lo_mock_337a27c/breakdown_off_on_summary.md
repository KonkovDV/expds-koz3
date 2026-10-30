# OFF vs ON mock e2e @ 337a27c — разбор

Источник: out_off/out_on result.geojson + compute_metrics.F1Calculator (тот же, что e2e_score); counts из e2e_off_on.json; логи main_on.log.

## Сводка
| Слой | n OFF→ON | F1 OFF | F1 ON | ΔF1 |
|---|---:|---:|---:|---:|
| ALL | 246→249 | 0.31287 | 0.29658 | **−0.01629** |
| li | 143→146 | 0.33698 | 0.32160 | **−0.01538** |
| ae | 59→59 | 0.09556 | 0.09556 | 0 |
| sp_or | 44→44 | 0.24506 | 0.24506 | 0 |

## li / gorodishcha (единственный изменившийся класс)
| | OFF | ON | Δ |
|---|---:|---:|---:|
| n | 9 | 12 | +3 |
| TP | 1 | 1 | 0 |
| FP | 8 | 11 | **+3** |
| FN | 0 | 0 | 0 |
| P | 0.2000 | 0.1538 | −0.0462 |
| R | 1.0 | 1.0 | 0 |
| F1 | 0.3333 | 0.2667 | −0.0667 |

Прочие классы: dtp=dfp=dfn=0 (kurgany/fort/arkhitektury без изменений счётчиков).

## Механизмы (логи main_on)
- `large_objects coarse hillfort 3` → +3 сырых gorodishcha
- `large_objects fort→gor копий 1` → копия создана, но net final gor +3 (не +4)
- shape_cores: площади крупных gor сжаты без сдвига TP/FP
- area_tau / protect_merge: нет следов в счётчиках других классов на этом mock

## Вердикт причины
**Главный виновник: coarse_hillfort** (+3 FP li/gorodishcha, TP без прироста).
fort→gor — не драйвер ΔF1 на mock. shape_cores — мутирует геометрию, на mock TP/FP gor не сдвинул.
