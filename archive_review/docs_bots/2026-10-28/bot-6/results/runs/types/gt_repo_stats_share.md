# Эталон обучающей выборки: сводка
Регионов: 210; файлов разметки: 456; объектов: 10979; слияний внутри markup_type: 4 ({'ae': 0, 'li': 0, 'or': 0, 'sp_or': 0, 'maggr': 4, 'terfl': 0}).

## Объекты по классам и модальностям
| class_name | markup_type | objects | regions | points | area_m2_p10 | area_m2_p50 | area_m2_p90 |
|---|---|---|---|---|---|---|---|
| kurgany_povrezhdennye | li | 1371 | 38 | 0 | 21.6 | 56.9 | 178.3 |
| kurgany_povrezhdennye | ae | 1162 | 23 | 0 | 31.7 | 78.7 | 1506.9 |
| kurgany_povrezhdennye | sp_or | 1087 | 60 | 0 | 94.5 | 491.5 | 2081.3 |
| kurgany_tselye | li | 783 | 23 | 0 | 16.1 | 39.1 | 147.9 |
| kurgany_tselye | ae | 587 | 18 | 0 | 21.2 | 62.1 | 329.9 |
| kurgany_tselye | sp_or | 207 | 30 | 0 | 44.7 | 449.3 | 2433.9 |
| gorodishcha | sp_or | 65 | 49 | 0 | 3505.4 | 9206.2 | 56586.5 |
| gorodishcha | li | 30 | 24 | 0 | 1704.0 | 6109.6 | 38303.0 |
| gorodishcha | ae | 19 | 12 | 0 | 4001.8 | 29922.7 | 83067.3 |
| gorodishcha | or | 4 | 2 | 0 | 3409.4 | 5802.4 | 13130.0 |
| arkhitektury | sp_or | 210 | 28 | 0 | 27.5 | 182.3 | 738.4 |
| arkhitektury | or | 178 | 4 | 0 | 0.1 | 14.8 | 84.8 |
| arkhitektury | ae | 18 | 2 | 0 | 429.5 | 1810.5 | 3494.3 |
| arkhitektury | li | 1 | 1 | 0 | 543.0 | 543.0 | 543.0 |
| object_poly | etgr | 658 | 6 | 0 | 0.4 | 2.7 | 16.9 |
| object_poly | maggr | 248 | 13 | 0 | 13.7 | 42.7 | 254.1 |
| object_poly | terfl | 177 | 2 | 0 | 3.3 | 18.7 | 85.6 |
| finds_points | maggr | 3478 | 13 | 3478 |  |  |  |
| fortifikatsii | ae | 393 | 13 | 0 | 77.2 | 339.3 | 3408.0 |
| fortifikatsii | li | 264 | 17 | 0 | 49.2 | 125.6 | 1068.9 |
| fortifikatsii | sp_or | 37 | 5 | 0 | 52.6 | 94.5 | 1610.8 |
| fortifikatsii | or | 2 | 1 | 0 | 1761.2 | 2055.9 | 2350.6 |

## Дубли между модальностями (центры ближе 5 м)
| class_name | a | b | pairs |
|---|---|---|---|
| fortifikatsii | ae | li | 3 |
| fortifikatsii | ae | or | 0 |
| fortifikatsii | ae | sp_or | 0 |
| fortifikatsii | or | sp_or | 1 |
| gorodishcha | ae | li | 2 |
| gorodishcha | ae | or | 0 |
| gorodishcha | ae | sp_or | 2 |
| gorodishcha | li | or | 2 |
| gorodishcha | li | sp_or | 1 |
| gorodishcha | or | sp_or | 0 |
| kurgany_povrezhdennye | ae | li | 386 |
| kurgany_povrezhdennye | ae | sp_or | 33 |
| kurgany_povrezhdennye | li | sp_or | 64 |
| kurgany_tselye | ae | li | 82 |
| kurgany_tselye | ae | sp_or | 0 |
| kurgany_tselye | li | sp_or | 1 |
| object_poly | etgr | maggr | 14 |
| object_poly | etgr | terfl | 101 |
| object_poly | maggr | terfl | 17 |

## Один курган — разные классы (целый/повреждённый)
| a | b | pairs |
|---|---|---|
| ae | ae | 6 |
| ae | li | 205 |
| ae | sp_or | 5 |
| li | li | 17 |
| li | sp_or | 2 |

## CRS файлов разметки
| crs | files |
|---|---|
| EPSG:32636 | 103 |
| EPSG:32637 | 4 |
| EPSG:3857 | 345 |
| EPSG:4326 | 4 |

## Проблемы
| problem | files |
|---|---|
| 1 линий/прочих геометрий | 16 |
| 10 линий/прочих геометрий | 1 |
| 11 линий/прочих геометрий | 2 |
| 15 линий/прочих геометрий | 2 |
| 2 линий/прочих геометрий | 8 |
| 23 линий/прочих геометрий | 1 |
| 3 линий/прочих геометрий | 8 |
| 33 линий/прочих геометрий | 1 |
| 4 линий/прочих геометрий | 5 |
| 5 линий/прочих геометрий | 3 |
| 6 линий/прочих геометрий | 1 |
| 8 линий/прочих геометрий | 1 |
| 9 линий/прочих геометрий | 1 |
| код модальности набран кириллицей | 7 |
