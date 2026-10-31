# lidar_a: ground.rasterise → Rust expds_fast (las_grid_fast), бот 8, 28.10.2026 (МСК)

Коммиты: bot8-cpu-a-clean: 7f8d714 (частичный cherry-pick 6602bb2, автор KonkovDV) + 95bc0a6 (подключение);
bot8-cpu-a-clean-on-v2: bf784eb + 662e7d1 (те же) + 18b2f66 (build_submission_git кладёт expds_fast.so из колеса).
Скрипт: bench/bench_las_grid_fast.py; JSON: bench/las_grid_fast/site_{016,001,022}.json.
Условия: box, 2 потока (OMP/MKL/OPENBLAS/RAYON/EXPDS_CPU_THREADS=2), nice 10, LAZ читается с диска (lazrs),
медиана из 3, wall / CPU процесса, load1 ≈ 6–8 (box общий). Пик RSS процесса ≤ 1347 МиБ.

## Равенство (побайтно, tobytes)
| участок | точек | ячеек | rasterise Rust == numpy | EXPDS_FAST=0 == numpy | bare/lowest Rust == numpy | DTM+valid Rust == numpy | DTM+valid Rust == эталон (oracle dtm.build) |
|---|---|---|---|---|---|---|---|
| 016 | 0,27 M | 2,77 M | да | да | да / да | да | да |
| 001 | 7,40 M | 6,42 M | да | да | да / да | да | да |
| 022 | 8,92 M | 13,5 M | да | да | да / да | да | да |
(все три участка в режиме "class"; bare/lowest проверены принудительно на тех же облаках.)

## Ускорение (wall с / CPU с, медиана из 3)
| участок | rasterise numpy | rasterise Rust | × wall / CPU | make_dtm numpy | make_dtm Rust | × wall / CPU |
|---|---|---|---|---|---|---|
| 016 | 0,165 / 0,225 | 0,074 / 0,136 | 2,24 / 1,65 | 0,507 / 0,643 | 0,462 / 0,622 | 1,10 / 1,03 |
| 001 | 2,886 / 4,278 | 1,727 / 2,914 | 1,67 / 1,47 | 3,981 / 5,890 | 2,818 / 4,609 | 1,41 / 1,28 |
| 022 | 3,719 / 5,146 | 2,059 / 3,804 | 1,81 / 1,35 | 6,527 / 8,036 | 4,660 / 6,287 | 1,40 / 1,28 |

Почему × меньше, чем у speedup (×2,5–3 на 105): там LAZ уже распакован в память; здесь в обоих путях
есть одинаковая распаковка LAZ (lazrs), она и остаётся в Rust-пути. Экономия на DTM: 1,2 с (001), 1,9 с (022) wall.
Экстраполяция [оц]: 101 (7421×10947 = 81 Мячеек) не запускался (numpy-эталон 2,97 ГБ > лимита 2 ГБ); по данным speedup Rust-путь
на 101 — пик 0,98 ГБ, 9,8 с; numpy-путь по нашим ячейкам/точкам ~×1,4–1,8 медленнее на rasterise.

## CPU-рельеф через torch_all_cores() (runner.stacks_for, dev=cpu)
016, 9 каналов: 1 поток 0,380 / 0,412 с, 2 потока 0,311 / 0,463 с → ×1,22 wall; результат 1 vs 2 потока побайтно равен.
На box число потоков ограничено EXPDS_CPU_THREADS=2 (добавлено в bot-8/env.sh); на площадке — физ. ядра/квота cgroup.
