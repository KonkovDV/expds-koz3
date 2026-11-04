# W0: профиль CPU постобработки и подготовки LiDAR (29.10.2026, бокс 8 ядер, высокая загрузка)

## Постобработка (watchdog.finish → writer)
Шейпли 2: векторные вызовы GEOS вместо поэлементного Python. Узкое место — накладные расходы Python, а не GEOS,
поэтому Rust-порт (geo/geos crates) не даёт выигрыша сверх векторизации. Фолбэк: `EXPDS_POST_FAST=0`.

| Набор | finish старый | finish новый | выход |
|---|---|---|---|
| li c3sc + dfine_or + ae03 (125k) | 27.8 с | 5.9 с | байт-в-байт |
| w2 c3sc_s0 + ae03, load avg 36 | 40.7 с | 23.4 с | байт-в-байт |

- finalize 15.8 → 0.5 с; micro_circles 3.2(4.9) → 0.7 с; merge_within_markup: пакетный STRtree.query + intersection массивом.
- Остаток: rerank_arch ≈5 с счёта + импорт lightgbm (в контейнере ~1–2 с, на загруженном боксе до 12 с), twins ~1 с — GEOS/C.
- Тесты: `tests/test_w0_post_fast.py` (equals_exact, tol 0) для finalize / micro_circles / merge_within_markup.

## LiDAR (expds_core: pyo3 abi3 + rayon, вход numpy zero-copy через Cow)
LAS 234 MB (003_Лубно, 8739×5063 @0,1 м, 4 потока): build_dem 7.7 с (fill_tin 3.9 с — последовательная триангуляция
Делоне, las_grid 1.3 с), hillshade_set 3.6 с, relief_channels 3.8 с.
- Сняты лишние копии `astype(np.float32)` после `np.where` (`copy=False`) — выход бит-в-бит (sha1 синтетики совпадает).
- Следующий кандидат: fill_tin (Делоне) — распараллеливание только при сохранении бит-точности; сейчас не трогаем.

## Раунд 2 (2026-10-30): rustc 1.99.1, Rust-матчинг, fill_tin
- Toolchain: rustup stable 1.99.1 (`$HOME/.cargo/bin`), сборка `scripts/build_rust.py linux` (avx2). Тесты на новом и старом .so: 48 passed / 3 skipped — одинаково.
- Матчинг гейтов: `EXPDS_RUST_MATCH=1` (по умолчанию выкл.). Rust (rstar+geo+rayon, `match_prefilter`) — только консервативный префильтр
  (cov ≥ thr−1e-4, dist ≤ r+1e-3, catch_unwind→пара остаётся); на выживших — официальный `_match_quality` ⇒ точность по построению.
  Наборы < 2000 кандидатов (`EXPDS_RUST_MATCH_MIN_PAIRS`) идут старым путём. Потоки: `EXPDS_THREADS`.
- Точность (hits/pairs sha py==rs): stack_arch 5748 hits c7934f9e56f6/dcb2e7206cf1; OOF f2 307 5b16f77c0375; OOF f4 401 2910d75bd177.
- Скорость: stack_arch (1.39M предсказаний) 3.6 с → 2.2 с (×1.6, 2 потока); OOF — без изменений (<0.1 с).
  Вывод: матчинг занимает секунды — 19-мин фолд тормозит НЕ здесь (смотреть обучение реранкера / nested-свипы / _collect JSON).
- fill_tin: уже в Rust (vendored delaunator + бит-точная параллельная растеризация). Последовательна только триангуляция;
  другие Delaunay (spade и т.п.) по-другому режут коциркулярные точки решётки ⇒ DEM не бит-в-бит. Пропущено.
