# w0: ускорение узких мест метрики/гейта на боксе (29.10.2026)

Профиль honest-10 на реальных данных (ae D-FINE OOF ≥0.2, 173 038 объектов; c3-sc OOF 85 647):
время уходит в чтение JSON и `_collect` (конструктор shapely.Polygon на объект + циклический GC);
`sweep_groups`/`honest_counts` дешёвые (0.05/0.8 с) — их не трогаем.

| узкое место | было | стало | как | флаг выключения |
|---|---|---|---|---|
| фильтр 3.27 ГБ OOF по conf≥0.2 (`cpu/filter_oof.py`, ijson yajl2_c) | 60.5 с | 7.8 с (×7.8) | Rust `expds_core.filter_geojson_conf` (потоковый сканер + serde_json + rayon) → `python -m expds.fastjson src dst thr` | `EXPDS_FASTJSON=0`, `EXPDS_BACKEND=numpy` |
| `build_curves` 173k (холодный кэш) | 10.9–11.2 с | 3.0 с (×3.7) | порт bot8 `flatten_polygons` (Rust) + `from_ragged_array`, вектор. `max_abs_x`, UTM в потоках; пауза GC | `EXPDS_FASTGEOM=0`, `EXPDS_METRIC_NOGC=0` |
| `build_curves` 85k | 4.1 с | 1.2–1.3 с (×3.3) | то же | то же |
| чтение pred+GT (310 МБ) | 5.4–7.4 с | 3.5 с | orjson, если установлен | `EXPDS_FASTJSON=0` |
| весь путь load→curves→honest-10, 173k | 17.5–19.2 с | 7.3 с (×2.5) | | |

Проверка равенства:
* фильтр: 173 038 / 1 920 482 — те же объекты, поэлементно `==` после разбора (выход не побайтный:
  Rust пишет исходный текст объектов, Python — переформатированный json.dump);
* геометрии `_collect`: WKB побайтно равны (173k); ClassCurve (conf, hit, регионы, группы, n_gt) равны;
  пороги sweep и honest-счётчики идентичны (repr совпадает) на обоих наборах;
* tests/test_w0_fast.py (5 тестов; `EXPDS_CORE_DEV=<каталог со свежим expds_core>` — conftest ставит inference/lib первым).

Не тронуто: `inference/lib/expds_core.abi3.so` (бинарь архива; новая сборка — rustc 1.99, manylinux_2_35,
совместимость с glibc контейнера не проверена). Для бокса: `maturin build --release` в rust/expds_core и
`pip install --target <dir>`; без модуля всё работает по-старому.
Сборка: системный rustc 1.85 не собирает delaunator 1.1/uuid 1.28 → rustup stable в ~/.cargo/bin.
