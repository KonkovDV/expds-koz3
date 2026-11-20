# inference/cnx — подготовка входа ConvNeXt-лидара

`expds/lidar_cnx.py` при `[li_cnx] enabled = true` запускает отдельный процесс

    python inference/cnx/cnx_prep.py LAS --ch 5 --crs <CRS> --vendor inference/cnx/vendor --out <tmp>.npz

с `PYTHONPATH=inference/cnx/_deps`. Скрипт строит из облака LAS каналы рельефа на сетке 0,5 м
(локальный рельеф двух масштабов, уклон, фактор открытости неба и положительная открытость) и пишет
их во временный `npz` в `EXPDS_TMP`; файл удаляется сразу после чтения.

| путь | что это |
|---|---|
| `cnx_prep.py` | LAS → каналы рельефа u8 [ch, H, W] и привязка |
| `vendor/expds/*.py`, `vendor/expds_core.abi3.so`, `vendor/relief_ext.py` | замороженная копия кода подготовки, на которой обучались веса ConvNeXt; `lidar_cnx` сверяет sha256 этих файлов с полем `prep_code_files` контракта весов и при расхождении не включает ветку |
| `_deps/rvt/*.py`, `_deps/rvt/LICENSE` | Relief Visualization Toolbox (rvt-py 2.2.1, Apache-2.0) — фактор открытости неба и открытость |
| `wheels/rvt_py-2.2.1-py3-none-any.whl` | исходное колесо rvt-py, из которого распакован `_deps/rvt` |

`MANIFEST.json` — sha256 каждого файла каталога (проверяется тестом `tests/test_bot8_cnx.py`).
Вендорный пакет `expds` — отдельная копия: в процесс пайплайна он не импортируется.
