# inference/cnx — подготовка входа ConvNeXt-лидара bot-9 (bot-8, ветка bot8-cnx)

Вызывается из `expds/lidar_cnx.py` только при `[li_cnx] enabled = true`: отдельный процесс
`python inference/cnx/cnx_prep.py LAS --ch 3|5 --crs <UTM.json> --vendor inference/cnx/vendor --out <tmp>.npz`,
`PYTHONPATH=inference/cnx/_deps` (rvt). Временный npz лежит в `EXPDS_TMP` и удаляется сразу.

| путь | откуда | зачем |
|---|---|---|
| `cnx_prep.py` | bot-9/for_bot8/cnx_prep.py (sha256 77d30749…) + 05.11 ветка `ch = 12` (TAGS12, те же вызовы вендора, что `kaggle/2026-11-05/li_c12_in/prep12.py`); ch 3/5 без изменений | LAS → u8 [ch,H,W] + geo |
| `vendor/expds/{__init__,accel,candidates,crs,lidar}.py`, `vendor/expds_core.abi3.so`, `vendor/relief_ext.py` | bot-9/kaggle/convnext_lidar/src/vendor, побайтно (sha = поле `prep_code_files` в весах = bot-9 vendor_manifest.json; `lidar_cnx.load` сверяет и при расхождении откатывается на li) | код подготовки, на котором учились веса (prep_code_sha 6ed68955…) |
| `_deps/rvt/*.py`, `_deps/rvt/LICENSE` | распакованный `wheels/rvt_py-2.2.1-py3-none-any.whl` (Apache-2.0) | svf_r / opp_r для cnx5 (`rvt.vis`) |
| `wheels/rvt_py-2.2.1-py3-none-any.whl` | bot-9/for_bot8/wheels (sha256 f3b171e0…) | исходник _deps, `pip install --no-deps` при сборке образа |

rvt-py 2.2.1 в образе (requirements.lock, docker/Dockerfile) **нет** — поэтому он вендорен как чистый Python
(зависимости rvt — numpy, scipy; в образе есть). `MANIFEST.json` — sha256 всех файлов каталога;
check_submission.py сверяет его с архивом. Вендорный `expds` — другой пакет с тем же именем, поэтому
в процесс пайплайна не импортируется (cnx_prep сам падает, если expds уже импортирован не из vendor).

`WEIGHTS_MANIFEST.json` (не входит в `MANIFEST.json`) — список весов [li_cnx] с sha256/size и `never_pack`
(smoke-веса). Веса в git не лежат (`models/cnx/` в .gitignore): `scripts/build_submission*.py` берут их по
этому списку и отказывают при несовпадении sha/size или при попадании never_pack; check_submission сверяет sha
внутри zip. Пороги — своя группа `[thresholds_by_markup.<threshold_group>]`. Имя по умолчанию — `li_cnx`.
Группы чужой шкалы (`li`, `li_coarse`, `li_mosaic` и остальные из `FORBIDDEN_GROUPS`) запрещены: без своей
таблицы cnx молчит. Архив 0,43274 задаёт группу `li_cnxin`, не имя по умолчанию.
