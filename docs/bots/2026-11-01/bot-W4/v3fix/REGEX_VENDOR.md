# W4 01.11 — v3_on: изоляция вендорного regex (D-FINE на любом regex окружения)

**Причина.** Вендор архива ebc8e4ba (`inference/lib/vendor`) нёс regex 2026.9.29 только с `_regex.cpython-312-…so`.
На Kaggle (python 3.13) `regex/_regex_core.py` → `from regex import _regex` падает (circular/partially initialized),
transformers импортирует regex при загрузке (`utils/auto_docstring.py`, dependency_versions_check) → вендорный путь
не грузится, фолбэк «env» отвергает transformers 5.16.1 (пин 5.21.0) → `SP_DFINE_UNAVAILABLE`, выдача = v2 (−0.02).

**Выбор: вендорить regex под все ABI** (cp311–cp314, одна версия 2026.9.29, .py одинаковые во всех колёсах, .so cp312
побайтно = прежнему) + в `sp_dfine.import_dfine` перед вендорной попыткой выгружать ВСЕ чужие (не из вендора) модули
пакетов вендора и `_regex`, предпроверка наличия `_regex{EXTENSION_SUFFIX}`. Почему не заглушка regex: transformers
использует regex на импорте (auto_docstring) — заглушка = правка чужого пакета и риск в D-FINE-процессоре; почему не
только purge: без .so под ABI вендор не грузится в принципе. Образ площадки (python 3.12) берёт тот же cp312 .so → поведение
на площадке не меняется. Из EN ничего не вендорится; колёса — PyPI:

| колесо | sha256 |
|---|---|
| regex-2026.9.29-cp311-…manylinux_2_28_x86_64.whl | 612b709381c0355b70d89cdb51b7f670591ed5cbbc0e3b5337488019dc667b65 |
| regex-2026.9.29-cp312-… | 39ab5894d971f9ac68baa6eca5c50387db579cfcacf36ae8df3feceb1815e6d0 |
| regex-2026.9.29-cp313-… | d49c18f1ea294cf4adde2e5ac256e98c82ea9d708462ce4bf799dffa7cfe8a2c |
| regex-2026.9.29-cp314-… | 2f7f7aa47b229f2b39a2ae2596d2ad5625d77b5eb9856fac2dab3eb506cdd0a0 |

.so в архиве: cp311 d75b44fa…, cp312 b625e4e7…, cp313 b9289d43…, cp314 4d13ce87….

**Громкий отказ.** `expds/degraded.py`: любая ветка v3 (sp_dfine: импорт/веса/пороги/ошибка участка; gor_gate: недоступен/нет
артефактов; watchdog: frame_prior в запасной цепочке) → WARNING `!!! V3_DEGRADED: … !!!` в лог и stderr + `V3_DEGRADED.json`
рядом с ответом. `check_submission` (раздел v3): строка V3_DEGRADED в логе или маркер → FAIL.

**Локально (py3.13.5, w1 venv), импорт из нового архива:** default / regex 2025.11.3 в PYTHONPATH / regex 2025.11.3
предзагружен в sys.modules / regex окружения заблокирован — во всех 4 случаях transformers 5.21.0 и regex 2026.9.29 из вендора.
Контроль: старый архив в тех же условиях берёт regex/transformers окружения.

**Официальное окружение:** docker/Dockerfile `FROM python:3.12-slim-bookworm`, torch 2.11.0; requirements.lock — без
regex/transformers/tokenizers (D-FINE на площадке только из вендора; .so cp312 есть). Организаторы (consult2): любой
публичный образ с Python ≥ 3.12.

## Вторая причина (Kaggle kernel v4, 22:23 MSK): huggingface_hub окружения

После фикса regex (архив 10ee67b, e40b2e8b…) на Kaggle C/A/B всё ещё `SP_DFINE_UNAVAILABLE`: `vendor: ImportError: cannot
import name 'httpx' from 'huggingface_hub.utils'` — huggingface_hub окружения Kaggle **1.29.0**, а transformers 5.21.0
требует `huggingface-hub>=1.31.0,<3.0`. Маркер/FAIL сработали как надо (выдача = v2, не тихо).

**Фикс b447649:** `inference/lib/vendor/_hf` = huggingface_hub **2.2.0** (ровно как requirements.lock образа) + httpx2 2.16.0,
httpcore2 2.16.0, h11 0.16.0, anyio 4.16.0, idna 3.20, truststore 0.10.5, typing_extensions 4.16.0 (чистый python, PyPI).
Подключается ТОЛЬКО если версия huggingface_hub окружения (по метаданным, без импорта) вне `>=1.31,<3` или его нет; тогда
`_hf` ставится вторым в sys.path (после вендора), загруженные huggingface_hub/httpx2/httpcore2/truststore не из `_hf`
выгружаются, WARNING в лог; `info["hf"]`. На площадке (2.2.0) и в venv с 2.4.0 — не используется, поведение прежнее.
Локально: hf 1.29.0 в PYTHONPATH (и предзагружен) → transformers 5.21.0 + hf 2.2.0 из `_hf`; без подмены → hf окружения.

| колесо | sha256 |
|---|---|
| huggingface_hub-2.2.0-py3-none-any.whl | 1667f145dc56dc210d60966069397df9ecfca9607a5d43db88b308c89dae56b3 |
| httpx2-2.16.0-py3-none-any.whl | fd2c5d7909e8d86fe92d53020c310ca0e4e431b838c207c5754b9bbf0308d49a |
| httpcore2-2.16.0-py3-none-any.whl | 5c4358319f2602d984ad93182c9fee5160f22e9da7babdda870a5e7c4e2dc34f |
| h11-0.16.0-py3-none-any.whl | 63cf8bbe7522de3bf65932fda1d9c2772064ffb3dae62d55932da54b31cb6c86 |
| anyio-4.16.0-py3-none-any.whl | c66ae634e0b8228396b801741beb3294326d4b6102f7bba4041c0dfb58c29026 |
| idna-3.20-py3-none-any.whl | ab7ae7122974553370f0bdb919e1a960b2cd1bc1ef0276416d896db81c14582c |
| truststore-0.10.5-py3-none-any.whl | 9aaaedaefaf06d8b206278cf8b5012bc897f485a874503501e12d776df78951c |
| typing_extensions-4.16.0-py3-none-any.whl | 481caa481374e813c1b176ada14e97f1f67a4539ce9cfeb3f350d78d6370c2e8 |

Архив: `koz3_v3on_b447649_6d4764f8_w6dfd2e07.zip` sha256 **1406c01a9aaafc996862a14f12cde14f6bf05b313ad0bbce14428df1d10ef41d**,
3902 файла, 689,2 МиБ, DOC-1 OK. Против ebc8e4ba: +396 (389 `_hf`, 3 regex .so, cloudfeat/degraded/w5_cloudfeat/w5cf_infer),
изменены gor_gate/pipeline/sp_dfine/watchdog, удалённых нет. Промежуточный zip 10ee67b удалён (манифест оставлен).
