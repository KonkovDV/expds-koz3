# W4 01.11 23:42 MSK — PREREG: паритет v2clean 545ef32a после b447649 (до запуска)

Архив `koz3_v2clean_70d335a_c028f4eb_wb042594f.zip` sha256 545ef32a… НЕ пересобирается. b447649 (и 9f7877b) его не
затрагивают: архив собран из 70d335a, вендора `inference/lib/vendor` в нём нет, `expds/degraded.py` нет,
`[sp_dfine]`/`[gor_gate]` enabled = false — правки regex/huggingface_hub-фолбэков в sp_dfine.py до него не доходят.
Меняется только проверяющий: новый check_submission.py (раздел v3: маркер V3_DEGRADED/строки в логе → FAIL).

Kernel aaaa0000004/koz3-w4-v2cleanchk2 (новая версия), окружения по порядку: REF_check4 (точный повтор check4: Kaggle
по умолчанию + старый check_submission из koz3-w4-v2clean-check, --keep), C_default, A_regex2025_preloaded (.pth import
regex), B_no_regex (pip uninstall regex), D_new_env521 (стек transformers 5.21 из офлайн-колёс) — новый check_submission.
В каждом: смоук check_submission (sha result.geojson) + прямой main.py на участке 103.

Критерии PASS: во всех 5 rc=0 / ИТОГ PASS; маркера V3_DEGRADED.json нет; sha смоука и участка 103 во всех = REF_check4;
смоук REF_check4 совпадает с check4 (out_v2clean_check4: 80 Polygon, 58184 байт, {finds_points 72, kurgany_tselye 3,
kurgany_povrezhdennye 3, arkhitektury 2}) — в check4 sha выдачи не сохранялся, поэтому побайтовый эталон = REF_check4
в том же образе и тем же проверяющим, связь с check4 — по размеру и составу.
