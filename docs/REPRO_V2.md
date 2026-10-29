# REPRO_V2 — сборка из main ≡ официальный v2 (eb268d6, LB 0.41238)

Причина разрыва (≈−0.012): в main нет 4 release-артефактов v2 (удалены интеграцией 29.10 09:18–09:29 MSK по правилам
«EN-derived hold» и «веса не в git»): `expds/geo_pixgbm.py`, `models/pixgbm/all/{poly,finds}.txt`,
`models/rerank_arch_lgbm.txt`. Без модуля pipeline молча уходит в классическую геофизику
(«pixgbm включён, но expds/geo_pixgbm.py нет в этой сборке — классика»); без весов build_submission падает.

Процедура (build-time injection, ничего не коммитится):

    python scripts/v2_release_artifacts.py        # достаёт из git-объектов eb268d6, сверяет sha256
    python scripts/v2_release_artifacts.py --check  # только проверка (rc=0 ⇔ все 4 на месте и sha совпадают)
    python scripts/build_submission.py
    python scripts/golden_compare.py RUN_A RUN_B  # постадийные счётчики, первая расходящаяся стадия

Проверка (box, CPU, threads=1): main(70dedf4)+артефакты даёт byte-identical result.geojson с пересборкой eb268d6
на synth, 210, 214, 217, 044, 181. Расхождение «main без артефактов» начинается на фазе geophysics (pixgbm).
Таблица стадий — в docs/STATUS.md ветки bots/w4-gate. Остаточное: 210 written 380 (box) vs 383 (лог бота-7) —
одинаково для обеих сборок, т.е. окружение/копия входа, не код.
Sha zip не сравним 1:1 (z.write с mtime) — сравнение по CRC/размерам членов.
