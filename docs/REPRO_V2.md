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

## 210: 380 vs 383 objects — attributed to the environment (bot 1's decision, 29.10 22:55 MSK)

* Original (bot-7 log `check_eb268d6_210_run.log`): written 383, before threshold fp1399 / op143.
* Box (py3.12.15, torch 2.11.0+cpu, threads=1): written 380, before threshold fp1400 / op142 — the same for both builds
  (eb268d6 rebuild and main 70dedf4 + artifacts); geophysics +1542 and object_poly after threshold 10 in all variants.
* Reasoning: the code is identical (members/CRC as in the v2 archive; result.geojson byte-identical between the two builds on all 6 inputs);
  synth/214/217 match the original exactly → the 3-object difference comes from the environment (Python 3.12.15 vs 3.12.16 / libraries)
  or the S3 copy of the 210 input (bot-7 used its own copy), not the code. Bot-5 got 383 on its environment.

## Status of the v2 line

v2 (eb268d6) is only a comparison reference (golden for golden_compare). The final archives must contain no EN-derived files:
`v2_release_artifacts.py` is for reference runs only, not for building a submission.
The geo_pixgbm.py copies under analysis/ in main are not opened or changed by W4.

*(Superseded on 04.11: the owner allowed the EN line with a declaration, `docs/PIXGBM_DECISION.md`. The final is v2.)*

## Пересборка архива v2 для финала (04.11.2026)

Оригинал `submission_bot7cand_v2_eb268d6.zip` (sha256 `8f3f2671…d589`) был только на ПК Елены; на Drive и на коробке его нет.
Все веса v2 лежат в git коммита `eb268d6` (обычные блобы, без LFS), поэтому архив пересобирается из самого коммита
тем же скриптом, без `v2_release_artifacts.py`:

    git archive eb268d6 | tar -x -C <dir>
    cd <dir> && python scripts/build_submission.py --out submission_v2_eb268d6_rebuild.zip

Результат: `submission_v2_eb268d6_rebuild.zip`, 525 158 901 байт, 50 файлов, sha256 `167478e3f5534bc553b61eef4244995553cba0da0542c9fe80c5dd4636f96fdd`.
Drive «2» → `KOZ3_backup_2026-11-04/00_SUBMIT_FINAL_v2_EN_167478e3` (рядом README, SHA256SUMS, лог проверки; md5 после загрузки совпал).

Почему это тот же архив, а не похожий:

| Проверка | Результат |
|---|---|
| Эталон: FR1h (`submission_fr1h_on_eb268d6.zip`, sha256 `fe8d60b9…`, Drive `fr1h.zip.bin`) — по записи 31.10 это оригинал v2 + 4 изменённых файла, остальные 46 побайтно равны оригиналу (`snapshot/2026-11-01/fr1h-pack/FR1H_PACK_20261031.md`) | sha256 скачанного файла = `fe8d60b9…` |
| Состав | 50 = 50 файлов, имена совпадают |
| 46 неизменённых членов | все 46 совпадают по размеру и CRC32 |
| 4 изменённых члена | различаются ровно `configs/default.toml`, `expds/pipeline.py`, `expds/post.py`, `expds/watchdog.py` |
| Размер правки FR1h от оригинала по аудиту (`orch/plans/AUDIT_HYPERDEEP_20261101.md`): toml +12/−4, pipeline +19, post +60, watchdog +10 | у пересборки против FR1h: +12/−4, +19/−0, +60/−0, +10/−0; 4 снятые строки toml = пороги архива 0.3645 / 0.3725 / 0.501645 / 0.645505 |
| Размер оригинала по записи 30.10 — «501M» | 525 158 901 Б = 500,8 МиБ |
| `scripts/check_submission.py` (main, `--cpus 0-3`) | PASS — FAIL 0, WARN 0, 29 проверок; смоук без сети 18,9 с; модели PIXGBM в архиве |

sha256 zip не совпадает с `8f3f2671…`: zip пишет время файлов (см. выше). Члены те же, значит и выход тот же.
Ожидание на площадке — 0,41238 (тест тот же: лидар и аэрофото совпадали до 3 знаков во всех четырёх посылках).
