# koz3-w2-oof — OOF-детекции W2 (c3-sc s0/s1/s2, li_hm), ~1 ГБ, 92 файла

**Что это:** out-of-fold предсказания лидарной модели c3-sc (каждый участок предсказан моделью, не видевшей его фолд). Это то, что подмешивается в стек и меряется гейтом.

## merged/ — склеенные 5/5 фолдов (главное)
| Файл | Детекций | Что | PREREG / итог гейта |
|---|---|---|---|
| `oof_cnx_c3-sc_s0.geojson` | 85 647 | сид 0, фолды 0–4 | P-W2-2 |
| `oof_cnx_c3-sc_s1.geojson` | 83 519 | сид 1 | P-W2-3 (контроль) |
| `oof_cnx_c3-sc_s2.geojson` | 94 959 | сид 2 | P-W2-2: аддитивно FAIL Δ−0,0162, P=0,000 |
| `oof_cnx_c3-sc_s01avg.geojson` | 81 665 | усреднение s0+s1 (`seed_merge_cnx.py`), отдан W5 | V6c W5: Δ−0,0103 |
| `oof_cnx_c3-sc_s012avg.geojson` (+`.json` сводка) | 82 996 | усреднение s0+s1+s2 | P-W2-2 аддитивно Δ−0,0071 P=0,106; P-W2-6 (замена li-слоя) Δ−0,0195 — FAIL |
| `merge_summary.json` | — | детекции по фолдам, complete=true, errors=[] | — |
`threshold_group="li_cnx"` (в варианте b переименовывается в `cnx012`).

## s0/, s1/, s2/, lihm/ — по ядрам `<seed>/<kernel-slug>/`
- `oof_cnx_c3-sc_f{K}.geojson` — OOF фолда K; `oof_cnx_c3-sc.geojson` — всё, что посчитало ядро.
- `gt_li_cnx_c3-sc_f{K}.geojson` — эталон Li для фолда (для Li-only оценки).
- `region_fold.json` — участок → фолд. `report_cnx_full.json` — конфиг (`cfg`) + Li-only метрики по фолдам. `*.log` — лог ядра.
- Ядра: s0 = dmitrijkv/koz3-cnx-li-c3sc-{f01,f23,f4-v2}; s1 = dmitrijkv/koz3-cnx-li-c3sc-s1f01, elenakonkova/koz3-cnx-li-c3sc-{s1f23,s1f4}; s2 = dmitrijkv/koz3-w2-cnx-c3sc-{s2f01,s2f23}, elenakonkova/koz3-w2-cnx-c3sc-s2f4; lihm = elenakonkova/koz3-w2-lihm-c3sc-f01234 (P-W2-7, heatmap-голова, ранний стоп FAIL: f0 0,2099 / f1 0,4175).
## seedmerge/ — `oof_cnx_c3-sc_s01avg.geojson` (+сводка), `check_s0.geojson` (самопроверка усреднения на одном сиде), `run.log`.

## Как использовать
- Склеить фолды заново: `w2_tools/merge_cnx.py` (проверяет threshold_group, region_fold, 0 детекций вне фолда, EPSG:3857).
- Усреднить сиды: `w2_tools/seed_merge_cnx.py`.
- Добавить в стек A аддитивно: `w2_tools/stack_add_cnx.py`; заменить li-группы: `w2_tools/stack_replace_li.py stack_arch.geojson <oof> <out> <group> [--classes ...]`. Затем гейт (см. koz3-w2-gate).
- Куда идёт: в стек W5 (`stack_arch.geojson`) → `prep.py` → `gate_v6c.py`.
