# koz3-w2-gate — всё для гейта honest-10, ~5,4 ГБ, 54 файла

## w5base/ — база A (от W5), с ней сравнивается любой вариант
| Файл | Что |
|---|---|
| `stack_arch.geojson` | стек A (все группы детекций, 1,2 млн строк; sha .gz начинается 42ff3087) → A=0,314615 |
| `gt_train_v2.geojson` | эталон train v2 |
| `base_counts.pkl` | счётчики TP/FP/FN базы по семьям → аргумент `--vs` у `gate_v6c.py` |
| `table.pkl` | таблица признаков W5 `prep.py` для базы |

## gate_s2/ — замеренные варианты W2
| Каталог | Вариант | Итог |
|---|---|---|
| `s2/` (`stack_arch_cnx_s2.geojson`, work/*.pkl, `w6c_gate_s2.json`) | A + c3-sc s2 аддитивно (P-W2-2) | 0,29840, Δ−0,0162, P=0,000, FAIL |
| `s012/` | A + s012avg | Δ−0,0071, P=0,106, FAIL |
| `varb/` | li-группы заменены на s012avg (P-W2-6) | 0,29513, Δ−0,0195, P=0,12, FAIL |
| `varb_kur/` | замена только курганов | Δ−0,0112, FAIL |
`work/`: `table.pkl`, `geom_wkb.pkl`, `gt_wkb.pkl` (выход prep.py), `counts_<tag>.pkl` (счётчики варианта), `w6c_gate_<tag>.json` (итог: score, d, boot, per_class, d_new34). `detail_<tag>.json` — разбор по классам/типам данных (`gate_detail.py`). `logs/` — все логи, `base_repro.log` — воспроизведение A. `check_s2only.geojson` — самопроверка. `run_varb.sh` — скрипт.

## Воспроизвести гейт (пример: A + s2)
```
K=/workspace/koz3; PK=$K/orch/pool_kaggle.sh
$PK elenakonkova datasets download -d elenakonkova/koz3-w2-gate -p $K/w2/restore/gate --unzip
$PK elenakonkova datasets download -d elenakonkova/koz3-w2-oof  -p $K/w2/restore/oof  --unzip
# пайплайн W5: $K/wt/w5/analysis/2026-10-30/w5_rerank (prep.py, gate_v6c.py); W5_DATA ожидает stack_arch*.geojson и gt_train_v2.geojson
D=$K/w2/restore/run_s2; mkdir -p $D; cp $K/w2/restore/gate/w5base/gt_train_v2.geojson $D/
python w2_tools/stack_add_cnx.py $K/w2/restore/gate/w5base/stack_arch.geojson $K/w2/restore/oof/merged/oof_cnx_c3-sc_s2.geojson $D/stack_arch_s2.geojson cnx_s2
cd $K/wt/w5/analysis/2026-10-30/w5_rerank
W5_DATA=$D W5_STACK=stack_arch_s2.geojson nice -n 10 python prep.py
W5_DATA=$D nice -n 10 python gate_v6c.py --vs $K/w2/restore/gate/w5base/base_counts.pkl --save counts_s2.pkl --out w6c_gate_s2.json
W5_DATA=$D PYTHONPATH=. python $K/wt/w2/w2_tools/gate_detail.py $K/w2/restore/gate/w5base/base_counts.pkl $D/work/counts_s2.pkl $D/detail_s2.json
```
Ожидается `score≈0,29840, d≈−0,0162` (как в `gate_s2/s2/work/w6c_gate_s2.json`). Для базы: тот же prep.py+gate_v6c.py на `stack_arch.geojson` → 0,314615. Для замены li-слоя — `stack_replace_li.py` (см. `run_varb.sh`). ~10–20 мин CPU и ~2 ГБ на вариант; ≤3 тяжёлых процесса, 8 ГБ ОЗУ.
