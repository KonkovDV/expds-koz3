
## 29.10.2026 12:50 МСК — задачи bot-1 11:09: C (EtGr по слоям), A (табло v2), B (C2/SATf) (бот 10, REPORT13_*)
- C: флаг `EXPDS_ETGR_LAYER` в рабочей копии `repo/expds/metric.py` (выкл. = байт-в-байт; дифф `work13/etgr_layer.diff`). При вкл. S/Ss6 теряют EtGr-выигрыш (etgr_pix TP 235→99): Ss6−A keep −0,011457…+0,003426, drop −0,011524 → П1/П2 не проходят. FR1h/D-FINE/ORGATE не меняются. Офиц. метрика слои пулит → вердикты оставить, риск уточнить у организаторов (через bot 1). `REPORT13_etgr_layers.md`.
- A: табло v2 выше OOF (0,412 vs 0,354); ae R 0,013 = ae-курганы только копии li→ae, в регионах без li 0 предсказаний (OOF 510/510 MISS). Остальные типы на табло лучше OOF. `REPORT13_lbv2.md`.
- B: C2 REJECT (к S keep −0,002733, P 0,118; s6 −0,027693; время ×2,1). SATf REJECT по PREREG bot-3. `REPORT13_C2.md`.
