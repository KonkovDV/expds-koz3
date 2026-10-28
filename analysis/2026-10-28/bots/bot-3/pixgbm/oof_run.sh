#!/bin/bash
# OOF для bot-6: сначала Гнёздово 214–217 etgr/terfl, затем 212–213, затем maggr 200–212.
cd /workspace/koz3_shared/bot-3/pixgbm
O=/workspace/koz3_shared/bot-3/pixgbm/oof; R=/workspace/data/train-expds-3; Mo=/workspace/koz3_shared/bot-3/pixgbm/models
bash run.sh oof_gn oof --root $R --models $Mo --out $O --markups etgr,terfl --regions 214_Гнёздово_Лесная_2 215_Гнёздово_Селище_1 216_Гнёздово_Селище_2 217_Гнёздово_Селище_3 212_Гнёздово_Усадьба 213_Гнёздово_Лесная_1
# maggr: свои маркеры READY_ ставятся после всех методов участка, поэтому отдельная папка шагом не нужна —
# READY_<участок> для 212 перезапишется после maggr (файл 212_..._etgr/terfl уже готов раньше).
bash run.sh oof_mag oof --root $R --models $Mo --out $O --markups maggr
