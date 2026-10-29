# models/pixgbm/all — веса не в git

Оригинальный кандидат v2 — коммит `eb268d6` (bot-7, ветка bot7-candidate-v2 = bot7-candidate-v3). В этой ветке (`bots/bot7-candidate-v2-nomodels`) его дерево без двух файлов моделей LightGBM PIXGBM (обучены на конкурсных данных; по правилу выгрузки 29.10 веса в git не добавляются):

| файл | sha256 |
|---|---|
| poly.txt | d1fe3efafcd4377d41430d8828e881576c9024f9d7a5c89f55e23554e02ec800 |
| finds.txt | 5fa8c09d77d0914cc925fb6bb925a0ab1b64ccecf3c823d7f3546e71c4ceb04a |

Источник моделей: `/workspace/koz3_shared/bot-3/pixgbm/models/all/`. Собранный архив сдачи: `submission_bot7cand_v2_eb268d6.zip`, sha256 8f3f2671…d589.
