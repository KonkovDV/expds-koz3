# STATUS bot-4 (28.10.2026, MSK)
- [x] COMMON.md прочитан; клон bot-4/repo (24bd916), venv bot-4/venv
- [x] база воспроизведена (work/res_base.json): честно 0,05041 (gor 0,2762, fort 0,1796)
- [x] путаница и потери: work/res_confusion.json, res_pool.json, res_ceiling.json
- [x] арбитр fort2gor + грубый проход по стеку: expds/gorfort.py, замеры work/res_gorfort_nms_s0.4.json — честный Δ<0, не включать
- [x] грубый проход по ЦМР: hillfort.COARSE / --preset coarse + тесты; замер только на ПК (нет LAS)
- [x] ветка bot4-gorfort (ab8ad26), diff bot-4/bot4-gorfort.diff, REPORT.md
- [ ] сообщить bot «1» путь к REPORT.md — у меня нет инструмента сообщений, передано родителю

## 28.10 20:54 МСК — GAP (срочно от бота 1)
- [x] GAP.md: класс×тип, FN/FP-потери, без лидара, Geo, IoU/размер, потолки (1а/1б/2/3) — `work/gap/gap.py`, `work/gap/res_gap.json`
- [ ] Перепроверка/бутстреп потолков (по желанию бота 1)
- [ ] DEM грубый проход на 21 LAZ-участке — на паузе, не запускался
- [ ] Уведомить бота «1» (через родителя)

## 28.10 20:58 МСК — DEM грубый проход ПРИОСТАНОВЛЕН (по команде бота 1)
- Готово 25 участков с LAS: `work/dem/out/*.geojson` (кандидаты gor_coarse_dem), лог `work/dem/run.log`. Не обработаны (крупные LAZ): 004, 007, 028 и всё, что докачается позже.
- Возобновление: `nohup work/dem/run.sh >> work/dem/run.log &` (пропускает готовые), затем `venv/bin/python work/dem/dem_eval.py` (оценка написана, не запускалась).
- Процессов DEM не осталось.
