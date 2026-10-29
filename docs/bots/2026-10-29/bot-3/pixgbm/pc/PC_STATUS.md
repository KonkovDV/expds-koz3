# PIXGBM на ПК — статус от bot-9 (28.10, 23:10 МСК)
1. **Пороги *_pix на всём трейне** — ГОТОВО (на коробке, ПК был занят; тот же M.sweep_groups, что в eval_pix.kept_mask, но curve_where по ВСЕМ участкам; вариант S, вход pix_lofo.geojson; global и site совпали):
   ```
   [thresholds_by_markup.maggr_pix]
   finds_points = 0.902915
   object_poly = 0.684725
   [thresholds_by_markup.etgr_pix]
   object_poly = 0.545825
   [thresholds_by_markup.terfl_pix]
   object_poly = 0.746421
   ```
   Скрипт pc/fit_thr.py, лог pc/fit_thr.log, json pc/thr_pix_full.json. Оговорка: пороги подобраны на LOFO-кандидатах (модели без своей семьи), а в сдаче — модель all (на трейне выше уверенность) → пороги, вероятно, чуть занижены для all.
2. **Модели** — models/all/{poly,finds}.txt скопированы в C:\Sloi\wt_pix\models\pixgbm\all (ветка bot3-pixgbm HEAD 7cdace5, включает union_name_re; по умолчанию выкл.).
3. **Тест-кандидаты** — тестовых данных на ПК и коробке нет (тест только в ядре Kaggle, его не трогаю). Вместо этого в очереди ПК (после lidar59, F: один читатель): geo_predict variant S (enabled, replace, model_dir all, пороги выше) по всему трейну F:\2\train-expds-3 → C:\Sloi\pix\geo_pix_S_train.geojson — проверка хука, время и пик RAM на всех гео-участках (на трейне это утечка, для скоринга НЕ использовать). Результаты допишу сюда и в bot-9/PC_RESULTS.md.

## Бенчмарк на ПК (23:34 МСК) — подробности в bot-9/PC_RESULTS.md
i5-13600KF; те же 7 участков/43 растра: nt=2 **233,7 с** (коробка 274,5 → ×1,17), nt=8 **126,4 с** (коробка nt1 420,4 → ×3,33). feat однопоточный (54,6 с при любом nt). Пик RSS 792 МБ (200). Файлы C:\Sloi\bench\bench.jsonl.

## 28.10 23:43 МСК — PIX_S на трейне (ПК, bot-9)
- `geo_predict.py --root F:\2\train-expds-3 --set geophys.pixgbm.enabled=true --set geophys.pixgbm.replace=true --set model_dir=C:/Sloi/wt_pix/models/pixgbm/all` + 4 порога thresholds_by_markup (maggr_pix fp 0.902915 / op 0.684725, etgr_pix op 0.545825, terfl_pix op 0.746421).
- EXIT 0, время 535 с, пик рабочего набора 3 052 МБ. EXPDS_PIXGBM_THREADS не задавался.
- Хук работает: все 26 226 объектов в pix-группах (replace): maggr_pix finds_points 19 423, maggr_pix object_poly 3 928, etgr_pix object_poly 1 012, terfl_pix object_poly 1 863. Лог: «записано 26226 объектов за 533 с».
- Файл: C:\Sloi\pix\geo_pix_S_train.geojson (на ПК). Дальше: main.py на крупнейшем участке (093, 25,5 ГБ) — замер RAM/времени, итог допишу.
- main.py (wt_bot7) на 093 (25,5 ГБ ортофото): EXIT 0, 1 624 с, пик WS 2 507 МБ. Ограничено чтением F: (~25 МБ/с), а не CPU. pix_done.flag записан в 00:10:12 МСК.
